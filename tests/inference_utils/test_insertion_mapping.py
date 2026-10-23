"""Test build_predicted_insertion_mapping target eligibility.

A proposed target may not already carry coordinate conditioning, since transferring a second,
different coordinate onto it is contradictory and makes `annotate_unindexed_insertion` raise.
Identity is intentionally NOT constrained: the head's logits carry where the motif actually went,
and a wrong-identity target should fail honestly in atom matching rather than be replaced by a
same-identity target elsewhere in the chain.
"""

import numpy as np
import torch
from atomworks.ml.conditions import C_CRD, C_IDX, C_SEQ
from biotite.structure import AtomArray

from rfproteina.inference_utils.insertion_mapping import build_predicted_insertion_mapping

# One atom per token keeps the token bookkeeping obvious; eligibility is a per-token property.
_GENERATED = [("ALA", 0), ("HIS", 1), ("GLY", 2), ("HIS", 3)]
_UNINDEXED = [("HIS", 4)]


def _make_array(
    generated: list[tuple[str, int]] = _GENERATED,
    unindexed: list[tuple[str, int]] = _UNINDEXED,
    conditioned_token_ids: tuple[int, ...] = (),
    seq_condition_unindexed: bool = True,
) -> AtomArray:
    """Build a single-chain array of generated tokens followed by unindexed in-context tokens."""
    all_tokens = generated + unindexed
    aa = AtomArray(len(all_tokens))
    aa.atom_name = np.array(["CA"] * len(all_tokens))
    aa.element = np.array(["C"] * len(all_tokens))
    aa.res_name = np.array([name for name, _ in all_tokens])
    aa.res_id = np.array([tid + 1 for _, tid in all_tokens])
    aa.chain_id = np.array(["A"] * len(all_tokens))
    aa.coord = np.tile(np.arange(len(all_tokens), dtype=float)[:, None], (1, 3))
    aa.set_annotation("token_id", np.array([tid for _, tid in all_tokens]))

    is_generated = np.arange(len(all_tokens)) < len(generated)
    C_IDX.set_annotation(aa, is_generated)

    coord_conditioned = np.isin(aa.token_id, np.asarray(conditioned_token_ids, dtype=int))
    C_CRD.set_annotation_from_ground_truth(aa, coord_conditioned)

    seq_mask = np.zeros(len(all_tokens), dtype=bool)
    if seq_condition_unindexed:
        seq_mask |= ~is_generated
    C_SEQ.set_annotation_from_ground_truth(aa, seq_mask)
    return aa


def _logits(preferred_col: int, n_gen: int = len(_GENERATED), n_unindexed: int = len(_UNINDEXED)) -> torch.Tensor:
    """Logits whose single unindexed row most prefers ``preferred_col`` among the generated tokens."""
    n_valid = n_gen + n_unindexed
    logits = torch.zeros((n_valid, n_valid))
    logits[n_gen, preferred_col] = 10.0
    return logits


def test_prefers_highest_logit_eligible_target() -> None:
    """With an eligible argmax the mapping is unchanged from plain Hungarian behaviour."""
    aa = _make_array()
    # Column 1 is the HIS at token_id 1: identity matches and it carries no conditioning.
    mapping = build_predicted_insertion_mapping(_logits(preferred_col=1), aa)
    assert mapping == {4: 1}


def test_skips_already_coordinate_conditioned_target() -> None:
    """A target that is part of the indexed motif is never proposed, even as the argmax."""
    aa = _make_array(conditioned_token_ids=(1,))
    n_gen = len(_GENERATED)
    logits = torch.zeros((n_gen + 1, n_gen + 1))
    logits[n_gen, 1] = 10.0  # best target, but coordinate-conditioned -> ineligible
    logits[n_gen, 3] = 5.0  # best eligible target
    mapping = build_predicted_insertion_mapping(logits, aa)
    assert mapping == {4: 3}


def test_identity_mismatch_is_not_overridden() -> None:
    """Identity is not constrained: the head's preferred target wins even on a mismatch.

    Substituting a same-identity target elsewhere in the chain would manufacture a geometrically
    wrong mapping; a mismatch is left to fail honestly in downstream atom matching.
    """
    aa = _make_array()
    # The head most wants token 0 (ALA) for a conditioned HIS motif; that choice is preserved.
    mapping = build_predicted_insertion_mapping(_logits(preferred_col=0), aa)
    assert mapping == {4: 0}


def test_unmapped_when_every_target_is_conditioned() -> None:
    """No unconditioned target leaves the token unmapped rather than raising or forcing a pair."""
    aa = _make_array(generated=[("ALA", 0), ("GLY", 1)], conditioned_token_ids=(0, 1))
    mapping = build_predicted_insertion_mapping(_logits(preferred_col=0, n_gen=2), aa)
    assert mapping == {}


def test_fewer_eligible_targets_than_tokens_maps_a_subset() -> None:
    """A shortage of unconditioned targets assigns what it can instead of raising."""
    aa = _make_array(
        generated=[("HIS", 0), ("HIS", 1)],
        unindexed=[("HIS", 4), ("HIS", 5)],
        conditioned_token_ids=(0,),
    )
    logits = torch.zeros((4, 4))
    logits[2, 1] = 5.0
    logits[3, 1] = 4.0
    mapping = build_predicted_insertion_mapping(logits, aa)
    # Only token 1 is unconditioned, so exactly one of the two motif tokens can be placed.
    assert len(mapping) == 1
    assert next(iter(mapping.values())) == 1


def test_distinct_targets_for_multiple_unindexed_tokens() -> None:
    """Two unindexed tokens competing for one target still receive distinct eligible targets."""
    aa = _make_array(unindexed=[("HIS", 4), ("HIS", 5)])
    n_gen = len(_GENERATED)
    logits = torch.zeros((n_gen + 2, n_gen + 2))
    logits[n_gen, 1] = 10.0  # both prefer column 1
    logits[n_gen + 1, 1] = 9.0
    logits[n_gen + 1, 3] = 1.0  # second choice is the other HIS
    mapping = build_predicted_insertion_mapping(logits, aa)
    assert mapping == {4: 1, 5: 3}


def test_no_unindexed_tokens_returns_empty() -> None:
    """Unchanged short-circuit: nothing unindexed means nothing to map."""
    aa = _make_array(unindexed=[])
    n_gen = len(_GENERATED)
    mapping = build_predicted_insertion_mapping(torch.zeros((n_gen, n_gen)), aa)
    assert mapping == {}
