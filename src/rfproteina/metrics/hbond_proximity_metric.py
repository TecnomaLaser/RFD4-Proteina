from typing import Any

import numpy as np
from biotite.structure import AtomArray

from rfproteina.conditions import C_HB
from rfproteina.metrics.metric import Metric
from rfproteina.utils.standard_annotations import SA_CRDSRC

# Heavy-atom donor/acceptor separation for a hydrogen bond. PLIP's own cutoff is 4.1 A before
# angle filtering; 3.4 A is the usual heavy-atom distance bound once geometry is taken into
# account, and 3.2 A is comfortably inside it.
HBOND_MAX_DISTANCE = 3.4
POLAR_ELEMENTS = ("N", "O", "S")


class HBondConditioningProximityMetric(Metric):
    """Distance from each ``C_HB``-conditioned atom to the nearest *generated* polar atom.

    ``C_HB`` is trained on **inter-chain** H-bond roles, and
    :py:class:`~rfproteina.metrics.interaction_metrics.InterchainInteractionMetric` scores
    adherence with ``inter_chain_only=True``. Neither says anything about an H-bond between a
    conditioned atom and a residue the model designed in the *same* chain, which is a thing
    users ask for and which nothing else here measures.

    This reports, over all ``C_HB``-conditioned atoms, the distance to the closest polar atom
    that the model was free to place -- i.e. one that is not coordinate-conditioned. A
    conditioned atom whose only polar neighbours are themselves fixed has no satisfied request
    in any useful sense, so those are excluded as partners.

    Reports NaN when nothing is H-bond-conditioned.
    """

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {"predicted_atom_array": ("predicted_atom_array",)}

    def compute(self, predicted_atom_array: AtomArray) -> dict[str, float]:
        """Return nearest-generated-polar-partner distances for H-bond-conditioned atoms."""
        conditioned = C_HB.mask(predicted_atom_array, default="generate")
        if not conditioned.any():
            return {
                "min_distance_to_generated_polar": float("nan"),
                "median_distance_to_generated_polar": float("nan"),
                "n_conditioned_atoms": 0.0,
                "n_satisfied": float("nan"),
                "frac_satisfied": float("nan"),
            }

        # Candidate partners: polar, and placed by the model rather than held fixed.
        is_polar = np.isin(predicted_atom_array.element, POLAR_ELEMENTS)
        is_generated = ~SA_CRDSRC.mask(predicted_atom_array)
        partners = predicted_atom_array.coord[is_polar & is_generated]

        n_conditioned = float(conditioned.sum())
        if len(partners) == 0:
            return {
                "min_distance_to_generated_polar": float("inf"),
                "median_distance_to_generated_polar": float("inf"),
                "n_conditioned_atoms": n_conditioned,
                "n_satisfied": 0.0,
                "frac_satisfied": 0.0,
            }

        # Nearest generated polar partner for each conditioned atom.
        deltas = predicted_atom_array.coord[conditioned][:, None, :] - partners[None, :, :]
        nearest = np.sqrt((deltas**2).sum(axis=-1)).min(axis=1)

        n_satisfied = float((nearest <= HBOND_MAX_DISTANCE).sum())
        return {
            "min_distance_to_generated_polar": float(nearest.min()),
            "median_distance_to_generated_polar": float(np.median(nearest)),
            "n_conditioned_atoms": n_conditioned,
            "n_satisfied": n_satisfied,
            "frac_satisfied": n_satisfied / n_conditioned,
        }
