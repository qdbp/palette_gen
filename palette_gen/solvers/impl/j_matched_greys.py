from __future__ import annotations

from typing import Any
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from palette_gen.solvers import JabColor
from palette_gen.solvers.color import FixedJabTargetSolver, OrganizedColors


class JMatchedGreys(FixedJabTargetSolver):
    """
    Returns a set of (whitepoint-adapted) greys of given lightnesses.
    """

    def __init__(self, j_dict: dict[str, float]):
        self.j_dict = j_dict

    def jab_target(self, ab_offset: NDArray[np.float64]) -> NDArray[np.float64]:
        out = np.empty((len(self.j_dict), 3))
        out[:, 1:] = ab_offset[None, :]
        out[:, 0] = np.array([*self.j_dict.values()])

        return out

    def organize_colors(self, raw_colors: Iterable[JabColor]) -> OrganizedColors:
        colors = list(raw_colors)
        for k, c in zip(self.j_dict.keys(), colors):
            c.name = k
        return colors

    # defect: py311 Self type
    @classmethod
    def construct_from_config(cls, conf: dict[str, Any]) -> JMatchedGreys:
        conf.pop("name", None)
        return cls(j_dict=conf)
