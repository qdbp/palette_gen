from typing import Any, Iterable, Type, Union

import numpy as np

from palette_gen.solvers import RGBColor, T
from palette_gen.solvers.color import FixedJabTargetSolver


class JMatchedGreys(FixedJabTargetSolver):
    """
    Returns a set of (whitepoint-adapted) greys of given lightnesses.
    """

    def __init__(self, j_dict: dict[str, float]):
        self.j_dict = j_dict

    def jab_target(self, ab_offset: np.ndarray) -> np.ndarray:

        out = np.empty((len(self.j_dict), 3))
        out[:, 1:] = ab_offset[None, :]
        out[:, 0] = np.array([*self.j_dict.values()])

        return out

    def organize_colors(
        self, raw_colors: Iterable[RGBColor]
    ) -> Union[list[RGBColor], dict[str, list[RGBColor]]]:
        colors = list(raw_colors)
        for k, c in zip(self.j_dict.keys(), colors):
            c.name = k
        return colors

    @classmethod
    def construct_from_config(cls: Type[T], conf: dict[str, Any]) -> T:
        conf.pop("name", None)
        return cls(j_dict=conf)
