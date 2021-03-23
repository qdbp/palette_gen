from dataclasses import dataclass
from typing import Any, Type

import numpy as np

from palette_gen.fastcolors import sRGB_to_XYZ_jit  # type: ignore
from palette_gen.punishedcam import (  # type: ignore
    XYZ_to_PUNISHEDCAM_JabQMsh_jit,
)
from palette_gen.solvers import T
from palette_gen.solvers.color import FixedJabTargetSolver


@dataclass()
class JabRingSpec(FixedJabTargetSolver):
    name: str
    n_colors: int
    m_lb: float
    m_ub: float
    j_lb: float
    j_ub: float

    hue_offset: float = 0.0

    def __post_init__(self) -> None:
        assert self.n_colors > 1 and not self.n_colors % 2

    def jab_target(self, ab_offset: np.ndarray) -> np.ndarray:
        """
        Gets the jab targets corresponding to the ring positions.

        Parameters
        ----------
        ab_offset: shifts the center of the ring to this point in the (a, b)
            plane. Intended to be used to center with respect to the background
            (a, b) position for some viewing conditions.

        Returns
        -------
        an array of shape (self.n_colors, 3) of the jab targets
        """

        out = np.ndarray((self.n_colors, 3))

        out[::2, 0] = self.j_ub * np.ones(self.n_colors // 2)
        out[1::2, 0] = self.j_lb * np.ones(self.n_colors // 2)

        # important - j lower corresponds to m upper
        mubs = self.m_ub * np.ones(self.n_colors // 2) / 100
        mlbs = self.m_lb * np.ones(self.n_colors // 2) / 100

        # start with 0.05 to avoid hue misalignmenet between pri/tones
        # etc in the output
        angles = np.linspace(0.05, 2 * np.pi, num=self.n_colors, endpoint=False)
        angles += self.hue_offset * (angles[1] - angles[0])
        jas = np.cos(angles) + ab_offset[0]
        jbs = np.sin(angles) + ab_offset[1]

        out[::2, 1] = jas[::2] * mlbs
        out[1::2, 1] = jas[1::2] * mubs
        out[::2, 2] = jbs[::2] * mlbs
        out[1::2, 2] = jbs[1::2] * mubs

        return out

    @classmethod
    def construct_from_config(cls: Type[T], conf: dict[str, Any]) -> T:
        return cls(**conf)  # type: ignore
