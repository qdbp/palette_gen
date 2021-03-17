from dataclasses import dataclass
from typing import Any, Type

import numpy as np
from matplotlib.colors import to_rgb
from numba import njit
from scipy.optimize import minimize
from scipy.special import expit

from palette_gen.fastcolors import sRGB_to_XYZ_jit  # type: ignore
from palette_gen.punishedcam import (  # type: ignore
    XYZ_to_PUNISHEDCAM_JabQMsh_jit,
)
from palette_gen.solvers import Color, T, ViewingSpec
from palette_gen.solvers.color import ColorSolver


@dataclass()
class JabRingSpec(ColorSolver):
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

    def solve_for_context(self, bg_hex: str, vs: ViewingSpec) -> list[Color]:
        logit_rgb = np.random.normal(size=(self.n_colors, 3)).ravel()
        ab_offset = vs.rgb_to_cam(np.array(to_rgb(bg_hex))[None, :])[1:3]
        jab_target = self.jab_target(ab_offset)

        print(
            f"Solving ring {self.name} with offset {ab_offset} from {bg_hex=}"
        )

        res = minimize(
            self.loss,
            logit_rgb,
            args=(
                jab_target,
                vs.XYZw,
                vs.Lsw,
                vs.Lb,
                vs.Lmax,
            ),
        )

        # noinspection PyTypeChecker
        return sorted(
            Color(rgb=tuple(expit(rgb)), vs=vs)  # type: ignore
            for rgb in res["x"].reshape((-1, 3))
        )

    # noinspection PyPep8Naming
    @staticmethod
    @njit  # type: ignore
    def loss(
        logit_rgb: np.ndarray,
        jab_target: np.ndarray,
        xyz_r: np.ndarray,
        Lsw: float,
        Lb: float,
        Lmax: float,
    ) -> float:
        rgb = (1 / (1 + np.exp(-logit_rgb))).reshape((-1, 3))
        jabqmsh = XYZ_to_PUNISHEDCAM_JabQMsh_jit(
            sRGB_to_XYZ_jit(rgb), xyz_r, Lsw=Lsw, Lb=Lb, Lmax=Lmax
        )
        jab = jabqmsh[..., :3]
        loss = ((jab - jab_target) ** 2).sum()
        return loss  # type: ignore

    @classmethod
    def construct_from_config(cls: Type[T], conf: dict[str, Any]) -> T:
        return cls(**conf)  # type: ignore
