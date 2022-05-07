from dataclasses import astuple, dataclass
from typing import Any, Iterable, Type

import numpy as np
from matplotlib.colors import to_rgb
from numba import njit
from numpy.random import MT19937, RandomState, SeedSequence
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

from palette_gen.fastcolors import sRGB_to_XYZ_jit  # type: ignore
from palette_gen.punishedcam import (  # type: ignore
    XYZ_to_PUNISHEDCAM_JabQMsh_jit,
    de_punished_jab,
)
from palette_gen.solvers import JabColor, T, ViewingSpec
from palette_gen.solvers.color import ColorSolver


@dataclass(frozen=True, order=True)
class HingeSpec:
    min: float
    max: float
    alpha: float = 1.0


@njit  # type: ignore
def hinge_loss(
    arr: NDArray[np.float64], lb: float, ub: float, α: float
) -> NDArray[np.float64]:
    flat_arr = arr.ravel()
    out = np.zeros_like(flat_arr)

    for ix in range(len(flat_arr)):
        item = flat_arr[ix]
        if item > ub:
            out[ix] = α * (item - ub)
        elif item < lb:
            out[ix] = α * (lb - item)

    return out.reshape(arr.shape)


# noinspection PyPep8Naming
@dataclass
class HingeMinDistSolver(ColorSolver):
    """
    Solves for a fixed number of colors by maximizing their log pairwise
    minimum distance, with hinge-loss constraints.
    """

    name: str
    n_colors: int

    m_hinge: HingeSpec
    j_hinge: HingeSpec
    de_hinge: HingeSpec
    hue_gap_hinge: HingeSpec

    seed: int | None = None

    def _solve_colors(self, bg_rgb: str, vs: ViewingSpec) -> Iterable[JabColor]:

        print(f"Palette {self.name}...")

        if self.seed is not None:
            state = RandomState(MT19937(SeedSequence(self.seed)))  # type: ignore
            np.random.set_state(state)  # type: ignore

        rgb = np.random.normal(size=(self.n_colors, 3)).ravel()
        out_jab = np.zeros((self.n_colors, 3))

        bg_jab = XYZ_to_PUNISHEDCAM_JabQMsh_jit(
            sRGB_to_XYZ_jit(np.array(to_rgb(bg_rgb)).reshape(-1, 3)),
            vs.XYZw,
            vs.Lsw,
            vs.Lb,
            vs.Lmax,
        )

        out_loss = np.zeros(5)
        res = minimize(
            self.loss,
            rgb,
            args=(
                out_jab,
                out_loss,
                vs.XYZw,
                vs.Lsw,
                vs.Lb,
                vs.Lmax,
                bg_jab,
                *astuple(self.j_hinge),
                *astuple(self.m_hinge),
                *astuple(self.de_hinge),
                *astuple(self.hue_gap_hinge),
            ),
        )

        loss_names = ["min(d)", "J", "M", "ΔE", "Δh"]
        print(f"loss: ", end="")
        for name, val in zip(loss_names, out_loss):
            print(f"{name}={val:.3f}; ", end="")
        print("")

        # noinspection PyTypeChecker
        return sorted(
            JabColor(rgb=tuple(expit(rgb)), vs=vs)  # type: ignore
            for rgb in res["x"].reshape((-1, 3))
        )

    @staticmethod
    @njit  # type: ignore
    def loss(
        logit_rgb: NDArray[np.float64],
        out_jab: NDArray[np.float64],
        out_loss: NDArray[np.float64],
        xyz_r: NDArray[np.float64],
        Lsw: float,
        Lb: float,
        Lmax: float,
        background_jab: NDArray[np.float64],
        j_min: float,
        j_max: float,
        j_alpha: float,
        m_min: float,
        m_max: float,
        m_alpha: float,
        de_min: float,
        de_max: float,
        de_alpha: float,
        hg_min: float,
        hg_max: float,
        hg_alpha: float,
    ) -> float:

        rgb = (1 / (1 + np.exp(-logit_rgb))).reshape((-1, 3))
        n_colors = len(rgb)

        jabqmsh = XYZ_to_PUNISHEDCAM_JabQMsh_jit(
            sRGB_to_XYZ_jit(rgb), xyz_r, Lsw=Lsw, Lb=Lb, Lmax=Lmax
        )
        out_jab[:] = jabqmsh[..., :3]

        pairwise_de = de_punished_jab(
            jabqmsh.reshape((-1, 1, 7)), jabqmsh.reshape((1, -1, 7))
        )
        pairwise_de += np.diag(np.ones(n_colors))  # type: ignore

        # calculate hue gaps
        hues = np.zeros(n_colors)
        hues[:] = jabqmsh[:, -1] / 360
        # insertion sort lol 'cause .sort ain't implemented
        i = 1
        while i < n_colors:
            j = i
            while j > 0 and hues[j - 1] > hues[j]:
                hues[j], hues[j - 1] = hues[j - 1], hues[j]
                j -= 1
            i += 1

        hg = np.zeros((1,))
        for i in range(n_colors - 1):
            hg[0] = max(hg[0], hues[i + 1] - hues[i])
        hg[0] = max(hg[0], hues[0] + 1 - hues[-1])

        out_loss[0] = -np.log(np.min(pairwise_de))  # type: ignore
        out_loss[1] = hinge_loss(jabqmsh[..., 0], j_min, j_max, j_alpha).mean()
        out_loss[2] = hinge_loss(jabqmsh[..., 4], m_min, m_max, m_alpha).mean()
        out_loss[3] = hinge_loss(
            de_punished_jab(background_jab, jabqmsh), de_min, de_max, de_alpha
        ).mean()
        # rescale the hue loss to have the same relative weight independent of
        out_loss[4] = hinge_loss(hg, hg_min, hg_max, hg_alpha).mean()

        return out_loss.sum()  # type: ignore

    @classmethod
    def construct_from_config(cls: Type[T], config: dict[str, Any]) -> T:
        raise NotImplementedError()
