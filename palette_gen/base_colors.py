from __future__ import annotations

import pickle
from dataclasses import dataclass
from multiprocessing import Pool
from string import ascii_uppercase
from typing import Iterator

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_hex, to_rgb
from matplotlib.figure import Figure
from numba import njit
from scipy.optimize import minimize, shgo
from scipy.special import expit

from palette_gen.fastcolors import (
    HSV_to_RGB_jit,
    cct_to_D_xyY_jit,
    dE_2000_sRGB_D65_jit,
    sRGB_to_XYZ_jit,
    xyY_to_XYZ_jit,
)
from palette_gen.punishedcam import XYZ_to_PunishedCAM20_JabQMsh_jit, de_jab_ucs

np.set_printoptions(precision=3)

# fundamental base colorspace definitions
N_CFUL = 12
N_BRIGHT = 12
N_HUES = 16

CFUL_MIN = 5
CFUL_MAX = 50
BRIGHT_MIN = 0.1
BRIGHT_MAX = 1.0

NORM_CFULS = np.linspace(0.0, 1.0, num=N_CFUL)
NORM_BRIGHTS = np.linspace(0.0, 1.0, num=N_BRIGHT)
NORM_HUES = np.linspace(0.0, 1.0, endpoint=False, num=N_HUES)

CFULS = NORM_CFULS * (CFUL_MAX - CFUL_MIN) + CFUL_MIN
BRIGHTS = NORM_BRIGHTS * (BRIGHT_MAX - BRIGHT_MIN) + BRIGHT_MIN
HUES = 360.0 * NORM_HUES


@dataclass()
class ViewingSpec:
    name: str
    T: float  # K
    Lsw: float  # cdm-2
    Lb: float  # cdm-2
    Lmax: float  # cdm-2

    # noinspection PyPep8Naming
    @property
    def XYZw(self) -> np.ndarray:
        return xyY_to_XYZ_jit(cct_to_D_xyY_jit(self.T))

    def get_cam_values(self, xyz: np.ndarray) -> np.ndarray:
        return XYZ_to_PunishedCAM20_JabQMsh_jit(
            xyz,
            self.XYZw.reshape(1, -1),
            Lsw=self.Lsw,
            Lb=self.Lb,
            Lmax=self.Lmax,
        )


NIGHT_VIEW_LIGHT = ViewingSpec("night_light", T=5000, Lsw=1, Lmax=10, Lb=8)
DAY_VIEW_LIGHT = ViewingSpec("day_light", T=6500, Lsw=100, Lmax=60, Lb=40)


@dataclass(order=True)
class ColorSpec:
    """
    Specifies a target color for optimization.
    """

    bx: int
    sx: int
    hx: int

    # targets
    t_bright: float
    t_cful: float
    t_hue: float

    loss_: float = np.nan
    qmh_: np.ndarray = None
    hsv_: np.ndarray = None

    @property
    def is_solved(self) -> bool:
        return self.hsv_ is not None

    @property
    def rgb_(self) -> np.ndarray:
        if self.hsv_ is None:
            raise ValueError("Spec is not fitted yet -- no HSV!")
        return HSV_to_RGB_jit(self.hsv_.reshape((1, -1))).squeeze()

    @classmethod
    def mk_grid(cls) -> Iterator[ColorSpec]:
        for bx, bright in enumerate(NORM_BRIGHTS):
            for sx, cful in enumerate(NORM_CFULS):
                for hx, hue in enumerate(NORM_HUES):

                    mask = cls.hue_mask_ladder(cful)
                    if hx % mask:
                        continue

                    yield cls(
                        hx=hx,
                        sx=sx,
                        bx=bx,
                        t_hue=hue,
                        t_cful=cful,
                        t_bright=bright,
                    )

    def __hash__(self) -> int:
        return hash((self.bx, self.hx, self.sx))

    # noinspection Mypy
    def __eq__(self, other) -> bool:
        return (self.hx, self.bx, self.sx) == (other.hx, other.bx, other.sx)

    @staticmethod
    def hue_mask_ladder(norm_cful: float) -> int:
        if norm_cful < 0.20:
            mask = N_HUES // 2
        elif norm_cful < 0.35:
            mask = N_HUES // 4
        elif norm_cful < 0.50:
            mask = N_HUES // 8
        else:
            mask = 1
        return mask

    @property
    def name(self) -> str:
        """Applies the standard naming scheme to an HCB color."""
        hue_let_ix = -self.hx - 1 - (1 if self.hx >= 11 else 0)
        return f"{self.bx:x}{self.sx:x}{ascii_uppercase[hue_let_ix]}"


ColorSpecMap = dict[ColorSpec, ColorSpec]


# noinspection PyPep8Naming
@njit  # type: ignore
def _pcam_loss_jit(
    logit_hsv: np.ndarray,
    out_jmh: np.ndarray,
    t_hue: float,
    t_cful: float,
    t_bright: float,
    XYZr: np.ndarray,
    Lsw: float,
    Lb: float,
    Lmax: float,
    w_hue: float = 1.0,
    w_hard_hue_elbow_deg: float = 3.0,
    w_hard_hue: float = 10.0,
    w_sat: float = 2.0,
    w_bright: float = 3.0,
) -> float:

    hsv = 1 / (1 + np.exp(-logit_hsv.reshape(1, -1)))
    rgb: np.ndarray = HSV_to_RGB_jit(hsv)
    xyz: np.ndarray = sRGB_to_XYZ_jit(rgb)

    jabqmsh = XYZ_to_PunishedCAM20_JabQMsh_jit(xyz, XYZr, Lsw, Lb, Lmax)

    norm_bright = (jabqmsh[..., 0] - BRIGHT_MIN) / (BRIGHT_MAX - BRIGHT_MIN)
    norm_cful = (jabqmsh[..., 4] - CFUL_MIN) / (CFUL_MAX - CFUL_MIN)
    norm_hue = jabqmsh[..., -1] / 360.0

    # print(jabqmsh)

    loss_arr = np.zeros_like(hsv)

    # circular difference for hue loss
    d_hue = (norm_hue - t_hue) % 1.0  # range [0, 1]
    d_ge_half = (d_hue >= 0.5).astype(np.uint8)
    d_hue = d_ge_half * (1.0 - d_hue) + (1 - d_ge_half) * d_hue
    loss_arr[..., 0] += w_hue * d_hue

    # past-elbow hard loss
    hue_elbow = w_hard_hue_elbow_deg / 360.0
    d_is_excess = (d_hue >= hue_elbow).astype(np.uint8)
    loss_arr[..., 0] += w_hard_hue * d_is_excess * (d_hue - hue_elbow)

    loss_arr[..., 1] = w_sat * np.abs(norm_cful - t_cful)
    loss_arr[..., 2] = w_bright * np.abs(norm_bright - t_bright)
    loss: float = loss_arr.sum()

    out_jmh[..., 0] = jabqmsh[..., 0]
    out_jmh[..., 1] = jabqmsh[..., 4]
    out_jmh[..., 2] = jabqmsh[..., -1]

    return loss


def minimizer_pcam(color: ColorSpec, view: ViewingSpec) -> ColorSpec:
    """
    Finds the best ciecam QMh
    Parameters
    ----------
    color
    view

    Returns
    -------

    """

    color.qmh_ = np.zeros((1, 3))

    res = shgo(
        _pcam_loss_jit,
        # bounds from one-byte color depth assumption
        # 256 * expit(-7) < 0.5
        # 256 * expit(7) >= 255.5
        bounds=[(-7, 7)] * 3,
        args=(
            # this modifies the array inplace
            color.qmh_,
            color.t_hue,
            color.t_cful,
            color.t_bright,
            view.XYZw,
            view.Lsw,
            view.Lb,
            view.Lmax,
        ),
    )
    if not res["success"]:
        print(res["message"])

    # return the modified spec to be multiprocessing-friendly
    color.loss_ = res["fun"]
    color.hsv_ = expit(res["x"])
    color.qmh_ = color.qmh_.squeeze()
    return color


def solve_uniform_color_grid(
    view_conditions: ViewingSpec,
    out_fn: str,
    do_plot: bool = False,
    extreme_loss_thresh: float = 0.25,
    plot_bg_rgb: str = None,
) -> list[ColorSpec]:

    grid = [(cs, view_conditions) for cs in ColorSpec.mk_grid()]
    with Pool(24) as pool:
        results = pool.starmap(minimizer_pcam, grid)

    out = []
    for spec in results:
        assert spec.is_solved
        print(spec)

        if spec.loss_ > extreme_loss_thresh:
            continue

        out.append(spec)

    out = sorted(out)

    if do_plot:
        dump_color_scheme(out, do_plot_loss=True, bg_rgb=plot_bg_rgb)

    with open(out_fn, "wb") as f:
        pickle.dump(out, f)

    return out


@njit  # type: ignore
def hinge_loss(arr: np.ndarray, lb: float, ub: float, α: float) -> np.ndarray:
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
@njit  # type: ignore
def palette_loss(
    logit_rgb: np.ndarray,
    out_jab: np.ndarray,
    xyz_r: np.ndarray,
    Lsw: float,
    Lb: float,
    Lmax: float,
    background_jab: np.ndarray,
    j_min: float,
    j_max: float,
    j_alpha: float,
    m_min: float,
    m_max: float,
    m_alpha: float,
    de_min: float,
    de_max: float,
    de_alpha: float,
) -> float:
    rgb = (1 / (1 + np.exp(-logit_rgb))).reshape((-1, 3))
    xyz = sRGB_to_XYZ_jit(rgb)
    jabqmsh = XYZ_to_PunishedCAM20_JabQMsh_jit(
        xyz, xyz_r, Lsw=Lsw, Lb=Lb, Lmax=Lmax
    )
    # print(rgb, xyz, jabqmh)
    pairwise_de = de_jab_ucs(
        jabqmsh.reshape((-1, 1, 7)), jabqmsh.reshape((1, -1, 7))
    ).ravel()

    # insertion sort cause .sort() isn't implemented rofl.
    for ix in range(1, len(pairwise_de)):
        for jx in range(ix, 0, -1):
            if pairwise_de[jx - 1] <= pairwise_de[jx]:
                break
            tmp = pairwise_de[jx]
            pairwise_de[jx] = pairwise_de[jx - 1]
            pairwise_de[jx - 1] = tmp

    loss = 0.0
    for i in range(len(rgb), len(pairwise_de), 2):
        loss -= pairwise_de[i] / (i - len(rgb) + 1) ** 2

    loss += hinge_loss(jabqmsh[..., 0], j_min, j_max, j_alpha).sum()
    loss += hinge_loss(jabqmsh[..., 4], m_min, m_max, m_alpha).sum()
    loss += hinge_loss(
        de_jab_ucs(background_jab, jabqmsh), de_min, de_max, de_alpha
    ).sum()

    out_jab[:] = jabqmsh[..., :3]
    print(loss)
    return loss


def solve_palette_global(
    view_conditions: ViewingSpec,
    n_colors: int,
    bg_rgb: str,
    j_hinge: tuple[float, float, float] = (0.0, 1.5, 1.0),
    m_hinge: tuple[float, float, float] = (0.0, 100.0, 1.0),
    de_hinge: tuple[float, float, float] = (0.0, 10.0, 3.0),
):

    rgb = np.random.normal(size=(n_colors, 3)).ravel()
    out_jab = np.zeros((n_colors, 3))

    bg_jab = XYZ_to_PunishedCAM20_JabQMsh_jit(
        sRGB_to_XYZ_jit(np.array(to_rgb(bg_rgb)).reshape(-1, 3)),
        view_conditions.XYZw,
        view_conditions.Lsw,
        view_conditions.Lb,
        view_conditions.Lmax,
    )

    res = minimize(
        palette_loss,
        rgb,
        args=(
            out_jab,
            view_conditions.XYZw,
            view_conditions.Lsw,
            view_conditions.Lb,
            view_conditions.Lmax,
            bg_jab,
            *j_hinge,
            *m_hinge,
            *de_hinge,
        ),
    )
    res["jab"] = out_jab
    return res


def dump_color_scheme(
    base_colors: list[ColorSpec],
    bg_rgb: str = None,
    do_plot_loss: bool = False,
) -> None:

    all_ax: dict[int, tuple[Figure, Axes]] = {}

    for spec in sorted(base_colors):
        assert spec.is_solved
        bx = spec.bx

        if spec.bx not in all_ax:
            fig = plt.figure(bx)
            ax = fig.add_subplot(111, projection="polar")
            all_ax[bx] = fig, ax
        else:
            _, ax = all_ax[bx]

        θ = np.pi * spec.qmh_[2] / 180
        r = spec.qmh_[1]

        ax.scatter(θ, r, s=300, marker="s", c=to_hex(spec.rgb_))
        if do_plot_loss:
            plt.scatter(θ, r, s=100 * spec.loss_, color="red")

    # change plot features in separate pass to avoid calling these functions
    # too often in original pass
    for bx, (fig, ax) in all_ax.items():
        if bg_rgb is not None:
            ax.set_facecolor(bg_rgb)

        ax.set_ylim(0, CFUL_MAX + 5)
        ax.set_xlabel("Hue (h**)")
        ax.set_ylabel("Colorfulness (M**)")
        ax.set_title(f"Corrected J** = {BRIGHTS[bx]:0.2f} [{bx=}]")
        fig.set_size_inches(6, 6)
        fig.tight_layout()
    plt.show()


def filter_hsb_by_de(
    colors: ColorSpecMap,
    ref_rgb: np.ndarray,
    min_de: float = 0.0,
    max_de: float = 100.0,
) -> ColorSpecMap:

    if ref_rgb.ndim == 1:
        ref_rgb = ref_rgb[None, :]

    dists = dE_2000_sRGB_D65_jit(ref_rgb, np.array([*colors.values()]))

    return {
        k: v
        for dist, (k, v) in zip(dists, colors.items())
        if (min_de <= dist <= max_de)
    }


def save_cam_grid(view_spec: ViewingSpec) -> None:

    rgb_grid = (
        (np.mgrid[0:256, 0:256, 0:256] / 255)
        .transpose(1, 2, 3, 0)
        .astype(np.float32)
    )

    xyz = sRGB_to_XYZ_jit(rgb_grid)
    cam_grid = view_spec.get_cam_values(xyz)

    with open(f"cam_block_{view_spec.name}.npz", "wb") as f:
        np.savez(f, cam_grid)


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)

    res = solve_palette_global(
        DAY_VIEW_LIGHT, 6, "#d9d9d9", de_hinge=(0.1, 0.2, 1.0)
    )
    with open("test_res_close.p", "wb") as f:
        pickle.dump(res, f)

    # save_cam_grid(view_spec=DAY_VIEW_LIGHT)
    # save_cam_grid(view_spec=NIGHT_VIEW_LIGHT)
