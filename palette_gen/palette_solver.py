from __future__ import annotations

from dataclasses import astuple, dataclass
from functools import cached_property
from itertools import chain
from typing import cast

import matplotlib.pyplot as plt

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_hex, to_rgb
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from numba import njit
from scipy.optimize import minimize
from scipy.special import expit

from palette_gen.fastcolors import (
    cct_to_D_xyY_jit,
    sRGB_to_XYZ_jit,
    xyY_to_XYZ_jit,
)
from palette_gen.punishedcam import XYZ_to_PunishedCAM20_JabQMsh_jit, de_jab_ucs


def mk_cube_surface_grid(pps: int = 20) -> np.ndarray:
    """
    Generates a regular grid over the surface of the {0,1}^3 corner unit cube.
    """

    base = np.linspace(1 / pps, 1 - 1 / pps, num=pps - 2)
    edges = np.array([0.0, 1.0])
    cube_dim = 3

    return np.concatenate(
        [
            np.stack(
                np.meshgrid(
                    *([base] * ix + [edges] + [base] * (cube_dim - ix - 1)),
                    indexing="ij",
                ),
                axis=-1,
            ).reshape(-1, 3)
            for ix in range(cube_dim)
        ],
    )


@dataclass(frozen=True, order=True)
class Color:
    rgb: tuple[float, float, float]
    vs: ViewingSpec

    @cached_property
    def jab(self) -> np.ndarray:
        return self.vs.xyz_to_cam(self.vs.rgb_to_xyz(np.array(self.rgb)))[0, :3]

    @cached_property
    def hex(self) -> str:
        return cast(str, to_hex(self.rgb))


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

    # TODO should be customizable based on additional fields
    # noinspection PyMethodMayBeStatic
    def rgb_to_xyz(self, rgb: np.narray) -> np.ndarray:
        return sRGB_to_XYZ_jit(rgb.reshape((-1, 3)))

    def xyz_to_cam(self, xyz: np.ndarray) -> np.ndarray:
        return XYZ_to_PunishedCAM20_JabQMsh_jit(
            xyz,
            self.XYZw.reshape(1, -1),
            Lsw=self.Lsw,
            Lb=self.Lb,
            Lmax=self.Lmax,
        )

    def rgb_to_cam(self, rgb: np.ndarray) -> np.ndarray:
        return self.xyz_to_cam(self.rgb_to_xyz(rgb)).squeeze()


# TODO the hinge spec can be generalized to include more general polygonal
# regions


@dataclass(frozen=True, order=True)
class HingeSpec:
    min: float
    max: float
    alpha: float = 1.0


@dataclass
class PaletteSpec:
    name: str
    n_colors: int

    m_hinge: HingeSpec
    j_hinge: HingeSpec
    de_hinge: HingeSpec

    def solve_in_context(self, bg_rgb: str, vs: ViewingSpec) -> list[Color]:

        rgb = np.random.normal(size=(self.n_colors, 3)).ravel()
        out_jab = np.zeros((self.n_colors, 3))

        bg_jab = XYZ_to_PunishedCAM20_JabQMsh_jit(
            sRGB_to_XYZ_jit(np.array(to_rgb(bg_rgb)).reshape(-1, 3)),
            vs.XYZw,
            vs.Lsw,
            vs.Lb,
            vs.Lmax,
        )

        res = minimize(
            palette_loss,
            rgb,
            args=(
                out_jab,
                vs.XYZw,
                vs.Lsw,
                vs.Lb,
                vs.Lmax,
                bg_jab,
                *astuple(self.j_hinge),
                *astuple(self.m_hinge),
                *astuple(self.de_hinge),
            ),
        )

        # noinspection PyTypeChecker
        return sorted(
            Color(rgb=tuple(expit(rgb)), vs=vs)
            for rgb in res["x"].reshape((-1, 3))
        )


NIGHT_VIEW_LIGHT = ViewingSpec("night_light", T=6500, Lsw=1, Lmax=10, Lb=8)
DAY_VIEW_LIGHT = ViewingSpec("day_light", T=6500, Lsw=100, Lmax=60, Lb=40)


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

    n_colors = len(rgb)

    # pairwise_de = de_jab_ucs(
    #     jabqmsh.reshape((-1, 1, 7)), jabqmsh.reshape((1, -1, 7))
    # ).ravel()
    # insertion sort cause .sort() isn't implemented rofl.
    # for ix in range(1, len(pairwise_de)):
    #     for jx in range(ix, 0, -1):
    #         if pairwise_de[jx - 1] <= pairwise_de[jx]:
    #             break
    #         tmp = pairwise_de[jx]
    #         pairwise_de[jx] = pairwise_de[jx - 1]
    #         pairwise_de[jx - 1] = tmp

    # loss = 0.0
    # for i in range(len(rgb), len(pairwise_de), 2):
    #     loss -= pairwise_de[i] / (i - len(rgb) + 1) ** 2

    pairwise_de = de_jab_ucs(
        jabqmsh.reshape((-1, 1, 7)), jabqmsh.reshape((1, -1, 7))
    )
    pairwise_de += np.diag(np.ones(n_colors))

    loss = -np.log(pairwise_de).sum()
    loss /= (n_colors ** 2)

    # scale adjustment factor to make loss sorta n_colors neutral
    # loss /= np.log(len(rgb))

    j_loss = hinge_loss(jabqmsh[..., 0], j_min, j_max, j_alpha)
    loss += j_loss.sum()

    m_loss = hinge_loss(jabqmsh[..., 4], m_min, m_max, m_alpha)
    loss += m_loss.sum()

    de_loss = hinge_loss(
        de_jab_ucs(background_jab, jabqmsh), de_min, de_max, de_alpha
    )
    loss += de_loss.sum()

    # print(de_jab_ucs(background_jab, jabqmsh))
    # print(np.vstack((j_loss, m_loss, de_loss)))

    out_jab[:] = jabqmsh[..., :3]
    return loss


def save_cam_grid(view_spec: ViewingSpec) -> None:

    rgb_grid = (
        (np.mgrid[0:256, 0:256, 0:256] / 255)
        .transpose(1, 2, 3, 0)
        .astype(np.float32)
    )

    xyz = sRGB_to_XYZ_jit(rgb_grid)
    cam_grid = view_spec.xyz_to_cam(xyz)

    with open(f"cam_block_{view_spec.name}.npz", "wb") as f:
        np.savez(f, cam_grid)


class ColorScheme:
    def __init__(
        self,
        name: str,
        bg_hex: str,
        vs: ViewingSpec,
        palettes: dict[str, PaletteSpec],
    ):
        self.name = name
        self.bg_hex = bg_hex
        self.vs = vs
        self.p_specs = palettes

        self.colors_dict = self.solve()

    def solve(self) -> dict[str, list[Color]]:

        all_colors = {}
        for name, spec in self.p_specs.items():
            print(f"Solving palette {name}")
            all_colors[name] = spec.solve_in_context(self.bg_hex, self.vs)

        return all_colors

    def draw_cone(self) -> None:
        import plotly.graph_objects as go

        marker_cycle = [
            "circle",
            "square",
            "diamond",
            "x",
            "cross",
        ]

        jab_arr = np.array(
            list(
                color.jab
                for color in chain.from_iterable(self.colors_dict.values())
            )
        )
        rgb_arr = np.array(
            list(
                color.rgb
                for color in chain.from_iterable(self.colors_dict.values())
            )
        )

        symbols = []
        for cx, colors in enumerate(self.colors_dict.values()):
            symbols.extend([marker_cycle[cx]] * len(colors))

        jab_edges = self.vs.rgb_to_cam(mk_cube_surface_grid())

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=jab_arr[..., 1],
                    y=jab_arr[..., 2],
                    z=jab_arr[..., 0],
                    marker=dict(
                        color=list(map(to_hex, rgb_arr)),
                        size=6,
                        symbol=symbols,
                    ),
                    mode="markers",
                ),
                go.Scatter3d(
                    x=jab_edges[:, 1],
                    y=jab_edges[:, 2],
                    z=jab_edges[:, 0],
                    marker=dict(color="black", size=1),
                    mode="markers",
                ),
            ],
        )
        fig.show()
        fig.write_html("test.html")

    def draw_colors(self) -> None:

        fig: Figure = plt.figure()

        gs = GridSpec(figure=fig, nrows=1, ncols=10)
        axp: Axes = fig.add_subplot(gs[0, :-2])
        axl: Axes = fig.add_subplot(gs[0, -2:])

        gx = 0
        for name, colors in self.colors_dict.items():
            for cx, color in enumerate(colors):
                j, a, b = color.jab
                marker = f"$\\rm{name}{cx:03d}$"
                axp.scatter(
                    10 * j * a,
                    10 * j * b,
                    s=500,
                    marker=marker,
                    color=color.hex,
                )
                axl.scatter(0, gx, s=500, marker=marker, color=color.hex)
                axl.text(
                    1,
                    gx,
                    color.hex,
                    color="grey",
                    fontsize="small",
                )
                gx += 1

        axl.set_xlim(-1, 2.5)
        for ax in [axl, axp]:
            ax.set_facecolor(self.bg_hex)

        plt.title(f"{self.name} color scheme colors.")
        plt.show()

    @property
    def is_dark(self) -> bool:
        return sum(to_rgb(self.bg_hex)) <= 1.5


if __name__ == "__main__":

    np.set_printoptions(precision=2, suppress=True)

    bg_hex = "#E8E8E8"
    specs = [
        light_bgs := PaletteSpec(
            name="BGL",
            n_colors=8,
            m_hinge=HingeSpec(0, 10, 0.1),
            j_hinge=HingeSpec(0.80, 0.95, 3.0),
            de_hinge=HingeSpec(0.00, 0.15, 10.0),
        ),
        primaries := PaletteSpec(
            name="PRI",
            n_colors=16,
            m_hinge=HingeSpec(15, 50, 0.1),
            j_hinge=HingeSpec(0.45, 0.60, 3.0),
            de_hinge=HingeSpec(0.3, 2.0, 10.0),
        ),
        secondaries := PaletteSpec(
            name="SND",
            n_colors=8,
            m_hinge=HingeSpec(0, 10, 0.1),
            j_hinge=HingeSpec(0.65, 0.75, 3.0),
            de_hinge=HingeSpec(0.20, 0.7, 10.0),
        ),
        highlights := PaletteSpec(
            name="HL",
            n_colors=6,
            m_hinge=HingeSpec(15, 25, 0.1),
            j_hinge=HingeSpec(0.75, 0.85, 3.0),
            de_hinge=HingeSpec(0.20, 0.20, 10.0),
        ),
    ]

    spec_dict = {ps.name: ps for ps in specs}

    scheme = ColorScheme(
        "Restraint",
        bg_hex,
        vs=NIGHT_VIEW_LIGHT,
        palettes=spec_dict,
    )

    scheme.draw_cone()