from __future__ import annotations

import pickle
from dataclasses import astuple, dataclass
from functools import cached_property
from itertools import chain
from math import atan2
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from idiotic_html_generator import HTML
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
from palette_gen.punishedcam import (
    XYZ_to_PUNISHEDCAM_JabQMsh_jit,
    de_punished_jab,
)


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
    def rgb_to_xyz(self, rgb: np.ndarray) -> np.ndarray:
        return sRGB_to_XYZ_jit(rgb.reshape((-1, 3)))

    def xyz_to_cam(self, xyz: np.ndarray) -> np.ndarray:
        return XYZ_to_PUNISHEDCAM_JabQMsh_jit(
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
    hue_gap_hinge: HingeSpec

    def solve_in_context(self, bg_rgb: str, vs: ViewingSpec) -> list[Color]:

        print(f"Solving palette {self.name}...")

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
            palette_loss,
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
            Color(rgb=tuple(expit(rgb)), vs=vs)
            for rgb in res["x"].reshape((-1, 3))
        )


NIGHT_VIEW_LIGHT = ViewingSpec("night_light", T=6500, Lsw=1, Lmax=15, Lb=8)
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
    out_loss: np.ndarray,
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
    pairwise_de += np.diag(np.ones(n_colors))

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

    out_loss[0] = -np.log(np.min(pairwise_de))
    out_loss[1] = hinge_loss(jabqmsh[..., 0], j_min, j_max, j_alpha).mean()
    out_loss[2] = hinge_loss(jabqmsh[..., 4], m_min, m_max, m_alpha).mean()
    out_loss[3] = hinge_loss(
        de_punished_jab(background_jab, jabqmsh), de_min, de_max, de_alpha
    ).mean()
    # rescale the hue loss to have the same relative weight independent of
    out_loss[4] = hinge_loss(hg, hg_min, hg_max, hg_alpha).mean()

    return out_loss.sum()


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

        self.colors_dict = {
            k: sorted(v, key=lambda x: atan2(x.jab[1], x.jab[2]))
            for k, v in self.solve().items()
        }

    def solve(self) -> dict[str, list[Color]]:

        all_colors = {}
        for name, spec in self.p_specs.items():
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

        plt.suptitle(f"{self.name} color scheme colors.")
        plt.show()

    def format_colors(self) -> str:
        html = HTML()

        with html as h:
            with h.table() as t:
                for key, colors in self.colors_dict.items():
                    with t.tr() as row:
                        for cx, color in enumerate(colors):
                            name = f"{key.upper()}{cx:02d}"
                            if sum(color.rgb) > 1.5:
                                fc = "black"
                            else:
                                fc = "white"
                            with row.td(
                                style=f"background:{color.hex};color:{fc};"
                            ) as cell:
                                print(name)
                                with cell.br():
                                    pass
                                print(color.hex.upper())

        return str(html)

    @property
    def is_dark(self) -> bool:
        return sum(to_rgb(self.bg_hex)) <= 1.5


if __name__ == "__main__":

    np.set_printoptions(precision=2, suppress=True)
    bg_hex = "#E8E8E8"

    do_fit: bool = True

    if do_fit:
        specs = [
            light_bgs := PaletteSpec(
                name="BGL",
                n_colors=(nc := 6),
                m_hinge=HingeSpec(1, 4, 1.0),
                j_hinge=HingeSpec(0.90, 0.95, 30.0),
                de_hinge=HingeSpec(0.05, 0.15, 10.0),
                hue_gap_hinge=HingeSpec(0.0, 0.1 + 2 / nc, 1.0),
            ),
            primaries := PaletteSpec(
                name="PRI",
                n_colors=(nc := 10),
                m_hinge=HingeSpec(17.5, 25.0, 1.0),
                j_hinge=HingeSpec(0.30, 0.50, 10.0),
                de_hinge=HingeSpec(0.40, 0.90, 10.0),
                hue_gap_hinge=HingeSpec(0.0, 1.5 / nc, 5.0),
            ),
            secondaries := PaletteSpec(
                name="SND",
                n_colors=(nc := 6),
                m_hinge=HingeSpec(0, 6, 1.0),
                j_hinge=HingeSpec(0.60, 0.75, 15.0),
                de_hinge=HingeSpec(0.35, 0.55, 10.0),
                hue_gap_hinge=HingeSpec(0.0, 3 / nc, 1.0),
            ),
            highlights := PaletteSpec(
                name="HL",
                n_colors=(nc := 6),
                m_hinge=HingeSpec(10.0, 12.5, 1.0),
                j_hinge=HingeSpec(0.825, 0.875, 20.0),
                de_hinge=HingeSpec(0.20, 0.20, 10.0),
                hue_gap_hinge=HingeSpec(0.0, 2 / nc, 0.5),
            ),
        ]

        spec_dict = {ps.name: ps for ps in specs}
        # spec_dict = {'test': primaries}

        scheme = ColorScheme(
            "Restraint",
            bg_hex,
            vs=NIGHT_VIEW_LIGHT,
            palettes=spec_dict,
        )

        with open("scheme.p", "wb") as f:
            pickle.dump(scheme, f)

    else:
        with open("scheme.p", "rb") as f:
            scheme = pickle.load(f)
    scheme.draw_cone()
    # scheme.draw_colors()
    out = scheme.format_colors()
    # print(out, file=open("../examples/example_scheme.html", "w"))
    print(out, file=open("test.html", "w"))
