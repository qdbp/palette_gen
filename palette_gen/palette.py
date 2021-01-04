from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import cpu_count
from typing import Any, Iterable, Literal, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
from graph_tool import Graph
from graph_tool.topology import max_independent_vertex_set
from matplotlib.axes import GridSpec
from matplotlib.colors import to_hex, to_rgb
from matplotlib.gridspec import GridSpecFromSubplotSpec, SubplotSpec
from pulp import (
    COIN_CMD,
    LpBinary,
    LpContinuous,
    LpMaximize,
    LpProblem,
    LpVariable,
)
from tqdm import trange

from .base_colors import (
    ColorSpecMap,
    N_BRIGHT,
    N_CFUL,
    filter_hsb_by_de,
    name_hsb_color,
    parse_hcb_color,
)
from .fastcolors import dE_2000_sRGB_D65_jit


@dataclass()
class ColorScheme:

    name: str
    bg_hex: str

    # ultras -- errors, unique/dangerous values, eye abuse, etc.
    ultra_brg: tuple[int, int]
    ultras_sat: tuple[int, int]

    # primaries -- code
    prim_brg: tuple[int, int]
    prim_sat: tuple[int, int]

    # secondaries -- comments, docstrings, line numbers, etc.
    sec_brg: tuple[int, int]
    sec_sat: tuple[int, int]

    # highlights -- diffs, errors, etc.
    hl_brg: tuple[int, int]
    hl_sat: tuple[int, int]

    # backgrounds -- stripes, selected lines, etc.
    bg_brg: tuple[int, int]
    bg_sat: tuple[int, int]

    extra_gen_kwargs: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.bg_rgb = np.array(to_rgb(self.bg_hex))

    @property
    def is_dark(self) -> bool:
        return sum(to_rgb(self.bg_hex)) <= 1.5

    def _generate_subspec(
        self,
        all_colors: ColorSpecMap,
        name: str,
        b_range: tuple[int, int],
        s_range: tuple[int, int],
        tolerance: int = 15,
        *,
        max_n_colors: int = 12,
        min_de: float = 5.0,
        max_de: float = 100.0,
    ) -> None:

        print(f"Solving for {self.name}:{name}...")

        gen_kwargs = dict(milp_away_from=self.bg_hex) | (
            self.extra_gen_kwargs or {}
        )

        spec = PaletteSpec(
            f"{self.name}_{name}",
            filter_hsb_by_de(
                all_colors, self.bg_rgb, min_de=min_de, max_de=max_de
            ),
            tolerance,
            target_n_colors=max_n_colors,
            b_range=b_range,
            s_range=s_range,
        )
        spec.gen_palette(**gen_kwargs).plot_palette(
            fn=spec.name, background=self.bg_hex
        )

    def generate(
        self,
        all_colors: ColorSpecMap,
        do_ultras: bool = True,
        do_primaries: bool = True,
        do_secondaries: bool = True,
        do_highlights: bool = True,
        do_backgrounds: bool = True,
    ) -> None:

        print(f"Building color scheme {self.name}:")

        if do_ultras:
            self._generate_subspec(
                all_colors,
                "ultras",
                self.ultra_brg,
                self.ultras_sat,
                tolerance=20,
                max_n_colors=8,
                min_de=30,
            )

        if do_primaries:
            self._generate_subspec(
                all_colors,
                "primaries",
                self.prim_brg,
                self.prim_sat,
                tolerance=16,
                max_n_colors=16,
                min_de=30,
                max_de=50,
            )

        if do_secondaries:
            self._generate_subspec(
                all_colors,
                "secondaries",
                self.sec_brg,
                self.sec_sat,
                tolerance=14,
                max_n_colors=16,
                min_de=15,
                max_de=29,
            )

        if do_highlights:
            self._generate_subspec(
                all_colors,
                "highlights",
                self.hl_brg,
                self.hl_sat,
                tolerance=14,
                max_n_colors=8,
                min_de=20,
                max_de=35,
            )

        if do_backgrounds:
            self._generate_subspec(
                all_colors,
                "backgrounds",
                self.bg_brg,
                self.bg_sat,
                tolerance=9,
                max_n_colors=8,
                min_de=5,
                max_de=15,
            )


class Palette:
    def __init__(self, colors: ColorSpecMap, spec: PaletteSpec):
        self.colors = {
            k: v
            for k, v in sorted(
                ((hbc, rgb) for hbc, rgb in colors.items()),
                key=lambda kv: (kv[0][1], kv[0][2], kv[0][0]),
            )[::-1]
        }
        self.spec = spec

    def plot_palette(
        self,
        fn: str = None,
        background: str = None,
        subplot_spec: SubplotSpec = None,
    ) -> None:

        if subplot_spec is None:
            fig = plt.figure()
            gs = GridSpec(figure=fig, nrows=1, ncols=10)
        else:
            # noinspection PyProtectedMember
            fig = subplot_spec._gridspec.figure
            gs = GridSpecFromSubplotSpec(
                nrows=1,
                ncols=10,
                subplot_spec=subplot_spec,
            )

        axp = fig.add_subplot(gs[0, :-2])
        axl = fig.add_subplot(gs[0, -2:])

        for ix, ((hx, sx, bx), rgb) in enumerate(self.colors.items()):
            marker = f"$\\rm {name_hsb_color(hx, sx, bx)}$"
            axp.scatter(hx, 10 * bx + sx, s=500, marker=marker, color=rgb)
            axl.scatter(0, ix, s=500, marker=marker, color=rgb)
            axl.text(1, ix, to_hex(rgb)[1:], color="black", fontsize="small")

        axl.set_xlim(-1, 2.5)
        axp.set_xticks([])
        axp.set_yticks([])
        axl.set_xticks([])
        axl.set_yticks([])

        if background is not None:
            axp.set_facecolor(background)
            axl.set_facecolor(background)

        if fn is None:
            fn = f"palette_{self.spec.name}.png"

        if subplot_spec is None:
            fig.set_size_inches((8, 8))
            fig.suptitle(
                f"Palette '{self.spec.name}', {len(self)} colors, "
                f"threshold {self.spec.de_threshold}"
            )
            fig.tight_layout()
            plt.savefig(fn)

    def __len__(self) -> int:
        return len(self.colors)


class PaletteSpec:
    """
    Specifies optimizer parameters for generating a specialized palette.
    """

    def __init__(
        self,
        name: str,
        base_colors: ColorSpecMap,
        de_threshold: int,
        target_n_colors: int = 100,
        min_n_colors: int = 5,
        b_range: tuple[int, int] = (0, N_BRIGHT),
        s_range: tuple[int, int] = (0, N_CFUL),
        solver: Literal["mis", "milp_max_n", "milp_max_d"] = "milp_max_d",
    ) -> None:

        self.name = name
        self.all_colors = base_colors
        self.b_range = b_range
        self.s_range = s_range

        self.target_n_colors = target_n_colors
        self.min_n_colors = min_n_colors
        self.de_threshold = de_threshold
        self.solver = solver

        self.valid_colors = {
            key: rgb
            for key, rgb in self.all_colors.items()
            for (hx, sx, bx) in [key]
            if self.b_range[0] <= bx <= self.b_range[1]
            and self.s_range[0] <= sx <= self.s_range[1]
        }

        if len(self.valid_colors) == 0:
            raise ValueError("Palette constraints exclude all base colors!")

        rgbs = np.array([*self.valid_colors.values()])

        self.pairwise_de = dE_2000_sRGB_D65_jit(
            rgbs.reshape((1, -1, 3)), rgbs.reshape((-1, 1, 3))
        )

    def gen_palette(self, **solver_kwargs: Any) -> Palette:

        print(
            "Solving for the best palette with "
            f"ΔEij >= {self.de_threshold:.1f} "
            f"with up to {self.target_n_colors} colors"
        )

        if self.solver == "mis":
            solved_colors = self._solve_mis(**solver_kwargs)
        elif self.solver == "milp_max_n":
            solved_colors = self._solve_milp_maxcolors(**solver_kwargs)
        elif self.solver == "milp_max_d":
            solved_colors = self._solve_milp_maxdistance(**solver_kwargs)
        else:
            raise ValueError(f"Invalid solver {self.solver}!")

        out = Palette(colors=solved_colors, spec=self)
        print(f"Found palette with {len(out)} colors.")
        return out

    def _solve_milp_maxcolors(
        self,
        milp_for_mode: Literal["dark", "light"] = "light",
        milp_away_from: str = None,
        **kwargs: Any,
    ) -> ColorSpecMap:
        from pulp_lparray import lparray

        print(f"Solving for  using MILP")

        prob = LpProblem("MaxPalette", sense=LpMaximize)
        n_colors = len(self.valid_colors)

        w = lparray.create_anon("Included", shape=(n_colors,), cat=LpBinary)
        not_w = 1 - w

        # big M -> max(dE) = 100, so this is valid for all situations
        big_m = 100
        d_eff: lparray = (
            big_m
            * (not_w[None, :] + not_w[:, None] + np.diag(2 * np.ones(n_colors)))
            + self.pairwise_de
        )

        print("Warm-starting with MIS")
        mis_rgbs = self._solve_mis()
        print(len(mis_rgbs), self.de_threshold)

        # undo overeager constriction
        if len(mis_rgbs) < self.de_threshold:
            self.de_threshold -= 1

        for wx, key in enumerate(self.valid_colors.keys()):
            if key in mis_rgbs:
                w[wx].setInitialValue(1)
            else:
                w[wx].setInitialValue(0)

        # maximize palette size
        obj = w.sum()
        # add auxiliary ordering loss
        rgbs = np.array([*self.valid_colors.values()])
        lightness = rgbs.sum(axis=-1) / 3

        # maximize total distance from reference
        if milp_away_from is None:
            if milp_for_mode == "dark":
                obj += w @ lightness / (len(w) * 2)
            else:
                obj += w @ (1 - lightness) / (len(w) * 2)
        else:
            rgb = np.array(to_rgb(milp_away_from)).reshape(-1, 3)
            d_from = dE_2000_sRGB_D65_jit(rgb, rgbs)
            obj += w @ d_from.squeeze() / (len(w) * 200)

        prob += obj.item()

        (w.sum() <= self.target_n_colors).constrain(prob, "MaxNColors")

        # subject to minimum distance
        (d_eff >= self.de_threshold).constrain(prob, "MinPairwiseDE")

        COIN_CMD(msg=True, threads=cpu_count() - 1, warmStart=True).solve(prob)

        return {
            k: v
            for include, (k, v) in zip(w.values, self.valid_colors.items())
            if include
        }

    def _solve_milp_maxdistance(
        self,
        milp_away_from: str = "#000000",
        milp_ref_alpha: float = 1.0,
        **kwargs: Any,
    ) -> ColorSpecMap:
        from pulp_lparray import lparray

        print(f"Solving for max distance using MILP, max-dist objective.")

        prob = LpProblem("MaxPalette", sense=LpMaximize)
        n_colors = len(self.valid_colors)

        w = lparray.create_anon("Included", shape=(n_colors,), cat=LpBinary)
        dist_floor = lparray.create_anon("DistFloor", (1,), cat=LpContinuous)

        not_w = 1 - w

        # big M -> max(dE) = 100, so this is valid for all situations
        big_m = 100
        d_eff: lparray = (
            big_m
            * (not_w[None, :] + not_w[:, None] + np.diag(2 * np.ones(n_colors)))
            + self.pairwise_de
        )

        # maximize min pairwise distance
        (d_eff >= dist_floor).constrain(prob, "DMatBoundBind")
        (dist_floor >= self.de_threshold).constrain(prob, "MinDistFloor")

        obj = dist_floor[0]

        # maximize min distance from reference
        rgbs = np.array([*self.valid_colors.values()])
        rgb = np.array(to_rgb(milp_away_from)).reshape(-1, 3)
        d_from_ref = dE_2000_sRGB_D65_jit(rgb, rgbs).squeeze()
        tot_dist: lparray = (
            milp_ref_alpha * w @ d_from_ref / self.target_n_colors
        )
        obj += tot_dist.item()

        # strongly prefer finding new colors
        obj += 100 * w.sum().item()

        prob += obj

        (w.sum() <= self.target_n_colors).constrain(prob, "MaxNColors")
        # (w.sum() >= self.min_n_colors).constrain(prob, "MinNColors")

        COIN_CMD(msg=True, threads=cpu_count() - 1, warmStart=True).solve(prob)

        min_ref_dist = (w.values * d_from_ref)[w.values > 0].min()
        print(
            f"Found solution with "
            f"pairwise dist floor {dist_floor[0].value():.2f}, "
            f"min ref dist {min_ref_dist:.2f}"
        )

        return {
            k: v
            for include, (k, v) in zip(w.values, self.valid_colors.items())
            if include
        }

    def _solve_mis(self, mis_effort: int = 10_000) -> ColorSpecMap:
        print(f"Solving using MIS with effort={mis_effort}")

        best_n_res = 0
        too_many_colors = False
        best_res = None

        g = Graph(directed=False)
        nodes = {key: g.add_vertex() for key in self.valid_colors.keys()}

        for ix, n1 in enumerate(nodes.values()):
            for jx, n2 in enumerate(nodes.values()):
                if jx >= ix:
                    continue
                if self.pairwise_de[ix, jx] <= self.de_threshold:
                    g.add_edge(n1, n2)

        for _ in (progress := trange(mis_effort)) :
            res = max_independent_vertex_set(g)
            if sum(res) > best_n_res:
                if best_n_res > 0:
                    progress.set_description(
                        f"best palette so far: {sum(res)} colors"
                    )
                    print("")
                best_n_res = sum(res)
                best_res = res

                if best_n_res > self.target_n_colors:
                    too_many_colors = True
                    break

        if too_many_colors:
            self.de_threshold += 1
            print(
                f"Found more colors than target, permanently increasing "
                f"thresh to {self.de_threshold} in place."
            )
            return self._solve_mis(mis_effort=mis_effort)
        else:
            assert best_res is not None
            return {
                k: v
                for include, (k, v) in zip(best_res, self.valid_colors.items())
                if include
            }


def dump_colors(
    colors: ColorSpecMap,
    fn: str,
    b_range: tuple[int, int] = (0, N_BRIGHT),
    c_range: tuple[int, int] = (0, N_CFUL),
) -> None:
    fig, ax = plt.subplots()
    for (hx, sx, bx), rgb in colors.items():
        if not (b_range[0] <= bx <= b_range[1]) and (
            c_range[0] <= sx <= c_range[1]
        ):
            continue
        print(hx, sx, bx)
        ax.scatter(
            [hx],
            [10 * bx + sx],
            s=500,
            marker=f"$\\rm {name_hsb_color(hx, sx, bx)}$",
            color=rgb,
        )

    plt.title("Full HCB color set.")
    plt.savefig(fn)
    plt.show()
    pass


def lookup_colors(colors: ColorSpecMap, hcb_name: str) -> str:
    return cast(str, to_hex(parse_hcb_color(colors, hcb_name)))
