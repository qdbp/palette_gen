from itertools import chain
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_hex, to_rgb
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray

from palette_gen.idiotic_html_generator import HTML
from palette_gen.solvers import JabColor, ViewingSpec
from palette_gen.solvers.color import ColorSolver


class PaletteSolver:
    def __init__(self, name: str, vs: ViewingSpec, palette_spec: dict[str, ColorSolver]) -> None:
        self.name = name
        self.vs = vs
        self.p_specs = palette_spec
        self.colors_dict = self._solve()

    def _solve(self) -> dict[str, list[JabColor]]:
        all_colors = {}
        for name, spec in self.p_specs.items():
            solved_colors = spec.solve_for_context(self.vs.bg_hex, self.vs)
            # don't sort here -- defer to organization defined by solver
            if isinstance(solved_colors, list):
                all_colors[name] = solved_colors.copy()
            else:
                for key, sublist in solved_colors.items():
                    if key in all_colors:
                        raise ValueError(
                            f"Palette-generated key {key} conflicts with anexisting colorset."
                        )
                    all_colors[key] = sublist.copy()

        return all_colors

    @staticmethod
    def mk_cube_surface_grid(pps: int = 20) -> NDArray[np.float64]:
        """
        Generates a regular grid over the surface of the {0,1}^3 corner unit
        cube.
        """

        base = np.linspace(1 / pps, 1 - 1 / pps, num=pps - 2)
        edges = np.array([0.0, 1.0])
        cube_dim = 3

        return np.concatenate(  # type: ignore
            [
                np.stack(
                    np.meshgrid(  # type: ignore
                        *([base] * ix + [edges] + [base] * (cube_dim - ix - 1)),
                        indexing="ij",
                    ),
                    axis=-1,
                ).reshape(-1, 3)
                for ix in range(cube_dim)
            ]
        )

    def draw_cone(self) -> Any:
        try:
            import plotly.graph_objects as go
        except ImportError as e:
            raise RuntimeError("Drawing the cone requires plotly.") from e

        marker_cycle = ["circle", "square", "diamond", "x"]

        jab_arr = np.array([color.jab for color in chain.from_iterable(self.colors_dict.values())])
        rgb_arr = np.array([color.rgb for color in chain.from_iterable(self.colors_dict.values())])

        symbols = []
        text = []
        for px, (name, colors) in enumerate(self.colors_dict.items()):
            symbols.extend([marker_cycle[px % len(marker_cycle)]] * len(colors))
            text.extend([f"{name.upper()}{cx}" for cx in range(len(colors))])

        jab_edges = self.vs.rgb_to_cam(self.mk_cube_surface_grid())

        return go.Figure(
            data=[
                go.Scatter3d(
                    x=jab_arr[..., 1],
                    y=jab_arr[..., 2],
                    z=jab_arr[..., 0],
                    marker={"color": list(map(to_hex, rgb_arr)), "size": 6, "symbol": symbols},
                    mode="markers",
                    text=text,
                ),
                go.Scatter3d(
                    x=jab_edges[:, 1],
                    y=jab_edges[:, 2],
                    z=jab_edges[:, 0],
                    marker={"color": "black", "size": 1},
                    mode="markers",
                ),
            ]
        )

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
                axl.text(1, gx, color.hex, color="grey", fontsize="small")
                gx += 1

        axl.set_xlim(-1, 2.5)
        for ax in [axl, axp]:
            ax.set_facecolor(self.vs.bg_hex)

        plt.suptitle(f"{self.name} color scheme colors.")
        plt.show()

    def dump_html(self) -> str:
        html = HTML()

        with html as h, h.body(style=f"background-color:{self.vs.bg_hex};") as b, b.table() as t:
            for key, colors in self.colors_dict.items():
                with t.tr() as row:
                    for cx, color in enumerate(colors):
                        seq_name = f"{key.upper()}{cx:02d}"
                        if sum(color.rgb) > 1.5:
                            fc = "black"
                        else:
                            fc = "white"
                        with row.td(style=f"background:{color.hex};color:{fc};") as cell:
                            print(seq_name)
                            if color.name is not None:
                                with cell.br():
                                    pass
                                print(color.name)
                            with cell.br():
                                pass
                            print(color.hex.upper())

        return str(html)

    def serialize(self) -> dict[str, Any]:
        p_dict = {
            p_name: [
                {
                    "name": f"{p_name}{cx:03d}" if c.name is None else c.name,
                    "hex": c.hex,
                }
                for cx, c in enumerate(colors)
            ]
            for p_name, colors in self.colors_dict.items()
        }
        return {"palette": p_dict}

    @property
    def is_dark(self) -> bool:
        return sum(to_rgb(self.vs.bg_hex)) <= 1.5
