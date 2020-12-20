# noinspection PyPep8Naming
import pickle
from multiprocessing import Pool
from itertools import product
from string import ascii_uppercase

import matplotlib.pyplot as plt
import numpy as np
from colour import (
    HSV_to_RGB,
    Luv_to_LCHuv,
    XYZ_to_Luv,
    XYZ_to_xyY,
)
from colour.models import RGB_to_HSV, sRGB_to_XYZ
from graph_tool import Graph, GraphView, Vertex
from graph_tool.draw import graph_draw
from graph_tool.topology import max_independent_vertex_set
from matplotlib.axes import Axes, GridSpec
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from numba import jit
from numpy import frompyfunc, linspace, log10, meshgrid
from numpy.linalg import norm
from scipy.optimize import OptimizeResult, basinhopping, fsolve, shgo
from scipy.special import expit
from tqdm import tqdm

from fastcolors import (
    HSV_to_RGB_jit,
    Luv_to_LCHuv_jit,
    XYZ_to_Lab_D65_jit,
    XYZ_to_Luv_D65_jit,
    XYZ_to_xyY_D65_jit,
    dE_2000_jit,
    sRGB_ILL,
    sRGB_to_XYZ_jit,
)


# Donofrio, R. L. (2011). Review Paper: The Helmholtz-Kohlrausch effect.
# Journal of the Society for Information Display,
# 19(10), 658. doi:10.1889/jsid19.10.658
@jit
def hk_F(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (
        0.256
        - 0.184 * y
        - 2.527 * x * y
        + 4.65 * y * x ** 3
        + 4.657 * x * y ** 4
    )


nh = 5
ns = 100
nv = 100

hs = linspace(0.4, 0.6, nh)
ss = linspace(0.0, 1.0, ns)
vs = linspace(0.0, 1.0, nv)


def main(axis=0, mk_plots: bool = True):
    ill = sRGB_ILL

    HSV = np.stack(meshgrid(hs, ss, vs, indexing="ij"), axis=-1)
    RGB = HSV_to_RGB(HSV)
    XYZ = sRGB_to_XYZ(RGB)
    Luv = XYZ_to_Luv(XYZ)
    Suv = (lchuv := Luv_to_LCHuv(Luv))[..., 1]  # / lchuv[..., 0]

    xyY = XYZ_to_xyY(XYZ)

    # our secret sauce
    F = hk_F(xyY[..., 0], xyY[..., 1])
    log_Y = log10(xyY[..., -1])

    plt.close("all")

    def colspace_contour(ax, ix, col_mat, fmt: str = "%.2f", **contour_kwargs):
        col_plane = np.rollaxis(col_mat, axis)[ix]
        contour = ax.contour(col_plane, **contour_kwargs)
        ax.clabel(contour, contour.levels, fmt=fmt)

    if mk_plots:
        for ix in range(nh):
            fig: Figure = plt.figure(ix)
            fig.clear()
            ax = fig.add_subplot()

            rgb_plane = np.rollaxis(RGB, axis)[ix]
            ax.imshow(rgb_plane)

            for mat, color in zip(
                [1 + log_Y + F, Suv, Luv[..., 0]], ["black", "white", "grey"]
            ):
                colspace_contour(ax, ix, mat, colors=color)

            ax.grid()
            fig.set_size_inches(8, 8)
            fig.tight_layout()

        plt.show()
        plt.close("all")


@jit
def get_KC_scalar_loss_D65(
    hsv: np.ndarray,
    t_hue: float,
    t_bight: float,
    t_norm_chr: float,
    λ: float = 2.0,
    θ: float = 5.0,
) -> float:
    hsv = hsv.reshape(1, -1)
    rgb: np.ndarray = HSV_to_RGB_jit(hsv)
    xyz: np.ndarray = sRGB_to_XYZ_jit(rgb)
    xyy: np.ndarray = XYZ_to_xyY_D65_jit(xyz)
    luv: np.ndarray = XYZ_to_Luv_D65_jit(xyz)
    lchuv: np.ndarray = Luv_to_LCHuv_jit(luv)

    B = 1 + log10(xyy[..., 2]) + hk_F(xyy[..., 0], xyy[..., 1])
    C: np.ndarray = lchuv[..., 1] / 100

    loss_arr = np.zeros_like(hsv)
    hue_loss_arr = np.zeros_like(hsv)
    hue_loss_arr[..., 0] = np.abs(t_hue - 1.0 - hsv[..., 0])
    hue_loss_arr[..., 1] = np.abs(t_hue - hsv[..., 0])
    hue_loss_arr[..., 2] = np.abs(t_hue + 1.0 - hsv[..., 0])
    # print(hue_loss_arr)
    loss_arr[..., 0] = np.where(
        hue_loss_arr[..., 0] < hue_loss_arr[..., 1],
        hue_loss_arr[..., 0],
        hue_loss_arr[..., 1],
    )
    loss_arr[..., 0] = np.where(
        loss_arr[..., 0] > hue_loss_arr[..., 2],
        hue_loss_arr[..., 2],
        loss_arr[..., 0],
    )
    loss_arr[..., 0] *= θ
    loss_arr[..., 1] = np.abs(C - t_norm_chr)
    loss_arr[..., 2] = λ * np.abs(B - t_bight)
    loss = loss_arr.sum()
    return loss


def minimizer(hue: float, brg: float, chr: float) -> OptimizeResult:
    res = shgo(
        get_KC_scalar_loss_D65,
        bounds=[(0.001, 0.999), (0.001, 0.999), (0.001, 0.999)],
        args=(hue, brg, chr / 100),
    )
    if not res["success"]:
        print(res["message"])
    return res


def solve_uniform_color_grid() -> dict[tuple[float, float, float], str]:
    nh = 16
    nb = 8
    nc = 8

    grid = list(
        product(
            hs := np.linspace(0.0, 1.0, num=nh, endpoint=False),
            # we need "expspace"-style increment here for perceptually uniform
            # relative brightness jumps, since ΔB ~ 10^ΔF
            bs := np.log(np.linspace(1.22, 2.22, num=nb)),
            cs := np.linspace(10, 90, num=nc),
        )
    )

    with Pool(24) as pool:
        results = pool.starmap(minimizer, grid)

    extreme_loss_thresh = 0.5
    out = {}
    for (hx, bx, cx), res in zip(
        product(range(nh), range(nb), range(nc)), results
    ):
        plt.figure(bx)
        loss = res["fun"]
        hsv = res["x"]
        if loss > extreme_loss_thresh:
            print("extreme loss for", hsv, loss)
            continue

        rgb = HSV_to_RGB_jit(hsv.reshape(1, -1)).squeeze()
        plt.scatter([hs[hx]], [cs[cx]], s=200, marker="s", color=rgb)
        plt.scatter(
            [hs[hx]],
            [cs[cx]],
            s=100 * loss,
            marker="x" if loss < extreme_loss_thresh else "o",
            color="red",
        )

        out[hx, bx, cx] = rgb

    for bx in range(nb):
        fig = plt.figure(bx)
        plt.xlabel("Hue (HSV)")
        plt.ylabel("Chroma (LCHuv)")
        plt.title(f"HKB = {bs[bx]:0.2f}")
        fig.set_size_inches(6, 6)
        fig.tight_layout()

    with open("cmap.p", "wb") as f:
        pickle.dump(out, f)

    plt.show()

    return out


def name_hcb_color(hx, cx, bx):
    hue_let_ix = -hx - 1 - (1 if hx >= 11 else 0)
    return f"{ascii_uppercase[bx]}{cx}{ascii_uppercase[hue_let_ix]}"


def get_max_indep(rgbs_by_hbc):

    rgbs = np.array([*rgbs_by_hbc.values()])

    labs = XYZ_to_Lab_D65_jit(sRGB_to_XYZ_jit(rgbs))

    dmat = dE_2000_jit(labs.reshape(1, -1, 3), labs.reshape(-1, 1, 3))

    thresh = 14
    node: Vertex

    g = Graph(directed=False)
    nodes = {key: g.add_vertex() for key in rgbs_by_hbc.keys()}

    vp_color = g.new_vertex_property("vector<float>")
    for key, node in nodes.items():
        vp_color[node] = rgbs_by_hbc[key]

    for ix, (key, n1) in enumerate(nodes.items()):
        for jx, (key, n2) in enumerate(nodes.items()):
            if jx >= ix:
                continue
            if dmat[ix, jx] <= thresh:
                g.add_edge(n1, n2)

    best_res = None
    best_n_res = 0
    for i in range(100):
        res = max_independent_vertex_set(g)
        if sum(res) > best_n_res:
            if best_n_res > 0:
                print(f"improved {best_n_res} -> {sum(res)}")
            best_n_res = sum(res)
            best_res = res

    # gv = GraphView(g, vfilt=best_res)
    # graph_draw(gv, vertex_fill_color=vp_color, vertex_color=vp_color)

    palette = sorted(
        (
            (hbc, rgb)
            for is_indep, (hbc, rgb) in zip(res, rgbs_by_hbc.items())
            if is_indep
        ),
        key=lambda hr: (hr[0][1], hr[0][2], hr[0][0]),
    )[::-1]

    fig = plt.figure()
    gs = GridSpec(figure=fig, nrows=1, ncols=10)
    axp = fig.add_subplot(gs[0, :-2])
    axl = fig.add_subplot(gs[0, -2:])
    for ix, ((hx, bx, cx), rgb) in enumerate(palette):
        marker = f"$\\rm {name_hcb_color(hx, cx, bx)}$"
        axp.text(
            x := hx,
            y := 10 * cx + bx,
            s=500,
            marker=marker,
            color=rgb,
        )
        axl.scatter(0, ix, s=500, marker=marker, color=rgb)
        axl.text(
            1,
            ix,
            to_hex(rgb)[1:],
            # s=150,
            # marker=f"$\\rm {to_hex(rgb)[1:]}$",
            color="grey",
            fontsize="small",
        )

    axl.set_xlim(-1, 2.5)
    axp.set_xticks([])
    axp.set_yticks([])
    axl.set_xticks([])
    axl.set_yticks([])

    fig.set_size_inches((8, 8))
    fig.suptitle(f"Palette, {len(palette)} colors, ΔE tolerance={thresh}")
    fig.tight_layout()
    plt.savefig("../examples/example_palette.png")
    plt.show()

    # graph_draw(g, vertex_fill_color=res, vertex_color=vp_color)


if __name__ == "__main__":

    # out = solve_uniform_color_grid()
    with open("cmap.p", "rb") as f:
        out = pickle.load(f)

    get_max_indep(out)
