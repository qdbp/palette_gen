# type: ignore
"""
Fast jit-compiled versions of common color conversion functions suitable for
use in optimization routines.
"""

from timeit import Timer
from typing import Any
from collections.abc import Mapping

import numpy as np
from colour import RGB_COLOURSPACES, xyY_to_XYZ
from colour.models import RGB_COLOURSPACE_sRGB
from numba import njit
from numpy.typing import NDArray

sRGB_to_XYZ_d65_mat = RGB_COLOURSPACES["sRGB"].matrix_RGB_to_XYZ
sRGB_ILL = D65_ILL = RGB_COLOURSPACE_sRGB.whitepoint

# CIE 1931, 2 degree observer
xyY_D50 = np.array([[0.3457, 0.3585, 1.00]])
XYZ_D50 = xyY_to_XYZ(xyY_D50)

# these should be more precise, but they are truncated at this point in
# the srgb standard
xyY_D65 = np.array([[0.3127, 0.3290, 1.00]])
XYZ_D65: NDArray[np.float64] = xyY_to_XYZ(xyY_D65)

xyY_E = np.array([[1 / 3, 1 / 3, 1.0]])

_LUV_DVEC: NDArray[np.float64] = np.asarray([[1.0, 15.0, 3.0]]).T
_LUV_49_VEC: NDArray[np.float64] = np.asarray([4.0, 9.0])


# noinspection PyPep8Naming
@njit
def cct_to_D_xyY_jit(T: float) -> NDArray[np.float64]:
    """
    Finds CIE D-series illuminant xyY coordinates from temperature.

    Parameters
    ----------
    T: temperature, Kelvin

    Returns
    -------
    xyY triple. Y is always 1.0.
    """

    out = np.zeros((1, 3))
    out[..., 2] = 1.0

    t = 1000 / T
    tt = t * t
    ttt = tt * t

    if 4000 <= T <= 7000:
        x = -4.6070 * ttt + 2.9678 * tt + 0.09911 * t + 0.244063
    elif 7000 < T <= 25000:
        x = -2.0064 * ttt + 1.9018 * tt + 0.24748 * t + 0.237040
    else:
        print("Invalid temperature", T, "in CCT conversion.")
        # this will break any consuming code in (hopefully) obvious ways
        out[...] = np.nan
        return out

    y = -3.0 * x * x + 2.870 * x - 0.275

    out[..., 0] = x
    out[..., 1] = y

    return out


@njit
def hk_f_kaiser(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    # noinspection SpellCheckingInspection
    """
    Calculates the "F" factor by Kaiser from xyY (x, y) coordinates.

    Donofrio, R. L. (2011). Review Paper: The Helmholtz-Kohlrausch
    effect. Journal of the Society for Information Display, 19(10), 658.
    doi:10.1889/jsid19.10.658
    """
    return 0.256 - 0.184 * y - 2.527 * x * y + 4.65 * y * x**3 + 4.657 * x * y**4


@njit
def hk_correct_lchab_inplace(lchab: NDArray[np.float64]) -> None:
    """
    Corrects L* of a L*a*b* array in-place using the Fairchild (1991) formula.

    Fairchild, M. D., & Pirrotta, E. (1991). Predicting the lightness of
    chromatic object colors using CIELAB. Color Research & Application,
    16(6), 385–393. doi:10.1002/col.5080160608
    """
    f2 = 2.5 - 0.025 * lchab[..., 0]
    f1 = 0.116 * np.abs(sin_deg(lchab[..., 2] / 2 - 45)) + 0.085
    lchab[..., 0] += f1 * f2 * lchab[..., 1]


# noinspection PyPep8Naming
@njit
def xyY_to_XYZ_jit(xyY: NDArray[np.float64]) -> NDArray[np.float64]:
    out: NDArray[np.float64] = np.zeros_like(xyY)
    x = xyY[..., 0]
    y = xyY[..., 1]
    Y = xyY[..., 2]
    _y_nonzero = (y > 1e-20).astype(np.uint8)
    out[..., 0] = _y_nonzero * x * Y / (y + 1e-30)
    out[..., 1] = Y
    out[..., 2] = _y_nonzero * (1 - x - y) * Y / (y + 1e-30)
    return out


# noinspection PyPep8Naming
@njit
def XYZ_to_xyY_jit(
    XYZ: NDArray[np.float64], black_xy: NDArray[np.float64] = xyY_D65
) -> NDArray[np.float64]:
    out: NDArray[np.float64] = np.zeros_like(XYZ)

    # keepdims doesn't work
    norm = XYZ.sum(axis=-1).reshape(-1, 1)

    out[..., 2] = XYZ[..., 1]
    zero_norm = (norm == 0.0).astype(np.uint8)
    out[..., 0:2] += zero_norm * black_xy[..., 0:2]
    out[..., 0:2] += (1 - zero_norm) * XYZ[..., 0:2] / (norm + 1e-20)
    return out


# noinspection PyPep8Naming
@njit
def XYZ_to_Luv_D65_jit(
    XYZ: NDArray[np.float64], XYZr: NDArray[np.float64] = XYZ_D65
) -> NDArray[np.float64]:
    ε = 0.008856
    κ = 903.3

    out: NDArray[np.float64] = np.zeros_like(XYZ)
    y_r: NDArray[np.float64] = XYZ[..., 1] / XYZr[..., 1]
    out[..., 0] = np.where(y_r > ε, 116 * y_r ** (1 / 3) - 16, κ * y_r)

    denom: NDArray[np.float64] = XYZ @ _LUV_DVEC
    uvp: NDArray[np.float64] = _LUV_49_VEC * XYZ[..., :2] / denom

    denom_r: NDArray[np.float64] = XYZr @ _LUV_DVEC
    uvp_r: NDArray[np.float64] = _LUV_49_VEC * XYZr[..., :2] / denom_r

    out[..., 1:] = 13 * out[..., 0:1] * (uvp - uvp_r)
    return out


# noinspection PyPep8Naming
@njit
def Luv_to_LCHuv_jit(Luv: NDArray[np.float64]) -> NDArray[np.float64]:
    out: NDArray[np.float64] = np.zeros_like(Luv)
    out[..., 0] = Luv[..., 0]
    out[..., 1] = np.sqrt(Luv[..., 1] ** 2 + Luv[..., 2] ** 2)
    out[..., 2] = atan2_360(Luv[..., 2], Luv[..., 1])
    return out


@njit
def atan2_360(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.arctan2(x, y)
    out *= 180 / np.pi
    np.maximum(out, out - 360 * np.sign(out), out)
    return out


@njit
def cos_deg(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.cos(np.deg2rad(arr))


@njit
def sin_deg(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sin(np.deg2rad(arr))


# noinspection PyPep8Naming
@njit
def XYZ_to_Lab_D65_jit(
    XYZ: NDArray[np.float64], XYZr: NDArray[np.float64] = XYZ_D65
) -> NDArray[np.float64]:
    ε = 216 / 24389
    κ = 24389 / 27

    xyz_r: NDArray[np.float64] = XYZ / XYZr
    f_xyz_r = np.where(xyz_r > ε, xyz_r ** (1 / 3), (κ * xyz_r + 16.0) / 116.0)

    out = np.zeros_like(XYZ)
    out[..., 0] = 116 * f_xyz_r[..., 1] - 16
    out[..., 1] = 500 * (f_xyz_r[..., 0] - f_xyz_r[..., 1])
    out[..., 2] = 200 * (f_xyz_r[..., 1] - f_xyz_r[..., 2])
    return out


Lab_to_LCHab_jit = Luv_to_LCHuv_jit


# noinspection PyPep8Naming
@njit
def sRGB_to_XYZ_jit(sRGB: NDArray[np.float64]) -> NDArray[np.float64]:
    srgb = np.where(sRGB < 0.04045, sRGB / 12.92, ((sRGB + 0.055) / 1.055) ** 2.4)
    out = np.zeros_like(srgb)
    # unrolled matmul to avoid unsupported dim expansion for jit
    for i in range(3):
        out[..., i] = (srgb * sRGB_to_XYZ_d65_mat[i]).sum(axis=-1)
    return out


# noinspection PyPep8Naming
@njit
def HSV_to_RGB_jit(HSV: NDArray[np.float64]) -> NDArray[np.float64]:
    hp: NDArray[np.float64] = HSV[..., 0:1] * 6
    ch: NDArray[np.float64] = HSV[..., 1:2] * HSV[..., 2:]
    x: NDArray[np.float64] = ch * (1 - np.abs(hp % 2 - 1))

    rgb = np.zeros_like(HSV)

    hp_ge0 = hp >= 0
    hp_le1 = hp <= 1
    gp_gt5 = hp > 5
    hp_le6 = hp <= 6
    hp_gt_1 = hp > 1

    rgb[..., 0:1] += ch * ((hp_ge0 & hp_le1) | (gp_gt5 & hp_le6)).astype(np.uint8)
    rgb[..., 0:1] += x * ((hp_gt_1 & (hp <= 2)) | ((hp > 4) & (hp <= 5))).astype(np.uint8)

    rgb[..., 1:2] += ch * (hp_gt_1 & (hp <= 3)).astype(np.uint8)
    rgb[..., 1:2] += x * ((hp_ge0 & hp_le1) | ((hp > 3) & (hp <= 4))).astype(np.uint8)

    rgb[..., 2:] += ch * ((hp > 3) & (hp <= 5)).astype(np.uint8)
    rgb[..., 2:] += x * (((hp > 2) & (hp <= 3)) | (gp_gt5 & hp_le6)).astype(np.uint8)

    rgb += HSV[..., 2:] - ch
    return rgb


# noinspection PyPep8Naming
@njit
def dE_2000_jit(Lab1: NDArray[np.float64], Lab2: NDArray[np.float64]) -> NDArray[np.float64]:
    L1, a1, b1 = Lab1[..., 0], Lab1[..., 1], Lab1[..., 2]
    L2, a2, b2 = Lab2[..., 0], Lab2[..., 1], Lab2[..., 2]

    C1 = np.sqrt(a1 * a1 + b1 * b1)
    C2 = np.sqrt(a2 * a2 + b2 * b2)

    Lbp = 0.5 * (L1 + L2)
    Cb = 0.5 * (C1 + C2)
    G = 0.5 * (1 - np.sqrt((Cb**7) / (Cb**7 + 25**7)))

    a1p = a1 * (1 + G)
    a2p = a2 * (1 + G)

    C1p = np.sqrt(a1p * a1p + b1 * b1)
    C2p = np.sqrt(a2p * a2p + b2 * b2)

    Cbp = 0.5 * (C1p + C2p)
    h1p = atan2_360(b1, a1p)
    h2p = atan2_360(b2, a2p)

    _ge_180 = (np.abs(h1p - h2p) > 180).astype(np.uint8)
    Hbp = _ge_180 * (h1p + h2p + 360) / 2 + (1 - _ge_180) * (h1p + h2p) / 2

    T = (
        1
        - 0.17 * cos_deg(Hbp - 30)
        + 0.24 * cos_deg(2 * Hbp)
        + 0.32 * cos_deg(3 * Hbp + 6)
        - 0.20 * cos_deg(4 * Hbp - 63)
    )

    _leq_180 = 1 - _ge_180
    _ge_180_and_2leq1 = _ge_180 * (h2p <= h1p).astype(np.uint8)

    Δhp = (
        _leq_180 * (h2p - h1p)
        + _ge_180_and_2leq1 * (h2p - h1p + 360)
        + (1 - _leq_180 - _ge_180_and_2leq1) * (h2p - h1p - 360)
    )

    ΔLp = L2 - L1
    ΔCp = C2p - C1p
    ΔHp = 2 * np.sqrt(C1p * C2p) * sin_deg(Δhp / 2)

    SL = 1 + (0.015 * (Lbp - 50) ** 2) / (np.sqrt(20 + (Lbp - 50) ** 2))
    SC = 1 + 0.045 * Cbp
    SH = 1 + 0.015 * Cbp * T

    Δθ = 30 * np.exp(-(((Hbp - 275) / 25) ** 2))
    RC = 2 * np.sqrt((Cbp**7) / (Cbp**7 + 25.0**7))
    RT = -RC * sin_deg(2 * Δθ)
    KL = KC = KH = 1

    ΔE = np.sqrt(
        (ΔLp / (KL * SL)) ** 2
        + (ΔCp / (KC * SC)) ** 2
        + (ΔHp / (KH * SH)) ** 2
        + RT * (ΔCp / (KC * SC)) * (ΔHp / (KH * SH))
    )
    return ΔE


# noinspection PyPep8Naming
@njit
def dE_2000_sRGB_D65_jit(rgb1: NDArray[np.float64], rgb2: NDArray[np.float64]) -> NDArray[np.float64]:
    return dE_2000_jit(
        XYZ_to_Lab_D65_jit(sRGB_to_XYZ_jit(rgb1)),
        XYZ_to_Lab_D65_jit(sRGB_to_XYZ_jit(rgb2)),
    )


def pretty_time(stmt: str, rows: int, glb: Mapping[str, Any]) -> None:
    n, t = Timer(stmt, globals=dict(glb)).autorange()
    if "dE_" in stmt:
        rows -= 1
    print(f"{stmt}: {t / (n * rows):.3g} seconds per call per color for {rows} rows")


# noinspection SpellCheckingInspection,PyUnusedLocal
def bench_funcs(size: int = 1000) -> None:
    hsv = globals()["__colmat_hsv"] = np.random.random(size=(size, 3))

    # ground truths
    rgb = globals()["__colmat_rgb"] = HSV_to_RGB_jit(hsv)
    xyz = globals()["__colmat_xyz"] = sRGB_to_XYZ_jit(rgb)
    globals()["__colmat_xyy"] = XYZ_to_xyY_jit(xyz)
    luv = globals()["__colmat_luv"] = XYZ_to_Luv_D65_jit(xyz)
    globals()["__colmat_lchuv"] = Luv_to_LCHuv_jit(luv)
    lab = globals()["__colmat_lab"] = XYZ_to_Lab_D65_jit(xyz)
    globals()["__colmat_lchab"] = Lab_to_LCHab_jit(lab)

    dE_2000_jit(lab[:2], lab[1:3])

    for metric in [
        "HSV_to_RGB_jit(__colmat_hsv)",
        "sRGB_to_XYZ_jit(__colmat_rgb)",
        "xyY_to_XYZ_jit(__colmat_xyy)",
        "XYZ_to_xyY_D65_jit(__colmat_xyz)",
        "XYZ_to_Luv_D65_jit(__colmat_xyz)",
        "XYZ_to_Lab_D65_jit(__colmat_xyz)",
        "Luv_to_LCHuv_jit(__colmat_luv)",
        "Lab_to_LCHab_jit(__colmat_luv)",
        "dE_2000_jit(__colmat_luv[:-1], __colmat_luv[1:])",
    ]:
        pretty_time(metric, hsv.shape[0], globals())

    while True:
        for key in globals():
            if key.startswith("__colmat"):
                del globals()[key]
                break
        else:
            break
