"""
Fast Numba-jitted versions of common color conversion functions suitable for
use in optimization routines.
"""
import itertools
from timeit import Timer

import numpy as np
from colour import (
    HSV_to_RGB,
    Lab_to_LCHab,
    Luv_to_LCHuv,
    RGB_COLOURSPACES,
    XYZ_to_Lab,
    XYZ_to_Luv,
    XYZ_to_xyY,
    sRGB_to_XYZ,
    xyY_to_XYZ,
)
from colour.difference import delta_E_CIE2000
from colour.models import RGB_COLOURSPACE_sRGB
from numba import njit

sRGB_to_XYZ_d65_mat = RGB_COLOURSPACES["sRGB"].matrix_RGB_to_XYZ
sRGB_ILL = D65_ILL = RGB_COLOURSPACE_sRGB.whitepoint

xyY_D65 = np.array([[0.3127, 0.3290, 1.00]])
XYZ_D65: np.ndarray = xyY_to_XYZ(xyY_D65)
_LUV_DVEC: np.ndarray = np.asarray([[1.0, 15.0, 3.0]]).T
_LUV_49_VEC: np.ndarray = np.asarray([4.0, 9.0])


@njit  # type: ignore
def xyY_to_XYZ_jit(xyY: np.ndarray) -> np.ndarray:
    out: np.ndarray = np.zeros_like(xyY)
    x = xyY[..., 0]
    y = xyY[..., 1]
    Y = xyY[..., 2]
    _y_nonzero = (y > 1e-12).astype(np.uint8)
    out[..., 0] = _y_nonzero * x * Y / (y + 1e-15)
    out[..., 1] = Y
    out[..., 2] = _y_nonzero * (1 - x - y) * Y / (y + 1e-15)
    return out


# noinspection PyPep8Naming
@njit  # type: ignore
def XYZ_to_xyY_D65_jit(XYZ: np.ndarray) -> np.ndarray:
    out: np.ndarray = np.zeros_like(XYZ)

    # keepdims doesn't work
    norm = XYZ.sum(axis=-1).reshape(-1, 1)

    out[..., 2] = XYZ[..., 1]
    zero_norm = (norm == 0.0).astype(np.uint8)
    out[..., 0:2] += zero_norm * xyY_D65[..., 0:2]
    out[..., 0:2] += (1 - zero_norm) * XYZ[..., 0:2] / (norm + 1e-20)
    return out


# noinspection PyPep8Naming
@njit  # type: ignore
# noinspection PyPep8Naming
def XYZ_to_Luv_D65_jit(
    XYZ: np.ndarray, XYZr: np.ndarray = XYZ_D65
) -> np.ndarray:
    ε = 0.008856
    κ = 903.3

    out: np.ndarray = np.zeros_like(XYZ)
    y_r: np.ndarray = XYZ[..., 1] / XYZr[..., 1]
    out[..., 0] = np.where(y_r > ε, 116 * y_r ** (1 / 3) - 16, κ * y_r)

    denom: np.ndarray = XYZ @ _LUV_DVEC
    uvp: np.ndarray = _LUV_49_VEC * XYZ[..., :2] / denom

    denom_r: np.ndarray = XYZr @ _LUV_DVEC
    uvp_r: np.ndarray = _LUV_49_VEC * XYZr[..., :2] / denom_r

    out[..., 1:] = 13 * out[..., 0:1] * (uvp - uvp_r)
    return out


@njit  # type: ignore
def Luv_to_LCHuv_jit(Luv: np.ndarray) -> np.ndarray:
    out: np.ndarray = np.zeros_like(Luv)
    out[..., 0] = Luv[..., 0]
    out[..., 1] = np.sqrt(Luv[..., 1] ** 2 + Luv[..., 2] ** 2)
    out[..., 2] = atan2_360(Luv[..., 2], Luv[..., 1])
    # out[..., 2] = 180 * np.arctan2(Luv[..., 2], Luv[..., 1]) / np.pi
    # np.maximum(
    #     out[..., 2],
    #     out[..., 2] - 360 * np.sign(out[..., 2]),
    #     out[..., 2],
    # )
    return out


@njit  # type: ignore
def atan2_360(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.arctan2(x, y)
    out *= 180 / np.pi
    np.maximum(out, out - 360 * np.sign(out), out)
    return out


@njit  # type: ignore
def cosdeg(arr: np.ndarray) -> np.ndarray:
    return np.cos(np.deg2rad(arr))


@njit  # type: ignore
def sindeg(arr: np.ndarray) -> np.ndarray:
    return np.sin(np.deg2rad(arr))


@njit  # type: ignore
# noinspection PyPep8Naming
def XYZ_to_Lab_D65_jit(
    XYZ: np.ndarray, XYZr: np.ndarray = XYZ_D65
) -> np.ndarray:

    ε = 216 / 24389
    κ = 24389 / 27

    xyz_r: np.ndarray = XYZ / XYZr
    f_xyz_r = np.where(xyz_r > ε, xyz_r ** (1 / 3), (κ * xyz_r + 16.0) / 116.0)

    out = np.zeros_like(XYZ)
    out[..., 0] = 116 * f_xyz_r[..., 1] - 16
    out[..., 1] = 500 * (f_xyz_r[..., 0] - f_xyz_r[..., 1])
    out[..., 2] = 200 * (f_xyz_r[..., 1] - f_xyz_r[..., 2])
    return out


Lab_to_LCHab_jit = Luv_to_LCHuv_jit


# noinspection PyPep8Naming
@njit  # type: ignore
def sRGB_to_XYZ_jit(sRGB: np.ndarray) -> np.ndarray:
    srgb = np.where(
        sRGB < 0.04045, sRGB / 12.92, ((sRGB + 0.055) / 1.055) ** 2.4
    )
    out = np.zeros_like(srgb)
    # unrolled matmul to avoid unsupported dim expansion for jit
    out[..., 0] = srgb @ sRGB_to_XYZ_d65_mat[0]
    out[..., 1] = srgb @ sRGB_to_XYZ_d65_mat[1]
    out[..., 2] = srgb @ sRGB_to_XYZ_d65_mat[2]
    return out


# noinspection PyPep8Naming
@njit  # type: ignore
def HSV_to_RGB_jit(HSV: np.ndarray) -> np.ndarray:
    hp: np.ndarray = HSV[..., 0:1] * 6
    ch: np.ndarray = HSV[..., 1:2] * HSV[..., 2:]
    x: np.ndarray = ch * (1 - np.abs(hp % 2 - 1))

    rgb = np.zeros_like(HSV)

    rgb[..., 0:1] += ch * (
        ((0 <= hp) & (hp <= 1)) | ((5 < hp) & (hp <= 6))
    ).astype(np.uint8)
    rgb[..., 0:1] += x * (
        ((1 < hp) & (hp <= 2)) | ((4 < hp) & (hp <= 5))
    ).astype(np.uint8)

    rgb[..., 1:2] += ch * ((1 < hp) & (hp <= 3)).astype(np.uint8)
    rgb[..., 1:2] += x * (
        ((0 <= hp) & (hp <= 1)) | ((3 < hp) & (hp <= 4))
    ).astype(np.uint8)

    rgb[..., 2:] += ch * ((3 < hp) & (hp <= 5)).astype(np.uint8)
    rgb[..., 2:] += x * (
        ((2 < hp) & (hp <= 3)) | ((5 < hp) & (hp <= 6))
    ).astype(np.uint8)

    rgb += HSV[..., 2:] - ch
    return rgb


# noinspection PyPep8Naming
@njit  # type: ignore
def dE_2000_jit(Lab1: np.ndarray, Lab2: np.ndarray) -> np.ndarray:
    L1, a1, b1 = Lab1[..., 0], Lab1[..., 1], Lab1[..., 2]
    L2, a2, b2 = Lab2[..., 0], Lab2[..., 1], Lab2[..., 2]

    C1 = np.sqrt(a1 * a1 + b1 * b1)
    C2 = np.sqrt(a2 * a2 + b2 * b2)

    Lbp = 0.5 * (L1 + L2)
    Cb = 0.5 * (C1 + C2)
    G = 0.5 * (1 - np.sqrt((Cb ** 7) / (Cb ** 7 + 25 ** 7)))

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
        - 0.17 * cosdeg(Hbp - 30)
        + 0.24 * cosdeg(2 * Hbp)
        + 0.32 * cosdeg(3 * Hbp + 6)
        - 0.20 * cosdeg(4 * Hbp - 63)
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
    ΔHp = 2 * np.sqrt(C1p * C2p) * sindeg(Δhp / 2)

    SL = 1 + (0.015 * (Lbp - 50) ** 2) / (np.sqrt(20 + (Lbp - 50) ** 2))
    SC = 1 + 0.045 * Cbp
    SH = 1 + 0.015 * Cbp * T

    Δθ = 30 * np.exp(-(((Hbp - 275) / 25) ** 2))
    RC = 2 * np.sqrt((Cbp ** 7) / (Cbp ** 7 + 25.0 ** 7))
    RT = -RC * sindeg(2 * Δθ)
    KL = KC = KH = 1

    ΔE = np.sqrt(
        (ΔLp / (KL * SL)) ** 2
        + (ΔCp / (KC * SC)) ** 2
        + (ΔHp / (KH * SH)) ** 2
        + RT * (ΔCp / (KC * SC)) * (ΔHp / (KH * SH))
    )
    return ΔE


def prettytime(stmt: str, rows: int, globals):
    n, t = Timer(stmt, globals=globals).autorange()
    if 'dE_' in stmt:
        rows -= 1
    print(
        f"{stmt}: {t / (n * rows):.3g} "
        f"seconds per call per color for {rows} rows"
    )


def bench_funcs(size: int = 1000) -> None:

    hsv = globals()["__colmat_hsv"] = np.random.random(size=(size, 3))

    # ground truths
    rgb = globals()["__colmat_rgb"] = HSV_to_RGB_jit(hsv)
    xyz = globals()["__colmat_xyz"] = sRGB_to_XYZ_jit(rgb)
    xyy = globals()["__colmat_xyy"] = XYZ_to_xyY_D65_jit(xyz)
    luv = globals()["__colmat_luv"] = XYZ_to_Luv_D65_jit(xyz)
    lchuv = globals()["__colmat_lchuv"] = Luv_to_LCHuv_jit(luv)
    lab = globals()["__colmat_lab"] = XYZ_to_Lab_D65_jit(xyz)
    lchab = globals()["__colmat_lchab"] = Lab_to_LCHab_jit(lab)

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
        prettytime(metric, hsv.shape[0], globals())

    while True:
        for key in globals():
            if key.startswith("__colmat"):
                del globals()[key]
                break
        else:
            break


if __name__ == "__main__":
    bench_funcs(size=10_000)
