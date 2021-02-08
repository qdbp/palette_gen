"""
Specialized color appearance model for monitor color scheme design.

Evgeny Naumov, 2020
"""

import numpy as np
from colour import XYZ_to_CIECAM02
from numba import njit

from palette_gen.fastcolors import (
    XYZ_D65,
    XYZ_to_xyY_jit,
    atan2_360,
    cos_deg,
    sin_deg,
    sRGB_to_XYZ_jit,
    xyY_E,
    xyY_to_XYZ_jit,
)

# XYZ -> sharpened LMS transforming matrix
# sharpened LMS -> Hunt–Pointer–Estévez transforming matrix
MCAT02 = np.array(
    [
        [0.7328, 0.4296, -0.1624],
        [-0.7036, 1.6975, 0.0061],
        [0.0030, 0.0136, 0.9834],
    ]
)
# precompute for single-pass LMS -> LMSc transform
MH = np.array(
    [
        [0.38971, 0.68898, -0.07868],
        [-0.22981, 1.18340, 0.04641],
        [0.00000, 0.00000, 1.00000],
    ]
)
_MH_compound = MH @ np.linalg.inv(MCAT02)
CIECAM02_hs = np.array([20.14, 90.00, 164.25, 237.53, 380.14])
CIECAM02_es = np.array([0.8, 0.7, 1.0, 1.2, 0.8])
CIECAM02_Hs = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
CIECAM02_arange = np.array([*range(len(CIECAM02_hs))])


@njit  # type: ignore
def matmul_last_axis(mat: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """
    Applies
        v = Mu
    to an arbitrarily-shaped array of vectors v.

    Parameters
    ----------
    mat: array of shape(n, m)
    vecs: array of shape(..., m)

    Returns
    -------
    array of shape (..., n)
    """

    out = np.zeros(vecs.shape[:-1] + (mat.shape[0],))
    for i in range(mat.shape[0]):
        out[..., i] = (vecs[..., :] * mat[i, :]).sum(axis=-1)
    return out


# noinspection PyPep8Naming
@njit  # type: ignore
def XYZ_to_PUNISHEDCAM_JabQMsh_jit(
    # no defaults -- perfection of bust!
    XYZ: np.ndarray,
    XYZr: np.ndarray,
    Lsw: float,
    Lb: float,
    Lmax: float,
    do_sr_interpolation: bool = True,
    do_ncb_fix: bool = True,
    do_hk_correction: bool = True,
    do_make_jab_ucs: bool = True,
    do_make_rest_ucs: bool = True,
) -> np.ndarray:
    """
    Converts XYZ to a cursed and punished version of CIECAM02-(UCS).

    No sane person should use this chimeric mishmash of dimly-understood
    research and copy-pasted formulas.

    Incorporates modifications from
    Kim, M., Jo, J.-H., Park, Y., & Lee, S.-W. (2018).
    Amendment of CIECAM02 with a technical extension to compensate
    Helmholtz-Kohlrausch effect for chromatic characterization of display
    devices. Displays.
    doi:10.1016/j.displa.2018.09.005

    Incorporates modifications from
    Park, Y., Luo, M. R., Li, C. J., & Kwak, Y. (2014).
    Refined CIECAM02 for bright surround conditions.
    Color Research & Application, 40(2), 114–124.
    doi:10.1002/col.21872

    Incorporates modifications from
    Sun, P. L., Li, H. C., & Ronnier Luo, M. (2017).
    Background luminance and subtense affects color appearance.
    Color Research & Application, 42(4), 440-449.

    Incorporates UCS transform from
    Luo, M. R., Cui, G., & Li, C. (2006).
    Uniform colour spaces based on CIECAM02 colour appearance model.
    Color Research & Application: Endorsed by Inter‐Society Color Council,
    The Colour Group (Great Britain), Canadian Society for Color,
    Color Science Association of Japan, Dutch Society for the Study of Color,
    The Swedish Colour Centre Foundation, Colour Society of Australia,
    Centre Français de la Couleur, 31(4), 320-330.

    Parameters
    ----------
    XYZ: XYZ tristimulus value of the point to convert
    XYZr: XYZ tristimulus value of the reference (adapted) white
    Lsw: absolute luminance of the reference white in the surround field, cdm-2
    Lb: gaussian-weighted luminance of the 13° background, cdm-2
    Lmax: absolute luminance of the brightest point in the 2° stimulus, cdm-2

    do_sr_interpolation: True -> interpolates S_R factors continuously.
    do_ncb_fix: True -> applies Ncb exponent fix for light backgrounds
    do_hk_correction: True -> applies extra HK correction factor to JM
    do_make_jab_ucs: True -> the UCS transform will be applied to Jab outputs
    do_make_rest_ucs: True -> implies do_make_jab_ucs. True -> every other
        correlate will be updated to reflect the UCS transform

    Returns
    -------
    all ciecam02 correlates.
    """

    ###
    # 1. PREAMBLE
    ###

    # Adapting field luminance reinterpretation due to Sun (2017)
    # the brightness of the adapting field is taken to depend on Lb which is the
    # luminance of the background. In the paper this is calculated using a
    # Gaussian kernel. This function takes it as a parameter.
    Ldw = max(Lmax, Lb)
    La = 0.5 * (Lb + Ldw / 5)
    # for way later
    Yb = max(0.04, Lb / Ldw)

    # Whitepoint correction step due to Sun (2017)
    # -- relevant for dark themes where Yb << Lmax
    # to avoid distorting light whitepoint, this is only applied if Yb < 0.2
    # Lmax (this conditional is original to this code -- done because the
    # interpolation by Sun appears overeager for lighter backgrounds)
    if 5 * Lb < Lmax:
        w = Lb / (Lb + Lmax)
        xyYr = XYZ_to_xyY_jit(XYZr)
        xyr = xyYr[..., 0:2]
        xyr *= w
        xyr += (1 - w) * xyY_E[..., 0:2]
        XYZr = xyY_to_XYZ_jit(xyr)

    # CIECAM02 surround ratio
    S_R = Lsw / Ldw

    ###
    # 1.1 SR INTERPOLATION
    # Linear interpolation of SR-derived factors due to Kim (2018)
    ###
    if do_sr_interpolation:
        F = -0.003 * S_R + 1.1474
        c = 0.023 * S_R + 0.7887
        Nc = 0.0203 * S_R + 1.2369
    else:
        if S_R > 0.2:
            F, c, Nc = 1.0, 0.69, 1.0
        elif 0 < S_R < 0.2:
            F, c, Nc = 0.9, 0.59, 0.95
        else:
            F, c, Nc = 0.8, 0.525, 0.8

    ###
    # 2. LMS TRANSFORM
    ###

    # CIECAM02 degree of chromatic adaptation
    D = F * (1 - np.exp(-(La + 42) / 92) / 3.6)

    # ...to sharpened LMS space
    LMS = matmul_last_axis(MCAT02, XYZ)
    LMSr = matmul_last_axis(MCAT02, XYZr)

    # ...perform chromatic adaptation using D factor in place
    LMS *= (XYZr[..., 1:2] / LMSr - 1) * D + 1
    LMSr *= (XYZr[..., 1:2] / LMSr - 1) * D + 1

    # ...to HPE space
    LMSp = matmul_last_axis(_MH_compound, LMS)
    LMSrp = matmul_last_axis(_MH_compound, LMSr)

    # the sources give an ambiguous description of what is to go here.
    # the 5LA factor is explained as encoding the gray-world assumption, and
    # that ideally the background luminance should go here. However, the
    # CIECAM97 standard also claims that "F_L is proportional to the luminance
    # of the adapting field", rather than that of the background, but still with
    # the 5x factor. The 5x factor, however, implies this should actually be Lw,
    # the luminance of the reference white under the test illuminant.
    # TODO to me Lb makes the most sense as the "intended" quantity here
    #   luminance adaptation, but this needs to be looked at better
    k = 1 / (Lb + 1)
    _k4 = k ** 4
    F_L = 0.2 * _k4 * Lb + 0.1 * ((1 - _k4) ** 2) * Lb ** (1 / 3)

    # noinspection DuplicatedCode
    LMS_fac = (LMSp * F_L / 100) ** 0.42
    LMSap = 400 * LMS_fac / (27.13 + LMS_fac) + 0.1
    L, M, S = LMSap[..., 0], LMSap[..., 1], LMSap[..., 2]

    # noinspection DuplicatedCode
    LMSr_fac = (LMSrp * F_L / 100) ** 0.42
    LMSrap = 400 * LMSr_fac / (27.13 + LMSr_fac) + 0.1
    Lr, Mr, Sr = LMSrap[..., 0], LMSrap[..., 1], LMSrap[..., 2]

    ###
    # 3. PSYCHOPHYSICAL CORRELATES
    ###

    a = L - (12 * M / 11) + (S / 11)
    b = (1 / 9) * (L + M - 2 * S)

    h = atan2_360(b, a)
    et = 0.25 * (cos_deg(h + 360 / np.pi) + 3.8)

    n = Yb / XYZr[..., 1]
    z = 1.48 + n ** 0.5

    # Park (2014) modifications: MobileCam-v1 exponent for Ncb
    # relevant for bright surrounds (working in daylight conditions)
    Nbb = 0.725 * n ** -0.2
    if do_ncb_fix:
        Ncb = 0.725 * n ** -0.1425
    else:
        Ncb = Nbb

    A = (2 * L + M + S / 20 - 0.305) * Nbb
    Ar = (2 * Lr + Mr + Sr / 20 - 0.305) * Nbb

    # J ∈ [0, 1]
    J = (A / Ar) ** (c * z)

    t = (
        (50_000 / 13)
        * (Nc * Ncb * et * np.sqrt(a * a + b * b))
        / (L + M + 21 * S / 20)
    )

    C = t ** 0.9 * (J ** 0.5) * (1.64 - 0.29 ** n) ** 0.73
    M = C * F_L ** 0.25

    ###
    # 3.2 second-pass HK-effect correction due to Kim (2018)
    ###
    # TODO Kim (2018) copy-pastes the HK correction functional form f1, f2 from
    #  Fairchild (1991) which was defined against L*a*b*. the raw data should be
    #  refitted natively in JMh. The sinusoid should be replaced with something
    #  from a richer space, like Kaiser (1986)'s xy
    if do_hk_correction:
        # formula adapted for J ∈ [0, 1]
        # NB. J can exceed 1.0 after this transformation -- deal with it
        f1 = 0.116 * np.abs(sin_deg(h / 2 - 45)) + 0.085
        f2 = 2.5 - 0.25 * J
        J += 0.0184 * f2 * f1 * M

    # define Q after updating J for HK-effect
    Q = (4 / c) * (J ** 0.5) * (Ar + 4) * F_L ** 0.25

    # s ∈ [0, 1]
    s = (M / Q) ** 0.5

    ###
    # 4 POSTPROCESSING
    ###
    if do_make_jab_ucs or do_make_rest_ucs:
        # formula adapted for rescaled J
        c1 = 0.007
        c2 = 0.0228
        J = (1 + 100 * c1) * J / (1 + 100 * c1 * J)
        # only Jab is changed in the output -- we keep the original M
        _Mp = np.log(1 + c2 * M) / c2
        a = _Mp * cos_deg(h)
        b = _Mp * sin_deg(h)
        if do_make_rest_ucs:
            M = _Mp
            Q = (4 / c) * (J ** 0.5) * (Ar + 4) * F_L ** 0.25
            s = (M / Q) ** 0.5
            # hue stays the same by construction

    out = np.zeros(XYZ.shape[:-1] + (7,))
    out[..., 0] = J
    # rescale to keep distance function trivial
    out[..., 1] = a / 100
    out[..., 2] = b / 100
    out[..., 3] = Q
    out[..., 4] = M
    out[..., 5] = s
    out[..., 6] = h

    return out


# noinspection PyPep8Naming
@njit  # type: ignore
def de_punished_jab(jab1: np.ndarray, jab2: np.ndarray) -> np.ndarray:
    return np.sqrt(((jab1[..., :3] - jab2[..., :3]) ** 2).sum(axis=-1))


if __name__ == "__main__":
    rgb = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    xyz = sRGB_to_XYZ_jit(rgb)

    print("reference JMh:")
    ref = XYZ_to_CIECAM02(xyz, XYZ_D65, L_A=5.0, Y_b=0.8)
    print(np.vstack([ref.J, ref.Q, ref.M, ref.h]).T)

    out = XYZ_to_PUNISHEDCAM_JabQMsh_jit(
        xyz,
        XYZ_D65,
        Lsw=1,
        Lb=4,
        Lmax=15,
    )
    out[..., 0] *= 100

    print(out[..., [0, 3, 5, 8]])
