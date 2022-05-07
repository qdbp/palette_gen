import itertools

import numpy as np
from colour import (
    HSV_to_RGB,
    Lab_to_LCHab,
    Luv_to_LCHuv,
    XYZ_to_Lab,
    XYZ_to_Luv,
    XYZ_to_xyY,
    sRGB_to_XYZ,
)
from colour.difference import delta_E_CIE2000
from numpy.typing import NDArray

from palette_gen.fastcolors import (
    HSV_to_RGB_jit,
    Lab_to_LCHab_jit,
    Luv_to_LCHuv_jit,
    XYZ_to_Lab_D65_jit,
    XYZ_to_Luv_D65_jit,
    XYZ_to_xyY_jit,
    cct_to_D_xyY_jit,
    dE_2000_jit,
    sRGB_to_XYZ_jit,
    xyY_D50,
    xyY_D65,
    xyY_to_XYZ_jit,
)


def get_random_01x01x01_with_corners(size: int = 1000) -> NDArray[np.float64]:

    hsv = np.random.random(size=(size, 3))
    # (literal) corner cases -- important!
    edges = np.array([*itertools.product([0, 1], repeat=3)])
    return np.concatenate([hsv, edges])


def test_converters():
    np.set_printoptions(precision=4)

    hsv = get_random_01x01x01_with_corners(size=10)

    # ground truths
    rgb = HSV_to_RGB(hsv)
    xyz = sRGB_to_XYZ(rgb)
    xyy = XYZ_to_xyY(xyz)
    luv = XYZ_to_Luv(xyz)
    lchuv = Luv_to_LCHuv(luv)
    lab = XYZ_to_Lab(xyz)
    lchab = Lab_to_LCHab(lab)

    for fun, src, trg in [
        [HSV_to_RGB_jit, hsv, rgb],
        [sRGB_to_XYZ_jit, rgb, xyz],
        [xyY_to_XYZ_jit, xyy, xyz],
        [XYZ_to_xyY_jit, xyz, xyy],
        [XYZ_to_Luv_D65_jit, xyz, luv],
        [XYZ_to_Lab_D65_jit, xyz, lab],
        [Luv_to_LCHuv_jit, luv, lchuv],
        [Lab_to_LCHab_jit, lab, lchab],
    ]:
        print(f"testing {fun.__name__}")
        my_trg = fun(src)
        succ = np.allclose(my_trg, trg, equal_nan=True, atol=1e-3)
        if not succ:
            bad_rows = (
                (~np.isclose(my_trg, trg, equal_nan=True, atol=1e-3))
                .sum(axis=-1)
                .astype(bool)
            )
            print("FAILED")
            print("output comparison (bad rows)")
            print(np.hstack([my_trg, trg])[bad_rows])
            print("absolute difference (bad rows)")
            print(np.hstack([src, np.abs(my_trg - trg)])[bad_rows])
            assert 0


def test_de2000() -> None:

    rgb = get_random_01x01x01_with_corners(1000)
    xyz = sRGB_to_XYZ(rgb)
    lab = XYZ_to_Lab(xyz)

    de = delta_E_CIE2000(lab[:-1], lab[1:])
    my_de = dE_2000_jit(lab[:-1], lab[1:])

    assert np.allclose(de, my_de), np.hstack([de.reshape(-1, 1), my_de.reshape(-1, 1)])


def test_cct() -> None:
    np.set_printoptions(precision=5)

    assert np.allclose(
        cct_to_D_xyY_jit(3000),
        np.array([np.nan, np.nan, np.nan]),
        equal_nan=True,
    )

    # low precision because standardized illuminants are not very precise
    assert np.allclose(cct_to_D_xyY_jit(5003), xyY_D50, atol=1e-3)
    assert np.allclose(cct_to_D_xyY_jit(6504), xyY_D65, atol=1e-3)


def test_ciecam02():
    pass
