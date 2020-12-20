import numpy as np
import itertools

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

from palette_gen.fastcolors import (
    HSV_to_RGB_jit,
    Lab_to_LCHab_jit,
    Luv_to_LCHuv_jit,
    XYZ_to_Lab_D65_jit,
    XYZ_to_Luv_D65_jit,
    XYZ_to_xyY_D65_jit,
    dE_2000_jit,
    sRGB_to_XYZ_jit,
    xyY_to_XYZ_jit,
)


def test_converters():
    np.set_printoptions(precision=3)

    hsv = np.random.random(size=(1_000, 3))
    # (literal) corner cases -- important!
    edges = np.array([*itertools.product([0, 1], repeat=3)])
    hsv = np.concatenate([hsv, edges])

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
        [XYZ_to_xyY_D65_jit, xyz, xyy],
        [XYZ_to_Luv_D65_jit, xyz, luv],
        [XYZ_to_Lab_D65_jit, xyz, lab],
        [Luv_to_LCHuv_jit, luv, lchuv],
        [Lab_to_LCHab_jit, lab, lchab],
    ]:
        print(f"testing {fun.__name__}")
        my_trg = fun(src)
        succ = np.allclose(my_trg, trg, equal_nan=True)
        if not succ:
            print("FAILED")
            print(np.hstack([my_trg, trg]))
            print(np.hstack([src, np.abs(my_trg - trg)]))
            assert 0

    de = delta_E_CIE2000(lab[:-1], lab[1:])
    my_de = dE_2000_jit(lab[:-1], lab[1:])
    assert np.allclose(de, my_de), np.hstack(
        [de.reshape(-1, 1), my_de.reshape(-1, 1)]
    )
