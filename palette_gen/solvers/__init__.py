from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Optional, TypeVar, cast

import numpy as np
from matplotlib.colors import to_hex

from palette_gen.fastcolors import (  # type: ignore
    cct_to_D_xyY_jit,
    sRGB_to_XYZ_jit,
    xyY_to_XYZ_jit,
)
from palette_gen.punishedcam import (  # type: ignore
    XYZ_to_PUNISHEDCAM_JabQMsh_jit,
)

T = TypeVar("T")


@dataclass()
class ViewingSpec:
    name: str
    T: float  # K
    Lsw: float  # cdm-2
    Lb: float  # cdm-2
    Lmax: float  # cdm-2
    bg_hex: str

    # noinspection PyPep8Naming
    @property
    def XYZw(self) -> np.ndarray:
        return xyY_to_XYZ_jit(cct_to_D_xyY_jit(self.T))  # type: ignore

    # TODO should be customizable based on additional fields
    # noinspection PyMethodMayBeStatic
    def rgb_to_xyz(self, rgb: np.ndarray) -> np.ndarray:
        return sRGB_to_XYZ_jit(rgb.reshape((-1, 3)))  # type: ignore

    def xyz_to_cam(self, xyz: np.ndarray) -> np.ndarray:
        return XYZ_to_PUNISHEDCAM_JabQMsh_jit(  # type: ignore
            xyz,
            self.XYZw.reshape(1, -1),
            Lsw=self.Lsw,
            Lb=self.Lb,
            Lmax=self.Lmax,
        )

    def rgb_to_cam(self, rgb: np.ndarray) -> np.ndarray:
        return self.xyz_to_cam(self.rgb_to_xyz(rgb)).squeeze()


@dataclass(frozen=True, order=True)
class Color:
    rgb: tuple[float, float, float]
    vs: ViewingSpec

    name: Optional[str] = None

    @cached_property
    def jab(self) -> np.ndarray:
        return self.vs.xyz_to_cam(  # type: ignore
            self.vs.rgb_to_xyz(np.array(self.rgb))
        )[0, :3]

    @cached_property
    def hex(self) -> str:
        return cast(str, to_hex(self.rgb))

    def __str__(self) -> str:
        return (
            f"Color{f'({self.name})' if self.name else ''}"
            f"[#{to_hex(self.rgb)}/{self.vs.name}]"
        )
