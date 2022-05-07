from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import TypeVar, cast

import numpy as np
from matplotlib.colors import to_hex, to_rgb
from numpy.typing import NDArray

from palette_gen.fastcolors import (  # type: ignore
    cct_to_D_xyY_jit,
    sRGB_to_XYZ_jit,
    xyY_to_XYZ_jit,
)
from palette_gen.punishedcam import XYZ_to_PUNISHEDCAM_JabQMsh_jit  # type: ignore

T = TypeVar("T")
COLOR_PAT = re.compile("#?[0-9a-fA-F]{6}")


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
    def XYZw(self) -> NDArray[np.float64]:
        return xyY_to_XYZ_jit(cct_to_D_xyY_jit(self.T))  # type: ignore

    # TODO should be customizable based on additional fields
    # noinspection PyMethodMayBeStatic
    def rgb_to_xyz(self, rgb: NDArray[np.float64]) -> NDArray[np.float64]:
        return sRGB_to_XYZ_jit(rgb.reshape((-1, 3)))  # type: ignore

    def xyz_to_cam(self, xyz: NDArray[np.float64]) -> NDArray[np.float64]:
        return XYZ_to_PUNISHEDCAM_JabQMsh_jit(  # type: ignore
            xyz,
            self.XYZw.reshape(1, -1),
            Lsw=self.Lsw,
            Lb=self.Lb,
            Lmax=self.Lmax,
        )

    def rgb_to_cam(self, rgb: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.xyz_to_cam(self.rgb_to_xyz(rgb)).squeeze()


@dataclass(order=True)
class RGBColor:

    rgb: tuple[float, float, float]
    name: str | None = field(init=False, default_factory=lambda: None)

    @classmethod
    def from_string(cls, s: str) -> RGBColor:
        return RGBColor(rgb=to_rgb(s))

    @cached_property
    def hex(self) -> str:
        return cast(str, to_hex(self.rgb))

    @cached_property
    def bare_hex(self) -> str:
        return self.hex.lstrip("#")

    def __str__(self) -> str:
        return f"Color[#{to_hex(self.rgb)}]"


@dataclass(order=True)
class JabColor(RGBColor):

    vs: ViewingSpec

    @cached_property
    def jab(self) -> NDArray[np.float64]:
        return self.vs.xyz_to_cam(self.vs.rgb_to_xyz(np.array(self.rgb)))[0, :3]

    def __str__(self) -> str:
        return f"JabColor[#{to_hex(self.rgb)}/{self.vs}]"
