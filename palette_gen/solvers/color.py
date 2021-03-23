from abc import ABC, abstractmethod
from math import atan2
from typing import Any, Iterable, Type, Union, final

import numpy as np
from matplotlib.colors import to_rgb
from numba import njit
from scipy.optimize import minimize
from scipy.special import expit

from palette_gen.fastcolors import sRGB_to_XYZ_jit
from palette_gen.punishedcam import XYZ_to_PUNISHEDCAM_JabQMsh_jit
from palette_gen.solvers import Color, T, ViewingSpec


class ColorSolver(ABC):
    """
    Abstract base class for objects capable of solving for a list of colors
    from a background color ViewingSpec, based on some desiderata.
    """

    @final
    def solve_for_context(
        self, bg_hex: str, vs: ViewingSpec
    ) -> Union[list[Color], dict[str, list[Color]]]:
        """
        Solves for a set of colors based on a viewing spec.
        """
        return self.organize_colors(self._solve_colors(bg_hex, vs))

    @abstractmethod
    def _solve_colors(self, bg_hex: str, vs: ViewingSpec) -> Iterable[Color]:
        """
        Implementation of solve_for_context.

        Solves for a set of colors based on a viewing spec.
        """

    @classmethod
    def construct_from_config(cls: Type[T], conf: dict[str, Any]) -> T:
        """
        Constructs the spec from the yaml configuration.
        """
        return cls(**conf)  # type: ignore

    def organize_colors(
        self, raw_colors: Iterable[Color]
    ) -> Union[list[Color], dict[str, list[Color]]]:
        """
        Groups, sorts and labels the colors returned by _solve.

        Parameters
        ----------
        raw_colors
            iterable of colors returned by _solve, in implementation-defined
            order

        Returns
        -------
            single organized list of colors, or dictionary of such lists.
            the order of elements here will define the default display order in
            the palette.
        """
        return self.hue_sort(raw_colors)

    @staticmethod
    def hue_sort(colors: Iterable[Color]) -> list[Color]:
        return sorted(colors, key=lambda c: -atan2(c.jab[1], c.jab[2]))


class FixedJabTargetSolver(ColorSolver, ABC):
    """
    Base class for color solvers with a Jab target vector known from config.
    """

    @abstractmethod
    def jab_target(self, ab_offset: np.ndarray) -> np.ndarray:
        """
        Gets the jab targets corresponding to the ring positions.

        Parameters
        ----------
        ab_offset: shifts the center of the ring to this point in the (a, b)
            plane. Intended to be used to center with respect to the background
            (a, b) position for some viewing conditions.

        Returns
        -------
        an array of shape (self.n_colors, 3) of the jab targets
        """

    @final
    def _solve_colors(self, bg_hex: str, vs: ViewingSpec) -> Iterable[Color]:

        ab_offset = vs.rgb_to_cam(np.array(to_rgb(bg_hex))[None, :])[1:3]
        jab_target = self.jab_target(ab_offset)

        if jab_target.ndim != 2 or jab_target.shape[1] != 3:
            raise RuntimeError(
                f"jab_target has unsuitable shape {jab_target.shape}; "
                f"should be (n_colors, 3)"
            )

        # initial logit values
        logit_rgb = np.random.normal(size=jab_target.shape).ravel()

        name = getattr(self, "name", f"anonymous {self.__class__.__name__}")
        print(f"... solving color set {name}...")

        res = minimize(
            self._loss,
            logit_rgb,
            args=(
                jab_target,
                vs.XYZw,
                vs.Lsw,
                vs.Lb,
                vs.Lmax,
            ),
        )

        return map(
            lambda x: Color(tuple(x), vs=vs), expit(res["x"]).reshape((-1, 3))
        )

    # noinspection PyPep8Naming
    @staticmethod
    @final
    @njit  # type: ignore
    def _loss(
        logit_rgb: np.ndarray,
        jab_target: np.ndarray,
        xyz_r: np.ndarray,
        Lsw: float,
        Lb: float,
        Lmax: float,
    ) -> float:

        rgb = (1 / (1 + np.exp(-logit_rgb))).reshape((-1, 3))

        jabqmsh = XYZ_to_PUNISHEDCAM_JabQMsh_jit(
            sRGB_to_XYZ_jit(rgb), xyz_r, Lsw=Lsw, Lb=Lb, Lmax=Lmax
        )
        jab = jabqmsh[..., :3]
        loss = ((jab - jab_target) ** 2).sum()
        return loss  # type: ignore
