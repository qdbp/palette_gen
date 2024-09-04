from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from palette_gen.solvers import JabColor
from palette_gen.solvers.color import ColorSolver, FixedJabTargetSolver, OrganizedColors


@dataclass()
class TriHexSolver(FixedJabTargetSolver):
    """
    Solves solves for a set of colors from an arrangement loosely based on
    hexagonal-close-packing.

    Produces three lightness-levels of colors, with six each in the bottom
    two levels, ans two sets of six from the top, for a total of 24 colors.

    Parameters
    __________

    j_scale:
        scales the lightness-spacing of the lattice by this factor.
    """

    name: str
    base_j: float
    pitch: float

    first_ring_offset: float = 0.0
    alt_tones: bool = False

    j_scale: float = 1.0

    ring0_name: str | None = None
    ring1_name: str | None = None
    ring2_name: str | None = None
    tints_name: str | None = None

    @staticmethod
    def unit_hex(offset: float) -> NDArray[np.float64]:
        """
        Offset measured in units of angle between colors.

        e.g. offset = 0.5 -> 30 deg
        """

        out = np.empty((6, 2))
        xs: NDArray[np.float64] = np.linspace(0, 2 * np.pi, num=6, endpoint=False) + offset * np.pi / 3
        out[:, 0] = np.cos(xs)
        out[:, 1] = np.sin(xs)
        return out

    def jab_target(self, ab_offset: NDArray[np.float64]) -> NDArray[np.float64]:
        # array layout:
        # 0:6 first ring primaries
        # 6:12 second ring primaries
        # 12:18 third ring primaries
        # 18:24 third ring tints
        out = np.zeros((24, 3))
        # NB. this is about the limit of how many lattice points are reasonable
        # to fill ad hoc. If the system is expanded, just iterate lattice
        # vectors.

        # fill array with normalized coordinates
        out[0:6, 1:] = self.pitch * self.unit_hex(self.first_ring_offset)
        out[6:12, 1:] = (
            self.pitch
            * (2 / np.sqrt(3))  # distance from zero to the second gap
            * self.unit_hex(self.first_ring_offset + 0.5)
        )
        out[12:18, 1:] = self.pitch * (1 + 1 / np.sqrt(3)) * self.unit_hex(self.first_ring_offset)
        out[18:24, 1:] = self.pitch * self.unit_hex(self.first_ring_offset + 0.5)

        # h as defined in the notebook
        h = self.pitch * np.sqrt(2 / 3) * self.j_scale

        out[:6, 0] = self.base_j  # first ring
        out[6:12, 0] = self.base_j + h  # second ring
        out[12:24, 0] = self.base_j + 2 * h  # third ring

        # apply offset
        out[:, 1:] += ab_offset.reshape((-1, 2))

        return out

    def organize_colors(self, raw_colors: Iterable[JabColor]) -> OrganizedColors:
        raw_colors = list(raw_colors)
        return {
            self.ring0_name or self.name + "r0": ColorSolver.hue_sort(raw_colors[0:6]),
            self.ring1_name or self.name + "r1": ColorSolver.hue_sort(raw_colors[6:12]),
            self.tints_name or self.name + "tint": ColorSolver.hue_sort(raw_colors[18:24]),
            self.ring2_name or self.name + "r2": ColorSolver.hue_sort(raw_colors[12:18]),
        }
