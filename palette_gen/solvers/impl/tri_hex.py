from dataclasses import dataclass, field
from typing import Iterable, Union

import numpy as np

from palette_gen.solvers import Color
from palette_gen.solvers.color import ColorSolver, FixedJabTargetSolver


@dataclass()
class TriHexSolver(FixedJabTargetSolver):
    """
    Hexagons are, after all, the bestagons.

    Solves a layout of three stacked, offset hexagonal rings of colors.

    One ring:

    Six colors with equal hue spread, 60 degrees apart. This is the unique color
    ring where neighbor colors are equally far from one another as their common
    distance from the central grey.

    In a well-behaved colorpsace, this distance (henceforth `d`) is exactly the
    perceptual distance, giving visual uniformity. It is equal to the normalized
    colorfulness m = √(a² + b²) of the colors in ring

    Two rings:

    This scheme can be extended by adding a second ring "above" the first
    (at higher j), whose 6 colors are rotated by 30 degrees, which gives
    maximally staggered hues.

    For this maximally staggered arrangement, there exists a specific
        δj = m * √(√3 - 1) ≈ 0.856 * d
    between the ring planes which makes the distance between "neighbor" colors
    in adjacent rings equal to d, so that they extend the visual uniformity
    scheme. Even in this maximally staggered arrangement (which minimized δj),
    the value of δj is close enough to 1 to consider the greys as "close enough"
    to being the uniform distance `d` apart.

    Third ring:

    A third ring can be added, staggered once more 30 degrees from the second,
    giving hue alignment with the first ring.
    The distance from their closest neighbors in the first ring (which will have
    the same hue) will be 2 * δj ≈ 1.71. This is close-ish to a "two-step" in
    the uniformity scheme, and the minimum distance between any two colors
    of identical hues in the scheme.

    In principle, more rings can be added. However, for large-enough values of
    d to create colorful primaries, the range of 2 * δj will exhaust the range
    of lightness reasonable for a single-purpose palette; beyond this membership
    in a uniformity scheme will be of limited usefulness.

    The overall effect is to give a set of "paired" primaries and tones with
    shared hues but a marked difference in hue. Between these is a set of
    hue-offset "auxiliary" colors with in-between brightness. This gives a
    palette of 18 distinguishable colors, all of which should be suitable for
    a shared visual role (in particular, as colors of text over a background
    of a given color).

    Skew
    ____

    The above specification can be modified by adding a "skew", such that
    higher rings have higher colorfulness with lower δj separation, with the
    constraint that nearest neighbors between successive ring maintain distance
    d.

    Please see the `hex_explore.ipynb` notebook.
    """

    name: str
    base_j: float
    neighbor_dist: float
    skew: float
    skew3: float = field(init=False)  # defined in notebook
    include_hues: bool = True  # defined in notebook as the b = 1 solution
    include_greys: bool = True
    first_ring_offset: float = 0.0

    def __post_init__(self):
        if not (1 / (np.sqrt(3) - 1) <= self.skew < np.sqrt(3)):
            raise ValueError("Skew outside of bounds [1.0, √3)")

        self.skew3 = np.sqrt(3) * self.skew - 1

    @staticmethod
    def unit_hex(offset: float) -> np.ndarray:
        """
        Offset measured in units of angle between colors.

        e.g. offset = 0.5 -> 30 deg
        """

        out = np.empty((6, 2))
        xs = (
            np.linspace(0, 2 * np.pi, num=6, endpoint=False)
            + offset * np.pi / 3
        )
        out[:, 0] = np.cos(xs)
        out[:, 1] = np.sin(xs)
        return out

    def jab_target(self, ab_offset: np.ndarray) -> np.ndarray:

        out = np.zeros((18, 3))

        # fill array with normalized coordinates
        out[:6, 1:] = out[12:, 1:] = self.unit_hex(self.first_ring_offset)
        out[6:12, 1:] = self.unit_hex(self.first_ring_offset + 0.5)

        # h as defined in the notebook
        h = self.neighbor_dist * np.sqrt(
            self.skew * np.sqrt(3) - self.skew ** 2
        )

        out[:6, 0] = self.base_j
        out[6:12, 0] = self.base_j + h
        out[12:, 0] = self.base_j + 2 * h

        if self.include_hues:
            tones = np.zeros((6, 3))
            tones[:, 1:] = self.unit_hex(self.first_ring_offset)
            tones[:, 0] = self.base_j + 2 * h
            out = np.concatenate([out, tones], axis=0)

        if self.include_greys:
            greys = np.zeros((3, 3))
            greys[:, 0] = self.base_j + h * np.array([0, 1, 2])
            out = np.concatenate([out, greys], axis=0)

        # skew is defined as the absolute distance of second ring colors from
        # the second ring grey. However, the skew process is additive, not
        # multiplicative, i.e. the distance of the third ring colors are
        # (1 + 2 * (skew - 1))
        out[0:6, 1:] *= self.neighbor_dist
        out[6:12, 1:] *= self.neighbor_dist * self.skew
        out[12:18, 1:] *= self.neighbor_dist * self.skew3

        # hues have the same spacing as the first ring
        if self.include_hues:
            out[18:24, 1:] *= self.neighbor_dist

        # apply offset
        out[:, 1:] += ab_offset.reshape((-1, 2))

        return out

    def organize_colors(
        self, raw_colors: Iterable[Color]
    ) -> Union[list[Color], dict[str, list[Color]]]:

        raw_colors = list(raw_colors)
        out = {}
        # hues appear first
        if self.include_hues:
            out["h"] = ColorSolver.hue_sort(raw_colors[18:24])

        out |= {
            "2": ColorSolver.hue_sort(raw_colors[12:18]),
            "1": ColorSolver.hue_sort(raw_colors[6:12]),
            "0": ColorSolver.hue_sort(raw_colors[0:6]),
        }

        # greys are always last
        if self.include_greys:
            out["2"].insert(0, raw_colors[-1])
            out["1"].insert(0, raw_colors[-2])
            out["0"].insert(0, raw_colors[-3])
        return out
