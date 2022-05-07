from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from palette_gen.solvers import JabColor
from palette_gen.solvers.color import FixedJabTargetSolver, OrganizedColors


@dataclass()
class CylinderMesh(FixedJabTargetSolver):
    """
    Solves for a cylinder of staggered color rings.

    Solves such that the distance between nearest neighbors across rings
    is equal to the distance between nearest neighbors within the same
    ring.
    """

    name: str

    base_j: float
    base_m: float

    n_colors: int
    n_rings: int
    ring_names: list[str]

    first_ring_offset: float = 0.0

    @staticmethod
    def unit_hex(offset: float) -> NDArray[np.float64]:
        """
        Offset measured in units of angle between colors.

        e.g. offset = 0.5 -> 30 deg
        """

        out = np.empty((6, 2))
        xs: NDArray[np.float64] = (
            np.linspace(0, 2 * np.pi, num=6, endpoint=False) + offset * np.pi / 3
        )
        out[:, 0] = np.cos(xs)
        out[:, 1] = np.sin(xs)
        return out

    def jab_target(self, ab_offset: NDArray[np.float64]) -> NDArray[np.float64]:

        θ = 2 * np.pi / self.n_colors
        h = np.sqrt(2) * np.sqrt(np.cos(θ / 2) - np.cos(θ)) * self.base_m / 100

        out = np.empty((self.n_colors * self.n_rings, 3))

        angles = np.linspace(0, 2 * np.pi, num=self.n_colors, endpoint=False)
        step = np.pi / self.n_colors
        for ring in range(self.n_rings):
            sl = slice(ring * self.n_colors, (ring + 1) * self.n_colors)
            print(sl)

            out[sl, 0] = self.base_j + ring * h
            out[sl, 1] = self.base_m * np.cos(angles) / 100
            out[sl, 2] = self.base_m * np.sin(angles) / 100

            angles += step * (-1 if ring % 2 else 1)

        return out

    def organize_colors(self, raw_colors: Iterable[JabColor]) -> OrganizedColors:

        raw_colors = list(raw_colors)
        out = {}
        for name, ring in zip(self.ring_names, range(self.n_rings)):
            out[name] = raw_colors[ring * self.n_colors : (ring + 1) * self.n_colors]
        return out
