from abc import ABC, abstractmethod
from typing import Any, Iterable, Type

from palette_gen.solvers import Color, T, ViewingSpec


class ColorSolver(ABC):
    """
    Abstract base class for objects capable of solving for a list of colors
    from a background color ViewingSpec, based on some desiderata.
    """

    @abstractmethod
    def solve_for_context(
        self, bg_hex: str, vs: ViewingSpec
    ) -> Iterable[Color]:
        """
        Solves for a set of colors based on a viewing spec.
        """

    @classmethod
    @abstractmethod
    def construct_from_config(cls: Type[T], conf: dict[str, Any]) -> T:
        """
        Constructs the spec from the yaml configuration.
        """
