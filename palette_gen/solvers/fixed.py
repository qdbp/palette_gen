from typing import Iterable, Type

from matplotlib.colors import to_rgb

from palette_gen.solvers import Color, ColorSolver, T, ViewingSpec


class FixedSolver(ColorSolver):
    """
    Trivial 'solver' that just returns a fixed dictionary of colors.
    """

    def __init__(self, fixed_dict: dict[str, str]):
        self.fixed_dict = fixed_dict
        self.fixed_dict.pop("name", None)

    def solve_for_context(
        self, bg_hex: str, vs: ViewingSpec
    ) -> Iterable[Color]:

        for key, hex in self.fixed_dict.items():
            yield Color(rgb=to_rgb(hex), vs=vs, name=key)

    @classmethod
    def construct_from_config(cls: Type[T], conf: dict[str, str]) -> T:
        return cls(fixed_dict=conf)  # type: ignore
