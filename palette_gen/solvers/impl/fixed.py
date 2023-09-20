from dataclasses import dataclass
from collections.abc import Iterable

from matplotlib.colors import to_rgb

from palette_gen.solvers import JabColor, T, ViewingSpec
from palette_gen.solvers.color import ColorSolver


@dataclass()
class FixedRGBSolver(ColorSolver):
    """
    Trivial 'solver' that returns a dictionary of colors of fixed RGB values.
    """

    fixed_dict: dict[str, str]

    def __post_init__(self) -> None:
        self.fixed_dict.pop("name", None)

    def _solve_colors(self, bg_hex: str, vs: ViewingSpec) -> Iterable[JabColor]:
        for name, hex_code in self.fixed_dict.items():
            out = JabColor(rgb=to_rgb(hex_code), vs=vs)
            out.name = name
            yield out

    @classmethod
    def construct_from_config(cls: type[T], conf: dict[str, str]) -> T:
        return cls(fixed_dict=conf)  # type: ignore
