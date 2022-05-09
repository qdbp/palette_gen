from __future__ import annotations

import re
from dataclasses import dataclass
from logging import warning
from typing import Any, Mapping

# TODO use color class instead of str for hex colors
from palette_gen.solvers import RGBColor


@dataclass()
class ConcretePalette:
    name: str
    view: str
    bg_hex: str
    hex_map: Mapping[str, str]

    @classmethod
    def from_config(cls, palette_dict: dict[str, Any]) -> ConcretePalette:
        hex_map = {}
        for palette_name, palette in palette_dict.pop("palette").items():
            for item in palette:
                name = item["name"]
                if name in hex_map:
                    warning(f"Duplicate color key {item}")
                hex_map[name] = item["hex"]
        return ConcretePalette(**palette_dict, hex_map=hex_map)

    def subs(self, color: int | str) -> RGBColor:
        """
        Translates a color "in the wild" into a hex color based on this
        palette.

        Should be very liberal, bending over backwards to find some some
        valid interpretation.
        """
        color = str(color).lower()

        try:
            name, ix = re.findall("([a-z]+)_?([0-9]+)", color)[0]
            ix = int(ix)
            keys = list(
                map(
                    lambda s: s.format(name, ix),
                    [
                        "{}{}",
                        "{}{:01d}",
                        "{}{:02d}",
                        "{}{:03d}",
                        "{}{:04d}",
                        "{}{:05d}",
                    ],
                )
            )
        except (IndexError, ValueError):
            keys = [color]

        for key in keys:
            if key in self.hex_map:
                mapped = self.hex_map[key]
                break
        else:
            mapped = self.hex_map.get(color, color)

        try:
            return RGBColor.from_string(mapped)
        except ValueError as e:
            raise ValueError(f"{mapped} is not a valid rgb color in this palette.") from e
