from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from logging import warning
from typing import Any, Mapping, Union

# TODO use color class instead of str for hex colors


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

    @staticmethod
    @lru_cache(maxsize=1 << 16)
    def _depad_zero(s: str) -> str:
        """
        Takes a string like xyz0005 and returns xyz005.

        Will be a nop if the string has no zeros when it does not end
        with a zero, or a single zero when it does.
        """

        if s.count("0") - int(s.endswith("0")) > 0:
            return re.sub("0", "", s, 1)
        return s

    def subs(self, color: Union[int, str]) -> str:
        """
        Translates a color "in the wild" into a hex color based on this
        palette.

        Should be very liberal, bending over backwards to find some some
        valid interpretation.
        """
        color = str(color)
        while color not in self.hex_map:
            new_color = self._depad_zero(color)
            if new_color == color:
                break
            color = new_color

        # TODO need to check if the returned color is valid
        return self.hex_map.get(color, color)
