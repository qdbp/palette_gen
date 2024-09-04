"""
Generates a lua color table for use in Neovim.
"""

from argparse import Namespace
from textwrap import dedent

from palette_gen.processors.core.concrete_palette import ConcretePalette
from palette_gen.processors.core.palette_processor import PaletteProcessor


class NvimProcessor(PaletteProcessor):
    """
    generate a lua color table for Neovim
    """

    cmd_name = "nvim"

    def _generate_body(self, concrete_palette: ConcretePalette, args: Namespace) -> str:
        return dedent(
            f"""
local palette = {{
    name = "{concrete_palette.name}",
    view = "{concrete_palette.view}",
    colors = {{
        {",\n        ".join(f'["{k}"] = "{v}"' for k, v in concrete_palette.hex_map.items())}
    }},
}}
""".strip()
        )


if __name__ == "__main__":
    NvimProcessor.process(
        Namespace(
            palette="/home/main/programming/projects/SalmonThemeBkp/build/palette.Twilight.yaml",
            output="test.lua",
        )
    )
