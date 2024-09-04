"""
Declarative JetBrains .theme.json generator using palette_gen colors.
"""

import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import yaml

from palette_gen.processors.core.concrete_palette import ConcretePalette
from palette_gen.processors.core.palette_processor import PaletteProcessor
from palette_gen.util import map_leaves


class JBThemeProcessor(PaletteProcessor):
    """
    generate a JetBrains .theme.json file
    """

    cmd_name = "jb_theme"

    @classmethod
    def _add_extra_parser_opts(cls, parser: ArgumentParser) -> None:
        parser.add_argument("-s", "--spec", help="theme spec config file, yaml", type=str, required=True)
        parser.add_argument(
            "--inline-colors",
            help='inline all colors and omit the "colors" dict',
            action="store_true",
        )

    def _generate_body(self, concrete_palette: ConcretePalette, args: Namespace) -> str:
        pal = concrete_palette
        logging.info(f"Generating theme {pal.name}, view {pal.view}")
        theme_config = yaml.full_load(Path(args.spec).read_text())

        meta = theme_config["meta"]
        name = meta["name"]
        author = meta["author"]
        # TODO should propagate these automatically
        dark = meta["dark"]
        editor_scheme = meta["scheme"] + f".{pal.view}.xml"

        icon_section: dict[str, Any] = map_leaves(  # type: ignore
            lambda x: pal.subs(x).hex,
            theme_config["icons"],  # type: ignore
        )

        if args.inline_colors:
            ui_dict = map_leaves(lambda x: pal.subs(x).hex, theme_config["ui"])
        else:
            ui_dict = theme_config["ui"]

        out = {
            "name": name,
            "author": author,
            "dark": dark,
            "editorScheme": "/" + editor_scheme,
            **({"colors": pal.hex_map} if not args.inline_colors else {}),
            "ui": ui_dict,
            "icons": icon_section,
        }
        return json.dumps(out, indent=2)
