"""
Declarative JetBrains .theme.json generator using palette_gen colors.
"""
import json
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any

import yaml

from palette_gen.processors import ConcretePalette
from palette_gen.util import map_leaves


def process_theme(args: Namespace) -> None:
    palette = ConcretePalette.from_config(yaml.full_load(Path(args.palette).read_text()))
    logging.info(f"Generating theme {palette.name}, view {palette.view}")
    theme_config = yaml.full_load(Path(args.spec).read_text())

    meta = theme_config["meta"]
    name = meta["name"]
    author = meta["author"]
    # TODO should propagate these automatically
    dark = meta["dark"]
    editor_scheme = meta["scheme"] + f".{palette.view}.xml"

    icon_section: dict[str, Any] = map_leaves(  # type: ignore
        lambda x: palette.subs(x).hex, theme_config["icons"]  # type: ignore
    )

    if args.inline_colors:
        ui_dict = map_leaves(lambda x: palette.subs(x).hex, theme_config["ui"])
    else:
        ui_dict = theme_config["ui"]

    out = {
        "name": name,
        "author": author,
        "dark": dark,
        "editorScheme": "/" + editor_scheme,
        **({"colors": palette.hex_map} if not args.inline_colors else {}),
        "ui": ui_dict,
        "icons": icon_section,
    }

    if (out_fn := args.out) is None:
        out_fn = name + f".{palette.view}.theme.json"

    logging.info(f"Saving generated theme to {out_fn}")
    Path(out_fn).write_text(json.dumps(out, indent=2))
