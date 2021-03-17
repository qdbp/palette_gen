"""
Declarative JetBrains .theme.json generator using palette_gen colors.
"""
import json
from argparse import Namespace
from typing import Any

import yaml

from palette_gen.processors import ConcretePalette
from palette_gen.util import map_leaves


def process_theme(args: Namespace) -> None:

    with open(args.palette, "r") as f:
        palette = ConcretePalette.from_config(yaml.full_load(f))

    print(f"Generating theme {palette.name}, view {palette.view}")

    with open(args.spec, "r") as f:
        theme_config = yaml.full_load(f)

    meta = theme_config["meta"]
    name = meta["name"]
    author = meta["author"]
    # TODO should propagate these automatically
    dark = meta["dark"]
    editor_scheme = meta["scheme"] + f".{palette.view}.xml"

    icon_section: dict[str, Any] = map_leaves(  # type: ignore
        lambda x: palette.subs(x), theme_config["icons"]  # type: ignore
    )

    if args.inline_colors:
        ui_dict = map_leaves(lambda x: palette.subs(x), theme_config["ui"])
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

    print(f"Saving generated theme to {out_fn}")
    with open(out_fn, "w") as f:
        json.dump(out, f, indent=2)
