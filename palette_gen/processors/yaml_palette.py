from argparse import Namespace
from os.path import splitext

import numpy as np
import yaml

from palette_gen.solvers import ViewingSpec
from palette_gen.solvers.color import ColorSolver
from palette_gen.solvers.impl.fixed import FixedSolver
from palette_gen.solvers.impl.jab_ring import JabRingSpec
from palette_gen.solvers.palette import PaletteSolver


def gen_palette_cmd(args: Namespace) -> None:
    """
    Entrypoint into the palette generator.

    Arguments defined in main.py
    """

    np.set_printoptions(precision=2, suppress=True)

    with open(args.spec, "r") as f:
        full_spec = yaml.full_load(f)

    views = {
        name: ViewingSpec(name=name, **view_args)
        for name, view_args in full_spec["views"].items()
    }

    constructors = {"jab_ring": JabRingSpec, "fixed": FixedSolver}

    palette_spec: dict[str, ColorSolver] = {
        name: constructors[d.pop("type")].construct_from_config(  # type: ignore
            d["args"] | {"name": name}
        )
        for name, d in full_spec["palette"].items()
    }

    if (out_fn := getattr(args, "out", None)) is None:
        base, ext = splitext(args.spec)
        out_fn = base + ".palette" + ext

    out = {
        "name": full_spec["name"],
        "views": {},
    }
    for view_name, view in views.items():
        print(f"Solving palette {view_name}...")
        palette = PaletteSolver(
            view_name + "_palette", vs=view, palette_spec=palette_spec
        )
        out["views"][view_name] = palette.serialize()

        if not args.html:
            continue

        html_fn = splitext(out_fn)[0] + f".{view_name}.html"
        with open(html_fn, "w") as f:
            print(f"Saving html to {html_fn}")
            f.write(palette.dump_html())

    print(f"Saving palettes to {out_fn}.")
    with open(out_fn, "w") as f:
        yaml.dump(out, f)
