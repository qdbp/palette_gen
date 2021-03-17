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

    for view_name, vs in views.items():

        if args.views and view_name not in args.views:
            continue

        out = {
            "name": full_spec["name"],
            "view": view_name,
            "bg_hex": vs.bg_hex,
        }

        path, ext = splitext(out_fn)
        view_fn = f"{path}.{view_name}{ext}"

        print(f"Solving palette {view_name}...")
        palette = PaletteSolver(
            view_name + "_palette", vs=vs, palette_spec=palette_spec
        )
        out |= palette.serialize()

        print(f"Saving palette to {out_fn}.")
        with open(view_fn, "w") as f:
            yaml.dump(out, f)

        if args.html:
            html_fn = splitext(view_fn)[0] + ".html"
            with open(html_fn, "w") as f:
                print(f"Saving html to {html_fn}")
                f.write(palette.dump_html())
