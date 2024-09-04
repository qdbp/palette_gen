import logging
from argparse import Namespace
from pathlib import Path

import numpy as np
import yaml

from palette_gen.solvers import ViewingSpec
from palette_gen.solvers.color import ColorSolver
from palette_gen.solvers.impl.cylinder_mesh import CylinderMesh
from palette_gen.solvers.impl.fixed import FixedRGBSolver
from palette_gen.solvers.impl.j_matched_greys import JMatchedGreys
from palette_gen.solvers.impl.jab_ring import JabRingSpec
from palette_gen.solvers.impl.tri_hex import TriHexSolver
from palette_gen.solvers.palette import PaletteSolver


def gen_palette_cmd(args: Namespace) -> None:
    """
    Entrypoint into the palette generator.

    Arguments defined in main.py
    """
    spec_path = Path(args.spec)
    if (out_fn := getattr(args, "out", None)) is None:
        out_path = spec_path.parent.joinpath(spec_path.stem + ".palette" + spec_path.suffix)
    else:
        out_path = Path(out_fn)

    gen_palette(spec_path, out_path, args.views or [], do_html=args.html, do_cone=args.cone)


def gen_palette(
    spec_path: Path,
    out_path: Path,
    explicit_views: list[str],
    *,
    do_html: bool,
    do_cone: bool,
) -> None:
    np.set_printoptions(precision=2, suppress=True)

    with spec_path.open() as f:
        full_spec = yaml.full_load(f)

    views = {name: ViewingSpec(name=name, **view_args) for name, view_args in full_spec["views"].items()}

    constructors = {
        "jab_ring": JabRingSpec,
        "fixed": FixedRGBSolver,
        "tri_hex": TriHexSolver,
        "cylinder": CylinderMesh,
        "greys": JMatchedGreys,
    }

    palette_spec: dict[str, ColorSolver] = {
        name: constructors[d.pop("type")].construct_from_config(  # type: ignore
            d["args"] | {"name": name}
        )
        for name, d in full_spec["palette"].items()
    }

    for view_name, vs in views.items():
        if explicit_views and view_name not in explicit_views:
            continue

        out = {
            "name": full_spec["name"],
            "view": view_name,
            "bg_hex": vs.bg_hex,
        }

        pal_out_fn = out_path.parent.joinpath(f"{out_path.stem}.{view_name}{out_path.suffix}")
        pal_out_fn.parent.mkdir(parents=True, exist_ok=True)

        print(f"Solving palette {view_name}...")
        palette = PaletteSolver(view_name + "_palette", vs=vs, palette_spec=palette_spec)
        out |= palette.serialize()

        print(f"Saving palette to {pal_out_fn.absolute()}.")
        pal_out_fn.write_text(yaml.dump(out, indent=2))

        if do_html:
            pal_out_fn.with_suffix(".html").write_text(palette.dump_html())

        if do_cone:
            fig = palette.draw_cone()
            cone_fn = pal_out_fn.with_suffix(".cone.html")
            logging.info(f"Saving cone plot to {cone_fn}")
            fig.write_html(cone_fn)
