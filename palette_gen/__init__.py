import sys
from argparse import ArgumentParser

from palette_gen.jb_scheme import JBScheme
from palette_gen.palette_solver import gen_palette_cmd


def main() -> None:
    parser = ArgumentParser()
    sub = parser.add_subparsers(
        description="subcommands",
        required=True,
        metavar="CMD",
        prog="palette-gen",
    )

    # palette generation
    sub_palette = sub.add_parser("palette", help="solve for palette colors")
    sub_palette.add_argument("spec", help="palette spec yaml config", type=str)
    sub_palette.add_argument(
        "--out", help="Save palette output to this file", type=str
    )
    sub_palette.add_argument(
        "--html",
        help="generate an html file showing the colors",
        action="store_true",
    )
    sub_palette.set_defaults(cmd=gen_palette_cmd)

    # scheme generation
    sub_scheme = sub.add_parser(
        "scheme", help="create a colorscheme from a template"
    )
    sub_scheme.add_argument("spec", help="scheme spec config, yaml", type=str)
    sub_scheme.add_argument(
        "palette", help="generated palette scheme, yaml", type=str
    )
    sub_scheme.add_argument(
        "--out", help="save scheme output to this file", type=str
    )
    sub_scheme.set_defaults(cmd=JBScheme.process_config)

    args = parser.parse_args()
    args.cmd(args)

    sys.exit(0)


if __name__ == "__main__":
    main()
