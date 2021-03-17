import sys
from argparse import ArgumentParser

from palette_gen.processors.jb_scheme import JBScheme
from palette_gen.processors.jb_theme import process_theme
from palette_gen.processors.yaml_palette import gen_palette_cmd


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--out", help="output file name. format depends on command.", type=str
    )

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
        "--html",
        help="generate an html file showing the colors",
        action="store_true",
    )
    sub_palette.add_argument(
        "views",
        help="Views to generate palettes for. Will generate all if omitted.",
        nargs="*",
        type=str,
    )
    sub_palette.set_defaults(cmd=gen_palette_cmd)

    # scheme generation
    sub_scheme = sub.add_parser(
        "scheme", help="create a colorscheme from a template"
    )
    sub_scheme.add_argument(
        "spec", help="scheme spec config file, yaml", type=str
    )
    sub_scheme.add_argument(
        "palette", help="generated palette scheme file, yaml", type=str
    )
    sub_scheme.set_defaults(cmd=JBScheme.process_config)

    sub_theme = sub.add_parser(
        "theme", help="create a .theme.json from a template"
    )
    sub_theme.add_argument(
        "spec", help="theme spec config file, yaml", type=str
    )
    sub_theme.add_argument(
        "palette", help="generated palette scheme file, yaml", type=str
    )
    sub_theme.add_argument(
        "--inline-colors",
        help='inline all colors and omit the "colors" dict',
        action="store_true",
    )
    sub_theme.set_defaults(cmd=process_theme)

    args = parser.parse_args()
    args.cmd(args)

    sys.exit(0)


if __name__ == "__main__":
    main()
