import sys
from argparse import ArgumentParser

from palette_gen.processors.jb_scheme import JBSchemeProcessor
from palette_gen.processors.jb_theme import JBThemeProcessor
from palette_gen.processors.nvim_scheme import NvimProcessor
from palette_gen.processors.yaml_palette import gen_palette_cmd


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
    sub_palette.add_argument("--out", help="output yaml file name.", type=str, required=True)
    sub_palette.add_argument("--spec", help="palette spec yaml config", type=str, required=True)
    sub_palette.add_argument(
        "views",
        help="Views to generate palettes for. If empty, generates all views.",
        nargs="*",
        type=str,
    )
    sub_palette.add_argument(
        "--html",
        help="generate an html table showing the generated colors",
        action="store_true",
    )
    sub_palette.add_argument(
        "--cone",
        help="generate a 3D colorspace cone visualization of generated colors",
        action="store_true",
    )
    sub_palette.set_defaults(cmd=gen_palette_cmd)

    # TODO dynamically generate these plugin style
    for proc_cls in [JBSchemeProcessor, JBThemeProcessor, NvimProcessor]:
        cls_parser = sub.add_parser(proc_cls.cmd_name, help=proc_cls.__doc__)
        proc_cls.configure_parser(cls_parser)
        cls_parser.set_defaults(cmd=proc_cls.process)

    args = parser.parse_args()
    args.cmd(args)


if __name__ == "__main__":
    sys.argv.extend(
        [
            "nvim",
            "-p",
            "/home/main/programming/projects/SalmonThemeBkp/build/palette.Twilight.yaml",
            "-o",
            "test.lua",
        ]
    )
    main()
