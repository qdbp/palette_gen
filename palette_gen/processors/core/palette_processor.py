import logging
from abc import abstractmethod
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self, final

import yaml

from palette_gen.processors.core.concrete_palette import ConcretePalette


@dataclass
class PaletteProcessor:
    cmd_name: ClassVar[str]
    source_palette: Path
    output_path: Path

    @abstractmethod
    def _generate_body(self, concrete_palette: ConcretePalette, args: Namespace) -> str:
        """
        Generate the body of the output file.

        Args:
            concrete_palette: the fully resolved palette to use

        Returns:
            The body of the output file according to the format of this class.
        """

    @classmethod
    @final
    def configure_parser(cls, parser: ArgumentParser) -> None:
        """
        Add arguments to the parser for all processors.

        Args:
            parser: the parser to add arguments to
        """
        parser.add_argument("-p", "--palette", help="palette yaml file", type=Path, required=True)
        parser.add_argument("-o", "--out", help="output file name", type=Path, required=True)
        cls._add_extra_parser_opts(parser)

    @classmethod
    def _add_extra_parser_opts(cls, parser: ArgumentParser) -> None:
        """
        Add arguments to the parser for this processor.

        Args:
            parser: the parser to add arguments to
        """

    @classmethod
    def construct_from_args(cls, args: Namespace) -> Self:
        """
        Construct a processor from parsed arguments.

        Args:
            args: the parsed arguments

        Returns:
            A new instance of the processor
        """
        return cls(source_palette=args.palette, output_path=args.out)

    @classmethod
    @final
    def process(cls, args: Namespace) -> None:
        self = cls.construct_from_args(args)

        palette = ConcretePalette.from_config(yaml.full_load(self.source_palette.read_text()))
        logging.info(f"Generating palette {palette.name}, view {palette.view}")

        out = self._generate_body(palette, args)

        logging.info(f"Saving generated theme to {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(out)
