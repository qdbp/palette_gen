"""
This is the sort of late-night absurdity that gives life the astringent meaning
we all crave.
"""

import builtins
import sys
from collections.abc import Callable
from io import StringIO
from typing import Any, Self, TypeVar, final

H = TypeVar("H", bound="HTML")

true_stdout = sys.stdout
true_print = print


@final
class HTML:
    PARENT = None

    def __init__(self, tag: str = "html", _parent: Self | None = None, **attrs: str) -> None:
        self.atts = attrs
        self.tag = tag
        self._parent = _parent

        self.out: StringIO = _parent.out if _parent else StringIO("")
        self.indent: int = _parent.indent + 1 if _parent else 0
        self._old_print = print

    def __getattr__(self, tag: str) -> Callable[[], H]:
        def _mk_node(**kwargs: str) -> H:
            return HTML(tag, _parent=self, **kwargs)  # type: ignore

        return _mk_node

    def __enter__(self) -> Self:
        sys.stdout = self.out

        # noinspection PyUnusedLocal
        def _print(*args: Any, end: str = "", **kwargs: Any) -> None:  # noqa: ARG001
            self.out.write("\n")
            self.out.write("\t" * (self.indent + 1))
            self.out.write(*args)
            self.out.write(end)

        # noinspection PyTypeHints
        builtins.print = _print  # type: ignore

        attr_string = " ".join(f'{k}="{v}"' for k, v in self.atts.items())
        self.out.write(
            "\n" + "\t" * self.indent + f"<{self.tag}{' ' if attr_string else ''}{attr_string}>"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        self.out.write("\n" + "\t" * self.indent + f"</{self.tag}>")
        sys.stdout = true_stdout
        builtins.print = self._old_print

    def __str__(self) -> str:
        return self.out.getvalue()
