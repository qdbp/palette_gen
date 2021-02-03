"""
This is the sort of late-night absurdity that gives life the astringent
meaning we all crave.
"""
from __future__ import annotations

import builtins
import sys
from io import StringIO
from typing import Any, Callable, TextIO, Type, TypeVar

H = TypeVar("H", bound="HTML")


class HTML:

    PARENT = None

    def __init__(self, _parent: HTML = None, **kwargs: str) -> None:

        self.kwargs = kwargs
        self._parent = _parent

        if _parent is None:
            self.out: StringIO = StringIO("")
            self.indent: int = 0
        else:
            self.out = _parent.out
            self.indent = _parent.indent + 1

        self.old_stdout: TextIO
        self.old_print = print

    def __getattr__(self, tag: str) -> Callable[[], H]:
        def _mk_node(**kwargs: str) -> H:
            return type(tag, (HTML,), {})(_parent=self, **kwargs)  # type: ignore

        return _mk_node

    def __enter__(self) -> HTML:
        self.old_stdout = sys.stdout
        sys.stdout = self.out

        # noinspection PyUnusedLocal
        def _print(*args: Any, end: str = "", **kwargs: Any) -> None:
            self.out.write("\n")
            self.out.write("\t" * (self.indent + 1))
            self.out.write(*args)
            self.out.write(end)

        builtins.print = _print

        attr_string = " ".join(f'{k}="{v}"' for k, v in self.kwargs.items())
        self.out.write(
            "\n" + "\t" * self.indent + f"<{self.__class__.__name__}"
            f"{' ' if attr_string else ''}"
            f"{attr_string}>"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        self.out.write(
            "\n" + "\t" * self.indent + f"</{self.__class__.__name__}>"
        )
        sys.stdout = self.old_stdout
        builtins.print = self.old_print

    def __str__(self) -> str:
        return self.out.getvalue()


if __name__ == "__main__":
    with (html := HTML(meta="foo")) as h:
        with h.p(style="font-size:20;") as p:
            print("HAHAHAH")
            with p.a(href="http://foobar.io"):
                print("CLICK HERE!!!")
            with p.b():
                print("FOR REAL")

    print(str(html))
