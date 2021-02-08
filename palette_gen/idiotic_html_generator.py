"""
This is the sort of late-night absurdity that gives life the astringent meaning
we all crave.
"""
from __future__ import annotations

import builtins
import sys
from io import StringIO
from typing import Any, Callable, TypeVar

H = TypeVar("H", bound="HTML")

true_stdout = sys.stdout
true_print = print


class HTML:

    PARENT = None

    def __init__(
        self, tag: str = "html", _parent: HTML = None, **attrs: str
    ) -> None:

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

    def __enter__(self) -> HTML:

        sys.stdout = self.out
        # noinspection PyUnusedLocal
        def _print(*args: Any, end: str = "", **kwargs: Any) -> None:
            self.out.write("\n")
            self.out.write("\t" * (self.indent + 1))
            self.out.write(*args)
            self.out.write(end)

        builtins.print = _print

        attr_string = " ".join(f'{k}="{v}"' for k, v in self.atts.items())
        self.out.write(
            "\n" + "\t" * self.indent + f"<{self.tag}"
            f"{' ' if attr_string else ''}"
            f"{attr_string}>"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        self.out.write("\n" + "\t" * self.indent + f"</{self.tag}>")
        sys.stdout = true_stdout
        builtins.print = self._old_print

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