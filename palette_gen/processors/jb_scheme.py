"""
Declarative Jetbrains Color Scheme Generator.
"""
from __future__ import annotations

from abc import abstractmethod
from argparse import Namespace
from dataclasses import dataclass, field, fields
from datetime import datetime
from functools import partial
from typing import Any, Callable, Iterable
from typing import Optional as Opt
from typing import Type, Union
from xml.dom import minidom

# noinspection PyPep8Naming
from xml.etree.ElementTree import Element, SubElement, tostring

import yaml

from palette_gen.processors import ConcretePalette


def strip_hex(s: str) -> str:
    return s.lstrip("#")


def jb_hex(s: Union[str, int]) -> str:
    if isinstance(s, int):
        s = f"{s:06d}"
    s = strip_hex(s)
    if s == "0":
        out = "00000"
    elif len(s) == 3:
        out = s[0] + s[0] + s[1] + s[1] + s[2] + s[2]
    else:
        out = s
    return out


class XMLAccessor:
    def __init__(self, f: Callable[..., XMLBuilder]):
        self.f = f

    def __getattr__(self, item: str) -> Callable[..., XMLBuilder]:
        return partial(self.f, item)


class XMLAccessorDescriptor:
    def __init__(self, push: bool, empty: bool):
        self.push = push
        self.empty = empty

    def __get__(
        self, instance: XMLBuilder, owner: Type[XMLBuilder]
    ) -> XMLAccessor:
        if self.empty and self.push:
            return XMLAccessor(instance.push_empty)
        elif self.empty:
            return XMLAccessor(instance.empty)
        elif self.push:
            return XMLAccessor(instance.push)
        else:
            return XMLAccessor(instance.elem)


class XMLBuilder(XMLAccessor):
    e = XMLAccessorDescriptor(push=False, empty=True)
    ep = XMLAccessorDescriptor(push=True, empty=True)
    p = XMLAccessorDescriptor(push=False, empty=False)

    def __init__(self, root: Element):
        self.parent_stack = [root]
        self._staged: Opt[Element] = None
        XMLAccessor.__init__(self, self.elem)

    @classmethod
    def mk_root(cls, name: str, text: Opt[str], /, **attrs: str) -> XMLBuilder:
        root = Element(name, {**attrs})
        if text is not None:
            root.text = text
        return XMLBuilder(root)

    def top(self) -> Element:
        return self.parent_stack[-1]

    def root(self) -> Element:
        return self.parent_stack[0]

    def last(self) -> Element:
        try:
            return list(self.top())[-1]
        except IndexError:
            return self.top()

    def __call__(self, next_parent: Element) -> XMLBuilder:
        if self._staged is not None:
            raise RuntimeError(
                "Staged parent twice without entering context..."
            )
        self._staged = next_parent
        return self

    def __enter__(self) -> None:
        if self._staged is None:
            raise RuntimeError("Entering context without parent...")
        self._staged = None

    def __exit__(self, *args: Any) -> None:
        self.parent_stack.pop()

    def append(self, node: Element) -> None:
        self.top().append(node)

    def elem(
        self, name: str, text: Opt[str] = None, /, **attrs: str
    ) -> XMLBuilder:
        elem = SubElement(self.top(), name, {**attrs})
        if text is not None:
            elem.text = text
        assert self.last() == elem
        return self

    def empty(self, name: str, /, **attrs: str) -> XMLBuilder:
        return self.elem(name, None, **attrs)

    def push(self, name: str, text: Opt[str], **attrs: str) -> XMLBuilder:
        self.elem(name, text, **attrs)
        self.parent_stack.append(self.last())
        self._staged = self.last()
        return self

    def push_empty(self, name: str, /, **attrs: str) -> XMLBuilder:
        return self.push(name, None, **attrs)


class XMLSerializable:
    @abstractmethod
    def to_xml(self) -> Element:
        pass


# TODO move to schema
COLOR_KEYS = {"fg", "bg", "effect_color", "stripe"}


@dataclass()
class JBAttrSpec(XMLSerializable):
    name: str

    fg: Opt[str] = None  # rgb hex
    bg: Opt[str] = None  # rgb hex
    ft: Opt[str] = None
    effect: Opt[str] = None
    effect_color: Opt[str] = None  # rgb hex
    stripe: Opt[str] = None  # rgb hex
    base: Opt[str] = None

    def to_xml(self) -> Element:

        root_attrs = dict(name=self.name.upper())
        if self.base is not None:
            root_attrs["baseAttributes"] = self.base.upper()

        xmlb = XMLBuilder.mk_root("option", None, **root_attrs)

        with xmlb.ep.value():
            if self.fg is not None:
                xmlb.e.option(name="FOREGROUND", value=jb_hex(self.fg))
            if self.bg is not None:
                xmlb.e.option(name="BACKGROUND", value=jb_hex(self.bg))
            if self.ft is not None:
                xmlb.e.option(name="FONT_TYPE", value=self.ft)
            if self.stripe is not None:
                xmlb.e.option(
                    name="ERROR_STRIPE_COLOR", value=jb_hex(self.stripe)
                )
            if self.effect is not None:
                xmlb.e.option(name="EFFECT_TYPE", value=self.effect)
            if self.effect_color is not None:
                xmlb.e.option(
                    name="EFFECT_COLOR", value=jb_hex(self.effect_color)
                )

        return xmlb.root()


@dataclass(frozen=True)
class JBAtomicOption(XMLSerializable):
    name: str
    value: Union[str, int, float, bool]

    def to_xml(self) -> Element:
        value = str(self.value)
        if isinstance(self.value, bool):
            value = value.lower()
        return Element("option", name=self.name.upper(), value=value)


@dataclass()
class JBColorSpec(XMLSerializable):
    options: Iterable[JBAtomicOption]

    def to_xml(self) -> Element:
        root = Element("colors")
        for option in self.options:
            root.append(option.to_xml())
        return root


@dataclass()
class JBFontSpec:
    font_scale: float = 1.0
    line_spacing: float = 0.9
    editor_font_size: float = 14.0
    editor_font_name: str = "JetBrains Mono"
    editor_ligatures: bool = True
    console_font_size: float = 12.0
    console_font_name: str = "JetBrains Mono"
    console_ligatures: bool = True
    console_line_spacing: float = 0.8

    def to_options(self) -> Iterable[JBAtomicOption]:
        yield from (
            JBAtomicOption(f.name, getattr(self, f.name)) for f in fields(self)
        )


@dataclass()
class JBScheme(XMLSerializable):
    name: str

    color_spec: JBColorSpec
    attrs: Iterable[JBAttrSpec]
    font_spec: JBFontSpec

    version: int = 142
    ide: str = "idea"
    parent_scheme: str = "Default"
    modified: datetime = field(default_factory=datetime.now)
    created: datetime = field(init=False)
    ide_version: str = "2020.3.2.0.0"

    def __post_init__(self) -> None:
        self.created = self.modified

    def to_xml(self) -> Element:
        b = XMLBuilder.mk_root(
            "scheme",
            None,
            name=self.name,
            version=str(self.version),
            parent_scheme=self.parent_scheme,
        )
        with b.push_empty("metaInfo"):
            b.property(self.name, name="originalScheme")
            b.property(self.created.isoformat(), name="created")
            b.property(self.modified.isoformat(), name="modified")
            b.property(self.ide_version, name="ideVersion")
            b.property(self.ide, name="idea")

        b.append(self.color_spec.to_xml())
        with b.ep.attributes():
            for attr in self.attrs:
                b.append(attr.to_xml())
        for option in self.font_spec.to_options():
            b.append(option.to_xml())

        return b.root()

    @classmethod
    def process_config(cls, args: Namespace) -> None:

        with open(args.spec, "r") as f:
            scheme = yaml.full_load(f)

        font = JBFontSpec(**scheme["font"])

        with open(args.palette, "r") as f:
            palette = ConcretePalette.from_config(yaml.full_load(f))

        print(f"Generating scheme for {palette.name}, view {palette.view}")

        color_spec = JBColorSpec(
            JBAtomicOption(key, palette.subs(val))
            for key, val in scheme["colors"].items()
        )
        attrs = [
            JBAttrSpec(
                name=key,
                **{
                    k: palette.subs(v) if k in COLOR_KEYS else str(v)
                    for k, v in val.items()
                },
            )
            for key, val in scheme["attributes"].items()
        ]

        jb_scheme = JBScheme(
            color_spec=color_spec,
            attrs=attrs,
            font_spec=font,
            **scheme["meta"],
        )

        root = jb_scheme.to_xml()
        dom: str = minidom.parseString(tostring(root)).toprettyxml(indent="  ")
        dom = dom[dom.index("\n") + 1 :]

        if (out_fn := getattr(args, "out")) is None:
            out_fn = jb_scheme.name + f"{palette.view}.xml"

        print(f"Writing generated scheme {out_fn}")
        with open(out_fn, "w") as f:
            f.write(dom)
