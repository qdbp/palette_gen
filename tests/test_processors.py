from argparse import Namespace
from pathlib import Path

import pytest

from palette_gen import JBSchemeProcessor, JBThemeProcessor, NvimProcessor
from palette_gen.processors.yaml_palette import gen_palette


@pytest.fixture(scope="module")
def palette_fn(data, tmp_path_factory) -> Path:
    out_fn = tmp_path_factory.mktemp("palette") / "palette.yaml"
    gen_palette(data.palette_spec, out_fn, ["Night"], do_html=False, do_cone=False)
    return out_fn.parent.glob("*.yaml").__next__()


def test_theme_processor(data, palette_fn, tmp_path):
    out_fn = tmp_path / "theme.yaml"
    args = Namespace(palette=palette_fn, out=out_fn, spec=data.theme, inline_colors=True)
    JBThemeProcessor.process(args)
    assert out_fn.exists()
    assert out_fn.stat().st_size > 0


def test_scheme_processor(data, palette_fn, tmp_path):
    out_fn = tmp_path / "scheme.xml"
    args = Namespace(palette=palette_fn, out=out_fn, spec=data.scheme)
    JBSchemeProcessor.process(args)
    assert out_fn.exists()
    assert out_fn.stat().st_size > 0


def test_nvim_processor(palette_fn, tmp_path):
    out_fn = tmp_path / "nvim.lua"
    args = Namespace(palette=palette_fn, out=out_fn)
    NvimProcessor.process(args)
    assert out_fn.exists()
    assert out_fn.stat().st_size > 0
