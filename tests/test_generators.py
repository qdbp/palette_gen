from pathlib import Path
from tempfile import NamedTemporaryFile

from palette_gen.processors.yaml_palette import gen_palette

TEST_YAML = Path(__file__).parent.joinpath("data", "test_palette_spec.yaml")


def test_palette_gen() -> None:
    with NamedTemporaryFile() as f:
        out_path = Path(f.name)
        gen_palette(TEST_YAML, out_path, [], do_html=True, do_cone=False)
