from pathlib import Path
from tempfile import NamedTemporaryFile

from palette_gen.processors.yaml_palette import gen_palette


def test_palette_gen(data) -> None:
    with NamedTemporaryFile() as f:
        out_path = Path(f.name)
        gen_palette(data.palette_spec, out_path, [], do_html=True, do_cone=False)
