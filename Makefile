.PHONY: example install palette theme scheme mkdir

install:
	pip install --user .

mkdir:
	@mkdir -p out

example: palette scheme theme

palette: mkdir
	palette-gen --out out/gen_palette.yaml  palette examples/spec.yaml --html

scheme: mkdir
	palette-gen --out out/gen_scheme.xml scheme examples/scheme.yaml out/gen_palette.yaml

theme: mkdir
	palette-gen --out out/gen.theme.json theme examples/theme.yaml out/gen_palette.yaml Night --inline-colors
