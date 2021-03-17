from setuptools import find_packages, setup  # type: ignore

setup(
    name="palette_gen",
    version="0.7",
    packages=find_packages(),
    url="http://github.com/qdbp/palette_gen",
    license="",
    author="qdbp",
    author_email="",
    description="color scheme generator",
    entry_points={"console_scripts": ["palette-gen=palette_gen:main"]},
)
