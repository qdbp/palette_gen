from pathlib import Path

from setuptools import find_packages, setup


def read_requirements() -> list[str]:
    req_file = Path(__file__).parent.joinpath("requirements.txt")
    with req_file.open() as f:
        return [line.strip() for line in f.readlines() if line.strip()]


setup(
    name="palette_gen",
    version="1.0.2",
    packages=find_packages(),
    url="http://github.com/qdbp/palette_gen",
    license="",
    author="qdbp",
    author_email="",
    description="color scheme generator",
    entry_points={"console_scripts": ["palette-gen=palette_gen:main"]},
    install_requires=read_requirements(),
)
