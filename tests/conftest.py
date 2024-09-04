from pathlib import Path

import pytest

TEST_DATA_ROOT = Path(__file__).parent.joinpath("data")


class TestData:
    palette_spec: Path = TEST_DATA_ROOT.joinpath("colorspec.yaml")
    theme: Path = TEST_DATA_ROOT.joinpath("theme.yaml")
    scheme: Path = TEST_DATA_ROOT.joinpath("scheme.yaml")


@pytest.fixture(scope="session")
def data():
    return TestData
