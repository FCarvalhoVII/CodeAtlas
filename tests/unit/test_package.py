from importlib.metadata import version

import codeatlas


def test_package_exposes_version() -> None:
    assert codeatlas.__version__ == "1.0.0"
    assert version("codeatlas") == codeatlas.__version__
