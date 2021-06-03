
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--plot", action="store_true", default=False, help="run slow plot tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "plot: mark plot test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--plot"):
        # --plot given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --plot option to run")
    for item in items:
        if "plot" in item.keywords:
            item.add_marker(skip_slow)