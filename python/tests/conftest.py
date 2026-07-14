import pytest
import gades


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")


def pytest_collection_modifyitems(config, items):
    if not gades.has_gpu():
        skip_gpu = pytest.mark.skip(reason="No CUDA GPU available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
