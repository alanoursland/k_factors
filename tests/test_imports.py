
import importlib
import pytest

@pytest.mark.parametrize("module", [
    "kfactors",
    "kfactors.algorithms",
    "kfactors.assignments",
    "kfactors.representations",
    "kfactors.updates",
    "kfactors.distances",
    "kfactors.initialization",
])
def test_submodules_exist(module):
    mod = importlib.import_module(module)
    assert mod is not None
