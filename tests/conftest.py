import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def A():
    from assumptions import load_assumptions
    return load_assumptions()


@pytest.fixture(scope="session")
def toy():
    from tests.toy import toy_city
    return toy_city()
