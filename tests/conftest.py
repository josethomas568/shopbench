import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shopbench.harness.server import StoreServer  # noqa: E402
from shopbench.store.catalog import Catalog  # noqa: E402


@pytest.fixture(scope="session")
def server():
    with StoreServer() as s:
        yield s


@pytest.fixture(scope="session")
def catalog():
    return Catalog()


@pytest.fixture(scope="session")
def tasks():
    return {t["id"]: t for t in json.loads((ROOT / "shopbench" / "tasks" / "tasks.json").read_text())}
