import os

import pytest


@pytest.fixture(autouse=True)
def set_testing_env():
    """Automatically set TESTING environment variable for all tests."""
    old_env = os.environ.get("TESTING")
    os.environ["TESTING"] = "True"
    yield
    if old_env is None:
        del os.environ["TESTING"]
    else:
        os.environ["TESTING"] = old_env
