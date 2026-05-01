"""Pytest configuration and custom CLI options."""

import os

import pytest

os.environ.pop("LD_LIBRARY_PATH", None)

DEFAULT_LOGN = 10
DEFAULT_BATCH = 4


def pytest_addoption(parser):
    """Register local CLI flags."""
    parser.addoption(
        "--logn",
        type=int,
        default=DEFAULT_LOGN,
        help=f"log2(N) for tests (default: {DEFAULT_LOGN})",
    )
    parser.addoption(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help=f"Batch size for tests (default: {DEFAULT_BATCH})",
    )


@pytest.fixture(scope="session")
def logn(request):
    """Return log2(N) for tests."""
    return request.config.getoption("--logn")


@pytest.fixture(scope="session")
def batch(request):
    """Return batch size for tests."""
    return request.config.getoption("--batch")
