"""
Pytest configuration for REEE tests.
"""

import pytest

# Configure pytest-asyncio to auto mode
pytest_plugins = ('pytest_asyncio',)


def pytest_configure(config):
    """Configure pytest with asyncio marker."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as an async test."
    )
