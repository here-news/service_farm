"""
Pytest fixtures for kernel validation with Neo4j.

Usage:
    @pytest.mark.integration
    async def test_full_weave(fresh_db):
        await fresh_db.load_fixture("corpus.json")
        # run tests...
"""

import os
import pytest
import pytest_asyncio

from .test_neo4j import TestNeo4jManager, TestNeo4jConfig


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring Neo4j"
    )


@pytest.fixture(scope="session")
def neo4j_config():
    """Test Neo4j configuration from environment or defaults."""
    return TestNeo4jConfig(
        uri=os.getenv("TEST_NEO4J_URI", "bolt://localhost:7688"),
        user=os.getenv("TEST_NEO4J_USER", "neo4j"),
        password=os.getenv("TEST_NEO4J_PASSWORD", "test_password"),
    )


@pytest_asyncio.fixture(scope="session")
async def test_neo4j(neo4j_config):
    """
    Session-scoped test Neo4j connection.

    Starts once per test session, shared across all tests.
    """
    manager = TestNeo4jManager(neo4j_config)
    try:
        await manager.connect()
        yield manager
    finally:
        await manager.close()


@pytest_asyncio.fixture
async def fresh_db(test_neo4j):
    """
    Per-test fresh database.

    Clears all data and creates indexes before each test.
    """
    await test_neo4j.setup_fresh()
    yield test_neo4j


@pytest_asyncio.fixture
async def corpus_db(test_neo4j):
    """
    Database loaded with golden corpus.

    For tests that need the full ~1000 claim corpus.
    """
    import os
    corpus_path = os.path.join(
        os.path.dirname(__file__),
        "..", "golden_macro", "corpus.json"
    )

    await test_neo4j.setup_fresh()

    if os.path.exists(corpus_path):
        await test_neo4j.load_fixture(corpus_path)

    yield test_neo4j


@pytest_asyncio.fixture
async def wfc_snapshot_db(test_neo4j):
    """
    Database loaded with WFC fire replay snapshot.

    For replay tests against frozen real-world data.
    """
    import os
    snapshot_path = os.path.join(
        os.path.dirname(__file__),
        "..", "fixtures", "replay_wfc_snapshot.json"
    )

    await test_neo4j.setup_fresh()

    if os.path.exists(snapshot_path):
        await test_neo4j.load_fixture(snapshot_path)

    yield test_neo4j
