"""
Test Neo4j Lifecycle
====================

Milestone 1 Stop/Go Gate Test

Tests:
- test_connect: Connect to test Neo4j
- test_load_fixture: Load 10 claims in deterministic order
- test_snapshot: Export ordered dict of state
- test_clear: Clear all nodes, verify zero

Full cycle must complete in < 30 seconds.
"""

import pytest
import asyncio
import os
import time
from pathlib import Path

# Skip if neo4j not available
neo4j_available = True
try:
    from neo4j import AsyncGraphDatabase
except ImportError:
    neo4j_available = False

# Import our test manager
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from reee.tests.db.test_neo4j import TestNeo4jManager, TestNeo4jConfig


# Fixture path
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
MICRO_FIXTURE = FIXTURES_DIR / "micro_10_claims.json"


@pytest.fixture
def neo4j_config():
    """Test Neo4j configuration."""
    return TestNeo4jConfig(
        uri=os.environ.get("TEST_NEO4J_URI", "bolt://localhost:7688"),
        user=os.environ.get("TEST_NEO4J_USER", "neo4j"),
        password=os.environ.get("TEST_NEO4J_PASSWORD", "test_password"),
    )


@pytest.fixture
async def neo4j_manager(neo4j_config):
    """Async fixture for Neo4j manager."""
    manager = TestNeo4jManager(neo4j_config)
    await manager.connect()
    yield manager
    await manager.close()


@pytest.mark.skipif(not neo4j_available, reason="neo4j package not installed")
@pytest.mark.integration
class TestNeo4jLifecycle:
    """
    Milestone 1 acceptance tests for Neo4j substrate.

    Stop/Go Gate:
        pytest backend/reee/tests/integration/test_neo4j_lifecycle.py -v
        Must pass: test_connect, test_load_fixture, test_snapshot, test_clear
    """

    @pytest.mark.asyncio
    async def test_connect(self, neo4j_config):
        """Test Neo4j connection succeeds."""
        start = time.time()

        manager = TestNeo4jManager(neo4j_config)
        await manager.connect()

        assert manager._connected is True
        assert manager.driver is not None

        # Verify we can run a query
        async with manager.driver.session() as session:
            result = await session.run("RETURN 1 as n")
            record = await result.single()
            assert record["n"] == 1

        await manager.close()
        assert manager._connected is False

        elapsed = time.time() - start
        print(f"\n  Connect/disconnect cycle: {elapsed:.2f}s")
        assert elapsed < 5, "Connection cycle should be < 5s"

    @pytest.mark.asyncio
    async def test_clear_all(self, neo4j_manager):
        """Test clearing all nodes and relationships."""
        manager = neo4j_manager

        # First create some test data
        async with manager.driver.session() as session:
            await session.run("CREATE (n:TestNode {name: 'test1'})")
            await session.run("CREATE (n:TestNode {name: 'test2'})")
            await session.run("""
                MATCH (a:TestNode {name: 'test1'}), (b:TestNode {name: 'test2'})
                CREATE (a)-[:TEST_REL]->(b)
            """)

        # Verify data exists
        async with manager.driver.session() as session:
            result = await session.run("MATCH (n) RETURN count(n) as count")
            record = await result.single()
            assert record["count"] > 0, "Should have test nodes"

        # Clear all
        await manager.clear_all()

        # Verify empty
        async with manager.driver.session() as session:
            result = await session.run("MATCH (n) RETURN count(n) as count")
            record = await result.single()
            assert record["count"] == 0, "Should have zero nodes after clear"

    @pytest.mark.asyncio
    async def test_load_fixture(self, neo4j_manager):
        """Test loading micro fixture with 10 claims."""
        manager = neo4j_manager
        start = time.time()

        # Setup fresh
        await manager.setup_fresh()

        # Load fixture
        await manager.load_fixture(str(MICRO_FIXTURE))

        # Verify claims loaded
        async with manager.driver.session() as session:
            result = await session.run("MATCH (c:Claim) RETURN count(c) as count")
            record = await result.single()
            claim_count = record["count"]

        assert claim_count == 10, f"Expected 10 claims, got {claim_count}"

        # Verify entities loaded
        async with manager.driver.session() as session:
            result = await session.run("MATCH (e:Entity) RETURN count(e) as count")
            record = await result.single()
            entity_count = record["count"]

        assert entity_count == 4, f"Expected 4 entities, got {entity_count}"

        # Verify deterministic order (claims sorted by id)
        async with manager.driver.session() as session:
            result = await session.run("""
                MATCH (c:Claim)
                RETURN c.id as id
                ORDER BY c.id
            """)
            records = await result.values()
            claim_ids = [r[0] for r in records]

        expected_ids = [f"claim_{i:03d}" for i in range(1, 11)]
        assert claim_ids == expected_ids, f"Claims should be in deterministic order"

        elapsed = time.time() - start
        print(f"\n  Load fixture cycle: {elapsed:.2f}s")
        assert elapsed < 10, "Fixture load should be < 10s"

    @pytest.mark.asyncio
    async def test_snapshot(self, neo4j_manager):
        """Test snapshot returns ordered dict of state."""
        manager = neo4j_manager

        # Setup and load fixture
        await manager.setup_fresh()
        await manager.load_fixture(str(MICRO_FIXTURE))

        # Take snapshot
        snapshot = await manager.snapshot()

        # Verify snapshot structure
        assert 'claims' in snapshot
        assert 'surfaces' in snapshot
        assert 'incidents' in snapshot
        assert 'cases' in snapshot
        assert 'relationships' in snapshot

        # Verify claims in snapshot
        assert len(snapshot['claims']) == 10

        # Verify deterministic ordering (sorted by id)
        claim_ids = [c['id'] for c in snapshot['claims']]
        assert claim_ids == sorted(claim_ids), "Snapshot claims should be sorted by id"

        # Verify first claim content
        first_claim = snapshot['claims'][0]
        assert first_claim['id'] == 'claim_001'
        assert 'World Food Council' in first_claim['text']

    @pytest.mark.asyncio
    async def test_full_cycle_under_30s(self, neo4j_config):
        """Test full cycle completes in under 30 seconds."""
        start = time.time()

        async with TestNeo4jManager(neo4j_config) as manager:
            # 1. Setup fresh
            await manager.setup_fresh()

            # 2. Load fixture
            await manager.load_fixture(str(MICRO_FIXTURE))

            # 3. Snapshot
            snapshot = await manager.snapshot()
            assert len(snapshot['claims']) == 10

            # 4. Clear
            await manager.clear_all()

            # 5. Verify empty
            async with manager.driver.session() as session:
                result = await session.run("MATCH (n) RETURN count(n) as count")
                record = await result.single()
                assert record["count"] == 0

        elapsed = time.time() - start
        print(f"\n  Full cycle: {elapsed:.2f}s")
        assert elapsed < 30, f"Full cycle should be < 30s, was {elapsed:.2f}s"


# =============================================================================
# DIRECT RUN SUPPORT
# =============================================================================

if __name__ == "__main__":
    # Allow running directly for debugging
    async def main():
        config = TestNeo4jConfig()
        tests = TestNeo4jLifecycle()

        print("Running Neo4j lifecycle tests...")
        print("-" * 40)

        try:
            print("1. test_connect")
            await tests.test_connect(config)
            print("   PASSED")

            async with TestNeo4jManager(config) as manager:
                print("2. test_clear_all")
                await tests.test_clear_all(manager)
                print("   PASSED")

            async with TestNeo4jManager(config) as manager:
                print("3. test_load_fixture")
                await tests.test_load_fixture(manager)
                print("   PASSED")

            async with TestNeo4jManager(config) as manager:
                print("4. test_snapshot")
                await tests.test_snapshot(manager)
                print("   PASSED")

            print("5. test_full_cycle_under_30s")
            await tests.test_full_cycle_under_30s(config)
            print("   PASSED")

            print("-" * 40)
            print("ALL TESTS PASSED - Milestone 1 Stop/Go Gate: GREEN")

        except Exception as e:
            print(f"   FAILED: {e}")
            raise

    asyncio.run(main())
