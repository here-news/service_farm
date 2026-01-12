"""
Test Soft Envelopes
===================

These are WARNINGS, not failures. They indicate parameter tuning issues
or corpus drift, but do not block deployment.

Soft Envelopes:
1. Story count within expected range
2. Max core size within expected range
3. Hub count within expected range
4. Periphery rate within expected range
5. Witness scarcity within expected range

These emit xfail/warnings on violation, not failures.

Prerequisites:
    Run kernel_validator.py first to populate decision traces:
    docker exec herenews-test-runner python -m reee.tests.scripts.kernel_validator

Usage:
    docker exec herenews-test-runner python -m pytest reee/tests/integration/test_soft_envelopes.py -v
"""

import pytest
import os
from pathlib import Path
from typing import Dict, List, Any
import warnings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Check for Neo4j availability
neo4j_available = True
try:
    from neo4j import AsyncGraphDatabase
except ImportError:
    neo4j_available = False


# Test Neo4j configuration
TEST_NEO4J_URI = os.environ.get("TEST_NEO4J_URI") or os.environ.get("NEO4J_URI", "bolt://localhost:7688")
TEST_NEO4J_USER = os.environ.get("TEST_NEO4J_USER") or os.environ.get("NEO4J_USER", "neo4j")
TEST_NEO4J_PASSWORD = os.environ.get("TEST_NEO4J_PASSWORD") or os.environ.get("NEO4J_PASSWORD", "test_password")


# =============================================================================
# ENVELOPE DEFINITIONS
# =============================================================================

# These ranges are expected for the macro corpus (seed=42)
# Adjust based on corpus size and archetype distribution

ENVELOPES = {
    'story_count': (20, 80),        # Expected story count range
    'max_core_size': (2, 30),       # Max incidents in any story core
    'hub_count': (1, 10),           # Number of hub entities detected
    'periphery_rate': (0.05, 0.40), # Fraction of PERIPHERY decisions
    'witness_scarcity': (0.0, 0.50), # Fraction of CORE_B without witnesses
}


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
async def neo4j_driver():
    """Create Neo4j driver for tests."""
    if not neo4j_available:
        pytest.skip("neo4j package not installed")

    driver = AsyncGraphDatabase.driver(
        TEST_NEO4J_URI,
        auth=(TEST_NEO4J_USER, TEST_NEO4J_PASSWORD),
    )

    # Verify data exists
    try:
        async with driver.session() as session:
            result = await session.run(
                "MATCH (st:Story) RETURN count(st) as count"
            )
            record = await result.single()
            if record['count'] == 0:
                await driver.close()
                pytest.skip("No stories - run kernel_validator.py first")
    except Exception as e:
        await driver.close()
        pytest.skip(f"Cannot connect to test Neo4j: {e}")

    yield driver
    await driver.close()


async def run_query(driver, query: str, params: dict = None) -> List[Dict]:
    """Run a Cypher query and return results."""
    async with driver.session() as session:
        result = await session.run(query, params or {})
        return await result.data()


def check_envelope(value, envelope_name: str, context: str = ""):
    """Check if value is within envelope, emit warning if not."""
    low, high = ENVELOPES[envelope_name]
    if value < low:
        warnings.warn(
            f"⚠ {envelope_name}: {value} below minimum {low}. {context}",
            UserWarning
        )
        return False
    if value > high:
        warnings.warn(
            f"⚠ {envelope_name}: {value} above maximum {high}. {context}",
            UserWarning
        )
        return False
    return True


# =============================================================================
# ENVELOPE 1: Story Count Range
# =============================================================================

@pytest.mark.skipif(not neo4j_available, reason="neo4j not available")
class TestStoryCountEnvelope:
    """
    ENVELOPE: Story count should be within expected range.
    Too few = under-clustering, too many = over-fragmentation.
    """

    @pytest.mark.asyncio
    async def test_story_count_in_range(self, neo4j_driver):
        """Story count should be within envelope."""
        results = await run_query(neo4j_driver, """
            MATCH (st:Story)
            RETURN count(st) as count
        """)

        count = results[0]['count']
        low, high = ENVELOPES['story_count']

        print(f"\n📊 Story count: {count} (envelope: {low}-{high})")

        # This is a soft check - emit warning but don't fail
        in_envelope = check_envelope(count, 'story_count')

        # Always pass but report status
        assert True, f"Story count: {count}"


# =============================================================================
# ENVELOPE 2: Max Core Size Range
# =============================================================================

@pytest.mark.skipif(not neo4j_available, reason="neo4j not available")
class TestMaxCoreSizeEnvelope:
    """
    ENVELOPE: No story core should be too large.
    Large cores may indicate insufficient discrimination.
    """

    @pytest.mark.asyncio
    async def test_max_core_size_in_range(self, neo4j_driver):
        """Max core size should be within envelope."""
        results = await run_query(neo4j_driver, """
            MATCH (st:Story)
            RETURN st.spine as story,
                   st.core_a_count + coalesce(st.core_b_count, 0) as core_size
            ORDER BY core_size DESC
            LIMIT 1
        """)

        if not results:
            pytest.skip("No stories found")

        max_size = results[0]['core_size']
        largest_story = results[0]['story']
        low, high = ENVELOPES['max_core_size']

        print(f"\n📊 Max core size: {max_size} in '{largest_story}' (envelope: {low}-{high})")

        in_envelope = check_envelope(
            max_size, 'max_core_size',
            f"Largest story: {largest_story}"
        )

        assert True


# =============================================================================
# ENVELOPE 3: Hub Count Range
# =============================================================================

@pytest.mark.skipif(not neo4j_available, reason="neo4j not available")
class TestHubCountEnvelope:
    """
    ENVELOPE: Hub entity count should be within range.
    Too few = hub detection underpowered, too many = over-detection.
    """

    @pytest.mark.asyncio
    async def test_hub_count_in_range(self, neo4j_driver):
        """Hub count should be within envelope."""
        results = await run_query(neo4j_driver, """
            MATCH (e:Entity)
            WHERE e.is_hub = true
            RETURN count(e) as count, collect(e.name) as hubs
        """)

        count = results[0]['count']
        hubs = results[0]['hubs'] or []
        low, high = ENVELOPES['hub_count']

        print(f"\n📊 Hub count: {count} (envelope: {low}-{high})")
        if hubs:
            print(f"   Hubs: {', '.join(hubs[:5])}{'...' if len(hubs) > 5 else ''}")

        in_envelope = check_envelope(count, 'hub_count', f"Hubs: {hubs}")

        assert True

    @pytest.mark.asyncio
    async def test_hub_lens_sanity_report(self, neo4j_driver):
        """
        Hub-lens sanity report: top entities by incident coverage.
        Shows if hubs are being treated appropriately (lens yes, story spine no).
        """
        # Get top 10 entities by incident coverage
        results = await run_query(neo4j_driver, """
            MATCH (e:Entity)<-[:HAS_ANCHOR|HAS_COMPANION]-(i:Incident)
            WITH e.name as entity, e.is_hub as is_hub, count(DISTINCT i) as incident_count
            ORDER BY incident_count DESC
            LIMIT 10
            RETURN entity, is_hub, incident_count
        """)

        # Get total incidents for percentage
        total_result = await run_query(neo4j_driver, """
            MATCH (i:Incident) RETURN count(i) as total
        """)
        total_incidents = total_result[0]['total'] if total_result else 1

        print(f"\n📊 Hub-Lens Sanity Report (Top 10 by incident coverage):")
        print(f"   {'Entity':<30} {'Coverage':>10} {'Is Hub':>8}")
        print(f"   {'-'*30} {'-'*10} {'-'*8}")

        for r in results:
            coverage_pct = r['incident_count'] / total_incidents * 100
            is_hub = "✓ hub" if r['is_hub'] else ""
            print(f"   {r['entity']:<30} {coverage_pct:>8.1f}% {is_hub:>8}")

        # Check for entities with high coverage NOT marked as hub
        high_coverage_non_hubs = [
            r for r in results
            if r['incident_count'] / total_incidents > 0.20 and not r['is_hub']
        ]

        if high_coverage_non_hubs:
            print(f"\n⚠ Entities with >20% coverage NOT marked as hub:")
            for r in high_coverage_non_hubs:
                pct = r['incident_count'] / total_incidents * 100
                print(f"   {r['entity']}: {pct:.1f}%")

        # Check for hubs that ARE story spines (should be 0)
        spine_results = await run_query(neo4j_driver, """
            MATCH (st:Story)-[:HAS_SPINE]->(e:Entity)
            WHERE e.is_hub = true
            RETURN st.spine as story, e.name as hub
        """)

        if spine_results:
            print(f"\n❌ Hub entities used as story spines (VIOLATION):")
            for r in spine_results:
                print(f"   {r['hub']} → {r['story']}")

        assert True  # Report only, no hard failure


# =============================================================================
# ENVELOPE 4: Periphery Rate
# =============================================================================

@pytest.mark.skipif(not neo4j_available, reason="neo4j not available")
class TestPeripheryRateEnvelope:
    """
    ENVELOPE: Periphery rate should be within expected range.
    Too low = over-inclusion in core, too high = under-inclusion.
    """

    @pytest.mark.asyncio
    async def test_periphery_rate_in_range(self, neo4j_driver):
        """Periphery rate should be within envelope."""
        results = await run_query(neo4j_driver, """
            MATCH ()-[d:MEMBERSHIP_DECISION]->()
            WITH d.membership as membership, count(*) as cnt
            RETURN membership, cnt
        """)

        total = sum(r['cnt'] for r in results)
        periphery = sum(r['cnt'] for r in results if r['membership'] == 'PERIPHERY')

        if total == 0:
            pytest.skip("No membership decisions")

        rate = periphery / total
        low, high = ENVELOPES['periphery_rate']

        print(f"\n📊 Periphery rate: {rate:.2%} ({periphery}/{total}) (envelope: {low*100:.0f}%-{high*100:.0f}%)")

        in_envelope = check_envelope(rate, 'periphery_rate')

        assert True


# =============================================================================
# ENVELOPE 5: Witness Scarcity
# =============================================================================

@pytest.mark.skipif(not neo4j_available, reason="neo4j not available")
class TestWitnessScarcityEnvelope:
    """
    ENVELOPE: Witness scarcity should be low.
    High scarcity = CORE_B decisions made without structural evidence.
    """

    @pytest.mark.asyncio
    async def test_witness_scarcity_in_range(self, neo4j_driver):
        """Witness scarcity should be within envelope."""
        results = await run_query(neo4j_driver, """
            MATCH ()-[d:MEMBERSHIP_DECISION {membership: 'CORE_B'}]->()
            RETURN count(*) as total,
                   sum(CASE WHEN d.witnesses IS NULL OR size(d.witnesses) = 0 THEN 1 ELSE 0 END) as without_witnesses
        """)

        total = results[0]['total']
        without = results[0]['without_witnesses']

        if total == 0:
            print("\n📊 No CORE_B decisions (scarcity N/A)")
            return  # No CORE_B is fine

        scarcity = without / total
        low, high = ENVELOPES['witness_scarcity']

        print(f"\n📊 Witness scarcity: {scarcity:.2%} ({without}/{total} CORE_B without witnesses)")
        print(f"   Envelope: {low*100:.0f}%-{high*100:.0f}%")

        in_envelope = check_envelope(scarcity, 'witness_scarcity')

        assert True


# =============================================================================
# SUMMARY
# =============================================================================

@pytest.mark.skipif(not neo4j_available, reason="neo4j not available")
class TestEnvelopeSummary:
    """Summary of all envelope checks."""

    @pytest.mark.asyncio
    async def test_print_envelope_summary(self, neo4j_driver):
        """Print summary of envelope checks."""
        print("\n" + "=" * 60)
        print("SOFT ENVELOPE CHECK SUMMARY")
        print("=" * 60)

        # Collect all metrics
        metrics = {}

        # Story count
        results = await run_query(neo4j_driver, "MATCH (st:Story) RETURN count(st) as count")
        metrics['story_count'] = results[0]['count']

        # Max core size
        results = await run_query(neo4j_driver, """
            MATCH (st:Story)
            RETURN max(st.core_a_count + coalesce(st.core_b_count, 0)) as max_core
        """)
        metrics['max_core_size'] = results[0]['max_core'] or 0

        # Hub count
        results = await run_query(neo4j_driver, """
            MATCH (e:Entity) WHERE e.is_hub = true RETURN count(e) as count
        """)
        metrics['hub_count'] = results[0]['count']

        # Periphery rate
        results = await run_query(neo4j_driver, """
            MATCH ()-[d:MEMBERSHIP_DECISION]->()
            RETURN sum(CASE WHEN d.membership = 'PERIPHERY' THEN 1 ELSE 0 END) as periphery,
                   count(*) as total
        """)
        total = results[0]['total']
        periphery = results[0]['periphery']
        metrics['periphery_rate'] = periphery / total if total > 0 else 0

        # Print summary
        violations = 0
        for name, value in metrics.items():
            low, high = ENVELOPES[name]
            status = "✓" if low <= value <= high else "⚠"
            if status == "⚠":
                violations += 1

            if isinstance(value, float):
                print(f"  {status} {name}: {value:.2%} (envelope: {low}-{high})")
            else:
                print(f"  {status} {name}: {value} (envelope: {low}-{high})")

        print("=" * 60)
        if violations > 0:
            print(f"⚠ {violations} envelope violations (review recommended)")
        else:
            print("✓ All envelopes within expected ranges")
        print("=" * 60)

        # Always pass - these are warnings only
        assert True


# =============================================================================
# DIRECT RUN SUPPORT
# =============================================================================

if __name__ == "__main__":
    print("Running Soft Envelope tests...")
    print("Prerequisites: Run kernel_validator.py first")
    print("-" * 50)
    print("Run with pytest:")
    print("  docker exec herenews-test-runner python -m pytest reee/tests/integration/test_soft_envelopes.py -v")
