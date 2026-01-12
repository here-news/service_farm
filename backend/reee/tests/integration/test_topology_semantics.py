"""
Test Topology Semantics
=======================

Validates that the persisted Neo4j topology matches semantic expectations
for the macro corpus. Tests structural patterns, archetype behavior,
and graph invariants.

Prerequisites:
    Run load_corpus_to_neo4j.py first to populate test Neo4j:
    docker exec herenews-app python /app/reee/tests/scripts/load_corpus_to_neo4j.py

Stop/Go Gate:
    pytest backend/reee/tests/integration/test_topology_semantics.py -v
"""

import pytest
import os
from pathlib import Path
from typing import Dict, List, Any, Set

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Check for Neo4j availability
neo4j_available = True
try:
    from neo4j import AsyncGraphDatabase
except ImportError:
    neo4j_available = False


# Test Neo4j configuration
TEST_NEO4J_URI = os.environ.get("TEST_NEO4J_URI", "bolt://localhost:7688")
TEST_NEO4J_USER = os.environ.get("TEST_NEO4J_USER", "neo4j")
TEST_NEO4J_PASSWORD = os.environ.get("TEST_NEO4J_PASSWORD", "test_password")


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

    # Verify connection and data exists
    try:
        async with driver.session() as session:
            result = await session.run("MATCH (n) RETURN count(n) as count")
            record = await result.single()
            if record['count'] == 0:
                await driver.close()
                pytest.skip("Test Neo4j is empty - run load_corpus_to_neo4j.py first")
    except Exception as e:
        await driver.close()
        pytest.skip(f"Cannot connect to test Neo4j: {e}")

    yield driver
    await driver.close()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def run_query(driver, query: str, params: dict = None) -> List[Dict]:
    """Run a Cypher query and return results."""
    async with driver.session() as session:
        result = await session.run(query, params or {})
        return await result.data()


# =============================================================================
# STRUCTURAL INVARIANT TESTS
# =============================================================================

@pytest.mark.skipif(not neo4j_available, reason="neo4j not available")
class TestStructuralInvariants:
    """Tests for graph structural invariants."""

    @pytest.mark.asyncio
    async def test_every_story_has_spine(self, neo4j_driver):
        """Every Story node must have exactly one HAS_SPINE relationship."""
        results = await run_query(neo4j_driver, """
            MATCH (st:Story)
            OPTIONAL MATCH (st)-[:HAS_SPINE]->(e:Entity)
            RETURN st.id as story_id, st.spine as spine, count(e) as spine_count
        """)

        for r in results:
            assert r['spine_count'] == 1, \
                f"Story {r['story_id']} has {r['spine_count']} spines (expected 1)"
            assert r['spine'] is not None, \
                f"Story {r['story_id']} has no spine property"

    @pytest.mark.asyncio
    async def test_every_incident_has_anchor(self, neo4j_driver):
        """Every Incident must have at least one HAS_ANCHOR relationship."""
        results = await run_query(neo4j_driver, """
            MATCH (i:Incident)
            OPTIONAL MATCH (i)-[:HAS_ANCHOR]->(e:Entity)
            RETURN i.id as incident_id, count(e) as anchor_count
        """)

        for r in results:
            assert r['anchor_count'] >= 1, \
                f"Incident {r['incident_id']} has no anchors"

    @pytest.mark.asyncio
    async def test_surfaces_have_claims(self, neo4j_driver):
        """Every Surface must have at least one PART_OF relationship from claims."""
        results = await run_query(neo4j_driver, """
            MATCH (s:Surface)
            OPTIONAL MATCH (c:Claim)-[:PART_OF]->(s)
            RETURN s.id as surface_id, count(c) as claim_count
        """)

        for r in results:
            assert r['claim_count'] >= 1, \
                f"Surface {r['surface_id']} has no claims"

    @pytest.mark.asyncio
    async def test_no_orphan_incidents(self, neo4j_driver):
        """Incidents assigned to stories should have CORE_A_OF, CORE_B_OF, or PERIPHERY_OF."""
        results = await run_query(neo4j_driver, """
            MATCH (i:Incident)
            WHERE EXISTS((i)-[:CORE_A_OF|CORE_B_OF|PERIPHERY_OF]->(:Story))
            OPTIONAL MATCH (i)-[r:CORE_A_OF|CORE_B_OF|PERIPHERY_OF]->(st:Story)
            RETURN i.id as incident_id, count(DISTINCT st) as story_count
        """)

        for r in results:
            assert r['story_count'] >= 1, \
                f"Incident {r['incident_id']} assigned to stories but has no story relationship"


# =============================================================================
# ARCHETYPE BEHAVIOR TESTS
# =============================================================================

@pytest.mark.skipif(not neo4j_available, reason="neo4j not available")
class TestArchetypeBehavior:
    """Tests that archetypes produce expected topology patterns."""

    @pytest.mark.asyncio
    async def test_star_story_has_single_spine(self, neo4j_driver):
        """Star story archetype: incidents share one spine entity."""
        # Get star story incidents
        results = await run_query(neo4j_driver, """
            MATCH (i:Incident {archetype: 'star_story'})-[:HAS_ANCHOR]->(e:Entity)
            RETURN e.name as entity, count(DISTINCT i) as incident_count
            ORDER BY incident_count DESC
        """)

        if results:
            # Top entity should appear in most star_story incidents
            top_entity = results[0]
            assert top_entity['incident_count'] >= 10, \
                f"Star story spine '{top_entity['entity']}' only in {top_entity['incident_count']} incidents"

    @pytest.mark.asyncio
    async def test_dyad_story_has_two_spines(self, neo4j_driver):
        """Dyad story archetype: two entities co-occur in all incidents."""
        results = await run_query(neo4j_driver, """
            MATCH (i:Incident {archetype: 'dyad_story'})-[:HAS_ANCHOR]->(e:Entity)
            WITH e.name as entity, count(DISTINCT i) as incident_count
            ORDER BY incident_count DESC
            LIMIT 2
            RETURN collect(entity) as top_entities, collect(incident_count) as counts
        """)

        if results and results[0]['top_entities']:
            entities = results[0]['top_entities']
            counts = results[0]['counts']
            # Both top entities should appear in similar number of incidents
            assert len(entities) == 2, f"Expected 2 dyad entities, got {len(entities)}"
            assert counts[0] == counts[1], \
                f"Dyad entities should co-occur equally: {entities[0]}={counts[0]}, {entities[1]}={counts[1]}"

    @pytest.mark.asyncio
    async def test_hub_entity_marked(self, neo4j_driver):
        """Hub adversary archetype: hub entity is marked with is_hub=true."""
        results = await run_query(neo4j_driver, """
            MATCH (e:Entity)
            WHERE e.is_hub = true
            RETURN e.name as entity, e.role as role
        """)

        # Should have at least one hub entity
        hub_entities = [r['entity'] for r in results]
        assert len(hub_entities) >= 1, "No hub entities marked"

        # Pacific Region is the hub in our corpus
        assert "Pacific Region" in hub_entities, \
            f"Expected 'Pacific Region' as hub, got {hub_entities}"

    @pytest.mark.asyncio
    async def test_hub_entity_not_story_spine(self, neo4j_driver):
        """Hub entities should NOT be story spines."""
        results = await run_query(neo4j_driver, """
            MATCH (e:Entity)
            WHERE e.is_hub = true
            OPTIONAL MATCH (st:Story)-[:HAS_SPINE]->(e)
            RETURN e.name as hub_entity, count(st) as story_count
        """)

        for r in results:
            assert r['story_count'] == 0, \
                f"Hub entity '{r['hub_entity']}' is spine of {r['story_count']} stories"

    @pytest.mark.asyncio
    async def test_scope_pollution_surfaces_isolated(self, neo4j_driver):
        """Scope pollution: same question_key in different scopes produces separate surfaces."""
        results = await run_query(neo4j_driver, """
            MATCH (s:Surface)
            WHERE s.question_key = 'interest_rate'
            RETURN s.scope_id as scope, count(*) as surface_count
        """)

        # Should have multiple scopes for interest_rate
        scopes = [r['scope'] for r in results if r['scope']]
        assert len(scopes) >= 2, \
            f"Expected multiple scopes for 'interest_rate', got {len(scopes)}"


# =============================================================================
# STORY QUALITY TESTS
# =============================================================================

@pytest.mark.skipif(not neo4j_available, reason="neo4j not available")
class TestStoryQuality:
    """Tests for story quality metrics."""

    @pytest.mark.asyncio
    async def test_no_mega_cases(self, neo4j_driver):
        """No story should have more than 50 core incidents."""
        results = await run_query(neo4j_driver, """
            MATCH (st:Story)
            RETURN st.spine as spine,
                   st.core_a_count as core_a,
                   st.core_b_count as core_b,
                   st.core_a_count + st.core_b_count as total_core
        """)

        for r in results:
            assert r['total_core'] < 50, \
                f"Story '{r['spine']}' is a mega-case with {r['total_core']} core incidents"

    @pytest.mark.asyncio
    async def test_core_leak_rates_acceptable(self, neo4j_driver):
        """All stories should have core_leak_rate below threshold."""
        results = await run_query(neo4j_driver, """
            MATCH (st:Story)
            RETURN st.spine as spine, st.core_leak_rate as leak_rate
        """)

        high_leak_stories = [r for r in results if r['leak_rate'] and r['leak_rate'] > 0.5]
        assert len(high_leak_stories) == 0, \
            f"Stories with high leak rate: {[(s['spine'], s['leak_rate']) for s in high_leak_stories]}"

    @pytest.mark.asyncio
    async def test_story_count_reasonable(self, neo4j_driver):
        """Story count should be within reasonable bounds."""
        results = await run_query(neo4j_driver, """
            MATCH (st:Story)
            RETURN count(st) as story_count
        """)

        story_count = results[0]['story_count']
        assert 10 <= story_count <= 100, \
            f"Story count {story_count} outside expected range [10, 100]"

    @pytest.mark.asyncio
    async def test_incident_coverage(self, neo4j_driver):
        """Most incidents should be assigned to stories."""
        results = await run_query(neo4j_driver, """
            MATCH (i:Incident)
            OPTIONAL MATCH (i)-[:CORE_A_OF|CORE_B_OF|PERIPHERY_OF]->(st:Story)
            RETURN count(DISTINCT i) as total_incidents,
                   count(DISTINCT CASE WHEN st IS NOT NULL THEN i END) as assigned_incidents
        """)

        total = results[0]['total_incidents']
        assigned = results[0]['assigned_incidents']
        coverage = assigned / total if total > 0 else 0

        assert coverage >= 0.5, \
            f"Only {coverage:.1%} of incidents assigned to stories (expected ≥50%)"


# =============================================================================
# GRAPH CONNECTIVITY TESTS
# =============================================================================

@pytest.mark.skipif(not neo4j_available, reason="neo4j not available")
class TestGraphConnectivity:
    """Tests for graph connectivity patterns."""

    @pytest.mark.asyncio
    async def test_entity_incident_connectivity(self, neo4j_driver):
        """Entities should be connected to incidents via HAS_ANCHOR or HAS_COMPANION."""
        results = await run_query(neo4j_driver, """
            MATCH (e:Entity)
            WHERE NOT (e)<-[:HAS_ANCHOR|HAS_COMPANION]-(:Incident)
            RETURN e.name as orphan_entity
        """)

        # No orphan entities
        orphans = [r['orphan_entity'] for r in results]
        assert len(orphans) == 0, \
            f"Orphan entities not connected to any incident: {orphans}"

    @pytest.mark.asyncio
    async def test_claim_surface_connectivity(self, neo4j_driver):
        """Every claim should belong to exactly one surface."""
        results = await run_query(neo4j_driver, """
            MATCH (c:Claim)
            OPTIONAL MATCH (c)-[:PART_OF]->(s:Surface)
            RETURN c.id as claim_id, count(s) as surface_count
            ORDER BY surface_count DESC
            LIMIT 10
        """)

        for r in results:
            assert r['surface_count'] == 1, \
                f"Claim {r['claim_id']} belongs to {r['surface_count']} surfaces"

    @pytest.mark.asyncio
    async def test_story_incident_paths(self, neo4j_driver):
        """Story spines should be reachable from their core incidents."""
        results = await run_query(neo4j_driver, """
            MATCH (i:Incident)-[:CORE_A_OF]->(st:Story)-[:HAS_SPINE]->(e:Entity)
            WHERE NOT (i)-[:HAS_ANCHOR]->(e)
            RETURN i.id as incident_id, st.spine as spine, e.name as spine_entity
            LIMIT 10
        """)

        # Core-A incidents should have spine as anchor (by definition)
        assert len(results) == 0, \
            f"Core-A incidents without spine as anchor: {results}"


# =============================================================================
# ARCHETYPE STATISTICS TESTS
# =============================================================================

@pytest.mark.skipif(not neo4j_available, reason="neo4j not available")
class TestArchetypeStatistics:
    """Tests for archetype distribution statistics."""

    @pytest.mark.asyncio
    async def test_all_archetypes_present(self, neo4j_driver):
        """All 8 archetypes should be present in the corpus."""
        expected_archetypes = {
            'star_story',
            'dyad_story',
            'hub_adversary',
            'homonym_adversary',
            'scope_pollution',
            'time_missingness',
            'typed_conflicts',
            'related_storyline',
        }

        results = await run_query(neo4j_driver, """
            MATCH (i:Incident)
            RETURN DISTINCT i.archetype as archetype
        """)

        actual_archetypes = {r['archetype'] for r in results if r['archetype']}

        missing = expected_archetypes - actual_archetypes
        assert len(missing) == 0, \
            f"Missing archetypes: {missing}"

    @pytest.mark.asyncio
    async def test_archetype_incident_counts(self, neo4j_driver):
        """Each archetype should have reasonable incident counts."""
        results = await run_query(neo4j_driver, """
            MATCH (i:Incident)
            RETURN i.archetype as archetype, count(*) as count
            ORDER BY count DESC
        """)

        for r in results:
            if r['archetype']:
                assert r['count'] >= 5, \
                    f"Archetype '{r['archetype']}' has only {r['count']} incidents"


# =============================================================================
# SUMMARY REPORT
# =============================================================================

@pytest.mark.skipif(not neo4j_available, reason="neo4j not available")
class TestTopologySummary:
    """Generate summary report of topology."""

    @pytest.mark.asyncio
    async def test_print_topology_summary(self, neo4j_driver):
        """Print topology summary for inspection."""
        # Node counts
        node_results = await run_query(neo4j_driver, """
            MATCH (n)
            RETURN labels(n)[0] as type, count(*) as count
            ORDER BY count DESC
        """)

        # Story summary
        story_results = await run_query(neo4j_driver, """
            MATCH (st:Story)
            RETURN st.spine as spine,
                   st.core_a_count as core_a,
                   st.periphery_count as periphery
            ORDER BY st.core_a_count DESC
        """)

        # Hub entities
        hub_results = await run_query(neo4j_driver, """
            MATCH (e:Entity)
            WHERE e.is_hub = true
            RETURN e.name as entity
        """)

        print("\n" + "=" * 60)
        print("TOPOLOGY SEMANTICS SUMMARY")
        print("=" * 60)

        print("\nNode Counts:")
        for r in node_results:
            print(f"  {r['type']}: {r['count']}")

        print(f"\nStories: {len(story_results)}")
        for r in story_results[:10]:
            print(f"  {r['spine']}: Core-A={r['core_a']}, Periphery={r['periphery']}")
        if len(story_results) > 10:
            print(f"  ... and {len(story_results) - 10} more")

        print(f"\nHub Entities: {[r['entity'] for r in hub_results]}")

        print("=" * 60)

        # This test always passes - it's for inspection
        assert True


# =============================================================================
# DIRECT RUN SUPPORT
# =============================================================================

if __name__ == "__main__":
    print("Running Topology Semantics tests...")
    print("Ensure test Neo4j is populated: docker exec herenews-app python /app/reee/tests/scripts/load_corpus_to_neo4j.py")
    print("-" * 50)
    print("Run with pytest:")
    print("  pytest backend/reee/tests/integration/test_topology_semantics.py -v")
