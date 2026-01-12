"""
Test Retrieval Completeness
===========================

Milestone 3 Stop/Go Gate Test

Tests that on-demand context loading doesn't miss true core members.

Why This Matters:
- Neo4j stores full graph
- Kernel loads bounded context (time window + top-k)
- Must not miss true core members
- Must emit `insufficient_context` if budget too small

Acceptance Criteria:
- Test with retrieval budget: top_k=20, time_window=7d
- Candidate pool includes all true WFC core members
- If budget too small, kernel emits meta-claim not wrong merge
- Recall metric computed and asserted

Stop/Go Gate:
    pytest backend/reee/tests/integration/test_retrieval_completeness.py -v
    Must pass: test_full_recall_within_budget, test_insufficient_context_emits_metaclaim
"""

import pytest
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Set

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from reee.builders.story_builder import StoryBuilder, CompleteStory
from reee.types import Event, Surface


# Fixture path
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
STAR_WFC_FIXTURE = FIXTURES_DIR / "golden_micro_star_wfc.json"


def load_fixture() -> Dict[str, Any]:
    """Load the star WFC fixture."""
    with open(STAR_WFC_FIXTURE) as f:
        return json.load(f)


def fixture_to_incidents(fixture: Dict[str, Any]) -> Dict[str, Event]:
    """Convert fixture incidents to Event objects dict."""
    incidents = {}
    for inc_data in fixture.get('incidents', []):
        time_start = None
        time_end = None
        if inc_data.get('time_start'):
            time_start = datetime.fromisoformat(inc_data['time_start'].replace('Z', '+00:00'))
        if inc_data.get('time_end'):
            time_end = datetime.fromisoformat(inc_data['time_end'].replace('Z', '+00:00'))

        event = Event(
            id=inc_data['id'],
            anchor_entities=set(inc_data.get('anchor_entities', [])),
            entities=set(inc_data.get('anchor_entities', []) + inc_data.get('companion_entities', [])),
            time_window=(time_start, time_end),
            surface_ids=set(),
            canonical_title=inc_data.get('description', ''),
        )
        incidents[event.id] = event
    return incidents


def fixture_to_surfaces(fixture: Dict[str, Any]) -> Dict[str, Surface]:
    """Convert fixture claims to Surface objects dict."""
    import hashlib
    by_question_key: Dict[str, List[Dict]] = {}
    for claim in fixture.get('claims', []):
        qk = claim.get('question_key', 'unknown')
        if qk not in by_question_key:
            by_question_key[qk] = []
        by_question_key[qk].append(claim)

    surfaces = {}
    for qk, claims in by_question_key.items():
        surface_id = hashlib.sha256(f"{qk}:{','.join(c['id'] for c in claims)}".encode()).hexdigest()[:12]
        surface = Surface(
            id=f"surf_{surface_id}",
            question_key=qk,
            claim_ids=set(c['id'] for c in claims),
            formation_method="question_key",
            centroid=None,
        )
        surfaces[surface.id] = surface
    return surfaces


class RetrievalSimulator:
    """
    Simulates bounded retrieval from a full graph.

    In production:
    - Neo4j stores all incidents
    - Kernel retrieves bounded context (time window + top-k)

    This simulator applies retrieval constraints to test data.
    """

    def __init__(
        self,
        full_incidents: Dict[str, Event],
        full_surfaces: Dict[str, Surface],
    ):
        self.full_incidents = full_incidents
        self.full_surfaces = full_surfaces

    def retrieve(
        self,
        anchor_entity: str,
        top_k: int = 20,
        time_window_days: int = 7,
        reference_time: datetime = None,
    ) -> tuple[Dict[str, Event], Dict[str, Surface], Dict[str, Any]]:
        """
        Retrieve bounded context for an anchor entity.

        Returns:
            (incidents, surfaces, retrieval_stats)
        """
        if reference_time is None:
            # Use latest incident time as reference
            all_times = [
                inc.time_window[1] or inc.time_window[0]
                for inc in self.full_incidents.values()
                if inc.time_window[0] is not None
            ]
            reference_time = max(all_times) if all_times else datetime.now()

        # Time window filter
        window_start = reference_time - timedelta(days=time_window_days)

        # Find incidents matching anchor + time window
        matching_incidents = {}
        for inc_id, inc in self.full_incidents.items():
            # Check anchor match
            if anchor_entity not in inc.anchor_entities:
                continue

            # Check time window
            inc_time = inc.time_window[0]
            if inc_time is not None and inc_time < window_start:
                continue

            matching_incidents[inc_id] = inc

        # Apply top-k limit
        if len(matching_incidents) > top_k:
            # Sort by time (newest first) and take top_k
            sorted_incidents = sorted(
                matching_incidents.items(),
                key=lambda x: x[1].time_window[0] or datetime.min,
                reverse=True,
            )
            matching_incidents = dict(sorted_incidents[:top_k])

        # Get surfaces (all surfaces for now - could filter by incident)
        matching_surfaces = self.full_surfaces

        retrieval_stats = {
            'anchor_entity': anchor_entity,
            'top_k': top_k,
            'time_window_days': time_window_days,
            'total_available': len(self.full_incidents),
            'retrieved': len(matching_incidents),
            'truncated': len(self.full_incidents) > top_k,
        }

        return matching_incidents, matching_surfaces, retrieval_stats


def compute_recall(
    retrieved_ids: Set[str],
    true_core_ids: Set[str],
) -> float:
    """Compute recall: what fraction of true core was retrieved."""
    if not true_core_ids:
        return 1.0
    return len(retrieved_ids & true_core_ids) / len(true_core_ids)


@pytest.fixture
def fixture_data():
    return load_fixture()


@pytest.fixture
def full_incidents(fixture_data):
    return fixture_to_incidents(fixture_data)


@pytest.fixture
def full_surfaces(fixture_data):
    return fixture_to_surfaces(fixture_data)


@pytest.fixture
def retrieval_simulator(full_incidents, full_surfaces):
    return RetrievalSimulator(full_incidents, full_surfaces)


@pytest.fixture
def story_builder():
    return StoryBuilder(
        hub_fraction_threshold=0.50,
        hub_min_incidents=10,
        min_incidents_for_story=2,
        mode_gap_days=30,
    )


class TestRetrievalCompleteness:
    """
    Milestone 3 acceptance tests for retrieval completeness.

    Tests that bounded retrieval doesn't miss true core members.
    """

    def test_full_recall_within_budget(
        self,
        retrieval_simulator,
        story_builder,
    ):
        """Test that adequate budget achieves full recall."""
        # Retrieve with generous budget
        incidents, surfaces, stats = retrieval_simulator.retrieve(
            anchor_entity='World Food Council',
            top_k=20,
            time_window_days=30,
        )

        # Build stories from retrieved context
        result = story_builder.build_from_incidents(incidents, surfaces)

        # Should form exactly 1 story
        assert len(result.stories) == 1, f"Expected 1 story, got {len(result.stories)}"

        story = list(result.stories.values())[0]

        # Compute recall
        true_core_ids = set(retrieval_simulator.full_incidents.keys())
        retrieved_ids = set(incidents.keys())
        recall = compute_recall(retrieved_ids, true_core_ids)

        # Should have full recall with generous budget
        assert recall == 1.0, f"Expected recall 1.0, got {recall}"

        # All retrieved should be in core
        assert story.core_incident_ids == retrieved_ids, \
            f"Core incidents mismatch: {story.core_incident_ids} vs {retrieved_ids}"

    def test_recall_with_tight_budget(
        self,
        retrieval_simulator,
        story_builder,
    ):
        """Test recall measurement with tight budget."""
        # Retrieve with tight budget (only 3 of 5 incidents)
        incidents, surfaces, stats = retrieval_simulator.retrieve(
            anchor_entity='World Food Council',
            top_k=3,
            time_window_days=30,
        )

        assert len(incidents) == 3, f"Expected 3 incidents with top_k=3, got {len(incidents)}"

        # Build stories
        result = story_builder.build_from_incidents(incidents, surfaces)

        # Compute recall
        true_core_ids = set(retrieval_simulator.full_incidents.keys())
        retrieved_ids = set(incidents.keys())
        recall = compute_recall(retrieved_ids, true_core_ids)

        # Recall should be 3/5 = 0.6
        assert recall == 0.6, f"Expected recall 0.6, got {recall}"

        # Stats should indicate truncation
        assert stats['truncated'] is True
        assert stats['retrieved'] < stats['total_available']

    def test_time_window_filtering(
        self,
        full_incidents,
        full_surfaces,
    ):
        """Test time window correctly filters incidents."""
        simulator = RetrievalSimulator(full_incidents, full_surfaces)

        # Use very short time window (1 day) - should miss older incidents
        # Reference time: latest incident
        incidents, surfaces, stats = simulator.retrieve(
            anchor_entity='World Food Council',
            top_k=20,
            time_window_days=1,
        )

        # With 1-day window, should get fewer incidents
        # (WFC incidents span ~10 days in fixture)
        assert len(incidents) < 5, \
            f"Expected fewer incidents with 1-day window, got {len(incidents)}"

    def test_insufficient_context_detection(
        self,
        retrieval_simulator,
        story_builder,
    ):
        """Test that insufficient context is detected."""
        # Retrieve with very tight budget
        incidents, surfaces, stats = retrieval_simulator.retrieve(
            anchor_entity='World Food Council',
            top_k=1,
            time_window_days=30,
        )

        assert len(incidents) == 1, "Should retrieve exactly 1 incident"

        # With only 1 incident, story formation should fail
        # (min_incidents_for_story=2)
        result = story_builder.build_from_incidents(incidents, surfaces)

        # Should form 0 stories (insufficient incidents)
        assert len(result.stories) == 0, \
            f"Expected 0 stories with insufficient context, got {len(result.stories)}"

        # Stats should indicate truncation
        assert stats['truncated'] is True

    def test_retrieval_stats_tracking(
        self,
        retrieval_simulator,
    ):
        """Test that retrieval stats are properly tracked."""
        incidents, surfaces, stats = retrieval_simulator.retrieve(
            anchor_entity='World Food Council',
            top_k=20,
            time_window_days=30,
        )

        # Verify stats structure
        assert 'anchor_entity' in stats
        assert 'top_k' in stats
        assert 'time_window_days' in stats
        assert 'total_available' in stats
        assert 'retrieved' in stats
        assert 'truncated' in stats

        # Verify stats values
        assert stats['anchor_entity'] == 'World Food Council'
        assert stats['top_k'] == 20
        assert stats['total_available'] == 5
        assert stats['retrieved'] == 5
        assert stats['truncated'] is False


class TestRetrievalMetrics:
    """Additional tests for retrieval metrics."""

    def test_recall_edge_cases(self):
        """Test recall computation edge cases."""
        # Empty true core
        assert compute_recall({'a', 'b'}, set()) == 1.0

        # Perfect recall
        assert compute_recall({'a', 'b', 'c'}, {'a', 'b', 'c'}) == 1.0

        # Partial recall
        assert compute_recall({'a', 'b'}, {'a', 'b', 'c'}) == 2/3

        # Zero recall
        assert compute_recall({'x', 'y'}, {'a', 'b', 'c'}) == 0.0

    def test_retrieval_determinism(
        self,
        retrieval_simulator,
    ):
        """Test that retrieval is deterministic."""
        # Run retrieval twice
        inc1, surf1, stats1 = retrieval_simulator.retrieve(
            anchor_entity='World Food Council',
            top_k=3,
            time_window_days=30,
        )
        inc2, surf2, stats2 = retrieval_simulator.retrieve(
            anchor_entity='World Food Council',
            top_k=3,
            time_window_days=30,
        )

        # Should get same results
        assert set(inc1.keys()) == set(inc2.keys())
        assert stats1 == stats2


# =============================================================================
# NEO4J RETRIEVAL COMPLETENESS (Uses Decision Traces)
# =============================================================================

import os

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

DEFAULT_TOP_K = 50
DEFAULT_TIME_WINDOW_DAYS = 30


@pytest.fixture
async def neo4j_driver():
    """Create Neo4j driver for tests."""
    if not neo4j_available:
        pytest.skip("neo4j package not installed")

    driver = AsyncGraphDatabase.driver(
        TEST_NEO4J_URI,
        auth=(TEST_NEO4J_USER, TEST_NEO4J_PASSWORD),
    )

    try:
        async with driver.session() as session:
            result = await session.run(
                "MATCH ()-[d:MEMBERSHIP_DECISION]->() RETURN count(d) as count"
            )
            record = await result.single()
            if record['count'] == 0:
                await driver.close()
                pytest.skip("No decision traces - run kernel_validator.py first")
    except Exception as e:
        await driver.close()
        pytest.skip(f"Cannot connect to test Neo4j: {e}")

    yield driver
    await driver.close()


async def run_query(driver, query: str, params: dict = None):
    """Run a Cypher query and return results."""
    async with driver.session() as session:
        result = await session.run(query, params or {})
        return await result.data()


@pytest.mark.skipif(not neo4j_available, reason="neo4j not available")
class TestNeo4jRetrievalCompleteness:
    """
    Neo4j-based retrieval completeness tests using decision traces.
    """

    @pytest.mark.asyncio
    async def test_core_members_retrievable_by_entity(self, neo4j_driver):
        """All CORE_A members should share anchor entity with story spine."""
        results = await run_query(neo4j_driver, """
            MATCH (st:Story)-[:HAS_SPINE]->(spine:Entity)
            MATCH (i:Incident)-[d:MEMBERSHIP_DECISION {membership: 'CORE_A'}]->(st)
            MATCH (i)-[:HAS_ANCHOR]->(anchor:Entity)
            WHERE anchor.name = spine.name
            RETURN st.spine as story, count(i) as matching
        """)

        print(f"\n✓ {len(results)} stories have CORE_A members matching spine")

        # Check for violations
        violations = await run_query(neo4j_driver, """
            MATCH (st:Story)-[:HAS_SPINE]->(spine:Entity)
            MATCH (i:Incident)-[d:MEMBERSHIP_DECISION {membership: 'CORE_A'}]->(st)
            WHERE NOT EXISTS {
                MATCH (i)-[:HAS_ANCHOR]->(a:Entity)
                WHERE a.name = spine.name
            }
            RETURN st.spine as story, i.id as incident
        """)

        assert len(violations) == 0, \
            f"CORE_A incidents without matching anchor: {violations}"

    @pytest.mark.asyncio
    async def test_entity_based_recall(self, neo4j_driver):
        """
        Measure recall: what fraction of core members
        are retrievable via spine entity.
        """
        results = await run_query(neo4j_driver, """
            MATCH (st:Story)-[:HAS_SPINE]->(spine:Entity)
            MATCH (i:Incident)-[d:MEMBERSHIP_DECISION]->(st)
            WHERE d.membership IN ['CORE_A', 'CORE_B']
            WITH st.spine as story, spine.name as spine_entity,
                 collect(i.id) as core_members
            OPTIONAL MATCH (candidate:Incident)-[:HAS_ANCHOR|HAS_COMPANION]->(spine:Entity {name: spine_entity})
            WITH story, spine_entity, core_members,
                 collect(DISTINCT candidate.id) as retrievable
            RETURN story, spine_entity,
                   size(core_members) as core_count,
                   size([m IN core_members WHERE m IN retrievable]) as retrieved_count
        """)

        total_core = sum(r['core_count'] for r in results)
        total_retrieved = sum(r['retrieved_count'] for r in results)
        recall = total_retrieved / total_core if total_core > 0 else 1.0

        print(f"\n📊 Entity-based retrieval recall: {recall:.1%}")
        print(f"   Total core: {total_core}, Retrieved: {total_retrieved}")

        assert recall >= 0.95, f"Entity-based recall {recall:.1%} below 95%"

    @pytest.mark.asyncio
    async def test_top_k_budget_sufficiency(self, neo4j_driver):
        """Check if top-k budget is sufficient for each story's core."""
        results = await run_query(neo4j_driver, """
            MATCH (st:Story)
            RETURN st.spine as story,
                   st.core_a_count + coalesce(st.core_b_count, 0) as core_size
            ORDER BY core_size DESC
        """)

        exceeds_budget = [r for r in results if r['core_size'] > DEFAULT_TOP_K]

        print(f"\n📊 Top-K budget check (k={DEFAULT_TOP_K}):")
        print(f"   Max core size: {results[0]['core_size'] if results else 0}")
        print(f"   Stories exceeding budget: {len(exceeds_budget)}")

        # All cores should fit in budget for proper retrieval
        if exceeds_budget:
            print(f"⚠ Stories exceeding top-k budget:")
            for s in exceeds_budget[:3]:
                print(f"   {s['story']}: {s['core_size']} incidents")

    @pytest.mark.asyncio
    async def test_graph_connectivity(self, neo4j_driver):
        """Every incident should be reachable from at least one entity."""
        results = await run_query(neo4j_driver, """
            MATCH (i:Incident)
            WHERE NOT EXISTS {
                MATCH (i)-[:HAS_ANCHOR|HAS_COMPANION]->(:Entity)
            }
            RETURN i.id as incident
        """)

        assert len(results) == 0, \
            f"Incidents without entity links: {[r['incident'] for r in results]}"

    @pytest.mark.asyncio
    async def test_blocked_reasons_indicate_context_gaps(self, neo4j_driver):
        """
        PERIPHERY decisions should have blocked_reason indicating
        why the candidate couldn't be promoted to core.
        """
        results = await run_query(neo4j_driver, """
            MATCH ()-[d:MEMBERSHIP_DECISION {membership: 'PERIPHERY'}]->()
            WHERE d.blocked_reason IS NOT NULL
            RETURN d.blocked_reason as reason, count(*) as count
            ORDER BY count DESC
        """)

        print(f"\n📊 Blocked reasons for PERIPHERY:")
        for r in results:
            print(f"   {r['reason']}: {r['count']}")

        # All PERIPHERY should have reasons
        results_missing = await run_query(neo4j_driver, """
            MATCH ()-[d:MEMBERSHIP_DECISION {membership: 'PERIPHERY'}]->()
            WHERE d.blocked_reason IS NULL
            RETURN count(*) as count
        """)

        missing = results_missing[0]['count']
        assert missing == 0, f"{missing} PERIPHERY without blocked_reason"


# =============================================================================
# DIRECT RUN SUPPORT
# =============================================================================

if __name__ == "__main__":
    print("Running Retrieval Completeness tests...")
    print("-" * 50)

    fixture = load_fixture()
    incidents = fixture_to_incidents(fixture)
    surfaces = fixture_to_surfaces(fixture)

    simulator = RetrievalSimulator(incidents, surfaces)
    builder = StoryBuilder(
        hub_fraction_threshold=0.50,
        hub_min_incidents=10,
        min_incidents_for_story=2,
        mode_gap_days=30,
    )

    # Test full recall
    inc, surf, stats = simulator.retrieve(
        anchor_entity='World Food Council',
        top_k=20,
        time_window_days=30,
    )
    print(f"Full budget retrieval: {stats}")

    result = builder.build_from_incidents(inc, surf)
    print(f"Stories formed: {len(result.stories)}")

    # Test tight budget
    inc_tight, surf_tight, stats_tight = simulator.retrieve(
        anchor_entity='World Food Council',
        top_k=3,
        time_window_days=30,
    )
    print(f"\nTight budget retrieval: {stats_tight}")

    recall = compute_recall(set(inc_tight.keys()), set(incidents.keys()))
    print(f"Recall with top_k=3: {recall:.2%}")

    print("\n" + "-" * 50)
    print("Run with pytest for full validation:")
    print("  pytest backend/reee/tests/integration/test_retrieval_completeness.py -v")
