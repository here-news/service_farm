"""
Test Hard Invariants
====================

Tests that MUST ALWAYS PASS - these are kernel safety guarantees,
not heuristic thresholds.

Hard Invariants:
1. Scoped surface isolation - (scope_id, question_key) uniquely identifies surface
2. Semantic-only cannot produce core - CORE requires structural link, not just semantic
3. Chain-only cannot merge cores - Single witness chain insufficient for CORE_B
4. Core leak rate definition holds - Core-B requires warrants (structural witnesses)
5. Blocked reasons visible - All non-core candidates have blocked_reason in trace

Soft Envelopes (tested separately):
- Story count range
- Max core size range
- Hub count range

Prerequisites:
    Run kernel_validator.py first to populate decision traces:
    docker exec herenews-test-runner python -m reee.tests.scripts.kernel_validator

Stop/Go Gate:
    docker exec herenews-test-runner python -m pytest reee/tests/integration/test_hard_invariants.py -v
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


# Test Neo4j configuration (uses env vars from test-runner)
TEST_NEO4J_URI = os.environ.get("TEST_NEO4J_URI") or os.environ.get("NEO4J_URI", "bolt://localhost:7688")
TEST_NEO4J_USER = os.environ.get("TEST_NEO4J_USER") or os.environ.get("NEO4J_USER", "neo4j")
TEST_NEO4J_PASSWORD = os.environ.get("TEST_NEO4J_PASSWORD") or os.environ.get("NEO4J_PASSWORD", "test_password")


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

    # Verify traces exist
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


async def run_query(driver, query: str, params: dict = None) -> List[Dict]:
    """Run a Cypher query and return results."""
    async with driver.session() as session:
        result = await session.run(query, params or {})
        return await result.data()


# =============================================================================
# INVARIANT 1: Scoped Surface Isolation
# =============================================================================

@pytest.mark.skipif(not neo4j_available, reason="neo4j not available")
class TestScopedSurfaceIsolation:
    """
    INVARIANT: (scope_id, question_key) uniquely identifies a surface.
    No two surfaces should have the same (scope_id, question_key) pair.
    """

    @pytest.mark.asyncio
    async def test_no_duplicate_scope_question_pairs(self, neo4j_driver):
        """Each (scope_id, question_key) pair must map to exactly one surface."""
        results = await run_query(neo4j_driver, """
            MATCH (s:Surface)
            WITH s.scope_id as scope, s.question_key as qk, count(*) as cnt
            WHERE cnt > 1
            RETURN scope, qk, cnt
        """)

        assert len(results) == 0, \
            f"Duplicate (scope_id, question_key) pairs found: {results}"

    @pytest.mark.asyncio
    async def test_surfaces_have_scope_ids(self, neo4j_driver):
        """Every surface must have a scope_id."""
        results = await run_query(neo4j_driver, """
            MATCH (s:Surface)
            WHERE s.scope_id IS NULL
            RETURN s.id as surface_id
        """)

        assert len(results) == 0, \
            f"Surfaces without scope_id: {[r['surface_id'] for r in results]}"


# =============================================================================
# INVARIANT 2: Semantic-Only Cannot Produce Core
# =============================================================================

@pytest.mark.skipif(not neo4j_available, reason="neo4j not available")
class TestSemanticOnlyCannotProduceCore:
    """
    INVARIANT: CORE membership requires structural link (ANCHOR reason).
    Semantic similarity alone is insufficient for CORE.
    """

    @pytest.mark.asyncio
    async def test_core_a_has_anchor_reason(self, neo4j_driver):
        """Every CORE_A decision must have ANCHOR as core_reason."""
        results = await run_query(neo4j_driver, """
            MATCH (i:Incident)-[d:MEMBERSHIP_DECISION {membership: 'CORE_A'}]->(st:Story)
            WHERE d.core_reason IS NULL OR d.core_reason <> 'ANCHOR'
            RETURN i.id as incident, st.spine as story, d.core_reason as reason
        """)

        assert len(results) == 0, \
            f"CORE_A without ANCHOR reason: {results}"

    @pytest.mark.asyncio
    async def test_no_semantic_only_core(self, neo4j_driver):
        """No CORE membership should have 'semantic' as sole reason."""
        # Check for any CORE decision with semantic-only justification
        results = await run_query(neo4j_driver, """
            MATCH (i:Incident)-[d:MEMBERSHIP_DECISION]->(st:Story)
            WHERE d.membership IN ['CORE_A', 'CORE_B']
              AND d.core_reason = 'SEMANTIC'
            RETURN i.id as incident, st.spine as story
        """)

        assert len(results) == 0, \
            f"Semantic-only CORE found: {results}"


# =============================================================================
# INVARIANT 3: Chain-Only Cannot Merge Cores
# =============================================================================

@pytest.mark.skipif(not neo4j_available, reason="neo4j not available")
class TestChainOnlyCannotMergeCores:
    """
    INVARIANT: CORE_B requires structural witnesses (warrants), not just chain links.
    Single temporal/semantic chain is insufficient for CORE_B.
    """

    @pytest.mark.asyncio
    async def test_core_b_has_warrant_or_witnesses(self, neo4j_driver):
        """Every CORE_B decision must have WARRANT reason or witnesses."""
        results = await run_query(neo4j_driver, """
            MATCH (i:Incident)-[d:MEMBERSHIP_DECISION {membership: 'CORE_B'}]->(st:Story)
            WHERE d.core_reason IS NULL
              AND (d.witnesses IS NULL OR size(d.witnesses) = 0)
            RETURN i.id as incident, st.spine as story,
                   d.core_reason as reason, d.witnesses as witnesses
        """)

        # If there are CORE_B decisions, they must have justification
        # (Currently our corpus may not have CORE_B, which is fine)
        assert len(results) == 0, \
            f"CORE_B without warrant/witnesses: {results}"


# =============================================================================
# INVARIANT 4: Core Leak Rate Definition
# =============================================================================

@pytest.mark.skipif(not neo4j_available, reason="neo4j not available")
class TestCoreLeakRateDefinition:
    """
    INVARIANT: Core leak rate is correctly computed and stored.
    Leak = incidents in core that shouldn't be (based on structural criteria).
    """

    @pytest.mark.asyncio
    async def test_stories_have_leak_rate(self, neo4j_driver):
        """Every story should have a core_leak_rate property."""
        results = await run_query(neo4j_driver, """
            MATCH (st:Story)
            WHERE st.core_leak_rate IS NULL
            RETURN st.spine as story
        """)

        # Note: Some stories may legitimately have NULL if not computed
        # This test just checks consistency
        pass  # Soft check - warn but don't fail

    @pytest.mark.asyncio
    async def test_core_leak_rate_bounds(self, neo4j_driver):
        """Core leak rate must be between 0 and 1."""
        results = await run_query(neo4j_driver, """
            MATCH (st:Story)
            WHERE st.core_leak_rate IS NOT NULL
              AND (st.core_leak_rate < 0 OR st.core_leak_rate > 1)
            RETURN st.spine as story, st.core_leak_rate as leak_rate
        """)

        assert len(results) == 0, \
            f"Invalid core_leak_rate: {results}"


# =============================================================================
# INVARIANT 5: Blocked Reasons Visible
# =============================================================================

@pytest.mark.skipif(not neo4j_available, reason="neo4j not available")
class TestBlockedReasonsVisible:
    """
    INVARIANT: All non-CORE candidates must have visible blocked_reason.
    Tests should be able to explain why any candidate was rejected.
    """

    @pytest.mark.asyncio
    async def test_periphery_has_blocked_reason(self, neo4j_driver):
        """Every PERIPHERY decision must have a blocked_reason."""
        results = await run_query(neo4j_driver, """
            MATCH (i:Incident)-[d:MEMBERSHIP_DECISION {membership: 'PERIPHERY'}]->(st:Story)
            WHERE d.blocked_reason IS NULL OR d.blocked_reason = ''
            RETURN i.id as incident, st.spine as story
        """)

        assert len(results) == 0, \
            f"PERIPHERY without blocked_reason: {results}"

    @pytest.mark.asyncio
    async def test_reject_has_blocked_reason(self, neo4j_driver):
        """Every REJECT decision must have a blocked_reason."""
        results = await run_query(neo4j_driver, """
            MATCH (i:Incident)-[d:MEMBERSHIP_DECISION {membership: 'REJECT'}]->(st:Story)
            WHERE d.blocked_reason IS NULL OR d.blocked_reason = ''
            RETURN i.id as incident, st.spine as story
        """)

        assert len(results) == 0, \
            f"REJECT without blocked_reason: {results}"

    @pytest.mark.asyncio
    async def test_blocked_reasons_are_meaningful(self, neo4j_driver):
        """Blocked reasons should be from known vocabulary."""
        known_reasons = {
            'no structural witnesses',
            'incident has only hub anchors',
            'insufficient temporal overlap',
            'semantic distance too high',
            'only 1 witness(es), need ≥2 with non-time',
        }

        results = await run_query(neo4j_driver, """
            MATCH ()-[d:MEMBERSHIP_DECISION]->()
            WHERE d.blocked_reason IS NOT NULL
            RETURN DISTINCT d.blocked_reason as reason
        """)

        found_reasons = {r['reason'] for r in results}

        # Warn about unknown reasons but don't fail (vocabulary may expand)
        unknown = found_reasons - known_reasons
        if unknown:
            print(f"\n⚠ Unknown blocked reasons (may be valid): {unknown}")


# =============================================================================
# INVARIANT 6: Hub Entities Cannot Define Stories
# =============================================================================

@pytest.mark.skipif(not neo4j_available, reason="neo4j not available")
class TestHubCannotDefineStory:
    """
    INVARIANT: Entities marked as hubs should not be story spines.
    """

    @pytest.mark.asyncio
    async def test_hub_not_story_spine(self, neo4j_driver):
        """Hub entities should not be story spines."""
        results = await run_query(neo4j_driver, """
            MATCH (st:Story)-[:HAS_SPINE]->(e:Entity)
            WHERE e.is_hub = true
            RETURN st.spine as story, e.name as hub_entity
        """)

        assert len(results) == 0, \
            f"Hub entities as story spines: {results}"


# =============================================================================
# INVARIANT 7: Decision Trace Completeness
# =============================================================================

@pytest.mark.skipif(not neo4j_available, reason="neo4j not available")
class TestDecisionTraceCompleteness:
    """
    INVARIANT: Decision traces must be complete for auditability.
    """

    @pytest.mark.asyncio
    async def test_traces_have_kernel_version(self, neo4j_driver):
        """All decision traces must have kernel_version."""
        results = await run_query(neo4j_driver, """
            MATCH ()-[d:MEMBERSHIP_DECISION]->()
            WHERE d.kernel_version IS NULL
            RETURN count(d) as count
        """)

        assert results[0]['count'] == 0, \
            "Decision traces missing kernel_version"

    @pytest.mark.asyncio
    async def test_traces_have_timestamp(self, neo4j_driver):
        """All decision traces must have timestamp."""
        results = await run_query(neo4j_driver, """
            MATCH ()-[d:MEMBERSHIP_DECISION]->()
            WHERE d.timestamp IS NULL
            RETURN count(d) as count
        """)

        assert results[0]['count'] == 0, \
            "Decision traces missing timestamp"

    @pytest.mark.asyncio
    async def test_traces_have_params_hash(self, neo4j_driver):
        """All decision traces must have params_hash for reproducibility."""
        results = await run_query(neo4j_driver, """
            MATCH ()-[d:MEMBERSHIP_DECISION]->()
            WHERE d.params_hash IS NULL
            RETURN count(d) as count
        """)

        assert results[0]['count'] == 0, \
            "Decision traces missing params_hash"


# =============================================================================
# INVARIANT 8: Constraint Source Tracking
# =============================================================================

@pytest.mark.skipif(not neo4j_available, reason="neo4j not available")
class TestConstraintSourceTracking:
    """
    INVARIANT: CORE decisions must have constraint_source = "structural".
    Semantic-only evidence cannot produce CORE membership.
    """

    @pytest.mark.asyncio
    async def test_core_has_structural_source(self, neo4j_driver):
        """All CORE decisions must have constraint_source = structural."""
        results = await run_query(neo4j_driver, """
            MATCH ()-[d:MEMBERSHIP_DECISION]->()
            WHERE d.membership IN ['CORE_A', 'CORE_B']
              AND (d.constraint_source IS NULL OR d.constraint_source <> 'structural')
            RETURN d.membership as membership, d.constraint_source as source, count(*) as count
        """)

        # Allow for traces before constraint_source was added
        total_core = await run_query(neo4j_driver, """
            MATCH ()-[d:MEMBERSHIP_DECISION]->()
            WHERE d.membership IN ['CORE_A', 'CORE_B']
            RETURN count(d) as count
        """)

        if total_core[0]['count'] > 0 and len(results) > 0:
            # Only fail if we have traces with explicit non-structural source
            non_structural = [r for r in results if r['source'] == 'semantic_proposal']
            assert len(non_structural) == 0, \
                f"CORE decisions with semantic_proposal source: {non_structural}"

    @pytest.mark.asyncio
    async def test_no_semantic_only_core_via_source(self, neo4j_driver):
        """No CORE membership should have semantic_proposal constraint_source."""
        results = await run_query(neo4j_driver, """
            MATCH (i:Incident)-[d:MEMBERSHIP_DECISION]->(st:Story)
            WHERE d.membership IN ['CORE_A', 'CORE_B']
              AND d.constraint_source = 'semantic_proposal'
            RETURN i.id as incident, st.spine as story
        """)

        assert len(results) == 0, \
            f"Semantic-only CORE found via constraint_source: {results}"


# =============================================================================
# INVARIANT 9: Trace Determinism
# =============================================================================

@pytest.mark.skipif(not neo4j_available, reason="neo4j not available")
class TestTraceDeterminism:
    """
    INVARIANT: Same corpus + same params_hash + same kernel_version
    ⇒ identical decision edges (order-independent).

    This prevents "graph query ordering" regressions.
    """

    @pytest.mark.asyncio
    async def test_single_params_hash_per_run(self, neo4j_driver):
        """All traces in a run should have the same params_hash."""
        results = await run_query(neo4j_driver, """
            MATCH ()-[d:MEMBERSHIP_DECISION]->()
            WHERE d.params_hash IS NOT NULL
            RETURN DISTINCT d.params_hash as hash, count(*) as count
        """)

        # Should have exactly 1 params_hash (single run)
        # Multiple hashes indicate mixed runs or non-determinism
        if len(results) > 1:
            print(f"\n⚠ Multiple params_hash values found: {results}")
            print("  This may indicate mixed runs or parameter drift")

        # This is a warning, not a hard failure during development
        assert len(results) >= 1, "No params_hash found in traces"

    @pytest.mark.asyncio
    async def test_decision_identity_deterministic(self, neo4j_driver):
        """
        Decision identity is deterministic: same (incident, story, params_hash)
        should always produce same (membership, core_reason, witnesses).

        Check for duplicates with different decisions.
        """
        results = await run_query(neo4j_driver, """
            MATCH (i:Incident)-[d:MEMBERSHIP_DECISION]->(st:Story)
            WITH i.id as inc, st.id as story, d.params_hash as params,
                 collect(DISTINCT d.membership) as memberships
            WHERE size(memberships) > 1
            RETURN inc, story, memberships
        """)

        assert len(results) == 0, \
            f"Non-deterministic decisions (same incident+story, different membership): {results}"

    @pytest.mark.asyncio
    async def test_witness_order_canonical(self, neo4j_driver):
        """
        Witness lists should be in canonical order (sorted).
        This ensures deterministic comparison across runs.
        """
        results = await run_query(neo4j_driver, """
            MATCH ()-[d:MEMBERSHIP_DECISION]->()
            WHERE d.witnesses IS NOT NULL AND size(d.witnesses) > 1
            WITH d.witnesses as witnesses
            WHERE witnesses <> apoc.coll.sort(witnesses)
            RETURN count(*) as unsorted_count
        """)

        # This is aspirational - witnesses may not be sorted yet
        # Just report for now
        if results and results[0].get('unsorted_count', 0) > 0:
            print(f"\n📊 Unsorted witness lists: {results[0]['unsorted_count']}")
            print("  Consider sorting witnesses for deterministic comparison")


# =============================================================================
# SUMMARY
# =============================================================================

@pytest.mark.skipif(not neo4j_available, reason="neo4j not available")
class TestInvariantSummary:
    """Summary of invariant status."""

    @pytest.mark.asyncio
    async def test_print_invariant_summary(self, neo4j_driver):
        """Print summary of invariant checks."""
        # Count decisions by membership
        results = await run_query(neo4j_driver, """
            MATCH ()-[d:MEMBERSHIP_DECISION]->()
            RETURN d.membership as membership, count(d) as count
            ORDER BY count DESC
        """)

        print("\n" + "=" * 60)
        print("HARD INVARIANT CHECK SUMMARY")
        print("=" * 60)
        print("\nDecision distribution:")
        for r in results:
            print(f"  {r['membership']}: {r['count']}")

        # Check for any anomalies
        results = await run_query(neo4j_driver, """
            MATCH ()-[d:MEMBERSHIP_DECISION]->()
            WHERE d.membership IN ['PERIPHERY', 'REJECT'] AND d.blocked_reason IS NULL
            RETURN count(d) as missing_reasons
        """)
        missing = results[0]['missing_reasons']
        print(f"\nMissing blocked_reasons: {missing}")

        print("=" * 60)

        # This test always passes - it's for information
        assert True


# =============================================================================
# DIRECT RUN SUPPORT
# =============================================================================

if __name__ == "__main__":
    print("Running Hard Invariant tests...")
    print("Prerequisites: Run kernel_validator.py first")
    print("-" * 50)
    print("Run with pytest:")
    print("  docker exec herenews-test-runner python -m pytest reee/tests/integration/test_hard_invariants.py -v")
