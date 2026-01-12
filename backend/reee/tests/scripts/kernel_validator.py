#!/usr/bin/env python3
"""
Kernel Validator - Real Kernel Path with Decision Traces
=========================================================

Runs the REAL kernel formation logic and persists decision traces
to test Neo4j for queryable validation.

SAFETY: This script ONLY connects to test Neo4j (localhost:7688).
        It will NOT touch production databases.

Usage (from service_farm directory):
    cd backend
    python -m reee.tests.scripts.kernel_validator

Acceptance Checkpoint:
    After running, you should be able to query Neo4j:
    "Why is incident X in story Y?" → follow MEMBERSHIP_DECISION edges

Decision Trace Schema (per candidate):
    (Incident)-[:MEMBERSHIP_DECISION {
        story_id,
        membership,      # CORE_A / CORE_B / PERIPHERY / REJECT
        core_reason,     # ANCHOR / WARRANT / null
        link_type,       # MEMBER / RELATED / null
        witnesses,       # [constraint_ids]
        blocked_reason,  # string or null
        timestamp,
        kernel_version,
        params_hash
    }]->(Story)
"""

import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
from dataclasses import dataclass, asdict

# Ensure we're using the local backend code
BACKEND_PATH = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(BACKEND_PATH))

# Test Neo4j configuration
# Uses TEST_NEO4J_* env vars, falls back to NEO4J_* (for test-runner container)
TEST_NEO4J_URI = os.environ.get("TEST_NEO4J_URI") or os.environ.get("NEO4J_URI", "bolt://localhost:7688")
TEST_NEO4J_USER = os.environ.get("TEST_NEO4J_USER") or os.environ.get("NEO4J_USER", "neo4j")
TEST_NEO4J_PASSWORD = os.environ.get("TEST_NEO4J_PASSWORD") or os.environ.get("NEO4J_PASSWORD", "test_password")

# SAFETY CHECK: Refuse to run against production URIs
PRODUCTION_PATTERNS = ["200:", "remote", "prod", "live"]
if any(p in TEST_NEO4J_URI.lower() for p in PRODUCTION_PATTERNS):
    raise RuntimeError(f"SAFETY: Refusing to run against production URI: {TEST_NEO4J_URI}")

# Kernel version for traces
KERNEL_VERSION = "story_builder_v1"


@dataclass
class KernelParams:
    """Kernel parameters for reproducibility."""
    hub_fraction_threshold: float = 0.20
    hub_min_incidents: int = 5
    min_incidents_for_story: int = 2
    mode_gap_days: int = 30

    def hash(self) -> str:
        """Deterministic hash of parameters."""
        params_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(params_str.encode()).hexdigest()[:12]


@dataclass
class DecisionTrace:
    """Full trace of a membership decision."""
    candidate_id: str
    target_id: str  # story_id
    membership: str  # CORE_A, CORE_B, PERIPHERY, REJECT
    core_reason: Optional[str]  # ANCHOR, WARRANT, null
    link_type: Optional[str]  # MEMBER, RELATED, null
    witnesses: List[str]  # constraint IDs
    blocked_reason: Optional[str]
    constraint_source: Optional[str]  # "structural" or "semantic_proposal"
    timestamp: str
    kernel_version: str
    params_hash: str


class KernelValidator:
    """
    Validates kernel by running real formation logic and persisting traces.
    """

    def __init__(self, params: KernelParams = None):
        self.params = params or KernelParams()
        self.traces: List[DecisionTrace] = []
        self.timestamp = datetime.utcnow().isoformat() + "Z"

    async def connect_test_neo4j(self):
        """Connect to test Neo4j ONLY."""
        from neo4j import AsyncGraphDatabase

        # SAFETY: Hardcoded test URI
        self.driver = AsyncGraphDatabase.driver(
            TEST_NEO4J_URI,
            auth=(TEST_NEO4J_USER, TEST_NEO4J_PASSWORD),
        )

        # Verify connection
        async with self.driver.session() as session:
            result = await session.run("RETURN 1 as n")
            await result.single()
        print(f"✓ Connected to TEST Neo4j at {TEST_NEO4J_URI}")

    async def close(self):
        """Close Neo4j connection."""
        if hasattr(self, 'driver'):
            await self.driver.close()

    async def clear_traces(self):
        """Clear existing decision traces (but keep corpus data)."""
        async with self.driver.session() as session:
            # Remove trace relationships only
            await session.run("""
                MATCH ()-[r:MEMBERSHIP_DECISION]->()
                DELETE r
            """)
            # Remove any DecisionTrace nodes
            await session.run("""
                MATCH (t:DecisionTrace)
                DETACH DELETE t
            """)
        print("✓ Cleared existing decision traces")

    def run_kernel(self, incidents: Dict[str, Any], surfaces: Dict[str, Any]) -> Any:
        """Run the real kernel and capture decision traces."""
        from reee.builders.story_builder import StoryBuilder
        from reee.types import Event, Surface

        # Convert to kernel types if needed
        kernel_incidents = {}
        for inc_id, inc in incidents.items():
            if isinstance(inc, Event):
                kernel_incidents[inc_id] = inc
            else:
                kernel_incidents[inc_id] = Event(
                    id=inc_id,
                    anchor_entities=set(inc.get('anchor_entities', [])),
                    entities=set(inc.get('anchor_entities', []) + inc.get('companion_entities', [])),
                    time_window=self._parse_time_window(inc),
                    surface_ids=set(inc.get('surface_ids', [])),
                    canonical_title=inc.get('description', ''),
                )

        kernel_surfaces = {}
        for surf_id, surf in surfaces.items():
            if isinstance(surf, Surface):
                kernel_surfaces[surf_id] = surf
            else:
                kernel_surfaces[surf_id] = Surface(
                    id=surf_id,
                    question_key=surf.get('question_key', 'unknown'),
                    claim_ids=set(surf.get('claim_ids', [])),
                    formation_method='kernel_validator',
                    centroid=None,
                )

        # Run StoryBuilder with real logic
        builder = StoryBuilder(
            hub_fraction_threshold=self.params.hub_fraction_threshold,
            hub_min_incidents=self.params.hub_min_incidents,
            min_incidents_for_story=self.params.min_incidents_for_story,
            mode_gap_days=self.params.mode_gap_days,
        )

        result = builder.build_from_incidents(kernel_incidents, kernel_surfaces)

        # Extract decision traces from membrane_decisions
        for story_id, story in result.stories.items():
            if story.membrane_decisions:
                for inc_id, decision in story.membrane_decisions.items():
                    # Determine membership category
                    if inc_id in story.core_a_ids:
                        membership = "CORE_A"
                    elif inc_id in story.core_b_ids:
                        membership = "CORE_B"
                    elif inc_id in story.periphery_incident_ids:
                        membership = "PERIPHERY"
                    else:
                        membership = "REJECT"

                    # Extract witnesses as constraint IDs
                    witnesses = []
                    if decision.witnesses:
                        for w in decision.witnesses:
                            if hasattr(w, 'id'):
                                witnesses.append(w.id)
                            elif isinstance(w, str):
                                witnesses.append(w)
                            else:
                                witnesses.append(str(w))

                    trace = DecisionTrace(
                        candidate_id=inc_id,
                        target_id=story_id,
                        membership=membership,
                        core_reason=decision.core_reason.name if decision.core_reason else None,
                        link_type=decision.link_type.name if decision.link_type else None,
                        witnesses=witnesses,
                        blocked_reason=decision.blocked_reason,
                        constraint_source=decision.constraint_source,
                        timestamp=self.timestamp,
                        kernel_version=KERNEL_VERSION,
                        params_hash=self.params.hash(),
                    )
                    self.traces.append(trace)

        print(f"✓ Ran kernel: {len(result.stories)} stories, {len(self.traces)} decision traces")
        return result

    def _parse_time_window(self, inc: dict):
        """Parse time window from incident dict."""
        time_start = None
        time_end = None
        if inc.get('time_start'):
            ts = inc['time_start'].replace('Z', '+00:00')
            time_start = datetime.fromisoformat(ts)
        if inc.get('time_end'):
            te = inc['time_end'].replace('Z', '+00:00')
            time_end = datetime.fromisoformat(te)
        return (time_start, time_end)

    async def persist_traces(self):
        """Persist decision traces to Neo4j."""
        async with self.driver.session() as session:
            for trace in self.traces:
                # Create MEMBERSHIP_DECISION relationship between Incident and Story
                await session.run("""
                    MATCH (i:Incident {id: $candidate_id})
                    MATCH (st:Story {id: $target_id})
                    MERGE (i)-[r:MEMBERSHIP_DECISION]->(st)
                    SET r.membership = $membership,
                        r.core_reason = $core_reason,
                        r.link_type = $link_type,
                        r.witnesses = $witnesses,
                        r.blocked_reason = $blocked_reason,
                        r.constraint_source = $constraint_source,
                        r.timestamp = $timestamp,
                        r.kernel_version = $kernel_version,
                        r.params_hash = $params_hash
                """, {
                    'candidate_id': trace.candidate_id,
                    'target_id': trace.target_id,
                    'membership': trace.membership,
                    'core_reason': trace.core_reason,
                    'link_type': trace.link_type,
                    'witnesses': trace.witnesses,
                    'blocked_reason': trace.blocked_reason,
                    'constraint_source': trace.constraint_source,
                    'timestamp': trace.timestamp,
                    'kernel_version': trace.kernel_version,
                    'params_hash': trace.params_hash,
                })

        print(f"✓ Persisted {len(self.traces)} decision traces to Neo4j")

    async def verify_traces_queryable(self):
        """Verify traces can answer 'why is incident X in story Y?'"""
        async with self.driver.session() as session:
            # Sample query: Get decision trace for a random incident
            result = await session.run("""
                MATCH (i:Incident)-[d:MEMBERSHIP_DECISION]->(st:Story)
                RETURN i.id as incident, st.spine as story,
                       d.membership as membership, d.core_reason as reason,
                       d.blocked_reason as blocked, d.witnesses as witnesses
                LIMIT 5
            """)
            records = await result.data()

            if records:
                print("\n✓ Traces are queryable. Sample:")
                for r in records:
                    print(f"  {r['incident']} → {r['story']}")
                    print(f"    membership={r['membership']}, reason={r['reason']}")
                    if r['blocked']:
                        print(f"    blocked: {r['blocked']}")
                    if r['witnesses']:
                        print(f"    witnesses: {r['witnesses']}")
            else:
                print("⚠ No traces found - check if corpus is loaded")

    async def print_summary(self):
        """Print validation summary."""
        async with self.driver.session() as session:
            # Count by membership
            result = await session.run("""
                MATCH (i:Incident)-[d:MEMBERSHIP_DECISION]->(st:Story)
                RETURN d.membership as membership, count(*) as count
                ORDER BY count DESC
            """)
            records = await result.data()

            print("\n" + "=" * 60)
            print("KERNEL VALIDATION SUMMARY")
            print("=" * 60)
            print(f"Kernel version: {KERNEL_VERSION}")
            print(f"Params hash: {self.params.hash()}")
            print(f"\nDecision counts by membership:")
            for r in records:
                print(f"  {r['membership']}: {r['count']}")

            # Count blocked reasons
            result = await session.run("""
                MATCH (i:Incident)-[d:MEMBERSHIP_DECISION]->(st:Story)
                WHERE d.blocked_reason IS NOT NULL
                RETURN d.blocked_reason as reason, count(*) as count
                ORDER BY count DESC
                LIMIT 10
            """)
            records = await result.data()

            if records:
                print(f"\nBlocked reasons:")
                for r in records:
                    print(f"  {r['reason']}: {r['count']}")

            print("=" * 60)


async def main():
    """Main entry point."""
    print("=" * 60)
    print("KERNEL VALIDATOR")
    print("=" * 60)
    print(f"Target: {TEST_NEO4J_URI} (TEST ONLY)")
    print()

    validator = KernelValidator()

    try:
        await validator.connect_test_neo4j()

        # Check if corpus is loaded
        async with validator.driver.session() as session:
            result = await session.run("MATCH (i:Incident) RETURN count(i) as count")
            record = await result.single()
            incident_count = record['count']

            if incident_count == 0:
                print("\n⚠ No incidents in test Neo4j!")
                print("Run load_corpus_to_neo4j.py first:")
                print("  docker exec herenews-app python /app/reee/tests/scripts/load_corpus_to_neo4j.py")
                return

            print(f"✓ Found {incident_count} incidents in corpus")

        # Load incidents and surfaces from Neo4j
        async with validator.driver.session() as session:
            # Load incidents
            result = await session.run("""
                MATCH (i:Incident)
                OPTIONAL MATCH (i)-[:HAS_ANCHOR]->(e:Entity)
                RETURN i.id as id, i.description as description,
                       i.time_start as time_start, i.time_end as time_end,
                       collect(e.name) as anchors
            """)
            incident_records = await result.data()

            # Load surfaces
            result = await session.run("""
                MATCH (s:Surface)
                OPTIONAL MATCH (c:Claim)-[:PART_OF]->(s)
                RETURN s.id as id, s.question_key as question_key,
                       collect(c.id) as claim_ids
            """)
            surface_records = await result.data()

        # Convert to dicts
        incidents = {}
        for r in incident_records:
            incidents[r['id']] = {
                'anchor_entities': r['anchors'] or [],
                'companion_entities': [],
                'time_start': r['time_start'],
                'time_end': r['time_end'],
                'description': r['description'] or '',
            }

        surfaces = {}
        for r in surface_records:
            surfaces[r['id']] = {
                'question_key': r['question_key'] or 'unknown',
                'claim_ids': r['claim_ids'] or [],
            }

        print(f"✓ Loaded {len(incidents)} incidents, {len(surfaces)} surfaces")

        # Clear old traces and run kernel
        await validator.clear_traces()
        validator.run_kernel(incidents, surfaces)

        # Persist traces
        await validator.persist_traces()

        # Verify queryable
        await validator.verify_traces_queryable()

        # Print summary
        await validator.print_summary()

    finally:
        await validator.close()


if __name__ == "__main__":
    asyncio.run(main())
