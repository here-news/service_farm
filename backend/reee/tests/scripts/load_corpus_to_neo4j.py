#!/usr/bin/env python3
"""
Load Macro Corpus into Test Neo4j
=================================

Loads the generated macro corpus into the test Neo4j instance for
persistent inspection and topology validation.

Usage:
    python -m reee.tests.scripts.load_corpus_to_neo4j

    # Or from backend directory:
    python reee/tests/scripts/load_corpus_to_neo4j.py
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from neo4j import AsyncGraphDatabase

from reee.tests.golden_macro.corpus_generator import MacroCorpusGenerator
from reee.types import Event, Surface


# Test Neo4j configuration
# Uses TEST_NEO4J_* env vars, falls back to NEO4J_* (for test-runner container)
TEST_NEO4J_URI = os.environ.get("TEST_NEO4J_URI") or os.environ.get("NEO4J_URI", "bolt://localhost:7688")
TEST_NEO4J_USER = os.environ.get("TEST_NEO4J_USER") or os.environ.get("NEO4J_USER", "neo4j")
TEST_NEO4J_PASSWORD = os.environ.get("TEST_NEO4J_PASSWORD") or os.environ.get("NEO4J_PASSWORD", "test_password")

# SAFETY: Refuse to run against production URIs
PRODUCTION_PATTERNS = ["200:", "remote", "prod", "live"]
if any(p in TEST_NEO4J_URI.lower() for p in PRODUCTION_PATTERNS):
    raise RuntimeError(f"SAFETY: Refusing to run against production URI: {TEST_NEO4J_URI}")


async def clear_database(driver):
    """Clear all nodes and relationships."""
    async with driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    print("✓ Cleared existing data")


async def create_constraints(driver):
    """Create uniqueness constraints."""
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Incident) REQUIRE i.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Surface) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (st:Story) REQUIRE st.id IS UNIQUE",
    ]
    async with driver.session() as session:
        for constraint in constraints:
            try:
                await session.run(constraint)
            except Exception as e:
                # Constraint may already exist
                pass
    print("✓ Created constraints")


async def load_entities(driver, corpus):
    """Load entities into Neo4j."""
    async with driver.session() as session:
        for entity in corpus.entities:
            await session.run("""
                MERGE (e:Entity {name: $name})
                SET e.type = $type,
                    e.role = $role,
                    e.archetype = $archetype
            """, {
                'name': entity.name,
                'type': entity.entity_type,
                'role': entity.role,
                'archetype': entity.archetype,
            })
    print(f"✓ Loaded {len(corpus.entities)} entities")


async def load_incidents(driver, corpus):
    """Load incidents into Neo4j with entity relationships."""
    async with driver.session() as session:
        for incident in corpus.incidents:
            # Create incident
            await session.run("""
                MERGE (i:Incident {id: $id})
                SET i.description = $description,
                    i.archetype = $archetype,
                    i.time_start = $time_start,
                    i.time_end = $time_end
            """, {
                'id': incident.id,
                'description': incident.description,
                'archetype': incident.archetype,
                'time_start': incident.time_start,
                'time_end': incident.time_end,
            })

            # Link to anchor entities
            for entity_name in incident.anchor_entities:
                await session.run("""
                    MATCH (i:Incident {id: $inc_id})
                    MATCH (e:Entity {name: $entity_name})
                    MERGE (i)-[:HAS_ANCHOR]->(e)
                """, {
                    'inc_id': incident.id,
                    'entity_name': entity_name,
                })

            # Link to companion entities
            for entity_name in incident.companion_entities:
                await session.run("""
                    MATCH (i:Incident {id: $inc_id})
                    MATCH (e:Entity {name: $entity_name})
                    MERGE (i)-[:HAS_COMPANION]->(e)
                """, {
                    'inc_id': incident.id,
                    'entity_name': entity_name,
                })

    print(f"✓ Loaded {len(corpus.incidents)} incidents")


async def load_claims(driver, corpus):
    """Load claims into Neo4j."""
    async with driver.session() as session:
        for claim in corpus.claims:
            await session.run("""
                MERGE (c:Claim {id: $id})
                SET c.text = $text,
                    c.publisher = $publisher,
                    c.question_key = $question_key,
                    c.scope_id = $scope_id,
                    c.archetype = $archetype,
                    c.anchor_entities = $anchor_entities
            """, {
                'id': claim.id,
                'text': claim.text,
                'publisher': claim.publisher,
                'question_key': claim.question_key,
                'scope_id': claim.scope_id,
                'archetype': claim.archetype,
                'anchor_entities': claim.anchor_entities,
            })

            # Link to anchor entities
            for entity_name in claim.anchor_entities:
                await session.run("""
                    MATCH (c:Claim {id: $claim_id})
                    MATCH (e:Entity {name: $entity_name})
                    MERGE (c)-[:MENTIONS]->(e)
                """, {
                    'claim_id': claim.id,
                    'entity_name': entity_name,
                })

    print(f"✓ Loaded {len(corpus.claims)} claims")


async def create_surfaces(driver, corpus):
    """Create surfaces from claims grouped by (scope_id, question_key)."""
    import hashlib
    from collections import defaultdict

    # Group claims by (scope_id, question_key)
    by_scope_qk = defaultdict(list)
    for claim in corpus.claims:
        key = (claim.scope_id, claim.question_key)
        by_scope_qk[key].append(claim)

    async with driver.session() as session:
        for (scope_id, qk), claims in by_scope_qk.items():
            surface_id = f"surf_{hashlib.sha256(f'{scope_id}:{qk}'.encode()).hexdigest()[:12]}"

            # Create surface
            await session.run("""
                MERGE (s:Surface {id: $id})
                SET s.question_key = $question_key,
                    s.scope_id = $scope_id,
                    s.claim_count = $claim_count
            """, {
                'id': surface_id,
                'question_key': qk,
                'scope_id': scope_id,
                'claim_count': len(claims),
            })

            # Link claims to surface
            for claim in claims:
                await session.run("""
                    MATCH (s:Surface {id: $surface_id})
                    MATCH (c:Claim {id: $claim_id})
                    MERGE (c)-[:PART_OF]->(s)
                """, {
                    'surface_id': surface_id,
                    'claim_id': claim.id,
                })

    print(f"✓ Created {len(by_scope_qk)} surfaces")


async def run_story_builder_and_persist(driver, corpus):
    """Run StoryBuilder and persist stories to Neo4j."""
    from reee.builders.story_builder import StoryBuilder

    # Convert corpus to kernel inputs
    incidents = {}
    for inc in corpus.incidents:
        time_start = None
        time_end = None
        if inc.time_start:
            time_start = datetime.fromisoformat(inc.time_start.replace('Z', '+00:00'))
        if inc.time_end:
            time_end = datetime.fromisoformat(inc.time_end.replace('Z', '+00:00'))

        event = Event(
            id=inc.id,
            anchor_entities=set(inc.anchor_entities),
            entities=set(inc.anchor_entities + inc.companion_entities),
            time_window=(time_start, time_end),
            surface_ids=set(),
            canonical_title=inc.description,
        )
        incidents[event.id] = event

    # Build stories
    builder = StoryBuilder(
        hub_fraction_threshold=0.20,
        hub_min_incidents=5,
        min_incidents_for_story=2,
        mode_gap_days=30,
    )

    result = builder.build_from_incidents(incidents, {})
    print(f"✓ Built {len(result.stories)} stories")

    # Persist stories
    async with driver.session() as session:
        for story_id, story in result.stories.items():
            # Create story node
            await session.run("""
                MERGE (st:Story {id: $id})
                SET st.spine = $spine,
                    st.core_a_count = $core_a_count,
                    st.core_b_count = $core_b_count,
                    st.periphery_count = $periphery_count,
                    st.core_leak_rate = $core_leak_rate
            """, {
                'id': story_id,
                'spine': story.spine,
                'core_a_count': len(story.core_a_ids),
                'core_b_count': len(story.core_b_ids),
                'periphery_count': len(story.periphery_incident_ids),
                'core_leak_rate': story.core_leak_rate,
            })

            # Link spine entity
            await session.run("""
                MATCH (st:Story {id: $story_id})
                MATCH (e:Entity {name: $spine})
                MERGE (st)-[:HAS_SPINE]->(e)
            """, {
                'story_id': story_id,
                'spine': story.spine,
            })

            # Link Core-A incidents
            for inc_id in story.core_a_ids:
                await session.run("""
                    MATCH (st:Story {id: $story_id})
                    MATCH (i:Incident {id: $inc_id})
                    MERGE (i)-[:CORE_A_OF]->(st)
                """, {
                    'story_id': story_id,
                    'inc_id': inc_id,
                })

            # Link Core-B incidents
            for inc_id in story.core_b_ids:
                await session.run("""
                    MATCH (st:Story {id: $story_id})
                    MATCH (i:Incident {id: $inc_id})
                    MERGE (i)-[:CORE_B_OF]->(st)
                """, {
                    'story_id': story_id,
                    'inc_id': inc_id,
                })

            # Link periphery incidents
            for inc_id in story.periphery_incident_ids:
                await session.run("""
                    MATCH (st:Story {id: $story_id})
                    MATCH (i:Incident {id: $inc_id})
                    MERGE (i)-[:PERIPHERY_OF]->(st)
                """, {
                    'story_id': story_id,
                    'inc_id': inc_id,
                })

    # Mark hub entities
    hub_entities = {e for e, s in result.spines.items() if s.is_hub}
    async with driver.session() as session:
        for entity in hub_entities:
            await session.run("""
                MATCH (e:Entity {name: $name})
                SET e.is_hub = true
            """, {'name': entity})

    print(f"✓ Persisted {len(result.stories)} stories to Neo4j")
    print(f"✓ Marked {len(hub_entities)} hub entities")

    return result


async def print_summary(driver):
    """Print summary of loaded data."""
    async with driver.session() as session:
        # Count nodes
        result = await session.run("""
            MATCH (n)
            RETURN labels(n)[0] as type, count(*) as count
            ORDER BY count DESC
        """)
        records = await result.data()

        print("\n" + "=" * 50)
        print("CORPUS LOADED INTO TEST NEO4J")
        print("=" * 50)
        print(f"URI: {TEST_NEO4J_URI}")
        print("\nNode counts:")
        for r in records:
            print(f"  {r['type']}: {r['count']}")

        # Count relationships
        result = await session.run("""
            MATCH ()-[r]->()
            RETURN type(r) as type, count(*) as count
            ORDER BY count DESC
        """)
        records = await result.data()
        print("\nRelationship counts:")
        for r in records:
            print(f"  {r['type']}: {r['count']}")

        # Story summary
        result = await session.run("""
            MATCH (st:Story)
            RETURN st.spine as spine, st.core_a_count as core_a,
                   st.core_b_count as core_b, st.periphery_count as periphery
            ORDER BY st.core_a_count DESC
            LIMIT 10
        """)
        records = await result.data()
        print("\nTop 10 stories by Core-A size:")
        for r in records:
            print(f"  {r['spine']}: Core-A={r['core_a']}, Core-B={r['core_b']}, Periphery={r['periphery']}")

        print("\n" + "=" * 50)


async def main():
    """Main entry point."""
    print(f"Connecting to test Neo4j at {TEST_NEO4J_URI}...")

    driver = AsyncGraphDatabase.driver(
        TEST_NEO4J_URI,
        auth=(TEST_NEO4J_USER, TEST_NEO4J_PASSWORD),
    )

    try:
        # Verify connection
        async with driver.session() as session:
            result = await session.run("RETURN 1 as n")
            await result.single()
        print("✓ Connected to Neo4j")

        # Generate corpus
        print("\nGenerating macro corpus (seed=42)...")
        generator = MacroCorpusGenerator(seed=42)
        corpus = generator.generate()
        print(f"✓ Generated {corpus.manifest.total_claims} claims, {corpus.manifest.total_incidents} incidents")

        # Clear and load
        await clear_database(driver)
        await create_constraints(driver)
        await load_entities(driver, corpus)
        await load_incidents(driver, corpus)
        await load_claims(driver, corpus)
        await create_surfaces(driver, corpus)

        # Run StoryBuilder and persist
        await run_story_builder_and_persist(driver, corpus)

        # Print summary
        await print_summary(driver)

    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
