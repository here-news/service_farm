"""
Test Neo4j Manager
==================

Manages test Neo4j lifecycle for kernel validation tests.

Features:
- Fresh database setup per test
- Deterministic fixture loading
- Ordered snapshots for comparison
- Hash-based stable IDs
"""

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
import logging

try:
    from neo4j import AsyncGraphDatabase
except ImportError:
    AsyncGraphDatabase = None  # Will fail at runtime if actually used

logger = logging.getLogger(__name__)


@dataclass
class TestNeo4jConfig:
    """Configuration for test Neo4j connection."""
    uri: str = "bolt://localhost:7688"  # Test port
    user: str = "neo4j"
    password: str = "test_password"
    database: str = "neo4j"


class TestNeo4jManager:
    """
    Manages test Neo4j lifecycle for kernel validation.

    Usage:
        async with TestNeo4jManager() as neo4j:
            await neo4j.setup_fresh()
            await neo4j.load_fixture("corpus.json")
            # run tests...
    """

    def __init__(self, config: Optional[TestNeo4jConfig] = None):
        self.config = config or TestNeo4jConfig()
        self.driver = None
        self._connected = False

    async def connect(self):
        """Connect to test Neo4j."""
        if self._connected:
            return

        self.driver = AsyncGraphDatabase.driver(
            self.config.uri,
            auth=(self.config.user, self.config.password),
        )
        self._connected = True
        logger.info(f"Connected to test Neo4j at {self.config.uri}")

    async def close(self):
        """Close connection."""
        if self.driver:
            await self.driver.close()
            self._connected = False
            logger.info("Closed test Neo4j connection")

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # =========================================================================
    # DATABASE SETUP
    # =========================================================================

    async def setup_fresh(self):
        """Clear database and create indexes."""
        await self.clear_all()
        await self.create_indexes()
        logger.info("Test database initialized fresh")

    async def clear_all(self):
        """Delete all nodes and relationships."""
        async with self.driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
        logger.info("Cleared all nodes and relationships")

    async def create_indexes(self):
        """Create indexes for efficient lookups."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (c:Claim) ON (c.id)",
            "CREATE INDEX IF NOT EXISTS FOR (s:Surface) ON (s.id)",
            "CREATE INDEX IF NOT EXISTS FOR (i:Incident) ON (i.id)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Case) ON (c.id)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.canonical_name)",
        ]
        async with self.driver.session() as session:
            for idx in indexes:
                await session.run(idx)
        logger.info(f"Created {len(indexes)} indexes")

    # =========================================================================
    # FIXTURE LOADING (DETERMINISTIC)
    # =========================================================================

    async def load_fixture(self, fixture_path: str):
        """
        Load fixture file into Neo4j.

        Supports JSON format with claims, surfaces, incidents, cases.
        All IDs are hash-based for stability.
        """
        with open(fixture_path, 'r') as f:
            fixture = json.load(f)

        # Load in deterministic order
        if 'entities' in fixture:
            await self._load_entities(fixture['entities'])

        if 'claims' in fixture:
            await self._load_claims(fixture['claims'])

        if 'surfaces' in fixture:
            await self._load_surfaces(fixture['surfaces'])

        if 'incidents' in fixture:
            await self._load_incidents(fixture['incidents'])

        if 'cases' in fixture:
            await self._load_cases(fixture['cases'])

        logger.info(f"Loaded fixture from {fixture_path}")

    async def _load_entities(self, entities: List[Dict]):
        """Load entities in sorted order."""
        sorted_entities = sorted(entities, key=lambda e: e.get('id', ''))

        async with self.driver.session() as session:
            for entity in sorted_entities:
                await session.run('''
                    MERGE (e:Entity {canonical_name: $name})
                    SET e.id = $id,
                        e.entity_type = $type,
                        e.role = $role
                ''', {
                    'id': entity.get('id', self._hash_id('entity', entity['name'])),
                    'name': entity['name'],
                    'type': entity.get('type', 'unknown'),
                    'role': entity.get('role', 'unknown'),
                })

    async def _load_claims(self, claims: List[Dict]):
        """Load claims in sorted order."""
        sorted_claims = sorted(claims, key=lambda c: c.get('id', ''))

        async with self.driver.session() as session:
            for claim in sorted_claims:
                await session.run('''
                    CREATE (c:Claim {
                        id: $id,
                        text: $text,
                        publisher: $publisher,
                        reported_time: $reported_time,
                        event_time: $event_time,
                        question_key: $question_key,
                        anchor_entities: $anchors
                    })
                ''', {
                    'id': claim.get('id', self._hash_id('claim', claim['text'])),
                    'text': claim.get('text', ''),
                    'publisher': claim.get('publisher', ''),
                    'reported_time': claim.get('reported_time'),
                    'event_time': claim.get('event_time'),
                    'question_key': claim.get('question_key', ''),
                    'anchors': claim.get('anchor_entities', []),
                })

    async def _load_surfaces(self, surfaces: List[Dict]):
        """Load surfaces in sorted order."""
        sorted_surfaces = sorted(surfaces, key=lambda s: s.get('id', ''))

        async with self.driver.session() as session:
            for surface in sorted_surfaces:
                # Create surface
                await session.run('''
                    CREATE (s:Surface {
                        id: $id,
                        question_key: $question_key,
                        scope_id: $scope_id,
                        claim_count: $claim_count
                    })
                ''', {
                    'id': surface.get('id'),
                    'question_key': surface.get('question_key', ''),
                    'scope_id': surface.get('scope_id', ''),
                    'claim_count': len(surface.get('claim_ids', [])),
                })

                # Link to claims
                for claim_id in surface.get('claim_ids', []):
                    await session.run('''
                        MATCH (s:Surface {id: $surface_id})
                        MATCH (c:Claim {id: $claim_id})
                        MERGE (s)-[:CONTAINS]->(c)
                    ''', {'surface_id': surface['id'], 'claim_id': claim_id})

    async def _load_incidents(self, incidents: List[Dict]):
        """Load incidents in sorted order."""
        sorted_incidents = sorted(incidents, key=lambda i: i.get('id', ''))

        async with self.driver.session() as session:
            for incident in sorted_incidents:
                # Create incident
                await session.run('''
                    CREATE (i:Incident {
                        id: $id,
                        anchor_entities: $anchors,
                        companion_entities: $companions,
                        time_start: $time_start,
                        time_end: $time_end
                    })
                ''', {
                    'id': incident.get('id'),
                    'anchors': incident.get('anchor_entities', []),
                    'companions': incident.get('companion_entities', []),
                    'time_start': incident.get('time_start'),
                    'time_end': incident.get('time_end'),
                })

                # Link to surfaces
                for surface_id in incident.get('surface_ids', []):
                    await session.run('''
                        MATCH (i:Incident {id: $incident_id})
                        MATCH (s:Surface {id: $surface_id})
                        MERGE (i)-[:CONTAINS]->(s)
                    ''', {'incident_id': incident['id'], 'surface_id': surface_id})

    async def _load_cases(self, cases: List[Dict]):
        """Load cases in sorted order."""
        sorted_cases = sorted(cases, key=lambda c: c.get('id', ''))

        async with self.driver.session() as session:
            for case in sorted_cases:
                # Create case
                await session.run('''
                    CREATE (c:Case {
                        id: $id,
                        title: $title,
                        spine: $spine,
                        case_type: $case_type
                    })
                ''', {
                    'id': case.get('id'),
                    'title': case.get('title', ''),
                    'spine': case.get('spine', ''),
                    'case_type': case.get('case_type', 'story'),
                })

                # Link to incidents
                for incident_id in case.get('incident_ids', []):
                    await session.run('''
                        MATCH (c:Case {id: $case_id})
                        MATCH (i:Incident {id: $incident_id})
                        MERGE (c)-[:CONTAINS]->(i)
                    ''', {'case_id': case['id'], 'incident_id': incident_id})

    # =========================================================================
    # SNAPSHOT & COMPARISON
    # =========================================================================

    async def snapshot(self) -> Dict[str, Any]:
        """
        Export current state for comparison.

        Returns ordered dict of all nodes/relationships.
        Order is deterministic (sorted by ID).
        """
        async with self.driver.session() as session:
            # Get all nodes by label
            claims = await self._get_nodes(session, 'Claim')
            surfaces = await self._get_nodes(session, 'Surface')
            incidents = await self._get_nodes(session, 'Incident')
            cases = await self._get_nodes(session, 'Case')

            # Get relationships
            surface_claims = await self._get_relationships(session, 'Surface', 'CONTAINS', 'Claim')
            incident_surfaces = await self._get_relationships(session, 'Incident', 'CONTAINS', 'Surface')
            case_incidents = await self._get_relationships(session, 'Case', 'CONTAINS', 'Incident')

        return {
            'claims': claims,
            'surfaces': surfaces,
            'incidents': incidents,
            'cases': cases,
            'relationships': {
                'surface_claims': surface_claims,
                'incident_surfaces': incident_surfaces,
                'case_incidents': case_incidents,
            },
        }

    async def _get_nodes(self, session, label: str) -> List[Dict]:
        """Get all nodes of a label, sorted by ID."""
        result = await session.run(f'''
            MATCH (n:{label})
            RETURN n
            ORDER BY n.id
        ''')
        records = await result.values()
        return [dict(r[0]) for r in records]

    async def _get_relationships(self, session, from_label: str, rel_type: str, to_label: str) -> List[tuple]:
        """Get all relationships of a type, sorted."""
        result = await session.run(f'''
            MATCH (a:{from_label})-[r:{rel_type}]->(b:{to_label})
            RETURN a.id, b.id
            ORDER BY a.id, b.id
        ''')
        records = await result.values()
        return [(r[0], r[1]) for r in records]

    # =========================================================================
    # QUERY HELPERS
    # =========================================================================

    async def get_incident_count(self) -> int:
        """Get total incident count."""
        async with self.driver.session() as session:
            result = await session.run("MATCH (i:Incident) RETURN count(i) as count")
            record = await result.single()
            return record['count']

    async def get_case_count(self) -> int:
        """Get total case count."""
        async with self.driver.session() as session:
            result = await session.run("MATCH (c:Case) RETURN count(c) as count")
            record = await result.single()
            return record['count']

    async def get_case_incident_counts(self) -> Dict[str, int]:
        """Get incident count per case, sorted by case ID."""
        async with self.driver.session() as session:
            result = await session.run('''
                MATCH (c:Case)-[:CONTAINS]->(i:Incident)
                WITH c, count(i) as incident_count
                RETURN c.id as case_id, incident_count
                ORDER BY c.id
            ''')
            records = await result.values()
            return {r[0]: r[1] for r in records}

    # =========================================================================
    # UTILITIES
    # =========================================================================

    @staticmethod
    def _hash_id(prefix: str, content: str) -> str:
        """Generate stable hash-based ID."""
        h = hashlib.sha256(content.encode()).hexdigest()[:12]
        return f"{prefix}_{h}"


# =============================================================================
# PYTEST FIXTURES
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring Neo4j"
    )


async def get_test_neo4j() -> TestNeo4jManager:
    """Get connected test Neo4j manager."""
    manager = TestNeo4jManager()
    await manager.connect()
    return manager
