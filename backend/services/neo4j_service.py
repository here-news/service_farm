"""
Neo4j Graph Service - Event Graph Operations

Provides interface for creating and querying the event graph in Neo4j.

Key operations:
- Create Event nodes with phases
- Create Claim nodes and link to events/phases
- Create Entity nodes and relationships (MENTIONS, ACTOR, SUBJECT, LOCATION)
- Query for event attachment scoring
- Generate narrative from graph traversal
"""
import os
import logging
import json
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
import uuid

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession

logger = logging.getLogger(__name__)


class Neo4jService:
    """Service for Neo4j graph operations"""

    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None
    ):
        """Initialize Neo4j connection"""
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://neo4j:7687')
        self.user = user or os.getenv('NEO4J_USER', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', 'herenews_neo4j_pass')

        self.driver: Optional[AsyncDriver] = None

    async def connect(self):
        """Establish connection to Neo4j"""
        if not self.driver:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # Verify connectivity
            await self.driver.verify_connectivity()
            logger.info(f"✅ Connected to Neo4j at {self.uri}")

    async def close(self):
        """Close Neo4j connection"""
        if self.driver:
            await self.driver.close()
            self.driver = None
            logger.info("🔌 Closed Neo4j connection")

    async def _execute_write(self, query: str, parameters: Dict = None):
        """Execute write query"""
        async with self.driver.session() as session:
            result = await session.run(query, parameters or {})
            return await result.single()

    async def _execute_read(self, query: str, parameters: Dict = None):
        """Execute read query"""
        async with self.driver.session() as session:
            result = await session.run(query, parameters or {})
            return await result.data()

    # ===== Event Operations =====

    async def create_event(
        self,
        event_id: str,
        canonical_name: str,
        event_type: str = "UNSPECIFIED",
        status: str = "provisional",
        confidence: float = 0.5,
        event_scale: str = "micro",
        earliest_time: datetime = None,
        latest_time: datetime = None,
        metadata: Dict = None
    ) -> str:
        """
        Create Event node in Neo4j

        Returns: event_id
        """
        query = """
        CREATE (e:Event {
            id: $event_id,
            canonical_name: $canonical_name,
            event_type: $event_type,
            status: $status,
            confidence: $confidence,
            event_scale: $event_scale,
            earliest_time: $earliest_time,
            latest_time: $latest_time,
            created_at: datetime(),
            updated_at: datetime(),
            metadata_json: $metadata_json
        })
        RETURN e.id as id
        """

        result = await self._execute_write(query, {
            'event_id': event_id,
            'canonical_name': canonical_name,
            'event_type': event_type,
            'status': status,
            'confidence': confidence,
            'event_scale': event_scale,
            'earliest_time': earliest_time,
            'latest_time': latest_time,
            'metadata_json': json.dumps(metadata) if metadata else '{}'
        })

        logger.info(f"✨ Created Event node: {canonical_name} ({event_id})")
        return result['id'] if result else None

    async def create_phase(
        self,
        phase_id: str,
        event_id: str,
        name: str,
        phase_type: str,
        start_time: datetime = None,
        end_time: datetime = None,
        confidence: float = 0.9,
        sequence: int = 1
    ) -> str:
        """
        Create Phase node and link to Event (graph structure only)

        Phase embedding is stored in PostgreSQL core.event_phases table

        Returns: phase_id
        """
        query = """
        MATCH (e:Event {id: $event_id})
        CREATE (p:Phase {
            id: $phase_id,
            name: $name,
            phase_type: $phase_type,
            start_time: $start_time,
            end_time: $end_time,
            confidence: $confidence
        })
        CREATE (e)-[:HAS_PHASE {sequence: $sequence}]->(p)
        RETURN p.id as id
        """

        result = await self._execute_write(query, {
            'event_id': event_id,
            'phase_id': phase_id,
            'name': name,
            'phase_type': phase_type,
            'start_time': start_time,
            'end_time': end_time,
            'confidence': confidence,
            'sequence': sequence
        })

        logger.info(f"📍 Created Phase: {name} ({phase_id})")
        return result['id'] if result else None

    async def create_claim(
        self,
        claim_id: str,
        text: str,
        modality: str = "observation",
        confidence: float = 0.8,
        event_time: datetime = None,
        page_id: str = None,
        page_embedding: List[float] = None
    ) -> str:
        """
        Create or merge Claim node (idempotent)

        Returns: claim_id
        """
        query = """
        MERGE (c:Claim {id: $claim_id})
        ON CREATE SET
            c.text = $text,
            c.modality = $modality,
            c.confidence = $confidence,
            c.event_time = $event_time,
            c.extracted_at = datetime()
        """

        params = {
            'claim_id': claim_id,
            'text': text,
            'modality': modality,
            'confidence': confidence,
            'event_time': event_time
        }

        # Link to page if provided
        if page_id:
            query += """
            WITH c
            MERGE (p:Page {id: $page_id})
            ON CREATE SET p.embedding = $page_embedding
            CREATE (c)-[:FROM_PAGE]->(p)
            """
            params['page_id'] = page_id
            params['page_embedding'] = page_embedding

        query += " RETURN c.id as id"

        result = await self._execute_write(query, params)

        logger.debug(f"📝 Created Claim: {text[:50]}... ({claim_id})")
        return result['id'] if result else None

    async def link_claim_to_phase(
        self,
        claim_id: str,
        phase_id: str,
        confidence: float = 0.9
    ):
        """Link Claim to Phase with SUPPORTED_BY relationship"""
        query = """
        MATCH (p:Phase {id: $phase_id})
        MATCH (c:Claim {id: $claim_id})
        CREATE (p)-[:SUPPORTED_BY {confidence: $confidence}]->(c)
        """

        await self._execute_write(query, {
            'phase_id': phase_id,
            'claim_id': claim_id,
            'confidence': confidence
        })

    async def link_claim_to_entity(
        self,
        claim_id: str,
        entity_id: str,
        relationship_type: str = "MENTIONS",  # or ACTOR, SUBJECT, LOCATION
        canonical_name: str = None,
        entity_type: str = None
    ):
        """
        Link Claim to Entity with specified relationship type

        Creates Entity node if it doesn't exist (using MERGE)
        """
        query = """
        MATCH (c:Claim {id: $claim_id})
        MERGE (e:Entity {id: $entity_id})
        ON CREATE SET
            e.canonical_name = $canonical_name,
            e.entity_type = $entity_type,
            e.mention_count = 1
        ON MATCH SET
            e.mention_count = e.mention_count + 1
        """

        # Add appropriate relationship based on type
        if relationship_type == "ACTOR":
            query += " CREATE (c)-[:ACTOR]->(e)"
        elif relationship_type == "SUBJECT":
            query += " CREATE (c)-[:SUBJECT]->(e)"
        elif relationship_type == "LOCATION":
            query += " CREATE (c)-[:LOCATION]->(e)"
        else:  # Default: MENTIONS
            query += " CREATE (c)-[:MENTIONS]->(e)"

        await self._execute_write(query, {
            'claim_id': claim_id,
            'entity_id': entity_id,
            'canonical_name': canonical_name,
            'entity_type': entity_type
        })

    async def create_event_relationship(
        self,
        from_event_id: str,
        to_event_id: str,
        relationship_type: str,  # CAUSED, TRIGGERED, PART_OF, RELATED_TO
        confidence: float = 0.8,
        metadata: Dict = None
    ):
        """Create relationship between two events"""
        query = f"""
        MATCH (e1:Event {{id: $from_event_id}})
        MATCH (e2:Event {{id: $to_event_id}})
        CREATE (e1)-[:{relationship_type} {{confidence: $confidence, metadata: $metadata}}]->(e2)
        """

        await self._execute_write(query, {
            'from_event_id': from_event_id,
            'to_event_id': to_event_id,
            'confidence': confidence,
            'metadata': metadata or {}
        })

        logger.info(f"🔗 Created {relationship_type} relationship: {from_event_id} → {to_event_id}")

    async def mark_claim_evolution(
        self,
        old_claim_id: str,
        new_claim_id: str
    ):
        """Mark temporal evolution of claims (e.g., 4 dead → 128 dead)"""
        query = """
        MATCH (c1:Claim {id: $old_claim_id})
        MATCH (c2:Claim {id: $new_claim_id})
        CREATE (c1)-[:EVOLVED_TO]->(c2)
        """

        await self._execute_write(query, {
            'old_claim_id': old_claim_id,
            'new_claim_id': new_claim_id
        })

    # ===== Query Operations =====

    async def get_event_with_phases(self, event_id: str) -> Dict:
        """Get event with all phases and claims"""
        query = """
        MATCH (e:Event {id: $event_id})
        OPTIONAL MATCH (e)-[:HAS_PHASE]->(p:Phase)
        OPTIONAL MATCH (p)-[:SUPPORTED_BY]->(c:Claim)
        RETURN e, collect(DISTINCT p) as phases, collect(DISTINCT c) as claims
        """

        results = await self._execute_read(query, {'event_id': event_id})

        if not results:
            return None

        return results[0]

    async def find_candidate_events(
        self,
        entity_names: Set[str],
        time_window_days: int = 7,
        reference_time: datetime = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Find candidate events for attachment based on entity overlap

        Returns list of events with entity overlap scores
        """
        query = """
        MATCH (e:Event)
        WHERE e.status IN ['provisional', 'emerging', 'stable']
        """

        # Add temporal filter if reference_time provided
        if reference_time:
            query += """
            AND duration.between(e.earliest_time, $reference_time).days <= $time_window_days
            """

        query += """
        OPTIONAL MATCH (e)-[:HAS_PHASE]->(p:Phase)-[:SUPPORTED_BY]->(c:Claim)-[:MENTIONS]->(entity:Entity)
        WHERE entity.canonical_name IN $entity_names
        WITH e, count(DISTINCT entity) as entity_overlap
        WHERE entity_overlap > 0
        RETURN e, entity_overlap
        ORDER BY entity_overlap DESC, e.updated_at DESC
        LIMIT $limit
        """

        results = await self._execute_read(query, {
            'entity_names': list(entity_names),
            'reference_time': reference_time,
            'time_window_days': time_window_days,
            'limit': limit
        })

        return results

    async def get_event_narrative(self, event_id: str) -> Dict:
        """
        Generate narrative from event graph traversal

        Returns structured narrative with timeline, phases, entities
        """
        query = """
        MATCH (e:Event {id: $event_id})
        OPTIONAL MATCH (e)-[:HAS_PHASE]->(p:Phase)
        OPTIONAL MATCH (p)-[:SUPPORTED_BY]->(c:Claim)
        OPTIONAL MATCH (c)-[:MENTIONS|ACTOR|SUBJECT|LOCATION]->(entity:Entity)
        RETURN
            e,
            collect(DISTINCT {
                phase: p,
                claims: collect(DISTINCT c),
                entities: collect(DISTINCT entity)
            }) as phases
        """

        results = await self._execute_read(query, {'event_id': event_id})

        if not results:
            return None

        return results[0]

    async def get_casualty_evolution(self, event_id: str) -> List[Dict]:
        """
        Get casualty evolution for event (tracking death toll updates)

        Follows EVOLVED_TO edges to track claim evolution
        """
        query = """
        MATCH (e:Event {id: $event_id})-[:HAS_PHASE]->(p:Phase)
        WHERE p.name CONTAINS 'Casualty' OR p.phase_type = 'CONSEQUENCE'
        MATCH (p)-[:SUPPORTED_BY]->(c1:Claim)
        OPTIONAL MATCH path = (c1)-[:EVOLVED_TO*]->(c2:Claim)
        RETURN c1, c2, path
        ORDER BY c1.event_time
        """

        results = await self._execute_read(query, {'event_id': event_id})
        return results

    async def initialize_constraints(self):
        """Create Neo4j constraints and indexes"""
        constraints = [
            "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT phase_id IF NOT EXISTS FOR (p:Phase) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT page_id IF NOT EXISTS FOR (p:Page) REQUIRE p.id IS UNIQUE",
            "CREATE INDEX event_status IF NOT EXISTS FOR (e:Event) ON (e.status)",
            "CREATE INDEX event_time IF NOT EXISTS FOR (e:Event) ON (e.earliest_time)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.canonical_name)"
        ]

        for constraint_query in constraints:
            try:
                await self._execute_write(constraint_query)
                logger.info(f"✅ {constraint_query.split()[1]} created")
            except Exception as e:
                logger.warning(f"⚠️  Constraint/index creation failed (may already exist): {e}")
