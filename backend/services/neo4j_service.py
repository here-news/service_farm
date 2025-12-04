"""
Neo4j Graph Service - Event Graph Operations

Provides interface for creating and querying the event graph in Neo4j.

Key operations:
- Create Event nodes and recursive sub-event structures
- Create Claim nodes and link to events
- Create Entity nodes and relationships (MENTIONS, ACTOR, SUBJECT, LOCATION, INVOLVES)
- Support recursive event formation with parent-child relationships
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

    async def get_parent_event_id(self, event_id: str) -> Optional[str]:
        """
        Get parent event ID from Neo4j relationships (:CONTAINS)

        Args:
            event_id: Child event ID

        Returns:
            Parent event ID or None if root event
        """
        query = """
        MATCH (parent:Event)-[:CONTAINS]->(child:Event {id: $event_id})
        RETURN parent.id as parent_id
        LIMIT 1
        """

        result = await self._execute_read(query, {'event_id': event_id})
        return result[0]['parent_id'] if result else None

    async def get_sub_event_ids(self, event_id: str) -> List[str]:
        """
        Get all sub-event IDs from Neo4j relationships (:CONTAINS)

        Args:
            event_id: Parent event ID

        Returns:
            List of sub-event IDs
        """
        query = """
        MATCH (parent:Event {id: $event_id})-[:CONTAINS]->(child:Event)
        RETURN child.id as child_id
        ORDER BY child.created_at
        """

        result = await self._execute_read(query, {'event_id': event_id})
        return [row['child_id'] for row in result]

    async def create_event_relationship(
        self,
        parent_id: str,
        child_id: str,
        relationship_type: str = "CONTAINS"
    ) -> bool:
        """
        Create relationship between parent and child events

        Args:
            parent_id: Parent event ID
            child_id: Child event ID
            relationship_type: Type of relationship (CONTAINS, PHASE_OF, etc.)

        Returns:
            True if successful
        """
        query = f"""
        MATCH (parent:Event {{id: $parent_id}})
        MATCH (child:Event {{id: $child_id}})
        MERGE (parent)-[r:{relationship_type} {{created_at: datetime()}}]->(child)
        RETURN r
        """

        result = await self._execute_write(query, {
            'parent_id': parent_id,
            'child_id': child_id
        })

        logger.info(f"🔗 Created {relationship_type} relationship: {parent_id} → {child_id}")
        return result is not None

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

    # ===== Entity Operations =====

    async def create_or_update_entity(
        self,
        entity_id: str,
        canonical_name: str,
        entity_type: str
    ) -> str:
        """
        Create or update Entity node in Neo4j (primary entity storage)

        Uses MERGE for idempotency based on (canonical_name, entity_type)
        Increments mention_count on match

        Returns: entity_id
        """
        query = """
        MERGE (e:Entity {canonical_name: $canonical_name, entity_type: $entity_type})
        ON CREATE SET
            e.id = $entity_id,
            e.mention_count = 1,
            e.created_at = datetime()
        ON MATCH SET
            e.mention_count = e.mention_count + 1,
            e.updated_at = datetime()
        RETURN e.id as id
        """

        result = await self._execute_write(query, {
            'entity_id': entity_id,
            'canonical_name': canonical_name,
            'entity_type': entity_type
        })

        logger.debug(f"📦 Created/updated Entity: {canonical_name} ({entity_type})")
        return result['id'] if result else entity_id

    async def get_entities_by_ids(self, entity_ids: List[str]) -> List[Dict]:
        """
        Retrieve multiple entities by IDs from Neo4j

        Args:
            entity_ids: List of Entity UUIDs

        Returns:
            List of entity dictionaries
        """
        if not entity_ids:
            return []

        query = """
        MATCH (e:Entity)
        WHERE e.id IN $entity_ids
        RETURN e.id as id, e.canonical_name as canonical_name,
               e.entity_type as entity_type, e.mention_count as mention_count,
               e.created_at as created_at, e.wikidata_qid as wikidata_qid,
               e.wikidata_label as wikidata_label,
               e.wikidata_description as wikidata_description,
               e.status as status, e.confidence as confidence,
               e.aliases as aliases, e.profile_summary as profile_summary
        """

        results = await self._execute_read(query, {'entity_ids': entity_ids})
        return results

    async def get_entity_by_id(self, entity_id: str) -> Optional[Dict]:
        """
        Get entity by ID

        Returns entity data or None
        """
        query = """
        MATCH (e:Entity {id: $entity_id})
        RETURN e.id as id, e.canonical_name as canonical_name,
               e.entity_type as entity_type, e.mention_count as mention_count,
               e.created_at as created_at, e.wikidata_qid as wikidata_qid,
               e.wikidata_label as wikidata_label,
               e.wikidata_description as wikidata_description,
               e.status as status, e.confidence as confidence,
               e.aliases as aliases
        """

        results = await self._execute_read(query, {'entity_id': entity_id})
        return results[0] if results else None

    async def get_entity_by_name_and_type(
        self,
        canonical_name: str,
        entity_type: str
    ) -> Optional[Dict]:
        """
        Get entity by canonical name and type

        Returns entity data or None
        """
        query = """
        MATCH (e:Entity {canonical_name: $canonical_name, entity_type: $entity_type})
        RETURN e.id as id, e.canonical_name as canonical_name,
               e.entity_type as entity_type, e.mention_count as mention_count,
               e.created_at as created_at, e.wikidata_qid as wikidata_qid,
               e.status as status
        """

        results = await self._execute_read(query, {
            'canonical_name': canonical_name,
            'entity_type': entity_type
        })

        return results[0] if results else None

    async def enrich_entity(
        self,
        entity_id: str,
        wikidata_qid: str,
        wikidata_label: str,
        wikidata_description: str,
        confidence: float,
        aliases: list,
        metadata: dict
    ) -> None:
        """
        Enrich entity with Wikidata information

        Stores: QID, label, description, confidence, aliases, metadata (thumbnail, coords, etc.)
        """
        query = """
        MATCH (e:Entity {id: $entity_id})
        SET e.wikidata_qid = $wikidata_qid,
            e.wikidata_label = $wikidata_label,
            e.wikidata_description = $wikidata_description,
            e.confidence = $confidence,
            e.aliases = $aliases,
            e.metadata_json = $metadata_json,
            e.status = 'enriched',
            e.updated_at = datetime()
        """

        await self._execute_write(query, {
            'entity_id': entity_id,
            'wikidata_qid': wikidata_qid,
            'wikidata_label': wikidata_label,
            'wikidata_description': wikidata_description,
            'confidence': confidence,
            'aliases': aliases,
            'metadata_json': json.dumps(metadata)
        })

        logger.info(f"📦 Enriched Entity {entity_id} with Wikidata {wikidata_qid}")

    async def mark_entity_checked(self, entity_id: str) -> None:
        """
        Mark entity as checked (no Wikidata match found)
        """
        query = """
        MATCH (e:Entity {id: $entity_id})
        SET e.status = 'checked',
            e.updated_at = datetime()
        """

        await self._execute_write(query, {
            'entity_id': entity_id
        })

        logger.debug(f"✓ Marked Entity {entity_id} as checked")

    async def update_entity_profile(
        self,
        entity_id: str,
        profile_summary: str
    ) -> None:
        """
        Update entity profile summary (generated from claim contexts)

        Args:
            entity_id: Entity UUID
            profile_summary: AI-generated description of entity
        """
        query = """
        MATCH (e:Entity {id: $entity_id})
        SET e.profile_summary = $profile_summary,
            e.updated_at = datetime()
        """

        await self._execute_write(query, {
            'entity_id': entity_id,
            'profile_summary': profile_summary
        })

        logger.debug(f"✓ Updated profile for Entity {entity_id}")

    async def find_duplicate_entities(self) -> List[Dict]:
        """
        Find all entities that share the same Wikidata QID (duplicates)

        Returns:
            List of duplicate groups with entity details
        """
        query = """
        MATCH (e:Entity)
        WHERE e.wikidata_qid IS NOT NULL
        WITH e.wikidata_qid as qid, collect(e) as entities
        WHERE size(entities) > 1
        RETURN qid as wikidata_qid,
               [entity IN entities | entity.id] as entity_ids,
               [entity IN entities | entity.canonical_name] as names,
               [entity IN entities | entity.mention_count] as mention_counts
        ORDER BY size(entities) DESC
        """

        results = await self._execute_read(query, {})
        return results

    async def merge_entities(
        self,
        canonical_id: str,
        duplicate_ids: List[str],
        total_mentions: int
    ) -> None:
        """
        Merge duplicate entities into canonical entity

        Steps:
        1. Re-point all relationships from duplicates to canonical
        2. Update canonical entity mention count
        3. Delete duplicate entities

        Args:
            canonical_id: ID of canonical entity to keep
            duplicate_ids: List of duplicate entity IDs to merge
            total_mentions: Total mention count after merge
        """
        query = """
        // Step 1: Find canonical entity and duplicates
        MATCH (canonical:Entity {id: $canonical_id})
        MATCH (duplicate:Entity)
        WHERE duplicate.id IN $duplicate_ids

        // Step 2: Re-point all MENTIONS relationships
        WITH canonical, collect(duplicate) as duplicates
        UNWIND duplicates as dup
        OPTIONAL MATCH (c:Claim)-[r:MENTIONS]->(dup)
        WITH canonical, dup, collect(c) as claims
        FOREACH (claim IN claims |
            MERGE (claim)-[:MENTIONS]->(canonical)
        )

        // Step 3: Re-point all other relationships (ACTOR, SUBJECT, LOCATION)
        WITH canonical, dup
        OPTIONAL MATCH (c:Claim)-[r:ACTOR|SUBJECT|LOCATION]->(dup)
        WITH canonical, dup, type(r) as rel_type, collect(c) as claims
        FOREACH (claim IN claims |
            FOREACH (ignoreMe IN CASE WHEN rel_type = 'ACTOR' THEN [1] ELSE [] END |
                MERGE (claim)-[:ACTOR]->(canonical)
            )
            FOREACH (ignoreMe IN CASE WHEN rel_type = 'SUBJECT' THEN [1] ELSE [] END |
                MERGE (claim)-[:SUBJECT]->(canonical)
            )
            FOREACH (ignoreMe IN CASE WHEN rel_type = 'LOCATION' THEN [1] ELSE [] END |
                MERGE (claim)-[:LOCATION]->(canonical)
            )
        )

        // Step 4: Re-point Event relationships
        WITH canonical, dup
        OPTIONAL MATCH (e:Event)-[r:INVOLVES]->(dup)
        WITH canonical, dup, collect(e) as events
        FOREACH (event IN events |
            MERGE (event)-[:INVOLVES]->(canonical)
        )

        // Step 5: Delete old relationships and duplicates
        WITH canonical, dup
        OPTIONAL MATCH (c)-[r]->(dup)
        DELETE r
        WITH canonical, dup
        DELETE dup

        // Step 6: Update canonical mention count
        WITH canonical
        SET canonical.mention_count = $total_mentions,
            canonical.updated_at = datetime()
        """

        await self._execute_write(query, {
            'canonical_id': canonical_id,
            'duplicate_ids': duplicate_ids,
            'total_mentions': total_mentions
        })

        logger.info(f"🔗 Merged {len(duplicate_ids)} entities into {canonical_id}")

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

        Note: Entity should already exist (created via create_or_update_entity)
        This just creates the relationship
        """
        query = """
        MATCH (c:Claim {id: $claim_id})
        MATCH (e:Entity {id: $entity_id})
        """

        # Add appropriate relationship based on type
        if relationship_type == "ACTOR":
            query += " MERGE (c)-[:ACTOR]->(e)"
        elif relationship_type == "SUBJECT":
            query += " MERGE (c)-[:SUBJECT]->(e)"
        elif relationship_type == "LOCATION":
            query += " MERGE (c)-[:LOCATION]->(e)"
        else:  # Default: MENTIONS
            query += " MERGE (c)-[:MENTIONS]->(e)"

        await self._execute_write(query, {
            'claim_id': claim_id,
            'entity_id': entity_id
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

    async def initialize_constraints(self):
        """Create Neo4j constraints and indexes"""
        constraints = [
            "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE",
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
