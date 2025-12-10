"""
Neo4j Graph Service - Knowledge Graph Operations

Neo4j is the SINGLE SOURCE OF TRUTH for the knowledge graph.
PostgreSQL only stores content text and embeddings.

Node Types:
- Page: {id, url, title, domain, pub_time, status} - metadata only
- Claim: {id, text, confidence, modality, event_time}
- Entity: {id, canonical_name, entity_type, wikidata_qid} - MERGE by QID for dedup
       - Publishers: Entity with is_publisher=true, domain property
- Event: {id, canonical_name, event_type, status, scale}

Relationships:
- (Page)-[:CONTAINS]->(Claim)
- (Page)-[:PUBLISHED_BY]->(Entity) - where Entity.is_publisher=true
- (Claim)-[:MENTIONS]->(Entity)
- (Claim)-[:ACTOR|SUBJECT|LOCATION]->(Entity) - semantic roles
- (Event)-[:SUPPORTS]->(Claim)
- (Event)-[:INVOLVES]->(Entity)
- (Entity)-[:PART_OF|LOCATED_IN|WORKS_FOR]->(Entity)

Entity Deduplication Strategy:
- If QID known: MERGE on wikidata_qid (one entity per real-world thing)
- If no QID: MERGE on (canonical_name, entity_type)
- Publishers: MERGE on dedup_key = 'publisher_{domain}'
- All mentions point to same entity node via relationships
"""
import os
import logging
import json
from typing import Dict, List, Optional, Set, Any
from datetime import datetime

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

    @staticmethod
    def _compute_event_dedup_key(canonical_name: str, event_type: str) -> str:
        """
        Compute deduplication key for event.

        Key = hash of normalized(canonical_name) + event_type
        """
        import hashlib
        normalized = canonical_name.lower().strip()
        key_string = f"{normalized}|{event_type}"
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    async def create_or_get_event(
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
    ) -> tuple[str, bool]:
        """
        Create or get existing Event node in Neo4j (deduplication).

        DEDUPLICATION STRATEGY:
        - MERGE on dedup_key (hash of name + event_type)
        - If event exists, increment page count and return existing ID
        - If new, create with provided ID

        Returns: (event_id, is_new)
        """
        dedup_key = self._compute_event_dedup_key(canonical_name, event_type)

        query = """
        MERGE (e:Event {dedup_key: $dedup_key})
        ON CREATE SET
            e.id = $event_id,
            e.canonical_name = $canonical_name,
            e.event_type = $event_type,
            e.status = $status,
            e.confidence = $confidence,
            e.event_scale = $event_scale,
            e.earliest_time = $earliest_time,
            e.latest_time = $latest_time,
            e.pages_count = 1,
            e.created_at = datetime(),
            e.metadata_json = $metadata_json
        ON MATCH SET
            e.pages_count = e.pages_count + 1,
            e.updated_at = datetime()
        RETURN e.id as id, e.created_at = e.updated_at as is_new
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
            'dedup_key': dedup_key,
            'metadata_json': json.dumps(metadata) if metadata else '{}'
        })

        if result:
            returned_id = result['id']
            is_new = result.get('is_new', True)
            if is_new:
                logger.info(f"✨ Created Event: {canonical_name} ({returned_id})")
            else:
                logger.info(f"📎 Matched Event: {canonical_name} ({returned_id})")
            return returned_id, is_new

        return event_id, True

    async def link_page_to_event(self, page_id: str, event_id: str) -> None:
        """Create ABOUT relationship between Page and Event."""
        query = """
        MATCH (p:Page {id: $page_id})
        MATCH (e:Event {id: $event_id})
        MERGE (p)-[r:ABOUT]->(e)
        ON CREATE SET r.created_at = datetime()
        """
        await self._execute_write(query, {
            'page_id': page_id,
            'event_id': event_id
        })
        logger.debug(f"🔗 Linked Page {page_id} → Event {event_id}")

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
        Create Event node in Neo4j (legacy method - prefer create_or_get_event)

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
        page_embedding: List[float] = None  # DEPRECATED - not stored in Neo4j
    ) -> str:
        """
        Create or merge Claim node (idempotent)

        Note: page_id/page_embedding are NOT stored in Neo4j
        Claims already have page_id in PostgreSQL for source tracking

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
        RETURN c.id as id
        """

        params = {
            'claim_id': claim_id,
            'text': text,
            'modality': modality,
            'confidence': confidence,
            'event_time': event_time
        }

        result = await self._execute_write(query, params)

        logger.debug(f"📝 Created Claim: {text[:50]}... ({claim_id})")
        return result['id'] if result else None

    # ===== Entity Operations =====

    @staticmethod
    def _compute_dedup_key(canonical_name: str, entity_type: str) -> str:
        """
        Compute deduplication key for entity.

        Key = hash of normalized(canonical_name) + entity_type
        This provides a unique constraint target for MERGE operations.
        """
        import hashlib
        normalized = canonical_name.lower().strip()
        key_string = f"{normalized}|{entity_type}"
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    async def create_or_update_entity(
        self,
        entity_id: str,
        canonical_name: str,
        entity_type: str,
        wikidata_qid: str = None
    ) -> str:
        """
        Create or update Entity node in Neo4j (primary entity storage).

        DEDUPLICATION STRATEGY (QID-first, then dedup_key):
        1. If QID provided: check if entity exists by QID (true identity)
        2. If found by QID: use it, add canonical_name as alias
        3. If not found by QID: MERGE on dedup_key (name+type)
        4. If found by dedup_key: use it, set QID if provided
        5. If neither found: create new entity

        This prevents:
        - "John Lee" and "John Lee Ka-chiu" creating two entities for Q9051824
        - Race condition duplicates (unique constraint on dedup_key)

        Returns: entity_id (may differ from input if entity already existed)
        """
        dedup_key = self._compute_dedup_key(canonical_name, entity_type)

        # Phase 1: If QID provided, check by QID first (true identity)
        if wikidata_qid:
            existing = await self.get_entity_by_qid(wikidata_qid)
            if existing:
                # Entity with this QID exists - use it
                existing_id = existing['id']
                # Add canonical_name as alias if different from existing name
                if existing['canonical_name'] != canonical_name:
                    await self.add_entity_alias(existing_id, canonical_name)
                # Increment mention count
                await self._execute_write("""
                    MATCH (e:Entity {id: $entity_id})
                    SET e.mention_count = e.mention_count + 1,
                        e.updated_at = datetime()
                """, {'entity_id': existing_id})
                logger.debug(f"📦 Entity by QID: {canonical_name} → existing {existing['canonical_name']} [{wikidata_qid}]")
                return existing_id

        # Phase 2: MERGE on dedup_key (name+type)
        query = """
        MERGE (e:Entity {dedup_key: $dedup_key})
        ON CREATE SET
            e.id = $entity_id,
            e.canonical_name = $canonical_name,
            e.entity_type = $entity_type,
            e.mention_count = 1,
            e.created_at = datetime()
        ON MATCH SET
            e.mention_count = e.mention_count + 1,
            e.updated_at = datetime()
        WITH e
        // If QID provided and entity doesn't have one, set it
        SET e.wikidata_qid = CASE
            WHEN $wikidata_qid IS NOT NULL AND e.wikidata_qid IS NULL
            THEN $wikidata_qid
            ELSE e.wikidata_qid
        END
        RETURN e.id as id, e.wikidata_qid as existing_qid
        """

        result = await self._execute_write(query, {
            'entity_id': entity_id,
            'canonical_name': canonical_name,
            'entity_type': entity_type,
            'wikidata_qid': wikidata_qid,
            'dedup_key': dedup_key
        })

        if result:
            returned_id = result['id']
            existing_qid = result.get('existing_qid')

            # Check for QID conflict: entity has different QID than provided
            if wikidata_qid and existing_qid and existing_qid != wikidata_qid:
                logger.warning(f"⚠️ QID conflict for '{canonical_name}': "
                             f"existing={existing_qid}, new={wikidata_qid}")

            qid_msg = f" [{existing_qid or wikidata_qid}]" if (existing_qid or wikidata_qid) else ""
            logger.debug(f"📦 Entity: {canonical_name} ({entity_type}){qid_msg} → {returned_id}")
            return returned_id

        return entity_id

    async def get_entity_by_qid(self, wikidata_qid: str) -> Optional[Dict]:
        """
        Get entity by Wikidata QID.

        This is the primary lookup for deduplication - if an entity with
        this QID exists, we should use it instead of creating a new one.
        """
        query = """
        MATCH (e:Entity {wikidata_qid: $qid})
        RETURN e.id as id, e.canonical_name as canonical_name,
               e.entity_type as entity_type, e.mention_count as mention_count,
               e.wikidata_qid as wikidata_qid, e.wikidata_label as wikidata_label,
               e.wikidata_description as wikidata_description,
               e.aliases as aliases, e.status as status
        """
        results = await self._execute_read(query, {'qid': wikidata_qid})
        return results[0] if results else None

    async def add_entity_alias(
        self,
        entity_id: str,
        alias: str
    ) -> None:
        """
        Add an alias to an entity (for fuzzy matching later).

        Aliases are surface forms that should resolve to this entity.
        """
        query = """
        MATCH (e:Entity {id: $entity_id})
        WHERE NOT $alias IN COALESCE(e.aliases, [])
        SET e.aliases = COALESCE(e.aliases, []) + $alias
        """
        await self._execute_write(query, {
            'entity_id': entity_id,
            'alias': alias
        })

    async def merge_duplicate_entities_by_qid(self) -> Dict:
        """
        Find and merge entities that share the same QID but have different IDs.

        This shouldn't happen with proper MERGE logic, but handles edge cases.

        Returns: Stats about merged entities
        """
        # Find duplicates
        duplicates = await self._execute_read("""
            MATCH (e:Entity)
            WHERE e.wikidata_qid IS NOT NULL
            WITH e.wikidata_qid as qid, COLLECT(e) as entities
            WHERE SIZE(entities) > 1
            RETURN qid,
                   [entity IN entities | entity.id] as ids,
                   [entity IN entities | entity.canonical_name] as names,
                   [entity IN entities | entity.mention_count] as counts
        """, {})

        merged_count = 0
        for dup in duplicates:
            # Keep the entity with highest mention count
            max_idx = dup['counts'].index(max(dup['counts']))
            canonical_id = dup['ids'][max_idx]
            duplicate_ids = [id for i, id in enumerate(dup['ids']) if i != max_idx]

            if duplicate_ids:
                await self.merge_entities(
                    canonical_id=canonical_id,
                    duplicate_ids=duplicate_ids,
                    total_mentions=sum(dup['counts'])
                )
                merged_count += len(duplicate_ids)
                logger.info(f"🔗 Merged {len(duplicate_ids)} duplicates for QID {dup['qid']}")

        return {'duplicates_found': len(duplicates), 'entities_merged': merged_count}

    async def get_entities_by_ids(self, entity_ids: List[str]) -> List[Dict]:
        """
        Retrieve multiple entities by IDs from Neo4j

        Args:
            entity_ids: List of Entity IDs (en_xxxxxxxx format)

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
        # Extract thumbnail_url from metadata to store as top-level property
        wikidata_image = metadata.get('thumbnail_url') if metadata else None

        query = """
        MATCH (e:Entity {id: $entity_id})
        SET e.wikidata_qid = $wikidata_qid,
            e.wikidata_label = $wikidata_label,
            e.wikidata_description = $wikidata_description,
            e.wikidata_image = $wikidata_image,
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
            'wikidata_image': wikidata_image,
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
        SET e.checked = true,
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
            entity_id: Entity ID (en_xxxxxxxx format)
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

    async def link_claim_to_event(
        self,
        event_id: str,
        claim_id: str,
        relationship_type: str = "SUPPORTS"  # SUPPORTS, CONTRADICTS, UPDATES
    ):
        """
        Link Claim to Event with specified relationship type

        Creates direct graph relationship: Event-[SUPPORTS]->Claim
        This makes it easy to query all claims for an event via Cypher
        """
        query = f"""
        MATCH (e:Event {{id: $event_id}})
        MATCH (c:Claim {{id: $claim_id}})
        MERGE (e)-[:{relationship_type} {{created_at: datetime()}}]->(c)
        """

        await self._execute_write(query, {
            'event_id': event_id,
            'claim_id': claim_id
        })

    async def create_causal_relationship(
        self,
        from_event_id: str,
        to_event_id: str,
        relationship_type: str,  # CAUSED, TRIGGERED, PART_OF, RELATED_TO
        confidence: float = 0.8,
        metadata: Dict = None
    ):
        """Create causal/peer relationship between two events (CAUSED, TRIGGERED, etc.)"""
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

    # ===== Page Operations =====

    async def create_or_update_page(
        self,
        page_id: str,
        url: str,
        title: str = None,
        domain: str = None,
        pub_time: str = None,
        status: str = "stub",
        language: str = "en",
        word_count: int = 0,
        metadata_confidence: float = 0.0,
        claims_count: int = 0,
        entities_count: int = 0
    ) -> str:
        """
        Create or update Page node in Neo4j.

        Page nodes store metadata only - content stays in PostgreSQL.
        Includes confidence metrics for the page.

        Returns: page_id
        """
        query = """
        MERGE (p:Page {id: $page_id})
        ON CREATE SET
            p.url = $url,
            p.title = $title,
            p.domain = $domain,
            p.pub_time = $pub_time,
            p.status = $status,
            p.language = $language,
            p.word_count = $word_count,
            p.metadata_confidence = $metadata_confidence,
            p.claims_count = $claims_count,
            p.entities_count = $entities_count,
            p.created_at = datetime()
        ON MATCH SET
            p.title = COALESCE($title, p.title),
            p.domain = COALESCE($domain, p.domain),
            p.pub_time = COALESCE($pub_time, p.pub_time),
            p.status = $status,
            p.language = COALESCE($language, p.language),
            p.word_count = CASE WHEN $word_count > 0 THEN $word_count ELSE p.word_count END,
            p.metadata_confidence = CASE WHEN $metadata_confidence > 0 THEN $metadata_confidence ELSE p.metadata_confidence END,
            p.claims_count = CASE WHEN $claims_count > 0 THEN $claims_count ELSE p.claims_count END,
            p.entities_count = CASE WHEN $entities_count > 0 THEN $entities_count ELSE p.entities_count END,
            p.updated_at = datetime()
        RETURN p.id as id
        """

        result = await self._execute_write(query, {
            'page_id': page_id,
            'url': url,
            'title': title,
            'domain': domain,
            'pub_time': pub_time,
            'status': status,
            'language': language,
            'word_count': word_count,
            'metadata_confidence': metadata_confidence,
            'claims_count': claims_count,
            'entities_count': entities_count
        })

        logger.debug(f"📄 Created/updated Page node: {title or url}")
        return result['id'] if result else page_id

    async def get_page_by_id(self, page_id: str) -> Optional[Dict]:
        """Get page by ID from Neo4j."""
        query = """
        MATCH (p:Page {id: $page_id})
        RETURN p.id as id, p.url as url, p.title as title,
               p.domain as domain, p.pub_time as pub_time,
               p.status as status, p.language as language,
               p.created_at as created_at
        """
        results = await self._execute_read(query, {'page_id': page_id})
        return results[0] if results else None

    async def update_page_status(self, page_id: str, status: str) -> None:
        """Update page status in Neo4j."""
        query = """
        MATCH (p:Page {id: $page_id})
        SET p.status = $status, p.updated_at = datetime()
        """
        await self._execute_write(query, {'page_id': page_id, 'status': status})

    async def link_page_to_claim(self, page_id: str, claim_id: str) -> None:
        """Create EXTRACTED relationship between Page and Claim."""
        query = """
        MATCH (p:Page {id: $page_id})
        MATCH (c:Claim {id: $claim_id})
        MERGE (p)-[r:EXTRACTED]->(c)
        ON CREATE SET r.created_at = datetime()
        """
        await self._execute_write(query, {
            'page_id': page_id,
            'claim_id': claim_id
        })

    async def get_page_claims(self, page_id: str) -> List[Dict]:
        """Get all claims for a page."""
        query = """
        MATCH (p:Page {id: $page_id})-[:CONTAINS]->(c:Claim)
        RETURN c.id as id, c.text as text, c.confidence as confidence,
               c.modality as modality, c.event_time as event_time
        ORDER BY c.created_at
        """
        return await self._execute_read(query, {'page_id': page_id})

    async def get_page_entities(self, page_id: str) -> List[Dict]:
        """Get all entities mentioned in a page's claims."""
        query = """
        MATCH (p:Page {id: $page_id})-[:CONTAINS]->(c:Claim)-[:MENTIONS]->(e:Entity)
        RETURN DISTINCT e.id as id, e.canonical_name as canonical_name,
               e.entity_type as entity_type, e.wikidata_qid as wikidata_qid,
               e.mention_count as mention_count
        ORDER BY e.mention_count DESC
        """
        return await self._execute_read(query, {'page_id': page_id})

    # ===== Query Operations =====

    async def initialize_constraints(self):
        """Create Neo4j constraints and indexes for knowledge graph."""
        constraints = [
            # Unique IDs for all node types
            "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT page_id IF NOT EXISTS FOR (p:Page) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT source_id IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE",
            # Entity QID uniqueness (key for deduplication)
            "CREATE CONSTRAINT entity_qid IF NOT EXISTS FOR (e:Entity) REQUIRE e.wikidata_qid IS UNIQUE",
            # Entity dedup_key uniqueness (prevents race condition duplicates)
            "CREATE CONSTRAINT entity_dedup_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.dedup_key IS UNIQUE",
            # TODO: Event dedup_key constraint - design event deduplication strategy first
            # "CREATE CONSTRAINT event_dedup_key IF NOT EXISTS FOR (e:Event) REQUIRE e.dedup_key IS UNIQUE",
            # Source domain uniqueness
            "CREATE CONSTRAINT source_domain IF NOT EXISTS FOR (s:Source) REQUIRE s.domain IS UNIQUE",
            # Indexes for common queries
            "CREATE INDEX event_status IF NOT EXISTS FOR (e:Event) ON (e.status)",
            "CREATE INDEX event_time IF NOT EXISTS FOR (e:Event) ON (e.earliest_time)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.canonical_name)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
            "CREATE INDEX page_status IF NOT EXISTS FOR (p:Page) ON (p.status)",
            "CREATE INDEX page_domain IF NOT EXISTS FOR (p:Page) ON (p.domain)",
            "CREATE INDEX claim_confidence IF NOT EXISTS FOR (c:Claim) ON (c.confidence)"
        ]

        for constraint_query in constraints:
            try:
                await self._execute_write(constraint_query)
                logger.info(f"✅ {constraint_query.split()[1]} created")
            except Exception as e:
                logger.warning(f"⚠️  Constraint/index creation failed (may already exist): {e}")
