"""
EntityManager - Unified entity deduplication and relationship management

This is the single source of truth for entity creation, deduplication, and merging.
All entity operations should go through this service to ensure:

1. Consistent deduplication strategy (QID-first, then dedup_key)
2. Proper relationship rewiring when entities are merged
3. Alias management across duplicates

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                  EntityManager                              │
├─────────────────────────────────────────────────────────────┤
│  get_or_create(name, type, qid=None, **props) → entity_id  │
│    1. If QID provided: check existing by QID → return      │
│    2. MERGE on dedup_key (name+type)                       │
│    3. Returns existing or new entity_id                     │
│                                                             │
│  merge_entities(source_id, target_id)                       │
│    - Transfer all relationships (MENTIONS, INVOLVES, etc)   │
│    - Merge aliases, mention_count                           │
│    - Delete source entity                                   │
│                                                             │
│  update_qid(entity_id, qid, label=None) → may trigger merge│
│    - If another entity has this QID → merge_entities()     │
│    - Returns final entity_id (may differ if merged)        │
│                                                             │
│  add_alias(entity_id, alias)                                │
│    - Add alias if not already present                       │
└─────────────────────────────────────────────────────────────┘
"""

import hashlib
import logging
import re
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)


class EntityManager:
    """
    Unified entity management with deduplication and relationship rewiring.
    """

    def __init__(self, neo4j_service):
        """
        Initialize with Neo4j service for database operations.

        Args:
            neo4j_service: Neo4jService instance for executing queries
        """
        self.neo4j = neo4j_service

    def _compute_dedup_key(self, canonical_name: str, entity_type: str) -> str:
        """
        Compute deterministic deduplication key from name and type.

        Normalization:
        - Lowercase
        - Remove punctuation
        - Collapse whitespace

        This ensures "John Lee" and "john lee" deduplicate.
        """
        normalized = canonical_name.lower().strip()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        key_string = f"{normalized}|{entity_type}"
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    async def get_or_create(
        self,
        entity_id: str,
        canonical_name: str,
        entity_type: str,
        wikidata_qid: str = None,
        wikidata_label: str = None,
        domain: str = None,
        is_publisher: bool = False,
        source_type: str = None,
        base_prior: float = None,
        status: str = None,
        **extra_props
    ) -> str:
        """
        Get existing entity or create new one with proper deduplication.

        DEDUPLICATION STRATEGY (QID-first, then dedup_key):
        1. If QID provided: check if entity exists by QID (true identity)
        2. If found by QID: use it, add canonical_name as alias
        3. If not found by QID: MERGE on dedup_key (name+type)
        4. Returns entity_id (may differ from input if entity already existed)

        This prevents:
        - "John Lee" and "John Lee Ka-chiu" creating two entities for Q9051824
        - Race condition duplicates (unique constraint on dedup_key)

        Args:
            entity_id: ID to use if creating new entity
            canonical_name: Entity name
            entity_type: PERSON, ORGANIZATION, LOCATION, EVENT
            wikidata_qid: Optional Wikidata QID for deduplication
            wikidata_label: Optional canonical name from Wikidata
            domain: Domain for publisher entities
            is_publisher: Whether this is a publisher entity
            source_type: Source classification for publishers
            base_prior: Bayesian prior for publishers
            status: Entity status (resolved, pending)
            **extra_props: Additional properties to set

        Returns:
            entity_id (may differ from input if entity already existed)
        """
        # Phase 1: If QID provided, check by QID first (true identity)
        if wikidata_qid:
            existing = await self.get_by_qid(wikidata_qid)
            if existing:
                existing_id = existing['id']
                existing_name = existing.get('canonical_name')

                # Add old canonical_name as alias if we're updating it
                if wikidata_label and existing_name and existing_name != wikidata_label:
                    await self.add_alias(existing_id, existing_name)

                # Add input canonical_name as alias if different from both
                if canonical_name and canonical_name != existing_name:
                    if not wikidata_label or canonical_name != wikidata_label:
                        await self.add_alias(existing_id, canonical_name)

                # Update canonical_name to wikidata_label if provided
                # Also increment mention count
                await self.neo4j._execute_write("""
                    MATCH (e:Entity {id: $entity_id})
                    SET e.mention_count = COALESCE(e.mention_count, 0) + 1,
                        e.canonical_name = COALESCE($wikidata_label, e.canonical_name),
                        e.wikidata_label = COALESCE($wikidata_label, e.wikidata_label),
                        e.updated_at = datetime()
                """, {
                    'entity_id': existing_id,
                    'wikidata_label': wikidata_label
                })

                new_name = wikidata_label or existing_name
                logger.debug(f"📦 Entity by QID: {canonical_name} → {new_name} [{wikidata_qid}]")
                return existing_id

        # Phase 2: MERGE on dedup_key
        # Use publisher-specific dedup_key if domain provided
        if is_publisher and domain:
            dedup_key = f"publisher_{domain.lower()}"
        else:
            dedup_key = self._compute_dedup_key(canonical_name, entity_type)

        # Build the query dynamically based on provided properties
        on_create_props = {
            'id': entity_id,
            'canonical_name': canonical_name,
            'entity_type': entity_type,
            'mention_count': 1,
            'created_at': 'datetime()'
        }

        # Add optional properties for ON CREATE
        if wikidata_qid:
            on_create_props['wikidata_qid'] = wikidata_qid
        if wikidata_label:
            on_create_props['wikidata_label'] = wikidata_label
        if domain:
            on_create_props['domain'] = domain
        if is_publisher:
            on_create_props['is_publisher'] = True
        if source_type:
            on_create_props['source_type'] = source_type
        if base_prior is not None:
            on_create_props['base_prior'] = base_prior
        if status:
            on_create_props['status'] = status

        # Build SET clause for ON CREATE
        on_create_sets = []
        params = {'dedup_key': dedup_key}

        for key, value in on_create_props.items():
            if value == 'datetime()':
                on_create_sets.append(f"e.{key} = datetime()")
            else:
                param_name = f"create_{key}"
                on_create_sets.append(f"e.{key} = ${param_name}")
                params[param_name] = value

        query = f"""
        MERGE (e:Entity {{dedup_key: $dedup_key}})
        ON CREATE SET
            {', '.join(on_create_sets)}
        ON MATCH SET
            e.mention_count = COALESCE(e.mention_count, 0) + 1,
            e.updated_at = datetime()
        RETURN e.id as id, e.wikidata_qid as existing_qid, e.canonical_name as canonical_name
        """

        result = await self.neo4j._execute_write(query, params)

        if result:
            returned_id = result['id']
            existing_qid = result.get('existing_qid')
            existing_name = result.get('canonical_name')

            # Log what happened
            if returned_id != entity_id:
                logger.debug(f"📦 Entity exists: {canonical_name} → {existing_name} [{existing_qid or 'no QID'}]")
            else:
                qid_msg = f" [{wikidata_qid}]" if wikidata_qid else ""
                logger.debug(f"✨ Entity created: {canonical_name} ({entity_type}){qid_msg}")

            return returned_id

        return entity_id

    async def get_by_qid(self, wikidata_qid: str) -> Optional[Dict]:
        """
        Get entity by Wikidata QID.

        This is the primary lookup for deduplication - if an entity with
        this QID exists, we should use it instead of creating a new one.

        Args:
            wikidata_qid: Wikidata Q-identifier

        Returns:
            Entity dict or None
        """
        query = """
        MATCH (e:Entity {wikidata_qid: $qid})
        RETURN e.id as id, e.canonical_name as canonical_name,
               e.entity_type as entity_type, e.mention_count as mention_count,
               e.wikidata_qid as wikidata_qid, e.wikidata_label as wikidata_label,
               e.domain as domain, e.is_publisher as is_publisher,
               e.source_type as source_type, e.base_prior as base_prior,
               e.aliases as aliases, e.status as status
        """
        result = await self.neo4j._execute_read(query, {'qid': wikidata_qid})
        if result and len(result) > 0:
            return dict(result[0])
        return None

    async def get_by_id(self, entity_id: str) -> Optional[Dict]:
        """
        Get entity by ID.

        Args:
            entity_id: Entity identifier

        Returns:
            Entity dict or None
        """
        query = """
        MATCH (e:Entity {id: $entity_id})
        RETURN e.id as id, e.canonical_name as canonical_name,
               e.entity_type as entity_type, e.mention_count as mention_count,
               e.wikidata_qid as wikidata_qid, e.wikidata_label as wikidata_label,
               e.domain as domain, e.is_publisher as is_publisher,
               e.source_type as source_type, e.base_prior as base_prior,
               e.aliases as aliases, e.status as status
        """
        result = await self.neo4j._execute_read(query, {'entity_id': entity_id})
        if result and len(result) > 0:
            return dict(result[0])
        return None

    async def get_by_domain(self, domain: str) -> Optional[Dict]:
        """
        Get publisher entity by domain.

        Args:
            domain: Publisher domain (e.g., 'theguardian.com')

        Returns:
            Entity dict or None
        """
        query = """
        MATCH (e:Entity {domain: $domain, is_publisher: true})
        RETURN e.id as id, e.canonical_name as canonical_name,
               e.entity_type as entity_type, e.mention_count as mention_count,
               e.wikidata_qid as wikidata_qid, e.wikidata_label as wikidata_label,
               e.domain as domain, e.is_publisher as is_publisher,
               e.source_type as source_type, e.base_prior as base_prior,
               e.aliases as aliases, e.status as status
        """
        result = await self.neo4j._execute_read(query, {'domain': domain})
        if result and len(result) > 0:
            return dict(result[0])
        return None

    async def update_qid(
        self,
        entity_id: str,
        qid: str,
        wikidata_label: str = None
    ) -> str:
        """
        Update entity with Wikidata QID and label.

        If another entity already has this QID, merge the current entity into it
        (transfer relationships and delete duplicate).

        Args:
            entity_id: Entity to update
            qid: Wikidata QID to set
            wikidata_label: Optional canonical name from Wikidata

        Returns:
            Final entity_id (may differ if merged into existing)
        """
        # Check if another entity already has this QID
        existing = await self.neo4j._execute_read("""
            MATCH (e:Entity {wikidata_qid: $qid})
            WHERE e.id <> $entity_id
            RETURN e.id as id, e.canonical_name as canonical_name
        """, {'qid': qid, 'entity_id': str(entity_id)})

        if existing and len(existing) > 0:
            # Another entity has this QID - merge into it
            target_id = existing[0]['id']
            target_name = existing[0].get('canonical_name', target_id)
            logger.info(f"🔄 QID conflict detected: {entity_id} and {target_id} both claim {qid}")

            await self.merge_entities(entity_id, target_id)

            # Also update target's canonical_name to wikidata_label if provided
            if wikidata_label and target_name != wikidata_label:
                await self.add_alias(target_id, target_name)
                await self.neo4j._execute_write("""
                    MATCH (e:Entity {id: $entity_id})
                    SET e.canonical_name = $wikidata_label,
                        e.wikidata_label = $wikidata_label,
                        e.updated_at = datetime()
                """, {
                    'entity_id': target_id,
                    'wikidata_label': wikidata_label
                })

            return target_id
        else:
            # No conflict - update QID, wikidata_label, and canonical_name
            # First, add old canonical_name as alias if it will change
            if wikidata_label:
                current = await self.get_by_id(entity_id)
                if current and current.get('canonical_name') != wikidata_label:
                    await self.add_alias(entity_id, current['canonical_name'])

            await self.neo4j._execute_write("""
                MATCH (e:Entity {id: $entity_id})
                WHERE e.wikidata_qid IS NULL
                SET e.wikidata_qid = $qid,
                    e.wikidata_label = $wikidata_label,
                    e.canonical_name = COALESCE($wikidata_label, e.canonical_name),
                    e.status = 'resolved',
                    e.updated_at = datetime()
            """, {
                'entity_id': str(entity_id),
                'qid': qid,
                'wikidata_label': wikidata_label
            })
            logger.info(f"🔗 Updated entity QID: {entity_id} → {qid} ({wikidata_label})")
            return entity_id

    async def merge_entities(self, source_id: str, target_id: str) -> None:
        """
        Merge source entity into target entity.

        Transfers all relationships from source to target:
        - MENTIONS relationships (from Claims)
        - INVOLVES relationships (from Events)
        - PUBLISHED_BY relationships (from Pages)
        - Aliases

        Then deletes the source entity.

        Args:
            source_id: Entity to merge from (will be deleted)
            target_id: Entity to merge into (will be kept)
        """
        logger.info(f"🔄 Merging entity {source_id} → {target_id}")

        # Get source entity info for alias merging
        source = await self.get_by_id(source_id)
        if not source:
            logger.warning(f"⚠️ Source entity {source_id} not found for merge")
            return

        # Transfer all relationships and merge data
        await self.neo4j._execute_write("""
            MATCH (source:Entity {id: $source_id})
            MATCH (target:Entity {id: $target_id})

            // Transfer incoming MENTIONS relationships (Claim → Entity)
            WITH source, target
            OPTIONAL MATCH (c:Claim)-[r:MENTIONS]->(source)
            FOREACH (_ IN CASE WHEN r IS NOT NULL THEN [1] ELSE [] END |
                MERGE (c)-[:MENTIONS]->(target)
            )

            // Transfer INVOLVES relationships (Event → Entity)
            WITH source, target
            OPTIONAL MATCH (ev:Event)-[r2:INVOLVES]->(source)
            FOREACH (_ IN CASE WHEN r2 IS NOT NULL THEN [1] ELSE [] END |
                MERGE (ev)-[:INVOLVES]->(target)
            )

            // Transfer PUBLISHED_BY relationships (Page → Entity)
            WITH source, target
            OPTIONAL MATCH (p:Page)-[r3:PUBLISHED_BY]->(source)
            FOREACH (_ IN CASE WHEN r3 IS NOT NULL THEN [1] ELSE [] END |
                MERGE (p)-[:PUBLISHED_BY]->(target)
            )

            // Transfer outgoing relationships (Entity → X)
            WITH source, target
            OPTIONAL MATCH (source)-[r4:RELATED_TO]->(other:Entity)
            FOREACH (_ IN CASE WHEN r4 IS NOT NULL THEN [1] ELSE [] END |
                MERGE (target)-[:RELATED_TO]->(other)
            )

            // Merge mention counts
            WITH source, target
            SET target.mention_count = COALESCE(target.mention_count, 0) + COALESCE(source.mention_count, 0)

            // Merge aliases: add source's canonical_name and aliases to target
            WITH source, target
            SET target.aliases =
                CASE
                    WHEN source.canonical_name IS NOT NULL
                         AND source.canonical_name <> target.canonical_name
                         AND NOT source.canonical_name IN COALESCE(target.aliases, [])
                    THEN COALESCE(target.aliases, []) + source.canonical_name
                    ELSE target.aliases
                END

            // Add source's aliases to target
            WITH source, target
            UNWIND COALESCE(source.aliases, []) AS alias
            WITH source, target, alias
            WHERE NOT alias IN COALESCE(target.aliases, []) AND alias <> target.canonical_name
            SET target.aliases = COALESCE(target.aliases, []) + alias

            // Update timestamp
            WITH source, target
            SET target.updated_at = datetime()

            // Delete the source entity
            WITH source
            DETACH DELETE source
        """, {
            'source_id': str(source_id),
            'target_id': str(target_id)
        })

        logger.info(f"✅ Merged entity {source_id} → {target_id}")

    async def add_alias(self, entity_id: str, alias: str) -> None:
        """
        Add an alias to an entity if not already present.

        Args:
            entity_id: Entity to add alias to
            alias: Alias to add
        """
        await self.neo4j._execute_write("""
            MATCH (e:Entity {id: $entity_id})
            WHERE NOT $alias IN COALESCE(e.aliases, [])
                  AND $alias <> e.canonical_name
            SET e.aliases = COALESCE(e.aliases, []) + $alias
        """, {
            'entity_id': str(entity_id),
            'alias': alias
        })

    async def update_publisher_classification(
        self,
        entity_id: str,
        source_type: str,
        base_prior: float
    ) -> None:
        """
        Update publisher entity with source classification.

        Args:
            entity_id: Publisher entity ID
            source_type: Classification (news_media, tabloid, wire_service, etc.)
            base_prior: Bayesian prior probability
        """
        await self.neo4j._execute_write("""
            MATCH (e:Entity {id: $entity_id})
            WHERE e.is_publisher = true
            SET e.source_type = COALESCE($source_type, e.source_type),
                e.base_prior = COALESCE($base_prior, e.base_prior),
                e.updated_at = datetime()
        """, {
            'entity_id': str(entity_id),
            'source_type': source_type,
            'base_prior': base_prior
        })
