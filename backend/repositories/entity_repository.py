"""
Entity Repository - Neo4j primary storage

Storage strategy:
- Neo4j: Primary entity storage (canonical_name, type, mention_count, created_at)
- PostgreSQL: Enrichment data only (Wikidata descriptions, external IDs)

Entities are fundamentally graph nodes, so Neo4j is the source of truth.
PostgreSQL only stores large text fields (descriptions) from enrichment workers.

Local Dedup Strategy:
- Before creating new entity, search graph for similar entities
- Use normalized string matching (case-insensitive, strip punctuation)
- Use fuzzy matching (>90% similarity) for near-duplicates
- This prevents duplicates at creation time, before Wikidata enrichment
"""
import uuid
import logging
import re
from typing import Optional, List, Tuple
import asyncpg
from rapidfuzz import fuzz

from models.entity import Entity
from services.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)


class EntityRepository:
    """
    Repository for Entity domain model

    Neo4j is the primary storage for entities.
    PostgreSQL enrichment handled separately.
    """

    def __init__(self, db_pool: asyncpg.Pool, neo4j_service: Neo4jService):
        self.db_pool = db_pool
        self.neo4j = neo4j_service

    async def create(self, entity: Entity) -> Entity:
        """
        Create or get existing entity in Neo4j (primary storage).

        DEDUPLICATION:
        - If QID provided: MERGE on QID → may return existing entity's ID
        - If no QID: MERGE on (canonical_name, entity_type)

        This means the returned entity may have a DIFFERENT ID than the input
        if an entity with the same QID already exists.

        Args:
            entity: Entity domain model

        Returns:
            Entity with correct ID (may differ from input if deduplicated)
        """
        # Create/get entity in Neo4j (may return existing entity's ID)
        returned_id = await self.neo4j.create_or_update_entity(
            entity_id=str(entity.id),
            canonical_name=entity.canonical_name,
            entity_type=entity.entity_type,
            wikidata_qid=entity.wikidata_qid
        )

        # If Neo4j returned a different ID, this entity already existed
        if returned_id != str(entity.id):
            logger.info(f"🔗 Entity deduplicated: {entity.canonical_name} → existing {returned_id}")
            entity.id = uuid.UUID(returned_id)

        qid_msg = f" [{entity.wikidata_qid}]" if entity.wikidata_qid else ""
        logger.debug(f"📦 Entity in Neo4j: {entity.canonical_name} ({entity.entity_type}){qid_msg} → {entity.id}")
        return entity

    async def get_by_id(self, entity_id: uuid.UUID) -> Optional[Entity]:
        """
        Retrieve entity by ID from Neo4j

        Args:
            entity_id: Entity UUID

        Returns:
            Entity model or None
        """
        entity_data = await self.neo4j.get_entity_by_id(entity_id=str(entity_id))

        if not entity_data:
            return None

        # Convert Neo4j data to Entity model
        return Entity(
            id=uuid.UUID(entity_data['id']),
            canonical_name=entity_data['canonical_name'],
            entity_type=entity_data['entity_type'],
            mention_count=entity_data.get('mention_count', 0),
            profile_summary=entity_data.get('profile_summary'),
            wikidata_qid=entity_data.get('wikidata_qid'),
            wikidata_label=entity_data.get('wikidata_label'),
            wikidata_description=entity_data.get('wikidata_description'),
            status=entity_data.get('status', 'pending'),
            confidence=entity_data.get('confidence', 0.0),
            aliases=entity_data.get('aliases', []),
            metadata={}
        )

    async def get_by_ids(self, entity_ids: List[uuid.UUID]) -> List[Entity]:
        """
        Retrieve multiple entities by IDs from Neo4j

        Args:
            entity_ids: List of Entity UUIDs

        Returns:
            List of Entity models
        """
        if not entity_ids:
            return []

        # Convert UUIDs to strings
        id_strings = [str(eid) for eid in entity_ids]

        # Query Neo4j
        entities_data = await self.neo4j.get_entities_by_ids(entity_ids=id_strings)

        # Convert to Entity models
        entities = []
        for entity_data in entities_data:
            entities.append(Entity(
                id=uuid.UUID(entity_data['id']),
                canonical_name=entity_data['canonical_name'],
                entity_type=entity_data['entity_type'],
                mention_count=entity_data.get('mention_count', 0),
                profile_summary=entity_data.get('profile_summary'),
                wikidata_qid=entity_data.get('wikidata_qid'),
                wikidata_label=entity_data.get('wikidata_label'),
                wikidata_description=entity_data.get('wikidata_description'),
                status=entity_data.get('status', 'pending'),
                confidence=entity_data.get('confidence', 0.0),
                aliases=entity_data.get('aliases', []),
                metadata={}
            ))

        return entities

    async def get_by_event_id(self, event_id: uuid.UUID) -> List[Entity]:
        """
        Retrieve all entities for an event from Neo4j

        Args:
            event_id: Event UUID

        Returns:
            List of Entity models involved in the event
        """
        # Query Neo4j for entities linked to this event
        results = await self.neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[:INVOLVES]->(entity:Entity)
            RETURN entity.id as id,
                   entity.canonical_name as canonical_name,
                   entity.entity_type as entity_type,
                   entity.mention_count as mention_count,
                   entity.confidence as confidence,
                   entity.wikidata_qid as wikidata_qid,
                   entity.wikidata_label as wikidata_label,
                   entity.wikidata_description as wikidata_description,
                   entity.wikidata_image as wikidata_image,
                   entity.profile_summary as profile_summary,
                   entity.status as status,
                   entity.aliases as aliases
            ORDER BY entity.canonical_name
        """, {'event_id': str(event_id)})

        # Convert to Entity models
        entities = []
        for row in results:
            entities.append(Entity(
                id=uuid.UUID(row['id']),
                canonical_name=row['canonical_name'],
                entity_type=row['entity_type'],
                mention_count=row.get('mention_count', 0),
                profile_summary=row.get('profile_summary'),
                wikidata_qid=row.get('wikidata_qid'),
                wikidata_label=row.get('wikidata_label'),
                wikidata_description=row.get('wikidata_description'),
                wikidata_image=row.get('wikidata_image'),
                status=row.get('status', 'pending'),
                confidence=row.get('confidence', 0.0),
                aliases=row.get('aliases', []),
                metadata={}
            ))

        return entities

    async def get_by_canonical_name(
        self, canonical_name: str, entity_type: Optional[str] = None
    ) -> Optional[Entity]:
        """
        Retrieve entity by canonical name from Neo4j

        Args:
            canonical_name: Entity canonical name
            entity_type: Required entity type

        Returns:
            Entity model or None
        """
        if not entity_type:
            logger.warning("get_by_canonical_name requires entity_type for Neo4j queries")
            return None

        # Query Neo4j
        entity_data = await self.neo4j.get_entity_by_name_and_type(
            canonical_name=canonical_name,
            entity_type=entity_type
        )

        if not entity_data:
            return None

        # Convert Neo4j data to Entity model
        return Entity(
            id=uuid.UUID(entity_data['id']),
            canonical_name=entity_data['canonical_name'],
            entity_type=entity_data['entity_type'],
            mention_count=entity_data.get('mention_count', 0),
            wikidata_qid=entity_data.get('wikidata_qid'),
            status=entity_data.get('status', 'pending'),
            aliases=[],
            metadata={}
        )

    @staticmethod
    def _normalize_name(name: str) -> str:
        """
        Normalize entity name for matching:
        - lowercase
        - remove possessives ('s, 's)
        - remove extra whitespace
        - strip punctuation except hyphens
        """
        normalized = name.lower()
        # Remove possessives
        normalized = re.sub(r"['']s\b", "", normalized)
        # Remove punctuation except hyphens (keep "Hong Kong" structure)
        normalized = re.sub(r"[^\w\s\-]", "", normalized)
        # Collapse whitespace
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    async def find_similar_entity(
        self, canonical_name: str, entity_type: str, threshold: float = 85.0
    ) -> Optional[Tuple[Entity, float]]:
        """
        Find most similar existing entity in graph using fuzzy matching.

        Strategy:
        1. Normalize input name
        2. Fetch all entities of same type from Neo4j (including aliases, wikidata_label)
        3. Match against canonical_name, wikidata_label, AND aliases
        4. Use token_set_ratio for PERSON (handles extra name parts)
        5. Return best match if above threshold

        This enables local matching after Wikidata enrichment:
        - "Michael Mo Kwan Tai" matches entity with canonical_name "Michael Mo"
          because wikidata_label "Michael Mo Kwan Tai" is stored and checked

        Args:
            canonical_name: Name to match
            entity_type: Entity type (PERSON, ORGANIZATION, LOCATION)
            threshold: Minimum similarity score (0-100), default 85

        Returns:
            Tuple of (matching Entity, similarity score) or None if no match
        """
        normalized_input = self._normalize_name(canonical_name)

        # Fetch all entities of this type from Neo4j, including enrichment data
        # TODO: For large graphs, consider using Neo4j full-text search index
        results = await self.neo4j._execute_read("""
            MATCH (e:Entity {entity_type: $entity_type})
            RETURN e.id as id,
                   e.canonical_name as canonical_name,
                   e.entity_type as entity_type,
                   e.mention_count as mention_count,
                   e.wikidata_qid as wikidata_qid,
                   e.wikidata_label as wikidata_label,
                   e.aliases as aliases,
                   e.status as status
            LIMIT 1000
        """, {'entity_type': entity_type})

        if not results:
            return None

        best_match = None
        best_score = 0.0
        best_match_source = None  # Track what matched (canonical, wikidata_label, alias)

        for row in results:
            # Collect all names to check: canonical_name, wikidata_label, aliases
            names_to_check = [row['canonical_name']]

            if row.get('wikidata_label'):
                names_to_check.append(row['wikidata_label'])

            if row.get('aliases'):
                names_to_check.extend(row['aliases'])

            for name_variant in names_to_check:
                if not name_variant:
                    continue

                normalized_existing = self._normalize_name(name_variant)

                # Exact normalized match is 100%
                if normalized_input == normalized_existing:
                    entity = Entity(
                        id=uuid.UUID(row['id']),
                        canonical_name=row['canonical_name'],
                        entity_type=row['entity_type'],
                        mention_count=row.get('mention_count', 0),
                        wikidata_qid=row.get('wikidata_qid'),
                        status=row.get('status', 'pending'),
                        aliases=row.get('aliases', []),
                        metadata={}
                    )
                    match_source = "canonical" if name_variant == row['canonical_name'] else (
                        "wikidata_label" if name_variant == row.get('wikidata_label') else "alias"
                    )
                    logger.debug(f"🔗 Exact match via {match_source}: '{canonical_name}' → '{row['canonical_name']}'")
                    return (entity, 100.0)

                # For PERSON entities, use token_set_ratio (handles extra name parts)
                # "Michael Mo" matches "Michael Mo Kwan Tai" (100% - all tokens present)
                if entity_type == 'PERSON':
                    score = fuzz.token_set_ratio(normalized_input, normalized_existing)
                else:
                    # For non-PERSON, use token_sort_ratio (word order independence)
                    score = fuzz.token_sort_ratio(normalized_input, normalized_existing)

                if score > best_score:
                    best_score = score
                    best_match = row
                    best_match_source = "canonical" if name_variant == row['canonical_name'] else (
                        "wikidata_label" if name_variant == row.get('wikidata_label') else "alias"
                    )

        if best_match and best_score >= threshold:
            entity = Entity(
                id=uuid.UUID(best_match['id']),
                canonical_name=best_match['canonical_name'],
                entity_type=best_match['entity_type'],
                mention_count=best_match.get('mention_count', 0),
                wikidata_qid=best_match.get('wikidata_qid'),
                status=best_match.get('status', 'pending'),
                aliases=best_match.get('aliases', []),
                metadata={}
            )
            logger.info(f"🔗 Fuzzy match via {best_match_source} ({best_score:.0f}%): '{canonical_name}' → '{entity.canonical_name}'")
            return (entity, best_score)

        return None

    async def increment_mention_count(self, entity_id: uuid.UUID) -> int:
        """
        Increment mention count (handled automatically by Neo4j MERGE)

        This is a no-op since Neo4j create_or_update_entity increments on MATCH

        Args:
            entity_id: Entity UUID

        Returns:
            Updated mention count (always returns 0 as placeholder)
        """
        # Neo4j handles mention_count increment automatically in create_or_update_entity
        # This method exists for compatibility but is a no-op
        logger.debug(f"Mention count for {entity_id} incremented via Neo4j MERGE")
        return 0

    async def enrich(
        self,
        entity_id: uuid.UUID,
        wikidata_qid: str,
        wikidata_label: str,
        wikidata_description: str,
        confidence: float,
        aliases: list = None,
        metadata: dict = None
    ) -> None:
        """
        Enrich entity with Wikidata information

        Args:
            entity_id: Entity UUID
            wikidata_qid: Wikidata QID (e.g., 'Q123')
            wikidata_label: Official Wikidata label
            wikidata_description: Wikidata description
            confidence: Confidence score of enrichment
            aliases: List of alternative names
            metadata: Additional metadata (thumbnail, coordinates, etc.)
        """
        await self.neo4j.enrich_entity(
            entity_id=str(entity_id),
            wikidata_qid=wikidata_qid,
            wikidata_label=wikidata_label,
            wikidata_description=wikidata_description,
            confidence=confidence,
            aliases=aliases or [],
            metadata=metadata or {}
        )
        logger.info(f"📦 Enriched entity {entity_id} with Wikidata QID {wikidata_qid}")

    async def mark_checked(self, entity_id: uuid.UUID) -> None:
        """
        Mark entity as checked (no Wikidata match found)

        Args:
            entity_id: Entity UUID
        """
        await self.neo4j.mark_entity_checked(entity_id=str(entity_id))
        logger.debug(f"✓ Marked entity {entity_id} as checked")

    async def update_profile(self, entity_id: uuid.UUID, profile_summary: str) -> None:
        """
        Update entity profile summary (AI-generated from claim contexts)

        Args:
            entity_id: Entity UUID
            profile_summary: Generated description
        """
        await self.neo4j.update_entity_profile(
            entity_id=str(entity_id),
            profile_summary=profile_summary
        )
        logger.debug(f"📝 Updated profile for entity {entity_id}")

    async def find_and_merge_duplicates(self) -> dict:
        """
        Find and merge duplicate entities based on normalized names.

        Strategy:
        1. Fetch all entities from Neo4j
        2. Group by (normalized_name, entity_type)
        3. For each group with >1 entities:
           - Keep first entity as canonical
           - Transfer all relationships from duplicates to canonical
           - Delete duplicate entities

        Returns:
            Dict with merge statistics
        """
        stats = {'groups_found': 0, 'entities_merged': 0, 'relationships_transferred': 0}

        # Fetch all entities
        results = await self.neo4j._execute_read("""
            MATCH (e:Entity)
            RETURN e.id as id,
                   e.canonical_name as canonical_name,
                   e.entity_type as entity_type,
                   e.mention_count as mention_count
            ORDER BY e.mention_count DESC
        """, {})

        if not results:
            return stats

        # Group by normalized name + type
        from collections import defaultdict
        groups = defaultdict(list)

        for row in results:
            normalized = self._normalize_name(row['canonical_name'])
            key = (normalized, row['entity_type'])
            groups[key].append(row)

        # Process groups with duplicates
        for key, entities in groups.items():
            if len(entities) <= 1:
                continue

            stats['groups_found'] += 1

            # Keep first entity (highest mention_count due to ORDER BY)
            canonical = entities[0]
            duplicates = entities[1:]

            logger.info(f"🔄 Merging {len(duplicates)} duplicates → '{canonical['canonical_name']}'")

            for dup in duplicates:
                # Transfer all relationships from duplicate to canonical
                # This includes: INVOLVES (events), MENTIONED_IN (claims), etc.
                transfer_result = await self.neo4j._execute_write("""
                    MATCH (dup:Entity {id: $dup_id})
                    MATCH (canonical:Entity {id: $canonical_id})

                    // Transfer incoming relationships
                    OPTIONAL MATCH (dup)<-[r_in]->(other)
                    WHERE NOT other:Entity OR other.id <> $canonical_id
                    WITH dup, canonical, collect({rel: r_in, other: other}) as in_rels

                    // For each relationship, create equivalent to canonical
                    UNWIND in_rels as rel_data
                    WITH dup, canonical, rel_data
                    WHERE rel_data.rel IS NOT NULL
                    CALL apoc.do.when(
                        rel_data.rel IS NOT NULL,
                        'MERGE (canonical)<-[new_rel:' + type(rel_data.rel) + ']-(other)
                         SET new_rel = properties(old_rel)
                         RETURN 1 as count',
                        'RETURN 0 as count',
                        {canonical: canonical, other: rel_data.other, old_rel: rel_data.rel}
                    ) YIELD value

                    WITH dup, canonical, count(value) as transferred

                    // Delete duplicate entity and its relationships
                    DETACH DELETE dup

                    RETURN transferred
                """, {
                    'dup_id': dup['id'],
                    'canonical_id': canonical['id']
                })

                stats['entities_merged'] += 1
                logger.info(f"   ✓ Merged '{dup['canonical_name']}' → '{canonical['canonical_name']}'")

        return stats

    async def find_duplicates_preview(self) -> List[dict]:
        """
        Preview duplicate entities without merging.

        Returns:
            List of duplicate groups with entity details
        """
        results = await self.neo4j._execute_read("""
            MATCH (e:Entity)
            RETURN e.id as id,
                   e.canonical_name as canonical_name,
                   e.entity_type as entity_type,
                   e.mention_count as mention_count,
                   e.wikidata_qid as wikidata_qid
            ORDER BY e.canonical_name
        """, {})

        if not results:
            return []

        # Group by normalized name + type
        from collections import defaultdict
        groups = defaultdict(list)

        for row in results:
            normalized = self._normalize_name(row['canonical_name'])
            key = (normalized, row['entity_type'])
            groups[key].append({
                'id': row['id'],
                'canonical_name': row['canonical_name'],
                'entity_type': row['entity_type'],
                'mention_count': row.get('mention_count', 0),
                'wikidata_qid': row.get('wikidata_qid')
            })

        # Return only groups with duplicates
        duplicates = []
        for key, entities in groups.items():
            if len(entities) > 1:
                duplicates.append({
                    'normalized_name': key[0],
                    'entity_type': key[1],
                    'count': len(entities),
                    'entities': entities
                })

        return sorted(duplicates, key=lambda x: -x['count'])
