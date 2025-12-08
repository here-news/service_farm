"""
IdentificationService - Resolve mentions to canonical entities

This service takes extraction output (mentions with context) and resolves
each mention to a canonical entity in the knowledge graph.

Pipeline:
  ExtractionResult → IdentificationService → IdentificationResult
                                                  ↓
                                         mention_id → entity_uuid

Strategy for each mention:
1. Search local graph (fuzzy match on name + aliases + wikidata_label)
2. Search Wikidata with context (if no local match)
3. Apply relationship constraints for disambiguation
4. Decide: reuse existing entity | create with QID | create local-only
"""
import uuid
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import asyncpg

from models.mention import Mention, MentionRelationship, ExtractionResult
from models.entity import Entity
from repositories.entity_repository import EntityRepository
from services.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)


@dataclass
class EntityMatch:
    """Result of identifying a single mention."""
    entity_id: uuid.UUID
    canonical_name: str
    entity_type: str
    confidence: float
    source: str  # "local_exact", "local_fuzzy", "wikidata", "new_local"
    wikidata_qid: Optional[str] = None
    is_new: bool = False


@dataclass
class IdentificationResult:
    """Complete result from identification stage."""
    # Mapping: mention_id → EntityMatch
    mention_to_entity: Dict[str, EntityMatch] = field(default_factory=dict)

    # New entities that need to be created
    new_entities: List[Entity] = field(default_factory=list)

    # Entities that were matched to existing
    matched_entities: List[Tuple[str, uuid.UUID]] = field(default_factory=list)

    # Statistics
    stats: Dict[str, int] = field(default_factory=dict)

    def get_entity_id(self, mention_id: str) -> Optional[uuid.UUID]:
        """Get entity UUID for a mention."""
        match = self.mention_to_entity.get(mention_id)
        return match.entity_id if match else None


class IdentificationService:
    """
    Service to identify mentions as canonical entities.

    Uses local graph search + Wikidata for entity resolution.
    Context and relationships help disambiguate.
    """

    # Type mapping from extraction hints to entity types
    TYPE_MAPPING = {
        'PERSON': 'PERSON',
        'ORGANIZATION': 'ORGANIZATION',
        'ORG': 'ORGANIZATION',
        'LOCATION': 'LOCATION',
        'GPE': 'LOCATION',
        'LOC': 'LOCATION',
    }

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        neo4j_service: Neo4jService,
        wikidata_client=None  # Optional WikidataClient for enrichment
    ):
        self.db_pool = db_pool
        self.neo4j = neo4j_service
        self.entity_repo = EntityRepository(db_pool, neo4j_service)
        self.wikidata = wikidata_client

    async def identify(
        self,
        extraction: ExtractionResult,
        page_context: Optional[Dict] = None
    ) -> IdentificationResult:
        """
        Identify all mentions in an extraction result.

        Args:
            extraction: ExtractionResult with mentions and relationships
            page_context: Optional page metadata (domain, language) for hints

        Returns:
            IdentificationResult with mention → entity mapping
        """
        result = IdentificationResult()
        result.stats = {
            "total_mentions": len(extraction.mentions),
            "local_exact": 0,
            "local_fuzzy": 0,
            "wikidata": 0,
            "new_local": 0,
            "skipped": 0
        }

        # Build relationship constraints map
        constraints = self._build_constraints(extraction.mention_relationships)

        # Process each mention
        for mention in extraction.mentions:
            try:
                match = await self._identify_mention(
                    mention,
                    constraints.get(mention.id, []),
                    page_context
                )

                if match:
                    result.mention_to_entity[mention.id] = match
                    result.stats[match.source] += 1

                    if match.is_new:
                        # Create new entity
                        entity = Entity(
                            id=match.entity_id,
                            canonical_name=match.canonical_name,
                            entity_type=match.entity_type,
                            wikidata_qid=match.wikidata_qid,
                            aliases=mention.aliases,
                            profile_summary=mention.description,
                            mention_count=1,
                            status='pending' if not match.wikidata_qid else 'enriched',
                            confidence=match.confidence
                        )
                        result.new_entities.append(entity)
                    else:
                        result.matched_entities.append((mention.id, match.entity_id))
                else:
                    result.stats["skipped"] += 1

            except Exception as e:
                logger.error(f"Failed to identify mention {mention.id}: {e}")
                result.stats["skipped"] += 1

        logger.info(
            f"✅ Identification complete: {result.stats['local_exact']} exact, "
            f"{result.stats['local_fuzzy']} fuzzy, {result.stats['wikidata']} wikidata, "
            f"{result.stats['new_local']} new"
        )

        return result

    async def _identify_mention(
        self,
        mention: Mention,
        constraints: List[str],
        page_context: Optional[Dict]
    ) -> Optional[EntityMatch]:
        """
        Identify a single mention.

        Strategy:
        1. Exact local match (canonical_name + type)
        2. Fuzzy local match (85% threshold, checks aliases too)
        3. Wikidata search with context
        4. Create new local entity
        """
        entity_type = self.TYPE_MAPPING.get(mention.type_hint, 'LOCATION')

        # Skip very short or generic names
        if len(mention.surface_form.strip()) < 2:
            logger.debug(f"Skipping short mention: {mention.surface_form}")
            return None

        # 1. Try exact local match
        existing = await self.entity_repo.get_by_canonical_name(
            mention.surface_form,
            entity_type
        )
        if existing:
            qid = existing.wikidata_qid

            # Enrich with Wikidata if entity lacks QID
            if not qid and self.wikidata and not self._is_generic_entity(mention.surface_form, entity_type):
                wikidata_match = await self._search_wikidata(mention, entity_type, page_context)
                if wikidata_match:
                    qid = wikidata_match['qid']
                    # Update entity with QID (will be saved when entity is created/updated)
                    logger.info(f"🔗 Enriching existing entity: {existing.canonical_name} → {qid}")

            logger.debug(f"🎯 Exact match: {mention.surface_form} → {existing.id}")
            return EntityMatch(
                entity_id=existing.id,
                canonical_name=existing.canonical_name,
                entity_type=existing.entity_type,
                confidence=1.0,
                source="local_exact",
                wikidata_qid=qid,
                is_new=False
            )

        # Also check aliases
        for alias in mention.aliases:
            existing = await self.entity_repo.get_by_canonical_name(alias, entity_type)
            if existing:
                logger.debug(f"🎯 Alias match: {alias} → {existing.id}")
                # Add the new surface form as alias
                existing.add_alias(mention.surface_form)
                return EntityMatch(
                    entity_id=existing.id,
                    canonical_name=existing.canonical_name,
                    entity_type=existing.entity_type,
                    confidence=0.95,
                    source="local_exact",
                    wikidata_qid=existing.wikidata_qid,
                    is_new=False
                )

        # 2. Try fuzzy local match
        similar = await self.entity_repo.find_similar_entity(
            canonical_name=mention.surface_form,
            entity_type=entity_type,
            threshold=85.0
        )
        if similar:
            entity, score = similar
            logger.debug(f"🔗 Fuzzy match ({score:.0f}%): {mention.surface_form} → {entity.canonical_name}")
            return EntityMatch(
                entity_id=entity.id,
                canonical_name=entity.canonical_name,
                entity_type=entity.entity_type,
                confidence=score / 100.0,
                source="local_fuzzy",
                wikidata_qid=entity.wikidata_qid,
                is_new=False
            )

        # 3. Try Wikidata search (if client available and not generic)
        wikidata_match = None
        if self.wikidata and not self._is_generic_entity(mention.surface_form, entity_type):
            wikidata_match = await self._search_wikidata(
                mention,
                entity_type,
                page_context
            )

        if wikidata_match:
            # Check if this QID already exists in our graph
            existing_by_qid = await self._find_by_qid(wikidata_match['qid'])
            if existing_by_qid:
                logger.debug(f"🔗 QID match: {mention.surface_form} → {existing_by_qid.canonical_name} ({wikidata_match['qid']})")
                return EntityMatch(
                    entity_id=existing_by_qid.id,
                    canonical_name=existing_by_qid.canonical_name,
                    entity_type=existing_by_qid.entity_type,
                    confidence=wikidata_match['confidence'],
                    source="wikidata",
                    wikidata_qid=wikidata_match['qid'],
                    is_new=False
                )
            else:
                # New entity with Wikidata QID
                logger.debug(f"✨ Wikidata new: {mention.surface_form} → {wikidata_match['qid']}")
                return EntityMatch(
                    entity_id=uuid.uuid4(),
                    canonical_name=wikidata_match.get('label', mention.surface_form),
                    entity_type=entity_type,
                    confidence=wikidata_match['confidence'],
                    source="wikidata",
                    wikidata_qid=wikidata_match['qid'],
                    is_new=True
                )

        # 4. Create new local entity (no Wikidata match)
        logger.debug(f"✨ New local: {mention.surface_form}")
        return EntityMatch(
            entity_id=uuid.uuid4(),
            canonical_name=mention.surface_form,
            entity_type=entity_type,
            confidence=0.7,  # Lower confidence without Wikidata
            source="new_local",
            wikidata_qid=None,
            is_new=True
        )

    def _build_constraints(
        self,
        relationships: List[MentionRelationship]
    ) -> Dict[str, List[str]]:
        """
        Build constraint map from relationships.

        Constraints help with disambiguation:
        - If X PART_OF Y, X must be containable (room, floor, building)
        - If X WORKS_FOR Y, X must be person, Y must be organization
        """
        constraints = {}

        for rel in relationships:
            # Subject constraints
            if rel.subject_id not in constraints:
                constraints[rel.subject_id] = []

            if rel.predicate == "PART_OF":
                constraints[rel.subject_id].append("is_containable")
            elif rel.predicate == "WORKS_FOR":
                constraints[rel.subject_id].append("is_person")
            elif rel.predicate == "MEMBER_OF":
                constraints[rel.subject_id].append("is_person")

            # Object constraints
            if rel.object_id not in constraints:
                constraints[rel.object_id] = []

            if rel.predicate == "PART_OF":
                constraints[rel.object_id].append("is_container")
            elif rel.predicate in ("WORKS_FOR", "MEMBER_OF"):
                constraints[rel.object_id].append("is_organization")

        return constraints

    def _is_generic_entity(self, name: str, entity_type: str) -> bool:
        """
        Check if entity name is too generic for Wikidata search.

        Generic names like "Block 6", "Floor 3" shouldn't be searched
        as they'll match wrong entities.
        """
        name_lower = name.lower().strip()
        words = name_lower.split()

        # Building/location components without proper names
        generic_patterns = ['block', 'floor', 'room', 'unit', 'level', 'section', 'building']

        if entity_type == 'LOCATION' and len(words) == 2:
            if words[0] in generic_patterns and (words[1].isdigit() or len(words[1]) == 1):
                return True

        # Single letters or numbers
        if len(name_lower) <= 2:
            return True

        return False

    async def _search_wikidata(
        self,
        mention: Mention,
        entity_type: str,
        page_context: Optional[Dict]
    ) -> Optional[Dict]:
        """
        Search Wikidata for entity match.

        Uses mention context and description for better matching.
        """
        if not self.wikidata:
            return None

        try:
            # Use only static description for entity identification
            # Dynamic context (what's happening TO the entity) confuses matching
            # e.g., "fire broke out" matches fire EVENT not the BUILDING
            search_context = mention.description or ""

            logger.info(f"🔍 Wikidata search: {mention.surface_form} | desc={search_context[:80] if search_context else 'EMPTY'}")

            # Call Wikidata search
            result = await self.wikidata.search_entity(
                name=mention.surface_form,
                entity_type=entity_type,
                context=search_context,
                aliases=mention.aliases
            )

            return result

        except Exception as e:
            logger.warning(f"Wikidata search failed for {mention.surface_form}: {e}")
            return None

    async def _find_by_qid(self, qid: str) -> Optional[Entity]:
        """Find existing entity by Wikidata QID."""
        try:
            results = await self.neo4j._execute_read("""
                MATCH (e:Entity {wikidata_qid: $qid})
                RETURN e.id as id,
                       e.canonical_name as canonical_name,
                       e.entity_type as entity_type,
                       e.wikidata_qid as wikidata_qid
                LIMIT 1
            """, {'qid': qid})

            if results:
                row = results[0]
                return Entity(
                    id=uuid.UUID(row['id']),
                    canonical_name=row['canonical_name'],
                    entity_type=row['entity_type'],
                    wikidata_qid=row['wikidata_qid']
                )
            return None

        except Exception as e:
            logger.error(f"Failed to find entity by QID {qid}: {e}")
            return None
