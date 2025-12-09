"""
KnowledgeGraph - Main interface for knowledge extraction and management

Clean API for knowledge operations:
- KnowledgeGraph.extract(page) → ExtractionResult
- KnowledgeGraph.amend(page, instruction) → AmendmentResult
- KnowledgeGraph.identify(mentions) → IdentificationResult
- KnowledgeGraph.commit(page) → persist to graph
- KnowledgeGraph.query(entity) → retrieve knowledge

This is the professional interface that orchestrates all knowledge services.
Uses repositories for data persistence (not direct Neo4j writes).
"""
import uuid
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
import asyncpg

from models.domain.mention import Mention, MentionRelationship, ExtractionResult
from models.domain.entity import Entity
from utils.id_generator import generate_entity_id
from models.domain.claim import Claim
from services.neo4j_service import Neo4jService
from services.identification_service import IdentificationService, IdentificationResult
from services.extraction_amendment import ExtractionAmendmentService, AmendmentResult
from repositories.entity_repository import EntityRepository
from repositories.claim_repository import ClaimRepository
from repositories.page_repository import PageRepository
from semantic_analyzer import EnhancedSemanticAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class PageArtifact:
    """
    Artifact representing a page and its extracted knowledge.

    This is the working object that accumulates extraction results,
    amendments, and identification before being committed to the graph.
    """
    page_id: uuid.UUID
    url: str
    title: str
    content: str
    language: str = "en"
    domain: Optional[str] = None

    # Extraction results (accumulated)
    mentions: List[Mention] = field(default_factory=list)
    claims: List[Dict] = field(default_factory=list)
    relationships: List[MentionRelationship] = field(default_factory=list)

    # Identification results
    mention_to_entity: Dict[str, uuid.UUID] = field(default_factory=dict)

    # Publisher (Entity with is_publisher=true)
    publisher: Optional[Entity] = None

    # Status
    is_extracted: bool = False
    is_identified: bool = False
    is_committed: bool = False

    # Extraction quality (0.0-1.0) - rated by LLM
    # < 0.5 means too many generic terms, page should not pass to knowledge_complete
    extraction_quality: float = 0.5

    # History of operations
    operations: List[Dict] = field(default_factory=list)

    def add_mentions(self, mentions: List[Mention]):
        """Add mentions (from extraction or amendment)."""
        for m in mentions:
            # Avoid duplicates by surface_form
            existing = next((x for x in self.mentions if x.surface_form == m.surface_form), None)
            if not existing:
                self.mentions.append(m)
            else:
                # Merge aliases
                for alias in m.aliases:
                    if alias not in existing.aliases:
                        existing.aliases.append(alias)
                # Update description if better
                if m.description and not existing.description:
                    existing.description = m.description

    def add_relationships(self, relationships: List[MentionRelationship]):
        """Add relationships (from extraction or amendment)."""
        for r in relationships:
            # Avoid duplicates
            existing = next(
                (x for x in self.relationships
                 if x.subject_id == r.subject_id and x.predicate == r.predicate and x.object_id == r.object_id),
                None
            )
            if not existing:
                self.relationships.append(r)


class KnowledgeGraph:
    """
    Main interface for knowledge graph operations.

    Usage:
        kg = KnowledgeGraph(db_pool, neo4j)

        # Extract knowledge from a page
        artifact = await kg.extract(page_id)

        # Review and amend if needed
        artifact = await kg.amend(artifact, "We missed Muhammad Suleman")
        artifact = await kg.amend(artifact, "Add Blocks 1,2,4,5 with PART_OF relationships")

        # Identify entities (local + wikidata matching)
        artifact = await kg.identify(artifact)

        # Commit to graph
        await kg.commit(artifact)
    """

    def __init__(self, db_pool: asyncpg.Pool, neo4j: Neo4jService):
        self.db_pool = db_pool
        self.neo4j = neo4j
        self.analyzer = EnhancedSemanticAnalyzer()
        self.identification_service = IdentificationService(db_pool, neo4j)
        self.amendment_service = ExtractionAmendmentService()
        # Repositories for proper data persistence
        self.entity_repository = EntityRepository(db_pool, neo4j)
        self.claim_repository = ClaimRepository(db_pool, neo4j)
        self.page_repository = PageRepository(db_pool)

    async def extract(self, page_id: Union[uuid.UUID, str]) -> PageArtifact:
        """
        Extract knowledge from a page.

        Args:
            page_id: Page UUID or string

        Returns:
            PageArtifact with extracted mentions, claims, relationships
        """
        if isinstance(page_id, str):
            page_id = uuid.UUID(page_id)

        # Fetch page from database
        async with self.db_pool.acquire() as conn:
            page = await conn.fetchrow("""
                SELECT id, url, title, content_text, language, domain, site_name
                FROM core.pages WHERE id = $1
            """, page_id)

        if not page:
            raise ValueError(f"Page {page_id} not found")

        # Create artifact
        artifact = PageArtifact(
            page_id=page_id,
            url=page['url'],
            title=page['title'],
            content=page['content_text'],
            language=page['language'] or 'en',
            domain=page['domain']
        )

        # Identify publisher
        artifact.publisher = await self._identify_publisher(page['domain'], page['site_name'])

        # Run extraction
        page_meta = {
            'title': page['title'],
            'site': page['site_name'] or page['domain']
        }
        page_text = [{'selector': 'article', 'text': page['content_text']}]

        result = await self.analyzer.extract_with_mentions(
            page_meta, page_text, page['url'], artifact.language
        )

        # Populate artifact
        artifact.mentions = result.mentions
        artifact.claims = result.claims
        artifact.relationships = result.mention_relationships
        artifact.extraction_quality = result.extraction_quality
        artifact.is_extracted = True

        artifact.operations.append({
            'op': 'extract',
            'mentions': len(result.mentions),
            'claims': len(result.claims),
            'relationships': len(result.mention_relationships),
            'quality': result.extraction_quality
        })

        logger.info(f"📝 Extracted: {len(artifact.mentions)} mentions, {len(artifact.claims)} claims")

        return artifact

    async def amend(self, artifact: PageArtifact, instruction: str) -> PageArtifact:
        """
        Amend an extraction based on instruction.

        Args:
            artifact: PageArtifact to amend
            instruction: What to fix, e.g., "We missed Muhammad Suleman"

        Returns:
            Updated PageArtifact with amendments applied
        """
        result = await self.amendment_service.amend(
            instruction=instruction,
            original_content=artifact.content,
            existing_mentions=artifact.mentions,
            existing_relationships=artifact.relationships
        )

        # Apply amendments
        new_mentions = []
        new_relationships = []

        for amendment in result.amendments:
            if amendment.action == 'add_mention' and amendment.mention:
                new_mentions.append(amendment.mention)

            elif amendment.action == 'add_alias' and amendment.target_surface_form and amendment.alias:
                # Find and update existing mention
                for m in artifact.mentions:
                    if m.surface_form == amendment.target_surface_form:
                        if amendment.alias not in m.aliases:
                            m.aliases.append(amendment.alias)
                        break

            elif amendment.action == 'correct_description' and amendment.target_surface_form:
                # Find and update existing mention
                for m in artifact.mentions:
                    if m.surface_form == amendment.target_surface_form:
                        m.description = amendment.new_description
                        break

            elif amendment.action == 'add_relationship' and amendment.relationship:
                new_relationships.append(amendment.relationship)

        # Add new items to artifact
        artifact.add_mentions(new_mentions)
        artifact.add_relationships(new_relationships)

        artifact.operations.append({
            'op': 'amend',
            'instruction': instruction,
            'new_mentions': len(new_mentions),
            'new_relationships': len(new_relationships)
        })

        logger.info(f"✏️ Amended: +{len(new_mentions)} mentions, +{len(new_relationships)} relationships")

        return artifact

    async def identify(self, artifact: PageArtifact) -> PageArtifact:
        """
        Identify mentions as canonical entities.

        Resolves mentions to entity UUIDs via local graph search and Wikidata.

        Args:
            artifact: PageArtifact with mentions to identify

        Returns:
            Updated PageArtifact with mention_to_entity mapping
        """
        # Create a temporary ExtractionResult for identification service
        extraction = ExtractionResult(
            mentions=artifact.mentions,
            claims=artifact.claims,
            mention_relationships=artifact.relationships
        )

        page_context = {
            'domain': artifact.domain,
            'language': artifact.language,
            'title': artifact.title
        }

        result = await self.identification_service.identify(extraction, page_context)

        # Store mapping
        artifact.mention_to_entity = {
            mention_id: match.entity_id
            for mention_id, match in result.mention_to_entity.items()
        }
        artifact.is_identified = True

        artifact.operations.append({
            'op': 'identify',
            'matched': len(result.matched_entities),
            'new': len(result.new_entities)
        })

        logger.info(f"🔍 Identified: {len(result.matched_entities)} matched, {len(result.new_entities)} new")

        return artifact

    async def commit(self, artifact: PageArtifact) -> bool:
        """
        Commit artifact to the knowledge graph.

        Creates entities, claims, and relationships using repositories.

        Args:
            artifact: PageArtifact to commit

        Returns:
            True if successful
        """
        if not artifact.is_extracted:
            raise ValueError("Artifact not extracted yet - call extract() first")

        if not artifact.is_identified:
            # Auto-identify if not done
            artifact = await self.identify(artifact)

        # Create entities using EntityRepository
        for mention in artifact.mentions:
            entity_id = artifact.mention_to_entity.get(mention.id)
            if entity_id:
                await self._create_or_update_entity(mention, entity_id)

        # Create claims using ClaimRepository
        claims_created = 0
        for claim_data in artifact.claims:
            # Get entity IDs for this claim's mentions
            claim_entity_ids = []
            claim_entity_names = []
            for mention_key in ['who', 'where', 'what']:
                for mention_id in claim_data.get(mention_key, []):
                    entity_id = artifact.mention_to_entity.get(mention_id)
                    if entity_id:
                        claim_entity_ids.append(entity_id)
                        # Find mention name
                        for m in artifact.mentions:
                            if m.id == mention_id:
                                claim_entity_names.append(m.surface_form)
                                break

            # Create Claim domain model
            claim = Claim(
                id=uuid.uuid4(),
                page_id=artifact.page_id,
                text=claim_data.get('text', ''),
                confidence=claim_data.get('confidence', 0.8),
                modality=claim_data.get('modality', 'observation'),
                metadata={}
            )

            # Use repository (handles PostgreSQL + Neo4j + MENTIONS relationships)
            await self.claim_repository.create(claim, claim_entity_ids, claim_entity_names)
            claims_created += 1

        # Create entity-entity relationships (PART_OF, LOCATED_IN, etc.)
        for rel in artifact.relationships:
            await self._create_relationship(rel, artifact)

        # Link page to publisher
        if artifact.publisher:
            await self._link_page_to_publisher(artifact)

        # Set page status based on extraction quality (integrity barrier)
        QUALITY_THRESHOLD = 0.5
        if artifact.extraction_quality >= QUALITY_THRESHOLD:
            await self.page_repository.update_status(artifact.page_id, 'knowledge_complete')
            logger.info(f"✅ Page {artifact.page_id} marked as knowledge_complete (quality: {artifact.extraction_quality})")
        else:
            await self.page_repository.update_status(artifact.page_id, 'knowledge_low_quality')
            logger.warning(f"⚠️ Page {artifact.page_id} marked as knowledge_low_quality (quality: {artifact.extraction_quality} < {QUALITY_THRESHOLD})")

        artifact.is_committed = True

        artifact.operations.append({
            'op': 'commit',
            'entities': len(artifact.mention_to_entity),
            'claims': claims_created,
            'relationships': len(artifact.relationships)
        })

        logger.info(f"💾 Committed: {len(artifact.mention_to_entity)} entities, {claims_created} claims, {len(artifact.relationships)} relationships")
        logger.info(f"✅ Page {artifact.page_id} marked as knowledge_complete")

        return True

    async def query_entity(self, name: str, entity_type: str = None) -> Optional[Entity]:
        """
        Query an entity from the graph.

        Args:
            name: Entity name to search
            entity_type: Optional type filter

        Returns:
            Entity if found
        """
        return await self.entity_repository.get_by_canonical_name(name, entity_type)

    async def _identify_publisher(self, domain: str, site_name: str) -> Entity:
        """Identify or create publisher entity (ORGANIZATION with is_publisher=true)."""
        # Check existing publisher by domain
        results = await self.neo4j._execute_read("""
            MATCH (e:Entity {domain: $domain, is_publisher: true})
            RETURN e.id as id, e.canonical_name as canonical_name,
                   e.wikidata_qid as wikidata_qid, e.status as status
        """, {'domain': domain})

        if results:
            row = results[0]
            return Entity(
                id=row['id'],
                canonical_name=row['canonical_name'],
                entity_type="ORGANIZATION",
                wikidata_qid=row.get('wikidata_qid'),
                status=row.get('status', 'pending')
            )

        # Create new publisher entity
        entity_id = generate_entity_id()
        publisher = Entity(
            id=entity_id,
            canonical_name=site_name or domain,
            entity_type="ORGANIZATION",
            status='pending'
        )

        # Create Entity node with is_publisher=true and domain for lookup
        dedup_key = f"publisher_{domain.lower()}"
        await self.neo4j._execute_write("""
            MERGE (e:Entity {dedup_key: $dedup_key})
            ON CREATE SET
                e.id = $id,
                e.canonical_name = $name,
                e.entity_type = 'ORGANIZATION',
                e.domain = $domain,
                e.is_publisher = true,
                e.mention_count = 1,
                e.status = 'pending',
                e.created_at = datetime()
            ON MATCH SET
                e.mention_count = e.mention_count + 1,
                e.updated_at = datetime()
        """, {
            'id': entity_id,
            'dedup_key': dedup_key,
            'domain': domain,
            'name': publisher.canonical_name
        })

        return publisher

    async def _create_or_update_entity(self, mention: Mention, entity_id: uuid.UUID):
        """Create or update entity using EntityRepository."""
        # Normalize type
        entity_type = mention.type_hint
        if entity_type in ('GPE', 'LOC'):
            entity_type = 'LOCATION'
        elif entity_type == 'ORG':
            entity_type = 'ORGANIZATION'

        # Create Entity domain model
        entity = Entity(
            id=entity_id,
            canonical_name=mention.surface_form,
            entity_type=entity_type,
            profile_summary=mention.description,
            aliases=mention.aliases
        )

        # Use repository (handles MERGE, mention_count, deduplication)
        await self.entity_repository.create(entity)

        # Update profile if we have a description (repository.create doesn't set this)
        if mention.description:
            await self.entity_repository.update_profile(entity_id, mention.description)

    async def _create_relationship(self, rel: MentionRelationship, artifact: PageArtifact):
        """Create relationship in Neo4j."""
        # Resolve mention IDs to entity IDs
        # For amendments, subject_id might be a surface_form instead of mention ID
        subject_id = None
        object_id = None

        # Try to find by mention ID first
        subject_id = artifact.mention_to_entity.get(rel.subject_id)
        object_id = artifact.mention_to_entity.get(rel.object_id)

        # If not found, try to find by surface_form (for amendments)
        if not subject_id:
            for m in artifact.mentions:
                if m.surface_form == rel.subject_id:
                    subject_id = artifact.mention_to_entity.get(m.id)
                    break

        if not object_id:
            for m in artifact.mentions:
                if m.surface_form == rel.object_id:
                    object_id = artifact.mention_to_entity.get(m.id)
                    break

        if not subject_id or not object_id:
            logger.warning(f"Cannot create relationship: {rel.subject_id} -> {rel.object_id}")
            return

        valid_predicates = {'PART_OF', 'LOCATED_IN', 'WORKS_FOR', 'MEMBER_OF', 'AFFILIATED_WITH'}
        if rel.predicate not in valid_predicates:
            return

        await self.neo4j._execute_write(f"""
            MATCH (s:Entity {{id: $subject_id}})
            MATCH (o:Entity {{id: $object_id}})
            MERGE (s)-[r:{rel.predicate}]->(o)
            ON CREATE SET r.created_at = datetime()
        """, {
            'subject_id': str(subject_id),
            'object_id': str(object_id)
        })

    async def _link_page_to_publisher(self, artifact: PageArtifact):
        """Link page to publisher entity in Neo4j."""
        # Ensure page node exists
        await self.neo4j._execute_write("""
            MERGE (p:Page {id: $page_id})
            ON CREATE SET p.url = $url, p.title = $title
        """, {
            'page_id': str(artifact.page_id),
            'url': artifact.url,
            'title': artifact.title
        })

        # Link to publisher entity
        await self.neo4j._execute_write("""
            MATCH (p:Page {id: $page_id})
            MATCH (e:Entity {id: $publisher_id})
            MERGE (p)-[r:PUBLISHED_BY]->(e)
            ON CREATE SET r.created_at = datetime()
        """, {
            'page_id': str(artifact.page_id),
            'publisher_id': artifact.publisher.id
        })
