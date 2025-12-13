"""
Page Repository - Split storage for web page data

Storage strategy:
- PostgreSQL: Content storage (content_text, embedding, status)
- Neo4j: Metadata and graph relationships (title, url, claims, entities)

The repository abstracts this split from consumers - they work with Page domain model.

ID format: pg_xxxxxxxx (11 chars)
"""
import logging
from typing import Optional, List
import asyncpg
from datetime import datetime

from pgvector.asyncpg import register_vector

from models.domain.page import Page
from services.neo4j_service import Neo4jService
from utils.id_generator import is_uuid, uuid_to_short_id

logger = logging.getLogger(__name__)


class PageRepository:
    """
    Repository for Page domain model

    Handles split storage:
    - PostgreSQL: content_text, embedding, processing status
    - Neo4j: metadata, graph relationships
    """

    def __init__(self, db_pool: asyncpg.Pool, neo4j_service: Optional[Neo4jService] = None):
        self.db_pool = db_pool
        self.neo4j = neo4j_service

    # =========================================================================
    # READ OPERATIONS
    # =========================================================================

    async def get_by_id(self, page_id: str) -> Optional[Page]:
        """
        Retrieve page by ID.

        Fetches from PostgreSQL (handles both old and new schemas).
        Optionally enriches with Neo4j metadata if available.

        Args:
            page_id: Page ID (pg_xxxxxxxx format)

        Returns:
            Page model or None
        """
        # Get from PostgreSQL (query all columns for backward compatibility)
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, url, canonical_url, content_text, status,
                       title, description, byline as author, thumbnail_url,
                       language, word_count, pub_time, metadata_confidence,
                       domain, site_name,
                       created_at, updated_at
                FROM core.pages
                WHERE id = $1
            """, page_id)

            if not row:
                return None

        # Build Page from PostgreSQL data (works with old schema)
        # Model handles UUID conversion in __post_init__ if needed
        page = Page(
            id=row['id'],
            url=row['url'],
            canonical_url=row['canonical_url'],
            content_text=row['content_text'],
            status=row['status'],
            title=row.get('title'),
            description=row.get('description'),
            author=row.get('author'),
            thumbnail_url=row.get('thumbnail_url'),
            language=row.get('language'),
            word_count=row.get('word_count', 0) or 0,
            pub_time=row.get('pub_time'),
            metadata_confidence=row.get('metadata_confidence', 0.0) or 0.0,
            domain=row.get('domain'),
            site_name=row.get('site_name'),
            created_at=row['created_at'],
            updated_at=row['updated_at']
        )

        # Optionally enrich with Neo4j metadata (for new architecture)
        if self.neo4j:
            try:
                neo4j_data = await self.neo4j.get_page_by_id(page_id)
                if neo4j_data:
                    # Neo4j data takes precedence if available
                    page.title = neo4j_data.get('title') or page.title
                    page.word_count = neo4j_data.get('word_count') or page.word_count
            except Exception:
                pass  # Neo4j not critical for reads

        return page

    async def get_by_canonical_url(self, canonical_url: str) -> Optional[Page]:
        """
        Retrieve page by canonical URL for deduplication.

        Args:
            canonical_url: Normalized canonical URL

        Returns:
            Page model or None
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id FROM core.pages WHERE canonical_url = $1
            """, canonical_url)

            if not row:
                return None

            return await self.get_by_id(row['id'])

    async def get_content(self, page_id: str) -> Optional[str]:
        """
        Get just the content text (for workers that only need content).

        Args:
            page_id: Page ID (pg_xxxxxxxx format)

        Returns:
            Content text or None
        """
        async with self.db_pool.acquire() as conn:
            return await conn.fetchval("""
                SELECT content_text FROM core.pages WHERE id = $1
            """, page_id)

    async def get_embedding(self, page_id: str) -> Optional[List[float]]:
        """
        Get page embedding vector (for event matching).

        Embedding is pre-computed by KnowledgeWorker during STAGE 4e.

        Args:
            page_id: Page ID (pg_xxxxxxxx or UUID format)

        Returns:
            Embedding vector as list of floats, or None if not available
        """
        try:
            async with self.db_pool.acquire() as conn:
                # Register pgvector type for native numpy array return
                await register_vector(conn)

                row = await conn.fetchrow("""
                    SELECT embedding
                    FROM core.pages
                    WHERE id = $1
                """, page_id)

                if row and row['embedding'] is not None:
                    # pgvector returns numpy array, convert to list
                    emb = row['embedding']
                    return [float(x) for x in emb]

                return None

        except Exception as e:
            logger.error(f"Failed to fetch page embedding for {page_id}: {e}")
            return None

    # =========================================================================
    # WRITE OPERATIONS
    # =========================================================================

    async def create(self, page: Page) -> Page:
        """
        Create new page.

        Creates stub in PostgreSQL with iframely metadata if available.

        Args:
            page: Page domain model

        Returns:
            Created page with timestamps
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO core.pages (
                    id, url, canonical_url, status,
                    title, description, byline, thumbnail_url,
                    language, metadata_confidence, domain, site_name
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """, page.id, page.url, page.canonical_url or page.url, page.status,
                page.title, page.description, page.author, page.thumbnail_url,
                page.language, page.metadata_confidence, page.domain, page.site_name)

            row = await conn.fetchrow("""
                SELECT created_at, updated_at FROM core.pages WHERE id = $1
            """, page.id)

            page.created_at = row['created_at']
            page.updated_at = row['updated_at']

            logger.debug(f"📄 Created page {page.id}: {page.url} (site: {page.site_name})")
            return page

    async def update_content(
        self,
        page_id: str,
        content_text: str,
        status: str = 'extracted',
        word_count: int = None
    ) -> None:
        """
        Update page content in PostgreSQL.

        Args:
            page_id: Page ID (pg_xxxxxxxx format)
            content_text: Extracted text content
            status: New status
            word_count: Word count (calculated if not provided)
        """
        if word_count is None:
            word_count = len(content_text.split()) if content_text else 0

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE core.pages
                SET content_text = $2, status = $3, word_count = $4, updated_at = NOW()
                WHERE id = $1
            """, page_id, content_text, status, word_count)

            logger.debug(f"📄 Updated page {page_id} content ({word_count} words)")

    async def update_metadata(
        self,
        page_id: str,
        title: str,
        description: Optional[str],
        author: Optional[str],
        thumbnail_url: Optional[str],
        language: str,
        word_count: int,
        pub_time: Optional[datetime],
        metadata_confidence: float,
        domain: str
    ) -> None:
        """
        Update page metadata in Neo4j.

        Args:
            page_id: Page ID (pg_xxxxxxxx format)
            title: Page title
            description: Meta description
            author: Author name
            thumbnail_url: Featured image URL
            language: Detected language
            word_count: Content word count
            pub_time: Publication timestamp
            metadata_confidence: Extraction confidence
            domain: Source domain
        """
        if not self.neo4j:
            logger.warning("Neo4j not available - skipping metadata update")
            return

        await self.neo4j.create_or_update_page(
            page_id=page_id,
            url=None,  # Already set on create
            title=title,
            domain=domain,
            status='extracted',
            word_count=word_count,
            metadata_confidence=metadata_confidence,
            description=description,
            author=author,
            thumbnail_url=thumbnail_url,
            language=language,
            pub_time=pub_time.isoformat() if pub_time else None
        )

        logger.debug(f"📄 Updated page {page_id} metadata in Neo4j")

    async def update_extracted_content(
        self,
        page_id: str,
        content_text: str,
        title: str,
        description: Optional[str],
        author: Optional[str],
        thumbnail_url: Optional[str],
        metadata_confidence: float,
        language: str,
        word_count: int,
        pub_time: Optional[datetime],
        domain: str = None,
        site_name: str = None
    ) -> None:
        """
        Update page with extracted content (both PostgreSQL and Neo4j).

        This is a convenience method that updates both stores.

        Args:
            page_id: Page ID (pg_xxxxxxxx format)
            content_text: Extracted text content
            title: Page title
            description: Page description
            author: Page author
            thumbnail_url: Thumbnail image URL
            metadata_confidence: Metadata extraction confidence
            language: Detected language
            word_count: Word count
            pub_time: Publication timestamp
            domain: Source domain
            site_name: Publisher name (from og:site_name)
        """
        # Update content and metadata in PostgreSQL
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE core.pages
                SET content_text = $2,
                    status = 'extracted',
                    word_count = $3,
                    title = COALESCE($4, title),
                    description = $5,
                    byline = $6,
                    thumbnail_url = $7,
                    language = $8,
                    pub_time = $9,
                    metadata_confidence = $10,
                    domain = $11,
                    site_name = $12,
                    updated_at = NOW()
                WHERE id = $1
            """, page_id, content_text, word_count, title, description, author,
                thumbnail_url, language, pub_time, metadata_confidence, domain, site_name)

        # Update metadata in Neo4j
        if self.neo4j and domain:
            await self.update_metadata(
                page_id=page_id,
                title=title,
                description=description,
                author=author,
                thumbnail_url=thumbnail_url,
                language=language,
                word_count=word_count,
                pub_time=pub_time,
                metadata_confidence=metadata_confidence,
                domain=domain
            )

        logger.debug(f"📄 Updated page {page_id} with extracted content ({word_count} words)")

    async def update_status(self, page_id: str, status: str) -> None:
        """
        Update page processing status.

        Updates both PostgreSQL (for worker coordination) and Neo4j (for queries).

        Args:
            page_id: Page ID (pg_xxxxxxxx format)
            status: New status
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE core.pages
                SET status = $2, updated_at = NOW()
                WHERE id = $1
            """, page_id, status)

        if self.neo4j:
            await self.neo4j.update_page_status(page_id, status)

        logger.debug(f"📄 Updated page {page_id} status: {status}")

    async def mark_failed(self, page_id: str, error_message: str = None) -> None:
        """
        Mark page as failed.

        Args:
            page_id: Page ID (pg_xxxxxxxx format)
            error_message: Optional error details
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE core.pages
                SET status = 'failed', updated_at = NOW()
                WHERE id = $1
            """, page_id)

        if self.neo4j:
            await self.neo4j.update_page_status(page_id, 'failed')

        logger.info(f"❌ Marked page {page_id} as failed")

    # =========================================================================
    # GRAPH QUERIES (via Neo4j)
    # =========================================================================

    async def get_claim_count(self, page_id: str) -> int:
        """
        Get count of claims linked to this page.

        Args:
            page_id: Page ID (pg_xxxxxxxx format)

        Returns:
            Count of claims
        """
        if not self.neo4j:
            return 0

        result = await self.neo4j._execute_read("""
            MATCH (p:Page {id: $page_id})-[:EMITS]->(c:Claim)
            RETURN count(c) as count
        """, {'page_id': page_id})

        return result[0]['count'] if result else 0

    async def get_entity_count(self, page_id: str) -> int:
        """
        Get count of unique entities mentioned in this page's claims.

        Args:
            page_id: Page ID (pg_xxxxxxxx format)

        Returns:
            Count of unique entities
        """
        if not self.neo4j:
            return 0

        result = await self.neo4j._execute_read("""
            MATCH (p:Page {id: $page_id})-[:EMITS]->(c:Claim)-[:MENTIONS]->(e:Entity)
            RETURN count(DISTINCT e) as count
        """, {'page_id': page_id})

        return result[0]['count'] if result else 0

    async def get_claims(self, page_id: str) -> List[dict]:
        """
        Get all claims for this page.

        Args:
            page_id: Page ID (pg_xxxxxxxx format)

        Returns:
            List of claim dictionaries
        """
        if not self.neo4j:
            return []

        results = await self.neo4j._execute_read("""
            MATCH (p:Page {id: $page_id})-[:EMITS]->(c:Claim)
            RETURN c.id as id, c.text as text, c.confidence as confidence,
                   c.event_time as event_time, c.created_at as created_at
            ORDER BY c.created_at
        """, {'page_id': page_id})

        return [dict(r) for r in results]

    async def get_entities(self, page_id: str) -> List[dict]:
        """
        Get all entities mentioned in this page's claims.

        Args:
            page_id: Page ID (pg_xxxxxxxx format)

        Returns:
            List of entity dictionaries
        """
        if not self.neo4j:
            return []

        results = await self.neo4j._execute_read("""
            MATCH (p:Page {id: $page_id})-[:EMITS]->(c:Claim)-[:MENTIONS]->(e:Entity)
            WITH DISTINCT e
            RETURN e.id as id, e.canonical_name as canonical_name,
                   e.entity_type as entity_type, e.wikidata_qid as wikidata_qid,
                   e.mention_count as mention_count, e.confidence as confidence
            ORDER BY e.mention_count DESC
        """, {'page_id': page_id})

        return [dict(r) for r in results]

    # =========================================================================
    # UTILITY
    # =========================================================================

    async def create_rogue_task(self, page_id: str, url: str) -> None:
        """
        Create rogue extraction task for pages needing browser-based extraction.

        Args:
            page_id: Page ID (pg_xxxxxxxx format)
            url: Page URL
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO core.rogue_extraction_tasks (page_id, url, status, created_at)
                VALUES ($1, $2, 'pending', NOW())
                ON CONFLICT DO NOTHING
            """, page_id, url)

            logger.info(f"🔴 Created rogue extraction task for {url}")
