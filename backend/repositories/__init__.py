"""
Repository Pattern - Storage abstraction layer

Repositories hide storage details (PostgreSQL, Neo4j) from business logic.
Consumers work with domain models, not storage-specific types.

Architecture (Neo4j-centric):
- Neo4j: Primary store for knowledge graph (Pages, Claims, Entities, Events)
- PostgreSQL: Content storage (page text, embeddings)

Storage Split:
- PageRepository: PostgreSQL (content) + Neo4j (metadata, relationships)
- ClaimRepository: Neo4j (claims, MENTIONS) + PostgreSQL (embeddings optional)
- EntityRepository: Neo4j only
- EventRepository: Neo4j + PostgreSQL (embeddings)
"""
import os
import asyncpg

# Shared database connection pool (initialized on first use)
db_pool = None


async def get_db_pool():
    """Get or create shared database connection pool"""
    global db_pool
    if db_pool is None:
        db_pool = await asyncpg.create_pool(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=int(os.getenv('POSTGRES_PORT', 5432)),
            user=os.getenv('POSTGRES_USER', 'herenews_user'),
            password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
            database=os.getenv('POSTGRES_DB', 'herenews'),
            min_size=2,
            max_size=10
        )
    return db_pool


# Conditional imports - some workers don't have Neo4j dependency
try:
    from .entity_repository import EntityRepository
    from .claim_repository import ClaimRepository
    from .event_repository import EventRepository
    from .page_repository import PageRepository
    from .rogue_task_repository import RogueTaskRepository
    from .user_repository import UserRepository
    from .comment_repository import CommentRepository
    from .chat_session_repository import ChatSessionRepository
    __all__ = [
        'EntityRepository',
        'ClaimRepository',
        'EventRepository',
        'PageRepository',
        'RogueTaskRepository',
        'UserRepository',
        'CommentRepository',
        'ChatSessionRepository',
        'db_pool',
        'get_db_pool',
    ]
except ModuleNotFoundError:
    # Neo4j not available - only basic repositories
    __all__ = ['db_pool', 'get_db_pool']
