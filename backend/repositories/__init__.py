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

# Conditional imports - some workers don't have Neo4j dependency
try:
    from .entity_repository import EntityRepository
    from .claim_repository import ClaimRepository
    from .event_repository import EventRepository
    from .page_repository import PageRepository
    from .rogue_task_repository import RogueTaskRepository
    __all__ = [
        'EntityRepository',
        'ClaimRepository',
        'EventRepository',
        'PageRepository',
        'RogueTaskRepository',
    ]
except ModuleNotFoundError:
    # Neo4j not available - only basic repositories
    __all__ = []
