"""
Repository Pattern - Storage abstraction layer

Repositories hide storage details (PostgreSQL, Neo4j) from business logic.

Architecture:
- Repositories provide CRUD operations for domain models
- PostgreSQL: Handles content, metadata, embeddings (via pgvector)
- Neo4j: Handles graph structure, relationships
- Dual-write: Some entities stored in both (e.g., entities, events)
"""

# Conditional imports - some workers don't have Neo4j dependency
try:
    from .entity_repository import EntityRepository
    from .claim_repository import ClaimRepository
    from .event_repository import EventRepository
    __all__ = [
        'EntityRepository',
        'ClaimRepository',
        'EventRepository',
    ]
except ModuleNotFoundError:
    # Neo4j not available - only basic repositories
    __all__ = []
