"""
REEE Repositories - Persistence layer for epistemic units.

Repositories handle:
- Neo4j storage for graph structures (claims, surfaces, events)
- Versioning and lifecycle management
- Query patterns for epistemic operations
"""

from .surface_repository import SurfaceRepository

__all__ = ['SurfaceRepository']
