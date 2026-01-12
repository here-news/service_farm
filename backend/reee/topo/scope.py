"""
Scope Computation - Derive scope_id from anchor entities.

Pure function - no DB, no LLM.

The scope_id determines which "real-world referent" a claim is about.
Claims with different scope_ids cannot be in the same surface.
"""

from typing import FrozenSet, Set, Tuple
import re

# Hub entities that are too generic to scope by
# These are suppressed from scope computation
DEFAULT_HUB_ENTITIES: FrozenSet[str] = frozenset({
    # Countries/regions that appear in many unrelated stories
    "United States",
    "China",
    "European Union",
    "United Nations",
    "Russia",
    "United Kingdom",
    "Asia",
    "Europe",
    "North America",
    "South America",
    "Africa",
    "Middle East",
    # Can be extended based on corpus analysis
})


def compute_scope_id(
    anchor_entities: FrozenSet[str],
    hub_entities: FrozenSet[str] = DEFAULT_HUB_ENTITIES,
) -> str:
    """Compute deterministic scope_id from anchor entities.

    Pure function - no DB, no LLM.

    Args:
        anchor_entities: Set of anchor entity names
        hub_entities: Set of hub entities to suppress

    Returns:
        Deterministic scope_id string

    Algorithm:
    1. Filter out hub entities (too generic)
    2. If all anchors are hubs, use them anyway (fallback)
    3. Normalize names (lowercase, remove spaces/apostrophes)
    4. Take top 2 for scope drift tolerance
    5. Join with underscore

    Examples:
        {"John Lee", "Hong Kong"} -> "scope_hongkong_johnlee"
        {"Gavin Newsom", "California"} -> "scope_california_gavinnewsom"
        {"United States"} -> "scope_unitedstates" (hub used as fallback)
        {} -> "scope_unscoped"
    """
    # Filter out hub entities
    scoping_anchors = anchor_entities - hub_entities

    if not scoping_anchors:
        # Fallback: all anchors are hubs, use them anyway
        scoping_anchors = anchor_entities

    if not scoping_anchors:
        return "scope_unscoped"

    # Normalize and sort
    normalized = sorted(_normalize_entity(a) for a in scoping_anchors)

    # Take top 2 for scope drift tolerance
    # This means adding a third anchor doesn't change the scope
    primary = normalized[:2]

    return "scope_" + "_".join(primary)


def _normalize_entity(name: str) -> str:
    """Normalize entity name for scope_id.

    - Lowercase
    - Remove spaces, apostrophes, hyphens
    - Limit length for readability
    """
    normalized = name.lower()
    normalized = re.sub(r"[\s'\-]", "", normalized)
    return normalized[:30]  # Limit length


def is_hub_entity(
    entity: str,
    hub_entities: FrozenSet[str] = DEFAULT_HUB_ENTITIES,
) -> bool:
    """Check if entity is a hub (too generic to scope by)."""
    return entity in hub_entities


def compute_scope_signature(scope_id: str) -> str:
    """Compute signature from scope_id.

    This is useful for debugging/logging, not for identity.
    """
    import hashlib
    return hashlib.sha256(scope_id.encode()).hexdigest()[:8]


def extract_primary_anchors(
    entities: FrozenSet[str],
    hub_entities: FrozenSet[str] = DEFAULT_HUB_ENTITIES,
    max_anchors: int = 5,
) -> Tuple[FrozenSet[str], bool]:
    """Extract primary anchor entities from a set of entities.

    Returns:
        Tuple of (anchors, all_hubs)
        - anchors: The primary anchor entities
        - all_hubs: True if all anchors are hub entities

    This is a helper for evidence providers to compute anchors.
    """
    non_hubs = entities - hub_entities

    if non_hubs:
        # Use non-hub entities as anchors
        sorted_anchors = sorted(non_hubs)[:max_anchors]
        return frozenset(sorted_anchors), False
    else:
        # All entities are hubs - use them anyway
        sorted_anchors = sorted(entities)[:max_anchors]
        return frozenset(sorted_anchors), True
