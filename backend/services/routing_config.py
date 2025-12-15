"""
Routing Configuration - Single source of truth for event matching weights

These weights determine how incoming pages are matched to existing events.
"""

# =============================================================================
# ROUTING WEIGHTS
# =============================================================================
# Semantic similarity is primary signal for same-story detection.
# Entity overlap is secondary because articles about the same event often
# mention different supporting entities (different sources, family members, etc.)
#
# Example: Two articles about "King Charles cancer update" may share only
# Charles III but mention completely different people (doctors, family, etc.)
# High semantic similarity (0.65+) should still match them.
#
WEIGHT_ENTITY = 0.40   # Entity overlap (Jaccard similarity)
WEIGHT_SEMANTIC = 0.60  # Semantic similarity (page vs event embedding)

# Minimum score to attach page to existing event
ROUTING_THRESHOLD = 0.20

# Minimum semantic similarity for fallback matching (when no entity overlap)
# Higher threshold since we're relying solely on embeddings
SEMANTIC_FALLBACK_THRESHOLD = 0.50
