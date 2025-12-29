"""
Force Field v0: Emergent Clustering via Intrinsic Forces
=========================================================

Claims attract/repel based on intrinsic properties.
Topology emerges from force equilibrium.

No LLM in this version - pure embedding math.

Affinity(a, b) = w_semantic * sim(emb_a, emb_b)
               + w_entity * jaccard(entities_a, entities_b)
               + w_temporal * temporal_proximity(t_a, t_b)
               - w_source_penalty * same_source(a, b)

Clusters = connected components where affinity >= threshold
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Callable
from datetime import datetime, timedelta
from collections import defaultdict


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ForceConfig:
    """Weights for affinity calculation."""
    w_semantic: float = 0.5      # Embedding similarity weight
    w_entity: float = 0.3        # Entity overlap weight
    w_temporal: float = 0.1      # Temporal proximity weight
    w_source_penalty: float = 0.2  # Same-source penalty (reduces copy bias)

    affinity_threshold: float = 0.4  # Edge creation threshold
    temporal_window_days: int = 14   # Time window for temporal proximity

    # Gating (hard constraints)
    require_anchor_entity: bool = True   # Require shared anchor entity for high-sim pairs
    anchor_sim_threshold: float = 0.6    # Semantic sim above which anchor is required
    time_gate_days: int = 30             # Block edge if time diff > this (0 = disabled)


# =============================================================================
# ITEM: A point in the force field
# =============================================================================

@dataclass
class FieldItem:
    """
    An item in the force field (claim, node, surface, event - any level).

    Has intrinsic properties that determine forces.
    """
    id: str
    text: str
    embedding: Optional[List[float]] = None
    entities: Set[str] = field(default_factory=set)
    anchor_entities: Set[str] = field(default_factory=set)  # Specific: persons, orgs
    source: str = ""
    timestamp: Optional[datetime] = None

    # Metadata for tracking
    level: int = 0  # 0=claim, 1=node, 2=surface, 3=event
    member_ids: Set[str] = field(default_factory=set)  # IDs of items this emerged from

    def __hash__(self):
        return hash(self.id)


@dataclass
class FieldEdge:
    """Edge between two items (created when affinity >= threshold)."""
    source_id: str
    target_id: str
    affinity: float
    components: Dict[str, float] = field(default_factory=dict)  # Breakdown


@dataclass
class FieldCluster:
    """Emergent cluster (connected component)."""
    id: int
    item_ids: Set[str]

    # Computed properties
    centroid: Optional[List[float]] = None
    label: str = ""
    entropy: float = 0.0  # Internal variance (for v1 tension detection)

    @property
    def size(self) -> int:
        return len(self.item_ids)


# =============================================================================
# FORCE CALCULATIONS
# =============================================================================

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between embeddings."""
    if not a or not b:
        return 0.0
    a, b = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity between entity sets."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def temporal_proximity(t1: Optional[datetime], t2: Optional[datetime],
                       window_days: int = 14) -> float:
    """Proximity score based on temporal distance."""
    if not t1 or not t2:
        return 0.0  # No contribution if unknown (avoid false positive bias)

    delta = abs((t1 - t2).total_seconds()) / 86400  # Days
    if delta > window_days:
        return 0.0
    return 1.0 - (delta / window_days)


def compute_affinity(a: FieldItem, b: FieldItem, config: ForceConfig) -> Tuple[float, Dict, bool]:
    """
    Compute affinity between two items.

    Returns: (affinity_score, component_breakdown, gated_out)

    gated_out = True means the edge should be blocked regardless of score.
    """
    components = {}
    gated_out = False

    # Semantic similarity (embedding)
    semantic = cosine_similarity(a.embedding, b.embedding) if a.embedding and b.embedding else 0.0
    components['semantic'] = semantic

    # Entity overlap (all entities)
    entity = jaccard_similarity(a.entities, b.entities)
    components['entity'] = entity

    # Anchor entity overlap (persons, orgs - specific identifiers)
    anchor_overlap = bool(a.anchor_entities & b.anchor_entities)
    components['anchor_overlap'] = anchor_overlap

    # Temporal proximity
    temporal = temporal_proximity(a.timestamp, b.timestamp, config.temporal_window_days)
    components['temporal'] = temporal

    # Source penalty (same source = reduce affinity to avoid copy bias)
    source_same = 1.0 if a.source and b.source and a.source == b.source else 0.0
    components['source_penalty'] = source_same

    # ==========================================================================
    # GATING (hard constraints that block edges)
    # ==========================================================================

    # Gate 1: Time gate - block if too far apart temporally
    if config.time_gate_days > 0 and a.timestamp and b.timestamp:
        days_apart = abs((a.timestamp - b.timestamp).total_seconds()) / 86400
        if days_apart > config.time_gate_days:
            gated_out = True
            components['gate_reason'] = f'time_apart:{days_apart:.0f}d'

    # Gate 2: Anchor entity gate - require anchor match for high-sim pairs
    # This prevents "same topic, different event" confusion
    # Stricter: if EITHER has anchors, require overlap (prevents topic bleeding)
    if config.require_anchor_entity and semantic >= config.anchor_sim_threshold:
        has_any_anchor = bool(a.anchor_entities or b.anchor_entities)
        if has_any_anchor and not anchor_overlap:
            gated_out = True
            components['gate_reason'] = 'anchor_mismatch'

    # Weighted sum
    affinity = (
        config.w_semantic * semantic +
        config.w_entity * entity +
        config.w_temporal * temporal -
        config.w_source_penalty * source_same
    )

    return affinity, components, gated_out


# =============================================================================
# FORCE FIELD
# =============================================================================

class ForceField:
    """
    A field of items connected by affinity forces.

    Clusters emerge as connected components.
    """

    def __init__(self, config: ForceConfig = None):
        self.config = config or ForceConfig()
        self.items: Dict[str, FieldItem] = {}
        self.edges: List[FieldEdge] = []
        self.clusters: List[FieldCluster] = []
        self._dirty = True

    def add_item(self, item: FieldItem):
        """Add an item to the field."""
        self.items[item.id] = item
        self._dirty = True

    def compute(self):
        """Compute edges and clusters from current items."""
        if not self._dirty:
            return

        self._compute_edges()
        self._find_clusters()
        self._compute_cluster_properties()
        self._dirty = False

    def _compute_edges(self):
        """Create edges where affinity >= threshold and not gated out."""
        self.edges = []
        self.gated_count = 0  # Track how many edges were blocked by gates
        item_list = list(self.items.values())

        for i, a in enumerate(item_list):
            for b in item_list[i+1:]:
                affinity, components, gated_out = compute_affinity(a, b, self.config)

                if gated_out:
                    self.gated_count += 1
                    continue

                if affinity >= self.config.affinity_threshold:
                    self.edges.append(FieldEdge(
                        source_id=a.id,
                        target_id=b.id,
                        affinity=affinity,
                        components=components
                    ))

    def _find_clusters(self):
        """Find connected components (emergent clusters)."""
        # Build adjacency
        adj = defaultdict(set)
        for e in self.edges:
            adj[e.source_id].add(e.target_id)
            adj[e.target_id].add(e.source_id)

        # DFS to find components
        visited = set()
        self.clusters = []
        cluster_id = 0

        for item_id in self.items:
            if item_id in visited:
                continue

            # BFS/DFS from this item
            component = set()
            stack = [item_id]
            while stack:
                curr = stack.pop()
                if curr in visited:
                    continue
                visited.add(curr)
                component.add(curr)
                stack.extend(adj[curr] - visited)

            self.clusters.append(FieldCluster(
                id=cluster_id,
                item_ids=component
            ))
            cluster_id += 1

        # Sort by size (largest first)
        self.clusters.sort(key=lambda c: -c.size)

    def _compute_cluster_properties(self):
        """Compute centroid, entropy, label for each cluster."""
        for cluster in self.clusters:
            items = [self.items[id] for id in cluster.item_ids]

            # Centroid (average embedding)
            embeddings = [i.embedding for i in items if i.embedding]
            if embeddings:
                cluster.centroid = np.mean(embeddings, axis=0).tolist()

            # Label (first item's text, truncated)
            if items:
                cluster.label = items[0].text[:50]

            # Entropy (semantic variance within cluster)
            if len(embeddings) > 1 and cluster.centroid:
                distances = [1 - cosine_similarity(e, cluster.centroid) for e in embeddings]
                cluster.entropy = float(np.mean(distances))
            else:
                cluster.entropy = 0.0

    def weave(self) -> List[FieldItem]:
        """
        The universal weave operation.

        Computes clusters and returns them as new FieldItems (next level up).
        """
        self.compute()

        next_level_items = []
        for cluster in self.clusters:
            if cluster.size < 2:
                continue  # Skip singletons (no emergence)

            items = [self.items[id] for id in cluster.item_ids]

            # Merge entities (preserve both all entities and anchor entities)
            merged_entities = set()
            merged_anchors = set()
            unique_sources = set()
            for item in items:
                merged_entities.update(item.entities)
                merged_anchors.update(item.anchor_entities)
                if item.source:
                    unique_sources.add(item.source)

            # Earliest/latest timestamps
            timestamps = [i.timestamp for i in items if i.timestamp]

            # Create emergent item (preserve anchor info for multi-level weaving)
            emergent = FieldItem(
                id=f"L{items[0].level + 1}_{cluster.id:03d}",
                text=cluster.label,
                embedding=cluster.centroid,
                entities=merged_entities,
                anchor_entities=merged_anchors,
                source=f"{len(unique_sources)} sources",
                timestamp=min(timestamps) if timestamps else None,
                level=items[0].level + 1,
                member_ids=cluster.item_ids
            )
            next_level_items.append(emergent)

        return next_level_items

    def summary(self) -> Dict:
        """Get field state summary."""
        self.compute()
        return {
            'items': len(self.items),
            'edges': len(self.edges),
            'clusters': len(self.clusters),
            'cluster_sizes': [c.size for c in self.clusters[:10]],
            'largest_cluster': {
                'size': self.clusters[0].size if self.clusters else 0,
                'label': self.clusters[0].label if self.clusters else '',
                'entropy': round(self.clusters[0].entropy, 3) if self.clusters else 0
            } if self.clusters else None
        }

    def purity(self, ground_truth: Dict[str, str]) -> float:
        """
        Compute cluster purity given ground truth labels.

        ground_truth: Dict mapping item_id -> true_label

        Returns: Average purity across clusters (0-1, higher = better)
        """
        self.compute()
        if not self.clusters:
            return 0.0

        total_purity = 0.0
        total_items = 0

        for cluster in self.clusters:
            if cluster.size == 0:
                continue

            # Count labels in this cluster
            label_counts = defaultdict(int)
            for item_id in cluster.item_ids:
                label = ground_truth.get(item_id, 'unknown')
                label_counts[label] += 1

            # Purity = majority class / total
            majority = max(label_counts.values()) if label_counts else 0
            cluster_purity = majority / cluster.size

            total_purity += cluster_purity * cluster.size
            total_items += cluster.size

        return total_purity / total_items if total_items > 0 else 0.0

    def b3_score(self, ground_truth: Dict[str, str]) -> Dict[str, float]:
        """
        B³ (B-cubed) clustering evaluation.

        Measures both precision (purity) and recall (completeness).
        Handles fragmentation properly unlike simple purity.

        Returns: {precision, recall, f1}
        """
        self.compute()

        # Build cluster assignment: item_id -> cluster_id
        item_to_cluster = {}
        for cluster in self.clusters:
            for item_id in cluster.item_ids:
                item_to_cluster[item_id] = cluster.id

        # Build ground truth clusters
        gt_clusters = defaultdict(set)
        for item_id, label in ground_truth.items():
            gt_clusters[label].add(item_id)

        precision_sum = 0.0
        recall_sum = 0.0
        n = len(ground_truth)

        for item_id, true_label in ground_truth.items():
            if item_id not in item_to_cluster:
                continue

            cluster_id = item_to_cluster[item_id]
            cluster = next((c for c in self.clusters if c.id == cluster_id), None)
            if not cluster:
                continue

            # Items in same predicted cluster
            pred_cluster_items = cluster.item_ids & set(ground_truth.keys())

            # Items in same ground truth cluster
            gt_cluster_items = gt_clusters[true_label]

            # Intersection: correctly clustered together
            correct = pred_cluster_items & gt_cluster_items

            # B³ precision for this item: |correct| / |pred_cluster|
            if pred_cluster_items:
                precision_sum += len(correct) / len(pred_cluster_items)

            # B³ recall for this item: |correct| / |gt_cluster|
            if gt_cluster_items:
                recall_sum += len(correct) / len(gt_cluster_items)

        precision = precision_sum / n if n > 0 else 0.0
        recall = recall_sum / n if n > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {'precision': precision, 'recall': recall, 'f1': f1}

    def completeness(self, ground_truth: Dict[str, str]) -> float:
        """
        Completeness: For each ground truth event, what fraction ends up in ONE cluster?

        Low completeness = fragmentation (event split across clusters)
        """
        self.compute()

        # Build cluster assignment
        item_to_cluster = {}
        for cluster in self.clusters:
            for item_id in cluster.item_ids:
                item_to_cluster[item_id] = cluster.id

        # For each ground truth event
        gt_clusters = defaultdict(set)
        for item_id, label in ground_truth.items():
            gt_clusters[label].add(item_id)

        total_completeness = 0.0
        for label, gt_items in gt_clusters.items():
            # Which predicted clusters contain items from this event?
            cluster_counts = defaultdict(int)
            for item_id in gt_items:
                if item_id in item_to_cluster:
                    cluster_counts[item_to_cluster[item_id]] += 1

            if cluster_counts:
                # Completeness = max items in one cluster / total items
                max_in_one = max(cluster_counts.values())
                total_completeness += max_in_one / len(gt_items)

        return total_completeness / len(gt_clusters) if gt_clusters else 0.0

    def merge_small_clusters(self, min_size: int = 2, entity_overlap_threshold: float = 0.3):
        """
        Post-processing: merge small clusters into larger ones if entity overlap is high.

        Reduces fragmentation.
        """
        self.compute()

        # Identify small and large clusters
        small = [c for c in self.clusters if c.size < min_size]
        large = [c for c in self.clusters if c.size >= min_size]

        if not large:
            return  # Nothing to merge into

        merged = set()
        for small_cluster in small:
            small_entities = set()
            for item_id in small_cluster.item_ids:
                small_entities.update(self.items[item_id].entities)

            # Find best large cluster by entity overlap
            best_match = None
            best_overlap = 0.0

            for large_cluster in large:
                large_entities = set()
                for item_id in large_cluster.item_ids:
                    large_entities.update(self.items[item_id].entities)

                if small_entities and large_entities:
                    overlap = len(small_entities & large_entities) / len(small_entities | large_entities)
                    if overlap > best_overlap and overlap >= entity_overlap_threshold:
                        best_overlap = overlap
                        best_match = large_cluster

            if best_match:
                # Merge small into large
                best_match.item_ids.update(small_cluster.item_ids)
                merged.add(small_cluster.id)

        # Remove merged clusters
        self.clusters = [c for c in self.clusters if c.id not in merged]
        self._compute_cluster_properties()


# =============================================================================
# MULTI-LEVEL WEAVING
# =============================================================================

def weave_hierarchy(items: List[FieldItem],
                    config: ForceConfig = None,
                    max_levels: int = 3) -> Dict[int, List[FieldItem]]:
    """
    Recursively weave items through multiple levels.

    Returns: Dict mapping level -> items at that level
    """
    config = config or ForceConfig()
    hierarchy = {0: items}

    current_items = items
    for level in range(1, max_levels + 1):
        if len(current_items) < 2:
            break

        field = ForceField(config)
        for item in current_items:
            field.add_item(item)

        emergent = field.weave()
        if not emergent:
            break

        hierarchy[level] = emergent
        current_items = emergent

    return hierarchy


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Simple test
    items = [
        FieldItem(id="c1", text="Fire kills 13 in Hong Kong", entities={"Hong Kong", "fire"}, source="BBC"),
        FieldItem(id="c2", text="Hong Kong fire death toll 13", entities={"Hong Kong", "fire"}, source="Reuters"),
        FieldItem(id="c3", text="13 dead in HK blaze", entities={"Hong Kong", "fire"}, source="SCMP"),
        FieldItem(id="c4", text="Jimmy Lai trial continues", entities={"Jimmy Lai", "Hong Kong"}, source="BBC"),
        FieldItem(id="c5", text="Lai faces security charges", entities={"Jimmy Lai"}, source="Guardian"),
    ]

    field = ForceField()
    for item in items:
        field.add_item(item)

    print("=== Force Field v0 Test ===")
    print(f"Items: {len(items)}")

    field.compute()
    print(f"Edges: {len(field.edges)}")
    print(f"Clusters: {len(field.clusters)}")

    for c in field.clusters:
        print(f"  Cluster {c.id}: {c.size} items - {c.label[:40]}")
        for item_id in c.item_ids:
            print(f"    - {item_id}: {field.items[item_id].text[:30]}")

    # Ground truth
    ground_truth = {
        "c1": "fire", "c2": "fire", "c3": "fire",
        "c4": "lai", "c5": "lai"
    }
    print(f"\nPurity: {field.purity(ground_truth):.2%}")
