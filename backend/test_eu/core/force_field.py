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

# =============================================================================
# SPRING FORCE FIELD (D3-style with principled hub detection)
# =============================================================================

@dataclass
class SpringNode:
    """A surface in the spring force field."""
    id: str
    embedding: Optional[List[float]] = None
    anchor_entities: Set[str] = field(default_factory=set)
    entities: Set[str] = field(default_factory=set)
    sources: Set[str] = field(default_factory=set)

    # Position (for visualization / actual simulation if needed)
    x: float = 0.0
    y: float = 0.0


@dataclass
class Spring:
    """Connection between two nodes with stiffness (evidence strength)."""
    node1_id: str
    node2_id: str
    stiffness: float  # 0 = no binding, 1 = rigid
    evidence: Dict[str, float] = field(default_factory=dict)  # breakdown


@dataclass
class SpringCluster:
    """An emerged event as a basin/attractor."""
    id: str
    core_nodes: Set[str] = field(default_factory=set)  # High connectivity
    periphery_nodes: Set[str] = field(default_factory=set)  # Low connectivity

    # Signature
    anchor_weights: Dict[str, float] = field(default_factory=dict)
    centroid: Optional[List[float]] = None


class SpringForceField:
    """
    D3-style force-based event clustering with principled hub detection.

    Mental model:
    - Nodes = surfaces
    - Springs = evidence connections
    - Stiffness = evidence strength
      - Rare anchors → stiff springs (bind tightly)
      - Hub entities → near-zero stiffness (don't bind)
      - Semantic similarity → weak springs (proximity, not identity)
    - Events = co-moving clusters / basins

    Key insight: Hub detection is PRINCIPLED
    - Entity is hub if it appears across MULTIPLE semantic clusters
    - Not just high df - must cross semantic boundaries
    """

    def __init__(self,
                 hub_cluster_threshold: int = 2,
                 semantic_cluster_sim: float = 0.55,
                 stiffness_threshold: float = 0.3,
                 known_hubs: Set[str] = None):
        """
        Args:
            hub_cluster_threshold: Entity appearing in this many semantic
                                   clusters is considered a hub
            semantic_cluster_sim: Similarity threshold for semantic clustering
            stiffness_threshold: Minimum stiffness for springs that form structure
            known_hubs: Set of entity names that are known hubs (e.g., common
                       locations like "Hong Kong", "China", "US"). These are
                       always treated as hubs regardless of detection.
        """
        self.hub_cluster_threshold = hub_cluster_threshold
        self.semantic_cluster_sim = semantic_cluster_sim
        self.stiffness_threshold = stiffness_threshold
        self.known_hubs = known_hubs or set()

        self.nodes: Dict[str, SpringNode] = {}
        self.springs: List[Spring] = []
        self.hub_entities: Set[str] = set()
        self.clusters: List[SpringCluster] = []

        # Document frequencies
        self.anchor_df: Dict[str, int] = defaultdict(int)
        self.entity_df: Dict[str, int] = defaultdict(int)

    def add_node(self, node: SpringNode):
        """Add a surface node to the force field."""
        self.nodes[node.id] = node
        for a in node.anchor_entities:
            self.anchor_df[a] += 1
        for e in node.entities:
            self.entity_df[e] += 1

    def build(self):
        """
        Build the force field:
        1. Detect hubs via component merge impact + known hubs
        2. Compute springs between all node pairs
        3. Find clusters via stiff-spring connectivity
        """
        nodes = list(self.nodes.values())
        if len(nodes) < 2:
            return

        # Step 1: Detect hubs via component merge impact
        detected_hubs = self._detect_hubs(nodes)

        # Merge with known hubs (e.g., common locations)
        self.hub_entities = detected_hubs | self.known_hubs

        # Step 2: Compute springs
        self.springs = self._compute_all_springs(nodes)

        # Step 3: Find clusters
        self.clusters = self._find_clusters()

    def _detect_hubs(self, nodes: List[SpringNode]) -> Set[str]:
        """
        Detect hub symbols using COMPONENT MERGE IMPACT on the surface graph.

        Key insight: Hub detection is a property of ANY symbol (anchor OR entity)
        that bridges across otherwise separate event clusters. This is NOT ad-hoc -
        it's derived from graph topology.

        Algorithm:
        1. Build surface adjacency WITHOUT each candidate symbol
        2. Find connected components without this symbol
        3. If adding this symbol's edges merges LARGE components that were
           previously separate → it's bridging across events (hub)
        4. If it only adds edges within existing components → event-defining

        This applies to BOTH anchors and entities uniformly.
        """
        node_ids = {n.id for n in nodes}
        node_map = {n.id: n for n in nodes}

        def find_components(adj: Dict[str, Set[str]]) -> List[Set[str]]:
            """Find all connected components, returning list of node ID sets."""
            visited: Set[str] = set()
            components: List[Set[str]] = []
            for node_id in node_ids:
                if node_id in visited:
                    continue
                component: Set[str] = set()
                queue = [node_id]
                while queue:
                    curr = queue.pop(0)
                    if curr in visited:
                        continue
                    visited.add(curr)
                    component.add(curr)
                    queue.extend(adj.get(curr, set()) - visited)
                components.append(component)
            return components

        def is_hub_symbol(symbol: str, symbol_surfaces: Set[str],
                         get_node_symbols: callable) -> bool:
            """
            Test if a symbol is a hub using component merge impact.

            Args:
                symbol: The symbol to test
                symbol_surfaces: Set of node IDs containing this symbol
                get_node_symbols: Function to get all symbols from a node
            """
            if len(symbol_surfaces) < 3:
                return False  # Not frequent enough to matter

            # Build adjacency WITHOUT this symbol
            adj_without: Dict[str, Set[str]] = defaultdict(set)
            for i, n1 in enumerate(nodes):
                for n2 in nodes[i+1:]:
                    # Shared symbols excluding the candidate
                    shared = (get_node_symbols(n1) & get_node_symbols(n2)) - {symbol}
                    if shared:
                        adj_without[n1.id].add(n2.id)
                        adj_without[n2.id].add(n1.id)

            # Find components WITHOUT this symbol
            components_without = find_components(adj_without)

            # Check: how many nodes would be orphaned (have no other symbols)?
            orphan_count = 0
            for nid in symbol_surfaces:
                node = node_map[nid]
                other_symbols = get_node_symbols(node) - {symbol}
                if not other_symbols:
                    orphan_count += 1

            # If marking this as hub would orphan many nodes, don't do it
            if orphan_count >= 3:
                return False  # Essential symbol for these nodes

            # Check if symbol bridges multiple components
            component_counts = []
            for comp in components_without:
                overlap = comp & symbol_surfaces
                if overlap:
                    component_counts.append(len(overlap))

            # Hub if symbol appears significantly (>=3) in 2+ separate components
            if len(component_counts) >= 2:
                large_presence = len([c for c in component_counts if c >= 3])
                if large_presence >= 2:
                    return True

            return False

        hubs: Set[str] = set()

        # 1. Detect anchor hubs
        anchor_surfaces: Dict[str, Set[str]] = defaultdict(set)
        for node in nodes:
            for a in node.anchor_entities:
                anchor_surfaces[a].add(node.id)

        for anchor, surfaces in anchor_surfaces.items():
            if is_hub_symbol(anchor, surfaces, lambda n: n.anchor_entities):
                hubs.add(anchor)

        # 2. Detect entity hubs (same principle applies!)
        # This catches "Hong Kong", "China", "John Lee" etc. as hubs
        entity_surfaces: Dict[str, Set[str]] = defaultdict(set)
        for node in nodes:
            for e in node.entities:
                entity_surfaces[e].add(node.id)

        for entity, surfaces in entity_surfaces.items():
            if is_hub_symbol(entity, surfaces, lambda n: n.entities):
                hubs.add(entity)

        return hubs

    def _semantic_cluster(self, nodes: List[SpringNode]) -> List[List[SpringNode]]:
        """
        Cluster nodes by semantic similarity (embedding cosine).
        Uses greedy single-linkage.
        """
        if not nodes:
            return []

        with_emb = [n for n in nodes if n.embedding]
        without_emb = [n for n in nodes if not n.embedding]

        if not with_emb:
            return [[n] for n in nodes]

        # Greedy single-linkage clustering
        clusters: List[Set[str]] = []
        node_to_cluster: Dict[str, int] = {}

        for node in with_emb:
            best_cluster = None
            for i, cluster in enumerate(clusters):
                for cid in cluster:
                    other = self.nodes[cid]
                    if other.embedding:
                        sim = cosine_similarity(node.embedding, other.embedding)
                        if sim >= self.semantic_cluster_sim:
                            best_cluster = i
                            break
                if best_cluster is not None:
                    break

            if best_cluster is not None:
                clusters[best_cluster].add(node.id)
                node_to_cluster[node.id] = best_cluster
            else:
                node_to_cluster[node.id] = len(clusters)
                clusters.append({node.id})

        # Nodes without embeddings are their own clusters
        for node in without_emb:
            clusters.append({node.id})

        return [[self.nodes[nid] for nid in cluster] for cluster in clusters]

    def _compute_all_springs(self, nodes: List[SpringNode]) -> List[Spring]:
        """Compute springs between all node pairs."""
        springs = []
        n = len(nodes)

        for i in range(n):
            for j in range(i + 1, n):
                spring = self._compute_spring(nodes[i], nodes[j])
                if spring.stiffness > 0.01:
                    springs.append(spring)

        return springs

    def _compute_spring(self, n1: SpringNode, n2: SpringNode) -> Spring:
        """
        Compute spring stiffness between two nodes.

        MULTI-SIGNAL REQUIREMENT:
        Event binding requires ≥2 strong signals. Single-entity overlap
        is NOT sufficient for event binding (even if non-hub).

        Signal types:
        1. Non-hub anchor overlap (strong)
        2. Non-hub entity overlap (moderate)
        3. Semantic similarity (weak)

        Stiffness is computed as:
        - If ≥2 non-hub anchors OR (1 non-hub anchor + 1 other signal): stiff (0.5-1.0)
        - If only 1 signal: weak (capped at 0.25)
        - Hubs contribute near-zero regardless

        This prevents single-entity incidental references from binding events.
        """
        evidence = {}
        n_surfaces = len(self.nodes)

        # Collect all signals
        signals = []  # List of (type, strength, details)

        # 1. Anchor overlap
        shared_anchors = n1.anchor_entities & n2.anchor_entities
        non_hub_anchors = shared_anchors - self.hub_entities
        hub_anchors = shared_anchors & self.hub_entities

        anchor_signal_strength = 0.0
        if non_hub_anchors:
            n_shared = len(non_hub_anchors)
            if n_shared >= 2:
                anchor_signal_strength = 0.7 + 0.3 * min(1.0, n_shared / 3)
                evidence['anchor_bundle'] = True
            else:
                anchor_signal_strength = 0.5
                # Check for co-anchor context (additional support)
                n1_other = n1.anchor_entities - non_hub_anchors - self.hub_entities
                n2_other = n2.anchor_entities - non_hub_anchors - self.hub_entities
                coanchor_overlap = n1_other & n2_other
                if coanchor_overlap:
                    anchor_signal_strength += 0.1 * min(len(coanchor_overlap), 2)
                    evidence['coanchor_support'] = list(coanchor_overlap)[:3]

            evidence['non_hub_anchors'] = list(non_hub_anchors)
            signals.append(('anchor', anchor_signal_strength, non_hub_anchors))

        if hub_anchors:
            evidence['hub_anchors'] = list(hub_anchors)
            # Hubs don't count as signals

        # 2. Entity overlap
        shared_entities = n1.entities & n2.entities
        non_hub_entities = shared_entities - self.hub_entities

        entity_signal_strength = 0.0
        if non_hub_entities:
            entity_score = 0.0
            for e in non_hub_entities:
                df = self.entity_df[e]
                idf = np.log(1 + n_surfaces / df) if df > 0 else 0
                entity_score += idf

            max_idf = np.log(1 + n_surfaces)
            normalized = min(1.0, entity_score / (max_idf * 3))
            entity_signal_strength = 0.2 + 0.2 * normalized

            evidence['non_hub_entities'] = list(non_hub_entities)[:5]
            signals.append(('entity', entity_signal_strength, non_hub_entities))

        # 3. Semantic similarity
        semantic_signal_strength = 0.0
        if n1.embedding and n2.embedding:
            sim = cosine_similarity(n1.embedding, n2.embedding)
            if sim > 0.5:
                semantic_signal_strength = (sim - 0.5) * 0.3
                evidence['semantic_sim'] = sim
                signals.append(('semantic', semantic_signal_strength, sim))

        # MULTI-SIGNAL LOGIC
        # Compute final stiffness based on signal count and strength
        n_signals = len(signals)

        if n_signals == 0:
            # No evidence at all
            stiffness = 0.0

        elif n_signals == 1:
            # Single signal - cap stiffness to prevent single-entity binding
            # This is the key structural prior: one entity can be incidental
            sig_type, sig_strength, _ = signals[0]
            if sig_type == 'anchor':
                # Single anchor: still weak unless it has multiple entries
                if evidence.get('anchor_bundle') or evidence.get('coanchor_support'):
                    stiffness = sig_strength  # Has supporting evidence
                else:
                    stiffness = min(0.25, sig_strength)  # Cap single-anchor
            elif sig_type == 'entity':
                # Single entity overlap is even weaker
                stiffness = min(0.15, sig_strength)
            else:
                # Semantic only
                stiffness = min(0.1, sig_strength)
            evidence['single_signal_cap'] = True

        else:
            # ≥2 signals - full multi-signal binding
            # Take max of signal strengths, boosted by multi-signal presence
            max_strength = max(s[1] for s in signals)
            signal_bonus = 0.1 * (n_signals - 1)  # Bonus for additional signals
            stiffness = min(1.0, max_strength + signal_bonus)
            evidence['multi_signal'] = True
            evidence['signal_count'] = n_signals

        evidence['stiffness'] = stiffness
        return Spring(
            node1_id=n1.id,
            node2_id=n2.id,
            stiffness=stiffness,
            evidence=evidence
        )

    def _find_clusters(self) -> List[SpringCluster]:
        """
        Find clusters via stiff-spring connectivity.
        Only springs above stiffness_threshold form cluster structure.
        """
        # Build adjacency from stiff springs only
        adj: Dict[str, Set[str]] = defaultdict(set)
        for spring in self.springs:
            if spring.stiffness >= self.stiffness_threshold:
                adj[spring.node1_id].add(spring.node2_id)
                adj[spring.node2_id].add(spring.node1_id)

        # Find connected components
        visited: Set[str] = set()
        clusters: List[SpringCluster] = []

        for node_id in self.nodes:
            if node_id in visited:
                continue

            component: Set[str] = set()
            queue = [node_id]
            while queue:
                curr = queue.pop(0)
                if curr in visited:
                    continue
                visited.add(curr)
                component.add(curr)
                queue.extend(adj[curr] - visited)

            if component:
                cluster = self._make_cluster(component, len(clusters))
                clusters.append(cluster)

        return clusters

    def _make_cluster(self, node_ids: Set[str], cluster_idx: int) -> SpringCluster:
        """Create a SpringCluster from a set of node IDs."""
        cluster_id = f"spring_event_{cluster_idx}"

        # Centroid
        embeddings = []
        for nid in node_ids:
            node = self.nodes[nid]
            if node.embedding:
                embeddings.append(node.embedding)

        centroid = None
        if embeddings:
            dim = len(embeddings[0])
            centroid = [sum(e[i] for e in embeddings) / len(embeddings)
                       for i in range(dim)]

        # Anchor weights (IDF-weighted, excluding hubs)
        anchor_weights: Dict[str, float] = defaultdict(float)
        n_surfaces = len(self.nodes)

        for nid in node_ids:
            node = self.nodes[nid]
            for a in node.anchor_entities:
                if a not in self.hub_entities:
                    df = self.anchor_df[a]
                    idf = np.log(1 + n_surfaces / df) if df > 0 else 0
                    anchor_weights[a] += idf

        # Classify core vs periphery
        core_nodes: Set[str] = set()
        periphery_nodes: Set[str] = set()

        for nid in node_ids:
            stiff_connections = sum(
                1 for s in self.springs
                if s.stiffness >= self.stiffness_threshold and
                ((s.node1_id == nid and s.node2_id in node_ids) or
                 (s.node2_id == nid and s.node1_id in node_ids))
            )
            if stiff_connections >= 2:
                core_nodes.add(nid)
            else:
                periphery_nodes.add(nid)

        return SpringCluster(
            id=cluster_id,
            core_nodes=core_nodes,
            periphery_nodes=periphery_nodes,
            anchor_weights=dict(anchor_weights),
            centroid=centroid
        )

    def compute_membership_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Compute continuous membership weights for all nodes to all clusters.

        Returns: Dict mapping node_id -> {cluster_id: weight}
        """
        weights: Dict[str, Dict[str, float]] = defaultdict(dict)

        for node_id in self.nodes:
            node = self.nodes[node_id]

            for cluster in self.clusters:
                weight = self._compute_node_cluster_affinity(node, cluster)
                if weight > 0.01:
                    weights[node_id][cluster.id] = weight

        # Normalize so weights sum to 1 per node
        for node_id in weights:
            total = sum(weights[node_id].values())
            if total > 0:
                for cid in weights[node_id]:
                    weights[node_id][cid] /= total

        return dict(weights)

    def _compute_node_cluster_affinity(self, node: SpringNode,
                                        cluster: SpringCluster) -> float:
        """Compute affinity of a node to a cluster."""
        affinity = 0.0

        # 1. Anchor match
        node_anchors = node.anchor_entities - self.hub_entities
        cluster_anchors = set(cluster.anchor_weights.keys())
        shared = node_anchors & cluster_anchors

        if shared:
            for a in shared:
                affinity += cluster.anchor_weights.get(a, 0) * 0.5

        # 2. Spring connections to cluster members
        cluster_members = cluster.core_nodes | cluster.periphery_nodes
        for spring in self.springs:
            if spring.node1_id == node.id and spring.node2_id in cluster_members:
                affinity += spring.stiffness * 0.3
            elif spring.node2_id == node.id and spring.node1_id in cluster_members:
                affinity += spring.stiffness * 0.3

        # 3. Semantic similarity to centroid (weak)
        if node.embedding and cluster.centroid:
            sim = cosine_similarity(node.embedding, cluster.centroid)
            if sim > 0.5:
                affinity += (sim - 0.5) * 0.2

        return affinity

    def get_diagnostics(self) -> Dict:
        """Get diagnostic info about the force field."""
        stiff_dist = {'0.0-0.1': 0, '0.1-0.3': 0, '0.3-0.5': 0, '0.5-0.7': 0, '0.7-1.0': 0}
        for s in self.springs:
            if s.stiffness < 0.1:
                stiff_dist['0.0-0.1'] += 1
            elif s.stiffness < 0.3:
                stiff_dist['0.1-0.3'] += 1
            elif s.stiffness < 0.5:
                stiff_dist['0.3-0.5'] += 1
            elif s.stiffness < 0.7:
                stiff_dist['0.5-0.7'] += 1
            else:
                stiff_dist['0.7-1.0'] += 1

        return {
            'n_nodes': len(self.nodes),
            'n_springs': len(self.springs),
            'n_hubs': len(self.hub_entities),
            'hub_entities': list(self.hub_entities)[:10],
            'n_clusters': len(self.clusters),
            'stiffness_distribution': stiff_dist,
        }


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
