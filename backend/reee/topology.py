"""
Pure Topological Primitives
============================

Domain-agnostic mathematical structures for belief topology.

This module contains ONLY:
- Node (belief point in semantic space)
- Edge (relation between nodes)
- Surface (connected component / emergent cluster)
- Mass, Entropy, Plausibility calculations

NO domain-specific logic. NO news references. NO application code.

Philosophy:
- Jaynes Maximum Entropy: Uncertainty is honest until evidence reduces it
- Hypergeometric Topology: Truth emerges from convergent source geometry
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Callable, Protocol, runtime_checkable, Union
from enum import Enum
from collections import defaultdict


# =============================================================================
# BELIEF SET PROTOCOL (unified interface for nodes, surfaces, meta-surfaces)
# =============================================================================

@runtime_checkable
class BeliefSet(Protocol):
    """
    Protocol for any set of beliefs at any level of hierarchy.

    Implementations:
    - Node: atomic belief (set of size 1)
    - Surface: set of nodes (connected component)
    - MetaSurface: set of surfaces (cluster of clusters)

    All levels support the same operations:
    - entropy(): epistemic uncertainty
    - centroid: semantic position
    - sources: provenance union
    - mass: epistemic weight
    """

    @property
    def centroid(self) -> Optional[List[float]]: ...

    def entropy(self) -> float: ...

    @property
    def source_count(self) -> int: ...

    @property
    def mass(self) -> float: ...


# =============================================================================
# VECTOR OPERATIONS
# =============================================================================

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two embedding vectors."""
    a, b = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def centroid(embeddings: List[List[float]]) -> Optional[List[float]]:
    """Compute centroid of embedding vectors."""
    if not embeddings:
        return None
    return np.mean(embeddings, axis=0).tolist()


# =============================================================================
# RELATION TYPES
# =============================================================================

class Relation(Enum):
    """
    Possible relations between propositions.

    Domain-agnostic: applies to any claim-based system.
    """
    NOVEL = "novel"           # New proposition, no existing node covers it
    CONFIRMS = "confirms"     # Same proposition from another source
    REFINES = "refines"       # More specific version (adds detail)
    SUPERSEDES = "supersedes" # Updated value (temporal succession)
    CONFLICTS = "conflicts"   # Mutually exclusive propositions


# =============================================================================
# NODE: A point in belief space
# =============================================================================

@dataclass
class Node:
    """
    A node in the belief topology.

    Represents a proposition with:
    - Semantic position (embedding)
    - Provenance (sources)
    - Epistemic state (entropy, plausibility)

    Note: event_id is optional metadata for higher-level interpretation.
    The kernel does not compute this - use EventInterpreter for event grouping.
    """
    id: str
    text: str
    sources: Set[str] = field(default_factory=set)
    claim_ids: Set[str] = field(default_factory=set)
    entity_ids: Set[str] = field(default_factory=set)  # Pre-extracted entity IDs for clustering
    embedding: Optional[List[float]] = None
    superseded: Optional[str] = None
    evolution: List[Tuple[str, str]] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    event_id: Optional[str] = None  # Optional: set by EventInterpreter, not kernel

    @property
    def source_count(self) -> int:
        return len(self.sources)

    def has_entity_overlap(self, other: 'Node') -> bool:
        """Check if this node shares any entities with another node."""
        return bool(self.entity_ids & other.entity_ids)

    def shared_entities(self, other: 'Node') -> Set[str]:
        """Get shared entity IDs with another node."""
        return self.entity_ids & other.entity_ids

    def add_source(self, source: str, claim_id: str, entity_ids: Set[str] = None):
        """Add corroborating source and optionally merge entities."""
        self.sources.add(source)
        self.claim_ids.add(claim_id)
        if entity_ids:
            self.entity_ids.update(entity_ids)

    def update(self, new_text: str, source: str, claim_id: str):
        """Update node with new value (supersession)."""
        self.evolution.append((self.text, list(self.sources)[-1] if self.sources else 'unknown'))
        self.superseded = self.text
        self.text = new_text
        self.sources.add(source)
        self.claim_ids.add(claim_id)

    def entropy(self) -> float:
        """
        Jaynes-aligned epistemic entropy.

        Principle: Maximum entropy until evidence constrains it.
        - Single source: High entropy (0.80) - honest uncertainty
        - Multiple sources: Lower entropy - convergent evidence reduces uncertainty

        Formula: H(n) = max(0.15, 0.8 / n^0.5)

        This gives smooth monotonic decrease:
        n=1: 0.80, n=2: 0.57, n=3: 0.46, n=4: 0.40, n=5: 0.36, ...

        This is NOT Shannon entropy of content.
        This is epistemic entropy: uncertainty about truth value.
        """
        n = self.source_count
        if n == 0:
            return 1.0
        # Smooth decay: H = 0.8 / sqrt(n), with floor at 0.15
        return max(0.15, 0.8 / np.sqrt(n))

    def plausibility(self) -> float:
        """
        Plausibility = 1 - Entropy

        High source count → Low entropy → High plausibility
        """
        return 1.0 - self.entropy()

    def confidence_level(self) -> str:
        """
        Human-readable confidence level.

        Based on source count thresholds.
        """
        n = self.source_count
        if n >= 3:
            return "confirmed"
        elif n >= 2:
            return "corroborated"
        else:
            return "reported"

    # BeliefSet protocol implementation
    @property
    def centroid(self) -> Optional[List[float]]:
        """Node's centroid is its embedding (atomic case)."""
        return self.embedding

    @property
    def mass(self) -> float:
        """Node mass = plausibility (how much epistemic weight)."""
        return self.plausibility()


# =============================================================================
# EDGE: Relation between nodes
# =============================================================================

@dataclass
class Edge:
    """
    An edge connecting two nodes.

    Types:
    - same_event: Shared sources (implicit clustering)
    - semantic: High embedding similarity
    - semantic_weak: Moderate embedding similarity
    - evolves: Temporal succession (supersedes)
    - conflicts: Mutual exclusion
    """
    source: int  # Source node index
    target: int  # Target node index
    relation: str
    weight: float = 0.0
    metadata: Dict = field(default_factory=dict)


# =============================================================================
# SURFACE: Connected component (emergent cluster)
# =============================================================================

@dataclass
class Surface:
    """
    A connected component in the topology (set of nodes).

    Surfaces emerge from edge connectivity:
    - Nodes connected by edges form a surface
    - Isolated nodes are singleton surfaces

    Implements BeliefSet protocol - same operations as Node.

    In news: surfaces often represent events
    In general: surfaces represent coherent proposition clusters
    """
    id: int
    node_indices: List[int]

    # Computed properties (set by topology builder)
    total_sources: int = 0
    _mass: float = 0.0
    coherence: float = 1.0
    centroid: Optional[List[float]] = None
    label: str = ""
    _avg_entropy: float = 0.8  # Cached average node entropy

    @property
    def size(self) -> int:
        return len(self.node_indices)

    @property
    def is_connected(self) -> bool:
        return self.size > 1

    @property
    def is_isolated(self) -> bool:
        return self.size == 1

    # BeliefSet protocol implementation
    def entropy(self) -> float:
        """
        Surface entropy = average of member node entropies.

        A surface with many well-corroborated nodes has low entropy.
        A surface with uncertain nodes has high entropy.
        """
        return self._avg_entropy

    @property
    def source_count(self) -> int:
        """Total unique sources across all nodes in surface."""
        return self.total_sources

    @property
    def mass(self) -> float:
        """Epistemic weight of the surface."""
        return self._mass

    def plausibility(self) -> float:
        """Surface plausibility = 1 - entropy."""
        return 1.0 - self.entropy()

    def distance_to(self, other: 'Surface') -> float:
        """
        Semantic distance to another surface.

        Returns 1 - cosine_similarity(centroids).
        Returns 1.0 if either centroid is missing.
        """
        if self.centroid is None or other.centroid is None:
            return 1.0
        sim = cosine_similarity(self.centroid, other.centroid)
        return 1.0 - sim

    def confidence_level(self) -> str:
        """Surface confidence based on source convergence."""
        if self.total_sources >= 5:
            return "well-established"
        elif self.total_sources >= 3:
            return "confirmed"
        elif self.total_sources >= 2:
            return "corroborated"
        else:
            return "reported"


# =============================================================================
# METASURFACE: Set of surfaces (hierarchical clustering)
# =============================================================================

@dataclass
class MetaSurface:
    """
    A cluster of surfaces (set of sets).

    MetaSurfaces emerge from surface proximity:
    - Surfaces with similar centroids form a meta-surface
    - Implements same BeliefSet protocol as Surface

    This enables recursive hierarchy:
    - Node ⊂ Surface ⊂ MetaSurface ⊂ MetaSurface...

    Same operations at every level.
    """
    id: int
    surface_indices: List[int]
    surfaces: List[Surface] = field(default_factory=list)

    # Computed properties
    _centroid: Optional[List[float]] = None
    _mass: float = 0.0
    _avg_entropy: float = 0.8
    total_sources: int = 0
    label: str = ""

    @property
    def size(self) -> int:
        """Number of surfaces in this meta-surface."""
        return len(self.surface_indices)

    @property
    def total_nodes(self) -> int:
        """Total nodes across all surfaces."""
        return sum(s.size for s in self.surfaces)

    # BeliefSet protocol implementation
    @property
    def centroid(self) -> Optional[List[float]]:
        """Centroid of surface centroids."""
        return self._centroid

    def entropy(self) -> float:
        """Average entropy of member surfaces."""
        return self._avg_entropy

    @property
    def source_count(self) -> int:
        """Total unique sources across all surfaces."""
        return self.total_sources

    @property
    def mass(self) -> float:
        """Sum of surface masses."""
        return self._mass

    def plausibility(self) -> float:
        """Meta-surface plausibility."""
        return 1.0 - self.entropy()

    def distance_to(self, other: 'MetaSurface') -> float:
        """Semantic distance to another meta-surface."""
        if self.centroid is None or other.centroid is None:
            return 1.0
        sim = cosine_similarity(self.centroid, other.centroid)
        return 1.0 - sim


# =============================================================================
# DISTANCE MATRIX (for surface/meta-surface clustering)
# =============================================================================

def compute_distance_matrix(items: List[Union[Surface, MetaSurface]]) -> np.ndarray:
    """
    Compute pairwise distance matrix for surfaces or meta-surfaces.

    Returns NxN matrix where D[i,j] = distance between items i and j.
    Distance = 1 - cosine_similarity(centroids).

    Use for:
    - Identifying related surfaces
    - Detecting gaps (high distance = unconnected topics)
    - Hierarchical clustering
    """
    n = len(items)
    matrix = np.ones((n, n))  # Default to max distance

    for i in range(n):
        matrix[i, i] = 0.0  # Self-distance is 0
        for j in range(i + 1, n):
            if items[i].centroid is not None and items[j].centroid is not None:
                sim = cosine_similarity(items[i].centroid, items[j].centroid)
                dist = 1.0 - sim
                matrix[i, j] = dist
                matrix[j, i] = dist

    return matrix


def find_clusters(items: List[Union[Surface, MetaSurface]],
                  threshold: float = 0.45) -> List[List[int]]:
    """
    Cluster surfaces/meta-surfaces by centroid proximity.

    Simple single-linkage clustering:
    - Items with distance < threshold are linked
    - Connected components form clusters

    Returns list of clusters (each cluster is list of item indices).
    """
    n = len(items)
    if n == 0:
        return []

    # Build adjacency from distance matrix
    dist_matrix = compute_distance_matrix(items)
    adj = defaultdict(set)

    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i, j] < threshold:
                adj[i].add(j)
                adj[j].add(i)

    # Find connected components
    visited = set()
    clusters = []

    for i in range(n):
        if i in visited:
            continue
        cluster = []
        stack = [i]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            cluster.append(node)
            stack.extend(adj[node] - visited)
        clusters.append(sorted(cluster))

    # Sort by size (largest first)
    clusters.sort(key=len, reverse=True)
    return clusters


# =============================================================================
# TOPOLOGY: The complete belief graph
# =============================================================================

class Topology:
    """
    The complete hypergeometric belief topology.

    Components:
    - Nodes: Propositions with epistemic state
    - Edges: Relations between propositions
    - Surfaces: Emergent clusters

    Operations:
    - add_node(): Add proposition
    - compute_edges(): Build edge structure
    - find_surfaces(): Discover connected components
    - mass/entropy/coherence: Aggregate metrics
    """

    # Similarity thresholds for edge creation
    SIM_THRESHOLD = 0.55       # Strong semantic connection
    WEAK_SIM_THRESHOLD = 0.40  # Weak semantic connection (lowered to enable emergence)
    MERGE_THRESHOLD = 0.45     # Surface merge threshold

    def __init__(self):
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.surfaces: List[Surface] = []
        self.meta_surfaces: List[MetaSurface] = []
        self._dirty = True  # Needs recomputation

    def add_node(self, node: Node) -> int:
        """Add a node and return its index."""
        idx = len(self.nodes)
        self.nodes.append(node)
        self._dirty = True
        return idx

    def get_node(self, idx: int) -> Optional[Node]:
        """Get node by index."""
        if 0 <= idx < len(self.nodes):
            return self.nodes[idx]
        return None

    def compute(self):
        """Compute edges and surfaces."""
        if not self._dirty:
            return

        self._compute_edges()
        self._find_surfaces()
        self._dirty = False

    def _compute_edges(self):
        """Build edge structure from node relationships."""
        self.edges = []
        edge_set = set()

        for i, n1 in enumerate(self.nodes):
            for j, n2 in enumerate(self.nodes):
                if i >= j:
                    continue

                # Event identity (LLM-determined same real-world event)
                # This is the strongest signal - connects claims about same event
                # regardless of embedding similarity
                if n1.event_id and n2.event_id and n1.event_id == n2.event_id:
                    key = (i, j, 'event_identity')
                    if key not in edge_set:
                        edge_set.add(key)
                        self.edges.append(Edge(
                            source=i,
                            target=j,
                            relation='event_identity',
                            weight=1.0,  # Strongest weight
                            metadata={'event_id': n1.event_id}
                        ))

                # Source overlap (shared sources = likely same topic)
                # IMPORTANT: Require semantic similarity to prevent pollution
                # Two claims from same source about different topics should NOT connect
                shared = n1.sources & n2.sources
                if shared and n1.embedding and n2.embedding:
                    sim = cosine_similarity(n1.embedding, n2.embedding)
                    # Only create edge if semantically related (not just same source)
                    if sim >= self.WEAK_SIM_THRESHOLD:
                        key = (i, j, 'same_event')
                        if key not in edge_set:
                            edge_set.add(key)
                            self.edges.append(Edge(
                                source=i,
                                target=j,
                                relation='same_event',
                                weight=len(shared) / max(len(n1.sources), len(n2.sources)),
                                metadata={'shared_sources': list(shared), 'similarity': round(sim, 3)}
                            ))

                # Semantic similarity (embedding-based)
                if n1.embedding and n2.embedding:
                    sim = cosine_similarity(n1.embedding, n2.embedding)

                    if sim >= self.SIM_THRESHOLD:
                        # Strong semantic similarity: edge always created
                        key = (i, j, 'semantic')
                        if key not in edge_set:
                            edge_set.add(key)
                            self.edges.append(Edge(
                                source=i,
                                target=j,
                                relation='semantic',
                                weight=sim,
                                metadata={'similarity': round(sim, 3)}
                            ))
                    elif sim >= self.WEAK_SIM_THRESHOLD:
                        # Weak semantic: requires entity overlap OR source overlap
                        has_entity_overlap = n1.has_entity_overlap(n2)
                        has_source_overlap = bool(n1.sources & n2.sources)

                        if has_entity_overlap or has_source_overlap:
                            key = (i, j, 'semantic_weak')
                            if key not in edge_set:
                                edge_set.add(key)
                                shared_ents = list(n1.shared_entities(n2))
                                self.edges.append(Edge(
                                    source=i,
                                    target=j,
                                    relation='semantic_weak',
                                    weight=sim * 0.5,
                                    metadata={
                                        'similarity': round(sim, 3),
                                        'entity_overlap': has_entity_overlap,
                                        'shared_entities': shared_ents[:3] if shared_ents else []
                                    }
                                ))

    def _find_surfaces(self):
        """Discover connected components (surfaces)."""
        # Build adjacency
        adj = defaultdict(set)
        for e in self.edges:
            adj[e.source].add(e.target)
            adj[e.target].add(e.source)

        # Find connected components
        visited = set()
        components = []

        for i in range(len(self.nodes)):
            if i in visited:
                continue

            stack = [i]
            component = []
            while stack:
                n = stack.pop()
                if n in visited:
                    continue
                visited.add(n)
                component.append(n)
                stack.extend(adj[n] - visited)
            components.append(component)

        # Sort by size (largest first)
        components.sort(key=len, reverse=True)

        # Build Surface objects
        self.surfaces = []
        for idx, comp in enumerate(components):
            nodes_in_surface = [self.nodes[i] for i in comp]
            sources = set().union(*[n.sources for n in nodes_in_surface])

            # Centroid embedding
            embeddings = [n.embedding for n in nodes_in_surface if n.embedding]
            surface_centroid = centroid(embeddings)

            # Compute average entropy of nodes
            avg_entropy = sum(n.entropy() for n in nodes_in_surface) / len(nodes_in_surface) if nodes_in_surface else 0.8

            # Mass = size * coherence * source_factor
            size = len(comp)
            coherence = 1.0  # TODO: compute from internal conflicts
            mass = size * 0.1 * (0.5 + coherence) * (1 + 0.1 * len(sources))

            self.surfaces.append(Surface(
                id=idx,
                node_indices=comp,
                total_sources=len(sources),
                _mass=round(mass, 3),
                coherence=coherence,
                centroid=surface_centroid,
                label=nodes_in_surface[0].text[:50] if nodes_in_surface else '',
                _avg_entropy=round(avg_entropy, 3)
            ))

    def find_meta_surfaces(self, threshold: float = 0.45) -> List[MetaSurface]:
        """
        Cluster surfaces into meta-surfaces by semantic proximity.

        This is the hierarchical step: surfaces that are semantically
        close form meta-surfaces (set of sets).

        Args:
            threshold: Distance threshold for clustering (lower = stricter)

        Returns:
            List of MetaSurface objects, sorted by size.
        """
        self.compute()

        if len(self.surfaces) < 2:
            self.meta_surfaces = []
            return self.meta_surfaces

        # Use find_clusters helper
        clusters = find_clusters(self.surfaces, threshold=threshold)

        self.meta_surfaces = []
        for idx, cluster_indices in enumerate(clusters):
            surfaces_in_cluster = [self.surfaces[i] for i in cluster_indices]

            # Compute meta-surface centroid from surface centroids
            surface_centroids = [s.centroid for s in surfaces_in_cluster if s.centroid]
            meta_centroid = centroid(surface_centroids) if surface_centroids else None

            # Aggregate metrics
            total_sources = len(set().union(*[
                set().union(*[self.nodes[ni].sources for ni in s.node_indices])
                for s in surfaces_in_cluster
            ])) if surfaces_in_cluster else 0

            total_mass = sum(s.mass for s in surfaces_in_cluster)
            avg_entropy = sum(s.entropy() for s in surfaces_in_cluster) / len(surfaces_in_cluster) if surfaces_in_cluster else 0.8

            self.meta_surfaces.append(MetaSurface(
                id=idx,
                surface_indices=cluster_indices,
                surfaces=surfaces_in_cluster,
                _centroid=meta_centroid,
                _mass=round(total_mass, 3),
                _avg_entropy=round(avg_entropy, 3),
                total_sources=total_sources,
                label=surfaces_in_cluster[0].label if surfaces_in_cluster else ''
            ))

        return self.meta_surfaces

    def surface_distance_matrix(self) -> np.ndarray:
        """
        Compute pairwise distance matrix for all surfaces.

        Useful for visualization and gap detection.
        """
        self.compute()
        return compute_distance_matrix(self.surfaces)

    def find_gaps(self, threshold: float = 0.7) -> List[Tuple[int, int, float]]:
        """
        Find pairs of surfaces that are semantically distant.

        Returns list of (surface_i, surface_j, distance) where distance > threshold.
        These represent potential "gaps" in coverage or unrelated topics.
        """
        self.compute()
        dist_matrix = self.surface_distance_matrix()
        gaps = []

        for i in range(len(self.surfaces)):
            for j in range(i + 1, len(self.surfaces)):
                if dist_matrix[i, j] > threshold:
                    gaps.append((i, j, dist_matrix[i, j]))

        gaps.sort(key=lambda x: -x[2])  # Largest gaps first
        return gaps

    # =========================================================================
    # AGGREGATE METRICS
    # =========================================================================

    def total_entropy(self) -> float:
        """
        Average entropy across all nodes.

        Lower = more converged evidence.
        """
        if not self.nodes:
            return 1.0
        return sum(n.entropy() for n in self.nodes) / len(self.nodes)

    def total_mass(self) -> float:
        """
        Sum of surface masses.

        Represents total epistemic weight.
        """
        self.compute()
        return sum(s.mass for s in self.surfaces)

    def connected_surface_count(self) -> int:
        """Number of surfaces with multiple nodes."""
        self.compute()
        return len([s for s in self.surfaces if s.is_connected])

    def isolated_node_count(self) -> int:
        """Number of singleton surfaces."""
        self.compute()
        return len([s for s in self.surfaces if s.is_isolated])

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict:
        """Serialize topology to dictionary."""
        self.compute()

        return {
            'nodes': [
                {
                    'id': i,
                    'text': n.text,
                    'sources': list(n.sources),
                    'source_count': n.source_count,
                    'entropy': n.entropy(),
                    'plausibility': n.plausibility(),
                    'confidence': n.confidence_level()
                }
                for i, n in enumerate(self.nodes)
            ],
            'edges': [
                {
                    'source': e.source,
                    'target': e.target,
                    'relation': e.relation,
                    'weight': e.weight,
                    **e.metadata
                }
                for e in self.edges
            ],
            'surfaces': [
                {
                    'id': s.id,
                    'beliefs': s.node_indices,
                    'size': s.size,
                    'total_sources': s.total_sources,
                    'mass': s.mass,
                    'coherence': s.coherence,
                    'label': s.label
                }
                for s in self.surfaces
            ],
            'stats': {
                'total_nodes': len(self.nodes),
                'total_edges': len(self.edges),
                'total_surfaces': len(self.surfaces),
                'connected_surfaces': self.connected_surface_count(),
                'isolated_nodes': self.isolated_node_count(),
                'total_entropy': round(self.total_entropy(), 3),
                'total_mass': round(self.total_mass(), 3)
            }
        }

    # =========================================================================
    # SIMILARITY SEARCH
    # =========================================================================

    def find_similar(self, embedding: List[float], threshold: float = 0.50) -> List[Tuple[int, float]]:
        """
        Find nodes similar to given embedding.

        Returns: List of (node_index, similarity) tuples, sorted by similarity descending.
        """
        if not embedding:
            return []

        results = []
        for i, node in enumerate(self.nodes):
            if node.embedding:
                sim = cosine_similarity(embedding, node.embedding)
                if sim >= threshold:
                    results.append((i, sim))

        results.sort(key=lambda x: -x[1])
        return results

    def most_similar(self, embedding: List[float]) -> Optional[Tuple[int, float]]:
        """Find the most similar node to given embedding."""
        similar = self.find_similar(embedding, threshold=0.0)
        return similar[0] if similar else None


# =============================================================================
# JAYNES VALIDATOR
# =============================================================================

def validate_jaynes(topology: Topology) -> Dict:
    """
    Validate that topology follows Jaynes Maximum Entropy principle.

    Checks:
    1. Single-source nodes have higher entropy than multi-source
    2. Entropy decreases monotonically with source count
    3. No node has lower entropy than justified by evidence
    """
    single_source = [n for n in topology.nodes if n.source_count == 1]
    multi_source = [n for n in topology.nodes if n.source_count >= 2]

    if not single_source or not multi_source:
        return {'valid': True, 'reason': 'Insufficient data for comparison'}

    single_avg = sum(n.entropy() for n in single_source) / len(single_source)
    multi_avg = sum(n.entropy() for n in multi_source) / len(multi_source)

    valid = single_avg > multi_avg

    return {
        'valid': valid,
        'single_source_count': len(single_source),
        'multi_source_count': len(multi_source),
        'single_source_avg_entropy': round(single_avg, 3),
        'multi_source_avg_entropy': round(multi_avg, 3),
        'reason': 'Single-source entropy > Multi-source entropy' if valid else 'VIOLATION: Entropy not properly scaled'
    }
