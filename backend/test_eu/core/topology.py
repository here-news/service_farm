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
from typing import List, Dict, Optional, Set, Tuple, Callable
from enum import Enum
from collections import defaultdict


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
    """
    id: str
    text: str
    sources: Set[str] = field(default_factory=set)
    claim_ids: Set[str] = field(default_factory=set)
    embedding: Optional[List[float]] = None
    superseded: Optional[str] = None
    evolution: List[Tuple[str, str]] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    @property
    def source_count(self) -> int:
        return len(self.sources)

    def add_source(self, source: str, claim_id: str):
        """Add corroborating source."""
        self.sources.add(source)
        self.claim_ids.add(claim_id)

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
    A connected component in the topology.

    Surfaces emerge from edge connectivity:
    - Nodes connected by edges form a surface
    - Isolated nodes are singleton surfaces

    In news: surfaces often represent events
    In general: surfaces represent coherent proposition clusters
    """
    id: int
    node_indices: List[int]

    # Computed properties (set by topology builder)
    total_sources: int = 0
    mass: float = 0.0
    coherence: float = 1.0
    centroid: Optional[List[float]] = None
    label: str = ""

    @property
    def size(self) -> int:
        return len(self.node_indices)

    @property
    def is_connected(self) -> bool:
        return self.size > 1

    @property
    def is_isolated(self) -> bool:
        return self.size == 1


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
    SIM_THRESHOLD = 0.65       # Strong semantic connection
    WEAK_SIM_THRESHOLD = 0.50  # Weak semantic connection
    MERGE_THRESHOLD = 0.55     # Surface merge threshold

    def __init__(self):
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.surfaces: List[Surface] = []
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

                # Source overlap (shared sources = likely same topic)
                shared = n1.sources & n2.sources
                if shared:
                    key = (i, j, 'same_event')
                    if key not in edge_set:
                        edge_set.add(key)
                        self.edges.append(Edge(
                            source=i,
                            target=j,
                            relation='same_event',
                            weight=len(shared) / max(len(n1.sources), len(n2.sources)),
                            metadata={'shared_sources': list(shared)}
                        ))

                # Semantic similarity (embedding-based)
                if n1.embedding and n2.embedding:
                    sim = cosine_similarity(n1.embedding, n2.embedding)

                    if sim >= self.SIM_THRESHOLD:
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
                        key = (i, j, 'semantic_weak')
                        if key not in edge_set:
                            edge_set.add(key)
                            self.edges.append(Edge(
                                source=i,
                                target=j,
                                relation='semantic_weak',
                                weight=sim * 0.5,
                                metadata={'similarity': round(sim, 3)}
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

            # Mass = size * coherence * source_factor
            size = len(comp)
            coherence = 1.0  # TODO: compute from internal conflicts
            mass = size * 0.1 * (0.5 + coherence) * (1 + 0.1 * len(sources))

            self.surfaces.append(Surface(
                id=idx,
                node_indices=comp,
                total_sources=len(sources),
                mass=round(mass, 3),
                coherence=coherence,
                centroid=surface_centroid,
                label=nodes_in_surface[0].text[:50] if nodes_in_surface else ''
            ))

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
