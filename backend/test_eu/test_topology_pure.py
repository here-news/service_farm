"""
Pure Topology Tests
====================

Domain-agnostic tests for hypergeometric topology behavior.

Tests focus on:
1. Jaynes Maximum Entropy principle
2. Source convergence → entropy reduction
3. Surface emergence from edge connectivity
4. Node/Edge/Surface mathematical properties

NO LLM calls. NO domain assumptions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_eu.core.topology import (
    Topology, Node, Edge, Surface, Relation,
    cosine_similarity, centroid, validate_jaynes
)
import numpy as np


def test_node_entropy_single_source():
    """Single-source nodes have high entropy (0.80)."""
    node = Node(id="n1", text="Proposition A", sources={"Source1"}, claim_ids={"c1"})
    assert abs(node.entropy() - 0.80) < 0.01, f"Expected ~0.80, got {node.entropy()}"
    assert abs(node.plausibility() - 0.20) < 0.01, f"Expected ~0.20, got {node.plausibility()}"
    print("PASS: Single-source entropy = 0.80")


def test_node_entropy_multi_source():
    """Multi-source nodes have lower entropy. Formula: H = 0.8 / sqrt(n)"""
    node2 = Node(id="n1", text="Proposition A", sources={"S1", "S2"}, claim_ids={"c1", "c2"})
    node3 = Node(id="n2", text="Proposition B", sources={"S1", "S2", "S3"}, claim_ids={"c1", "c2", "c3"})
    node4 = Node(id="n3", text="Proposition C", sources={"S1", "S2", "S3", "S4"}, claim_ids={"c1", "c2", "c3", "c4"})

    # H = 0.8 / sqrt(n): n=2 -> 0.566, n=3 -> 0.462, n=4 -> 0.40
    assert node2.entropy() < 0.80, f"Expected < 0.80, got {node2.entropy()}"
    assert node3.entropy() < node2.entropy(), f"Expected < {node2.entropy()}, got {node3.entropy()}"
    assert node4.entropy() < node3.entropy(), f"Expected < {node3.entropy()}, got {node4.entropy()}"

    print("PASS: Multi-source entropy decreases with more sources")


def test_jaynes_monotonic_decrease():
    """Entropy must decrease monotonically with source count."""
    entropies = []
    for n in range(1, 10):
        node = Node(
            id=f"n{n}",
            text=f"Proposition {n}",
            sources={f"S{i}" for i in range(n)},
            claim_ids={f"c{i}" for i in range(n)}
        )
        entropies.append(node.entropy())

    for i in range(1, len(entropies)):
        assert entropies[i] <= entropies[i-1], \
            f"Jaynes violation: H({i+1}) = {entropies[i]} > H({i}) = {entropies[i-1]}"

    print("PASS: Entropy monotonically decreases with source count")


def test_confidence_levels():
    """Confidence levels based on source count."""
    n1 = Node(id="n1", text="P1", sources={"S1"})
    n2 = Node(id="n2", text="P2", sources={"S1", "S2"})
    n3 = Node(id="n3", text="P3", sources={"S1", "S2", "S3"})

    assert n1.confidence_level() == "reported"
    assert n2.confidence_level() == "corroborated"
    assert n3.confidence_level() == "confirmed"

    print("PASS: Confidence levels correctly assigned")


def test_topology_empty():
    """Empty topology has correct defaults."""
    topo = Topology()
    assert len(topo.nodes) == 0
    assert len(topo.edges) == 0
    assert topo.total_entropy() == 1.0

    print("PASS: Empty topology defaults")


def test_topology_add_nodes():
    """Adding nodes to topology."""
    topo = Topology()

    idx1 = topo.add_node(Node(id="n1", text="A", sources={"S1"}))
    idx2 = topo.add_node(Node(id="n2", text="B", sources={"S1", "S2"}))

    assert idx1 == 0
    assert idx2 == 1
    assert len(topo.nodes) == 2

    print("PASS: Nodes added correctly")


def test_source_overlap_creates_edges():
    """Nodes with shared sources get connected."""
    topo = Topology()

    topo.add_node(Node(id="n1", text="A", sources={"S1", "S2"}))
    topo.add_node(Node(id="n2", text="B", sources={"S2", "S3"}))  # Shares S2
    topo.add_node(Node(id="n3", text="C", sources={"S4"}))  # No overlap

    topo.compute()

    # Should have edge between n1 and n2 (share S2)
    edge_pairs = [(e.source, e.target) for e in topo.edges]
    assert (0, 1) in edge_pairs or (1, 0) in edge_pairs, "Expected edge between nodes sharing S2"

    # Should NOT have edge between n3 and others
    assert (0, 2) not in edge_pairs and (2, 0) not in edge_pairs
    assert (1, 2) not in edge_pairs and (2, 1) not in edge_pairs

    print("PASS: Source overlap creates edges")


def test_semantic_similarity_creates_edges():
    """High embedding similarity creates edges."""
    # Create simple embeddings
    emb1 = [1.0, 0.0, 0.0]
    emb2 = [0.95, 0.05, 0.0]  # Very similar to emb1
    emb3 = [0.0, 1.0, 0.0]    # Orthogonal to emb1

    topo = Topology()
    topo.add_node(Node(id="n1", text="A", sources={"S1"}, embedding=emb1))
    topo.add_node(Node(id="n2", text="B", sources={"S2"}, embedding=emb2))
    topo.add_node(Node(id="n3", text="C", sources={"S3"}, embedding=emb3))

    topo.compute()

    # Find edge between n1 and n2
    edge_01 = next((e for e in topo.edges if (e.source == 0 and e.target == 1) or (e.source == 1 and e.target == 0)), None)
    assert edge_01 is not None, "Expected edge between similar embeddings"
    assert edge_01.relation in ('semantic', 'semantic_weak')

    print("PASS: Semantic similarity creates edges")


def test_surfaces_emerge():
    """Connected components form surfaces."""
    topo = Topology()

    # Cluster 1: n0, n1 connected via source overlap
    topo.add_node(Node(id="n0", text="A", sources={"S1"}))
    topo.add_node(Node(id="n1", text="B", sources={"S1"}))

    # Cluster 2: n2 isolated
    topo.add_node(Node(id="n2", text="C", sources={"S2"}))

    topo.compute()

    assert len(topo.surfaces) == 2, f"Expected 2 surfaces, got {len(topo.surfaces)}"

    # Largest surface should have 2 nodes (n0, n1)
    largest = topo.surfaces[0]
    assert largest.size == 2, f"Expected size 2, got {largest.size}"

    # Second surface should be isolated
    isolated = topo.surfaces[1]
    assert isolated.size == 1
    assert isolated.is_isolated

    print("PASS: Surfaces emerge from connectivity")


def test_surface_mass():
    """Surface mass calculation."""
    topo = Topology()

    # Add nodes with varying sources
    topo.add_node(Node(id="n0", text="A", sources={"S1"}))
    topo.add_node(Node(id="n1", text="B", sources={"S1", "S2"}))
    topo.add_node(Node(id="n2", text="C", sources={"S1", "S2", "S3"}))

    topo.compute()

    # All connected via S1
    assert len(topo.surfaces) == 1
    surface = topo.surfaces[0]

    # Mass should be > 0
    assert surface.mass > 0, f"Expected positive mass, got {surface.mass}"

    print(f"PASS: Surface mass = {surface.mass}")


def test_validate_jaynes():
    """Jaynes validation function."""
    topo = Topology()

    # Mix of single and multi-source nodes
    topo.add_node(Node(id="n0", text="A", sources={"S1"}))
    topo.add_node(Node(id="n1", text="B", sources={"S2"}))
    topo.add_node(Node(id="n2", text="C", sources={"S1", "S2", "S3"}))

    result = validate_jaynes(topo)

    assert result['valid'], f"Jaynes validation failed: {result['reason']}"
    assert result['single_source_avg_entropy'] > result['multi_source_avg_entropy']

    print("PASS: Jaynes validation")


def test_topology_serialization():
    """Topology to_dict() produces valid structure."""
    topo = Topology()

    topo.add_node(Node(id="n0", text="Proposition A", sources={"S1", "S2"}))
    topo.add_node(Node(id="n1", text="Proposition B", sources={"S2"}))

    data = topo.to_dict()

    assert 'nodes' in data
    assert 'edges' in data
    assert 'surfaces' in data
    assert 'stats' in data

    assert len(data['nodes']) == 2
    assert data['nodes'][0]['source_count'] == 2
    assert data['nodes'][0]['confidence'] == 'corroborated'

    print("PASS: Topology serialization")


def test_find_similar():
    """Similarity search in topology."""
    topo = Topology()

    emb1 = [1.0, 0.0, 0.0]
    emb2 = [0.9, 0.1, 0.0]
    emb3 = [0.0, 1.0, 0.0]

    topo.add_node(Node(id="n0", text="A", sources={"S1"}, embedding=emb1))
    topo.add_node(Node(id="n1", text="B", sources={"S2"}, embedding=emb2))
    topo.add_node(Node(id="n2", text="C", sources={"S3"}, embedding=emb3))

    # Search for similar to emb1
    query = [0.95, 0.05, 0.0]
    results = topo.find_similar(query, threshold=0.5)

    assert len(results) >= 2, f"Expected at least 2 results, got {len(results)}"
    assert results[0][0] in [0, 1], "Top result should be n0 or n1"

    print("PASS: Similarity search")


def test_cosine_similarity():
    """Cosine similarity function."""
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    c = [0.0, 1.0, 0.0]
    d = [-1.0, 0.0, 0.0]

    assert cosine_similarity(a, b) == 1.0, "Same vector should be 1.0"
    assert cosine_similarity(a, c) == 0.0, "Orthogonal should be 0.0"
    assert cosine_similarity(a, d) == -1.0, "Opposite should be -1.0"

    print("PASS: Cosine similarity")


def test_centroid():
    """Centroid calculation."""
    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ]
    c = centroid(embeddings)

    expected = [2/3, 2/3]
    assert abs(c[0] - expected[0]) < 0.01
    assert abs(c[1] - expected[1]) < 0.01

    print("PASS: Centroid calculation")


def test_node_evolution():
    """Node update tracking."""
    node = Node(id="n1", text="Value is 10", sources={"S1"})
    node.update("Value is 20", "S2", "c2")

    assert node.text == "Value is 20"
    assert node.superseded == "Value is 10"
    assert len(node.evolution) == 1
    assert node.source_count == 2

    print("PASS: Node evolution")


def run_all_tests():
    """Run all pure topology tests."""
    print("=" * 60)
    print("PURE TOPOLOGY TESTS")
    print("=" * 60)
    print()

    tests = [
        test_node_entropy_single_source,
        test_node_entropy_multi_source,
        test_jaynes_monotonic_decrease,
        test_confidence_levels,
        test_topology_empty,
        test_topology_add_nodes,
        test_source_overlap_creates_edges,
        test_semantic_similarity_creates_edges,
        test_surfaces_emerge,
        test_surface_mass,
        test_validate_jaynes,
        test_topology_serialization,
        test_find_similar,
        test_cosine_similarity,
        test_centroid,
        test_node_evolution,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {test.__name__}: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
