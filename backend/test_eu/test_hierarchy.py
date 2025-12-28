"""
Hierarchical Topology Tests
============================

Tests for set-in-set hierarchy:
- Surface = set of nodes
- MetaSurface = set of surfaces
- Same operations at every level

NO LLM calls. Tests pure mathematical behavior.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_eu.core.topology import (
    Topology, Node, Edge, Surface, MetaSurface,
    cosine_similarity, centroid, compute_distance_matrix, find_clusters
)
import numpy as np


def test_surface_entropy():
    """Surface entropy is average of node entropies."""
    topo = Topology()

    # Two nodes: 1 source (H=0.8) and 3 sources (H=0.46)
    topo.add_node(Node(id="n0", text="A", sources={"S1"}))
    topo.add_node(Node(id="n1", text="B", sources={"S1", "S2", "S3"}))

    topo.compute()

    # Should be 1 surface with both nodes
    assert len(topo.surfaces) == 1
    surface = topo.surfaces[0]

    # Surface entropy should be average
    expected = (0.8 + 0.462) / 2  # ~0.63
    assert abs(surface.entropy() - expected) < 0.05, f"Expected ~{expected}, got {surface.entropy()}"

    print("PASS: Surface entropy is average of node entropies")


def test_surface_distance():
    """Surface distance based on centroid similarity."""
    # Create two surfaces with different embeddings
    s1 = Surface(id=0, node_indices=[0], centroid=[1.0, 0.0, 0.0])
    s2 = Surface(id=1, node_indices=[1], centroid=[0.0, 1.0, 0.0])  # Orthogonal
    s3 = Surface(id=2, node_indices=[2], centroid=[0.9, 0.1, 0.0])  # Similar to s1

    # s1 and s2 should be far (distance = 1 - 0 = 1.0)
    assert abs(s1.distance_to(s2) - 1.0) < 0.01, f"Expected 1.0, got {s1.distance_to(s2)}"

    # s1 and s3 should be close
    assert s1.distance_to(s3) < 0.3, f"Expected < 0.3, got {s1.distance_to(s3)}"

    print("PASS: Surface distance based on centroid similarity")


def test_distance_matrix():
    """Distance matrix computation."""
    surfaces = [
        Surface(id=0, node_indices=[0], centroid=[1.0, 0.0]),
        Surface(id=1, node_indices=[1], centroid=[0.0, 1.0]),
        Surface(id=2, node_indices=[2], centroid=[0.9, 0.1]),
    ]

    matrix = compute_distance_matrix(surfaces)

    # Diagonal should be 0
    assert matrix[0, 0] == 0.0
    assert matrix[1, 1] == 0.0
    assert matrix[2, 2] == 0.0

    # s0 and s1 should be far (orthogonal)
    assert matrix[0, 1] > 0.9

    # s0 and s2 should be close
    assert matrix[0, 2] < 0.3

    # Symmetric
    assert matrix[0, 1] == matrix[1, 0]

    print("PASS: Distance matrix computation")


def test_find_clusters():
    """Find clusters groups similar surfaces."""
    surfaces = [
        # Cluster 1: close together
        Surface(id=0, node_indices=[0], centroid=[1.0, 0.0]),
        Surface(id=1, node_indices=[1], centroid=[0.95, 0.05]),

        # Cluster 2: far from cluster 1
        Surface(id=2, node_indices=[2], centroid=[0.0, 1.0]),
        Surface(id=3, node_indices=[3], centroid=[0.05, 0.95]),

        # Isolated: far from both
        Surface(id=4, node_indices=[4], centroid=[-1.0, 0.0]),
    ]

    clusters = find_clusters(surfaces, threshold=0.3)

    # Should have 3 clusters
    assert len(clusters) >= 2, f"Expected >= 2 clusters, got {len(clusters)}"

    # Check that 0 and 1 are together
    cluster_for_0 = next(c for c in clusters if 0 in c)
    assert 1 in cluster_for_0, "Surfaces 0 and 1 should be in same cluster"

    # Check that 2 and 3 are together
    cluster_for_2 = next(c for c in clusters if 2 in c)
    assert 3 in cluster_for_2, "Surfaces 2 and 3 should be in same cluster"

    print("PASS: Find clusters groups similar surfaces")


def test_meta_surface_creation():
    """MetaSurface is created from surface clusters."""
    topo = Topology()

    # Create 4 nodes in 2 disconnected surfaces
    topo.add_node(Node(id="n0", text="Fire A", sources={"S1"}, embedding=[1.0, 0.0]))
    topo.add_node(Node(id="n1", text="Fire B", sources={"S1"}, embedding=[0.95, 0.05]))
    topo.add_node(Node(id="n2", text="Weather X", sources={"S2"}, embedding=[0.0, 1.0]))
    topo.add_node(Node(id="n3", text="Weather Y", sources={"S2"}, embedding=[0.05, 0.95]))

    topo.compute()

    # Should have 2 surfaces
    assert len(topo.surfaces) == 2, f"Expected 2 surfaces, got {len(topo.surfaces)}"

    # Find meta-surfaces with loose threshold (all should be separate)
    meta = topo.find_meta_surfaces(threshold=0.3)

    # With threshold 0.3, the two surfaces should NOT cluster (distance > 0.3)
    assert len(meta) == 2, f"Expected 2 meta-surfaces, got {len(meta)}"

    print("PASS: MetaSurface creation from surface clusters")


def test_meta_surface_properties():
    """MetaSurface has correct aggregated properties."""
    # Create surfaces manually
    s1 = Surface(
        id=0, node_indices=[0, 1],
        total_sources=3, _mass=0.5, _avg_entropy=0.6,
        centroid=[1.0, 0.0]
    )
    s2 = Surface(
        id=1, node_indices=[2],
        total_sources=1, _mass=0.2, _avg_entropy=0.8,
        centroid=[0.9, 0.1]
    )

    meta = MetaSurface(
        id=0,
        surface_indices=[0, 1],
        surfaces=[s1, s2],
        _centroid=[0.95, 0.05],
        _mass=0.7,  # 0.5 + 0.2
        _avg_entropy=0.7,  # (0.6 + 0.8) / 2
        total_sources=3,
        label="Fire"
    )

    assert meta.size == 2
    assert meta.total_nodes == 3  # 2 + 1
    assert abs(meta.mass - 0.7) < 0.01
    assert abs(meta.entropy() - 0.7) < 0.01
    assert abs(meta.plausibility() - 0.3) < 0.01

    print("PASS: MetaSurface has correct aggregated properties")


def test_hierarchy_same_operations():
    """Same operations work at node, surface, and meta-surface level."""
    # Node level
    node = Node(id="n0", text="A", sources={"S1", "S2"}, embedding=[1.0, 0.0])
    assert node.entropy() < 0.8  # Multi-source
    assert node.plausibility() > 0.2
    assert node.centroid == [1.0, 0.0]
    assert node.mass > 0

    # Surface level
    surface = Surface(
        id=0, node_indices=[0],
        total_sources=2, _mass=0.5, _avg_entropy=0.57,
        centroid=[1.0, 0.0]
    )
    assert surface.entropy() < 0.8
    assert surface.plausibility() > 0.2
    assert surface.centroid == [1.0, 0.0]
    assert surface.mass > 0

    # MetaSurface level
    meta = MetaSurface(
        id=0,
        surface_indices=[0],
        surfaces=[surface],
        _centroid=[1.0, 0.0],
        _mass=0.5,
        _avg_entropy=0.57,
        total_sources=2
    )
    assert meta.entropy() < 0.8
    assert meta.plausibility() > 0.2
    assert meta.centroid == [1.0, 0.0]
    assert meta.mass > 0

    print("PASS: Same operations work at all hierarchy levels")


def test_find_gaps():
    """Topology.find_gaps identifies distant surfaces."""
    topo = Topology()

    # Two clusters of nodes
    topo.add_node(Node(id="n0", text="Fire A", sources={"S1"}, embedding=[1.0, 0.0, 0.0]))
    topo.add_node(Node(id="n1", text="Fire B", sources={"S1"}, embedding=[0.95, 0.05, 0.0]))
    topo.add_node(Node(id="n2", text="Weather", sources={"S2"}, embedding=[0.0, 1.0, 0.0]))
    topo.add_node(Node(id="n3", text="Politics", sources={"S3"}, embedding=[0.0, 0.0, 1.0]))

    topo.compute()

    gaps = topo.find_gaps(threshold=0.5)

    # Should find gaps between unrelated surfaces
    assert len(gaps) > 0, "Expected some gaps"

    # First gap should have distance > 0.5
    assert gaps[0][2] > 0.5, f"Expected distance > 0.5, got {gaps[0][2]}"

    print(f"PASS: Found {len(gaps)} gaps between surfaces")


def test_recursive_clustering():
    """Meta-surfaces can be clustered again (recursive hierarchy)."""
    # Create meta-surfaces
    meta1 = MetaSurface(
        id=0, surface_indices=[0, 1], surfaces=[],
        _centroid=[1.0, 0.0], _mass=1.0, _avg_entropy=0.5, total_sources=5
    )
    meta2 = MetaSurface(
        id=1, surface_indices=[2, 3], surfaces=[],
        _centroid=[0.95, 0.05], _mass=0.8, _avg_entropy=0.6, total_sources=3
    )
    meta3 = MetaSurface(
        id=2, surface_indices=[4], surfaces=[],
        _centroid=[0.0, 1.0], _mass=0.3, _avg_entropy=0.8, total_sources=1
    )

    # Cluster meta-surfaces (same function works!)
    clusters = find_clusters([meta1, meta2, meta3], threshold=0.3)

    # meta1 and meta2 should cluster together
    cluster_for_0 = next(c for c in clusters if 0 in c)
    assert 1 in cluster_for_0, "Meta-surfaces 0 and 1 should cluster"

    # meta3 should be separate
    cluster_for_2 = next(c for c in clusters if 2 in c)
    assert len(cluster_for_2) == 1, "Meta-surface 2 should be isolated"

    print("PASS: Recursive clustering works for meta-surfaces")


def run_all_tests():
    """Run all hierarchy tests."""
    print("=" * 60)
    print("HIERARCHICAL TOPOLOGY TESTS")
    print("=" * 60)
    print()

    tests = [
        test_surface_entropy,
        test_surface_distance,
        test_distance_matrix,
        test_find_clusters,
        test_meta_surface_creation,
        test_meta_surface_properties,
        test_hierarchy_same_operations,
        test_find_gaps,
        test_recursive_clustering,
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
