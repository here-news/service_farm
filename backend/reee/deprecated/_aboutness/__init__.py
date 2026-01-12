"""Aboutness scoring module for L2 -> L3 surface-to-event clustering."""

from .scorer import AboutnessScorer, compute_aboutness_edges, compute_events_from_aboutness
from .metrics import (
    B3Metrics,
    PurityMetrics,
    ClusteringEvaluation,
    compute_b3_metrics,
    compute_purity_metrics,
    evaluate_clustering,
    print_evaluation_report,
)

__all__ = [
    'AboutnessScorer',
    'compute_aboutness_edges',
    'compute_events_from_aboutness',
    'B3Metrics',
    'PurityMetrics',
    'ClusteringEvaluation',
    'compute_b3_metrics',
    'compute_purity_metrics',
    'evaluate_clustering',
    'print_evaluation_report',
]
