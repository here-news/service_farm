"""Aboutness scoring module for L2 -> L3 surface-to-event clustering."""

from .scorer import AboutnessScorer, compute_aboutness_edges, compute_events_from_aboutness

__all__ = ['AboutnessScorer', 'compute_aboutness_edges', 'compute_events_from_aboutness']
