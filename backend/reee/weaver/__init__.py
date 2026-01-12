"""
REEE Weaver Adapters - DB boundary layer for TopologyKernel.

This module contains thin adapters that:
- Load data from DB into kernel contracts
- Apply kernel output back to DB
- Keep DB logic OUTSIDE the pure kernel

Layers:
- SnapshotLoader: DB → PartitionSnapshot
- EvidenceBuilder: DB → ClaimEvidence
- DeltaApplier: TopologyDelta → DB
- ShadowRunner: Validation runner for kernel vs legacy comparison
"""

from .snapshot_loader import SnapshotLoader
from .evidence_builder import EvidenceBuilder, EnrichedEvidenceBuilder
from .delta_applier import DeltaApplier, ShadowDeltaApplier
from .shadow_runner import ShadowRunner, run_shadow_validation

__all__ = [
    "SnapshotLoader",
    "EvidenceBuilder",
    "EnrichedEvidenceBuilder",
    "DeltaApplier",
    "ShadowDeltaApplier",
    "ShadowRunner",
    "run_shadow_validation",
]
