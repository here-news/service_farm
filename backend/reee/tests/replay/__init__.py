"""
Replay Infrastructure for Kernel Regression Testing
====================================================

Records and replays kernel inputs for deterministic regression testing.

Components:
- recorder.py: Record kernel inputs to snapshot files
- replayer.py: Replay snapshots and compare results
"""

from .recorder import SnapshotRecorder, record_snapshot
from .replayer import SnapshotReplayer, replay_snapshot, ReplayResult

__all__ = [
    'SnapshotRecorder',
    'SnapshotReplayer',
    'record_snapshot',
    'replay_snapshot',
    'ReplayResult',
]
