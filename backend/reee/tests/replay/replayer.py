"""
Snapshot Replayer for Kernel Regression Testing
================================================

Replays kernel inputs from snapshot files and compares results
to expected outcomes for deterministic regression testing.

Usage:
    from reee.tests.replay import replay_snapshot

    result = replay_snapshot("fixtures/replay_wfc_snapshot.json")

    assert result.deterministic, "Replay should be deterministic"
    assert result.matches_expected, "Should match expected outcomes"
    for diff in result.diffs:
        print(f"Changed: {diff}")
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from reee.types import Event, Surface
from reee.builders.story_builder import StoryBuilder, StoryBuilderResult, CompleteStory


@dataclass
class DecisionDiff:
    """Represents a diff in kernel decision."""
    incident_id: str
    expected: str  # "core", "periphery", "reject"
    actual: str
    expected_reason: Optional[str] = None
    actual_reason: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.incident_id}: expected={self.expected} ({self.expected_reason}), actual={self.actual} ({self.actual_reason})"


@dataclass
class ReplayResult:
    """Result of replaying a snapshot."""
    snapshot_id: str
    deterministic: bool  # Same result on multiple runs
    matches_expected: bool  # Matches expected_outcome in meta

    # Actual results
    stories_formed: int
    story_spines: List[str]
    core_incidents: Set[str]
    periphery_incidents: Set[str]
    rejected_incidents: Set[str]

    # Comparisons
    diffs: List[DecisionDiff] = field(default_factory=list)

    # Metrics
    core_leak_rate: float = 0.0
    expected_core_leak_rate: float = 0.0

    # Raw result for inspection
    build_result: Optional[StoryBuilderResult] = None

    def __str__(self) -> str:
        status = "✓ PASS" if self.matches_expected and self.deterministic else "✗ FAIL"
        lines = [
            f"Replay Result: {self.snapshot_id} {status}",
            f"  Stories: {self.stories_formed} ({', '.join(self.story_spines)})",
            f"  Core: {len(self.core_incidents)}, Periphery: {len(self.periphery_incidents)}, Rejected: {len(self.rejected_incidents)}",
            f"  Core leak rate: {self.core_leak_rate:.2%} (expected {self.expected_core_leak_rate:.2%})",
            f"  Deterministic: {self.deterministic}",
            f"  Matches expected: {self.matches_expected}",
        ]
        if self.diffs:
            lines.append(f"  Diffs ({len(self.diffs)}):")
            for diff in self.diffs[:10]:  # Show first 10
                lines.append(f"    - {diff}")
            if len(self.diffs) > 10:
                lines.append(f"    ... and {len(self.diffs) - 10} more")
        return "\n".join(lines)


class SnapshotReplayer:
    """Replays snapshots and compares results."""

    def __init__(self, story_builder: Optional[StoryBuilder] = None):
        self.story_builder = story_builder or StoryBuilder(
            hub_fraction_threshold=0.20,
            hub_min_incidents=5,
            min_incidents_for_story=2,
            mode_gap_days=30,
        )

    def load_snapshot(self, path: Path) -> Dict[str, Any]:
        """Load snapshot from file."""
        with open(path) as f:
            return json.load(f)

    def snapshot_to_incidents(self, snapshot: Dict[str, Any]) -> Dict[str, Event]:
        """Convert snapshot incidents to Event objects."""
        incidents = {}
        raw_incidents = snapshot.get("incidents", {})

        # Also generate background incidents if specified
        background = snapshot.get("unrelated_background", {})
        if background.get("count"):
            for i in range(background["count"]):
                bg_id = f"bg_{i:03d}"
                raw_incidents[bg_id] = {
                    "anchor_entities": [f"Entity_A_{i}", f"Entity_B_{i}"],
                    "time_start": "2025-11-26T00:00:00Z",
                    "time_end": "2025-11-26T01:00:00Z",
                    "expected_membership": "separate",  # Not part of WFC story
                }

        for inc_id, inc_data in raw_incidents.items():
            time_start = None
            time_end = None
            if inc_data.get("time_start"):
                ts = inc_data["time_start"].replace("Z", "+00:00")
                time_start = datetime.fromisoformat(ts)
            if inc_data.get("time_end"):
                te = inc_data["time_end"].replace("Z", "+00:00")
                time_end = datetime.fromisoformat(te)

            event = Event(
                id=inc_id,
                anchor_entities=set(inc_data.get("anchor_entities", [])),
                entities=set(inc_data.get("anchor_entities", [])),
                time_window=(time_start, time_end),
                surface_ids=set(inc_data.get("surface_ids", [])),
                canonical_title=inc_data.get("_note", ""),
            )
            incidents[inc_id] = event

        return incidents

    def snapshot_to_surfaces(self, snapshot: Dict[str, Any]) -> Dict[str, Surface]:
        """Convert snapshot surfaces to Surface objects."""
        surfaces = {}
        for surf_id, surf_data in snapshot.get("surfaces", {}).items():
            surface = Surface(
                id=surf_id,
                question_key=surf_data.get("question_key", "unknown"),
                claim_ids=set(surf_data.get("claim_ids", [])) if surf_data.get("claim_ids") else set(),
                formation_method="snapshot",
                centroid=None,
            )
            surfaces[surf_id] = surface
        return surfaces

    def get_expected_memberships(self, snapshot: Dict[str, Any]) -> Dict[str, Tuple[str, Optional[str]]]:
        """Get expected memberships from snapshot.

        Returns:
            Dict mapping incident_id to (membership, reason)
        """
        expected = {}
        for inc_id, inc_data in snapshot.get("incidents", {}).items():
            membership = inc_data.get("expected_membership")
            reason = inc_data.get("expected_reason")
            if membership:
                expected[inc_id] = (membership, reason)
        return expected

    def classify_incidents(
        self,
        result: StoryBuilderResult,
        all_incident_ids: Set[str],
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """Classify incidents into core, periphery, rejected.

        Returns:
            (core_ids, periphery_ids, rejected_ids)
        """
        core_ids = set()
        periphery_ids = set()

        for story in result.stories.values():
            core_ids.update(story.core_a_ids)
            core_ids.update(story.core_b_ids)
            periphery_ids.update(story.periphery_incident_ids)

        # Rejected = all incidents not in any story
        assigned = core_ids | periphery_ids
        rejected_ids = all_incident_ids - assigned

        return core_ids, periphery_ids, rejected_ids

    def compare_memberships(
        self,
        expected: Dict[str, Tuple[str, Optional[str]]],
        core_ids: Set[str],
        periphery_ids: Set[str],
        rejected_ids: Set[str],
    ) -> List[DecisionDiff]:
        """Compare actual memberships to expected.

        Returns:
            List of diffs where actual != expected
        """
        diffs = []

        for inc_id, (exp_membership, exp_reason) in expected.items():
            if exp_membership == "separate":
                # Background incidents - skip
                continue

            actual_membership = "unknown"
            if inc_id in core_ids:
                actual_membership = "core"
            elif inc_id in periphery_ids:
                actual_membership = "periphery"
            elif inc_id in rejected_ids:
                actual_membership = "reject"

            if actual_membership != exp_membership:
                diffs.append(DecisionDiff(
                    incident_id=inc_id,
                    expected=exp_membership,
                    actual=actual_membership,
                    expected_reason=exp_reason,
                ))

        return diffs

    def replay(self, path: Path, runs: int = 2) -> ReplayResult:
        """Replay a snapshot and compare results.

        Args:
            path: Path to snapshot file
            runs: Number of runs to check determinism (default 2)

        Returns:
            ReplayResult with comparison data
        """
        snapshot = self.load_snapshot(path)
        meta = snapshot.get("_meta", {})
        snapshot_id = meta.get("description", path.stem)

        # Convert to kernel inputs
        incidents = self.snapshot_to_incidents(snapshot)
        surfaces = self.snapshot_to_surfaces(snapshot)
        expected = self.get_expected_memberships(snapshot)
        expected_outcome = meta.get("expected_outcome", {})

        # Run multiple times to check determinism
        results: List[StoryBuilderResult] = []
        for _ in range(runs):
            result = self.story_builder.build_from_incidents(incidents, surfaces)
            results.append(result)

        # Check determinism (same number of stories, same spines)
        deterministic = True
        first_result = results[0]
        for r in results[1:]:
            if len(r.stories) != len(first_result.stories):
                deterministic = False
            if set(s.spine for s in r.stories.values()) != set(s.spine for s in first_result.stories.values()):
                deterministic = False

        # Classify incidents
        all_incident_ids = set(incidents.keys())
        core_ids, periphery_ids, rejected_ids = self.classify_incidents(first_result, all_incident_ids)

        # Compare to expected
        diffs = self.compare_memberships(expected, core_ids, periphery_ids, rejected_ids)

        # Calculate core leak rate
        expected_core_count = expected_outcome.get("core_incidents", 0)
        actual_core_count = len(core_ids)
        core_leak_rate = 0.0
        if expected_core_count > 0:
            # Leak = (actual - expected) / expected, clamped to 0
            extra = max(0, actual_core_count - expected_core_count)
            core_leak_rate = extra / expected_core_count

        # Check if matches expected outcome
        matches_expected = len(diffs) == 0
        if expected_outcome.get("core_leak_rate") is not None:
            if core_leak_rate > expected_outcome["core_leak_rate"]:
                matches_expected = False

        return ReplayResult(
            snapshot_id=snapshot_id,
            deterministic=deterministic,
            matches_expected=matches_expected,
            stories_formed=len(first_result.stories),
            story_spines=[s.spine for s in first_result.stories.values()],
            core_incidents=core_ids,
            periphery_incidents=periphery_ids,
            rejected_incidents=rejected_ids,
            diffs=diffs,
            core_leak_rate=core_leak_rate,
            expected_core_leak_rate=expected_outcome.get("core_leak_rate", 0.0),
            build_result=first_result,
        )


def replay_snapshot(
    path: str,
    story_builder: Optional[StoryBuilder] = None,
    runs: int = 2,
) -> ReplayResult:
    """
    Convenience function to replay a snapshot.

    Args:
        path: Path to snapshot file
        story_builder: Optional custom StoryBuilder
        runs: Number of runs for determinism check

    Returns:
        ReplayResult with comparison data
    """
    replayer = SnapshotReplayer(story_builder)
    return replayer.replay(Path(path), runs)
