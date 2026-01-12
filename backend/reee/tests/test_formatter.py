"""
Tests for REEE Explain formatter.

These tests verify:
- Decision trace formatting across styles
- Belief trace formatting across styles
- Output format correctness
"""

import pytest
from datetime import datetime

from reee.contracts.traces import DecisionTrace, BeliefUpdateTrace, FeatureVector
from reee.explain import format_trace, format_belief_trace, TraceStyle


class TestDecisionTraceFormatting:
    """Tests for DecisionTrace formatting."""

    @pytest.fixture
    def sample_decision_trace(self) -> DecisionTrace:
        """Create a sample decision trace for testing."""
        return DecisionTrace(
            id="trace_abc123",
            decision_type="surface_membership",
            outcome="joined",
            subject_id="claim_123",
            target_id="surface_456",
            candidate_ids=frozenset({"surface_456", "surface_789"}),
            features=FeatureVector(
                anchor_overlap=0.8,
                companion_jaccard=0.6,
                time_delta_hours=2.5,
                question_key_confidence=0.85,
            ),
            rules_fired=frozenset({"ANCHOR_OVERLAP_PASS", "TIME_WINDOW_PASS"}),
            params_hash="params_xyz",
            kernel_version="1.0.0",
            timestamp=datetime(2024, 1, 15, 12, 0, 0),
        )

    def test_short_format(self, sample_decision_trace):
        """Short format should be one line."""
        output = format_trace(sample_decision_trace, TraceStyle.SHORT)

        assert "surface_membership" in output
        assert "joined" in output
        assert "\n" not in output  # Single line

    def test_ui_format(self, sample_decision_trace):
        """UI format should have markdown structure."""
        output = format_trace(sample_decision_trace, TraceStyle.UI)

        assert "**" in output  # Has bold markers
        assert "Decision:" in output
        assert "joined" in output
        assert "Anchor overlap:" in output
        assert "80%" in output  # 0.8 -> 80%

    def test_debug_format(self, sample_decision_trace):
        """Debug format should be valid JSON."""
        import json

        output = format_trace(sample_decision_trace, TraceStyle.DEBUG)
        parsed = json.loads(output)

        assert parsed["decision_type"] == "surface_membership"
        assert parsed["outcome"] == "joined"
        assert "features" in parsed

    def test_log_format(self, sample_decision_trace):
        """Log format should be key=value pairs."""
        output = format_trace(sample_decision_trace, TraceStyle.LOG)

        assert "decision_type=surface_membership" in output
        assert "outcome=joined" in output
        assert "subject=claim_123" in output

    def test_rules_displayed(self, sample_decision_trace):
        """Rules should be displayed in UI format."""
        output = format_trace(sample_decision_trace, TraceStyle.UI)

        assert "ANCHOR_OVERLAP_PASS" in output
        assert "TIME_WINDOW_PASS" in output

    def test_candidates_displayed(self, sample_decision_trace):
        """Multiple candidates should be noted."""
        output = format_trace(sample_decision_trace, TraceStyle.UI)

        assert "2 candidates" in output


class TestBeliefTraceFormatting:
    """Tests for BeliefUpdateTrace formatting."""

    @pytest.fixture
    def sample_belief_trace(self) -> BeliefUpdateTrace:
        """Create a sample belief trace for testing."""
        return BeliefUpdateTrace(
            id="trace_belief_123",
            surface_id="sf_abc123",
            question_key="fire_death_count",
            claim_id="claim_456",
            prior_entropy=1.5,
            prior_map=10.0,
            prior_support=3,
            observation_value=15.0,
            observation_confidence=0.9,
            observation_authority=0.85,
            noise_model="calibrated",
            posterior_entropy=0.8,
            posterior_map=15.0,
            posterior_support=4,
            surprisal=2.5,
            conflict_detected=False,
            timestamp=datetime(2024, 1, 15, 12, 0, 0),
        )

    @pytest.fixture
    def conflict_belief_trace(self) -> BeliefUpdateTrace:
        """Create a belief trace with conflict."""
        return BeliefUpdateTrace(
            id="trace_conflict_123",
            surface_id="sf_abc123",
            question_key="fire_death_count",
            claim_id="claim_789",
            prior_entropy=0.8,
            prior_map=15.0,
            prior_support=4,
            observation_value=50.0,
            observation_confidence=0.8,
            observation_authority=0.7,
            noise_model="calibrated",
            posterior_entropy=1.2,
            posterior_map=15.0,  # MAP unchanged despite conflict
            posterior_support=5,
            surprisal=3.5,
            conflict_detected=True,
            timestamp=datetime(2024, 1, 15, 12, 0, 0),
        )

    def test_short_format(self, sample_belief_trace):
        """Short format should be one line."""
        output = format_belief_trace(sample_belief_trace, TraceStyle.SHORT)

        assert "fire_death_count" in output
        # Note: MAP values are floats now (10.0, 15.0)
        assert "10" in output and "15" in output
        assert "\n" not in output

    def test_ui_format(self, sample_belief_trace):
        """UI format should have markdown structure."""
        output = format_belief_trace(sample_belief_trace, TraceStyle.UI)

        assert "**Belief Update:" in output
        assert "Observed value:" in output
        assert "15" in output
        assert "Prior MAP:" in output
        assert "Entropy:" in output

    def test_debug_format(self, sample_belief_trace):
        """Debug format should be valid JSON."""
        import json

        output = format_belief_trace(sample_belief_trace, TraceStyle.DEBUG)
        parsed = json.loads(output)

        assert parsed["surface_id"] == "sf_abc123"
        assert parsed["prior_map"] == 10.0
        assert parsed["posterior_map"] == 15.0

    def test_log_format(self, sample_belief_trace):
        """Log format should be key=value pairs."""
        output = format_belief_trace(sample_belief_trace, TraceStyle.LOG)

        assert "surface=sf_abc123" in output
        assert "question=fire_death_count" in output
        assert "conflict=False" in output

    def test_conflict_warning(self, conflict_belief_trace):
        """Conflict should be prominently displayed."""
        output = format_belief_trace(conflict_belief_trace, TraceStyle.UI)

        assert "Conflict" in output or "conflict" in output

    def test_conflict_in_short(self, conflict_belief_trace):
        """Conflict should appear in short format."""
        output = format_belief_trace(conflict_belief_trace, TraceStyle.SHORT)

        assert "CONFLICT" in output

    def test_surprisal_highlighted(self, sample_belief_trace):
        """High surprisal should be noted."""
        output = format_belief_trace(sample_belief_trace, TraceStyle.UI)

        # surprisal > 2.0 should be highlighted
        assert "Surprising" in output or "surprisal" in output.lower()

    def test_entropy_direction(self, sample_belief_trace):
        """Entropy change direction should be shown."""
        output = format_belief_trace(sample_belief_trace, TraceStyle.UI)

        # Entropy went 1.5 -> 0.8 (decreased)
        assert "decreased" in output


class TestFormatTraceDispatch:
    """Tests for format_trace dispatch function."""

    def test_decision_trace_dispatch(self):
        """format_trace should handle DecisionTrace."""
        trace = DecisionTrace(
            id="trace_test_1",
            decision_type="test",
            outcome="test_outcome",
            subject_id="s1",
            target_id=None,
            candidate_ids=frozenset(),
            features=FeatureVector(),
            rules_fired=frozenset(),
            params_hash="p1",
            kernel_version="1.0",
            timestamp=datetime.now(),
        )

        output = format_trace(trace, TraceStyle.SHORT)
        assert "test" in output
        assert "test_outcome" in output

    def test_belief_trace_dispatch(self):
        """format_trace should handle BeliefUpdateTrace."""
        trace = BeliefUpdateTrace(
            id="trace_test_2",
            surface_id="sf_1",
            question_key="test_key",
            claim_id="c1",
            prior_entropy=1.0,
            prior_map=None,
            prior_support=0,
            observation_value="val",
            observation_confidence=0.9,
            observation_authority=0.9,
            noise_model="uniform",
            posterior_entropy=0.5,
            posterior_map=1.0,
            posterior_support=1,
            surprisal=0.5,
            conflict_detected=False,
            timestamp=datetime.now(),
        )

        output = format_trace(trace, TraceStyle.SHORT)
        assert "test_key" in output

    def test_invalid_trace_type(self):
        """format_trace should raise on invalid type."""
        with pytest.raises(TypeError):
            format_trace("not a trace", TraceStyle.SHORT)


class TestEdgeCases:
    """Tests for edge cases in formatting."""

    def test_empty_rules(self):
        """Trace with no rules should format cleanly."""
        trace = DecisionTrace(
            id="trace_empty_rules",
            decision_type="test",
            outcome="test",
            subject_id="s1",
            target_id=None,
            candidate_ids=frozenset(),
            features=FeatureVector(),
            rules_fired=frozenset(),
            params_hash="p1",
            kernel_version="1.0",
            timestamp=datetime.now(),
        )

        output = format_trace(trace, TraceStyle.UI)
        assert "Rules" not in output  # No rules section

    def test_no_target(self):
        """Trace with no target should format cleanly."""
        trace = DecisionTrace(
            id="trace_no_target",
            decision_type="test",
            outcome="test",
            subject_id="s1",
            target_id=None,
            candidate_ids=frozenset(),
            features=FeatureVector(),
            rules_fired=frozenset(),
            params_hash="p1",
            kernel_version="1.0",
            timestamp=datetime.now(),
        )

        output = format_trace(trace, TraceStyle.UI)
        assert "Target:" not in output

    def test_low_confidence_warning(self):
        """Low question_key confidence should be noted."""
        trace = DecisionTrace(
            id="trace_low_conf",
            decision_type="test",
            outcome="test",
            subject_id="s1",
            target_id=None,
            candidate_ids=frozenset(),
            features=FeatureVector(question_key_confidence=0.3),
            rules_fired=frozenset(),
            params_hash="p1",
            kernel_version="1.0",
            timestamp=datetime.now(),
        )

        output = format_trace(trace, TraceStyle.UI)
        assert "(low)" in output or "30%" in output
