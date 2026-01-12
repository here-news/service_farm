"""
Deterministic Trace Formatter - Always available, no LLM required.

Formats DecisionTrace and BeliefUpdateTrace into human-readable output.
Supports multiple styles for different use cases.
"""

from enum import Enum
from typing import Union
import json

from ..contracts.traces import DecisionTrace, BeliefUpdateTrace


class TraceStyle(Enum):
    """Output style for trace formatting."""

    SHORT = "short"  # One-line summary
    UI = "ui"  # User-facing markdown
    DEBUG = "debug"  # Full JSON for debugging
    LOG = "log"  # Structured log format


def format_trace(
    trace: Union[DecisionTrace, BeliefUpdateTrace],
    style: TraceStyle = TraceStyle.UI,
) -> str:
    """Format a trace for display.

    Args:
        trace: DecisionTrace or BeliefUpdateTrace
        style: Output style

    Returns:
        Formatted string
    """
    if isinstance(trace, DecisionTrace):
        return _format_decision_trace(trace, style)
    elif isinstance(trace, BeliefUpdateTrace):
        return _format_belief_trace(trace, style)
    else:
        raise TypeError(f"Unknown trace type: {type(trace)}")


def format_belief_trace(
    trace: BeliefUpdateTrace,
    style: TraceStyle = TraceStyle.UI,
) -> str:
    """Format a belief update trace.

    Convenience function for direct BeliefUpdateTrace formatting.
    """
    return _format_belief_trace(trace, style)


def _format_decision_trace(trace: DecisionTrace, style: TraceStyle) -> str:
    """Format a decision trace."""
    if style == TraceStyle.SHORT:
        return _format_decision_short(trace)
    elif style == TraceStyle.UI:
        return _format_decision_ui(trace)
    elif style == TraceStyle.DEBUG:
        return json.dumps(trace.to_debug_dict(), indent=2)
    elif style == TraceStyle.LOG:
        return _format_decision_log(trace)
    else:
        raise ValueError(f"Unknown style: {style}")


def _format_belief_trace(trace: BeliefUpdateTrace, style: TraceStyle) -> str:
    """Format a belief update trace."""
    if style == TraceStyle.SHORT:
        return _format_belief_short(trace)
    elif style == TraceStyle.UI:
        return _format_belief_ui(trace)
    elif style == TraceStyle.DEBUG:
        return json.dumps(trace.to_debug_dict(), indent=2)
    elif style == TraceStyle.LOG:
        return _format_belief_log(trace)
    else:
        raise ValueError(f"Unknown style: {style}")


# =============================================================================
# Decision Trace Formatters
# =============================================================================


def _format_decision_short(trace: DecisionTrace) -> str:
    """One-line summary of decision."""
    rules = ", ".join(sorted(trace.rules_fired)[:3])
    if len(trace.rules_fired) > 3:
        rules += f" (+{len(trace.rules_fired) - 3})"
    return f"{trace.decision_type}: {trace.outcome} [{rules}]"


def _format_decision_ui(trace: DecisionTrace) -> str:
    """User-facing markdown format."""
    lines = []

    # Header
    outcome_emoji = _get_outcome_emoji(trace.outcome)
    lines.append(f"**{outcome_emoji} Decision: {trace.outcome}**")
    lines.append("")

    # Type info
    lines.append(f"- Type: `{trace.decision_type}`")
    lines.append(f"- Subject: `{trace.subject_id}`")
    if trace.target_id:
        lines.append(f"- Target: `{trace.target_id}`")

    # Features (if relevant)
    if trace.features.anchor_overlap > 0:
        lines.append(f"- Anchor overlap: {trace.features.anchor_overlap:.0%}")
    if trace.features.companion_jaccard > 0:
        lines.append(f"- Companion similarity: {trace.features.companion_jaccard:.0%}")
    if trace.features.time_delta_hours is not None:
        lines.append(f"- Time gap: {trace.features.time_delta_hours:.1f} hours")
    if trace.features.question_key_confidence < 0.7:
        lines.append(
            f"- Question key confidence: {trace.features.question_key_confidence:.0%} (low)"
        )

    # Rules
    if trace.rules_fired:
        lines.append("")
        lines.append("**Rules applied:**")
        for rule in sorted(trace.rules_fired):
            rule_desc = _get_rule_description(rule)
            lines.append(f"- {rule}: {rule_desc}")

    # Candidates considered
    if trace.candidate_ids and len(trace.candidate_ids) > 1:
        lines.append("")
        lines.append(f"*Considered {len(trace.candidate_ids)} candidates*")

    return "\n".join(lines)


def _format_decision_log(trace: DecisionTrace) -> str:
    """Structured log format."""
    parts = [
        f"decision_type={trace.decision_type}",
        f"outcome={trace.outcome}",
        f"subject={trace.subject_id}",
        f"target={trace.target_id or 'none'}",
        f"rules={','.join(sorted(trace.rules_fired))}",
        f"anchor_overlap={trace.features.anchor_overlap:.2f}",
    ]
    return " ".join(parts)


# =============================================================================
# Belief Trace Formatters
# =============================================================================


def _format_belief_short(trace: BeliefUpdateTrace) -> str:
    """One-line summary of belief update."""
    conflict_marker = " CONFLICT" if trace.conflict_detected else ""
    return (
        f"belief: {trace.question_key} "
        f"{trace.prior_map}->{trace.posterior_map} "
        f"(entropy {trace.prior_entropy:.2f}->{trace.posterior_entropy:.2f})"
        f"{conflict_marker}"
    )


def _format_belief_ui(trace: BeliefUpdateTrace) -> str:
    """User-facing markdown format."""
    lines = []

    # Header with conflict warning
    if trace.conflict_detected:
        lines.append(f"**Warning: Conflicting observation detected**")
        lines.append("")

    lines.append(f"**Belief Update: {trace.question_key}**")
    lines.append("")

    # Observation
    lines.append(f"- Observed value: `{trace.observation_value}`")
    lines.append(f"- Confidence: {trace.observation_confidence:.0%}")
    lines.append(f"- Source authority: {trace.observation_authority:.0%}")

    # State change
    lines.append("")
    lines.append("**State change:**")
    lines.append(f"- Prior MAP: {trace.prior_map} (n={trace.prior_support})")
    lines.append(f"- Posterior MAP: {trace.posterior_map} (n={trace.posterior_support})")

    # Entropy
    entropy_delta = trace.posterior_entropy - trace.prior_entropy
    entropy_direction = "decreased" if entropy_delta < 0 else "increased"
    lines.append(
        f"- Entropy: {trace.prior_entropy:.2f} -> {trace.posterior_entropy:.2f} "
        f"({entropy_direction})"
    )

    # Surprisal
    if trace.surprisal > 2.0:
        lines.append(f"- **Surprising** (surprisal = {trace.surprisal:.1f})")

    return "\n".join(lines)


def _format_belief_log(trace: BeliefUpdateTrace) -> str:
    """Structured log format."""
    parts = [
        f"surface={trace.surface_id}",
        f"question={trace.question_key}",
        f"value={trace.observation_value}",
        f"prior_map={trace.prior_map}",
        f"posterior_map={trace.posterior_map}",
        f"entropy={trace.prior_entropy:.2f}->{trace.posterior_entropy:.2f}",
        f"conflict={trace.conflict_detected}",
    ]
    return " ".join(parts)


# =============================================================================
# Helpers
# =============================================================================


def _get_outcome_emoji(outcome: str) -> str:
    """Get emoji for outcome."""
    emoji_map = {
        "created_new": "+",
        "joined": "->",
        "rejected": "X",
        "key_explicit": "=",
        "key_pattern": "~",
        "key_entity": "#",
        "key_page_scope": "@",
        "key_singleton": "1",
    }
    return emoji_map.get(outcome, "?")


def _get_rule_description(rule: str) -> str:
    """Get human-readable description of rule."""
    descriptions = {
        # Fallback rules
        "FALLBACK_EXPLICIT": "Used LLM-extracted question key",
        "FALLBACK_PATTERN": "Matched typed question pattern",
        "FALLBACK_ENTITY": "Derived key from anchor entities",
        "FALLBACK_PAGE_SCOPE": "Fell back to page/source scope",
        "FALLBACK_SINGLETON": "Created singleton surface (no collapse)",
        # Membership rules
        "ANCHOR_OVERLAP_PASS": "Sufficient anchor overlap",
        "ANCHOR_OVERLAP_FAIL": "Insufficient anchor overlap",
        "COMPANION_COMPATIBLE": "Companion entities are compatible",
        "COMPANION_DISJOINT": "Companion entities are disjoint (bridge blocked)",
        "TIME_WINDOW_PASS": "Within time window",
        "TIME_WINDOW_FAIL": "Outside time window",
        # Hub rules
        "HUB_SUPPRESSED": "Hub entity suppressed from scope",
        "ALL_HUBS_FALLBACK": "All anchors are hubs (using anyway)",
    }
    return descriptions.get(rule, rule)
