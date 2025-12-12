"""
Metabolism - Data structures for living event metabolism

The metabolism system is the event organism's reactive nervous system.
It responds to stimuli (claims, time, commands) with principled actions.

Key concepts:
- MetabolismAction: Decision about what action to take
- MetabolismResult: Outcome of executing an action
- Thought: An epistemic observation the organism wants to emit

Based on Jaynes' probability theory: the organism maintains beliefs,
updates them with evidence, and surfaces uncertainty honestly.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any


class ActionType(str, Enum):
    """Types of metabolism actions the organism can take"""
    NO_OP = "no_op"                          # Do nothing
    FULL_TOPOLOGY = "full_topology"          # Run complete Bayesian topology analysis
    INCREMENTAL_TOPOLOGY = "incremental_topology"  # Update topology with new claims only
    REGENERATE_NARRATIVE = "regenerate_narrative"  # Regenerate narrative from topology
    EMIT_THOUGHT = "emit_thought"            # Surface an epistemic observation
    HIBERNATE = "hibernate"                  # Enter dormant state
    SPAWN_SUB_EVENT = "spawn_sub_event"      # Create child event for divergent facet


class TriggerType(str, Enum):
    """Types of triggers that can initiate metabolism"""
    CLAIMS_ADDED = "claims_added"            # New claims accepted into event
    TIME_TICK = "time_tick"                  # Periodic metabolism check
    COMMAND = "command"                      # External command received
    CONTRADICTION_DETECTED = "contradiction_detected"  # High-confidence contradiction found
    HYDRATION_COMPLETE = "hydration_complete"  # Event just loaded from storage
    COHERENCE_SHIFT = "coherence_shift"      # Significant coherence change


class ThoughtType(str, Enum):
    """Types of thoughts the organism can emit"""
    QUESTION = "question"                    # "Why did X happen after Y?"
    ANOMALY = "anomaly"                      # "Unusual: death toll revised downward"
    STATE_CHANGE = "state_change"            # "Entering stable state - no updates in 48h"
    CONTRADICTION = "contradiction"          # "Sources disagree on X"
    CONTRADICTION_DETECTED = "contradiction_detected"  # High temperature - many contradictions
    COHERENCE_DROP = "coherence_drop"        # Significant coherence decrease
    EMERGENCE = "emergence"                  # "New angle detected: should this branch?"
    PROGRESS = "progress"                    # "Death toll updated: 36 → 156"
    TOPOLOGY_INSIGHT = "topology_insight"    # LLM-generated summary of epistemic state


@dataclass
class MetabolismAction:
    """
    A decision about what metabolism action to take.

    Every action has a type and a reason (for auditability).
    The reason explains WHY this action was chosen.
    """
    type: ActionType
    trigger: TriggerType  # What triggered this action
    reason: str
    priority: int = 0  # Higher = more urgent
    params: Dict[str, Any] = field(default_factory=dict)  # Parameters for action
    context: Dict[str, Any] = field(default_factory=dict)  # Legacy - use params

    def __post_init__(self):
        # Set default priorities based on action type
        if self.priority == 0:
            priority_map = {
                ActionType.HIBERNATE: 100,
                ActionType.FULL_TOPOLOGY: 90,
                ActionType.INCREMENTAL_TOPOLOGY: 80,
                ActionType.REGENERATE_NARRATIVE: 70,
                ActionType.EMIT_THOUGHT: 60,
                ActionType.SPAWN_SUB_EVENT: 50,
                ActionType.NO_OP: 0,
            }
            self.priority = priority_map.get(self.type, 0)


@dataclass
class MetabolismResult:
    """
    Outcome of executing a metabolism action.

    Includes what was done, whether it succeeded, and any outputs.
    """
    action: ActionType
    success: bool = True
    reason: str = ""

    # Outputs from the action
    claims_processed: int = 0
    topology_pattern: Optional[str] = None
    contradictions_found: int = 0
    thought_emitted: Optional['Thought'] = None
    narrative_updated: bool = False

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None

    def __post_init__(self):
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            self.duration_ms = int(delta.total_seconds() * 1000)


@dataclass
class Thought:
    """
    An epistemic observation the organism wants to emit.

    Thoughts are the organism's way of surfacing:
    - Questions it has about the evidence
    - Anomalies it detected
    - State changes worth noting
    - Contradictions that need attention

    Thoughts are stored in Neo4j on the Event node and can be
    consumed by downstream applications (UI, alerts, community).
    """
    id: str  # th_xxxxxxxx format
    event_id: str
    type: ThoughtType
    content: str  # Human-readable thought text

    # Context for the thought
    related_claims: List[str] = field(default_factory=list)  # claim IDs
    related_entities: List[str] = field(default_factory=list)  # entity IDs

    # Epistemic state when thought was generated
    temperature: float = 0.0  # Event temperature at time of thought
    coherence: float = 0.0    # Event coherence at time of thought

    # Lifecycle
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False  # Has a human/system seen this?
    acknowledged_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dict for storage/API"""
        return {
            'id': self.id,
            'event_id': self.event_id,
            'type': self.type.value,
            'content': self.content,
            'related_claims': self.related_claims,
            'related_entities': self.related_entities,
            'temperature': self.temperature,
            'coherence': self.coherence,
            'created_at': self.created_at.isoformat(),
            'acknowledged': self.acknowledged,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Thought':
        """Create from dict"""
        return cls(
            id=data['id'],
            event_id=data['event_id'],
            type=ThoughtType(data['type']),
            content=data['content'],
            related_claims=data.get('related_claims', []),
            related_entities=data.get('related_entities', []),
            temperature=data.get('temperature', 0.0),
            coherence=data.get('coherence', 0.0),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.utcnow(),
            acknowledged=data.get('acknowledged', False),
            acknowledged_at=datetime.fromisoformat(data['acknowledged_at']) if data.get('acknowledged_at') else None,
        )


@dataclass
class MetabolismState:
    """
    Complete metabolism state for a living event.

    Tracks everything needed to make metabolism decisions:
    - Current temperature and coherence
    - Timestamps of last actions
    - Recent thoughts emitted
    """
    # Current epistemic state
    temperature: float = 0.0  # 0=stable, 1=chaotic
    coherence: float = 0.5    # 0=fragmented, 1=unified

    # Timestamps for decision making
    last_claim_added: Optional[datetime] = None
    last_topology_update: Optional[datetime] = None
    last_narrative_update: Optional[datetime] = None
    last_thought_emitted: Optional[datetime] = None

    # Counts for decision making
    claims_since_topology: int = 0
    claims_since_narrative: int = 0
    contradictions_active: int = 0

    # Previous coherence (for delta detection)
    previous_coherence: float = 0.5

    # Hibernation flag
    is_hibernating: bool = False

    def coherence_delta(self) -> float:
        """Calculate change in coherence"""
        return self.coherence - self.previous_coherence

    def update_coherence(self, new_coherence: float):
        """Update coherence, tracking previous value"""
        self.previous_coherence = self.coherence
        self.coherence = new_coherence


# ============================================================================
# METABOLISM RULES - Configurable thresholds
# ============================================================================

@dataclass
class MetabolismRules:
    """
    Configurable rules for metabolism decisions.

    These thresholds determine when actions are triggered.
    Can be tuned based on operational experience.
    """
    # Topology triggers
    min_claims_for_topology: int = 3
    max_claims_for_incremental: int = 20  # Beyond this, do full topology
    topology_stale_hours: float = 6.0

    # Narrative triggers
    coherence_delta_threshold: float = 0.1  # Trigger narrative on this delta
    coherence_boost_threshold: float = 0.1  # Trigger narrative on coherence boost
    coherence_drop_threshold: float = 0.15  # Emit thought on coherence drop
    narrative_stale_hours: float = 1.0

    # Thought triggers
    temperature_thought_threshold: float = 0.6  # Emit thought when temp exceeds
    high_temperature_threshold: float = 0.5  # Consider temperature "high"
    thought_cooldown_hours: float = 2.0  # Don't emit thoughts too frequently

    # Hibernation triggers
    idle_hours_to_hibernate: float = 24.0

    # Activity window
    active_window_hours: float = 6.0  # Consider "active" if claims within this window


# Default rules instance
DEFAULT_RULES = MetabolismRules()
