#!/usr/bin/env python3
"""
Experiment: Event State Machine

Test the event lifecycle as defined in docs/66.product.liveevent.md:
- 🔴 LIVE (active) - High metabolism, immediate processing
- 🟡 WARM (recent activity) - Medium metabolism, batched processing
- 🟢 STABLE (converged) - Low metabolism, hourly cycles
- ⚪ DORMANT (hibernating) - No metabolism, wake on high-stake

Key questions:
1. When should events transition between states?
2. How does contribution responsiveness change with state?
3. When should dormant events wake?
"""

import asyncio
import json
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum

sys.path.insert(0, '/app')


class EventState(Enum):
    LIVE = "🔴 LIVE"
    WARM = "🟡 WARM"
    STABLE = "🟢 STABLE"
    DORMANT = "⚪ DORMANT"


@dataclass
class StateConfig:
    """Configuration for each state"""
    metabolism_interval: timedelta  # How often to run metabolism
    response_mode: str  # immediate, batched, queued, wake_only
    min_stake_to_wake: int  # Minimum stake to wake from dormant


STATE_CONFIGS = {
    EventState.LIVE: StateConfig(
        metabolism_interval=timedelta(seconds=30),
        response_mode="immediate",
        min_stake_to_wake=1
    ),
    EventState.WARM: StateConfig(
        metabolism_interval=timedelta(minutes=5),
        response_mode="batched",
        min_stake_to_wake=1
    ),
    EventState.STABLE: StateConfig(
        metabolism_interval=timedelta(hours=1),
        response_mode="queued",
        min_stake_to_wake=10
    ),
    EventState.DORMANT: StateConfig(
        metabolism_interval=timedelta(hours=24),
        response_mode="wake_only",
        min_stake_to_wake=100
    )
}


@dataclass
class EventActivity:
    """Tracks event activity for state decisions"""
    claims_added_last_hour: int = 0
    claims_added_last_day: int = 0
    last_claim_at: Optional[datetime] = None
    last_metabolism_at: Optional[datetime] = None
    coherence: float = 1.0
    coherence_trend: float = 0.0  # Positive = improving, negative = degrading
    contradiction_count: int = 0
    unresolved_contradictions: int = 0


@dataclass
class SimulatedEvent:
    """Event with state machine"""
    id: str
    state: EventState = EventState.LIVE
    activity: EventActivity = field(default_factory=EventActivity)
    state_history: List[Dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def transition_to(self, new_state: EventState, reason: str):
        """Record state transition"""
        old_state = self.state
        self.state = new_state
        self.state_history.append({
            'from': old_state.name,
            'to': new_state.name,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })
        print(f"   📍 {old_state.value} → {new_state.value}: {reason}")


class EventStateMachine:
    """Manages event state transitions"""

    # Transition thresholds
    LIVE_TO_WARM_HOURS = 1  # No claims in 1 hour → WARM
    WARM_TO_STABLE_HOURS = 6  # No claims in 6 hours → STABLE
    STABLE_TO_DORMANT_DAYS = 3  # No claims in 3 days → DORMANT

    CLAIMS_TO_STAY_LIVE = 3  # Need 3+ claims/hour to stay LIVE
    CONTRADICTION_THRESHOLD = 0.2  # >20% contradictions keeps ACTIVE/LIVE

    def __init__(self):
        self.events: Dict[str, SimulatedEvent] = {}

    def evaluate_state(self, event: SimulatedEvent, now: datetime) -> Optional[EventState]:
        """Evaluate if event should transition"""
        activity = event.activity
        hours_since_claim = (now - activity.last_claim_at).total_seconds() / 3600 if activity.last_claim_at else 999

        current = event.state

        # Check for contradictions keeping event active
        has_tension = activity.unresolved_contradictions > 0 or \
                      (activity.contradiction_count / max(1, activity.claims_added_last_day) > self.CONTRADICTION_THRESHOLD)

        if current == EventState.LIVE:
            if hours_since_claim > self.LIVE_TO_WARM_HOURS and not has_tension:
                return EventState.WARM
            if activity.claims_added_last_hour < self.CLAIMS_TO_STAY_LIVE and not has_tension:
                if hours_since_claim > 0.5:  # 30 min without activity
                    return EventState.WARM

        elif current == EventState.WARM:
            # Can go back to LIVE
            if activity.claims_added_last_hour >= self.CLAIMS_TO_STAY_LIVE or has_tension:
                return EventState.LIVE
            # Or down to STABLE
            if hours_since_claim > self.WARM_TO_STABLE_HOURS and not has_tension:
                return EventState.STABLE

        elif current == EventState.STABLE:
            # Can wake to WARM/LIVE
            if activity.claims_added_last_hour >= 1:
                return EventState.WARM
            if has_tension:
                return EventState.LIVE
            # Or hibernate
            days_since_claim = hours_since_claim / 24
            if days_since_claim > self.STABLE_TO_DORMANT_DAYS:
                return EventState.DORMANT

        elif current == EventState.DORMANT:
            # Only wake on significant activity
            if activity.claims_added_last_hour >= 1 or has_tension:
                return EventState.WARM

        return None  # No transition

    def should_wake(self, event: SimulatedEvent, stake: int) -> bool:
        """Check if contribution stake is enough to wake dormant event"""
        config = STATE_CONFIGS[event.state]
        return stake >= config.min_stake_to_wake

    def process_contribution(self, event: SimulatedEvent, stake: int, is_contradiction: bool = False):
        """Update activity metrics on contribution"""
        now = datetime.now()

        # Wake check for dormant
        if event.state == EventState.DORMANT:
            if self.should_wake(event, stake):
                event.transition_to(EventState.WARM, f"High-stake contribution ({stake}c) woke event")
            else:
                print(f"   💤 Event dormant, stake {stake}c insufficient (need {STATE_CONFIGS[EventState.DORMANT].min_stake_to_wake}c)")
                return

        # Update activity
        event.activity.claims_added_last_hour += 1
        event.activity.claims_added_last_day += 1
        event.activity.last_claim_at = now

        if is_contradiction:
            event.activity.contradiction_count += 1
            event.activity.unresolved_contradictions += 1

        # Evaluate state transition
        new_state = self.evaluate_state(event, now)
        if new_state:
            reason = self._get_transition_reason(event.state, new_state, event.activity)
            event.transition_to(new_state, reason)

    def _get_transition_reason(self, old: EventState, new: EventState, activity: EventActivity) -> str:
        """Generate human-readable transition reason"""
        if new == EventState.LIVE:
            if activity.unresolved_contradictions > 0:
                return f"Unresolved contradiction detected"
            return f"High activity ({activity.claims_added_last_hour} claims/hour)"

        elif new == EventState.WARM:
            if old == EventState.LIVE:
                return "Activity slowed"
            elif old == EventState.DORMANT:
                return "New activity detected"
            return "Recent activity"

        elif new == EventState.STABLE:
            return f"No activity for {self.WARM_TO_STABLE_HOURS}+ hours, coherent"

        elif new == EventState.DORMANT:
            return f"No activity for {self.STABLE_TO_DORMANT_DAYS}+ days"

        return "State transition"

    def decay_activity(self, event: SimulatedEvent, hours_passed: float):
        """Decay activity counters as time passes"""
        # Simple exponential decay
        decay_factor = 0.5 ** (hours_passed / 1)  # Half-life of 1 hour

        event.activity.claims_added_last_hour = int(event.activity.claims_added_last_hour * decay_factor)

        if hours_passed >= 24:
            day_decay = 0.5 ** (hours_passed / 24)
            event.activity.claims_added_last_day = int(event.activity.claims_added_last_day * day_decay)


async def simulate_event_lifecycle():
    """Simulate an event's lifecycle through various states"""

    print("=" * 70)
    print("EVENT STATE MACHINE SIMULATION")
    print("=" * 70)

    sm = EventStateMachine()

    # Create event
    event = SimulatedEvent(id="test_event")
    event.activity.last_claim_at = datetime.now()
    sm.events[event.id] = event

    print(f"\n🎯 Event created: {event.id}")
    print(f"   Initial state: {event.state.value}")

    # Simulation scenarios
    scenarios = [
        # (description, action, params)
        ("Breaking news - rapid claims", "burst", {"count": 5, "interval_min": 2}),
        ("Activity slows down", "wait", {"hours": 1.5}),
        ("Single new source", "contribution", {"stake": 10}),
        ("Contradiction detected", "contribution", {"stake": 50, "contradiction": True}),
        ("Contradiction resolved", "resolve_contradiction", {}),
        ("Extended quiet period", "wait", {"hours": 8}),
        ("Minor contribution", "contribution", {"stake": 5}),
        ("Very long quiet", "wait", {"hours": 80}),  # 3+ days
        ("Low-stake attempt to wake", "contribution", {"stake": 10}),
        ("High-stake contribution", "contribution", {"stake": 100}),
    ]

    print("\n📋 Running simulation scenarios:\n")

    for desc, action, params in scenarios:
        print(f"\n▶️  Scenario: {desc}")

        if action == "burst":
            for i in range(params['count']):
                sm.process_contribution(event, stake=10)
                print(f"      +claim {i+1}/{params['count']}")

        elif action == "wait":
            hours = params['hours']
            print(f"   ⏳ {hours} hours pass...")
            sm.decay_activity(event, hours)

            # Simulate time passing for last_claim_at
            if event.activity.last_claim_at:
                event.activity.last_claim_at -= timedelta(hours=hours)

            # Evaluate state after time passes
            now = datetime.now()
            new_state = sm.evaluate_state(event, now)
            if new_state:
                reason = sm._get_transition_reason(event.state, new_state, event.activity)
                event.transition_to(new_state, reason)

        elif action == "contribution":
            stake = params.get('stake', 10)
            is_contra = params.get('contradiction', False)
            sm.process_contribution(event, stake=stake, is_contradiction=is_contra)
            print(f"   💰 Contribution ({stake}c, {'contradiction' if is_contra else 'normal'})")

        elif action == "resolve_contradiction":
            event.activity.unresolved_contradictions = 0
            print(f"   ✅ Contradiction resolved")

        print(f"   Current: {event.state.value}")
        print(f"   Activity: {event.activity.claims_added_last_hour}/hr, {event.activity.claims_added_last_day}/day")
        print(f"   Unresolved: {event.activity.unresolved_contradictions} contradictions")

    # Summary
    print("\n" + "=" * 70)
    print("STATE TRANSITION HISTORY")
    print("=" * 70)

    for i, transition in enumerate(event.state_history):
        print(f"   {i+1}. {transition['from']} → {transition['to']}: {transition['reason']}")

    # State duration analysis
    print(f"\n📊 State Analysis:")
    state_counts = {}
    for t in event.state_history:
        to_state = t['to']
        state_counts[to_state] = state_counts.get(to_state, 0) + 1

    for state, count in state_counts.items():
        print(f"   Entered {state}: {count} times")

    # Config summary
    print(f"\n⚙️  State Configurations:")
    for state, config in STATE_CONFIGS.items():
        print(f"   {state.value}:")
        print(f"      Metabolism: every {config.metabolism_interval}")
        print(f"      Response: {config.response_mode}")
        print(f"      Min wake stake: {config.min_stake_to_wake}c")

    return {
        'final_state': event.state.name,
        'transitions': len(event.state_history),
        'state_counts': state_counts,
        'history': event.state_history
    }


async def main():
    results = await simulate_event_lifecycle()

    print(f"\n💾 Saving results...")
    with open('/app/test_eu/state_machine_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n✅ State machine simulation complete!")


if __name__ == "__main__":
    asyncio.run(main())
