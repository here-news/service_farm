"""
LiveEvent - Event as a living organism with metabolism

A LiveEvent is an in-memory representation of an event that:
- Bootstraps from claims
- Hydrates state from storage
- Executes metabolism via trigger-based state machine
- Uses Bayesian topology for plausibility-weighted narratives
- Emits thoughts for epistemic observations
- Hibernates when dormant

The pool maintains these organisms, preventing duplicates and managing lifecycle.

Metabolism Triggers:
- on_claims(): Called when new claims are added
- on_time_tick(): Called periodically by pool metabolism cycle
- on_hydration_complete(): Called after loading from storage
- on_command(): Called when external command is received

All triggers route through _decide_metabolism_action() which determines
what action to take based on current state and rules.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Set, Optional, Dict, Tuple
from dataclasses import dataclass

from models.domain.event import Event, StructuredNarrative
from models.domain.claim import Claim
from models.domain.metabolism import (
    ActionType, TriggerType, ThoughtType,
    MetabolismAction, MetabolismState, MetabolismRules, Thought
)
from utils.datetime_utils import neo4j_datetime_to_python
from utils.id_generator import generate_id

logger = logging.getLogger(__name__)


@dataclass
class MetabolismResult:
    """Result of executing a metabolism action in LiveEvent"""
    action_type: ActionType
    success: bool = True
    error: Optional[str] = None
    data: Optional[Dict] = None
    thoughts: Optional[List] = None
    # For backwards compatibility with examine()
    claims_added: Optional[List] = None
    claims_rejected: Optional[List] = None
    sub_events_created: Optional[List] = None


@dataclass
class LiveEventState:
    """State of a living event"""
    event: Event
    claims: List[Claim]
    entity_ids: Set[str]
    last_claim_added: Optional[datetime]
    last_narrative_update: datetime
    created_at: datetime


class LiveEvent:
    """
    A living event organism that maintains itself through metabolism.

    Lifecycle:
    1. Bootstrap - created from initial claims
    2. Hydrate - loads full state from storage, triggers on_hydration_complete()
    3. Metabolism - responds to triggers (claims, time, commands)
    4. Hibernate - archives when dormant

    Trigger-Based State Machine:
    - on_claims(new_claims) - New claims received
    - on_time_tick() - Periodic metabolism cycle
    - on_hydration_complete() - Just loaded from storage
    - on_command(cmd, params) - External command received

    Each trigger calls _decide_metabolism_action() which returns an action
    based on current state. Actions are then executed by action handlers.

    Uses Bayesian topology (Jaynes-informed) for:
    - Plausibility scoring of claims
    - Weighted narrative generation
    - Contradiction detection and resolution
    """

    def __init__(self, event: Event, event_service, topology_service=None):
        self.event = event
        self.service = event_service
        self.topology_service = topology_service  # Optional: ClaimTopologyService

        # Internal state
        self.claims: List[Claim] = []
        self.entity_ids: Set[str] = set()

        # Source priors - stored on publisher at extraction time
        # publisher_priors: claim_id -> {'base_prior': float, 'source_type': str, 'publisher_name': str}
        self.publisher_priors: Dict[str, dict] = {}
        # page_urls: fallback for claims without stored priors
        self.page_urls: Dict[str, str] = {}

        self.last_claim_added: Optional[datetime] = None
        self.last_narrative_update: datetime = datetime.utcnow()
        self.last_topology_update: Optional[datetime] = None
        self.created_at: datetime = datetime.utcnow()

        # Cached topology result
        self._topology_result = None

        # Metabolism state
        self._metabolism_state = MetabolismState()
        self._metabolism_rules = MetabolismRules()
        self._thoughts: List[Thought] = []

        logger.debug(f"🌱 LiveEvent created: {event.canonical_name} ({event.id})")

    # ==================== Trigger Interface ====================
    # These are the entry points for the metabolism state machine.
    # External code should call these triggers, not internal methods.

    async def on_claims(self, new_claims: List[Claim]) -> MetabolismResult:
        """
        Trigger: New claims received for this event.

        This is called when the pool routes new claims to this event.
        Decides and executes appropriate metabolism action.

        Returns:
            MetabolismResult with action taken and any side effects
        """
        logger.info(f"⚡ on_claims trigger: {self.event.canonical_name} received {len(new_claims)} claims")

        # Capture state before processing
        old_coherence = self.event.coherence if self.event.coherence else 0.5
        old_claim_count = len(self.claims)

        # Event examines claims via its metabolism (acceptance/rejection)
        examination = await self.service.examine_claims(self.event, new_claims)

        # Update internal state with accepted claims
        claims_added = []
        if examination.claims_added:
            existing_claims = list(self.claims)
            self.claims.extend(examination.claims_added)
            claims_added = examination.claims_added

            for claim in examination.claims_added:
                self.entity_ids.update(claim.entity_ids)
            self.last_claim_added = datetime.utcnow()

            # Load priors for new claims
            new_claim_ids = [c.id for c in examination.claims_added]
            new_priors = await self.service.claim_repo.get_publisher_priors_for_claims(new_claim_ids)
            new_urls = await self.service.claim_repo.get_page_urls_for_claims(new_claim_ids)
            self.publisher_priors.update(new_priors)
            self.page_urls.update(new_urls)

            logger.info(f"✅ {self.event.canonical_name} accepted {len(claims_added)} claims")

        if examination.claims_rejected:
            logger.info(f"❌ {self.event.canonical_name} rejected {len(examination.claims_rejected)} claims")

        if examination.sub_events_created:
            logger.info(f"🌿 {self.event.canonical_name} created {len(examination.sub_events_created)} sub-events")

        # Update metabolism state
        self._metabolism_state.claims_since_topology += len(claims_added)
        self._metabolism_state.claims_since_narrative += len(claims_added)

        # Decide what metabolism action to take
        action = self._decide_metabolism_action(
            trigger=TriggerType.CLAIMS_ADDED,
            context={
                'claims_added': len(claims_added),
                'old_coherence': old_coherence,
                'old_claim_count': old_claim_count,
                'existing_claims': existing_claims if claims_added else []
            }
        )

        # Execute the action
        result = await self._execute_action(action, context={
            'new_claims': claims_added,
            'existing_claims': existing_claims if claims_added else [],
            'old_coherence': old_coherence
        })

        # Attach examination results
        result.claims_added = claims_added
        result.claims_rejected = examination.claims_rejected
        result.sub_events_created = examination.sub_events_created

        return result

    async def on_time_tick(self) -> MetabolismResult:
        """
        Trigger: Periodic metabolism cycle.

        Called by the pool's metabolism_cycle() at regular intervals.
        Checks if any maintenance actions are needed.

        Returns:
            MetabolismResult with action taken
        """
        logger.debug(f"⏱️ on_time_tick trigger: {self.event.canonical_name}")

        # Decide what action to take based on current state
        action = self._decide_metabolism_action(
            trigger=TriggerType.TIME_TICK,
            context={}
        )

        # Execute the action
        return await self._execute_action(action)

    async def on_hydration_complete(self) -> MetabolismResult:
        """
        Trigger: Called after hydration from storage completes.

        This is the key trigger that fixes the Jimmy Lai bug:
        Events loaded from storage with claims but no topology
        will now automatically get topology computed.

        Returns:
            MetabolismResult with action taken
        """
        logger.info(f"💧 on_hydration_complete trigger: {self.event.canonical_name} "
                   f"({len(self.claims)} claims, topology={'yes' if self._topology_result else 'no'})")

        # Decide what action to take
        action = self._decide_metabolism_action(
            trigger=TriggerType.HYDRATION_COMPLETE,
            context={}
        )

        # Execute the action
        return await self._execute_action(action)

    async def on_command(self, command: str, params: dict = None) -> MetabolismResult:
        """
        Trigger: External command received.

        Commands are MCP-like paths that trigger specific behaviors.
        This is the event's "listening" interface.

        Supported commands:
        - /retopologize: Force full Bayesian topology analysis
        - /regenerate: Regenerate narrative
        - /rethink: Regenerate just the thought (no topology recompute)
        - /status: Return current event status
        - /rehydrate: Reload claims from storage

        Returns:
            MetabolismResult with action taken and command result
        """
        params = params or {}

        # Normalize command (strip slashes, lowercase)
        cmd = command.strip('/').lower()

        logger.info(f"🎯 on_command trigger: {self.event.canonical_name} received /{cmd}")

        # Map commands to action types
        command_actions = {
            'retopologize': ActionType.FULL_TOPOLOGY,
            'regenerate': ActionType.REGENERATE_NARRATIVE,
            'rethink': ActionType.EMIT_THOUGHT,  # Just regenerate thought
            'status': ActionType.NO_OP,  # Status is handled specially
            'rehydrate': ActionType.NO_OP,  # Rehydrate is handled specially
        }

        if cmd not in command_actions:
            return MetabolismResult(
                action_type=ActionType.NO_OP,
                success=False,
                error=f'Unknown command: {command}',
                data={'available_commands': list(command_actions.keys())}
            )

        # Handle special commands that bypass the action system
        if cmd == 'status':
            return self._get_status()
        if cmd == 'rehydrate':
            return await self._handle_rehydrate()
        if cmd == 'rethink':
            return await self._handle_rethink()

        # Create action for the command
        action = MetabolismAction(
            type=command_actions[cmd],
            trigger=TriggerType.COMMAND,
            reason=f"Command: /{cmd}",
            params=params
        )

        # For retopologize, force full re-analysis
        if cmd == 'retopologize':
            self._topology_result = None
            self.last_topology_update = None

        return await self._execute_action(action)

    # ==================== Decision Engine ====================

    def _decide_metabolism_action(
        self,
        trigger: TriggerType,
        context: dict = None
    ) -> MetabolismAction:
        """
        Central decision engine for metabolism.

        Based on current state and trigger, decides what action to take.
        This is the "brain" of the living event.

        Decision Rules (in priority order):
        1. No claims -> NO_OP
        2. Hydration complete + no topology + claims >= threshold -> FULL_TOPOLOGY
        3. Claims trigger + has topology + claims added -> INCREMENTAL_TOPOLOGY
        4. Claims trigger + no topology + claims >= threshold -> FULL_TOPOLOGY
        5. Time tick + needs narrative update -> REGENERATE_NARRATIVE
        6. Time tick + should hibernate -> HIBERNATE
        7. Anomaly detected -> EMIT_THOUGHT
        8. Default -> NO_OP

        Args:
            trigger: What triggered this decision
            context: Additional context (claims_added, etc.)

        Returns:
            MetabolismAction to execute
        """
        context = context or {}
        rules = self._metabolism_rules

        # Rule 1: No claims -> NO_OP
        if len(self.claims) == 0:
            return MetabolismAction(
                type=ActionType.NO_OP,
                trigger=trigger,
                reason="No claims to process"
            )

        # Rule 2: Hydration complete + no topology + sufficient claims -> FULL_TOPOLOGY
        if trigger == TriggerType.HYDRATION_COMPLETE:
            if not self._topology_result and len(self.claims) >= rules.min_claims_for_topology:
                # Check if we have any plausibilities (might have been loaded)
                has_stored_plausibilities = hasattr(self, 'claim_plausibilities') and self.claim_plausibilities
                if not has_stored_plausibilities:
                    return MetabolismAction(
                        type=ActionType.FULL_TOPOLOGY,
                        trigger=trigger,
                        reason=f"Post-hydration: {len(self.claims)} claims without topology"
                    )

        # Rule 3: Claims trigger + has topology -> INCREMENTAL_TOPOLOGY
        if trigger == TriggerType.CLAIMS_ADDED:
            claims_added = context.get('claims_added', 0)
            if claims_added > 0 and self._topology_result and self.topology_service:
                return MetabolismAction(
                    type=ActionType.INCREMENTAL_TOPOLOGY,
                    trigger=trigger,
                    reason=f"Incremental update for {claims_added} new claims",
                    params={
                        'new_claims': context.get('new_claims', []),
                        'existing_claims': context.get('existing_claims', [])
                    }
                )

        # Rule 4: Claims trigger + no topology + sufficient claims -> FULL_TOPOLOGY
        if trigger == TriggerType.CLAIMS_ADDED:
            if not self._topology_result and len(self.claims) >= rules.min_claims_for_topology:
                return MetabolismAction(
                    type=ActionType.FULL_TOPOLOGY,
                    trigger=trigger,
                    reason=f"Initial topology for {len(self.claims)} claims"
                )

        # Rule 5: Time tick + needs narrative update -> REGENERATE_NARRATIVE
        if trigger == TriggerType.TIME_TICK:
            if self.needs_narrative_update():
                return MetabolismAction(
                    type=ActionType.REGENERATE_NARRATIVE,
                    trigger=trigger,
                    reason="Narrative stale, claims recently added"
                )

        # Rule 6: Time tick + should hibernate -> HIBERNATE
        if trigger == TriggerType.TIME_TICK:
            if self.should_hibernate():
                return MetabolismAction(
                    type=ActionType.HIBERNATE,
                    trigger=trigger,
                    reason=f"Idle for {self.idle_time_seconds():.0f}s"
                )

        # Rule 7: Check for anomalies that warrant thought emission
        thought = self._detect_anomaly(trigger, context)
        if thought:
            return MetabolismAction(
                type=ActionType.EMIT_THOUGHT,
                trigger=trigger,
                reason=thought.content,
                params={'thought': thought}
            )

        # Default: NO_OP
        return MetabolismAction(
            type=ActionType.NO_OP,
            trigger=trigger,
            reason="No action needed"
        )

    def _detect_anomaly(self, trigger: TriggerType, context: dict) -> Optional[Thought]:
        """
        Detect epistemic anomalies that warrant thought emission.

        Anomalies include:
        - High temperature (many contradictions)
        - Sudden coherence drop
        - Source diversity concern
        - Stale topology

        Returns:
            Thought object if anomaly detected, None otherwise
        """
        rules = self._metabolism_rules

        # Check temperature (contradiction ratio)
        temperature = self._calculate_temperature()
        if temperature > rules.high_temperature_threshold:
            return Thought(
                id=generate_id('th'),
                event_id=self.event.id,
                type=ThoughtType.CONTRADICTION_DETECTED,
                content=f"High temperature ({temperature:.2f}): Many contradicting claims detected",
                related_claims=[],
                related_entities=[],
                temperature=temperature,
                coherence=self.event.coherence or 0.5,
                created_at=datetime.utcnow(),
                acknowledged=False
            )

        # Check coherence drop on claims trigger
        if trigger == TriggerType.CLAIMS_ADDED:
            old_coherence = context.get('old_coherence', 0.5)
            new_coherence = self.event.coherence or 0.5
            delta = new_coherence - old_coherence
            if delta < -rules.coherence_drop_threshold:
                return Thought(
                    id=generate_id('th'),
                    event_id=self.event.id,
                    type=ThoughtType.COHERENCE_DROP,
                    content=f"Coherence dropped significantly ({old_coherence:.2f} → {new_coherence:.2f})",
                    related_claims=[c.id for c in context.get('new_claims', [])[:5]],
                    related_entities=[],
                    temperature=temperature,
                    coherence=new_coherence,
                    created_at=datetime.utcnow(),
                    acknowledged=False
                )

        return None

    def _calculate_temperature(self) -> float:
        """
        Calculate event temperature (0=stable, 1=chaotic).

        Temperature is based on contradiction ratio in topology.
        """
        if not self._topology_result or not self._topology_result.contradictions:
            return 0.0

        # Temperature = contradictions / total_claims (capped at 1.0)
        num_contradictions = len(self._topology_result.contradictions)
        num_claims = len(self.claims)

        if num_claims == 0:
            return 0.0

        return min(1.0, num_contradictions / num_claims)

    # ==================== Action Executors ====================

    async def _execute_action(self, action: MetabolismAction, context: dict = None) -> MetabolismResult:
        """
        Execute a metabolism action.

        Dispatches to appropriate handler based on action type.
        """
        context = context or {}

        executors = {
            ActionType.NO_OP: self._exec_no_op,
            ActionType.FULL_TOPOLOGY: self._exec_full_topology,
            ActionType.INCREMENTAL_TOPOLOGY: self._exec_incremental_topology,
            ActionType.REGENERATE_NARRATIVE: self._exec_regenerate_narrative,
            ActionType.EMIT_THOUGHT: self._exec_emit_thought,
            ActionType.HIBERNATE: self._exec_hibernate,
        }

        executor = executors.get(action.type)
        if not executor:
            return MetabolismResult(
                action_type=action.type,
                success=False,
                error=f"No executor for action type: {action.type}"
            )

        try:
            result = await executor(action, context)
            logger.info(f"🔄 Executed {action.type.value}: {action.reason}")
            return result
        except Exception as e:
            logger.error(f"❌ Action {action.type.value} failed: {e}")
            return MetabolismResult(
                action_type=action.type,
                success=False,
                error=str(e)
            )

    async def _exec_no_op(self, action: MetabolismAction, context: dict) -> MetabolismResult:
        """Execute NO_OP action (do nothing)"""
        return MetabolismResult(
            action_type=ActionType.NO_OP,
            success=True,
            data={'reason': action.reason}
        )

    async def _exec_full_topology(self, action: MetabolismAction, context: dict) -> MetabolismResult:
        """
        Execute FULL_TOPOLOGY action.

        Runs complete Bayesian topology analysis:
        1. Analyze all claims
        2. Compute plausibilities
        3. Detect contradictions
        4. Persist to graph
        5. Regenerate narrative
        """
        if not self.topology_service:
            return MetabolismResult(
                action_type=ActionType.FULL_TOPOLOGY,
                success=False,
                error="Topology service not available"
            )

        if len(self.claims) < self._metabolism_rules.min_claims_for_topology:
            return MetabolismResult(
                action_type=ActionType.FULL_TOPOLOGY,
                success=False,
                error=f"Need at least {self._metabolism_rules.min_claims_for_topology} claims"
            )

        logger.info(f"🧬 Running full topology analysis for {len(self.claims)} claims")

        # Run full topology analysis
        topology = await self.topology_service.analyze(
            claims=self.claims,
            publisher_priors=self.publisher_priors,
            page_urls=self.page_urls
        )

        # Cache result
        self._topology_result = topology
        self.last_topology_update = datetime.utcnow()
        self._metabolism_state.claims_since_topology = 0

        # Persist topology to graph
        await self._persist_topology(topology)

        # Generate narrative with topology context
        topology_context = self.topology_service.get_topology_context(topology)
        narrative = await self.service.generate_structured_narrative(
            self.event,
            self.claims,
            topology_context
        )

        # Update storage
        flat_narrative = narrative.to_flat_text()
        await self.service.event_repo.update_narrative(
            self.event.id,
            flat_narrative,
            narrative.to_dict()
        )

        # Update internal state
        self.event.narrative = narrative
        self.event.summary = flat_narrative
        self.last_narrative_update = datetime.utcnow()
        self._metabolism_state.claims_since_narrative = 0

        # Calculate and update coherence
        new_coherence = await self._calculate_coherence()
        await self.service.event_repo.update_coherence(self.event.id, new_coherence)
        self.event.coherence = new_coherence

        logger.info(f"✨ Full topology complete: pattern={topology.pattern}, "
                   f"contradictions={len(topology.contradictions)}, coherence={new_coherence:.3f}")

        # Generate epistemic thought summarizing topology + narrative state
        thought = await self._generate_epistemic_thought(
            topology=topology,
            narrative=narrative,
            coherence=new_coherence
        )
        if thought:
            self._thoughts.append(thought)
            await self.service.event_repo.add_thought(self.event.id, thought)
            logger.info(f"💭 Generated thought: {thought.content[:80]}...")

        return MetabolismResult(
            action_type=ActionType.FULL_TOPOLOGY,
            success=True,
            thoughts=[thought] if thought else [],
            data={
                'claims_analyzed': len(self.claims),
                'pattern': topology.pattern,
                'contradictions': len(topology.contradictions),
                'coherence': new_coherence,
                'narrative_sections': len(narrative.sections)
            }
        )

    async def _exec_incremental_topology(self, action: MetabolismAction, context: dict) -> MetabolismResult:
        """
        Execute INCREMENTAL_TOPOLOGY action.

        Updates topology with new claims only:
        1. Compare new claims vs existing
        2. Classify relationships
        3. Update posteriors locally
        4. Persist incremental changes
        """
        new_claims = context.get('new_claims', action.params.get('new_claims', []))
        existing_claims = context.get('existing_claims', action.params.get('existing_claims', []))

        if not new_claims:
            return MetabolismResult(
                action_type=ActionType.INCREMENTAL_TOPOLOGY,
                success=True,
                data={'reason': 'No new claims to process'}
            )

        if not self._topology_result or not self.topology_service:
            # Fall back to full topology
            return await self._exec_full_topology(action, context)

        logger.info(f"🔄 Running incremental topology for {len(new_claims)} new claims")

        try:
            # Run incremental update
            updated_topology, new_relationships = await self.topology_service.incremental_update(
                new_claims=new_claims,
                existing_topology=self._topology_result,
                existing_claims=existing_claims,
                publisher_priors=self.publisher_priors,
                page_urls=self.page_urls
            )

            # Update cached topology
            self._topology_result = updated_topology
            self.last_topology_update = datetime.utcnow()
            self._metabolism_state.claims_since_topology = 0

            # Persist incremental changes
            if hasattr(self.service, 'topology_persistence') and self.service.topology_persistence:
                new_claim_ids = [c.id for c in new_claims]
                for claim_id in new_claim_ids:
                    if claim_id in updated_topology.claim_plausibilities:
                        result = updated_topology.claim_plausibilities[claim_id]
                        await self.service.topology_persistence.update_claim_plausibility(
                            event_id=self.event.id,
                            claim_id=claim_id,
                            plausibility=result.posterior,
                            is_superseded=claim_id in updated_topology.superseded_by
                        )

                for rel in new_relationships:
                    await self.service.topology_persistence.add_claim_relationship(
                        claim1_id=rel['source'],
                        claim2_id=rel['target'],
                        rel_type=rel['type'],
                        similarity=rel.get('similarity')
                    )

            # Update coherence
            new_coherence = await self._calculate_coherence()
            await self.service.event_repo.update_coherence(self.event.id, new_coherence)
            self.event.coherence = new_coherence

            # Check if narrative regeneration needed
            old_coherence = context.get('old_coherence', 0.5)
            delta = new_coherence - old_coherence
            if delta > self._metabolism_rules.coherence_boost_threshold:
                logger.info(f"📈 Coherence boost (Δ={delta:.3f}) - triggering narrative regeneration")
                await self._exec_regenerate_narrative(
                    MetabolismAction(
                        type=ActionType.REGENERATE_NARRATIVE,
                        trigger=action.trigger,
                        reason="Coherence boost"
                    ),
                    context
                )

            logger.info(f"✨ Incremental topology: +{len(new_claims)} claims, "
                       f"+{len(new_relationships)} relationships, coherence={new_coherence:.3f}")

            return MetabolismResult(
                action_type=ActionType.INCREMENTAL_TOPOLOGY,
                success=True,
                data={
                    'new_claims': len(new_claims),
                    'new_relationships': len(new_relationships),
                    'coherence': new_coherence
                }
            )

        except Exception as e:
            logger.warning(f"Incremental topology failed, falling back to full: {e}")
            self._topology_result = None
            return await self._exec_full_topology(action, context)

    async def _exec_regenerate_narrative(self, action: MetabolismAction, context: dict) -> MetabolismResult:
        """
        Execute REGENERATE_NARRATIVE action.

        Regenerates narrative using current topology.
        """
        logger.info(f"📝 Regenerating narrative for {self.event.canonical_name}")

        if self.topology_service and len(self.claims) >= 3:
            # Use topology context if available
            if self._topology_result:
                topology_context = self.topology_service.get_topology_context(self._topology_result)
            else:
                # Run topology first
                topology = await self.topology_service.analyze(
                    claims=self.claims,
                    publisher_priors=self.publisher_priors,
                    page_urls=self.page_urls
                )
                self._topology_result = topology
                self.last_topology_update = datetime.utcnow()
                await self._persist_topology(topology)
                topology_context = self.topology_service.get_topology_context(topology)

            # Generate structured narrative
            narrative = await self.service.generate_structured_narrative(
                self.event,
                self.claims,
                topology_context
            )
            flat_narrative = narrative.to_flat_text()

            # Update storage
            await self.service.event_repo.update_narrative(
                self.event.id,
                flat_narrative,
                narrative.to_dict()
            )
        else:
            # Fallback: simple narrative
            narrative = StructuredNarrative(
                sections=[],
                pattern="unknown",
                generated_at=datetime.utcnow()
            )
            flat_narrative = await self.service._generate_event_narrative(self.event, self.claims)
            await self.service.event_repo.update_narrative(self.event.id, flat_narrative)

        # Update internal state
        self.event.narrative = narrative
        self.event.summary = flat_narrative
        self.last_narrative_update = datetime.utcnow()
        self._metabolism_state.claims_since_narrative = 0

        return MetabolismResult(
            action_type=ActionType.REGENERATE_NARRATIVE,
            success=True,
            data={
                'narrative_length': len(flat_narrative),
                'sections': len(narrative.sections) if narrative else 0
            }
        )

    async def _exec_emit_thought(self, action: MetabolismAction, context: dict) -> MetabolismResult:
        """
        Execute EMIT_THOUGHT action.

        Stores thought to graph and adds to local cache.
        """
        thought = action.params.get('thought')
        if not thought:
            return MetabolismResult(
                action_type=ActionType.EMIT_THOUGHT,
                success=False,
                error="No thought provided"
            )

        logger.info(f"💭 Emitting thought: {thought.type.value} - {thought.content[:50]}...")

        # Add to local cache
        self._thoughts.append(thought)

        # Persist to Neo4j Event node
        if hasattr(self.service, 'event_repo'):
            await self.service.event_repo.add_thought(self.event.id, thought)

        return MetabolismResult(
            action_type=ActionType.EMIT_THOUGHT,
            success=True,
            thoughts=[thought],
            data={'thought_id': thought.id}
        )

    async def _exec_hibernate(self, action: MetabolismAction, context: dict) -> MetabolismResult:
        """
        Execute HIBERNATE action.

        Marks event for hibernation (pool will remove from active set).
        """
        logger.info(f"😴 Event entering hibernation: {self.event.canonical_name}")

        self._metabolism_state.is_hibernating = True

        return MetabolismResult(
            action_type=ActionType.HIBERNATE,
            success=True,
            data={
                'idle_seconds': self.idle_time_seconds(),
                'reason': action.reason
            }
        )

    # ==================== Thought Generation ====================

    async def _generate_epistemic_thought(
        self,
        topology,
        narrative,
        coherence: float
    ) -> Optional[Thought]:
        """
        Generate an epistemic thought summarizing topology + narrative state.

        Uses lightweight LLM (gpt-4o-mini) to create a concise, informative
        summary of what we know and how reliable it is.

        Example output:
        "13 sources corroborate death toll of 17. Two local reports conflict
        on fire origin (electrical vs arson). Timeline spans Nov 15-17."

        Args:
            topology: TopologyResult from analysis
            narrative: StructuredNarrative generated
            coherence: Event coherence score

        Returns:
            Thought object or None if generation fails
        """
        if not self.topology_service or not hasattr(self.topology_service, 'openai'):
            logger.debug("Skipping thought generation - no OpenAI client")
            return None

        try:
            # Build context for LLM
            num_claims = len(self.claims)
            num_sources = len(set(self.page_urls.values())) if self.page_urls else 0
            num_contradictions = len(topology.contradictions) if topology.contradictions else 0
            pattern = topology.pattern if topology else "unknown"

            # Get high/low plausibility claims for context
            high_plaus_claims = []
            low_plaus_claims = []
            if topology and topology.claim_plausibilities:
                sorted_claims = sorted(
                    topology.claim_plausibilities.items(),
                    key=lambda x: x[1].posterior,
                    reverse=True
                )
                # Top 3 high plausibility
                for claim_id, result in sorted_claims[:3]:
                    claim = next((c for c in self.claims if c.id == claim_id), None)
                    if claim:
                        high_plaus_claims.append(f"- {claim.text[:100]}... (p={result.posterior:.2f})")

                # Bottom 3 (if low)
                for claim_id, result in sorted_claims[-3:]:
                    if result.posterior < 0.4:
                        claim = next((c for c in self.claims if c.id == claim_id), None)
                        if claim:
                            low_plaus_claims.append(f"- {claim.text[:100]}... (p={result.posterior:.2f})")

            # Build contradiction summary
            contradiction_text = ""
            if topology.contradictions:
                contradiction_pairs = []
                for contra in topology.contradictions[:3]:
                    # Contradictions are dicts with claim1_id, claim2_id, text1, text2
                    text1 = contra.get('text1', '')[:50]
                    text2 = contra.get('text2', '')[:50]
                    if text1 and text2:
                        contradiction_pairs.append(f"'{text1}...' vs '{text2}...'")
                if contradiction_pairs:
                    contradiction_text = "Contradictions:\n" + "\n".join(contradiction_pairs)

            # Narrative sections summary
            sections_text = ""
            if narrative and narrative.sections:
                section_names = [s.title for s in narrative.sections[:5]]
                sections_text = f"Narrative covers: {', '.join(section_names)}"

            # Get high-plausibility claims (>0.5) for grounding facts
            reliable_claims_text = ""
            if topology and topology.claim_plausibilities:
                reliable = [(cid, res.posterior) for cid, res in topology.claim_plausibilities.items()
                           if res.posterior >= 0.5]
                reliable.sort(key=lambda x: x[1], reverse=True)
                texts = []
                for cid, plaus in reliable[:8]:
                    claim = next((c for c in self.claims if c.id == cid), None)
                    if claim:
                        texts.append(f"- {claim.text[:200]}")
                reliable_claims_text = "\n".join(texts) if texts else "(still analyzing...)"

            # Get update chains (what's changed over time)
            updates_text = ""
            if topology and hasattr(topology, 'superseded_by') and topology.superseded_by:
                update_items = []
                for old_id, new_id in list(topology.superseded_by.items())[:3]:
                    old_claim = next((c for c in self.claims if c.id == old_id), None)
                    new_claim = next((c for c in self.claims if c.id == new_id), None)
                    if old_claim and new_claim:
                        update_items.append(f"- UPDATED: '{old_claim.text[:60]}...' → '{new_claim.text[:60]}...'")
                if update_items:
                    updates_text = "Recent updates:\n" + "\n".join(update_items)

            prompt = f"""You're a witty, sharp-eyed journalist writing a one-liner that captures the ESSENCE of this developing story - what's surprising, ironic, or demands attention.

Event: {self.event.canonical_name}

Key facts:
{reliable_claims_text}

{updates_text}

{contradiction_text if contradiction_text else ""}

Write ONE punchy sentence (max 180 chars) that:
- Captures what makes this story INTERESTING right now
- Highlights irony, tension, or the unexpected angle
- Makes a reader say "wait, what?" or "I need to know more"

STYLE:
- Be sharp, not dry. Think NYT breaking news meets Twitter wit.
- Lead with the hook, not the background
- Specific details > vague summaries
- OK to be slightly provocative if grounded in facts above

Good examples:
- "Death toll hits 17 as residents say fire alarms never sounded. Building passed inspection last month."
- "Lai spends 77th birthday in prison cell. UK calls it 'politically motivated'; Beijing calls it justice."
- "19-year-old suspect in court as Seyfried doubles down: 'I'm not apologizing for calling him hateful.'"

Bad examples:
- "Fire kills many in building. Investigation ongoing."
- "Activist sentenced. International reaction mixed."

Your line:"""

            response = await self.topology_service.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )

            content = response.choices[0].message.content.strip()
            # Clean up any quotes
            content = content.strip('"\'')

            thought = Thought(
                id=generate_id('thought'),
                event_id=self.event.id,
                type=ThoughtType.TOPOLOGY_INSIGHT,
                content=content,
                related_claims=[],
                related_entities=[],
                temperature=self._calculate_temperature(),
                coherence=coherence,
                created_at=datetime.utcnow(),
                acknowledged=False
            )

            return thought

        except Exception as e:
            logger.warning(f"Failed to generate epistemic thought: {e}")
            return None

    # ==================== Helper Methods ====================

    async def hydrate(self, claim_repo):
        """
        Load full state from storage.

        Called when loading existing event into pool.
        Restores claims, priors, and cached plausibility data.
        After hydration, triggers on_hydration_complete().
        """
        # Load all claims for this event
        self.claims = await claim_repo.get_by_event(self.event.id)

        # Hydrate entities for each claim (needed for narrative generation)
        for claim in self.claims:
            await claim_repo.hydrate_entities(claim)

        # Extract entity IDs
        for claim in self.claims:
            self.entity_ids.update(claim.entity_ids)

        claim_ids = [c.id for c in self.claims]

        # Load publisher priors (stored on publisher entity at extraction time)
        self.publisher_priors = await claim_repo.get_publisher_priors_for_claims(claim_ids)

        # Also load page URLs as fallback for claims without stored priors
        self.page_urls = await claim_repo.get_page_urls_for_claims(claim_ids)

        # Load cached plausibilities from Neo4j (stored on SUPPORTS relationship)
        await self._hydrate_plausibilities(claim_repo)

        # Set last claim time (convert Neo4j DateTime to Python datetime)
        if self.claims:
            claim_times = []
            for c in self.claims:
                if c.created_at:
                    dt = neo4j_datetime_to_python(c.created_at)
                    if dt:
                        claim_times.append(dt)
            if claim_times:
                self.last_claim_added = max(claim_times)

        # Log source prior coverage
        stored_priors = sum(1 for p in self.publisher_priors.values() if p.get('base_prior'))
        plaus_count = len(self.claim_plausibilities) if hasattr(self, 'claim_plausibilities') else 0
        logger.info(f"💧 Hydrated event: {self.event.canonical_name} ({len(self.claims)} claims, "
                   f"{stored_priors}/{len(claim_ids)} with stored priors, {plaus_count} with plausibility)")

        # CRITICAL: Trigger post-hydration metabolism
        # This is what fixes the Jimmy Lai bug - events without topology
        # will now automatically get topology computed
        await self.on_hydration_complete()

    async def _hydrate_plausibilities(self, claim_repo):
        """
        Load cached plausibility scores and topology from Neo4j.

        Plausibilities are stored on SUPPORTS relationships during topology analysis.
        Topology metadata is stored on Event node.
        Loading them avoids re-running expensive topology on every hydrate.
        """
        # Dict to store plausibilities: claim_id -> float
        self.claim_plausibilities = {}

        # Load from Neo4j via event_repo (accessed through service)
        plausibilities = await self.service.event_repo.get_all_claim_plausibilities(self.event.id)

        if plausibilities:
            self.claim_plausibilities = plausibilities
            # Mark topology as already computed (don't need to re-run)
            self.last_topology_update = datetime.utcnow()
            logger.debug(f"📊 Loaded {len(plausibilities)} cached plausibilities")

        # Also try to reconstruct cached TopologyResult from stored data
        await self._hydrate_topology_result()

    async def _hydrate_topology_result(self):
        """
        Reconstruct TopologyResult from persisted topology data.

        This allows skipping expensive LLM re-analysis when topology is fresh.
        """
        if not hasattr(self.service, 'topology_persistence') or not self.service.topology_persistence:
            return

        try:
            topology_data = await self.service.topology_persistence.get_topology(self.event.id)
            if not topology_data:
                return

            # Import here to avoid circular dependency
            from services.claim_topology import TopologyResult, PlausibilityResult

            # Reconstruct PlausibilityResults from stored data
            claim_plausibilities = {}
            for c in topology_data.claims:
                claim_plausibilities[c.id] = PlausibilityResult(
                    claim_id=c.id,
                    prior=c.prior,
                    posterior=c.plausibility,
                    evidence_for=[],  # Not stored, but not needed for narrative
                    evidence_against=[],
                    confidence=c.plausibility
                )

            # Reconstruct superseded_by from update chains
            superseded_by = {}
            for chain in topology_data.update_chains:
                chain_list = chain.chain
                for i in range(len(chain_list) - 1):
                    superseded_by[chain_list[i]] = chain_list[i + 1]

            # Build minimal TopologyResult
            self._topology_result = TopologyResult(
                claim_plausibilities=claim_plausibilities,
                consensus_date=topology_data.consensus_date,
                contradictions=topology_data.contradictions,
                pattern=topology_data.pattern,
                superseded_by=superseded_by
            )

            logger.info(f"📊 Hydrated topology: pattern={topology_data.pattern}, "
                       f"{len(claim_plausibilities)} claims")

        except Exception as e:
            logger.warning(f"Failed to hydrate topology result: {e}")
            self._topology_result = None

    async def _persist_topology(self, topology):
        """
        Persist full topology to Neo4j graph.

        Stores:
        - Claim-to-claim edges (CORROBORATES, CONTRADICTS, UPDATES)
        - Plausibility scores on SUPPORTS edges
        - Topology metadata on Event node (pattern, consensus_date, etc.)
        """
        if not topology.claim_plausibilities:
            return

        try:
            if hasattr(self.service, 'topology_persistence') and self.service.topology_persistence:
                source_diversity = self._compute_source_diversity()

                await self.service.topology_persistence.store_topology(
                    event_id=self.event.id,
                    topology_result=topology,
                    source_diversity=source_diversity
                )
                logger.info(f"💾 Persisted full topology: {len(topology.claim_plausibilities)} claims, "
                           f"{len(topology.contradictions)} contradictions")
            else:
                # Fallback: just store plausibilities
                await self._update_claim_plausibilities_legacy(topology)

        except Exception as e:
            logger.warning(f"Failed to persist topology: {e}")
            await self._update_claim_plausibilities_legacy(topology)

    def _compute_source_diversity(self) -> Dict[str, Dict]:
        """
        Compute source diversity stats from publisher priors.

        Returns: {source_type: {count: N, avg_prior: X}}
        """
        source_stats = {}
        for claim_id, prior_data in self.publisher_priors.items():
            source_type = prior_data.get('source_type', 'unknown')
            if source_type not in source_stats:
                source_stats[source_type] = {'count': 0, 'total_prior': 0.0}
            source_stats[source_type]['count'] += 1
            source_stats[source_type]['total_prior'] += prior_data.get('base_prior', 0.5)

        # Compute averages
        for source_type, stats in source_stats.items():
            if stats['count'] > 0:
                stats['avg_prior'] = stats['total_prior'] / stats['count']
            del stats['total_prior']

        return source_stats

    async def _update_claim_plausibilities_legacy(self, topology):
        """
        Legacy: Update plausibility scores on SUPPORTS relationships in graph.
        """
        if not topology.claim_plausibilities:
            return

        try:
            for claim_id, result in topology.claim_plausibilities.items():
                await self.service.event_repo.update_claim_plausibility(
                    event_id=self.event.id,
                    claim_id=claim_id,
                    plausibility=result.posterior
                )
            logger.debug(f"📊 Updated plausibility for {len(topology.claim_plausibilities)} claims")
        except Exception as e:
            logger.warning(f"Failed to update claim plausibilities: {e}")

    async def _calculate_coherence(self) -> float:
        """
        Calculate event coherence using Jaynes' maximum entropy principle.

        Formula: coherence = 0.6 * hub_coverage + 0.4 * graph_connectivity
        """
        if not self.claims or not self.entity_ids:
            return 0.5  # Neutral coherence for empty event

        hub_coverage = await self._calculate_hub_coverage()
        graph_connectivity = await self._calculate_graph_connectivity()

        coherence = 0.6 * hub_coverage + 0.4 * graph_connectivity

        logger.debug(f"🧮 Coherence components: hub={hub_coverage:.3f}, connectivity={graph_connectivity:.3f}")

        return coherence

    async def _calculate_hub_coverage(self) -> float:
        """Calculate percentage of claims that touch hub entities."""
        if not self.claims:
            return 0.0

        # Count entity mentions across all claims
        entity_mention_counts = {}
        for claim in self.claims:
            for entity_id in claim.entity_ids:
                entity_mention_counts[entity_id] = entity_mention_counts.get(entity_id, 0) + 1

        # Identify hub entities (3+ mentions)
        hub_entities = {entity_id for entity_id, count in entity_mention_counts.items() if count >= 3}

        if not hub_entities:
            return 0.0

        # Count claims that touch at least one hub
        claims_touching_hubs = 0
        for claim in self.claims:
            if any(entity_id in hub_entities for entity_id in claim.entity_ids):
                claims_touching_hubs += 1

        return claims_touching_hubs / len(self.claims)

    async def _calculate_graph_connectivity(self) -> float:
        """Calculate graph connectivity using union-find algorithm."""
        if not self.claims or not self.entity_ids:
            return 0.0

        # Build union-find structure
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_x] = root_y

        # Create edges: claim <-> entity
        for claim in self.claims:
            claim_node = f"claim:{claim.id}"
            for entity_id in claim.entity_ids:
                entity_node = f"entity:{entity_id}"
                union(claim_node, entity_node)

        # Count connected components
        components = len(set(find(node) for node in parent.keys()))

        if components == 0:
            return 0.0

        return 1.0 / components

    def needs_narrative_update(self) -> bool:
        """Check if narrative needs regeneration."""
        if not self.last_claim_added:
            return False

        now = datetime.utcnow()

        # Handle timezone comparison
        last_claim = self.last_claim_added
        if last_claim.tzinfo is not None:
            last_claim = last_claim.replace(tzinfo=None)

        last_update = self.last_narrative_update
        if last_update.tzinfo is not None:
            last_update = last_update.replace(tzinfo=None)

        time_since_claim = (now - last_claim).total_seconds()
        time_since_update = (now - last_update).total_seconds()

        is_active = time_since_claim < 6 * 3600  # 6 hours
        is_stale = time_since_update > 3600  # 1 hour

        return is_active and is_stale

    def needs_topology_update(self) -> bool:
        """Check if topology analysis should be re-run."""
        if not self.last_topology_update:
            return True

        last_claim = self.last_claim_added
        last_topo = self.last_topology_update

        if last_claim:
            if hasattr(last_claim, 'tzinfo') and last_claim.tzinfo is not None:
                last_claim = last_claim.replace(tzinfo=None)
            if hasattr(last_topo, 'tzinfo') and last_topo.tzinfo is not None:
                last_topo = last_topo.replace(tzinfo=None)

            if last_claim > last_topo:
                return True

        topo_for_compare = self.last_topology_update
        if hasattr(topo_for_compare, 'tzinfo') and topo_for_compare.tzinfo is not None:
            topo_for_compare = topo_for_compare.replace(tzinfo=None)

        hours_since = (datetime.utcnow() - topo_for_compare).total_seconds() / 3600
        return hours_since > 6

    def idle_time_seconds(self) -> float:
        """Seconds since last claim was added"""
        now = datetime.utcnow()
        ref_time = self.last_claim_added if self.last_claim_added else self.created_at

        if ref_time and hasattr(ref_time, 'tzinfo') and ref_time.tzinfo is not None:
            from datetime import timezone
            now = datetime.now(timezone.utc)

        if not ref_time:
            return 0.0

        return (now - ref_time).total_seconds()

    def should_hibernate(self) -> bool:
        """Check if event should be archived (idle for 24+ hours)."""
        return self.idle_time_seconds() > 24 * 3600

    def get_state(self) -> LiveEventState:
        """Export current state"""
        return LiveEventState(
            event=self.event,
            claims=self.claims,
            entity_ids=self.entity_ids,
            last_claim_added=self.last_claim_added,
            last_narrative_update=self.last_narrative_update,
            created_at=self.created_at
        )

    def _get_status(self) -> MetabolismResult:
        """Return current event status."""
        claims_with_plausibility = 0
        if self._topology_result:
            claims_with_plausibility = len(self._topology_result.claim_plausibilities)

        return MetabolismResult(
            action_type=ActionType.NO_OP,
            success=True,
            data={
                'event_id': self.event.id,
                'canonical_name': self.event.canonical_name,
                'status': self.event.status,
                'confidence': self.event.confidence,
                'coherence': self.event.coherence,
                'temperature': self._calculate_temperature(),
                'claims_count': len(self.claims),
                'entity_count': len(self.entity_ids),
                'claims_with_plausibility': claims_with_plausibility,
                'last_claim_added': self.last_claim_added.isoformat() if self.last_claim_added else None,
                'last_narrative_update': self.last_narrative_update.isoformat(),
                'last_topology_update': self.last_topology_update.isoformat() if self.last_topology_update else None,
                'idle_seconds': self.idle_time_seconds(),
                'needs_narrative_update': self.needs_narrative_update(),
                'needs_topology_update': self.needs_topology_update(),
                'topology_pattern': self._topology_result.pattern if self._topology_result else None,
                'has_topology_service': self.topology_service is not None,
                'metabolism_state': {
                    'claims_since_topology': self._metabolism_state.claims_since_topology,
                    'claims_since_narrative': self._metabolism_state.claims_since_narrative,
                    'is_hibernating': self._metabolism_state.is_hibernating
                },
                'thoughts_count': len(self._thoughts)
            }
        )

    async def _handle_rehydrate(self) -> MetabolismResult:
        """Handle /rehydrate command - reload from storage."""
        old_claim_count = len(self.claims)

        claim_repo = self.service.claim_repo

        # Clear and reload
        self.claims = []
        self.entity_ids = set()
        self.publisher_priors = {}
        self.page_urls = {}
        self._topology_result = None

        await self.hydrate(claim_repo)

        return MetabolismResult(
            action_type=ActionType.NO_OP,
            success=True,
            data={
                'claims_before': old_claim_count,
                'claims_after': len(self.claims),
                'entities': len(self.entity_ids),
                'priors_loaded': len(self.publisher_priors)
            }
        )

    async def _handle_rethink(self) -> MetabolismResult:
        """
        Handle /rethink command - regenerate thought without recomputing topology.

        Uses existing topology result to generate a fresh thought.
        Much faster than /retopologize since it skips the expensive LLM
        relationship classification.
        """
        if not self._topology_result:
            return MetabolismResult(
                action_type=ActionType.EMIT_THOUGHT,
                success=False,
                error="No topology available. Run /retopologize first."
            )

        # Generate new thought using existing topology
        thought = await self._generate_epistemic_thought(
            topology=self._topology_result,
            narrative=self.event.narrative,
            coherence=self.event.coherence or 0.5
        )

        if thought:
            self._thoughts.append(thought)
            await self.service.event_repo.add_thought(self.event.id, thought)
            logger.info(f"💭 Rethink generated: {thought.content[:80]}...")

            return MetabolismResult(
                action_type=ActionType.EMIT_THOUGHT,
                success=True,
                thoughts=[thought],
                data={'thought_id': thought.id, 'content': thought.content}
            )
        else:
            return MetabolismResult(
                action_type=ActionType.EMIT_THOUGHT,
                success=False,
                error="Failed to generate thought"
            )

    # ==================== Legacy Interface ====================
    # These methods maintain backwards compatibility with existing code.
    # They delegate to the new trigger-based interface.

    async def examine(self, new_claims: List[Claim]):
        """
        Legacy interface: Examine new claims.

        Delegates to on_claims() trigger.
        Returns an ExaminationResult-compatible object for backwards compatibility.
        """
        result = await self.on_claims(new_claims)

        # Create ExaminationResult-like object for backwards compatibility
        @dataclass
        class LegacyExaminationResult:
            claims_added: List[Claim]
            claims_rejected: List[Claim]
            sub_events_created: List

        return LegacyExaminationResult(
            claims_added=result.claims_added or [],
            claims_rejected=result.claims_rejected or [],
            sub_events_created=result.sub_events_created or []
        )

    async def regenerate_narrative(self):
        """Legacy interface: Regenerate narrative."""
        await self._exec_regenerate_narrative(
            MetabolismAction(
                type=ActionType.REGENERATE_NARRATIVE,
                trigger=TriggerType.COMMAND,
                reason="Legacy regenerate_narrative() call"
            ),
            {}
        )

    async def handle_command(self, command: str, params: dict = None) -> dict:
        """
        Legacy interface: Handle command.

        Delegates to on_command() trigger and converts result to dict.
        """
        result = await self.on_command(command, params)

        return {
            'success': result.success,
            'data': result.data,
            'error': result.error
        }

    def __repr__(self):
        return f"<LiveEvent {self.event.canonical_name} ({len(self.claims)} claims, idle={self.idle_time_seconds():.0f}s)>"
