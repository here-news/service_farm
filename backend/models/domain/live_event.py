"""
LiveEvent - Event as a living organism

A LiveEvent is an in-memory representation of an event that:
- Bootstraps from claims
- Hydrates state from storage
- Executes metabolism (examines claims, updates narrative)
- Uses Bayesian topology for plausibility-weighted narratives
- Hibernates when dormant

The pool maintains these organisms, preventing duplicates and managing lifecycle.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Set, Optional, Dict
from dataclasses import dataclass

from models.domain.event import Event, StructuredNarrative
from models.domain.claim import Claim
from utils.datetime_utils import neo4j_datetime_to_python

logger = logging.getLogger(__name__)


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
    A living event organism that maintains itself.

    Lifecycle:
    1. Bootstrap - created from initial claims
    2. Hydrate - loads full state from storage
    3. Metabolism - examines new claims, updates narrative
    4. Hibernate - archives when dormant

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

        logger.debug(f"🌱 LiveEvent created: {event.canonical_name} ({event.id})")

    async def hydrate(self, claim_repo):
        """
        Load full state from storage.

        Called when loading existing event into pool.
        Restores claims, priors, and cached plausibility data.
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

    async def examine(self, new_claims: List[Claim]):
        """
        Layer 3: Event's own metabolism processes claims.

        The living event examines claims and decides:
        - Accept claim into this event (same story)
        - Reject claim (unrelated)
        - Create sub-event for claim (related but distinct)

        This is the event's autonomous behavior - it processes claims
        based on its own understanding of what it is.

        Returns:
            ExaminationResult with decisions
        """
        logger.info(f"🔬 {self.event.canonical_name} examining {len(new_claims)} claims")

        # Capture coherence before examination
        old_coherence = self.event.coherence if self.event.coherence else 0.5

        # Event examines claims via its metabolism
        result = await self.service.examine_claims(self.event, new_claims)

        # Update internal state based on examination results
        if result.claims_added:
            self.claims.extend(result.claims_added)
            for claim in result.claims_added:
                self.entity_ids.update(claim.entity_ids)
            self.last_claim_added = datetime.utcnow()

            logger.info(f"✅ {self.event.canonical_name} accepted {len(result.claims_added)} claims")

            # Calculate new coherence after adding claims
            new_coherence = await self._calculate_coherence()

            # Update event coherence in storage
            await self.service.event_repo.update_coherence(self.event.id, new_coherence)
            self.event.coherence = new_coherence

            # Check coherence delta and trigger narrative regeneration if significant
            delta = new_coherence - old_coherence
            logger.info(f"📊 Coherence: {old_coherence:.3f} → {new_coherence:.3f} (Δ {delta:+.3f})")

            if delta > 0.1:
                logger.info(f"📈 Significant coherence boost (Δ={delta:.3f}) - triggering narrative regeneration")
                await self.regenerate_narrative()

        if result.claims_rejected:
            logger.info(f"❌ {self.event.canonical_name} rejected {len(result.claims_rejected)} claims")

        if result.sub_events_created:
            logger.info(f"🌿 {self.event.canonical_name} created {len(result.sub_events_created)} sub-events")

        return result

    async def _calculate_coherence(self) -> float:
        """
        Calculate event coherence using Jaynes' maximum entropy principle.

        Formula: coherence = 0.6 * hub_coverage + 0.4 * graph_connectivity

        Hub coverage: % of claims touching entities with 3+ mentions
        Graph connectivity: 1.0 / num_connected_components (union-find)

        Based on GitHub issue #3 specification.
        """
        if not self.claims or not self.entity_ids:
            return 0.5  # Neutral coherence for empty event

        hub_coverage = await self._calculate_hub_coverage()
        graph_connectivity = await self._calculate_graph_connectivity()

        coherence = 0.6 * hub_coverage + 0.4 * graph_connectivity

        logger.debug(f"🧮 Coherence components: hub={hub_coverage:.3f}, connectivity={graph_connectivity:.3f}")

        return coherence

    async def _calculate_hub_coverage(self) -> float:
        """
        Calculate percentage of claims that touch hub entities.

        Hub entity = entity mentioned in 3+ claims in this event.

        Returns: 0.0 to 1.0
        """
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

        hub_coverage = claims_touching_hubs / len(self.claims)

        logger.debug(f"🎯 Hub coverage: {claims_touching_hubs}/{len(self.claims)} claims touch {len(hub_entities)} hubs")

        return hub_coverage

    async def _calculate_graph_connectivity(self) -> float:
        """
        Calculate graph connectivity using union-find algorithm.

        Creates bipartite graph of claims and entities:
        - Nodes: claims + entities
        - Edges: claim mentions entity

        Connectivity = 1.0 / num_connected_components
        (1.0 = fully connected, lower = fragmented)

        Returns: 0.0 to 1.0
        """
        if not self.claims or not self.entity_ids:
            return 0.0

        # Build union-find structure
        parent = {}

        def find(x):
            """Find root with path compression"""
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            """Union two sets"""
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

        connectivity = 1.0 / components

        logger.debug(f"🕸️  Graph connectivity: {components} components → {connectivity:.3f}")

        return connectivity

    def needs_narrative_update(self) -> bool:
        """
        Check if narrative needs regeneration.

        Regenerate if:
        - Claims added in last 6 hours AND
        - Narrative not updated in last hour
        """
        if not self.last_claim_added:
            return False

        now = datetime.utcnow()

        # Handle timezone-aware vs naive datetime comparison
        last_claim = self.last_claim_added
        if last_claim.tzinfo is not None:
            # Convert to naive UTC for comparison
            last_claim = last_claim.replace(tzinfo=None)

        last_update = self.last_narrative_update
        if last_update.tzinfo is not None:
            last_update = last_update.replace(tzinfo=None)

        time_since_claim = (now - last_claim).total_seconds()
        time_since_update = (now - last_update).total_seconds()

        # Active period: claims added recently
        is_active = time_since_claim < 6 * 3600  # 6 hours

        # Stale narrative: not updated in a while
        is_stale = time_since_update > 3600  # 1 hour

        return is_active and is_stale

    async def regenerate_narrative(self):
        """
        Metabolism - update narrative based on accumulated claims.

        Uses Bayesian topology for structured narrative generation.
        Generates both StructuredNarrative (for API) and flat summary (for backwards compat).
        """
        logger.info(f"📝 Regenerating narrative for {self.event.canonical_name}")

        if self.topology_service and len(self.claims) >= 3:
            # Use Bayesian topology for structured narrative
            structured_narrative = await self._regenerate_narrative_bayesian()
        else:
            # Fallback: create minimal structured narrative
            structured_narrative = StructuredNarrative(
                sections=[],
                pattern="unknown",
                generated_at=datetime.utcnow()
            )
            # Generate legacy flat narrative
            flat_narrative = await self.service._generate_event_narrative(self.event, self.claims)
            # We'll set summary separately below

        # Update in storage - both structured and flat
        if structured_narrative.sections:
            flat_narrative = structured_narrative.to_flat_text()
            await self.service.event_repo.update_narrative(
                self.event.id,
                flat_narrative,
                structured_narrative.to_dict()
            )
        else:
            # Fallback path - no structured narrative available
            await self.service.event_repo.update_narrative(self.event.id, flat_narrative)

        # Update internal state
        self.event.narrative = structured_narrative
        self.event.summary = flat_narrative
        self.last_narrative_update = datetime.utcnow()

        section_count = len(structured_narrative.sections) if structured_narrative else 0
        logger.info(f"✨ Narrative updated: {section_count} sections, {len(flat_narrative)} chars")

    async def _regenerate_narrative_bayesian(self) -> StructuredNarrative:
        """
        Regenerate narrative using Bayesian claim topology.

        Jaynes-informed approach:
        1. Use stored publisher priors (assigned at extraction time)
        2. Compute plausibility posteriors for all claims (or use cached)
        3. Detect date consensus and penalize outliers
        4. Identify and report contradictions
        5. Persist topology to graph (for visualization/API)
        6. Generate structured narrative via EventService with topology context

        Returns:
            StructuredNarrative with sections and key figures
        """
        logger.info(f"🧬 Using Bayesian topology for {len(self.claims)} claims")

        # Check if we can use cached topology (skip expensive LLM analysis)
        if self._topology_result and not self.needs_topology_update():
            topology = self._topology_result
            logger.info(f"📊 Using cached topology (pattern={topology.pattern})")
        else:
            # Run full topology analysis with stored publisher priors
            topology = await self.topology_service.analyze(
                claims=self.claims,
                publisher_priors=self.publisher_priors,
                page_urls=self.page_urls
            )

            # Cache result for potential reuse
            self._topology_result = topology
            self.last_topology_update = datetime.utcnow()

            # Persist full topology to graph (edges + metadata)
            await self._persist_topology(topology)

        # Get topology context for narrative generation
        topology_context = self.topology_service.get_topology_context(topology)

        # Generate STRUCTURED narrative using EventService
        narrative = await self.service.generate_structured_narrative(
            self.event,
            self.claims,
            topology_context
        )

        logger.info(f"🧬 Bayesian narrative: pattern={topology.pattern}, "
                   f"sections={len(narrative.sections)}, "
                   f"contradictions={len(topology.contradictions)}, "
                   f"consensus_date={topology.consensus_date}")

        return narrative

    async def _persist_topology(self, topology):
        """
        Persist full topology to Neo4j graph.

        Stores:
        - Claim-to-claim edges (CORROBORATES, CONTRADICTS, UPDATES)
        - Plausibility scores on SUPPORTS edges
        - Topology metadata on Event node (pattern, consensus_date, etc.)

        This enables the /topology API endpoint to serve pre-computed data.
        """
        if not topology.claim_plausibilities:
            return

        try:
            # Use TopologyPersistence service if available via event service
            if hasattr(self.service, 'topology_persistence') and self.service.topology_persistence:
                # Compute source diversity from publisher_priors
                source_diversity = self._compute_source_diversity()

                await self.service.topology_persistence.store_topology(
                    event_id=self.event.id,
                    topology_result=topology,
                    source_diversity=source_diversity
                )
                logger.info(f"💾 Persisted full topology: {len(topology.claim_plausibilities)} claims, "
                           f"{len(topology.contradictions)} contradictions")
            else:
                # Fallback: just store plausibilities (legacy behavior)
                await self._update_claim_plausibilities_legacy(topology)

        except Exception as e:
            logger.warning(f"Failed to persist topology: {e}")
            # Fallback to legacy plausibility storage
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

        Used as fallback when TopologyPersistence is not available.
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

    def needs_topology_update(self) -> bool:
        """
        Check if topology analysis should be re-run.

        Re-analyze if:
        - No topology computed yet, OR
        - Claims added since last topology update, OR
        - More than 6 hours since last update
        """
        if not self.last_topology_update:
            return True

        # Handle timezone-aware vs naive datetime comparison
        last_claim = self.last_claim_added
        last_topo = self.last_topology_update

        if last_claim:
            # Normalize to naive UTC for comparison
            if hasattr(last_claim, 'tzinfo') and last_claim.tzinfo is not None:
                last_claim = last_claim.replace(tzinfo=None)
            if hasattr(last_topo, 'tzinfo') and last_topo.tzinfo is not None:
                last_topo = last_topo.replace(tzinfo=None)

            if last_claim > last_topo:
                return True

        # Check time since last topology update
        topo_for_compare = self.last_topology_update
        if hasattr(topo_for_compare, 'tzinfo') and topo_for_compare.tzinfo is not None:
            topo_for_compare = topo_for_compare.replace(tzinfo=None)

        hours_since = (datetime.utcnow() - topo_for_compare).total_seconds() / 3600
        return hours_since > 6

    def idle_time_seconds(self) -> float:
        """Seconds since last claim was added"""
        now = datetime.utcnow()
        ref_time = self.last_claim_added if self.last_claim_added else self.created_at

        # Handle timezone-aware vs naive datetime comparison
        if ref_time and hasattr(ref_time, 'tzinfo') and ref_time.tzinfo is not None:
            # ref_time is timezone-aware, make now UTC-aware too
            from datetime import timezone
            now = datetime.now(timezone.utc)

        if not ref_time:
            return 0.0

        return (now - ref_time).total_seconds()

    def should_hibernate(self) -> bool:
        """
        Check if event should be archived.

        Hibernate if idle for 24+ hours.
        """
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

    # ==================== Command Handlers ====================
    # MCP-like command interface for external control of living events

    async def handle_command(self, command: str, params: dict = None) -> dict:
        """
        Handle a command sent to this living event.

        Commands are MCP-like paths that trigger specific behaviors.
        This is the event's "listening" interface - other parts of the
        system can communicate with living events through commands.

        Supported commands:
        - /retopologize: Re-run full Bayesian topology analysis and regenerate narrative
        - /regenerate: Regenerate narrative (uses cached topology if recent)
        - /status: Return current event status
        - /rehydrate: Reload claims from storage

        Args:
            command: Command path (e.g., '/retopologize')
            params: Optional parameters dict

        Returns:
            Result dict with 'success' and optional 'data' or 'error'
        """
        params = params or {}

        # Normalize command (strip slashes, lowercase)
        cmd = command.strip('/').lower()

        logger.info(f"🎯 {self.event.canonical_name} handling command: /{cmd}")

        # Dispatch to handler
        handlers = {
            'retopologize': self._cmd_retopologize,
            'regenerate': self._cmd_regenerate,
            'status': self._cmd_status,
            'rehydrate': self._cmd_rehydrate,
        }

        handler = handlers.get(cmd)
        if not handler:
            return {
                'success': False,
                'error': f'Unknown command: {command}',
                'available_commands': list(handlers.keys())
            }

        return await handler(params)

    async def _cmd_retopologize(self, params: dict) -> dict:
        """
        /retopologize - Re-run full Bayesian topology analysis.

        This forces a complete re-analysis of all claims:
        1. Regenerate embeddings if needed
        2. Build similarity network
        3. Classify relations (corroborate/contradict/update)
        4. Compute Bayesian posteriors
        5. Update plausibility scores in graph
        6. Regenerate weighted narrative

        Use this when:
        - New claims have been added and plausibilities are stale
        - Debugging claim scoring
        - After fixing topology bugs
        """
        logger.info(f"🧬 /retopologize: Starting full topology analysis for {self.event.canonical_name}")

        if not self.topology_service:
            return {
                'success': False,
                'error': 'Topology service not available'
            }

        if len(self.claims) < 3:
            return {
                'success': False,
                'error': f'Need at least 3 claims for topology analysis (have {len(self.claims)})'
            }

        # Force topology re-run by clearing cache
        self._topology_result = None
        self.last_topology_update = None

        # Run full Bayesian analysis and narrative regeneration
        structured_narrative = await self._regenerate_narrative_bayesian()

        # Update in storage - pass both flat and structured
        flat_narrative = structured_narrative.to_flat_text()
        await self.service.event_repo.update_narrative(
            self.event.id,
            flat_narrative,
            structured_narrative.to_dict()
        )
        self.event.narrative = structured_narrative
        self.event.summary = flat_narrative
        self.last_narrative_update = datetime.utcnow()

        topology = self._topology_result

        return {
            'success': True,
            'data': {
                'claims_analyzed': len(self.claims),
                'pattern': topology.pattern if topology else 'unknown',
                'contradictions': len(topology.contradictions) if topology else 0,
                'consensus_date': topology.consensus_date if topology else None,
                'narrative_length': len(flat_narrative),
                'sections': len(structured_narrative.sections),
                'key_figures': len(structured_narrative.key_figures)
            }
        }

    async def _cmd_regenerate(self, params: dict) -> dict:
        """
        /regenerate - Regenerate narrative.

        Uses existing topology if recent (< 1 hour), otherwise runs fresh analysis.
        Lighter weight than /retopologize.

        Params:
            force: If True, always regenerate (default: False)
        """
        force = params.get('force', False)

        logger.info(f"📝 /regenerate: Regenerating narrative for {self.event.canonical_name} (force={force})")

        if not force and not self.needs_narrative_update():
            return {
                'success': True,
                'data': {
                    'skipped': True,
                    'reason': 'Narrative is already up to date'
                }
            }

        await self.regenerate_narrative()

        return {
            'success': True,
            'data': {
                'narrative_length': len(self.event.summary) if self.event.summary else 0,
                'updated_at': self.last_narrative_update.isoformat()
            }
        }

    async def _cmd_status(self, params: dict) -> dict:
        """
        /status - Return current event status.

        Returns detailed status info for debugging and monitoring.
        """
        logger.info(f"📊 /status: Returning status for {self.event.canonical_name}")

        # Count claims with plausibility
        claims_with_plausibility = 0
        if self._topology_result:
            claims_with_plausibility = len(self._topology_result.claim_plausibilities)

        return {
            'success': True,
            'data': {
                'event_id': self.event.id,
                'canonical_name': self.event.canonical_name,
                'status': self.event.status,
                'confidence': self.event.confidence,
                'coherence': self.event.coherence,
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
                'has_topology_service': self.topology_service is not None
            }
        }

    async def _cmd_rehydrate(self, params: dict) -> dict:
        """
        /rehydrate - Reload claims and state from storage.

        Use this after manual database changes or to sync state.
        """
        logger.info(f"💧 /rehydrate: Reloading state for {self.event.canonical_name}")

        old_claim_count = len(self.claims)

        # Need claim_repo - get it from service
        claim_repo = self.service.claim_repo

        # Clear and reload
        self.claims = []
        self.entity_ids = set()
        self.publisher_priors = {}
        self.page_urls = {}

        await self.hydrate(claim_repo)

        return {
            'success': True,
            'data': {
                'claims_before': old_claim_count,
                'claims_after': len(self.claims),
                'entities': len(self.entity_ids),
                'priors_loaded': len(self.publisher_priors)
            }
        }

    def __repr__(self):
        return f"<LiveEvent {self.event.canonical_name} ({len(self.claims)} claims, idle={self.idle_time_seconds():.0f}s)>"
