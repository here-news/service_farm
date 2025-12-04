"""
Event Worker (Neo4j Graph-Based) - Universal Event Formation

Universal discovery-based event formation using two-layer architecture:
1. Abstract Model Layer: Discover patterns from data (temporal + semantic)
2. Natural Language Layer: Translate patterns to human-readable phase names

Key principles:
- No hardcoded phase types - discovered from data
- Works for any event domain (fire, election, scandal, etc.)
- Temporal segmentation discovers natural time boundaries
- Semantic clustering discovers topic groupings
- Coherence metric determines event complexity
- Umbrella strategy for multi-phase complex events

Design:
- PostgreSQL: Pages, raw claims, embeddings (content layer)
- Neo4j: Events, phases, claim nodes, relationships (structure layer)
- Worker: Read from PostgreSQL, discover structure, write to Neo4j
"""
import asyncio
import json
import logging
import os
import re
import uuid
from typing import Dict, List, Set, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter

import asyncpg
import numpy as np
from openai import AsyncOpenAI
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.job_queue import JobQueue
from services.neo4j_service import Neo4jService
from workers.event_attachment import EventAttachmentScorer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===== Discovery Layer: Universal Event Formation =====

class TemporalSegment:
    """Represents a discovered temporal segment"""
    def __init__(self, start_time: datetime, end_time: datetime, claims: List[Dict]):
        self.start_time = start_time
        self.end_time = end_time
        self.claims = claims
        self.span_hours = (end_time - start_time).total_seconds() / 3600 if end_time and start_time else 0

    def __repr__(self):
        return f"TemporalSegment({self.start_time.strftime('%Y-%m-%d %H:%M') if self.start_time else 'None'} - {self.end_time.strftime('%H:%M') if self.end_time else 'None'}, {len(self.claims)} claims)"


# Legacy SemanticCluster class removed - using new implementation below

class SemanticCluster:
    """Represents a discovered semantic cluster of claims"""
    def __init__(self, claims: List[Dict], centroid: np.ndarray, keywords: List[str]):
        self.claims = claims
        self.centroid = centroid
        self.keywords = keywords
        self.size = len(claims)


class EventScaffold:
    """
    Universal event scaffold with semantic clustering

    Strategy: Discover multi-phase structure from claim embeddings
    - Semantic clustering of claims using DBSCAN
    - Extract keywords from each cluster
    - Determine complexity and coherence
    - Create phases based on discovered clusters
    """
    def __init__(self, claims: List[Dict], page: Dict):
        self.claims = claims
        self.page = page

        # Temporal bounds from page pub_time (reliable anchor)
        self.earliest_time: Optional[datetime] = page.get('pub_time')
        self.latest_time: Optional[datetime] = page.get('pub_time')
        self.temporal_span_hours: float = 0.0

        # Discovered structure (populated by initialize())
        self.semantic_clusters: List[SemanticCluster] = []
        self.phase_metadata: List[Dict] = []  # LLM phase data
        self.complexity_score: int = 0
        self.coherence: float = 0.3  # Default low confidence

    async def _discover_with_llm(self, openai_client):
        """
        Discover semantic phases using LLM analysis

        Returns discovered clusters for phase creation
        """
        if len(self.claims) < 3:
            logger.info("🔍 First page - few claims, using single phase")
            logger.info(f"  ⏱️  Using pub_time as temporal anchor: {self.earliest_time}")
            logger.info(f"  🎯 Confidence: {self.coherence} (low - needs validation)")
            # Create single default cluster
            self.semantic_clusters = [SemanticCluster(
                claims=self.claims,
                centroid=None,
                keywords=['initial', 'reports']
            )]
            self.phase_metadata = [{"name": "Initial Reports", "description": "First reports", "claim_numbers": list(range(1, len(self.claims) + 1))}]
            return

        # Prepare claims for LLM analysis
        claims_text = "\n".join([
            f"{i+1}. {claim['text']}"
            for i, claim in enumerate(self.claims)
        ])

        prompt = f"""Analyze these claims from a news article and identify natural semantic phases/topics.

Claims:
{claims_text}

Instructions:
1. Group claims into 1-4 semantic phases based on their topics
2. Each phase should represent a distinct aspect of the event (e.g., "Fire Breakout", "Casualties", "Emergency Response", "Investigation")
3. Provide a clear, descriptive name for each phase (2-3 words)
4. List which claim numbers belong to each phase
5. All claims must be assigned to exactly one phase

Return JSON with this structure:
{{
  "phases": [
    {{
      "name": "Phase Name",
      "description": "What this phase covers",
      "claim_numbers": [1, 2, 3]
    }}
  ]
}}"""

        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing news content and identifying semantic structure."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            phases = result.get('phases', [])

            if not phases:
                logger.warning("⚠️  LLM returned no phases, using single default phase")
                self.semantic_clusters = [SemanticCluster(
                    claims=self.claims,
                    centroid=None,
                    keywords=['initial', 'reports']
                )]
                self.phase_metadata = [{"name": "Initial Reports", "description": "First reports", "claim_numbers": list(range(1, len(self.claims) + 1))}]
                return

            # Convert LLM phases to SemanticCluster objects
            self.semantic_clusters = []
            self.phase_metadata = []  # Store LLM phase data for logging

            for phase in phases:
                # Get claims for this phase
                claim_indices = [n - 1 for n in phase['claim_numbers'] if 0 <= n - 1 < len(self.claims)]
                phase_claims = [self.claims[i] for i in claim_indices]

                if not phase_claims:
                    continue

                # Extract keywords from phase name
                keywords = phase['name'].lower().split()

                self.semantic_clusters.append(SemanticCluster(
                    claims=phase_claims,
                    centroid=None,
                    keywords=keywords
                ))

                self.phase_metadata.append(phase)

            # Compute complexity and coherence
            self.complexity_score = self._compute_complexity()
            self.coherence = self._compute_coherence()

            logger.info(f"🔍 LLM discovered {len(self.semantic_clusters)} semantic phases from {len(self.claims)} claims")
            for i, (cluster, phase) in enumerate(zip(self.semantic_clusters, phases), 1):
                logger.info(f"  📊 Phase {i}: {phase['name']} ({len(cluster.claims)} claims)")
                logger.info(f"      {phase['description']}")
            logger.info(f"  🎯 Complexity score: {self.complexity_score}")
            logger.info(f"  🎯 Coherence: {self.coherence:.2f}")

        except Exception as e:
            logger.error(f"❌ LLM phase discovery failed: {e}")
            # Fallback to single phase
            self.semantic_clusters = [SemanticCluster(
                claims=self.claims,
                centroid=None,
                keywords=['initial', 'reports']
            )]
            self.phase_metadata = [{"name": "Initial Reports", "description": "First reports", "claim_numbers": list(range(1, len(self.claims) + 1))}]

    def _discover_semantic_clusters(self, claims_with_embeddings: List[Dict]) -> List[SemanticCluster]:
        """
        Cluster claims by semantic similarity using DBSCAN

        Returns list of SemanticCluster objects
        """
        # Extract embeddings
        embeddings = []
        for claim in claims_with_embeddings:
            emb = claim.get('embedding')
            if isinstance(emb, str):
                # Parse string embedding
                emb = np.array([float(x) for x in emb.strip('[]').split(',')], dtype=np.float32)
            elif isinstance(emb, (list, tuple)):
                emb = np.array(emb, dtype=np.float32)
            embeddings.append(emb)

        embeddings_matrix = np.array(embeddings)

        # DBSCAN clustering
        # eps=0.3 means claims within 0.3 cosine distance are neighbors
        # min_samples=2 means at least 2 claims to form a cluster
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
        labels = clustering.fit_predict(embeddings_matrix)

        # Group claims by cluster
        clusters_dict = defaultdict(list)
        for i, label in enumerate(labels):
            if label == -1:
                # Noise point - create singleton cluster
                clusters_dict[f'noise_{i}'] = [claims_with_embeddings[i]]
            else:
                clusters_dict[label].append(claims_with_embeddings[i])

        # Create SemanticCluster objects
        clusters = []
        for label, cluster_claims in clusters_dict.items():
            # Compute centroid
            cluster_embeddings = [embeddings[claims_with_embeddings.index(c)] for c in cluster_claims]
            centroid = np.mean(cluster_embeddings, axis=0)

            # Extract keywords
            keywords = self._extract_keywords(cluster_claims)

            clusters.append(SemanticCluster(
                claims=cluster_claims,
                centroid=centroid,
                keywords=keywords
            ))

        return clusters

    def _extract_keywords(self, claims: List[Dict]) -> List[str]:
        """Extract salient keywords from cluster claims"""
        # Combine all claim texts
        text = ' '.join([c['text'] for c in claims])

        # Simple keyword extraction (can be improved with TF-IDF)
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())

        # Count and return top keywords
        word_counts = Counter(words)
        # Filter out common words
        stopwords = {'that', 'this', 'with', 'from', 'have', 'been', 'were', 'said', 'would', 'their', 'there', 'about', 'which', 'when', 'where', 'after', 'before'}
        filtered_words = [(w, c) for w, c in word_counts.items() if w not in stopwords]

        # Return top 5 keywords
        top_keywords = [w for w, c in sorted(filtered_words, key=lambda x: x[1], reverse=True)[:5]]

        return top_keywords if top_keywords else ['event', 'reports']

    def _compute_complexity(self) -> int:
        """
        Compute complexity score based on discovered structure

        Indicators:
        - Multiple semantic clusters (≥3)
        - Many claims (≥8)
        """
        score = 0
        if len(self.semantic_clusters) >= 3:
            score += 1
        if len(self.claims) >= 8:
            score += 1
        return score

    def _compute_coherence(self) -> float:
        """
        Compute coherence score

        Simple heuristic for first page:
        - Single cluster = 0.8 (high coherence)
        - 2 clusters = 0.5 (moderate)
        - 3+ clusters = 0.3 (low - complex event)
        """
        num_clusters = len(self.semantic_clusters)
        if num_clusters == 1:
            return 0.8
        elif num_clusters == 2:
            return 0.5
        else:
            return 0.3

    def should_create_umbrella(self) -> bool:
        """Determine if umbrella event needed based on complexity"""
        return self.complexity_score >= 2

    # Legacy methods removed - see git history for old claim-based clustering
    def _discover_temporal_segments_LEGACY(self) -> List[TemporalSegment]:
        """
        Discover natural temporal boundaries from claims

        Algorithm:
        1. Extract all temporal markers from claims
        2. Cluster by temporal proximity (DBSCAN)
        3. Return temporal segments
        """
        # Extract times from claims
        timed_claims = []
        for claim in self.claims:
            if claim.get('event_time'):
                timed_claims.append({
                    'claim': claim,
                    'time': claim['event_time']
                })

        if not timed_claims:
            # No temporal info, return single segment
            return [TemporalSegment(None, None, self.claims)]

        # Sort by time
        timed_claims.sort(key=lambda x: x['time'])

        # Update time bounds
        self.earliest_time = timed_claims[0]['time']
        self.latest_time = timed_claims[-1]['time']
        self.temporal_span_hours = (self.latest_time - self.earliest_time).total_seconds() / 3600

        # If span is small (<2 hours), treat as single segment
        if self.temporal_span_hours < 2:
            return [TemporalSegment(self.earliest_time, self.latest_time, self.claims)]

        # Cluster by temporal proximity using DBSCAN
        times_array = np.array([(tc['time'] - self.earliest_time).total_seconds() / 3600 for tc in timed_claims]).reshape(-1, 1)

        # eps = 2 hours, min_samples = 1 (allow single-claim segments)
        clustering = DBSCAN(eps=2.0, min_samples=1).fit(times_array)

        # Group claims by cluster
        segments = []
        for cluster_id in set(clustering.labels_):
            cluster_indices = [i for i, label in enumerate(clustering.labels_) if label == cluster_id]
            cluster_claims = [timed_claims[i]['claim'] for i in cluster_indices]
            cluster_times = [timed_claims[i]['time'] for i in cluster_indices]

            segment = TemporalSegment(
                start_time=min(cluster_times),
                end_time=max(cluster_times),
                claims=cluster_claims
            )
            segments.append(segment)

        # Sort segments by start time
        segments.sort(key=lambda s: s.start_time if s.start_time else datetime.min)

        return segments

    # Legacy _discover_semantic_clusters(), _compute_coherence(), _assess_complexity(), and should_create_umbrella() removed - now in EventScaffold class


class DomainTranslator:
    """
    Translate abstract patterns to domain-specific natural language

    Uses keyword matching to infer phase names based on event domain
    """

    def __init__(self, event_type: str):
        self.event_type = event_type

    def translate_cluster_to_phase_name(
        self,
        semantic_cluster: SemanticCluster,
        temporal_segment: TemporalSegment,
        temporal_position: str
    ) -> Tuple[str, str]:
        """
        Translate abstract cluster to natural phase name

        Returns: (phase_name, phase_type)
        """
        keywords = semantic_cluster.keywords

        if self.event_type == "FIRE" or self.event_type == "DISASTER":
            return self._translate_disaster_phase(keywords, temporal_position)
        elif self.event_type == "ELECTION":
            return self._translate_election_phase(keywords, temporal_position)
        elif self.event_type == "SCANDAL":
            return self._translate_scandal_phase(keywords, temporal_position)
        elif self.event_type == "SHOOTING":
            return self._translate_violence_phase(keywords, temporal_position)
        else:
            return self._generic_phase_name(keywords, temporal_position)

    def _translate_disaster_phase(self, keywords: List[str], position: str) -> Tuple[str, str]:
        """Fire/Disaster domain translation"""
        keywords_str = ' '.join(keywords)

        # Check for incident phase
        if any(kw in keywords_str for kw in ['broke', 'started', 'ignited', 'occurred', 'engulfs', 'blaze', 'fire']):
            return ("Fire Breakout", "INCIDENT")

        # Check for response phase
        elif any(kw in keywords_str for kw in ['firefighters', 'rescue', 'evacuate', 'battle', 'extinguish', 'respond']):
            return ("Emergency Response", "RESPONSE")

        # Check for casualty phase
        elif any(kw in keywords_str for kw in ['dead', 'casualties', 'injured', 'toll', 'killed', 'victim']):
            return ("Casualty Assessment", "CONSEQUENCE")

        # Check for investigation phase
        elif any(kw in keywords_str for kw in ['investigation', 'probe', 'arrested', 'charged', 'cause']):
            return ("Investigation", "INVESTIGATION")

        # Check for political phase
        elif any(kw in keywords_str for kw in ['reform', 'regulations', 'inquiry', 'government', 'policy']):
            return ("Political Response", "POLITICAL")

        # Temporal fallback
        else:
            if position == "early":
                return ("Initial Incident", "INCIDENT")
            elif position == "middle":
                return ("Ongoing Response", "RESPONSE")
            else:
                return ("Aftermath", "CONSEQUENCE")

    def _translate_election_phase(self, keywords: List[str], position: str) -> Tuple[str, str]:
        """Election domain translation"""
        keywords_str = ' '.join(keywords)

        if any(kw in keywords_str for kw in ['campaign', 'rally', 'debate']):
            return ("Campaign Period", "BUILDUP")
        elif any(kw in keywords_str for kw in ['vote', 'ballot', 'polls', 'voting']):
            return ("Voting Day", "INCIDENT")
        elif any(kw in keywords_str for kw in ['results', 'winner', 'declared', 'victory']):
            return ("Results Announcement", "CONSEQUENCE")
        elif any(kw in keywords_str for kw in ['transition', 'inauguration', 'sworn']):
            return ("Transition Period", "AFTERMATH")
        else:
            return (f"Election Phase ({position})", "UNSPECIFIED")

    def _translate_scandal_phase(self, keywords: List[str], position: str) -> Tuple[str, str]:
        """Scandal domain translation"""
        keywords_str = ' '.join(keywords)

        if any(kw in keywords_str for kw in ['revealed', 'leaked', 'exposed']):
            return ("Initial Revelation", "INCIDENT")
        elif any(kw in keywords_str for kw in ['denied', 'statement', 'response', 'defense']):
            return ("Official Response", "RESPONSE")
        elif any(kw in keywords_str for kw in ['investigation', 'probe', 'inquiry']):
            return ("Investigation", "INVESTIGATION")
        elif any(kw in keywords_str for kw in ['resigned', 'stepped', 'consequences', 'impeach']):
            return ("Consequences", "CONSEQUENCE")
        else:
            return (f"Scandal Development ({position})", "UNSPECIFIED")

    def _translate_violence_phase(self, keywords: List[str], position: str) -> Tuple[str, str]:
        """Violence/Shooting domain translation"""
        keywords_str = ' '.join(keywords)

        if any(kw in keywords_str for kw in ['shoot', 'gunfire', 'attack', 'opened']):
            return ("Attack", "INCIDENT")
        elif any(kw in keywords_str for kw in ['police', 'respond', 'scene', 'arrest']):
            return ("Police Response", "RESPONSE")
        elif any(kw in keywords_str for kw in ['dead', 'casualties', 'injured', 'victim']):
            return ("Casualties Reported", "CONSEQUENCE")
        elif any(kw in keywords_str for kw in ['investigation', 'motive', 'suspect']):
            return ("Investigation", "INVESTIGATION")
        else:
            return (f"Shooting Phase ({position})", "UNSPECIFIED")

    def _generic_phase_name(self, keywords: List[str], position: str) -> Tuple[str, str]:
        """Generic fallback using top keywords"""
        if not keywords:
            return (f"Phase ({position})", "UNSPECIFIED")

        # Use top 2-3 keywords
        top_keywords = keywords[:min(3, len(keywords))]
        name = " ".join(top_keywords).title()

        # Add temporal qualifier
        if position == "early":
            return (f"Initial {name}", "INCIDENT")
        elif position == "late":
            return (f"Late {name}", "CONSEQUENCE")
        else:
            return (name, "UNSPECIFIED")

    def _assess_temporal_position(
        self,
        segment: TemporalSegment,
        earliest_time: datetime,
        latest_time: datetime
    ) -> str:
        """Is this segment early, middle, or late in event timeline?"""
        if not segment.start_time or not earliest_time or not latest_time:
            return "middle"

        total_span = (latest_time - earliest_time).total_seconds()
        if total_span == 0:
            return "early"

        segment_position = (segment.start_time - earliest_time).total_seconds() / total_span

        if segment_position < 0.33:
            return "early"
        elif segment_position > 0.67:
            return "late"
        else:
            return "middle"


class EventWorkerNeo4j:
    """
    Event worker using Neo4j graph scaffold
    """

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        neo4j_service: Neo4jService,
        job_queue: JobQueue,
        worker_id: int = 1
    ):
        self.db_pool = db_pool
        self.neo4j = neo4j_service
        self.job_queue = job_queue
        self.worker_id = worker_id
        self.attachment_scorer = EventAttachmentScorer()

        # Initialize OpenAI for title synthesis
        api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = AsyncOpenAI(api_key=api_key) if api_key else None

    async def start(self):
        """Start worker loop"""
        logger.info(f"📊 event-worker-neo4j-{self.worker_id} started")

        try:
            while True:
                try:
                    # Listen to job queue for new pages
                    job = await self.job_queue.dequeue('queue:event:high', timeout=5)

                    if job:
                        page_id = uuid.UUID(job['page_id'])
                        await self.process_page(page_id)

                except Exception as e:
                    logger.error(f"❌ Event worker error: {e}", exc_info=True)
                    await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")

    async def process_page(self, page_id: uuid.UUID):
        """
        Process page and create/attach to event graph

        Steps:
        1. Fetch page + claims from PostgreSQL
        2. Find candidate events from Neo4j (entity overlap)
        3. Score candidates using existing attachment scorer
        4. Decision: attach to existing event OR create new event
        5. Create graph structure in Neo4j
        """
        async with self.db_pool.acquire() as conn:
            # Fetch page data
            page = await conn.fetchrow("""
                SELECT id, url, title, embedding, pub_time, metadata_confidence
                FROM core.pages
                WHERE id = $1
            """, page_id)

            if not page:
                logger.warning(f"⚠️  Page {page_id} not found")
                return

            # Fetch claims with entities and embeddings
            claims = await conn.fetch("""
                SELECT
                    c.id, c.text, c.event_time, c.confidence, c.modality, c.embedding,
                    ARRAY_AGG(DISTINCT ce.entity_id) FILTER (WHERE ce.entity_id IS NOT NULL) as entity_ids,
                    ARRAY_AGG(DISTINCT e.canonical_name) FILTER (WHERE e.canonical_name IS NOT NULL) as entity_names,
                    ARRAY_AGG(DISTINCT e.entity_type) FILTER (WHERE e.entity_type IS NOT NULL) as entity_types
                FROM core.claims c
                LEFT JOIN core.claim_entities ce ON c.id = ce.claim_id
                LEFT JOIN core.entities e ON ce.entity_id = e.id
                WHERE c.page_id = $1
                GROUP BY c.id
            """, page_id)

            if not claims:
                logger.warning(f"⚠️  No claims found for page {page_id}")
                return

            claims_data = [{
                'id': c['id'],
                'text': c['text'],
                'event_time': c['event_time'],
                'confidence': c['confidence'],
                'modality': c['modality'],
                'embedding': c['embedding'],
                'entity_ids': list(c['entity_ids']) if c['entity_ids'] else [],
                'entity_names': list(c['entity_names']) if c['entity_names'] else [],
                'entity_types': list(c['entity_types']) if c['entity_types'] else []
            } for c in claims]

            logger.info(f"📄 Processing page {page['url']} ({len(claims_data)} claims)")

            # Collect all entity names from claims
            all_entity_names = set()
            for claim in claims_data:
                all_entity_names.update(claim['entity_names'])

            # Find or create event
            await self._find_or_create_event_graph(page, claims_data, all_entity_names)

    async def _find_or_create_event_graph(
        self,
        page: Dict,
        claims: List[Dict],
        entity_names: Set[str]
    ):
        """
        Find matching event in Neo4j or create new event graph

        Scoring is similar to old approach but we query Neo4j for candidates
        """
        # Get candidate events from Neo4j based on entity overlap
        reference_time = page['pub_time'] or (claims[0]['event_time'] if claims else datetime.utcnow())

        candidates = await self.neo4j.find_candidate_events(
            entity_names=entity_names,
            time_window_days=7,
            reference_time=reference_time,
            limit=10
        )

        if not candidates:
            logger.info("📝 No candidate events found, creating new event graph")
            await self._create_event_graph(page, claims, entity_names)
            return

        # Score each candidate (reuse existing EventAttachmentScorer)
        best_event = None
        best_score_value = 0.0

        # Query PostgreSQL for event embeddings (one batch query)
        candidate_ids = [c['e']['id'] for c in candidates]
        event_embeddings = {}

        async with self.db_pool.acquire() as conn:
            embedding_rows = await conn.fetch("""
                SELECT id, embedding FROM core.events WHERE id = ANY($1::uuid[])
            """, candidate_ids)

            for row in embedding_rows:
                event_embeddings[str(row['id'])] = row['embedding']

        for candidate in candidates:
            event_data = candidate['e']
            entity_overlap = candidate['entity_overlap']

            # Get candidate entity names from Neo4j
            candidate_entity_names = await self._get_event_entity_names(event_data['id'])

            # Convert Neo4j DateTime to Python datetime
            event_start = event_data.get('earliest_time')
            event_end = event_data.get('latest_time')

            # Neo4j returns DateTime objects, convert to Python datetime
            if event_start and hasattr(event_start, 'to_native'):
                event_start = event_start.to_native()
            if event_end and hasattr(event_end, 'to_native'):
                event_end = event_end.to_native()

            # Get event embedding from PostgreSQL
            event_embedding = event_embeddings.get(event_data['id'])

            # Score using existing scorer
            score = self.attachment_scorer.score_page_to_event(
                page_embedding=page.get('embedding'),
                page_entities=entity_names,
                page_time=page.get('pub_time'),
                page_claims=claims,
                event={
                    'id': event_data['id'],
                    'title': event_data.get('canonical_name', 'Untitled'),
                    'status': event_data.get('status', 'provisional'),
                    'confidence': event_data.get('confidence', 0.5),
                    'event_start': event_start,
                    'event_end': event_end,
                    'pages_count': 1,  # TODO: get from Neo4j
                    'embedding': event_embedding  # From PostgreSQL
                },
                event_entities=candidate_entity_names,
                event_claims=[]
            )

            # Apply status boost
            boosted_score = score['total_score']
            status = event_data.get('status', 'provisional')

            if status == 'stable':
                boosted_score += 0.08
            elif status == 'emerging':
                boosted_score += 0.04

            logger.info(
                f"  📊 Event '{event_data.get('canonical_name', 'Untitled')[:50]}...': "
                f"score={boosted_score:.3f} (entity_overlap={entity_overlap}, status={status})"
            )

            if boosted_score > best_score_value:
                best_score_value = boosted_score
                best_event = event_data

        # Decision: attach or create new
        if best_score_value >= 0.40:  # Threshold
            logger.info(
                f"🔗 Attaching to event '{best_event.get('canonical_name', 'Untitled')}' "
                f"(score: {best_score_value:.3f})"
            )
            await self._attach_to_event_graph(best_event['id'], page, claims)
        else:
            logger.info(
                f"📝 Creating new event (best match score: {best_score_value:.3f} below threshold)"
            )
            await self._create_event_graph(page, claims, entity_names)

    async def _get_event_entity_names(self, event_id: str) -> Set[str]:
        """Get all entity names for an event from Neo4j"""
        query = """
        MATCH (e:Event {id: $event_id})-[:HAS_PHASE]->(p:Phase)-[:SUPPORTED_BY]->(c:Claim)
        MATCH (c)-[:MENTIONS|ACTOR|SUBJECT|LOCATION]->(entity:Entity)
        RETURN collect(DISTINCT entity.canonical_name) as entity_names
        """

        async with self.neo4j.driver.session() as session:
            result = await session.run(query, {'event_id': event_id})
            record = await result.single()
            if record and record['entity_names']:
                return set(record['entity_names'])
            return set()

    async def _synthesize_canonical_title(
        self,
        page: Dict,
        claims: List[Dict],
        entity_names: Set[str],
        event_type: str
    ) -> str:
        """
        Synthesize canonical event title immediately

        Format: "YYYY Location Event Type" (e.g., "2025 Hong Kong Tai Po Fire")
        """
        if not self.openai_client:
            # Fallback to page title if no OpenAI key
            return page['title'][:100] if page.get('title') else 'Untitled Event'

        # Extract location entities
        locations = [name for name in entity_names if any(loc_word in name for loc_word in ['Hong Kong', 'Tai Po', 'China', 'Korea', 'Japan', 'Singapore'])]

        # Get event date
        event_times = [c['event_time'] for c in claims if c['event_time']]
        event_date = min(event_times) if event_times else page.get('pub_time')
        year = event_date.strftime("%Y") if event_date else "2025"

        # Prepare prompt
        entity_list = "\n".join([f"- {name}" for name in list(entity_names)[:8]])
        claims_text = "\n".join([f"- {c['text'][:80]}" for c in claims[:5]])

        prompt = f"""Create a canonical event name from this news article.

ARTICLE: {page['title']}

KEY ENTITIES:
{entity_list}

SAMPLE CLAIMS:
{claims_text}

FORMAT: "{year} [Location] [Event Type]"

REQUIREMENTS:
1. Start with {year}
2. Include specific location (district/city level)
3. End with event type: {event_type}
4. Keep under 8 words
5. Use proper nouns, be specific
6. Timeless (no "breaking", "toll rises")

GOOD EXAMPLES:
- "2025 Hong Kong Tai Po Residential Fire"
- "2024 Baltimore Francis Scott Key Bridge Collapse"
- "2025 Los Angeles Palisades Wildfire"

BAD EXAMPLES (article headlines):
- "Death toll rises as blaze engulfs high-rise"
- "Survivors asking how it was allowed to happen"

Return ONLY the canonical event name, nothing else."""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=30
            )

            title = response.choices[0].message.content.strip().strip('"').strip("'")
            logger.info(f"📝 Synthesized canonical title: '{title}'")
            return title[:100]

        except Exception as e:
            logger.error(f"❌ Title synthesis error: {e}")
            # Fallback to simple title
            return f"{year} {list(entity_names)[0] if entity_names else 'Event'} {event_type}".title()

    async def _create_event_graph(
        self,
        page: Dict,
        claims: List[Dict],
        entity_names: Set[str]
    ):
        """
        Create new event graph in Neo4j using discovery-based approach

        Steps:
        1. Create EventScaffold to discover structure (temporal + semantic)
        2. Determine event type and synthesize canonical title
        3. Decide: umbrella event or simple event
        4. Create Event node
        5. Create Phase nodes based on discovered clusters
        6. Link claims to phases
        """
        # === Step 1: Discovery ===
        scaffold = EventScaffold(claims, page)
        await scaffold._discover_with_llm(self.openai_client)

        # === Step 2: Event type and title ===
        event_type = self._infer_event_type(claims, entity_names)
        canonical_name = await self._synthesize_canonical_title(page, claims, entity_names, event_type)

        # === Step 3: Event scale (first page always provisional/micro) ===
        event_scale = 'micro'

        logger.info(
            f"🏗️  Event structure: SIMPLE event (first page, low confidence={scaffold.coherence})"
        )

        # === Step 4: Create Event node ===
        event_id = str(uuid.uuid4())

        # Compute event embedding from page embedding
        event_embedding = page.get('embedding')
        if isinstance(event_embedding, str):
            event_embedding = self._parse_embedding(event_embedding)
            if event_embedding is not None:
                event_embedding = event_embedding.tolist()

        metadata = {
            'source_page': str(page['id']),
            'source_url': page['url'],
            'coherence': scaffold.coherence,
            'first_page_pub_time': page['pub_time'].isoformat() if page.get('pub_time') else None
        }

        # Create Event node in Neo4j (graph structure only, no embedding)
        await self.neo4j.create_event(
            event_id=event_id,
            canonical_name=canonical_name,
            event_type=event_type,
            status='provisional',
            confidence=scaffold.coherence,  # Low confidence (0.3)
            event_scale=event_scale,
            earliest_time=scaffold.earliest_time,
            latest_time=scaffold.latest_time,
            metadata=metadata
        )

        # Store event in PostgreSQL with embedding
        if event_embedding:
            await self._store_event_in_postgres(
                event_id=event_id,
                title=canonical_name,
                event_type=event_type,
                event_start=scaffold.earliest_time,
                event_end=scaffold.latest_time,
                status='provisional',
                confidence=scaffold.coherence,
                event_scale=event_scale,
                embedding=event_embedding,
                metadata=metadata
            )

        # === Step 5: Create phases from discovered semantic clusters ===
        phases_created = []

        for i, cluster in enumerate(scaffold.semantic_clusters, 1):
            phase_id = str(uuid.uuid4())

            # Generate phase name from cluster keywords
            phase_name = self._generate_phase_name(cluster, event_type, i)
            phase_type = self._infer_phase_type(cluster, event_type)

            # Compute phase embedding: mean of claim embeddings in this cluster
            phase_embedding = self._compute_phase_embedding(cluster.claims)

            # Create Phase node in Neo4j (graph structure only)
            await self.neo4j.create_phase(
                phase_id=phase_id,
                event_id=event_id,
                name=phase_name,
                phase_type=phase_type,
                start_time=scaffold.earliest_time,
                end_time=scaffold.latest_time,
                sequence=i
            )

            # Store phase embedding in PostgreSQL
            if phase_embedding:
                await self._store_phase_embedding(
                    phase_id=phase_id,
                    event_id=event_id,
                    name=phase_name,
                    phase_type=phase_type,
                    sequence=i,
                    embedding=phase_embedding
                )

            phases_created.append({
                'id': phase_id,
                'name': phase_name,
                'claims': cluster.claims,
                'keywords': cluster.keywords
            })

            logger.info(f"  📍 Created phase {i}: {phase_name} ({len(cluster.claims)} claims)")

        # === Step 6: Create claim nodes and link to phases ===
        for phase_info in phases_created:
            phase_id = phase_info['id']
            phase_claims = phase_info['claims']

            for claim in phase_claims:
                claim_id = str(claim['id'])

                # Create claim node
                await self.neo4j.create_claim(
                    claim_id=claim_id,
                    text=claim['text'],
                    modality=claim.get('modality', 'observation'),
                    confidence=claim.get('confidence', 0.8),
                    event_time=claim.get('event_time'),
                    page_id=str(page['id']),
                    page_embedding=page.get('embedding')
                )

                # Link claim to phase
                await self.neo4j.link_claim_to_phase(
                    claim_id=claim_id,
                    phase_id=phase_id,
                    confidence=0.9
                )

                # Link claim to entities
                for i, entity_id in enumerate(claim['entity_ids']):
                    if i < len(claim['entity_names']) and i < len(claim['entity_types']):
                        entity_name = claim['entity_names'][i]
                        entity_type = claim['entity_types'][i]

                        rel_type = self._determine_entity_relationship(claim['text'], entity_name, entity_type)

                        await self.neo4j.link_claim_to_entity(
                            claim_id=claim_id,
                            entity_id=str(entity_id),
                            relationship_type=rel_type,
                            canonical_name=entity_name,
                            entity_type=entity_type
                        )

        logger.info(
            f"✨ Created provisional event: {canonical_name}\n"
            f"   Phase: {phases_created[0]['name']} (all {len(claims)} claims)\n"
            f"   Entities: {len(entity_names)}\n"
            f"   Confidence: {scaffold.coherence} (low - needs more pages)"
        )

        # Trigger enrichment for canonical title synthesis
        if len(claims) >= 3:
            await self.job_queue.enqueue('enrichment_queue', {
                'event_id': event_id,
                'trigger': 'new_event_created'
            })
            logger.info(f"🎨 Enqueued enrichment for event {event_id}")

    def _parse_embedding(self, emb) -> Optional[np.ndarray]:
        """Parse embedding from any format to numpy array"""
        if emb is None:
            return None

        # Already a list/tuple
        if isinstance(emb, (list, tuple)):
            return np.array(emb, dtype=np.float32)

        # Already numpy array
        if isinstance(emb, np.ndarray):
            return emb.astype(np.float32)

        # String representation
        if isinstance(emb, str):
            try:
                if emb.startswith('[') and emb.endswith(']'):
                    emb_list = [float(x.strip()) for x in emb[1:-1].split(',')]
                    return np.array(emb_list, dtype=np.float32)
            except Exception as e:
                logger.warning(f"⚠️  Failed to parse embedding: {e}")
                return None

        return None

    async def _find_best_phase_for_page(
        self,
        event_id: str,
        phases: List[Dict],
        new_page_embedding: List[float],
        new_page_pub_time: datetime
    ) -> str:
        """
        Find best matching phase for new page using phase embedding similarity

        Algorithm:
        1. Get phase embedding for each existing phase
        2. Compare new page embedding to each phase embedding
        3. Return phase with highest similarity if > 0.70
        4. Otherwise create new phase

        Returns: phase_id (existing or newly created)
        """
        # Parse new page embedding first
        new_page_vec = self._parse_embedding(new_page_embedding)
        if new_page_vec is None:
            logger.warning("⚠️  New page has no valid embedding, creating new phase")
            new_phase_id = str(uuid.uuid4())
            await self.neo4j.create_phase(
                phase_id=new_phase_id,
                event_id=event_id,
                name=f"Development Phase {len(phases) + 1}",
                phase_type="UNSPECIFIED",
                start_time=new_page_pub_time,
                end_time=new_page_pub_time,
                sequence=len(phases) + 1
            )
            return new_phase_id

        phase_similarities = []

        # Query PostgreSQL for phase embeddings
        async with self.db_pool.acquire() as conn:
            for phase in phases:
                phase_id = phase['id']

                # Get phase embedding from PostgreSQL
                row = await conn.fetchrow("""
                    SELECT embedding
                    FROM core.event_phases
                    WHERE id = $1
                """, uuid.UUID(phase_id))

                if not row or not row['embedding']:
                    # Phase has no embedding, skip
                    logger.debug(f"⚠️  Phase {phase['name']} has no embedding in PostgreSQL, skipping")
                    continue

                # Parse phase embedding
                phase_emb = self._parse_embedding(row['embedding'])
                if phase_emb is None:
                    continue

                # Cosine similarity
                similarity = float(np.dot(new_page_vec, phase_emb) / (
                    np.linalg.norm(new_page_vec) * np.linalg.norm(phase_emb)
                ))

                phase_similarities.append({
                    'phase_id': phase_id,
                    'phase_name': phase['name'],
                    'similarity': similarity
                })

                logger.info(
                    f"  📊 Phase '{phase['name']}': similarity={similarity:.3f}"
                )

        # Find best match
        if phase_similarities:
            best_match = max(phase_similarities, key=lambda x: x['similarity'])

            if best_match['similarity'] >= 0.70:
                logger.info(
                    f"  ✅ Attaching to existing phase '{best_match['phase_name']}' "
                    f"(similarity: {best_match['similarity']:.3f})"
                )
                return best_match['phase_id']

        # No good match, create new phase
        if phase_similarities:
            max_sim = max([p['similarity'] for p in phase_similarities])
            logger.info(f"  🆕 Creating new phase (max similarity: {max_sim:.3f} < 0.70)")
        else:
            logger.info(f"  🆕 Creating new phase (no existing pages to compare)")

        new_phase_id = str(uuid.uuid4())
        new_phase_name = f"Development Phase {len(phases) + 1}"
        new_phase_type = "UNSPECIFIED"

        await self.neo4j.create_phase(
            phase_id=new_phase_id,
            event_id=event_id,
            name=new_phase_name,
            phase_type=new_phase_type,
            start_time=new_page_pub_time,
            end_time=new_page_pub_time,
            sequence=len(phases) + 1
        )

        logger.info(f"  📍 Created new phase: {new_phase_name}")

        return new_phase_id

    async def _attach_to_event_graph(
        self,
        event_id: str,
        page: Dict,
        claims: List[Dict]
    ):
        """
        Attach page claims to existing event graph using page embedding similarity

        Strategy:
        1. Get all existing phases in event
        2. For each phase, compute centroid of page embeddings
        3. Compare new page embedding to each phase centroid
        4. If max_similarity > 0.70: attach to that phase
        5. If max_similarity < 0.70: create new phase (different focus!)
        """
        # Get event with phases
        event_data = await self.neo4j.get_event_with_phases(event_id)

        if not event_data or not event_data.get('phases'):
            logger.warning(f"Event {event_id} has no phases, creating Initial Reports phase")
            phase_id = str(uuid.uuid4())
            await self.neo4j.create_phase(
                phase_id=phase_id,
                event_id=event_id,
                name='Initial Reports',
                phase_type='INCIDENT',
                sequence=1
            )
        else:
            # Page-based phase matching
            phases = event_data['phases']
            page_embedding = page.get('embedding')

            if page_embedding is None:
                # No embedding, default to latest phase
                logger.warning("⚠️  Page has no embedding, defaulting to latest phase")
                phase_id = phases[-1]['id']
            else:
                # Find best matching phase using page embedding similarity
                phase_id = await self._find_best_phase_for_page(
                    event_id=event_id,
                    phases=phases,
                    new_page_embedding=page_embedding,
                    new_page_pub_time=page.get('pub_time')
                )

        # Create claim nodes and link to phase
        for claim in claims:
            claim_id = str(claim['id'])

            await self.neo4j.create_claim(
                claim_id=claim_id,
                text=claim['text'],
                modality=claim.get('modality', 'observation'),
                confidence=claim.get('confidence', 0.8),
                event_time=claim.get('event_time'),
                page_id=str(page['id']),
                page_embedding=page.get('embedding')
            )

            # Link claim to phase
            await self.neo4j.link_claim_to_phase(
                claim_id=claim_id,
                phase_id=phase_id,
                confidence=0.9
            )

            # Link claim to entities
            for i, entity_id in enumerate(claim['entity_ids']):
                if i < len(claim['entity_names']) and i < len(claim['entity_types']):
                    entity_name = claim['entity_names'][i]
                    entity_type = claim['entity_types'][i]

                    rel_type = self._determine_entity_relationship(claim['text'], entity_name, entity_type)

                    await self.neo4j.link_claim_to_entity(
                        claim_id=claim_id,
                        entity_id=str(entity_id),
                        relationship_type=rel_type,
                        canonical_name=entity_name,
                        entity_type=entity_type
                    )

        # Update event status in Neo4j
        await self._update_event_status(event_id)

        # Update event in PostgreSQL (embedding, counts, status)
        await self._update_event_in_postgres(event_id)

        logger.info(f"📎 Attached {len(claims)} claims to event {event_id}")

        # Trigger enrichment for multi-source events
        await self.job_queue.enqueue('enrichment_queue', {
            'event_id': event_id,
            'trigger': 'claims_added',
            'new_claims_count': len(claims)
        })

    async def _update_event_status(self, event_id: str):
        """
        Update event status based on accumulated evidence

        - Count unique pages via FROM_PAGE relationships
        - Promote provisional → emerging → stable
        """
        query = """
        MATCH (e:Event {id: $event_id})-[:HAS_PHASE]->(p:Phase)-[:SUPPORTED_BY]->(c:Claim)-[:FROM_PAGE]->(page:Page)
        WITH e, count(DISTINCT page) as page_count, count(DISTINCT c) as claim_count
        SET e.updated_at = datetime(),
            e.status = CASE
                WHEN e.status = 'provisional' AND page_count >= 2 THEN 'emerging'
                WHEN e.status = 'emerging' AND page_count >= 5 THEN 'stable'
                ELSE e.status
            END,
            e.confidence = CASE
                WHEN page_count >= 5 THEN 0.95
                WHEN page_count >= 2 THEN 0.75
                ELSE 0.5
            END
        RETURN e.status as status, page_count, claim_count
        """

        async with self.neo4j.driver.session() as session:
            result = await session.run(query, {'event_id': event_id})
            record = await result.single()
            if record:
                logger.info(
                    f"📊 Updated event {event_id}: "
                    f"status={record['status']}, pages={record['page_count']}, claims={record['claim_count']}"
                )

    async def _update_event_in_postgres(self, event_id: str):
        """
        Update event in PostgreSQL after attaching pages

        Updates:
        - Event embedding (mean of all page embeddings)
        - Pages count and claims count
        - Status and confidence based on page count
        """
        # Get all page IDs linked to this event from Neo4j
        query = """
        MATCH (e:Event {id: $event_id})-[:HAS_PHASE]->(p:Phase)-[:SUPPORTED_BY]->(c:Claim)-[:FROM_PAGE]->(page:Page)
        RETURN DISTINCT page.id as page_id
        """

        async with self.neo4j.driver.session() as session:
            result = await session.run(query, {'event_id': event_id})
            records = await result.data()
            page_ids = [r['page_id'] for r in records]

        if not page_ids:
            logger.warning(f"⚠️  No pages found for event {event_id}, cannot update PostgreSQL")
            return

        # Query PostgreSQL for page embeddings
        async with self.db_pool.acquire() as conn:
            page_rows = await conn.fetch("""
                SELECT id, embedding FROM core.pages WHERE id = ANY($1::uuid[])
            """, page_ids)

            # Compute mean embedding from all page embeddings
            page_embeddings = []
            for row in page_rows:
                emb = self._parse_embedding(row['embedding'])
                if emb is not None:
                    page_embeddings.append(emb)

            if not page_embeddings:
                logger.warning(f"⚠️  No page embeddings found for event {event_id}")
                return

            mean_embedding = np.mean(np.array(page_embeddings), axis=0).tolist()
            embedding_str = '[' + ','.join(str(x) for x in mean_embedding) + ']'

            # Count claims for this event
            claim_count = await conn.fetchval("""
                SELECT COUNT(DISTINCT c.id)
                FROM core.claims c
                WHERE c.page_id = ANY($1::uuid[])
            """, page_ids)

            # Determine status and confidence based on page count
            page_count = len(page_ids)
            if page_count >= 5:
                status = 'stable'
                confidence = 0.95
            elif page_count >= 2:
                status = 'emerging'
                confidence = 0.75
            else:
                status = 'provisional'
                confidence = 0.5

            # Update event in PostgreSQL
            await conn.execute("""
                UPDATE core.events
                SET
                    embedding = $1::vector,
                    pages_count = $2,
                    claims_count = $3,
                    status = $4,
                    confidence = $5,
                    updated_at = NOW()
                WHERE id = $6
            """, embedding_str, page_count, claim_count, status, confidence, uuid.UUID(event_id))

            logger.debug(
                f"💾 Updated event in PostgreSQL: {event_id} "
                f"(pages={page_count}, claims={claim_count}, status={status})"
            )

    def _compute_phase_embedding(self, claims: List[Dict]) -> Optional[List[float]]:
        """
        Compute phase embedding as mean of claim embeddings

        Returns None if no claim embeddings available
        """
        claim_embeddings = []

        for claim in claims:
            emb = claim.get('embedding')
            parsed = self._parse_embedding(emb)
            if parsed is not None:
                claim_embeddings.append(parsed)

        if not claim_embeddings:
            logger.debug(f"⚠️  No claim embeddings available for phase (claims without embeddings)")
            return None

        # Compute mean
        mean_embedding = np.mean(np.array(claim_embeddings), axis=0)
        return mean_embedding.tolist()

    async def _store_event_in_postgres(
        self,
        event_id: str,
        title: str,
        event_type: str,
        event_start: datetime,
        event_end: datetime,
        status: str,
        confidence: float,
        event_scale: str,
        embedding: List[float],
        metadata: Dict = None
    ):
        """
        Store event in PostgreSQL core.events table with embedding

        Architecture: PostgreSQL stores content + embeddings, Neo4j stores graph structure
        """
        # Convert embedding to pgvector format
        embedding_str = '[' + ','.join(str(x) for x in embedding) + ']' if embedding else None

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO core.events (
                    id, title, event_type, event_start, event_end,
                    status, confidence, event_scale, embedding, metadata,
                    pages_count, claims_count
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::vector, $10, 0, 0)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    event_type = EXCLUDED.event_type,
                    event_start = EXCLUDED.event_start,
                    event_end = EXCLUDED.event_end,
                    status = EXCLUDED.status,
                    confidence = EXCLUDED.confidence,
                    event_scale = EXCLUDED.event_scale,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
            """,
                uuid.UUID(event_id),
                title,
                event_type,
                event_start,
                event_end,
                status,
                confidence,
                event_scale,
                embedding_str,
                json.dumps(metadata) if metadata else '{}'
            )

    async def _store_phase_embedding(
        self,
        phase_id: str,
        event_id: str,
        name: str,
        phase_type: str,
        sequence: int,
        embedding: List[float]
    ):
        """
        Store phase embedding in PostgreSQL core.event_phases table
        """
        # Convert embedding to pgvector format
        embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO core.event_phases (id, event_id, name, phase_type, sequence, embedding)
                VALUES ($1, $2, $3, $4, $5, $6::vector)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    phase_type = EXCLUDED.phase_type,
                    embedding = EXCLUDED.embedding,
                    updated_at = NOW()
            """, uuid.UUID(phase_id), uuid.UUID(event_id), name, phase_type, sequence, embedding_str)

        logger.debug(f"💾 Stored phase embedding in PostgreSQL: {name}")

    def _infer_event_type(self, claims: List[Dict], entity_names: Set[str]) -> str:
        """
        Infer event type from claims and entities

        Simple keyword-based heuristic for now
        """
        text = ' '.join([c['text'].lower() for c in claims])

        if 'fire' in text or 'blaze' in text or 'burn' in text:
            return 'FIRE'
        elif 'earthquake' in text or 'quake' in text:
            return 'EARTHQUAKE'
        elif 'shoot' in text or 'gunman' in text or 'gunfire' in text:
            return 'SHOOTING'
        elif 'explo' in text or 'blast' in text:
            return 'EXPLOSION'
        elif 'flood' in text or 'storm' in text or 'hurricane' in text:
            return 'NATURAL_DISASTER'
        elif 'crash' in text or 'accident' in text or 'collision' in text:
            return 'ACCIDENT'
        else:
            return 'UNSPECIFIED'

    def _generate_phase_name(self, cluster: SemanticCluster, event_type: str, sequence: int) -> str:
        """
        Generate human-readable phase name from semantic cluster

        When using LLM-based discovery, keywords already contain the phase name
        """
        # LLM-provided phase names are in keywords (e.g., ['fire', 'breakout'] -> "Fire Breakout")
        if cluster.keywords:
            return ' '.join(word.capitalize() for word in cluster.keywords)
        else:
            return f"Phase {sequence}"

    def _infer_phase_type(self, cluster: SemanticCluster, event_type: str) -> str:
        """
        Infer phase type from cluster keywords

        Returns: INCIDENT, RESPONSE, CONSEQUENCE, INVESTIGATION, or AFTERMATH
        """
        keywords = cluster.keywords
        text = ' '.join(keywords)

        if any(kw in text for kw in ['broke', 'started', 'occurred', 'began', 'fire', 'blaze']):
            return "INCIDENT"
        elif any(kw in text for kw in ['rescue', 'firefighters', 'response', 'evacuate', 'battle']):
            return "RESPONSE"
        elif any(kw in text for kw in ['dead', 'killed', 'casualties', 'toll', 'injured', 'missing']):
            return "CONSEQUENCE"
        elif any(kw in text for kw in ['arrested', 'investigation', 'probe', 'charged', 'police']):
            return "INVESTIGATION"
        else:
            return "INCIDENT"

    def _determine_entity_relationship(self, claim_text: str, entity_name: str, entity_type: str) -> str:
        """
        Determine relationship type based on entity position in claim

        Heuristic:
        - Locations → LOCATION
        - First PERSON/ORG before verb → ACTOR
        - PERSON/ORG after "killed", "injured" → SUBJECT
        - Default → MENTIONS
        """
        if entity_type in ('GPE', 'LOC', 'FAC'):
            return 'LOCATION'

        text_lower = claim_text.lower()
        entity_lower = entity_name.lower()

        # Check if entity appears after victim keywords
        if any(keyword in text_lower for keyword in ['killed', 'injured', 'dead', 'wounded', 'victim']):
            entity_pos = text_lower.find(entity_lower)
            for keyword in ['killed', 'injured', 'dead', 'wounded', 'victim']:
                keyword_pos = text_lower.find(keyword)
                if keyword_pos > 0 and entity_pos > keyword_pos:
                    return 'SUBJECT'

        # Check if entity is likely an actor (appears early, before verbs)
        words = text_lower.split()
        if entity_lower in ' '.join(words[:5]):  # First 5 words
            return 'ACTOR'

        return 'MENTIONS'


async def main():
    """Main worker entry point"""
    worker_id = int(os.getenv('WORKER_ID', '1'))

    # PostgreSQL connection
    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        user=os.getenv('POSTGRES_USER', 'herenews_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
        database=os.getenv('POSTGRES_DB', 'herenews'),
        min_size=2,
        max_size=5
    )

    # Neo4j connection
    neo4j_service = Neo4jService()
    await neo4j_service.connect()

    # Initialize constraints/indexes
    await neo4j_service.initialize_constraints()

    # Job queue
    job_queue = JobQueue(os.getenv('REDIS_URL', 'redis://redis:6379'))
    await job_queue.connect()

    worker = EventWorkerNeo4j(db_pool, neo4j_service, job_queue, worker_id=worker_id)
    logger.info(f"📊 Starting Event worker (Neo4j) {worker_id}")

    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    finally:
        await db_pool.close()
        await neo4j_service.close()
        await job_queue.close()


if __name__ == "__main__":
    asyncio.run(main())
