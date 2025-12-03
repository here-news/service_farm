"""
Event Worker - Multi-Pass Clustering for Event Formation

Implements recursive multi-pass clustering to form events from claims.

Algorithm (TODO.md Phase 4):
- Pass 1: Tight temporal clusters (2-day window + 2+ entity overlap)
- Pass 2: Bridge temporal gaps (14-day relaxed + 4+ entity overlap)
- Pass 3: Transitive merging (graph DFS for connected components)

Design principles:
- Agnostic: Let data speak through entity/temporal signals
- Incremental: New claims trigger re-clustering on affected components
- Conservative: Require minimum entity overlap to avoid spurious merges
"""
import asyncio
import json
import logging
import os
import re
import uuid
from typing import Dict, List, Set, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

import asyncpg

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.job_queue import JobQueue
from workers.event_consolidation import (
    compute_log_prior, compute_log_likelihood,
    consolidate_claims, detect_event_phases
)
from workers.event_attachment import EventAttachmentScorer, find_best_event_match

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_event_title(title: str) -> str:
    """
    Clean page title to create better event title

    Removes:
    - Source indicators: "– DW –", "| CNN", "- BBC"
    - Date patterns: "– 11/26/2025", "| Jan 15, 2024"
    - Trailing punctuation
    """
    if not title:
        return title

    # Remove common source indicators
    title = re.sub(r'\s*[–|-]\s*(DW|CNN|BBC|Reuters|AP|NYT|NYPost|Bloomberg|Guardian)\s*[–|-]\s*', ' ', title, flags=re.IGNORECASE)
    title = re.sub(r'\s*\|\s*(DW|CNN|BBC|Reuters|AP|NYT|NYPost|Bloomberg|Guardian)\s*', ' ', title, flags=re.IGNORECASE)

    # Remove date patterns like "– 11/26/2025" or "| Jan 15, 2024"
    title = re.sub(r'\s*[–|-]\s*\d{1,2}/\d{1,2}/\d{4}\s*', ' ', title)
    title = re.sub(r'\s*[–|-]\s*\w{3,9}\s+\d{1,2},?\s+\d{4}\s*', ' ', title)
    title = re.sub(r'\s*\|\s*\d{1,2}/\d{1,2}/\d{4}\s*', ' ', title)

    # Remove trailing numbers (partial dates like "– 11")
    title = re.sub(r'\s*[–|-]\s*\d{1,2}\s*$', '', title)

    # Remove trailing separators and whitespace
    title = re.sub(r'\s*[–|-]\s*$', '', title)
    title = re.sub(r'\s*\|\s*$', '', title)
    title = title.strip()

    return title


# Clustering parameters (from TODO.md)
PASS1_TIME_WINDOW_DAYS = 2  # Tight temporal window
PASS1_MIN_ENTITY_OVERLAP = 2  # Minimum shared entities

PASS2_TIME_WINDOW_DAYS = 14  # Relaxed temporal window
PASS2_MIN_ENTITY_OVERLAP = 4  # Higher entity requirement


class EventWorker:
    """
    Worker that forms events from claims using multi-pass clustering
    """

    def __init__(self, db_pool: asyncpg.Pool, job_queue: JobQueue, worker_id: int = 1):
        self.db_pool = db_pool
        self.job_queue = job_queue
        self.worker_id = worker_id
        self.attachment_scorer = EventAttachmentScorer()

    async def start(self):
        """Start worker loop"""
        logger.info(f"📊 event-worker-{self.worker_id} started (incremental event formation)")

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
        Process a page and find/create event

        1. Fetch page embedding and metadata
        2. Search for similar events (using page embedding as proxy)
        3. If found (>= 0.7 similarity): attach to existing event
        4. If not found: create provisional event
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

            # Fetch claims with entities
            claims = await conn.fetch("""
                SELECT
                    c.id, c.text, c.event_time, c.confidence, c.metadata,
                    array_agg(ce.entity_id) FILTER (WHERE ce.entity_id IS NOT NULL) as entity_ids
                FROM core.claims c
                LEFT JOIN core.claim_entities ce ON c.id = ce.claim_id
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
                'metadata': c['metadata'],
                'entity_ids': set(c['entity_ids']) if c['entity_ids'] else set()
            } for c in claims]

            logger.info(f"📄 Processing page {page['url']} ({len(claims_data)} claims)")

            # Find or create event
            if page['embedding']:
                await self._find_or_create_event(conn, page, claims_data)
            else:
                logger.warning(f"⚠️  No embedding for page {page_id}, creating provisional event")
                await self._create_provisional_event(conn, page, claims_data)

    async def _find_or_create_event(self, conn: asyncpg.Connection, page: Dict, claims: List[Dict]):
        """
        Find best matching event using multi-signal scoring or create new one

        Uses:
        - Semantic similarity (embedding)
        - Entity overlap
        - Temporal proximity
        - Topic coherence

        Decides: attach, spawn, or create relationship
        """
        # Get candidate events (top-K by embedding, then multi-signal score)
        candidates = await conn.fetch("""
            SELECT
                e.id, e.title, e.status, e.confidence, e.event_scale,
                e.event_start, e.event_end, e.embedding,
                1 - (e.embedding <=> $1::vector) as emb_similarity,
                ARRAY_AGG(DISTINCT en.canonical_name) FILTER (WHERE en.canonical_name IS NOT NULL) as entity_names
            FROM core.events e
            LEFT JOIN core.event_entities ee ON e.id = ee.event_id
            LEFT JOIN core.entities en ON ee.entity_id = en.id
            WHERE e.embedding IS NOT NULL
                AND e.status IN ('provisional', 'emerging', 'stable')
            GROUP BY e.id, e.title, e.status, e.confidence, e.event_scale,
                     e.event_start, e.event_end, e.embedding
            ORDER BY e.embedding <=> $1::vector
            LIMIT 10
        """, page['embedding'])

        if not candidates:
            logger.info("📝 No candidate events found, creating provisional event")
            await self._create_provisional_event(conn, page, claims)
            return

        # Get page entities
        page_entities = set()
        for claim in claims:
            page_entities.update(claim.get('entity_ids', []))

        # Convert page_entities UUIDs to names for comparison
        if page_entities:
            entity_rows = await conn.fetch("""
                SELECT canonical_name FROM core.entities WHERE id = ANY($1::uuid[])
            """, list(page_entities))
            page_entity_names = {row['canonical_name'] for row in entity_rows}
        else:
            page_entity_names = set()

        # Score each candidate
        best_event = None
        best_score = None
        best_score_value = 0.0

        for candidate in candidates:
            candidate_dict = dict(candidate)
            candidate_entity_names = set(candidate_dict.get('entity_names') or [])

            # Score this candidate
            score = self.attachment_scorer.score_page_to_event(
                page_embedding=page.get('embedding'),
                page_entities=page_entity_names,
                page_time=page.get('pub_time'),
                page_claims=claims,
                event=candidate_dict,
                event_entities=candidate_entity_names,
                event_claims=[]  # TODO: fetch event claims if needed
            )

            # Boost score for stable/large events (prefer consolidation into main events)
            boosted_score = score['total_score']
            status = candidate_dict.get('status', 'provisional')
            pages_count = candidate_dict.get('pages_count', 1) or 1

            if status == 'stable':
                boosted_score += 0.08  # Significant boost for stable events
            elif status == 'emerging':
                boosted_score += 0.04  # Moderate boost for emerging

            # Additional boost for events with multiple pages (log scale)
            import math
            if pages_count > 1:
                boosted_score += min(0.05, math.log10(pages_count) * 0.03)

            logger.info(
                f"  📊 Event '{candidate_dict['title'][:50]}...': "
                f"raw={score['total_score']:.3f} boosted={boosted_score:.3f} "
                f"(status={status}, pages={pages_count}) {score['rationale']}"
            )

            if boosted_score > best_score_value:
                best_score_value = boosted_score
                best_event = candidate_dict
                best_score = score  # Keep original score for decision logic

        # Decision based on best score
        if not best_score:
            logger.info("📝 No suitable matches, creating provisional event")
            await self._create_provisional_event(conn, page, claims)
            return

        decision = best_score['decision']
        relationship_type = best_score['relationship_type']

        if decision == 'attach':
            logger.info(
                f"🔗 Attaching to event '{best_event['title']}' "
                f"(score: {best_score_value:.3f}, {best_score['rationale']})"
            )
            await self._attach_to_event(conn, best_event['id'], page, claims)

        elif decision == 'relate':
            logger.info(
                f"🔗 Creating {relationship_type} relationship with '{best_event['title']}' "
                f"(score: {best_score_value:.3f}, {best_score['rationale']})"
            )
            # Create new event and relationship
            new_event_id = await self._create_provisional_event(conn, page, claims, return_id=True)
            await self._create_relationship(conn, new_event_id, best_event['id'], relationship_type, best_score['confidence'])

        else:  # spawn
            logger.info(
                f"📝 Creating new event (best match: '{best_event['title']}', "
                f"score: {best_score_value:.3f} below threshold)"
            )
            await self._create_provisional_event(conn, page, claims)

    async def _create_provisional_event(self, conn: asyncpg.Connection, page: Dict, claims: List[Dict], return_id: bool = False):
        """Create new provisional event from page"""
        # Extract temporal bounds
        event_times = [c['event_time'] for c in claims if c['event_time']]
        if event_times:
            event_start = min(event_times)
            event_end = max(event_times)
        else:
            event_start = event_end = page['pub_time']

        # Generate title from page title (better than claim fragments)
        raw_title = page['title'] if page.get('title') else (claims[0]['text'][:80] if claims else 'Untitled Event')
        title = clean_event_title(raw_title)[:100]
        summary = f"Provisional event created from {page['url']}"

        # Collect all entities
        all_entities = set()
        for c in claims:
            all_entities.update(c['entity_ids'])

        # Calculate scale and confidence based on reporting coverage
        # More claims and entities = wider coverage = higher confidence
        claims_count = len(claims)
        entities_count = len(all_entities)
        page_credibility = page['metadata_confidence'] or 0.5

        # Scale determination
        if claims_count >= 8 or entities_count >= 8:
            scale_type = 'regional'
            scale_rationale = f'{claims_count} claims, {entities_count} entities - significant coverage'
        elif claims_count >= 5 or entities_count >= 5:
            scale_type = 'local'
            scale_rationale = f'{claims_count} claims, {entities_count} entities - moderate coverage'
        else:
            scale_type = 'micro'
            scale_rationale = f'{claims_count} claims, {entities_count} entities - limited coverage'

        # Confidence calculation (0.5-0.8 range for provisional)
        # Base: 0.5, +0.05 per claim (max 0.15), +0.05 per entity (max 0.1), +0.05 for high credibility
        base_confidence = 0.5
        claim_boost = min(claims_count * 0.05, 0.15)
        entity_boost = min(entities_count * 0.05, 0.1)
        credibility_boost = 0.05 if page_credibility >= 0.7 else 0
        provisional_confidence = min(base_confidence + claim_boost + entity_boost + credibility_boost, 0.8)

        # Create enriched_json
        enriched_json = {
            'temporal': {
                'start': event_start.isoformat() if event_start else None,
                'end': event_end.isoformat() if event_end else None,
                'precision': 'approximate'
            },
            'scale': {'type': scale_type, 'rationale': scale_rationale},
            'participants': {
                'entity_ids': [str(eid) for eid in all_entities],
                'count': len(all_entities)
            },
            'evidence': {
                'claims_count': len(claims),
                'pages_count': 1,
                'page_ids': [str(page['id'])]
            },
            'quality': {
                'confidence': provisional_confidence,
                'credibility': page_credibility,
                'completeness': 0.3
            },
            'provenance': {
                'method': 'provisional_from_page',
                'created_from_page': str(page['id']),
                'status_history': ['provisional']
            }
        }

        # Convert embedding to pgvector format if present
        embedding_str = None
        if page['embedding']:
            if isinstance(page['embedding'], list):
                embedding_str = '[' + ','.join(str(x) for x in page['embedding']) + ']'
            else:
                embedding_str = str(page['embedding'])

        # Insert event
        event_id = await conn.fetchval("""
            INSERT INTO core.events (
                title, summary, event_start, event_end,
                confidence, status, event_scale, claims_count, pages_count,
                embedding, enriched_json
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::vector, $11)
            RETURNING id
        """,
            title, summary[:500], event_start, event_end,
            0.5, 'provisional', 'micro', len(claims), 1,
            embedding_str, json.dumps(enriched_json)
        )

        # Link entities
        for entity_id in all_entities:
            await conn.execute("""
                INSERT INTO core.event_entities (event_id, entity_id)
                VALUES ($1, $2)
                ON CONFLICT DO NOTHING
            """, event_id, entity_id)

        # Link page
        await conn.execute("""
            INSERT INTO core.page_events (page_id, event_id)
            VALUES ($1, $2)
            ON CONFLICT DO NOTHING
        """, page['id'], event_id)

        logger.info(f"✨ Created provisional event {event_id}: \"{title}\" ({len(claims)} claims, {len(all_entities)} entities)")

        # Trigger enrichment if event has sufficient claims
        if len(claims) >= 3:
            await self.job_queue.enqueue('enrichment_queue', {
                'event_id': str(event_id),
                'trigger': 'provisional_created'
            })
            logger.info(f"🎨 Enqueued enrichment for event {event_id}")

        if return_id:
            return event_id

    async def _create_relationship(
        self,
        conn: asyncpg.Connection,
        from_event_id: uuid.UUID,
        to_event_id: uuid.UUID,
        relationship_type: str,
        confidence: float
    ):
        """Create relationship between events"""
        await conn.execute("""
            INSERT INTO core.event_relationships (
                event_id, related_event_id, relationship_type, confidence, metadata
            )
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (event_id, related_event_id, relationship_type) DO UPDATE
            SET confidence = GREATEST(core.event_relationships.confidence, EXCLUDED.confidence),
                updated_at = NOW()
        """,
            from_event_id,
            to_event_id,
            relationship_type,
            confidence,
            json.dumps({'created_by': 'multi_signal_scorer'})
        )

        logger.info(
            f"🔗 Created {relationship_type} relationship: "
            f"{from_event_id} → {to_event_id} (confidence: {confidence:.3f})"
        )

    async def _attach_to_event(self, conn: asyncpg.Connection, event_id: uuid.UUID, page: Dict, claims: List[Dict]):
        """Attach page to existing event"""
        # Collect all entities
        all_entities = set()
        for c in claims:
            all_entities.update(c['entity_ids'])

        # Link entities
        for entity_id in all_entities:
            await conn.execute("""
                INSERT INTO core.event_entities (event_id, entity_id)
                VALUES ($1, $2)
                ON CONFLICT DO NOTHING
            """, event_id, entity_id)

        # Link page
        await conn.execute("""
            INSERT INTO core.page_events (page_id, event_id)
            VALUES ($1, $2)
            ON CONFLICT DO NOTHING
        """, page['id'], event_id)

        # Get updated counts
        event_stats = await conn.fetchrow("""
            SELECT
                claims_count + $2 as total_claims,
                (SELECT COUNT(DISTINCT page_id) FROM core.page_events WHERE event_id = $1) as total_pages,
                (SELECT COUNT(DISTINCT entity_id) FROM core.event_entities WHERE event_id = $1) as total_entities,
                enriched_json
            FROM core.events
            WHERE id = $1
        """, event_id, len(claims))

        # Recalculate scale based on updated counts
        total_claims = event_stats['total_claims']
        total_entities = event_stats['total_entities']

        if total_claims >= 8 or total_entities >= 8:
            scale_type = 'regional'
            scale_rationale = f'{total_claims} claims, {total_entities} entities, {event_stats["total_pages"]} sources - significant coverage'
        elif total_claims >= 5 or total_entities >= 5:
            scale_type = 'local'
            scale_rationale = f'{total_claims} claims, {total_entities} entities, {event_stats["total_pages"]} sources - moderate coverage'
        else:
            scale_type = 'micro'
            scale_rationale = f'{total_claims} claims, {total_entities} entities, {event_stats["total_pages"]} sources - limited coverage'

        # Update enriched_json scale
        # Parse from JSON string to dict if needed
        enriched_json = event_stats['enriched_json']
        if isinstance(enriched_json, str):
            enriched_json = json.loads(enriched_json)
        elif enriched_json is None:
            enriched_json = {}
        enriched_json['scale'] = {'type': scale_type, 'rationale': scale_rationale}

        # Update event and capture new status
        new_status = await conn.fetchval("""
            UPDATE core.events
            SET claims_count = claims_count + $2,
                pages_count = (SELECT COUNT(DISTINCT page_id) FROM core.page_events WHERE event_id = $1),
                confidence = LEAST(confidence + 0.1, 0.95),
                status = CASE
                    WHEN status = 'provisional' AND (SELECT COUNT(DISTINCT page_id) FROM core.page_events WHERE event_id = $1) >= 2 THEN 'emerging'
                    WHEN status = 'emerging' AND (SELECT COUNT(DISTINCT page_id) FROM core.page_events WHERE event_id = $1) >= 5 THEN 'stable'
                    ELSE status
                END,
                enriched_json = $3,
                updated_at = NOW()
            WHERE id = $1
            RETURNING status
        """, event_id, len(claims), json.dumps(enriched_json))

        # Consolidate all claims for this event
        await self._consolidate_event(conn, event_id)

        logger.info(f"📎 Attached {len(claims)} claims to event {event_id} (scale: {scale_type}, status: {new_status})")

        # Trigger enrichment when event becomes multi-source or reaches stable status
        if new_status in ('emerging', 'stable') and event_stats['total_pages'] >= 2:
            await self.job_queue.enqueue('enrichment_queue', {
                'event_id': str(event_id),
                'trigger': f'status_transition_{new_status}',
                'pages_count': event_stats['total_pages'],
                'claims_count': total_claims
            })
            logger.info(f"🎨 Enqueued enrichment for event {event_id} (status: {new_status})")

    async def _consolidate_event(self, conn: asyncpg.Connection, event_id: uuid.UUID):
        """
        Consolidate all claims for an event into structured summary

        Updates enriched_json with:
        - Timeline of key events
        - Casualty tracking (death tolls, injuries with ranges and contradictions)
        - Key participants and locations
        - Bayesian scores (priors, posteriors, coherence)
        """
        # Fetch ALL claims for this event
        all_claims = await conn.fetch("""
            SELECT c.id, c.text, c.event_time, c.confidence, c.modality,
                   p.url as page_url, p.metadata_confidence as page_credibility,
                   ARRAY_AGG(DISTINCT ce.entity_id) as entity_ids
            FROM core.claims c
            JOIN core.pages p ON c.page_id = p.id
            JOIN core.page_events pe ON p.id = pe.page_id
            LEFT JOIN core.claim_entities ce ON c.id = ce.claim_id
            WHERE pe.event_id = $1
            GROUP BY c.id, c.text, c.event_time, c.confidence, c.modality, p.url, p.metadata_confidence
            ORDER BY c.event_time NULLS LAST
        """, event_id)

        # Fetch all entities for this event
        all_entities = await conn.fetch("""
            SELECT e.id, e.canonical_name as name, e.entity_type, e.metadata
            FROM core.entities e
            JOIN core.event_entities ee ON e.id = ee.entity_id
            WHERE ee.event_id = $1
        """, event_id)

        # Convert to dicts
        claims_list = [dict(c) for c in all_claims]
        entities_list = [dict(e) for e in all_entities]

        # Consolidate
        consolidated = consolidate_claims(claims_list, entities_list)

        # Detect phases
        phases = detect_event_phases(consolidated['timeline'])
        consolidated['phases'] = [
            {
                'start': p['start'].isoformat() if p['start'] else None,
                'end': p['end'].isoformat() if p['end'] else None,
                'entry_count': len(p['entries'])
            }
            for p in phases
        ]

        # Get current event data
        event_data = await conn.fetchrow("""
            SELECT event_start, event_end, confidence, enriched_json,
                   claims_count, pages_count
            FROM core.events
            WHERE id = $1
        """, event_id)

        # Compute Bayesian scores
        event_stats = {
            'confidence': event_data['confidence'],
            'event_start': event_data['event_start'],
            'event_end': event_data['event_end'],
            'entity_count': len(all_entities),
            'scale_type': 'regional'  # Get from enriched_json
        }

        log_prior = compute_log_prior(event_stats)

        # Sum log likelihoods for all claims
        log_likelihood_sum = 0.0
        event_entity_ids = {str(e['id']) for e in all_entities}
        event_dict = {
            'event_start': event_data['event_start'],
            'event_end': event_data['event_end']
        }

        for claim in claims_list:
            claim_entity_ids = {str(eid) for eid in (claim.get('entity_ids') or []) if eid}
            shared_entities = len(event_entity_ids & claim_entity_ids)
            total_claim_entities = len(claim_entity_ids)

            log_lik = compute_log_likelihood(claim, event_dict, shared_entities, total_claim_entities)
            log_likelihood_sum += log_lik

        log_posterior = log_prior + log_likelihood_sum

        # Compute coherence (1.0 = perfect, lower with contradictions)
        coherence = 1.0
        if consolidated['contradictions']:
            for contradiction in consolidated['contradictions']:
                if contradiction['severity'] == 'high':
                    coherence *= 0.8
                elif contradiction['severity'] == 'moderate':
                    coherence *= 0.9

        # Parse enriched_json
        enriched_json = event_data['enriched_json']
        if isinstance(enriched_json, str):
            enriched_json = json.loads(enriched_json)
        elif enriched_json is None:
            enriched_json = {}

        # Update enriched_json with consolidation
        enriched_json['consolidated'] = {
            'casualty_summary': {
                'deaths': consolidated['casualties']['deaths'],
                'injured': consolidated['casualties']['injured'],
                'hospitalized': consolidated['casualties']['hospitalized']
            },
            'timeline_entries': len(consolidated['timeline']),
            'phases': consolidated['phases'],
            'key_locations': list(consolidated['locations']),
            'participants': consolidated['participants'],
            'contradictions': consolidated['contradictions']
        }

        # Update event with Bayesian scores and consolidation
        await conn.execute("""
            UPDATE core.events
            SET enriched_json = $2,
                log_prior = $3,
                log_likelihood = $4,
                log_posterior = $5,
                coherence = $6,
                updated_at = NOW()
            WHERE id = $1
        """, event_id, json.dumps(enriched_json), log_prior, log_likelihood_sum, log_posterior, coherence)

        logger.info(
            f"📊 Consolidated event {event_id}: "
            f"{len(consolidated['timeline'])} timeline entries, "
            f"{len(consolidated['contradictions'])} contradictions, "
            f"coherence={coherence:.3f}, log_posterior={log_posterior:.2f}"
        )

    async def run_clustering_cycle(self):
        """
        Run a full 3-pass clustering cycle on all claims

        This is a simple initial implementation that re-clusters everything.
        Later: make incremental (only re-cluster affected components)
        """
        async with self.db_pool.acquire() as conn:
            # Fetch all claims with their entities
            claims = await self._fetch_claims_with_entities(conn)

            if not claims:
                logger.debug("No claims to cluster")
                return

            logger.info(f"🔍 Clustering {len(claims)} claims...")

            # Pass 1: Tight temporal clustering
            clusters_pass1 = await self._cluster_pass1(claims)
            logger.info(f"✅ Pass 1: {len(clusters_pass1)} tight clusters")

            # Pass 2: Bridge temporal gaps
            clusters_pass2 = await self._cluster_pass2(clusters_pass1, claims)
            logger.info(f"✅ Pass 2: {len(clusters_pass2)} bridged clusters")

            # Pass 3: Transitive merging
            final_clusters = await self._cluster_pass3(clusters_pass2)
            logger.info(f"✅ Pass 3: {len(final_clusters)} final events")

            # Materialize events
            await self._materialize_events(conn, final_clusters, claims)

    async def _fetch_claims_with_entities(self, conn: asyncpg.Connection) -> List[Dict]:
        """
        Fetch all claims with their associated entities

        Returns list of dicts with claim metadata + entity list
        """
        rows = await conn.fetch("""
            SELECT
                c.id,
                c.page_id,
                c.text,
                c.event_time,
                c.confidence,
                c.created_at,
                COALESCE(
                    array_agg(ce.entity_id) FILTER (WHERE ce.entity_id IS NOT NULL),
                    ARRAY[]::uuid[]
                ) as entity_ids
            FROM core.claims c
            LEFT JOIN core.claim_entities ce ON c.id = ce.claim_id
            WHERE c.event_time IS NOT NULL  -- Only cluster claims with temporal info
            GROUP BY c.id
            ORDER BY c.event_time
        """)

        claims = []
        for row in rows:
            claims.append({
                'id': row['id'],
                'page_id': row['page_id'],
                'text': row['text'],
                'event_time': row['event_time'],
                'confidence': row['confidence'],
                'created_at': row['created_at'],
                'entity_ids': set(row['entity_ids'])  # Convert to set for fast overlap
            })

        return claims

    async def _cluster_pass1(self, claims: List[Dict]) -> List[Set[uuid.UUID]]:
        """
        Pass 1: Tight temporal clustering (2-day window + 2+ entity overlap)

        Returns list of claim ID sets (each set is a cluster)
        """
        clusters = []

        # Sort claims by event_time
        sorted_claims = sorted(claims, key=lambda c: c['event_time'])

        # Sliding window approach
        for i, claim in enumerate(sorted_claims):
            cluster = {claim['id']}
            claim_entities = claim['entity_ids']

            # Look ahead within time window
            for j in range(i + 1, len(sorted_claims)):
                other_claim = sorted_claims[j]

                # Check time window
                time_diff = (other_claim['event_time'] - claim['event_time']).days
                if time_diff > PASS1_TIME_WINDOW_DAYS:
                    break  # Beyond window

                # Check entity overlap
                shared_entities = claim_entities & other_claim['entity_ids']
                if len(shared_entities) >= PASS1_MIN_ENTITY_OVERLAP:
                    cluster.add(other_claim['id'])

            if len(cluster) > 1:
                clusters.append(cluster)

        # Merge overlapping clusters
        return self._merge_overlapping_clusters(clusters)

    async def _cluster_pass2(
        self, pass1_clusters: List[Set[uuid.UUID]], claims: List[Dict]
    ) -> List[Set[uuid.UUID]]:
        """
        Pass 2: Bridge temporal gaps (14-day window + 4+ entity overlap)

        Takes Pass 1 clusters and attempts to merge them using relaxed constraints
        """
        # Build claim index for fast lookup
        claim_index = {c['id']: c for c in claims}

        # Get aggregate info for each cluster
        cluster_info = []
        for cluster in pass1_clusters:
            cluster_claims = [claim_index[cid] for cid in cluster]

            # Aggregate temporal bounds
            min_time = min(c['event_time'] for c in cluster_claims)
            max_time = max(c['event_time'] for c in cluster_claims)

            # Aggregate entities (union)
            all_entities = set()
            for c in cluster_claims:
                all_entities.update(c['entity_ids'])

            cluster_info.append({
                'claim_ids': cluster,
                'min_time': min_time,
                'max_time': max_time,
                'entities': all_entities
            })

        # Try to merge clusters
        merged = []
        for i, cluster_a in enumerate(cluster_info):
            merged_cluster = cluster_a['claim_ids'].copy()

            for j, cluster_b in enumerate(cluster_info):
                if i >= j:  # Skip self and already compared
                    continue

                # Check temporal proximity (14-day window between cluster bounds)
                time_gap = min(
                    abs((cluster_a['min_time'] - cluster_b['max_time']).days),
                    abs((cluster_b['min_time'] - cluster_a['max_time']).days)
                )

                if time_gap > PASS2_TIME_WINDOW_DAYS:
                    continue

                # Check entity overlap (4+ shared entities)
                shared_entities = cluster_a['entities'] & cluster_b['entities']
                if len(shared_entities) >= PASS2_MIN_ENTITY_OVERLAP:
                    merged_cluster.update(cluster_b['claim_ids'])

            merged.append(merged_cluster)

        return self._merge_overlapping_clusters(merged)

    async def _cluster_pass3(self, pass2_clusters: List[Set[uuid.UUID]]) -> List[Set[uuid.UUID]]:
        """
        Pass 3: Transitive merging via graph DFS

        Build a graph where clusters are nodes, edges when they share claims
        Then extract connected components
        """
        # Build adjacency graph
        graph = defaultdict(set)

        for i, cluster_a in enumerate(pass2_clusters):
            for j, cluster_b in enumerate(pass2_clusters):
                if i >= j:
                    continue

                # If clusters share any claims, they should be merged
                if cluster_a & cluster_b:
                    graph[i].add(j)
                    graph[j].add(i)

        # DFS to find connected components
        visited = set()
        components = []

        def dfs(node, component):
            visited.add(node)
            component.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)

        for i in range(len(pass2_clusters)):
            if i not in visited:
                component = set()
                dfs(i, component)

                # Merge all clusters in this component
                merged = set()
                for cluster_idx in component:
                    merged.update(pass2_clusters[cluster_idx])

                components.append(merged)

        return components

    def _merge_overlapping_clusters(self, clusters: List[Set[uuid.UUID]]) -> List[Set[uuid.UUID]]:
        """
        Merge clusters that share any claims (transitive closure)
        """
        if not clusters:
            return []

        # Build union-find structure
        merged = []
        visited = [False] * len(clusters)

        for i in range(len(clusters)):
            if visited[i]:
                continue

            # Start new merged cluster
            current = clusters[i].copy()
            visited[i] = True

            # Keep merging until no more overlaps found
            changed = True
            while changed:
                changed = False
                for j in range(len(clusters)):
                    if visited[j]:
                        continue

                    if current & clusters[j]:  # Overlap found
                        current.update(clusters[j])
                        visited[j] = True
                        changed = True

            merged.append(current)

        return merged

    async def _materialize_events(
        self, conn: asyncpg.Connection, clusters: List[Set[uuid.UUID]],
        claims: List[Dict]
    ):
        """
        Create/update event records from final clusters
        """
        claim_index = {c['id']: c for c in claims}

        for cluster_claims in clusters:
            if not cluster_claims:
                continue

            # Aggregate cluster metadata
            cluster_claim_objs = [claim_index[cid] for cid in cluster_claims]

            # Temporal bounds
            event_start = min(c['event_time'] for c in cluster_claim_objs)
            event_end = max(c['event_time'] for c in cluster_claim_objs)

            # Aggregate entities
            all_entities = set()
            for c in cluster_claim_objs:
                all_entities.update(c['entity_ids'])

            # Average confidence
            avg_confidence = sum(c['confidence'] for c in cluster_claim_objs) / len(cluster_claim_objs)

            # Get page IDs
            page_ids = {c['page_id'] for c in cluster_claim_objs if c['page_id']}

            # Create event (simple placeholder title for now)
            event_title = f"Event ({event_start.strftime('%Y-%m-%d')})"

            logger.info(
                f"💾 Creating event: {len(cluster_claims)} claims, "
                f"{len(all_entities)} entities, {len(page_ids)} pages"
            )

            # Insert event
            event_id = await conn.fetchval("""
                INSERT INTO core.events (
                    title, summary, event_start, event_end,
                    confidence, claims_count, pages_count,
                    event_scale, status
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
            """,
                event_title,
                f"{len(cluster_claims)} claims from {len(page_ids)} sources",
                event_start,
                event_end,
                avg_confidence,
                len(cluster_claims),
                len(page_ids),
                'micro',  # Default scale for now
                'active'
            )

            # Link entities to event
            for entity_id in all_entities:
                await conn.execute("""
                    INSERT INTO core.event_entities (event_id, entity_id)
                    VALUES ($1, $2)
                    ON CONFLICT (event_id, entity_id) DO NOTHING
                """, event_id, entity_id)

            # Link pages to event
            for page_id in page_ids:
                await conn.execute("""
                    INSERT INTO core.page_events (page_id, event_id)
                    VALUES ($1, $2)
                    ON CONFLICT (page_id, event_id) DO NOTHING
                """, page_id, event_id)


async def main():
    """Main worker entry point"""
    worker_id = int(os.getenv('WORKER_ID', '1'))

    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        user=os.getenv('POSTGRES_USER', 'herenews_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
        database=os.getenv('POSTGRES_DB', 'herenews'),
        min_size=2,
        max_size=5
    )

    job_queue = JobQueue(os.getenv('REDIS_URL', 'redis://redis:6379'))
    await job_queue.connect()

    worker = EventWorker(db_pool, job_queue, worker_id=worker_id)
    logger.info(f"📊 Starting Event worker {worker_id}")

    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    finally:
        await db_pool.close()
        await job_queue.close()


if __name__ == "__main__":
    asyncio.run(main())
