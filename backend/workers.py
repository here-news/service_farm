"""
Demo Workers - Extraction, Semantic (entities + claims), Event formation
"""
import asyncio
import json
import os
from datetime import datetime
from urllib.parse import urlparse
import asyncpg
import redis.asyncio as redis
from trafilatura import extract, fetch_url
from langdetect import detect_langs
from openai import AsyncOpenAI

# Use existing services
import sys
sys.path.append('/media/im3/plus/lab4/re_news/service_farm')

# Import production semantic analyzer
from semantic_analyzer import EnhancedSemanticAnalyzer

openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
semantic_analyzer = EnhancedSemanticAnalyzer()


class ExtractionWorker:
    """Worker 1: URL → Content"""

    def __init__(self, pool: asyncpg.Pool, redis_client):
        self.pool = pool
        self.redis = redis_client

    async def process(self, job: dict):
        """
        Extract content from URL

        Input: {page_id, url}
        Output: Updates pages table, queues semantic_worker
        """
        page_id = job['page_id']
        url = job['url']

        print(f"[Extraction] Processing {url}")

        async with self.pool.acquire() as conn:
            # Update status
            await conn.execute("""
                UPDATE pages SET status = 'extracting', updated_at = NOW()
                WHERE id = $1
            """, page_id)

        try:
            # Fetch and extract (use urllib directly as fetch_url has issues)
            import urllib.request
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; BreathingKB/1.0)'}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                downloaded = response.read().decode('utf-8', errors='ignore')

            if not downloaded:
                raise Exception("Failed to fetch URL")

            result = extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                output_format='json'
            )

            if not result:
                raise Exception("Failed to extract content")

            result_dict = json.loads(result) if isinstance(result, str) else result

            # Detect language from content
            language = 'en'
            lang_conf = 0.6
            if result_dict.get('text'):
                try:
                    langs = detect_langs(result_dict['text'][:500])
                    if langs:
                        language = langs[0].lang
                        lang_conf = langs[0].prob
                except:
                    pass

            # Update pages table
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    UPDATE pages
                    SET title = $1,
                        content_text = $2,
                        language = $3,
                        language_confidence = $4,
                        word_count = $5,
                        status = 'extracted',
                        updated_at = NOW()
                    WHERE id = $6
                """,
                    result_dict.get('title', ''),
                    result_dict.get('text', ''),
                    language,
                    lang_conf,
                    len(result_dict.get('text', '').split()),
                    page_id
                )

            print(f"[Extraction] ✓ Extracted: {result_dict.get('title', 'No title')[:50]}")

            # Queue semantic worker
            await self.redis.lpush(
                'queue:semantic:normal',
                json.dumps({'page_id': page_id})
            )

        except Exception as e:
            print(f"[Extraction] ✗ Failed: {e}")

            async with self.pool.acquire() as conn:
                await conn.execute("""
                    UPDATE pages
                    SET status = 'failed',
                        updated_at = NOW()
                    WHERE id = $1
                """, page_id)


class SemanticWorker:
    """Worker 2: Content → Entities + Claims"""

    def __init__(self, pool: asyncpg.Pool, redis_client):
        self.pool = pool
        self.redis = redis_client

    async def process(self, job: dict):
        """
        Extract entities and claims from content using production semantic analyzer

        Input: {page_id}
        Output: Creates entities, claims, queues event_worker
        """
        page_id = job['page_id']

        print(f"[Semantic] Processing page {page_id}")

        async with self.pool.acquire() as conn:
            # Get page content
            page = await conn.fetchrow("""
                SELECT id, url, canonical_url, title, content_text, language
                FROM pages
                WHERE id = $1
            """, page_id)

            if not page or not page['content_text']:
                print(f"[Semantic] No content for {page_id}")
                return

            # Update status
            await conn.execute("""
                UPDATE pages SET status = 'extracting_entities', updated_at = NOW()
                WHERE id = $1
            """, page_id)

        try:
            # Use production semantic analyzer with proper temporal extraction
            page_meta = {
                'title': page['title'],
                'pub_time': datetime.now().isoformat(),  # TODO: extract from page
                'site': urlparse(page['url']).netloc
            }

            page_text = [{'selector': 'article', 'text': page['content_text']}]

            result = await semantic_analyzer.extract_enhanced_claims(
                page_meta=page_meta,
                page_text=page_text,
                url=page['canonical_url'] or page['url'],
                lang=page['language']
            )

            # Extract entities and claims from production format
            entities = self._convert_entities_from_production(result['entities'])
            claims = self._convert_claims_from_production(result['claims'], pub_time=page_meta['pub_time'])

            # Save to database
            async with self.pool.acquire() as conn:
                # Save entities
                for entity in entities:
                    # Normalize entity type to uppercase for consistency
                    entity_type_normalized = entity['type'].upper()

                    entity_id = await conn.fetchval("""
                        INSERT INTO entities (
                            canonical_name, entity_type, language,
                            confidence, created_at
                        )
                        VALUES ($1, $2, $3, $4, NOW())
                        ON CONFLICT (canonical_name, entity_type, language)
                        DO UPDATE SET confidence = GREATEST(entities.confidence, EXCLUDED.confidence)
                        RETURNING id
                    """,
                        entity['name'],
                        entity_type_normalized,
                        page['language'],
                        entity['confidence']
                    )

                    # Link page to entity
                    if entity_id:
                        await conn.execute("""
                            INSERT INTO page_entities (page_id, entity_id, mention_count)
                            VALUES ($1, $2, $3)
                            ON CONFLICT DO NOTHING
                        """, page['id'], entity_id, entity.get('mentions', 1))

                # Save claims
                for claim in claims:
                    # Parse event_time if it's a string
                    event_time = claim.get('event_time')
                    if event_time and isinstance(event_time, str):
                        from dateutil import parser as dateutil_parser
                        try:
                            event_time = dateutil_parser.parse(event_time)
                        except:
                            event_time = None

                    await conn.execute("""
                        INSERT INTO claims (
                            page_id, text, confidence, event_time, created_at
                        )
                        VALUES ($1, $2, $3, $4, NOW())
                    """,
                        page['id'],
                        claim['text'],
                        claim['confidence'],
                        event_time
                    )

                # Update page status
                await conn.execute("""
                    UPDATE pages
                    SET status = 'entities_extracted',
                        updated_at = NOW()
                    WHERE id = $1
                """, page_id)

            print(f"[Semantic] ✓ Extracted {len(entities)} entities, {len(claims)} claims")

            # Queue BOTH event worker AND entity enrichment worker (parallel)
            await self.redis.lpush(
                'queue:event:normal',
                json.dumps({'page_id': page_id})
            )
            await self.redis.lpush(
                'queue:entity_enrichment:normal',
                json.dumps({'page_id': page_id})
            )

        except Exception as e:
            print(f"[Semantic] ✗ Failed: {e}")
            import traceback
            traceback.print_exc()

    def _convert_entities_from_production(self, entities_dict: dict) -> list[dict]:
        """Convert production entities format to demo format"""
        entities = []

        # Production format: {"people": [...], "organizations": [...], "locations": [...]}
        for person in entities_dict.get('people', []):
            entities.append({
                'name': person,
                'type': 'PERSON',
                'confidence': 0.8,
                'mentions': 1
            })

        for org in entities_dict.get('organizations', []):
            entities.append({
                'name': org,
                'type': 'ORG',
                'confidence': 0.8,
                'mentions': 1
            })

        for loc in entities_dict.get('locations', []):
            entities.append({
                'name': loc,
                'type': 'GPE',
                'confidence': 0.8,
                'mentions': 1
            })

        return entities

    def _convert_claims_from_production(self, claims_list: list, pub_time: str = None) -> list[dict]:
        """
        Convert production claims format to demo format

        Filters out historical background timestamps (>6 months before pub_time)
        and defaults to pub_time for current events
        """
        from dateutil import parser
        demo_claims = []

        # Parse pub_time for filtering
        pub_dt = None
        if pub_time:
            try:
                pub_dt = parser.parse(pub_time)
            except:
                pass

        for claim in claims_list:
            # Production format has 'when' object with temporal info
            when_info = claim.get('when', {}) or {}
            event_time_str = None

            # Try to construct ISO timestamp from when info
            if when_info.get('date'):
                date_str = when_info['date']
                time_str = when_info.get('time', '00:00:00')
                event_time_str = f"{date_str}T{time_str}Z"
            elif when_info.get('event_time'):
                event_time_str = when_info['event_time']

            # FILTER: If event_time is >6 months before pub_time, it's background - use pub_time instead
            if event_time_str and pub_dt:
                try:
                    event_dt = parser.parse(event_time_str)
                    days_diff = (pub_dt - event_dt).days
                    if days_diff > 180:  # 6 months
                        print(f"⏰ Claim: Filtering old timestamp {event_time_str} ({days_diff} days old), using pub_time")
                        event_time_str = pub_time
                except:
                    pass

            # Fallback to pub_time if no event_time
            if not event_time_str and pub_dt:
                event_time_str = pub_time

            demo_claims.append({
                'text': claim.get('text', ''),
                'confidence': claim.get('confidence', 0.7),
                'event_time': event_time_str
            })

        return demo_claims


class EventWorker:
    """Worker 3: Claims → Events using Temporal-Entity Clustering"""

    def __init__(self, pool: asyncpg.Pool, redis_client):
        self.pool = pool
        self.redis = redis_client
        # Clustering parameters from validated experiments
        self.temporal_window_days = 2
        self.min_entity_overlap = 2
        self.min_claims_per_event = 2  # Lowered from 5 for demo

    async def process(self, job: dict):
        """
        Cluster claims into events using temporal proximity + entity overlap

        Strategy:
        1. Fetch recent claims (30 days) not yet clustered
        2. Cluster claims by temporal proximity (2 days) + entity overlap (2+)
        3. For each cluster: create event or match to existing parent
        4. Detect hierarchy: small clusters become sub-events of larger ones

        Input: {page_id}
        Output: Links page to events with hierarchical structure
        """
        page_id = job['page_id']
        print(f"[Event] Processing page {page_id}")

        async with self.pool.acquire() as conn:
            # Get claims for this page with entity context
            page_claims = await conn.fetch("""
                SELECT
                    c.id, c.text, c.event_time, c.page_id,
                    array_agg(DISTINCT e.id) as entity_ids,
                    array_agg(DISTINCT e.canonical_name) as entity_names
                FROM claims c
                LEFT JOIN page_entities pe ON c.page_id = pe.page_id
                LEFT JOIN entities e ON pe.entity_id = e.id
                WHERE c.page_id = $1
                GROUP BY c.id, c.text, c.event_time, c.page_id
            """, page_id)

            if not page_claims:
                print(f"[Event] No claims for {page_id}")
                return

            # Also fetch recent claims from other pages for clustering context
            # (within 30 days window for broader clustering)
            all_recent_claims = await conn.fetch("""
                SELECT
                    c.id, c.text, c.event_time, c.page_id,
                    array_agg(DISTINCT e.id) as entity_ids,
                    array_agg(DISTINCT e.canonical_name) as entity_names
                FROM claims c
                LEFT JOIN page_entities pe ON c.page_id = pe.page_id
                LEFT JOIN entities e ON pe.entity_id = e.id
                WHERE c.event_time >= NOW() - INTERVAL '30 days'
                   OR c.event_time IS NULL
                GROUP BY c.id, c.text, c.event_time, c.page_id
            """)

            # Convert to clustering format
            claims_data = self.prepare_claims_for_clustering(all_recent_claims)

            # Cluster claims
            clusters = self.cluster_claims_temporal_entity(claims_data)

            print(f"[Event] Found {len(clusters)} clusters from recent claims")

            # Process clusters: create/match events
            await self.process_clusters(conn, clusters, page_id)

    def prepare_claims_for_clustering(self, claim_records):
        """Convert DB records to clustering format"""
        claims = []
        for rec in claim_records:
            # Filter out None entity_ids
            entity_ids = [eid for eid in (rec['entity_ids'] or []) if eid is not None]

            claims.append({
                'id': str(rec['id']),
                'text': rec['text'],
                'time': rec['event_time'],
                'page_id': str(rec['page_id']),
                'entities': entity_ids,
                'entity_names': rec['entity_names'] or []
            })
        return claims

    def cluster_claims_temporal_entity(self, claims):
        """
        Recursive multi-pass clustering with error tolerance

        Pass 1: Tight clusters (strict temporal + entity overlap)
        Pass 2: Bridge gaps (relaxed temporal, strong entity overlap)
        Pass 3: Merge related clusters (transitive relationships)
        """
        print(f"[Event] Clustering {len(claims)} claims (multi-pass)...")
        print(f"[Event]   Temporal window: {self.temporal_window_days} days")
        print(f"[Event]   Min entity overlap: {self.min_entity_overlap}")

        # PASS 1: Form tight clusters with strict temporal constraint
        print(f"[Event] Pass 1: Forming tight clusters...")
        tight_clusters = self._form_tight_clusters(claims)
        print(f"[Event]   → {len(tight_clusters)} tight clusters")

        # PASS 2: Bridge temporal gaps with strong entity overlap
        print(f"[Event] Pass 2: Bridging gaps with entity overlap...")
        bridged_clusters = self._bridge_cluster_gaps(tight_clusters)
        print(f"[Event]   → {len(bridged_clusters)} clusters after bridging")

        # PASS 3: Merge transitively related clusters
        print(f"[Event] Pass 3: Merging transitively related clusters...")
        final_clusters = self._merge_related_clusters(bridged_clusters)
        print(f"[Event]   → {len(final_clusters)} final clusters")

        # Filter clusters by minimum size
        valid_clusters = [c for c in final_clusters if c['claim_count'] >= self.min_claims_per_event]
        print(f"[Event]   → {len(valid_clusters)} clusters meet minimum size ({self.min_claims_per_event} claims)")

        return valid_clusters

    def _form_tight_clusters(self, claims):
        """Pass 1: Strict temporal + entity clustering"""
        clusters = []

        for claim in claims:
            if not claim['entities'] or not claim['time']:
                continue

            matched_cluster = None
            best_overlap = 0

            for cluster in clusters:
                # Strict temporal check (within configured window)
                temporal_match = False
                for clustered_claim in cluster['claims']:
                    if claim['time'] and clustered_claim['time']:
                        try:
                            time_diff = abs((claim['time'] - clustered_claim['time']).days)
                            if time_diff <= self.temporal_window_days:
                                temporal_match = True
                                break
                        except:
                            continue

                if not temporal_match:
                    continue

                # Entity overlap
                claim_entities = set(claim['entities'])
                cluster_entities = set(cluster['all_entities'])
                overlap = len(claim_entities & cluster_entities)

                if overlap >= self.min_entity_overlap and overlap > best_overlap:
                    matched_cluster = cluster
                    best_overlap = overlap

            if matched_cluster:
                matched_cluster['claims'].append(claim)
                matched_cluster['all_entities'].update(claim['entities'])
                matched_cluster['claim_count'] += 1
                if claim['time'] < matched_cluster['start_time']:
                    matched_cluster['start_time'] = claim['time']
                if claim['time'] > matched_cluster['end_time']:
                    matched_cluster['end_time'] = claim['time']
            else:
                clusters.append({
                    'cluster_id': len(clusters),
                    'claims': [claim],
                    'claim_count': 1,
                    'all_entities': set(claim['entities']),
                    'start_time': claim['time'],
                    'end_time': claim['time']
                })

        return clusters

    def _bridge_cluster_gaps(self, clusters):
        """
        Pass 2: Bridge temporal gaps between clusters with strong entity overlap

        Tolerates timing errors by looking at entity continuity across gaps
        """
        if len(clusters) < 2:
            return clusters

        # Sort by start time
        sorted_clusters = sorted(clusters, key=lambda c: c['start_time'])

        merged = []
        used = set()

        for i, cluster1 in enumerate(sorted_clusters):
            if i in used:
                continue

            # Try to merge with later clusters
            current_group = [cluster1]
            used.add(i)

            for j, cluster2 in enumerate(sorted_clusters[i+1:], start=i+1):
                if j in used:
                    continue

                # Check entity overlap (high threshold for bridging)
                entities1 = cluster1['all_entities']
                entities2 = cluster2['all_entities']
                overlap = len(entities1 & entities2)

                # Relaxed temporal check (within 30 days for same story)
                time_gap = abs((cluster2['start_time'] - cluster1['end_time']).days)

                # Bridge if: strong entity overlap (4+) OR moderate overlap (2+) + reasonable time gap (≤14 days)
                should_bridge = (
                    (overlap >= 4) or  # Very strong entity signal
                    (overlap >= self.min_entity_overlap and time_gap <= 14)  # Moderate signal + not too distant
                )

                if should_bridge:
                    current_group.append(cluster2)
                    used.add(j)
                    # Update cluster1 bounds to include cluster2
                    cluster1['all_entities'].update(entities2)
                    if cluster2['end_time'] > cluster1['end_time']:
                        cluster1['end_time'] = cluster2['end_time']

            # Merge all clusters in this group
            if len(current_group) > 1:
                merged_cluster = self._merge_cluster_group(current_group)
                merged.append(merged_cluster)
            else:
                merged.append(cluster1)

        return merged

    def _merge_cluster_group(self, cluster_group):
        """Merge multiple clusters into one"""
        merged = {
            'cluster_id': cluster_group[0]['cluster_id'],
            'claims': [],
            'claim_count': 0,
            'all_entities': set(),
            'start_time': None,
            'end_time': None
        }

        for cluster in cluster_group:
            merged['claims'].extend(cluster['claims'])
            merged['claim_count'] += cluster['claim_count']
            merged['all_entities'].update(cluster['all_entities'])

            if merged['start_time'] is None or cluster['start_time'] < merged['start_time']:
                merged['start_time'] = cluster['start_time']
            if merged['end_time'] is None or cluster['end_time'] > merged['end_time']:
                merged['end_time'] = cluster['end_time']

        return merged

    def _merge_related_clusters(self, clusters):
        """
        Pass 3: Merge clusters that share significant entities (transitive closure)

        Handles cases where A shares entities with B, B shares with C, so A-B-C should merge
        """
        if len(clusters) < 2:
            return clusters

        # Build adjacency graph
        n = len(clusters)
        adjacent = [[False] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                entities_i = clusters[i]['all_entities']
                entities_j = clusters[j]['all_entities']
                overlap = len(entities_i & entities_j)

                # Clusters are related if they share 3+ entities (high confidence in same story)
                if overlap >= 3:
                    adjacent[i][j] = True
                    adjacent[j][i] = True

        # Find connected components (transitive groups)
        visited = [False] * n
        components = []

        def dfs(node, component):
            visited[node] = True
            component.append(node)
            for neighbor in range(n):
                if adjacent[node][neighbor] and not visited[neighbor]:
                    dfs(neighbor, component)

        for i in range(n):
            if not visited[i]:
                component = []
                dfs(i, component)
                components.append(component)

        # Merge each connected component
        merged_clusters = []
        for component in components:
            if len(component) == 1:
                merged_clusters.append(clusters[component[0]])
            else:
                cluster_group = [clusters[idx] for idx in component]
                merged = self._merge_cluster_group(cluster_group)
                merged_clusters.append(merged)

        return merged_clusters

    async def process_clusters(self, conn, clusters, current_page_id):
        """
        For each cluster:
        1. Check if event already exists for this cluster
        2. If not, create new event or match to parent event
        3. Link pages to events
        4. Detect hierarchies
        """
        # Track which event the current page should be linked to
        current_page_event_id = None

        for cluster in clusters:
            # Get unique page IDs in this cluster
            page_ids = list(set(c['page_id'] for c in cluster['claims']))

            # Check if this cluster contains the current page's claims
            current_page_in_cluster = current_page_id in page_ids

            # Check if cluster already has an event
            # (by checking if all pages share a common event)
            existing_event = await self.find_cluster_event(conn, page_ids, cluster)

            if existing_event:
                event_id = existing_event['id']
                print(f"[Event] Cluster matches existing event: {existing_event['title']}")
            else:
                # Create new event for this cluster
                event_id = await self.create_event_from_cluster(conn, cluster)
                print(f"[Event] Created new event from cluster")

            # Remember which event this current page should link to
            if current_page_in_cluster:
                current_page_event_id = event_id

            # Link all pages in cluster to this event
            for page_id in page_ids:
                await conn.execute("""
                    INSERT INTO page_events (page_id, event_id)
                    VALUES ($1, $2)
                    ON CONFLICT DO NOTHING
                """, page_id, event_id)

        # Ensure current page is linked to an event and marked complete
        if current_page_event_id:
            await conn.execute("""
                INSERT INTO page_events (page_id, event_id)
                VALUES ($1, $2)
                ON CONFLICT DO NOTHING
            """, current_page_id, current_page_event_id)

            await conn.execute("""
                UPDATE pages
                SET status = 'complete', updated_at = NOW()
                WHERE id = $1
            """, current_page_id)
            print(f"[Event] ✓ Linked page {current_page_id} to event {current_page_event_id}")
        else:
            # Fallback: if page wasn't in any cluster, try matching by entities
            print(f"[Event] Page {current_page_id} not in any cluster, trying entity-based matching...")

            # Get entities for this page
            page_entities = await conn.fetch("""
                SELECT entity_id FROM page_entities WHERE page_id = $1
            """, current_page_id)

            if page_entities:
                entity_ids = [e['entity_id'] for e in page_entities]

                # Find event with most shared entities (prioritize best match)
                matching_event = await conn.fetchrow("""
                    SELECT e.id, e.title, COUNT(DISTINCT ee.entity_id) as shared_entities
                    FROM events e
                    JOIN event_entities ee ON e.id = ee.event_id
                    WHERE ee.entity_id = ANY($1)
                    GROUP BY e.id, e.title
                    HAVING COUNT(DISTINCT ee.entity_id) >= 2
                    ORDER BY COUNT(DISTINCT ee.entity_id) DESC, e.updated_at DESC
                    LIMIT 1
                """, entity_ids)

                if matching_event:
                    await conn.execute("""
                        INSERT INTO page_events (page_id, event_id)
                        VALUES ($1, $2)
                        ON CONFLICT DO NOTHING
                    """, current_page_id, matching_event['id'])

                    await conn.execute("""
                        UPDATE pages
                        SET status = 'complete', updated_at = NOW()
                        WHERE id = $1
                    """, current_page_id)

                    print(f"[Event] ✓ Fallback matched to event: {matching_event['title']} ({matching_event['shared_entities']} shared entities)")
                else:
                    print(f"[Event] ⚠ No matching event found for page {current_page_id}")

        # After processing clusters, detect hierarchies
        await self.detect_hierarchies(conn, clusters)

        # Detect temporal phases within large merged clusters
        await self.detect_temporal_phases(conn, clusters)

    async def find_cluster_event(self, conn, page_ids, cluster):
        """
        Find if an event already exists for this cluster
        by checking if pages share a common event with matching entities
        """
        if not page_ids:
            return None

        # Find events that have pages from this cluster
        events = await conn.fetch("""
            SELECT DISTINCT e.id, e.title, e.summary,
                   COUNT(DISTINCT pe.page_id) as page_count
            FROM events e
            JOIN page_events pe ON e.id = pe.event_id
            WHERE pe.page_id = ANY($1)
            GROUP BY e.id, e.title, e.summary
            ORDER BY page_count DESC
            LIMIT 1
        """, page_ids)

        if events and events[0]['page_count'] >= len(page_ids) * 0.5:
            # If majority of pages already linked to same event, use it
            return events[0]

        return None

    async def create_event_from_cluster(self, conn, cluster):
        """Create event from claim cluster using LLM"""
        claim_texts = [c['text'] for c in cluster['claims']]

        # Get full entity records (with entity_type) for the cluster
        entity_ids = list(cluster['all_entities'])
        entities = await conn.fetch("""
            SELECT id, canonical_name, entity_type
            FROM entities
            WHERE id = ANY($1)
        """, entity_ids)

        # Use LLM to synthesize event
        event_data = await self.synthesize_event(claim_texts, entities)

        if not event_data:
            # Fallback
            entity_names = [e['canonical_name'] for e in entities]
            event_data = {
                'title': f"Event involving {', '.join(entity_names[:3])}",
                'summary': None,
                'event_type': None,
                'location': None,
                'event_start': cluster['start_time']
            }

        # Parse event_start if string
        event_start = event_data.get('event_start') or cluster['start_time']
        if event_start and isinstance(event_start, str):
            from dateutil import parser as dateutil_parser
            try:
                event_start = dateutil_parser.parse(event_start)
            except:
                event_start = cluster['start_time']

        # Insert event
        event_id = await conn.fetchval("""
            INSERT INTO events (
                title, summary, event_type, location, event_start,
                confidence, created_at
            )
            VALUES ($1, $2, $3, $4, $5, 0.7, NOW())
            RETURNING id
        """,
            event_data['title'],
            event_data['summary'],
            event_data['event_type'],
            event_data['location'],
            event_start
        )

        # Link entities to event
        for entity_id in cluster['all_entities']:
            await conn.execute("""
                INSERT INTO event_entities (event_id, entity_id)
                VALUES ($1, $2)
                ON CONFLICT DO NOTHING
            """, event_id, entity_id)

        return str(event_id)

    async def detect_hierarchies(self, conn, clusters):
        """
        Detect parent-child relationships between events

        Strategy:
        - Larger clusters (more claims/pages) become parent events
        - Smaller clusters with overlapping entities + temporal proximity become sub-events (PHASE_OF)
        - Mark sub-events with parent_event_id and relationship_type
        """
        if len(clusters) < 2:
            return

        # Sort by size (largest first)
        sorted_clusters = sorted(clusters, key=lambda c: c['claim_count'], reverse=True)

        for i, small_cluster in enumerate(sorted_clusters):
            # Check if this cluster should be a sub-event of a larger one
            for large_cluster in sorted_clusters[:i]:
                if large_cluster['claim_count'] <= small_cluster['claim_count']:
                    continue

                # Check entity overlap
                small_entities = small_cluster['all_entities']
                large_entities = large_cluster['all_entities']
                overlap = len(small_entities & large_entities)

                if overlap >= self.min_entity_overlap:
                    # Check temporal overlap
                    small_start = small_cluster['start_time']
                    small_end = small_cluster['end_time']
                    large_start = large_cluster['start_time']
                    large_end = large_cluster['end_time']

                    # If small event is within or near large event's timespan
                    time_gap = min(
                        abs((small_start - large_start).days),
                        abs((small_end - large_end).days)
                    )

                    if time_gap <= self.temporal_window_days * 3:  # 6 days max
                        # Mark as sub-event
                        await self.mark_as_subevent(conn, small_cluster, large_cluster)
                        break

    async def mark_as_subevent(self, conn, sub_cluster, parent_cluster):
        """Mark events from sub_cluster as children of parent_cluster events"""
        # Get event IDs for both clusters
        sub_page_ids = list(set(c['page_id'] for c in sub_cluster['claims']))
        parent_page_ids = list(set(c['page_id'] for c in parent_cluster['claims']))

        # Find events for these pages
        sub_event = await conn.fetchrow("""
            SELECT DISTINCT e.id, e.title
            FROM events e
            JOIN page_events pe ON e.id = pe.event_id
            WHERE pe.page_id = ANY($1)
            LIMIT 1
        """, sub_page_ids)

        parent_event = await conn.fetchrow("""
            SELECT DISTINCT e.id, e.title
            FROM events e
            JOIN page_events pe ON e.id = pe.event_id
            WHERE pe.page_id = ANY($1)
            LIMIT 1
        """, parent_page_ids)

        if sub_event and parent_event and sub_event['id'] != parent_event['id']:
            # Update sub-event to reference parent
            await conn.execute("""
                UPDATE events
                SET parent_event_id = $1,
                    relationship_type = 'PHASE_OF',
                    event_scale = 'micro',
                    updated_at = NOW()
                WHERE id = $2
            """, parent_event['id'], sub_event['id'])

            # Update parent to macro scale
            await conn.execute("""
                UPDATE events
                SET event_scale = 'macro',
                    updated_at = NOW()
                WHERE id = $1
            """, parent_event['id'])

            print(f"[Event] ✓ Hierarchy: '{sub_event['title']}' → PHASE_OF → '{parent_event['title']}'")

    async def detect_temporal_phases(self, conn, clusters):
        """
        Detect temporal phases WITHIN large merged clusters

        When recursive clustering merges everything into one big cluster,
        we still want to detect temporal sub-events (phases) based on claim distribution

        Strategy:
        - For clusters with 20+ claims spanning >7 days
        - Re-cluster the claims with STRICT temporal window (2 days)
        - Create parent event for the whole story
        - Create child events for each temporal phase
        """
        for cluster in clusters:
            if cluster['claim_count'] < 20:
                continue

            time_span = (cluster['end_time'] - cluster['start_time']).days
            if time_span <= 7:
                continue

            print(f"[Event] Large cluster detected: {cluster['claim_count']} claims over {time_span} days")
            print(f"[Event]   Detecting temporal phases...")

            # Re-cluster with strict temporal window to find phases
            phase_clusters = self._form_tight_clusters(cluster['claims'])

            if len(phase_clusters) < 2:
                print(f"[Event]   No distinct phases found")
                continue

            print(f"[Event]   Found {len(phase_clusters)} temporal phases")

            # Get the parent event for this cluster
            page_ids = list(set(c['page_id'] for c in cluster['claims']))
            parent_event = await conn.fetchrow("""
                SELECT DISTINCT e.id, e.title
                FROM events e
                JOIN page_events pe ON e.id = pe.event_id
                WHERE pe.page_id = ANY($1)
                LIMIT 1
            """, page_ids)

            if not parent_event:
                continue

            # Create sub-events for each phase
            for i, phase_cluster in enumerate(sorted(phase_clusters, key=lambda c: c['start_time'])):
                if phase_cluster['claim_count'] < 3:
                    continue

                # Check if a phase event already exists for this time period
                phase_page_ids = list(set(c['page_id'] for c in phase_cluster['claims']))
                existing_phase = await conn.fetchrow("""
                    SELECT DISTINCT e.id
                    FROM events e
                    JOIN page_events pe ON e.id = pe.event_id
                    WHERE pe.page_id = ANY($1)
                      AND e.parent_event_id = $2
                      AND e.relationship_type = 'PHASE_OF'
                    LIMIT 1
                """, phase_page_ids, parent_event['id'])

                if existing_phase:
                    # Phase already exists, just ensure pages are linked
                    phase_event_id = existing_phase['id']
                    print(f"[Event]   ✓ Phase {i+1}: Reusing existing phase event")
                else:
                    # Create new phase event
                    phase_event_id = await self.create_event_from_cluster(conn, phase_cluster)

                    # Mark as sub-event of parent
                    await conn.execute("""
                        UPDATE events
                        SET parent_event_id = $1,
                            relationship_type = 'PHASE_OF',
                            event_scale = 'micro',
                            updated_at = NOW()
                        WHERE id = $2
                    """, parent_event['id'], phase_event_id)

                    phase_event_title = await conn.fetchval(
                        "SELECT title FROM events WHERE id = $1", phase_event_id
                    )
                    print(f"[Event]   ✓ Phase {i+1}: '{phase_event_title}' → PHASE_OF → '{parent_event['title']}'")

                # Link phase pages to phase event (already computed above)
                for page_id in phase_page_ids:
                    await conn.execute("""
                        INSERT INTO page_events (page_id, event_id)
                        VALUES ($1, $2)
                        ON CONFLICT DO NOTHING
                    """, page_id, phase_event_id)

            # Update parent to macro scale
            await conn.execute("""
                UPDATE events
                SET event_scale = 'macro',
                    updated_at = NOW()
                WHERE id = $1
            """, parent_event['id'])

    async def detect_story_level_events(self, conn):
        """
        Detect story-level (super-macro) events that group multiple macro events

        Strategy:
        - Find root events (no parent) that share 2+ entities
        - Group them under a story-level event if they're temporally related (within 30 days)
        - Create story event with event_scale='story'
        """
        # Get all root events (no parent)
        root_events = await conn.fetch("""
            SELECT e.id, e.title, e.event_scale, e.event_start,
                   array_agg(DISTINCT ee.entity_id) as entity_ids,
                   COUNT(DISTINCT pe.page_id) as page_count
            FROM events e
            LEFT JOIN event_entities ee ON e.id = ee.event_id
            LEFT JOIN page_events pe ON e.id = pe.event_id
            WHERE e.parent_event_id IS NULL
            GROUP BY e.id
            HAVING COUNT(DISTINCT ee.entity_id) >= 2
            ORDER BY e.event_start NULLS LAST
        """)

        if len(root_events) < 2:
            return

        # Group events that share entities and are temporally related
        event_groups = []
        used_events = set()

        for i, event1 in enumerate(root_events):
            if event1['id'] in used_events:
                continue

            group = {
                'events': [event1],
                'all_entities': set(event1['entity_ids']),
                'total_pages': event1['page_count'],
                'earliest_time': event1['event_start'],
                'latest_time': event1['event_start']
            }

            for event2 in root_events[i+1:]:
                if event2['id'] in used_events:
                    continue

                # Check entity overlap
                event1_entities = set(event1['entity_ids']) if event1['entity_ids'] else set()
                event2_entities = set(event2['entity_ids']) if event2['entity_ids'] else set()
                overlap = len(event1_entities & event2_entities)

                if overlap >= self.min_entity_overlap:
                    # Check temporal proximity (30 days)
                    if event1['event_start'] and event2['event_start']:
                        time_gap = abs((event1['event_start'] - event2['event_start']).days)
                        if time_gap <= 30:  # Same story if within 30 days
                            group['events'].append(event2)
                            group['all_entities'].update(event2_entities)
                            group['total_pages'] += event2['page_count']
                            if event2['event_start']:
                                if group['earliest_time'] is None or event2['event_start'] < group['earliest_time']:
                                    group['earliest_time'] = event2['event_start']
                                if group['latest_time'] is None or event2['event_start'] > group['latest_time']:
                                    group['latest_time'] = event2['event_start']
                            used_events.add(event2['id'])

            # Only create story event if we have multiple events grouped
            if len(group['events']) >= 2:
                event_groups.append(group)
                used_events.add(event1['id'])

        # Create story-level events for each group
        for group in event_groups:
            await self.create_story_event(conn, group)

    async def create_story_event(self, conn, group):
        """Create a story-level event that encompasses multiple macro events"""
        # Get entity names for LLM prompt
        entity_ids = list(group['all_entities'])
        entities = await conn.fetch("""
            SELECT id, canonical_name, entity_type
            FROM entities
            WHERE id = ANY($1)
        """, entity_ids)

        # Use LLM to synthesize story title
        event_titles = [e['title'] for e in group['events']]
        entity_names = [e['canonical_name'] for e in entities]

        prompt = f"""You are analyzing multiple related events to create an overarching story title.

Events:
{chr(10).join(f'- {title}' for title in event_titles)}

Key entities: {', '.join(entity_names[:5])}

Create a concise, factual story title (max 10 words) that encompasses all these events.
Focus on the central subject and the overarching situation.

Return JSON: {{"title": "story title here"}}"""

        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )

            result = json.loads(response.choices[0].message.content)
            story_title = result.get('title', f"Story involving {', '.join(entity_names[:3])}")
        except:
            story_title = f"Story involving {', '.join(entity_names[:3])}"

        # Create story event
        story_event_id = await conn.fetchval("""
            INSERT INTO events (
                title, event_type, event_start, event_scale, confidence, created_at
            )
            VALUES ($1, 'ongoing_story', $2, 'story', 0.8, NOW())
            RETURNING id
        """, story_title, group['earliest_time'])

        # Link entities to story event
        for entity_id in entity_ids:
            await conn.execute("""
                INSERT INTO event_entities (event_id, entity_id)
                VALUES ($1, $2)
                ON CONFLICT DO NOTHING
            """, story_event_id, entity_id)

        # Update all events in group to be children of story event
        for event in group['events']:
            await conn.execute("""
                UPDATE events
                SET parent_event_id = $1,
                    relationship_type = 'PART_OF',
                    updated_at = NOW()
                WHERE id = $2
            """, story_event_id, event['id'])

            print(f"[Event] ✓ Story: '{event['title']}' → PART_OF → '{story_title}'")

    async def match_or_create_event(
        self,
        conn,
        claims: list,
        entities: list
    ) -> str:
        """
        Match claims to existing event or create new one using LLM

        Strategy:
        1. Find events with 2+ shared entities
        2. Use LLM to check if claims are about same event
        3. If match: link to existing event, update summary if needed
        4. If no match: use LLM to synthesize new event from claims
        """
        if not entities:
            return None

        entity_ids = [e['id'] for e in entities]
        claim_texts = [c['text'] for c in claims]

        # Find events that share entities
        existing_events = await conn.fetch("""
            SELECT DISTINCT e.id, e.title, e.summary, e.event_type,
                   COUNT(*) as shared_entities
            FROM events e
            JOIN event_entities ee ON e.id = ee.event_id
            WHERE ee.entity_id = ANY($1)
            GROUP BY e.id, e.title, e.summary, e.event_type
            ORDER BY shared_entities DESC
            LIMIT 3
        """, entity_ids)

        # Check if claims match existing events
        if existing_events and existing_events[0]['shared_entities'] >= 2:
            for existing in existing_events:
                # Use LLM to check if claims belong to this event
                is_match = await self.check_event_match(
                    claim_texts,
                    existing['title'],
                    existing['summary']
                )

                if is_match:
                    event_id = existing['id']
                    print(f"[Event] Matched to existing event: {existing['title']}")

                    # Update event summary to incorporate new claims
                    await self.update_event_summary(
                        conn,
                        event_id,
                        claim_texts,
                        existing['summary']
                    )

                    return str(event_id)

        # No match - create new event using LLM
        print(f"[Event] Calling LLM to synthesize event from {len(claim_texts)} claims")
        event_data = await self.synthesize_event(claim_texts, entities)

        if not event_data:
            # Fallback to simple approach
            print(f"[Event] LLM synthesis returned None, using fallback")
            entity_names = [e['canonical_name'] for e in entities]
            event_data = {
                'title': f"Event involving {', '.join(entity_names[:3])}",
                'summary': None,
                'event_type': None,
                'location': None,
                'event_start': None
            }
        else:
            print(f"[Event] LLM synthesized event: {event_data['title']}")

        # Parse event_start if it's a string
        event_start = event_data.get('event_start')
        if event_start and isinstance(event_start, str):
            from dateutil import parser as dateutil_parser
            try:
                event_start = dateutil_parser.parse(event_start)
            except:
                event_start = None

        event_id = await conn.fetchval("""
            INSERT INTO events (
                title, summary, event_type, location, event_start,
                confidence, created_at
            )
            VALUES ($1, $2, $3, $4, $5, 0.7, NOW())
            RETURNING id
        """,
            event_data['title'],
            event_data['summary'],
            event_data['event_type'],
            event_data['location'],
            event_start
        )

        # Link entities to event
        for entity in entities:
            await conn.execute("""
                INSERT INTO event_entities (event_id, entity_id)
                VALUES ($1, $2)
                ON CONFLICT DO NOTHING
            """, event_id, entity['id'])

        print(f"[Event] Created new event: {event_data['title']}")
        return str(event_id)

    async def check_event_match(
        self,
        claim_texts: list[str],
        event_title: str,
        event_summary: str
    ) -> bool:
        """Use LLM to check if claims belong to existing event"""

        prompt = f"""Are these claims about the same overarching event or closely related sub-events?

Existing Event:
Title: {event_title}
Summary: {event_summary or "No summary yet"}

New Claims:
{chr(10).join(f"- {claim}" for claim in claim_texts)}

IMPORTANT: Consider these as the SAME event if:
- They describe the same legal case (even if different proceedings/motions)
- They cover different aspects/phases of the same ongoing situation
- They involve the same core parties and legal matter

Examples of SAME event:
- "DOJ Prosecution of John Doe" and "John Doe Files Dismissal Motion" → SAME
- "Charges Against Jane Smith" and "Jane Smith's Trial Begins" → SAME
- "Investigation into Company X" and "Company X Subpoenaed" → SAME

Only mark as DIFFERENT if:
- Completely unrelated subject matter
- Different people/organizations (unless related case)
- Different legal matters entirely

Answer with JSON: {{"is_same_event": true/false, "reasoning": "brief explanation"}}"""

        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an event matching system. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )

            result = json.loads(response.choices[0].message.content)
            return result.get('is_same_event', False)

        except Exception as e:
            print(f"[Event] LLM match check error: {e}")
            return True  # Conservative: assume match if LLM fails

    async def synthesize_event(
        self,
        claim_texts: list[str],
        entities: list
    ) -> dict:
        """Use LLM to synthesize event from claims"""

        entity_info = [f"{e['canonical_name']} ({e['entity_type']})" for e in entities]

        prompt = f"""Synthesize an event from these claims.

Claims:
{chr(10).join(f"- {claim}" for claim in claim_texts)}

Entities involved:
{chr(10).join(f"- {e}" for e in entity_info)}

Create a structured event description with:
1. Event title: Describe the CORE EVENT as a noun phrase, NOT a headline
   - Good: "DOJ Prosecution of James Comey for False Statements"
   - Good: "Federal Investigation into Hunter Biden Laptop"
   - Bad: "James Comey Seeks Dismissal of DOJ Case" (this is a headline)
   - Bad: "Prosecutors Urge Judge to..." (this is news reporting)
2. Summary (2-3 sentences) explaining what the event is about
3. Event type (legal_proceeding, investigation, incident, policy_announcement, political_event, etc.)
4. Location (if mentioned)
5. Timeframe (if mentioned, ISO format)

Return JSON:
{{
  "title": "DOJ Prosecution of James Comey for False Statements to Congress",
  "summary": "James Comey faces federal charges for allegedly making false statements to Congress and obstructing a Congressional investigation. He has filed a motion to dismiss, claiming vindictive prosecution.",
  "event_type": "legal_proceeding",
  "location": "United States",
  "event_start": null
}}

CRITICAL: Title must be a descriptive noun phrase of the core event, not a news headline or action statement."""

        try:
            print(f"[Event] Sending prompt to LLM (model=gpt-4o-mini)")
            response = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an event synthesis system. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )

            print(f"[Event] LLM response received, parsing JSON")
            result = json.loads(response.choices[0].message.content)
            print(f"[Event] Successfully synthesized event: {result.get('title', 'NO TITLE')}")
            return result

        except Exception as e:
            print(f"[Event] LLM synthesis error: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def update_event_summary(
        self,
        conn,
        event_id: str,
        new_claims: list[str],
        current_summary: str
    ):
        """Update event summary to incorporate new information"""

        if not current_summary:
            # First article - create summary from claims
            prompt = f"""Create a 2-3 sentence summary from these claims:

{chr(10).join(f"- {claim}" for claim in new_claims)}

Be concise and factual."""
        else:
            # Update existing summary
            prompt = f"""Update this event summary with new information:

Current Summary:
{current_summary}

New Claims:
{chr(10).join(f"- {claim}" for claim in new_claims)}

Return updated 2-3 sentence summary that incorporates both perspectives.
Keep it concise and balanced."""

        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a summary synthesis system. Return only the summary text, no JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            updated_summary = response.choices[0].message.content.strip()

            await conn.execute("""
                UPDATE events
                SET summary = $1, updated_at = NOW()
                WHERE id = $2
            """, updated_summary, event_id)

            print(f"[Event] Updated summary for event {event_id}")

        except Exception as e:
            print(f"[Event] Summary update error: {e}")


async def worker_loop(worker_class, queue_name: str):
    """Generic worker loop"""
    pool = await asyncpg.create_pool(
        host='localhost',
        port=5433,
        user='demo_user',
        password='demo_pass',
        database='demo_phi_here'
    )

    redis_client = await redis.from_url('redis://localhost:6379')

    worker = worker_class(pool, redis_client)

    print(f"[Worker] {worker_class.__name__} started, listening on {queue_name}")

    while True:
        try:
            # Pop job from queue (blocking)
            job_data = await redis_client.brpop(queue_name, timeout=5)

            if job_data:
                job = json.loads(job_data[1])
                await worker.process(job)

        except Exception as e:
            print(f"[Worker] Error: {e}")
            await asyncio.sleep(1)


if __name__ == '__main__':
    # Run all workers concurrently
    asyncio.run(asyncio.gather(
        worker_loop(ExtractionWorker, 'queue:extraction:high'),
        worker_loop(SemanticWorker, 'queue:semantic:normal'),
        worker_loop(EventWorker, 'queue:event:normal')
    ))
