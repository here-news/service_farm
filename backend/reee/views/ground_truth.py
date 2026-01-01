"""
Ground Truth Generation
=======================

Generates incident-level ground truth from legacy case-level events.

Legacy events in the database are often case/storyline level (e.g., "Do Kwon Sentencing"
spans 2018-2025). For proper evaluation of incident-level clustering, we need to
split these into incident-level chunks.

Splitting heuristics:
1. Time gaps: Claims >N days apart → different incidents
2. Anchor shifts: Different primary anchors → different incidents
3. Manual overrides: Known case-level events to split

The output is a mapping: claim_id → incident_label
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict


@dataclass
class IncidentGT:
    """An incident-level ground truth cluster."""
    id: str
    legacy_event_id: str
    legacy_event_name: str
    claim_ids: Set[str]
    primary_anchors: Set[str]
    time_window: Tuple[Optional[datetime], Optional[datetime]]

    @property
    def duration_days(self) -> Optional[int]:
        t_start, t_end = self.time_window
        if t_start and t_end:
            return (t_end - t_start).days
        return None


@dataclass
class GTSplitResult:
    """Result of splitting legacy events into incidents."""
    incidents: Dict[str, IncidentGT]  # incident_id -> IncidentGT
    claim_to_incident: Dict[str, str]  # claim_id -> incident_id
    legacy_to_incidents: Dict[str, List[str]]  # legacy_event_id -> [incident_ids]

    # Statistics
    legacy_events_processed: int = 0
    incidents_created: int = 0
    claims_labeled: int = 0
    splits_by_time: int = 0
    splits_by_anchor: int = 0


def split_legacy_events_to_incidents(
    legacy_events: List[Dict],
    time_gap_days: int = 7,
    min_anchor_overlap: float = 0.3,
    anchor_salience_top_k: int = 3,
) -> GTSplitResult:
    """
    Split legacy case-level events into incident-level ground truth.

    Splitting rules (must satisfy BOTH to split):
    1. Time gap: Claims >time_gap_days apart
    2. Anchor shift: Low overlap (<min_anchor_overlap) of salient anchors

    Only splits when there's clear evidence of different incidents,
    not just different aspects of the same incident.

    Args:
        legacy_events: List of dicts with keys:
            - id: legacy event ID
            - name: legacy event name
            - claims: List of dicts with {id, text, timestamp, anchors}
        time_gap_days: Max days between claims in same incident
        min_anchor_overlap: Split only if overlap < this (Jaccard)
        anchor_salience_top_k: Only consider top-k anchors by frequency

    Returns:
        GTSplitResult with incident-level labels
    """
    # Publisher/context entities to exclude from anchor matching
    EXCLUDE_ANCHORS = {
        'Time', 'TIME', 'New York Times', 'Washington Post', 'BBC',
        'CNN', 'Fox News', 'Reuters', 'AP', 'Associated Press',
        'The Guardian', 'Daily Mail', 'SCMP', 'South China Morning Post',
        'Hong Kong', 'China', 'United States', 'US', 'UK', 'United Kingdom',
        'Beijing', 'New York', 'London', 'Washington',
    }

    def filter_anchors(anchors: Set[str]) -> Set[str]:
        """Remove publisher/context anchors."""
        return {a for a in anchors if a not in EXCLUDE_ANCHORS}

    def compute_anchor_overlap(anchors1: Set[str], anchors2: Set[str]) -> float:
        """Compute Jaccard overlap of filtered anchors."""
        a1 = filter_anchors(anchors1)
        a2 = filter_anchors(anchors2)
        if not a1 or not a2:
            return 1.0  # Don't split if no meaningful anchors
        intersection = len(a1 & a2)
        union = len(a1 | a2)
        return intersection / union if union > 0 else 0.0

    def get_salient_anchors(claims: List[Dict], top_k: int) -> Set[str]:
        """Get top-k most frequent anchors across claims."""
        anchor_counts: Dict[str, int] = defaultdict(int)
        for c in claims:
            for a in filter_anchors(set(c.get('anchors', []))):
                anchor_counts[a] += 1
        sorted_anchors = sorted(anchor_counts.items(), key=lambda x: -x[1])
        return {a for a, _ in sorted_anchors[:top_k]}

    incidents = {}
    claim_to_incident = {}
    legacy_to_incidents = defaultdict(list)
    splits_by_time = 0
    splits_by_anchor = 0

    for legacy_event in legacy_events:
        legacy_id = legacy_event['id']
        legacy_name = legacy_event['name']
        claims = legacy_event.get('claims', [])

        if not claims:
            continue

        # Sort claims by timestamp
        def safe_timestamp(ts):
            if ts is None:
                return datetime.max
            if ts.tzinfo is not None:
                return ts.replace(tzinfo=None)
            return ts

        claims_with_time = [
            (c, c.get('timestamp')) for c in claims
        ]
        claims_with_time.sort(key=lambda x: safe_timestamp(x[1]))

        # Split into incidents using sliding window approach
        current_incident_claims = []
        current_incident_start = None
        current_incident_end = None
        incident_counter = 0

        def flush_incident():
            nonlocal incident_counter

            if not current_incident_claims:
                return

            incident_id = f"{legacy_id}_inc{incident_counter:02d}"
            incident_counter += 1

            claim_ids = {c['id'] for c in current_incident_claims}
            salient_anchors = get_salient_anchors(current_incident_claims, anchor_salience_top_k)

            incidents[incident_id] = IncidentGT(
                id=incident_id,
                legacy_event_id=legacy_id,
                legacy_event_name=legacy_name,
                claim_ids=claim_ids,
                primary_anchors=salient_anchors,
                time_window=(current_incident_start, current_incident_end),
            )

            for claim in current_incident_claims:
                claim_to_incident[claim['id']] = incident_id

            legacy_to_incidents[legacy_id].append(incident_id)

        for claim, timestamp in claims_with_time:
            claim_anchors = set(claim.get('anchors', []))

            should_split = False
            split_reason = None

            if current_incident_claims:
                # Rule 1: Time gap check
                time_gap_exceeded = False
                if timestamp and current_incident_end:
                    gap = abs((safe_timestamp(timestamp) - safe_timestamp(current_incident_end)).days)
                    if gap > time_gap_days:
                        time_gap_exceeded = True

                # Rule 2: Anchor overlap check (using salient anchors)
                low_anchor_overlap = False
                if time_gap_exceeded:
                    # Only check anchor overlap if time gap is exceeded
                    current_salient = get_salient_anchors(current_incident_claims, anchor_salience_top_k)
                    new_claim_anchors = filter_anchors(claim_anchors)

                    if current_salient and new_claim_anchors:
                        overlap = len(current_salient & new_claim_anchors) / len(current_salient)
                        if overlap < min_anchor_overlap:
                            low_anchor_overlap = True

                # Split only if BOTH conditions met
                if time_gap_exceeded and low_anchor_overlap:
                    should_split = True
                    split_reason = 'time_gap_and_anchor_shift'
                    splits_by_time += 1
                    splits_by_anchor += 1
                elif time_gap_exceeded and not low_anchor_overlap:
                    # Time gap but same anchors - could be continuation
                    # Only split if gap is very large (>30 days)
                    if timestamp and current_incident_end:
                        gap = abs((safe_timestamp(timestamp) - safe_timestamp(current_incident_end)).days)
                        if gap > 30:
                            should_split = True
                            split_reason = 'large_time_gap'
                            splits_by_time += 1

            if should_split:
                flush_incident()
                current_incident_claims = []
                current_incident_start = None
                current_incident_end = None

            # Add claim to current incident
            current_incident_claims.append(claim)

            if timestamp:
                ts = safe_timestamp(timestamp)
                if current_incident_start is None:
                    current_incident_start = timestamp
                current_incident_end = timestamp

        # Flush final incident
        flush_incident()

    return GTSplitResult(
        incidents=incidents,
        claim_to_incident=claim_to_incident,
        legacy_to_incidents=dict(legacy_to_incidents),
        legacy_events_processed=len(legacy_events),
        incidents_created=len(incidents),
        claims_labeled=len(claim_to_incident),
        splits_by_time=splits_by_time,
        splits_by_anchor=splits_by_anchor,
    )


async def load_legacy_events_for_gt(ctx, limit: int = 50) -> List[Dict]:
    """
    Load legacy events with claims for ground truth generation.

    Returns list of event dicts with claims including timestamps and anchors.
    """
    # Get events with their claims
    events_data = await ctx.neo4j._execute_read('''
        MATCH (e:Event)-[:INTAKES]->(c:Claim)
        OPTIONAL MATCH (p:Page)-[:EMITS]->(c)
        OPTIONAL MATCH (c)-[:MENTIONS]->(ent:Entity)
        WHERE ent.entity_type IN ['PERSON', 'ORGANIZATION', 'ORG']
           OR (ent.entity_type = 'LOCATION' AND NOT ent.canonical_name IN
               ['Hong Kong', 'China', 'United States', 'US', 'UK', 'United Kingdom'])
        WITH e, c, p, collect(DISTINCT ent.canonical_name) as anchors
        RETURN e.id as event_id, e.canonical_name as event_name,
               c.id as claim_id, c.text as claim_text,
               p.pub_time as pub_time, anchors
        ORDER BY e.id, p.pub_time
        LIMIT $limit
    ''', {'limit': limit * 20})  # Over-fetch since we group by event

    # Group by event
    events_by_id = defaultdict(lambda: {'claims': []})
    for row in events_data:
        event_id = row['event_id']
        events_by_id[event_id]['id'] = event_id
        events_by_id[event_id]['name'] = row['event_name']

        timestamp = None
        pub_time = row.get('pub_time')
        if pub_time:
            try:
                if isinstance(pub_time, str):
                    timestamp = datetime.fromisoformat(pub_time.replace('Z', '+00:00'))
                else:
                    timestamp = pub_time
            except:
                pass

        events_by_id[event_id]['claims'].append({
            'id': row['claim_id'],
            'text': row['claim_text'],
            'timestamp': timestamp,
            'anchors': row['anchors'] or [],
        })

    return list(events_by_id.values())[:limit]


def print_gt_report(result: GTSplitResult):
    """Print a report of ground truth generation."""
    print("=" * 70)
    print("INCIDENT GROUND TRUTH GENERATION")
    print("=" * 70)
    print(f"Legacy events processed: {result.legacy_events_processed}")
    print(f"Incidents created: {result.incidents_created}")
    print(f"Claims labeled: {result.claims_labeled}")
    print(f"Splits by time gap: {result.splits_by_time}")
    print(f"Splits by anchor shift: {result.splits_by_anchor}")
    print()

    # Show split events
    multi_incident_legacies = [
        (legacy_id, inc_ids)
        for legacy_id, inc_ids in result.legacy_to_incidents.items()
        if len(inc_ids) > 1
    ]

    if multi_incident_legacies:
        print("Events split into multiple incidents:")
        for legacy_id, inc_ids in multi_incident_legacies:
            inc = result.incidents[inc_ids[0]]
            print(f"\n  {inc.legacy_event_name} → {len(inc_ids)} incidents")
            for inc_id in inc_ids:
                inc = result.incidents[inc_id]
                duration = inc.duration_days
                duration_str = f"{duration} days" if duration is not None else "unknown"
                print(f"    - {inc_id}: {len(inc.claim_ids)} claims, {duration_str}")
                print(f"      Anchors: {list(inc.primary_anchors)[:3]}")
    else:
        print("No events were split (all were already incident-level)")
