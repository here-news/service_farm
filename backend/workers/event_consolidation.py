"""
Event Consolidation and Bayesian Scoring

Implements synthesis, contradiction tracking, and probabilistic event formation
per /tmp/event_emergent_horizon_design.md
"""
import math
import re
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime
from collections import defaultdict


def compute_log_prior(event_stats: Dict) -> float:
    """
    Compute log prior P(H) for event hypothesis

    Based on: quality_overall, recency, participant centrality, scale fit
    """
    base_prior = -2.0  # log(0.135) ≈ -2.0, modest starting prior

    # Quality boost
    quality = event_stats.get('confidence', 0.5)
    quality_boost = (quality - 0.5) * 2.0  # ±1.0 range

    # Recency boost (events in last 7 days get boost)
    recency_boost = 0.0
    if event_stats.get('event_start'):
        days_ago = (datetime.now(event_stats['event_start'].tzinfo) - event_stats['event_start']).days
        if days_ago <= 7:
            recency_boost = 0.5
        elif days_ago <= 30:
            recency_boost = 0.2

    # Participant centrality (more entities = higher prior)
    entity_count = event_stats.get('entity_count', 0)
    centrality_boost = math.log1p(entity_count) * 0.3  # log scale

    # Scale fit (regional > local > micro)
    scale_boost = {'micro': 0.0, 'local': 0.3, 'regional': 0.6, 'macro': 0.9}.get(
        event_stats.get('scale_type', 'micro'), 0.0
    )

    log_prior = base_prior + quality_boost + recency_boost + centrality_boost + scale_boost

    return log_prior


def compute_log_likelihood(claim: Dict, event: Dict, shared_entities: int, total_claim_entities: int) -> float:
    """
    Compute log likelihood P(claim | event_hypothesis)

    Factors: time_match * entity_match * modality_weight * source_weight * coherence_penalty
    """
    # Time match (claims within event temporal bounds get high score)
    time_match_score = 0.0
    if claim.get('event_time') and event.get('event_start') and event.get('event_end'):
        if event['event_start'] <= claim['event_time'] <= event['event_end']:
            time_match_score = 0.5  # Inside bounds
        else:
            # Penalize by distance
            if claim['event_time'] < event['event_start']:
                days_off = (event['event_start'] - claim['event_time']).days
            else:
                days_off = (claim['event_time'] - event['event_end']).days
            time_match_score = -0.1 * min(days_off, 10)

    # Entity match (Jaccard similarity)
    entity_match_score = 0.0
    if total_claim_entities > 0:
        entity_overlap = shared_entities / total_claim_entities
        entity_match_score = math.log1p(entity_overlap * 5)  # 0 to ~1.8

    # Modality weight (higher confidence claims contribute more)
    claim_confidence = claim.get('confidence', 0.5)
    modality_score = math.log(claim_confidence + 0.1)  # Avoid log(0)

    # Source weight (page credibility)
    source_credibility = claim.get('page_credibility', 0.5)
    source_score = math.log(source_credibility + 0.1)

    # Coherence penalty (applied later when contradictions detected)
    coherence_score = 0.0  # Neutral for now

    log_likelihood = time_match_score + entity_match_score + modality_score + source_score + coherence_score

    return log_likelihood


def consolidate_claims(claims: List[Dict], entities: List[Dict]) -> Dict:
    """
    Consolidate claims into structured summary with whole picture

    Returns:
        - Temporal timeline (key events with timestamps)
        - Casualty tracking (death toll range, hospitalization)
        - Key participants (who, roles)
        - Locations (where)
        - Narrative phases
        - Contradictions
    """
    consolidated = {
        'timeline': [],
        'casualties': {
            'deaths': {'values': [], 'range': None, 'latest': None},
            'injured': {'values': [], 'range': None, 'latest': None},
            'hospitalized': {'values': [], 'range': None, 'latest': None}
        },
        'locations': set(),
        'participants': defaultdict(list),  # role -> [entities]
        'key_facts': [],
        'contradictions': [],
        'phases': []
    }

    # Extract numbers from claims
    death_pattern = re.compile(r'(\d+)\s*(?:people\s+)?(?:were\s+)?(?:killed|deaths?|dead|died)', re.IGNORECASE)
    injured_pattern = re.compile(r'(\d+)\s*(?:people\s+)?(?:were\s+)?(?:injured|hurt)', re.IGNORECASE)
    hospitalized_pattern = re.compile(r'(\d+)\s*(?:people\s+)?(?:were\s+)?hospitalized', re.IGNORECASE)

    for claim in claims:
        text = claim['text']
        event_time = claim.get('event_time')

        # Timeline entry
        if event_time:
            consolidated['timeline'].append({
                'time': event_time,
                'text': text[:200]
            })

        # Extract death toll
        death_match = death_pattern.search(text)
        if death_match:
            count = int(death_match.group(1))
            consolidated['casualties']['deaths']['values'].append({
                'count': count,
                'source': claim.get('page_url', 'unknown'),
                'time': event_time
            })

        # Extract injuries
        injured_match = injured_pattern.search(text)
        if injured_match:
            count = int(injured_match.group(1))
            consolidated['casualties']['injured']['values'].append({
                'count': count,
                'source': claim.get('page_url', 'unknown'),
                'time': event_time
            })

        # Extract hospitalized
        hospitalized_match = hospitalized_pattern.search(text)
        if hospitalized_match:
            count = int(hospitalized_match.group(1))
            consolidated['casualties']['hospitalized']['values'].append({
                'count': count,
                'source': claim.get('page_url', 'unknown'),
                'time': event_time
            })

    # Sort timeline
    consolidated['timeline'].sort(key=lambda x: x['time'] if x['time'] else datetime.min.replace(tzinfo=None))

    # Calculate casualty ranges
    if consolidated['casualties']['deaths']['values']:
        death_counts = [v['count'] for v in consolidated['casualties']['deaths']['values']]
        consolidated['casualties']['deaths']['range'] = (min(death_counts), max(death_counts))
        # Latest by time
        latest = max(consolidated['casualties']['deaths']['values'],
                    key=lambda x: x['time'] if x['time'] else datetime.min.replace(tzinfo=None))
        consolidated['casualties']['deaths']['latest'] = latest['count']

    if consolidated['casualties']['injured']['values']:
        injured_counts = [v['count'] for v in consolidated['casualties']['injured']['values']]
        consolidated['casualties']['injured']['range'] = (min(injured_counts), max(injured_counts))

    if consolidated['casualties']['hospitalized']['values']:
        hosp_counts = [v['count'] for v in consolidated['casualties']['hospitalized']['values']]
        consolidated['casualties']['hospitalized']['range'] = (min(hosp_counts), max(hosp_counts))

    # Detect contradictions (different death tolls)
    if consolidated['casualties']['deaths']['range']:
        min_deaths, max_deaths = consolidated['casualties']['deaths']['range']
        if max_deaths - min_deaths > 5:  # Significant discrepancy
            consolidated['contradictions'].append({
                'type': 'death_toll',
                'values': consolidated['casualties']['deaths']['values'],
                'severity': 'moderate' if max_deaths - min_deaths < 20 else 'high'
            })

    # Extract locations from entities
    for entity in entities:
        if entity.get('entity_type') in ('LOCATION', 'GPE'):
            consolidated['locations'].add(entity['name'])

    # Extract participants
    for entity in entities:
        if entity.get('entity_type') == 'PERSON':
            # Handle metadata as string or dict
            metadata = entity.get('metadata', {})
            if isinstance(metadata, str):
                import json
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            role = metadata.get('role', 'mentioned') if isinstance(metadata, dict) else 'mentioned'
            consolidated['participants'][role].append(entity['name'])
        elif entity.get('entity_type') == 'ORGANIZATION':
            consolidated['participants']['organizations'].append(entity['name'])

    # Convert sets to lists for JSON serialization
    consolidated['locations'] = list(consolidated['locations'])
    consolidated['participants'] = dict(consolidated['participants'])

    return consolidated


def detect_event_phases(timeline: List[Dict]) -> List[Dict]:
    """
    Detect narrative phases from timeline

    E.g., "Fire outbreak", "Rescue operations", "Investigation"
    """
    phases = []

    # Simple heuristic: group timeline entries by time gaps
    if not timeline:
        return phases

    current_phase = {
        'start': timeline[0]['time'],
        'end': timeline[0]['time'],
        'entries': [timeline[0]]
    }

    for i in range(1, len(timeline)):
        entry = timeline[i]
        prev_time = current_phase['end']
        curr_time = entry['time']

        if curr_time and prev_time:
            gap_hours = (curr_time - prev_time).total_seconds() / 3600

            if gap_hours > 6:  # More than 6 hours apart = new phase
                phases.append(current_phase)
                current_phase = {
                    'start': curr_time,
                    'end': curr_time,
                    'entries': [entry]
                }
            else:
                current_phase['end'] = curr_time
                current_phase['entries'].append(entry)
        else:
            current_phase['entries'].append(entry)

    phases.append(current_phase)

    return phases
