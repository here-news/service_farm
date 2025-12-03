"""
General Event Attachment Logic

Multi-signal scoring to decide: attach, spawn new, or create relationship

Principles:
1. Multiple independent signals (not just embedding)
2. Hierarchical matching (try micro, meso, macro)
3. Relationship detection (PART_OF, CAUSED_BY, SIMILAR_TO, etc.)
4. Bayesian decision framework
"""
import re
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime, timedelta
import math


class EventAttachmentScorer:
    """
    Score how well a page/claims fit with existing events

    Uses multi-signal approach:
    - Semantic similarity (embedding)
    - Entity overlap (participants)
    - Temporal proximity (time windows)
    - Topic coherence (claim-level similarity)
    """

    def __init__(self):
        # Signal weights (sum to 1.0)
        self.weights = {
            'semantic': 0.35,
            'entity': 0.30,
            'temporal': 0.20,
            'topic': 0.15
        }

        # Thresholds (empirically tuned for real-world event coverage)
        self.attach_threshold = 0.40  # Combined score to attach
        self.relationship_threshold = 0.28  # Create relationship instead

    def score_page_to_event(
        self,
        page_embedding: List[float],
        page_entities: Set[str],
        page_time: datetime,
        page_claims: List[Dict],
        event: Dict,
        event_entities: Set[str],
        event_claims: List[Dict]
    ) -> Dict:
        """
        Score how well page fits with event

        Returns:
            {
                'total_score': float,
                'signals': {...},
                'decision': 'attach' | 'spawn' | 'relate',
                'relationship_type': 'PART_OF' | 'CAUSED_BY' | 'SIMILAR_TO' | None,
                'attachment_level': 'micro' | 'meso' | 'macro' | None,
                'confidence': float
            }
        """
        signals = {}

        # 1. Semantic similarity (embedding)
        signals['semantic'] = self._semantic_similarity(page_embedding, event.get('embedding'))

        # 2. Entity overlap (Jaccard)
        signals['entity'] = self._entity_overlap(page_entities, event_entities)

        # 3. Temporal proximity
        signals['temporal'] = self._temporal_proximity(page_time, event.get('event_start'), event.get('event_end'))

        # 4. Topic coherence (claim-level)
        signals['topic'] = self._topic_coherence(page_claims, event_claims)

        # Combined score
        total_score = sum(
            signals[signal] * self.weights[signal]
            for signal in signals
        )

        # Detect relationship type from claims
        relationship_type = self._detect_relationship_type(page_claims, event)

        # Decision logic
        if total_score >= self.attach_threshold:
            decision = 'attach'
            attachment_level = self._determine_attachment_level(signals, event.get('event_scale'))
            confidence = total_score
        elif total_score >= self.relationship_threshold or relationship_type:
            decision = 'relate'
            # Use detected type or default to RELATED_TO
            if not relationship_type:
                relationship_type = 'RELATED_TO'
            attachment_level = None
            confidence = total_score * 0.8
        else:
            decision = 'spawn'
            attachment_level = None
            confidence = 1.0 - total_score

        return {
            'total_score': total_score,
            'signals': signals,
            'decision': decision,
            'relationship_type': relationship_type,
            'attachment_level': attachment_level,
            'confidence': confidence,
            'rationale': self._generate_rationale(signals, decision, relationship_type)
        }

    def _parse_embedding(self, emb) -> Optional[List[float]]:
        """Parse embedding from various formats (list or pgvector string)"""
        if not emb:
            return None

        # Already a list
        if isinstance(emb, list):
            return emb

        # String format: '[0.1,0.2,...]'
        if isinstance(emb, str):
            try:
                if emb.startswith('[') and emb.endswith(']'):
                    return [float(x.strip()) for x in emb[1:-1].split(',')]
            except:
                return None

        return None

    def _semantic_similarity(self, embedding1: Optional[List[float]], embedding2: Optional[List[float]]) -> float:
        """Cosine similarity between embeddings"""
        # Parse both embeddings
        emb1 = self._parse_embedding(embedding1)
        emb2 = self._parse_embedding(embedding2)

        if not emb1 or not emb2:
            return 0.0

        if len(emb1) != len(emb2):
            return 0.0

        # Simple cosine similarity
        dot = sum(a * b for a, b in zip(emb1, emb2))
        mag1 = math.sqrt(sum(a * a for a in emb1))
        mag2 = math.sqrt(sum(b * b for b in emb2))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot / (mag1 * mag2)

    def _entity_overlap(self, entities1: Set[str], entities2: Set[str]) -> float:
        """Jaccard similarity of entity sets"""
        if not entities1 or not entities2:
            return 0.0

        intersection = len(entities1 & entities2)
        union = len(entities1 | entities2)

        if union == 0:
            return 0.0

        return intersection / union

    def _temporal_proximity(
        self,
        page_time: Optional[datetime],
        event_start: Optional[datetime],
        event_end: Optional[datetime]
    ) -> float:
        """Score temporal overlap/proximity"""
        if not page_time:
            return 0.5  # Neutral if no time

        if not event_start:
            return 0.5

        # If page time is within event bounds, high score
        event_end_actual = event_end or event_start
        if event_start <= page_time <= event_end_actual:
            return 1.0

        # If outside, decay by days
        if page_time < event_start:
            days_diff = (event_start - page_time).days
        else:
            days_diff = (page_time - event_end_actual).days

        # Exponential decay: 0.9 at 1 day, 0.5 at 7 days, 0.1 at 30 days
        return math.exp(-days_diff / 10.0)

    def _topic_coherence(self, claims1: List[Dict], claims2: List[Dict]) -> float:
        """
        Score how similar the topics/themes are

        Uses keyword overlap in claims
        """
        if not claims1 or not claims2:
            return 0.5

        # Extract keywords from claims
        keywords1 = self._extract_keywords([c['text'] for c in claims1])
        keywords2 = self._extract_keywords([c['text'] for c in claims2])

        if not keywords1 or not keywords2:
            return 0.5

        # Jaccard on keywords
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)

        return intersection / union if union > 0 else 0.0

    def _extract_keywords(self, texts: List[str]) -> Set[str]:
        """Extract significant keywords from texts"""
        # Simple approach: lowercase words, filter stopwords, > 3 chars
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                     'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}

        keywords = set()
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            keywords.update(
                word for word in words
                if len(word) > 3 and word not in stopwords
            )

        return keywords

    def _detect_relationship_type(self, claims: List[Dict], event: Dict) -> Optional[str]:
        """
        Detect if claims create a relationship rather than direct attachment

        Relationship types:
        - CAUSED_BY: causal language
        - SIMILAR_TO: comparison language
        - PHASE_OF: temporal progression language
        - EXTENDS: adds new dimension
        """
        combined_text = ' '.join(c['text'].lower() for c in claims)

        # Causal indicators
        causal_patterns = [
            r'caused? by', r'due to', r'because of', r'resulted? (?:in|from)',
            r'led to', r'attributed to', r'contributed to', r'sparked by'
        ]
        if any(re.search(pattern, combined_text) for pattern in causal_patterns):
            return 'CAUSED_BY'

        # Comparison indicators
        comparison_patterns = [
            r'similar to', r'like the', r'reminiscent of', r'echoes? of',
            r'comparable to', r'parallels? with', r'resembles?'
        ]
        if any(re.search(pattern, combined_text) for pattern in comparison_patterns):
            return 'SIMILAR_TO'

        # Phase/temporal progression and responses
        phase_patterns = [
            r'following', r'after', r'subsequently', r'then', r'next',
            r'in the wake of', r'aftermath', r'response to', r'in response',
            r'reacting to', r'regarding', r'about the', r'concerning'
        ]
        if any(re.search(pattern, combined_text) for pattern in phase_patterns):
            return 'PHASE_OF'

        # Condolence/sympathy messages
        condolence_patterns = [
            r'condolences?', r'sympathy', r'saddened', r'heartfelt',
            r'thoughts (?:are|and prayers)', r'prayers?', r'mourn'
        ]
        if any(re.search(pattern, combined_text) for pattern in condolence_patterns):
            return 'RESPONSE_TO'

        # Investigation/explanation (EXTENDS)
        extends_patterns = [
            r'investigation', r'analysis', r'report', r'findings',
            r'experts? (?:say|found|discovered)', r'according to'
        ]
        if any(re.search(pattern, combined_text) for pattern in extends_patterns):
            # Check if this is about WHY/HOW (extends understanding)
            if 'why' in combined_text or 'how' in combined_text:
                return 'EXTENDS'

        return None

    def _determine_attachment_level(self, signals: Dict[str, float], event_scale: str) -> str:
        """
        Decide whether to attach at micro, meso, or macro level

        High topic coherence → attach to specific micro event
        High entity/time overlap but lower topic → attach to meso/macro
        """
        topic_score = signals.get('topic', 0.0)
        entity_score = signals.get('entity', 0.0)

        if topic_score >= 0.7:
            # Very specific topic match → micro level
            return 'micro'
        elif entity_score >= 0.6 or event_scale in ('meso', 'macro'):
            # General participant overlap → parent level
            return 'meso'
        else:
            # Default: attach to same level
            return event_scale or 'micro'

    def _generate_rationale(
        self,
        signals: Dict[str, float],
        decision: str,
        relationship_type: Optional[str]
    ) -> str:
        """Generate human-readable rationale for decision"""
        parts = []

        # Highlight strongest signals
        sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
        for signal, score in sorted_signals[:2]:
            if score > 0.5:
                parts.append(f"{signal}={score:.2f}")

        if relationship_type:
            parts.append(f"relationship={relationship_type}")

        return f"{decision}: {', '.join(parts)}"


def find_best_event_match(
    page: Dict,
    page_entities: Set[str],
    page_claims: List[Dict],
    candidate_events: List[Dict],
    scorer: EventAttachmentScorer
) -> Tuple[Optional[Dict], Dict]:
    """
    Find best matching event from candidates

    Returns:
        (best_event, score_details)
    """
    best_event = None
    best_score_details = None
    best_total_score = 0.0

    for event in candidate_events:
        # Get event entities
        event_entities = set(event.get('entity_names', []))

        # Get event claims (if available)
        event_claims = event.get('claims', [])

        # Score this event
        score_details = scorer.score_page_to_event(
            page_embedding=page.get('embedding'),
            page_entities=page_entities,
            page_time=page.get('pub_time'),
            page_claims=page_claims,
            event=event,
            event_entities=event_entities,
            event_claims=event_claims
        )

        # Track best
        if score_details['total_score'] > best_total_score:
            best_total_score = score_details['total_score']
            best_event = event
            best_score_details = score_details

    return best_event, best_score_details
