"""
TCF (Timely-Coherence-Funding) Feed Service

Implements the coherence-first feed algorithm:
- Coherence: 70% weight (entropy reduction, graph centrality)
- Timely: 20% weight (recency with decay)
- Funding: 10% weight (community credits staked)

This is the core differentiator of the platform: we prioritize epistemic
coherence above popularity and funding.
"""

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional
from .neo4j_client import neo4j_client
from .coherence_service import CoherenceService
from .cache import coherence_cache, feed_cache


class TCFFeedService:
    def __init__(self):
        # Use singleton neo4j_client instead of creating new instance
        self.neo4j_client = neo4j_client
        self.coherence_service = CoherenceService()

    def calculate_tcf_score(
        self,
        story_id: str,
        created_at: Optional[datetime] = None,
        funding: float = 0.0
    ) -> Dict:
        """
        Calculate combined TCF score for a story.

        Args:
            story_id: Story UUID
            created_at: When story was created (for timely component)
            funding: Total credits staked on this story

        Returns:
            {
                'tcf_score': float,      # 0-100 combined score
                'coherence': float,      # 0-100 coherence score
                'timely': float,         # 0-100 recency score
                'funding': float,        # 0-100 funding score
                'explanation': str       # Human-readable explanation
            }
        """
        # Check cache first (coherence is expensive to calculate)
        cache_key = f"coherence:{story_id}"
        cached_coherence = coherence_cache.get(cache_key)

        if cached_coherence:
            coherence_data = cached_coherence
            coherence_score = coherence_data['score']
        else:
            # Calculate coherence (70% weight) - EXPENSIVE
            coherence_data = self.coherence_service.calculate_story_coherence(story_id)
            coherence_score = coherence_data['score']
            # Cache for 5 minutes
            coherence_cache.set(cache_key, coherence_data)

        # Calculate timely score (20% weight)
        timely_score = self._calculate_timely_score(created_at) if created_at else 50.0

        # Calculate funding score (10% weight)
        funding_score = self._calculate_funding_score(funding)

        # Weighted combination
        tcf_score = (
            coherence_score * 0.70 +
            timely_score * 0.20 +
            funding_score * 0.10
        )

        return {
            'tcf_score': round(tcf_score, 2),
            'coherence': round(coherence_score, 2),
            'timely': round(timely_score, 2),
            'funding': round(funding_score, 2),
            'coherence_breakdown': coherence_data,
            'explanation': self._generate_explanation(
                tcf_score, coherence_score, timely_score, funding_score
            )
        }

    def get_feed(
        self,
        limit: int = 20,
        offset: int = 0,
        min_coherence: float = 0.0
    ) -> List[Dict]:
        """
        Get stories ranked by update time (most recently updated first).

        Optimized version: Just fetch from DB without recalculating coherence.
        Stories are pre-computed and stored in Neo4j with their metrics.

        Args:
            limit: Number of stories to return
            offset: Number of stories to skip
            min_coherence: Minimum coherence threshold (0-100)

        Returns:
            List of stories sorted by last update time
        """
        # Check feed cache first (cache entire feed for 5 minutes)
        cache_key = f"feed:{limit}:{offset}:{min_coherence}"
        cached_feed = feed_cache.get(cache_key)
        if cached_feed:
            return cached_feed

        if not self.neo4j_client.connected:
            return []

        # Fetch stories directly from Neo4j with their stored metrics
        # Fetch extra stories to account for coherence filtering, but not too many
        fetch_limit = limit + offset + 10  # Small buffer for filtering
        stories = self.neo4j_client.get_recent_stories(limit=fetch_limit)

        feed_items = []
        for story in stories:
            # Use last_updated for timely score (when story was last updated, not created)
            # This matches storychat behavior
            timestamp = story.get('last_updated') or story.get('created_at')
            story_time = None
            if timestamp:
                try:
                    story_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    pass

            # Use stored coherence if available, otherwise calculate once
            if 'coherence_score' in story and story['coherence_score'] is not None:
                coherence = float(story['coherence_score'])
            else:
                # Calculate once for new stories
                tcf_data = self.calculate_tcf_score(
                    story['id'],
                    created_at=story_time,
                    funding=0.0
                )
                coherence = tcf_data['coherence']

            # Filter by min coherence
            if coherence < min_coherence:
                continue

            # Calculate just timely score (fast)
            timely = self._calculate_timely_score(story_time) if story_time else 50.0

            # Simple weighted score without full coherence recalculation
            tcf_score = round(
                (coherence * 0.70) +
                (timely * 0.20),
                2
            )

            feed_items.append({
                'story_id': story['id'],
                'title': story['title'],
                'description': story['description'],
                'created_at': story.get('created_at'),
                'last_updated': story.get('last_updated'),
                'health_indicator': story.get('health_indicator'),
                'claim_count': story.get('claim_count', 0),
                'people': story.get('people', []),
                'cover_image': story.get('cover_image', ''),
                'tcf_score': tcf_score,
                'coherence': coherence,
                'timely': timely,
                'funding': 0.0,
                'explanation': f"Coherence {round(coherence)}/100 • Recently updated"
            })

        # Keep the order from Neo4j (already sorted by last_updated DESC)
        # This matches storychat's channel behavior - most recently updated first

        # Apply pagination
        result = feed_items[offset:offset + limit]

        # Cache the result for 5 minutes (longer since we're not recalculating)
        feed_cache.set(cache_key, result, ttl=300)

        return result

    def _calculate_timely_score(self, created_at: datetime) -> float:
        """
        Calculate recency score with exponential decay.

        Decay parameters:
        - Events: 24 hours (breaking news)
        - Stories: 168 hours = 7 days (investigations)
        - Default: 72 hours = 3 days

        Score formula: 100 * exp(-hours_old / decay_hours)

        Returns: 0-100 score
        """
        if not created_at:
            return 50.0  # Neutral score if no date

        # Ensure timezone-aware
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        hours_old = (now - created_at).total_seconds() / 3600

        # Default decay: 72 hours (3 days)
        decay_hours = 72

        # Calculate exponential decay
        score = 100 * math.exp(-hours_old / decay_hours)

        return min(score, 100.0)

    def _calculate_funding_score(self, funding: float) -> float:
        """
        Calculate funding score based on credits staked.

        Uses logarithmic scale to prevent whale domination:
        score = 100 * log(1 + funding) / log(1 + max_funding)

        Assumes max_funding = 10,000 credits for normalization.

        Returns: 0-100 score
        """
        if funding <= 0:
            return 0.0

        max_funding = 10000.0  # Reference max
        score = 100 * math.log(1 + funding) / math.log(1 + max_funding)

        return min(score, 100.0)

    def _generate_explanation(
        self,
        tcf_score: float,
        coherence: float,
        timely: float,
        funding: float
    ) -> str:
        """
        Generate human-readable explanation for why this story ranks where it does.

        This is shown as "Why this ranks #X" on the frontend.
        """
        reasons = []

        # Coherence is the primary driver
        if coherence > 70:
            reasons.append(f"High coherence ({coherence:.0f}/100)")
        elif coherence > 40:
            reasons.append(f"Moderate coherence ({coherence:.0f}/100)")
        else:
            reasons.append(f"Low coherence ({coherence:.0f}/100)")

        # Timely component
        if timely > 80:
            reasons.append("Very recent")
        elif timely > 50:
            reasons.append("Recent")

        # Funding component
        if funding > 50:
            reasons.append(f"Community backed ({funding:.0f}/100)")

        return " • ".join(reasons)

    def get_story_with_tcf(self, story_id: str) -> Optional[Dict]:
        """
        Get a single story with its TCF score.

        Useful for detail pages showing ranking explanation.
        """
        with self.neo4j_client.driver.session(database=self.neo4j_client.database) as session:
            result = session.run('''
                MATCH (s:Story {id: $story_id})
                RETURN s.id as id,
                       coalesce(s.title, s.topic) as title,
                       s.gist as description,
                       s.created_at as created_at,
                       s.content as content
            ''', story_id=story_id)

            record = result.single()
            if not record:
                return None

            # Parse created_at
            created_at = None
            if record['created_at']:
                try:
                    created_at = datetime.fromisoformat(record['created_at'].replace('Z', '+00:00'))
                except:
                    pass

            # Calculate TCF
            tcf_data = self.calculate_tcf_score(story_id, created_at=created_at)

            return {
                'story_id': story_id,
                'title': record['title'],
                'description': record['description'],
                'content': record['content'],
                'created_at': record['created_at'],
                **tcf_data
            }

    def close(self):
        """Close connections."""
        if self.coherence_service:
            self.coherence_service.close()
        if self.neo4j_client:
            self.neo4j_client.close()
