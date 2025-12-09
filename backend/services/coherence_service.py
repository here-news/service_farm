"""
Coherence calculation service for the coherence-first platform.

Calculates epistemic coherence scores based on:
1. Entity overlap with other stories (50% weight)
2. Graph centrality of mentioned entities (30% weight)
3. Claim density (20% weight)

The coherence score measures how well a story connects to the existing
knowledge graph and contributes to reducing overall uncertainty.
"""

from typing import Dict, List, Optional
from .neo4j_client import neo4j_client


class CoherenceService:
    def __init__(self):
        # Use singleton neo4j_client instead of creating new instance
        self.neo4j_client = neo4j_client

    def calculate_story_coherence(self, story_id: str) -> Dict:
        """
        Calculate comprehensive coherence score for a story.

        Returns:
            {
                'score': float,           # 0-100 overall coherence score
                'entity_overlap': float,  # 0-100 how many entities are shared
                'centrality': float,      # 0-100 avg importance of entities
                'claim_density': float,   # 0-100 claim richness
                'breakdown': {
                    'shared_entities': int,
                    'total_entities': int,
                    'avg_entity_stories': float,
                    'claim_count': int
                }
            }
        """
        if not self.neo4j_client.connected:
            return self._default_score()

        # Calculate components
        entity_overlap_score, entity_breakdown = self._calculate_entity_overlap(story_id)
        centrality_score, centrality_breakdown = self._calculate_graph_centrality(story_id)
        claim_density_score, claim_breakdown = self._calculate_claim_density(story_id)

        # Weighted combination (aligned with TCF algorithm weights)
        coherence_score = (
            entity_overlap_score * 0.50 +
            centrality_score * 0.30 +
            claim_density_score * 0.20
        )

        return {
            'score': round(coherence_score, 2),
            'entity_overlap': round(entity_overlap_score, 2),
            'centrality': round(centrality_score, 2),
            'claim_density': round(claim_density_score, 2),
            'breakdown': {
                **entity_breakdown,
                **centrality_breakdown,
                **claim_breakdown
            }
        }

    def _calculate_entity_overlap(self, story_id: str) -> tuple[float, Dict]:
        """
        Calculate how many entities this story shares with other stories.
        Higher overlap = more coherent with existing knowledge.

        Returns: (score: 0-100, breakdown: dict)
        """
        with self.neo4j_client.driver.session(database=self.neo4j_client.database) as session:
            # Get entities mentioned in this story
            result = session.run('''
                MATCH (s:Story)-[:MENTIONS_PERSON|MENTIONS_ORG|MENTIONS_LOCATION]->(e)
                WHERE s.id = $story_id
                RETURN count(DISTINCT e) as entity_count
            ''', story_id=story_id)

            record = result.single()
            total_entities = record['entity_count'] if record else 0

            if total_entities == 0:
                return 0.0, {'shared_entities': 0, 'total_entities': 0}

            # Find entities shared with other stories
            result = session.run('''
                MATCH (s1:Story)-[:MENTIONS_PERSON|MENTIONS_ORG|MENTIONS_LOCATION]->(e)
                      <-[:MENTIONS_PERSON|MENTIONS_ORG|MENTIONS_LOCATION]-(s2:Story)
                WHERE s1.id = $story_id AND s1 <> s2
                RETURN count(DISTINCT e) as shared_entities
            ''', story_id=story_id)

            record = result.single()
            shared_entities = record['shared_entities'] if record else 0

            # Score: percentage of entities that are shared
            overlap_ratio = shared_entities / total_entities if total_entities > 0 else 0

            # Scale to 0-100
            score = min(overlap_ratio * 100, 100)

            return score, {
                'shared_entities': shared_entities,
                'total_entities': total_entities
            }

    def _calculate_graph_centrality(self, story_id: str) -> tuple[float, Dict]:
        """
        Calculate average centrality of entities mentioned in this story.
        Centrality = how many stories mention each entity.
        Higher centrality = entities are important hubs in the knowledge graph.

        Returns: (score: 0-100, breakdown: dict)
        """
        with self.neo4j_client.driver.session(database=self.neo4j_client.database) as session:
            # Get centrality for each entity mentioned in story
            result = session.run('''
                MATCH (s:Story)-[:MENTIONS_PERSON|MENTIONS_ORG|MENTIONS_LOCATION]->(e)
                WHERE s.id = $story_id
                WITH DISTINCT e

                // Count how many stories mention this entity
                MATCH (e)<-[:MENTIONS_PERSON|MENTIONS_ORG|MENTIONS_LOCATION]-(story:Story)
                WITH e, count(DISTINCT story) as story_count

                // Get total number of stories for normalization
                MATCH (all_stories:Story)
                WITH e, story_count, count(DISTINCT all_stories) as total_stories

                // Centrality = what fraction of all stories mention this entity
                RETURN avg(story_count * 1.0 / total_stories) as avg_centrality,
                       avg(story_count) as avg_story_count
            ''', story_id=story_id)

            record = result.single()

            if not record or record['avg_centrality'] is None:
                return 0.0, {'avg_entity_stories': 0.0}

            avg_centrality = record['avg_centrality']
            avg_story_count = record['avg_story_count']

            # Scale to 0-100 (assuming max centrality is 0.25 = entity in 25% of stories)
            score = min(avg_centrality * 400, 100)

            return score, {
                'avg_entity_stories': round(avg_story_count, 2)
            }

    def _calculate_claim_density(self, story_id: str) -> tuple[float, Dict]:
        """
        Calculate claim density for this story.
        More claims = more epistemic value.

        Note: Claims are connected via Page artifacts:
        Story → HAS_ARTIFACT → Page → HAS_CLAIM → Claim

        Returns: (score: 0-100, breakdown: dict)
        """
        with self.neo4j_client.driver.session(database=self.neo4j_client.database) as session:
            # Get claim count for this story (through Page artifacts)
            result = session.run('''
                MATCH (s:Story)-[:HAS_ARTIFACT]->(p:Page)-[:HAS_CLAIM]->(c:Claim)
                WHERE s.id = $story_id
                RETURN count(DISTINCT c) as claim_count
            ''', story_id=story_id)

            record = result.single()
            claim_count = record['claim_count'] if record else 0

            # Get max claims for normalization
            result = session.run('''
                MATCH (s:Story)-[:HAS_ARTIFACT]->(p:Page)-[:HAS_CLAIM]->(c:Claim)
                WITH s, count(DISTINCT c) as claims
                RETURN max(claims) as max_claims
            ''')

            record = result.single()
            max_claims = record['max_claims'] if record and record['max_claims'] else 106

            # Score: percentage of max claims
            score = min((claim_count / max_claims) * 100, 100) if max_claims > 0 else 0

            return score, {
                'claim_count': claim_count
            }

    def _default_score(self) -> Dict:
        """Return default score when Neo4j is not connected."""
        return {
            'score': 0.0,
            'entity_overlap': 0.0,
            'centrality': 0.0,
            'claim_density': 0.0,
            'breakdown': {
                'shared_entities': 0,
                'total_entities': 0,
                'avg_entity_stories': 0.0,
                'claim_count': 0
            }
        }

    def get_coherence_rankings(self, story_ids: List[str]) -> List[Dict]:
        """
        Calculate coherence scores for multiple stories and rank them.

        Args:
            story_ids: List of story UUIDs

        Returns:
            List of stories with coherence scores, sorted by score descending
        """
        results = []

        for story_id in story_ids:
            coherence = self.calculate_story_coherence(story_id)
            results.append({
                'story_id': story_id,
                **coherence
            })

        # Sort by coherence score (highest first)
        results.sort(key=lambda x: x['score'], reverse=True)

        return results

    def explain_coherence(self, story_id: str) -> str:
        """
        Generate human-readable explanation of why this story has its coherence score.

        Returns:
            String explanation for "Why this ranks #X" display
        """
        coherence = self.calculate_story_coherence(story_id)
        score = coherence['score']
        breakdown = coherence['breakdown']

        explanations = []

        # Entity overlap explanation
        if breakdown['shared_entities'] > 0:
            explanations.append(
                f"Connects to {breakdown['shared_entities']} entities across other stories"
            )

        # Centrality explanation
        if breakdown['avg_entity_stories'] > 1:
            explanations.append(
                f"Mentions entities appearing in avg {breakdown['avg_entity_stories']:.1f} stories"
            )

        # Claim density explanation
        if breakdown['claim_count'] > 0:
            explanations.append(
                f"Contains {breakdown['claim_count']} verifiable claims"
            )

        if not explanations:
            return "New story with limited graph connections"

        return " • ".join(explanations)

    def close(self):
        """Close Neo4j connection."""
        if self.neo4j_client:
            self.neo4j_client.close()
