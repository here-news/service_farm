"""
TopologyPersistence - Persistent storage for claim topology graph

Stores and retrieves the epistemic topology for events:
- Claim-to-claim relationships (CORROBORATES, CONTRADICTS, UPDATES)
- Plausibility scores (on SUPPORTS edges)
- Topology metadata (pattern, consensus_date, coherence, temperature)
- Update chains (metric progressions)

All storage is in Neo4j - the topology IS the graph structure.
"""
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from services.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)


@dataclass
class TopologyClaimData:
    """Claim data for topology visualization"""
    id: str
    text: str
    plausibility: float
    prior: float
    is_superseded: bool
    event_time: Optional[str]
    source_type: Optional[str] = None
    corroboration_count: int = 0


@dataclass
class TopologyRelationship:
    """Relationship between two claims"""
    source_id: str
    target_id: str
    rel_type: str  # CORROBORATES, CONTRADICTS, UPDATES
    similarity: Optional[float] = None


@dataclass
class TopologyUpdateChain:
    """A chain of updates for a metric (e.g., death toll progression)"""
    metric: str
    chain: List[Dict]  # [{claim_id, value, plausibility}, ...]
    current_claim_id: str


@dataclass
class TopologyData:
    """Complete topology data for an event"""
    event_id: str
    pattern: str  # consensus, progressive, contradictory, mixed
    consensus_date: Optional[str]
    coherence: float
    temperature: float  # 0=stable, 1=chaotic
    claims: List[TopologyClaimData]
    relationships: List[TopologyRelationship]
    update_chains: List[TopologyUpdateChain]
    contradictions: List[Dict]
    source_diversity: Dict[str, Dict]
    last_updated: datetime


class TopologyPersistence:
    """
    Manages persistent topology storage in Neo4j.

    The topology is stored as:
    - Claim-to-Claim edges: CORROBORATES, CONTRADICTS, UPDATES
    - Event node properties: topology_* fields
    - SUPPORTS edge properties: plausibility, prior, is_superseded
    """

    def __init__(self, neo4j_service: Neo4jService):
        self.neo4j = neo4j_service

    # =========================================================================
    # STORE OPERATIONS
    # =========================================================================

    async def store_topology(
        self,
        event_id: str,
        topology_result,  # TopologyResult from ClaimTopologyService
        source_diversity: Dict[str, Dict] = None
    ) -> None:
        """
        Persist full topology result to Neo4j.

        Args:
            event_id: Event ID
            topology_result: TopologyResult from ClaimTopologyService.analyze()
            source_diversity: Optional source diversity stats
        """
        logger.info(f"💾 Persisting topology for event {event_id}")

        # 1. Store topology metadata on Event node
        await self._store_topology_metadata(
            event_id,
            pattern=topology_result.pattern,
            consensus_date=topology_result.consensus_date,
            contradictions_count=len(topology_result.contradictions),
            superseded_count=len(topology_result.superseded_by),
            source_diversity=source_diversity
        )

        # 2. Store plausibilities on SUPPORTS edges
        for claim_id, result in topology_result.claim_plausibilities.items():
            is_superseded = claim_id in topology_result.superseded_by
            await self._store_claim_plausibility(
                event_id=event_id,
                claim_id=claim_id,
                plausibility=result.posterior,
                prior=result.prior,
                is_superseded=is_superseded,
                corroboration_count=len(result.evidence_for),
                contradiction_count=len(result.evidence_against)
            )

        # 3. Store claim-to-claim relationships
        # First, clear existing topology edges for this event's claims
        await self._clear_topology_edges(event_id)

        # Store new edges from topology analysis
        await self._store_topology_edges(event_id, topology_result)

        # 4. Store update chains as JSON on Event node
        if topology_result.superseded_by:
            await self._store_update_chains(event_id, topology_result.superseded_by)

        logger.info(f"✅ Topology persisted: {len(topology_result.claim_plausibilities)} claims, "
                   f"pattern={topology_result.pattern}")

    async def _store_topology_metadata(
        self,
        event_id: str,
        pattern: str,
        consensus_date: Optional[str],
        contradictions_count: int,
        superseded_count: int,
        source_diversity: Dict = None
    ) -> None:
        """Store topology metadata on Event node."""
        await self.neo4j._execute_write("""
            MATCH (e:Event {id: $event_id})
            SET e.topology_pattern = $pattern,
                e.topology_consensus_date = $consensus_date,
                e.topology_contradictions = $contradictions_count,
                e.topology_superseded = $superseded_count,
                e.topology_source_diversity = $source_diversity_json,
                e.topology_updated_at = datetime()
        """, {
            'event_id': event_id,
            'pattern': pattern,
            'consensus_date': consensus_date,
            'contradictions_count': contradictions_count,
            'superseded_count': superseded_count,
            'source_diversity_json': json.dumps(source_diversity) if source_diversity else None
        })

    async def _store_claim_plausibility(
        self,
        event_id: str,
        claim_id: str,
        plausibility: float,
        prior: float,
        is_superseded: bool,
        corroboration_count: int,
        contradiction_count: int
    ) -> None:
        """Store plausibility data on SUPPORTS edge."""
        await self.neo4j._execute_write("""
            MATCH (e:Event {id: $event_id})-[s:SUPPORTS]->(c:Claim {id: $claim_id})
            SET s.plausibility = $plausibility,
                s.prior = $prior,
                s.is_superseded = $is_superseded,
                s.corroboration_count = $corroboration_count,
                s.contradiction_count = $contradiction_count,
                s.updated_at = datetime()
        """, {
            'event_id': event_id,
            'claim_id': claim_id,
            'plausibility': plausibility,
            'prior': prior,
            'is_superseded': is_superseded,
            'corroboration_count': corroboration_count,
            'contradiction_count': contradiction_count
        })

    async def _clear_topology_edges(self, event_id: str) -> None:
        """Clear existing topology edges between claims in this event."""
        await self.neo4j._execute_write("""
            MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c1:Claim)
            MATCH (e)-[:SUPPORTS]->(c2:Claim)
            MATCH (c1)-[r:CORROBORATES|CONTRADICTS|UPDATES]->(c2)
            DELETE r
        """, {'event_id': event_id})

    async def _store_topology_edges(self, event_id: str, topology_result) -> None:
        """
        Store claim-to-claim relationships from topology analysis.

        Relations dict contains:
        - key: tuple of (claim_id, claim_id) - sorted for corroborates/contradicts
        - value: 'corroborates', 'contradicts', 'complements', or tuple for updates
        """
        # We need to reconstruct which pairs had which relations
        # The topology_result has claim_plausibilities with evidence_for/against

        edges_created = 0

        # First handle update chains (superseded_by maps old -> new)
        for old_id, new_id in topology_result.superseded_by.items():
            await self.neo4j._execute_write("""
                MATCH (c_old:Claim {id: $old_id})
                MATCH (c_new:Claim {id: $new_id})
                MERGE (c_old)-[r:UPDATES]->(c_new)
                ON CREATE SET r.created_at = datetime()
            """, {'old_id': old_id, 'new_id': new_id})
            edges_created += 1

        # Then handle corroborations and contradictions from evidence
        seen_pairs = set()

        for claim_id, result in topology_result.claim_plausibilities.items():
            # Corroborations
            for corr_id in result.evidence_for:
                pair = tuple(sorted([claim_id, corr_id]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    await self.neo4j._execute_write("""
                        MATCH (c1:Claim {id: $id1})
                        MATCH (c2:Claim {id: $id2})
                        MERGE (c1)-[r:CORROBORATES]-(c2)
                        ON CREATE SET r.created_at = datetime()
                    """, {'id1': pair[0], 'id2': pair[1]})
                    edges_created += 1

            # Contradictions
            for contra_id in result.evidence_against:
                pair = tuple(sorted([claim_id, contra_id]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    await self.neo4j._execute_write("""
                        MATCH (c1:Claim {id: $id1})
                        MATCH (c2:Claim {id: $id2})
                        MERGE (c1)-[r:CONTRADICTS]-(c2)
                        ON CREATE SET r.created_at = datetime()
                    """, {'id1': pair[0], 'id2': pair[1]})
                    edges_created += 1

        logger.debug(f"📊 Created {edges_created} topology edges")

    async def _store_update_chains(
        self,
        event_id: str,
        superseded_by: Dict[str, str]
    ) -> None:
        """
        Store update chains as JSON on Event node.

        Reconstructs chains from superseded_by mapping:
        e.g., {cl_1: cl_2, cl_2: cl_3} becomes chain [cl_1, cl_2, cl_3]
        """
        # Build chains by following the superseded_by links
        chains = []
        used = set()

        # Find chain roots (claims that supersede but aren't superseded)
        all_old = set(superseded_by.keys())
        all_new = set(superseded_by.values())
        roots = all_old - all_new  # Starting points of chains

        for root in roots:
            chain = [root]
            current = root
            while current in superseded_by:
                next_claim = superseded_by[current]
                chain.append(next_claim)
                used.add(current)
                current = next_claim
            used.add(current)

            if len(chain) > 1:
                chains.append({
                    'chain': chain,
                    'current': chain[-1]  # Most recent claim
                })

        # Store as JSON
        await self.neo4j._execute_write("""
            MATCH (e:Event {id: $event_id})
            SET e.topology_update_chains = $chains_json
        """, {
            'event_id': event_id,
            'chains_json': json.dumps(chains)
        })

    # =========================================================================
    # RETRIEVE OPERATIONS
    # =========================================================================

    async def get_topology(self, event_id: str) -> Optional[TopologyData]:
        """
        Retrieve complete topology data for an event.

        Single query fetches everything needed for visualization.
        """
        # Query 1: Get event metadata and claims with plausibilities
        result = await self.neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})
            OPTIONAL MATCH (e)-[s:SUPPORTS]->(c:Claim)
            OPTIONAL MATCH (p:Page)-[:CONTAINS]->(c)
            OPTIONAL MATCH (p)-[:PUBLISHED_BY]->(pub:Entity {is_publisher: true})
            RETURN e.topology_pattern as pattern,
                   e.topology_consensus_date as consensus_date,
                   e.coherence as coherence,
                   e.topology_contradictions as contradictions_count,
                   e.topology_source_diversity as source_diversity_json,
                   e.topology_update_chains as update_chains_json,
                   e.topology_updated_at as updated_at,
                   collect(DISTINCT {
                       id: c.id,
                       text: c.text,
                       plausibility: s.plausibility,
                       prior: s.prior,
                       is_superseded: s.is_superseded,
                       event_time: c.event_time,
                       source_type: pub.source_type,
                       corroboration_count: s.corroboration_count
                   }) as claims
        """, {'event_id': event_id})

        if not result or not result[0].get('pattern'):
            # No topology computed yet
            return None

        row = result[0]

        # Query 2: Get claim-to-claim relationships
        rel_result = await self.neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})-[:SUPPORTS]->(c1:Claim)
            MATCH (e)-[:SUPPORTS]->(c2:Claim)
            MATCH (c1)-[r:CORROBORATES|CONTRADICTS|UPDATES]->(c2)
            RETURN c1.id as source_id, c2.id as target_id,
                   type(r) as rel_type, r.similarity as similarity
        """, {'event_id': event_id})

        # Build claims list
        claims = []
        for c in row['claims']:
            if c['id']:  # Filter out null entries
                claims.append(TopologyClaimData(
                    id=c['id'],
                    text=c['text'] or '',
                    plausibility=c['plausibility'] or 0.5,
                    prior=c['prior'] or 0.5,
                    is_superseded=c['is_superseded'] or False,
                    event_time=c['event_time'],
                    source_type=c['source_type'],
                    corroboration_count=c['corroboration_count'] or 0
                ))

        # Build relationships list
        relationships = [
            TopologyRelationship(
                source_id=r['source_id'],
                target_id=r['target_id'],
                rel_type=r['rel_type'],
                similarity=r['similarity']
            )
            for r in rel_result
        ]

        # Parse JSON fields
        source_diversity = {}
        if row['source_diversity_json']:
            try:
                source_diversity = json.loads(row['source_diversity_json'])
            except:
                pass

        update_chains = []
        if row['update_chains_json']:
            try:
                chains_data = json.loads(row['update_chains_json'])
                for chain in chains_data:
                    update_chains.append(TopologyUpdateChain(
                        metric='update',  # TODO: detect metric type
                        chain=chain.get('chain', []),
                        current_claim_id=chain.get('current', '')
                    ))
            except:
                pass

        # Extract contradictions from relationships
        contradictions = [
            {'claim1_id': r.source_id, 'claim2_id': r.target_id}
            for r in relationships
            if r.rel_type == 'CONTRADICTS'
        ]

        # Calculate temperature (0=stable, 1=chaotic)
        # Based on contradictions vs total claims
        temperature = 0.0
        if claims:
            contradiction_ratio = len(contradictions) / len(claims)
            temperature = min(1.0, contradiction_ratio * 2)  # Scale up

        return TopologyData(
            event_id=event_id,
            pattern=row['pattern'] or 'unknown',
            consensus_date=row['consensus_date'],
            coherence=row['coherence'] or 0.5,
            temperature=temperature,
            claims=claims,
            relationships=relationships,
            update_chains=update_chains,
            contradictions=contradictions,
            source_diversity=source_diversity,
            last_updated=row['updated_at'] or datetime.utcnow()
        )

    async def has_topology(self, event_id: str) -> bool:
        """Check if topology has been computed for an event."""
        result = await self.neo4j._execute_read("""
            MATCH (e:Event {id: $event_id})
            RETURN e.topology_pattern IS NOT NULL as has_topology
        """, {'event_id': event_id})

        return result[0]['has_topology'] if result else False

    # =========================================================================
    # INCREMENTAL UPDATE OPERATIONS
    # =========================================================================

    async def add_claim_relationship(
        self,
        claim1_id: str,
        claim2_id: str,
        rel_type: str,
        similarity: float = None
    ) -> None:
        """
        Add a single relationship between claims.

        Used for incremental updates when new claims arrive.
        """
        rel_type_upper = rel_type.upper()
        if rel_type_upper not in ('CORROBORATES', 'CONTRADICTS', 'UPDATES'):
            raise ValueError(f"Invalid relationship type: {rel_type}")

        query = f"""
            MATCH (c1:Claim {{id: $id1}})
            MATCH (c2:Claim {{id: $id2}})
            MERGE (c1)-[r:{rel_type_upper}]->(c2)
            ON CREATE SET r.created_at = datetime()
        """
        if similarity is not None:
            query = f"""
                MATCH (c1:Claim {{id: $id1}})
                MATCH (c2:Claim {{id: $id2}})
                MERGE (c1)-[r:{rel_type_upper}]->(c2)
                ON CREATE SET r.created_at = datetime(), r.similarity = $similarity
                ON MATCH SET r.similarity = $similarity
            """

        await self.neo4j._execute_write(query, {
            'id1': claim1_id,
            'id2': claim2_id,
            'similarity': similarity
        })

    async def update_claim_plausibility(
        self,
        event_id: str,
        claim_id: str,
        plausibility: float,
        is_superseded: bool = None
    ) -> None:
        """
        Update plausibility for a single claim.

        Used for incremental updates after new evidence arrives.
        """
        params = {
            'event_id': event_id,
            'claim_id': claim_id,
            'plausibility': plausibility
        }

        if is_superseded is not None:
            await self.neo4j._execute_write("""
                MATCH (e:Event {id: $event_id})-[s:SUPPORTS]->(c:Claim {id: $claim_id})
                SET s.plausibility = $plausibility,
                    s.is_superseded = $is_superseded,
                    s.updated_at = datetime()
            """, {**params, 'is_superseded': is_superseded})
        else:
            await self.neo4j._execute_write("""
                MATCH (e:Event {id: $event_id})-[s:SUPPORTS]->(c:Claim {id: $claim_id})
                SET s.plausibility = $plausibility,
                    s.updated_at = datetime()
            """, params)
