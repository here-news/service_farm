"""
Experiment framework for epistemic unit emergence testing.

Provides:
- Scalable test harness for different claim counts
- Threshold sweep experiments
- Visualization of emergence curves
- JSON export for analysis
"""

import asyncio
import json
import os
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import asyncpg
from openai import AsyncOpenAI
from pgvector.asyncpg import register_vector

from services.neo4j_service import Neo4jService
from test_eu.core.epistemic_unit import Claim, EmergenceEngine
from test_eu.core.evaluation import GroundTruth, LabeledClaim, evaluate_clustering, ClusterMetrics


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    name: str
    claims_per_event: List[int] = field(default_factory=lambda: [3, 5, 7, 10])
    num_events: int = 3
    association_thresholds: List[float] = field(default_factory=lambda: [0.50])
    use_associations: bool = True
    random_seed: Optional[int] = None

    # Weight configurations to sweep
    weight_configs: List[Dict[str, float]] = field(default_factory=lambda: [
        {'semantic': 0.35, 'entity_strength': 0.25, 'anchor_match': 0.30, 'source_diversity': 0.10}
    ])


@dataclass
class TrialResult:
    """Result of a single trial."""
    claims_per_event: int
    total_claims: int
    num_surfaces: int
    fragmentation: float
    b3_f1: float
    b3_precision: float
    b3_recall: float
    completeness: float
    purity: float
    pure_count: int
    mixed_count: int

    # Config used
    association_threshold: float
    weights: Dict[str, float]
    use_associations: bool

    # Timing
    llm_calls: int = 0
    duration_seconds: float = 0.0

    # Diagnostic: connection analysis
    total_pairs: int = 0           # n*(n-1)/2
    gate_passed: int = 0           # pairs that passed should_compare gate
    llm_connected: int = 0         # pairs where LLM found identity relation
    soft_edges: int = 0            # Tier-2 soft association edges
    candidate_recall: float = 0.0  # gate_passed / gt_same_event_pairs
    llm_recall: float = 0.0        # llm_connected / gate_passed


@dataclass
class ExperimentResult:
    """Full experiment results."""
    config: ExperimentConfig
    trials: List[TrialResult]
    events_used: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            'config': asdict(self.config),
            'trials': [asdict(t) for t in self.trials],
            'events_used': self.events_used,
            'timestamp': self.timestamp
        }

    def save(self, path: str):
        """Save results to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ExperimentRunner:
    """Runs emergence experiments with different configurations."""

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        neo4j: Neo4jService,
        llm: AsyncOpenAI
    ):
        self.db_pool = db_pool
        self.neo4j = neo4j
        self.llm = llm

    async def load_claims(
        self,
        events: List[Dict],
        claims_per_event: int
    ) -> Tuple[List[Claim], List[LabeledClaim]]:
        """Load claims from database."""
        claims = []
        labeled_claims = []

        for ev in events:
            claims_data = await self.neo4j._execute_read('''
                MATCH (e:Event {id: $eid})-[:INTAKES]->(c:Claim)
                WHERE c.text IS NOT NULL
                OPTIONAL MATCH (p:Page)-[:EMITS]->(c)
                OPTIONAL MATCH (c)-[:MENTIONS]->(ent:Entity)
                WITH c, p, collect({name: ent.canonical_name, type: ent.entity_type}) as entities
                RETURN c.id as id, c.text as text, p.domain as source, entities
                ORDER BY rand()
                LIMIT $limit
            ''', {'eid': ev['id'], 'limit': claims_per_event})

            async with self.db_pool.acquire() as conn:
                await register_vector(conn)
                for row in claims_data:
                    if not row['text']:
                        continue

                    embedding = await conn.fetchval(
                        'SELECT embedding FROM core.claim_embeddings WHERE claim_id = $1',
                        row['id']
                    )

                    all_ent = set()
                    anchor_ent = set()
                    for ent in row['entities']:
                        ent_name = ent.get('name')
                        if ent_name:
                            all_ent.add(ent_name)
                            if ent.get('type') in ('PERSON', 'ORGANIZATION', 'ORG'):
                                anchor_ent.add(ent_name)

                    emb = None
                    if embedding is not None and len(embedding) > 0:
                        emb = [float(x) for x in embedding]

                    claim = Claim(
                        id=row['id'],
                        text=row['text'],
                        source=row['source'] or 'unknown',
                        embedding=emb,
                        entities=all_ent,
                        anchor_entities=anchor_ent
                    )

                    claims.append(claim)
                    labeled_claims.append(LabeledClaim(
                        claim=claim,
                        event_label=ev['name'][:25]
                    ))

        return claims, labeled_claims

    async def run_trial(
        self,
        claims: List[Claim],
        labeled_claims: List[LabeledClaim],
        gt: GroundTruth,
        association_threshold: float,
        weights: Dict[str, float],
        use_associations: bool,
        claims_per_event: int
    ) -> TrialResult:
        """Run a single trial with given configuration."""
        import time
        start = time.time()

        engine = EmergenceEngine(llm=self.llm)

        for claim in claims:
            await engine.add_claim(claim)

        surfaces = engine.compute_surfaces(
            use_associations=use_associations,
            association_threshold=association_threshold,
            affinity_weights=weights
        )

        duration = time.time() - start

        # Evaluate
        metrics = evaluate_clustering(surfaces, gt)

        # Calculate purity
        pure_count = 0
        mixed_count = 0
        for s in surfaces:
            events_in = set(
                lc.event_label for lc in labeled_claims
                if lc.claim.id in s.claim_ids
            )
            if len(events_in) == 1:
                pure_count += 1
            else:
                mixed_count += 1

        purity = pure_count / len(surfaces) if surfaces else 0.0

        # Diagnostic: analyze connection patterns
        n = len(claims)
        total_pairs = n * (n - 1) // 2

        # Count GT same-event pairs
        claim_to_event = gt.claim_to_event()
        gt_same_event_pairs = 0
        for i, c1 in enumerate(claims):
            for c2 in claims[i+1:]:
                if claim_to_event.get(c1.id) == claim_to_event.get(c2.id):
                    gt_same_event_pairs += 1

        # Count gate passes and LLM connections
        gate_passed = 0
        for i, c1 in enumerate(claims):
            for c2 in claims[i+1:]:
                if c1.should_compare(c2):
                    gate_passed += 1

        llm_connected = len(engine.claim_edges)

        # Count soft edges (computed during compute_surfaces)
        # This is approximate - edges from Tier-2 associations
        soft_edges = 0
        if use_associations:
            import math
            entity_df = {}
            for c in claims:
                for e in c.entities:
                    entity_df[e] = entity_df.get(e, 0) + 1
            entity_idf = {e: math.log(1 + n / df) for e, df in entity_df.items()}
            for i, c1 in enumerate(claims):
                for c2 in claims[i+1:]:
                    score = c1.event_affinity(c2, entity_idf=entity_idf, weights=weights)
                    if score >= association_threshold:
                        soft_edges += 1

        candidate_recall = gate_passed / gt_same_event_pairs if gt_same_event_pairs > 0 else 0.0
        llm_recall = llm_connected / gate_passed if gate_passed > 0 else 0.0

        return TrialResult(
            claims_per_event=claims_per_event,
            total_claims=len(claims),
            num_surfaces=len(surfaces),
            fragmentation=len(surfaces) / len(gt.events) if gt.events else 0,
            b3_f1=metrics.b3_f1,
            b3_precision=metrics.b3_precision,
            b3_recall=metrics.b3_recall,
            completeness=metrics.completeness,
            purity=purity,
            pure_count=pure_count,
            mixed_count=mixed_count,
            association_threshold=association_threshold,
            weights=weights,
            use_associations=use_associations,
            llm_calls=llm_connected,
            duration_seconds=duration,
            total_pairs=total_pairs,
            gate_passed=gate_passed,
            llm_connected=llm_connected,
            soft_edges=soft_edges,
            candidate_recall=candidate_recall,
            llm_recall=llm_recall
        )

    async def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run full experiment with given configuration."""
        # Get events
        events = await self.neo4j._execute_read('''
            MATCH (e:Event)-[:INTAKES]->(c:Claim)
            WITH e, count(c) as cnt
            WHERE cnt >= 15
            RETURN e.id as id, e.canonical_name as name, cnt
            ORDER BY cnt DESC
            LIMIT $limit
        ''', {'limit': config.num_events})

        trials = []

        for n_claims in config.claims_per_event:
            for threshold in config.association_thresholds:
                for weights in config.weight_configs:
                    # Load fresh claims for each trial
                    claims, labeled_claims = await self.load_claims(events, n_claims)

                    gt = GroundTruth(
                        claims=labeled_claims,
                        events={ev['name'][:25]: ev['name'] for ev in events},
                        confounders=[]
                    )

                    trial = await self.run_trial(
                        claims=claims,
                        labeled_claims=labeled_claims,
                        gt=gt,
                        association_threshold=threshold,
                        weights=weights,
                        use_associations=config.use_associations,
                        claims_per_event=n_claims
                    )

                    trials.append(trial)

                    print(f"  {n_claims} claims: {trial.num_surfaces} surfaces, "
                          f"B³ F1={trial.b3_f1:.1%}, P={trial.b3_precision:.0%}, "
                          f"R={trial.b3_recall:.0%}, purity={trial.pure_count}/{trial.num_surfaces}")

        return ExperimentResult(
            config=config,
            trials=trials,
            events_used=[ev['name'] for ev in events]
        )


def plot_emergence_curve(result: ExperimentResult, output_path: str = None):
    """Plot B³ metrics vs claims per event."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    claims = [t.claims_per_event for t in result.trials]
    b3_f1 = [t.b3_f1 for t in result.trials]
    precision = [t.b3_precision for t in result.trials]
    recall = [t.b3_recall for t in result.trials]
    surfaces = [t.num_surfaces for t in result.trials]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: B³ metrics
    ax1.plot(claims, b3_f1, 'b-o', label='B³ F1', linewidth=2)
    ax1.plot(claims, precision, 'g--s', label='Precision', linewidth=2)
    ax1.plot(claims, recall, 'r--^', label='Recall', linewidth=2)
    ax1.set_xlabel('Claims per Event')
    ax1.set_ylabel('Score')
    ax1.set_title('B³ Metrics vs Scale')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Right: Surface count
    ax2.bar(claims, surfaces, color='steelblue', alpha=0.7)
    ax2.axhline(y=result.config.num_events, color='red', linestyle='--',
                label=f'Target ({result.config.num_events} events)')
    ax2.set_xlabel('Claims per Event')
    ax2.set_ylabel('Number of Surfaces')
    ax2.set_title('Fragmentation vs Scale')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_threshold_sweep(results: List[ExperimentResult], output_path: str = None):
    """Plot metrics across different thresholds."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    # Group by threshold
    by_threshold = {}
    for result in results:
        for trial in result.trials:
            thresh = trial.association_threshold
            if thresh not in by_threshold:
                by_threshold[thresh] = []
            by_threshold[thresh].append(trial)

    thresholds = sorted(by_threshold.keys())
    avg_f1 = [sum(t.b3_f1 for t in by_threshold[th]) / len(by_threshold[th]) for th in thresholds]
    avg_precision = [sum(t.b3_precision for t in by_threshold[th]) / len(by_threshold[th]) for th in thresholds]
    avg_recall = [sum(t.b3_recall for t in by_threshold[th]) / len(by_threshold[th]) for th in thresholds]
    avg_purity = [sum(t.purity for t in by_threshold[th]) / len(by_threshold[th]) for th in thresholds]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(thresholds, avg_f1, 'b-o', label='B³ F1', linewidth=2)
    ax.plot(thresholds, avg_precision, 'g--s', label='Precision', linewidth=2)
    ax.plot(thresholds, avg_recall, 'r--^', label='Recall', linewidth=2)
    ax.plot(thresholds, avg_purity, 'm-.d', label='Purity', linewidth=2)

    ax.set_xlabel('Association Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Threshold Sweep: Precision-Recall Tradeoff')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


async def run_quick_experiment():
    """Quick experiment for testing."""
    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        database=os.getenv('POSTGRES_DB', 'phi_here'),
        user=os.getenv('POSTGRES_USER', 'phi_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'phi_password_dev'),
        min_size=1, max_size=5
    )

    neo4j = Neo4jService(
        uri=os.getenv('NEO4J_URI', 'bolt://neo4j:7687'),
        user=os.getenv('NEO4J_USER', 'neo4j'),
        password=os.getenv('NEO4J_PASSWORD')
    )
    await neo4j.connect()

    llm = AsyncOpenAI()

    runner = ExperimentRunner(db_pool, neo4j, llm)

    config = ExperimentConfig(
        name="quick_test",
        claims_per_event=[3, 5],
        num_events=3,
        association_thresholds=[0.50],
        use_associations=True
    )

    print(f"Running experiment: {config.name}")
    result = await runner.run_experiment(config)

    # Save results
    result.save('/app/backend/test_eu/results/experiment_quick.json')

    # Plot
    plot_emergence_curve(result, '/app/backend/test_eu/results/emergence_curve.png')

    await db_pool.close()
    await neo4j.close()

    return result


if __name__ == '__main__':
    asyncio.run(run_quick_experiment())
