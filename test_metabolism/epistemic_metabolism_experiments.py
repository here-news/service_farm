"""
Epistemic Metabolism Experiments

Goal: stress-test the LiveEvent metabolism ideas on three epistemic axes without
calling external services. Each experiment builds synthetic claims with fixed
embeddings and priors, runs ClaimTopologyService.analyze(), and reports the
posteriors plus qualitative takeaways.

Experiments:
1) Monotone updates vs late contradiction (casualty counts).
2) Source diversity vs echo chamber (same text, different priors).
3) Date consensus with an off-by-year outlier (penalty check).
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND = os.path.join(ROOT, "backend")
for path in (ROOT, BACKEND):
    if path not in sys.path:
        sys.path.insert(0, path)

from models.domain.claim import Claim
from services.claim_topology import ClaimTopologyService, TopologyResult
from utils.id_generator import generate_claim_id, generate_page_id


# Fixed embeddings to avoid OpenAI calls; cosine similarity will be 1.0 for
# identical vectors and high (>0.6) for nearby ones.
def _vec(x: float, y: float, z: float) -> List[float]:
    return [x, y, z]


def make_claim(
    text: str,
    event_time: datetime,
    embedding: List[float],
    entity_ids: List[str],
    page_id: str = None,
    claim_id: str = None,
    topic_key: str = None,
) -> Claim:
    return Claim(
        id=claim_id or generate_claim_id(),
        page_id=page_id or generate_page_id(),
        text=text,
        event_time=event_time,
        embedding=embedding,
        metadata={"entity_ids": entity_ids},
        topic_key=topic_key,
    )


def pretty_results(title: str, topology: TopologyResult, order: List[str], label_lookup: Dict[str, str]):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")
    print(f"Pattern: {topology.pattern}, Consensus date: {topology.consensus_date}")
    for cid in order:
        r = topology.claim_plausibilities[cid]
        label = label_lookup.get(cid, cid)
        print(f"[{r.posterior:0.3f}] prior={r.prior:0.2f} conf={r.confidence:0.2f} :: {label}")
    if topology.contradictions:
        print("\nContradictions detected:")
        for c in topology.contradictions:
            print(f" - {c['text1'][:60]} ... VS ... {c['text2'][:60]} ...")


async def experiment_monotone_updates(topology: ClaimTopologyService):
    """
    Scenario: three monotone casualty updates from higher-reliability outlets,
    then a late low-prior drop (echo-chamber rumor).
    Expectation: monotone updates stay high; drop gets penalized and flagged.
    """
    base_time = datetime(2024, 5, 1, 10, 0)
    claims = [
        make_claim("5 confirmed dead in tower fire", base_time, _vec(1, 0, 0), ["en_tower"], topic_key="deaths"),
        make_claim("Deaths rise to 7 as rescue continues", base_time + timedelta(hours=2), _vec(1, 0.05, 0), ["en_tower"], topic_key="deaths"),
        make_claim("Officials: death toll now at 10", base_time + timedelta(hours=5), _vec(0.98, 0.02, 0), ["en_tower"], topic_key="deaths"),
        make_claim("Rumor: only 2 deaths actually confirmed", base_time + timedelta(hours=7), _vec(0.97, 0.01, 0), ["en_tower"], topic_key="deaths"),
    ]

    priors = {
        claims[0].id: {"base_prior": 0.62, "source_type": "wire", "publisher_name": "Reuters"},
        claims[1].id: {"base_prior": 0.60, "source_type": "wire", "publisher_name": "AP"},
        claims[2].id: {"base_prior": 0.58, "source_type": "official", "publisher_name": "Fire Dept"},
        claims[3].id: {"base_prior": 0.48, "source_type": "aggregator", "publisher_name": "SocialRumors"},
    }

    topology_result = await topology.analyze(claims, publisher_priors=priors)
    order = [c.id for c in claims]
    labels = {c.id: c.text for c in claims}
    pretty_results("Experiment 1: Monotone updates vs rumor drop", topology_result, order, labels)


async def experiment_source_diversity(topology: ClaimTopologyService):
    """
    Scenario A: three corroborating claims, all low-prior aggregators.
    Scenario B: same content but from diverse, higher-prior outlets.
    Expectation: B yields materially higher posteriors and a consensus pattern.
    """
    event_time = datetime(2024, 8, 12, 9, 0)

    def build_claims(prefix: str, sources: List[Tuple[str, float]]) -> Tuple[List[Claim], Dict[str, dict]]:
        claims: List[Claim] = []
        priors: Dict[str, dict] = {}
        for i, (source_name, prior) in enumerate(sources):
            claim = make_claim(
                text="Bridge collapse injures 12 people",
                event_time=event_time + timedelta(minutes=i),
                embedding=_vec(0, 1, 0),
                entity_ids=["en_bridge"],
                page_id=f"pg_{prefix}{i}",
                claim_id=f"cl_{prefix}{i}",
                topic_key="injured",
            )
            claims.append(claim)
            priors[claim.id] = {"base_prior": prior, "source_type": source_name, "publisher_name": source_name}
        return claims, priors

    claims_a, priors_a = build_claims("agg", [("aggregator", 0.48), ("aggregator", 0.48), ("aggregator", 0.48)])
    claims_b, priors_b = build_claims("mix", [("wire", 0.62), ("local_news", 0.57), ("official", 0.55)])

    topo_a = await topology.analyze(claims_a, publisher_priors=priors_a)
    topo_b = await topology.analyze(claims_b, publisher_priors=priors_b)

    print("\n\n=== Experiment 2: Source diversity vs echo chamber ===")
    print("Scenario A (all aggregators):")
    for c in claims_a:
        r = topo_a.claim_plausibilities[c.id]
        print(f"[{r.posterior:0.3f}] prior={r.prior:0.2f} :: {c.text} ({priors_a[c.id]['source_type']})")

    print("\nScenario B (mixed sources):")
    for c in claims_b:
        r = topo_b.claim_plausibilities[c.id]
        print(f"[{r.posterior:0.3f}] prior={r.prior:0.2f} :: {c.text} ({priors_b[c.id]['source_type']})")

    print(f"\nPattern A: {topo_a.pattern}, Pattern B: {topo_b.pattern}")


async def experiment_date_outlier(topology: ClaimTopologyService):
    """
    Scenario: three claims agree on 2024-05-01, one claims 2023-11-02.
    Expectation: consensus date is 2024-05-01; outlier posterior is penalized.
    """
    claims = [
        make_claim("Fire started around noon on May 1", datetime(2024, 5, 1, 12, 0), _vec(0, 0, 1), ["en_city"]),
        make_claim("May 1 blaze under control by evening", datetime(2024, 5, 1, 18, 0), _vec(0, 0, 1), ["en_city"]),
        make_claim("Residents evacuated on May 1 night", datetime(2024, 5, 1, 22, 0), _vec(0, 0, 1), ["en_city"]),
        make_claim("Report: fire actually November 2, 2023", datetime(2023, 11, 2, 9, 0), _vec(0, 0, 0.95), ["en_city"]),
    ]

    priors = {c.id: {"base_prior": 0.55, "source_type": "local_news", "publisher_name": "Local"} for c in claims}
    priors[claims[3].id]["base_prior"] = 0.50  # weaker prior for the outlier

    topology_result = await topology.analyze(claims, publisher_priors=priors)
    order = [c.id for c in claims]
    labels = {c.id: c.text for c in claims}
    pretty_results("Experiment 3: Date consensus with outlier", topology_result, order, labels)


async def main():
    topology = ClaimTopologyService(openai_client=None)  # We avoid remote calls by pre-setting embeddings.

    await experiment_monotone_updates(topology)
    await experiment_source_diversity(topology)
    await experiment_date_outlier(topology)


if __name__ == "__main__":
    asyncio.run(main())
