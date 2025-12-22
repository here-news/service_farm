#!/usr/bin/env python3
"""
Experiment: Contribution Simulation

Test how the EU system handles community contributions as described in
docs/66.product.liveevent.md

Simulates:
1. URL contributions → fetch, extract claims, absorb
2. Text claim contributions → evaluate, absorb or reject
3. Coherence delta calculation for rewards
4. Event responses (accepted, duplicate, rejected, high_value)

Run inside container:
    docker exec herenews-app python /app/test_eu/contribution_simulation.py
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
from datetime import datetime
from enum import Enum
import httpx


# Config
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"


class ContributionStatus(Enum):
    PROCESSING = "processing"
    ACCEPTED = "accepted"
    HIGH_VALUE = "high_value"
    NOTED = "noted"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"
    SKEPTICAL = "skeptical"


@dataclass
class Contribution:
    """Community contribution"""
    id: str
    type: Literal['url', 'text_claim', 'evidence', 'dispute']
    content: str
    stake: int  # Credits committed
    user_id: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ContributionResult:
    """Event's response to contribution"""
    contribution_id: str
    status: ContributionStatus
    reason: str
    claims_added: int = 0
    coherence_delta: float = 0.0
    reward: int = 0
    response: str = ""


@dataclass
class SimulatedEU:
    """Simplified EU for contribution testing"""
    id: str
    embedding: List[float]
    claim_ids: List[str] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    page_ids: set = field(default_factory=set)
    coherence: float = 1.0
    internal_corr: int = 0
    internal_contra: int = 0

    def update_coherence(self):
        total = self.internal_corr + self.internal_contra
        self.coherence = self.internal_corr / total if total > 0 else 1.0


def get_embedding(text: str) -> List[float]:
    """Get embedding for text"""
    response = httpx.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=30
    )
    return response.json()['data'][0]['embedding']


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for multiple texts"""
    response = httpx.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={"model": EMBEDDING_MODEL, "input": texts},
        timeout=60
    )
    return [d['embedding'] for d in response.json()['data']]


def llm_complete(prompt: str, max_tokens: int = 100) -> str:
    """Simple LLM completion"""
    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0
        },
        timeout=30
    )
    return response.json()['choices'][0]['message']['content']


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0


class ContributionProcessor:
    """Processes community contributions into EU system"""

    def __init__(self, event: SimulatedEU):
        self.event = event
        self.known_texts: set = set()
        self.contribution_history: List[ContributionResult] = []

    def process_contribution(self, contribution: Contribution) -> ContributionResult:
        """Process a contribution and return result"""

        old_coherence = self.event.coherence
        old_claims = len(self.event.claim_ids)

        if contribution.type == 'url':
            result = self._process_url(contribution)
        elif contribution.type == 'text_claim':
            result = self._process_text_claim(contribution)
        elif contribution.type == 'dispute':
            result = self._process_dispute(contribution)
        else:
            result = ContributionResult(
                contribution_id=contribution.id,
                status=ContributionStatus.REJECTED,
                reason="Unknown contribution type"
            )

        if result.status in [ContributionStatus.ACCEPTED, ContributionStatus.HIGH_VALUE]:
            result.coherence_delta = self.event.coherence - old_coherence
            result.claims_added = len(self.event.claim_ids) - old_claims
            result.reward = self._calculate_reward(
                contribution.stake,
                result.coherence_delta,
                result.claims_added
            )

            if result.reward > 0:
                result.status = ContributionStatus.HIGH_VALUE

        result.response = self._generate_response(result, contribution)
        self.contribution_history.append(result)
        return result

    def _process_url(self, contribution: Contribution) -> ContributionResult:
        """Process URL contribution - simulates fetch and extract"""
        url = contribution.content

        # Simulate claim extraction
        simulated_claims = [
            f"Claim from {url[:30]}: Detail about the incident",
            f"Claim from {url[:30]}: Official statement released",
            f"Claim from {url[:30]}: Timeline of events"
        ]

        new_claims = [c for c in simulated_claims if c not in self.known_texts]

        if not new_claims:
            return ContributionResult(
                contribution_id=contribution.id,
                status=ContributionStatus.DUPLICATE,
                reason="All claims from this source are already known"
            )

        embeddings = get_embeddings_batch(new_claims)

        relevance_scores = [
            cosine_similarity(emb, self.event.embedding)
            for emb in embeddings
        ]

        relevant_claims = [
            (c, e) for c, e, s in zip(new_claims, embeddings, relevance_scores)
            if s > 0.5
        ]

        if not relevant_claims:
            return ContributionResult(
                contribution_id=contribution.id,
                status=ContributionStatus.REJECTED,
                reason="Content doesn't appear related to this event"
            )

        for claim_text, embedding in relevant_claims:
            self._absorb_claim(claim_text, embedding, contribution.id)

        return ContributionResult(
            contribution_id=contribution.id,
            status=ContributionStatus.ACCEPTED,
            reason=f"Extracted and integrated {len(relevant_claims)} claims"
        )

    def _process_text_claim(self, contribution: Contribution) -> ContributionResult:
        """Process text claim contribution"""
        claim_text = contribution.content.strip()

        is_verifiable = self._check_verifiable(claim_text)
        if not is_verifiable:
            return ContributionResult(
                contribution_id=contribution.id,
                status=ContributionStatus.SKEPTICAL,
                reason="Couldn't identify a verifiable factual claim"
            )

        if claim_text in self.known_texts:
            return ContributionResult(
                contribution_id=contribution.id,
                status=ContributionStatus.DUPLICATE,
                reason="This information is already known"
            )

        embedding = get_embedding(claim_text)
        relevance = cosine_similarity(embedding, self.event.embedding)

        if relevance < 0.4:
            return ContributionResult(
                contribution_id=contribution.id,
                status=ContributionStatus.REJECTED,
                reason="Claim doesn't appear related to this event"
            )

        is_contradiction = self._check_contradiction(claim_text)

        self._absorb_claim(claim_text, embedding, contribution.id, is_contradiction)

        if is_contradiction:
            return ContributionResult(
                contribution_id=contribution.id,
                status=ContributionStatus.ACCEPTED,
                reason="Claim recorded as potential contradiction - needs verification"
            )

        return ContributionResult(
            contribution_id=contribution.id,
            status=ContributionStatus.NOTED,
            reason="Claim recorded but unverified (needs corroborating source)"
        )

    def _process_dispute(self, contribution: Contribution) -> ContributionResult:
        """Process dispute contribution"""
        return ContributionResult(
            contribution_id=contribution.id,
            status=ContributionStatus.ACCEPTED,
            reason="Dispute flagged for review - may trigger re-analysis"
        )

    def _check_verifiable(self, text: str) -> bool:
        """Check if text contains a verifiable factual claim"""
        prompt = f"""Is this a verifiable factual claim (not opinion/question)?

Text: {text[:300]}

Answer YES or NO."""

        response = llm_complete(prompt, max_tokens=10)
        return 'YES' in response.upper()

    def _check_contradiction(self, new_claim: str) -> bool:
        """Check if new claim contradicts existing claims"""
        if not self.event.texts:
            return False

        sample = self.event.texts[:5]
        sample_text = "\n".join(f"- {t[:100]}" for t in sample)

        prompt = f"""Does the new claim CONTRADICT any existing claims?

Existing claims:
{sample_text}

New claim: {new_claim[:200]}

Answer YES or NO."""

        response = llm_complete(prompt, max_tokens=10)
        return 'YES' in response.upper()

    def _absorb_claim(
        self,
        text: str,
        embedding: List[float],
        source_id: str,
        is_contradiction: bool = False
    ):
        """Absorb claim into event"""
        claim_id = f"contrib_{source_id}_{len(self.event.claim_ids)}"

        self.event.claim_ids.append(claim_id)
        self.event.texts.append(text)
        self.known_texts.add(text)

        # Update embedding
        n = len(self.event.claim_ids)
        self.event.embedding = [
            (self.event.embedding[i] * (n - 1) + embedding[i]) / n
            for i in range(len(embedding))
        ]

        if is_contradiction:
            self.event.internal_contra += 1
        else:
            self.event.internal_corr += 1
        self.event.update_coherence()

    def _calculate_reward(self, stake: int, coherence_delta: float, claims_added: int) -> int:
        """Calculate reward based on coherence improvement"""
        if coherence_delta > 0.05 and claims_added >= 3:
            return int(stake * 0.5)
        elif coherence_delta > 0.02:
            return int(stake * 0.2)
        elif claims_added >= 2:
            return int(stake * 0.1)
        return 0

    def _generate_response(self, result: ContributionResult, contribution: Contribution) -> str:
        """Generate human-readable response"""
        templates = {
            ContributionStatus.ACCEPTED: (
                f"✅ Added {result.claims_added} claims. "
                f"Coherence {'improved' if result.coherence_delta >= 0 else 'adjusted'} by {result.coherence_delta:+.2%}."
            ),
            ContributionStatus.HIGH_VALUE: (
                f"💎 Excellent contribution! Added {result.claims_added} claims, "
                f"coherence +{result.coherence_delta:.2%}. "
                f"Rewarding {result.reward}c!"
            ),
            ContributionStatus.NOTED: (
                f"📝 Claim recorded but unverified. "
                f"Consider providing a source link for corroboration."
            ),
            ContributionStatus.DUPLICATE: f"🔄 {result.reason}",
            ContributionStatus.REJECTED: f"❌ {result.reason}",
            ContributionStatus.SKEPTICAL: (
                f"🤔 {result.reason}. "
                f"Consider submitting a factual claim with a source."
            ),
            ContributionStatus.PROCESSING: "🧠 Processing..."
        }
        return templates.get(result.status, result.reason)


def simulate_contribution_flow():
    """Simulate a series of contributions to test the flow"""

    # Create a simulated event
    seed_text = "A fire broke out in Wang Fuk Court, Tai Po, Hong Kong on November 26, 2025"
    print(f"🎯 Creating event with seed: {seed_text[:50]}...")
    seed_embedding = get_embedding(seed_text)

    event = SimulatedEU(
        id="test_event_1",
        embedding=seed_embedding,
        texts=[seed_text],
        claim_ids=["seed_1"],
        internal_corr=1
    )

    processor = ContributionProcessor(event)

    # Simulate contributions
    contributions = [
        Contribution(
            id="c1", type="url", stake=10, user_id="alice",
            content="https://bbc.com/news/hong-kong-fire-death-toll-rises"
        ),
        Contribution(
            id="c2", type="text_claim", stake=1, user_id="bob",
            content="17 people confirmed dead in the fire"
        ),
        Contribution(
            id="c3", type="text_claim", stake=1, user_id="charlie",
            content="This is a government coverup!!!"  # Opinion
        ),
        Contribution(
            id="c4", type="url", stake=100, user_id="chen",
            content="https://hkfd.gov.hk/official-report.pdf"
        ),
        Contribution(
            id="c5", type="text_claim", stake=10, user_id="diana",
            content="Fire started due to electrical fault according to initial investigation"
        ),
        Contribution(
            id="c6", type="text_claim", stake=10, user_id="eve",
            content="Witnesses report seeing suspicious person before fire started"
        ),
        Contribution(
            id="c7", type="text_claim", stake=1, user_id="frank",
            content="What time did the fire start?"  # Question
        ),
        Contribution(
            id="c8", type="url", stake=10, user_id="alice",
            content="https://bbc.com/news/hong-kong-fire-death-toll-rises"  # Duplicate
        ),
    ]

    print("=" * 70)
    print("CONTRIBUTION SIMULATION")
    print("=" * 70)
    print(f"\n🎯 Event: {seed_text[:50]}...")
    print(f"   Initial coherence: {event.coherence:.2%}")

    print("\n📥 Processing contributions:\n")

    for contrib in contributions:
        print(f"[{contrib.user_id}] ({contrib.stake}c) {contrib.type}: {contrib.content[:50]}...")

        result = processor.process_contribution(contrib)

        icon = {
            ContributionStatus.ACCEPTED: "✅",
            ContributionStatus.HIGH_VALUE: "💎",
            ContributionStatus.NOTED: "📝",
            ContributionStatus.DUPLICATE: "🔄",
            ContributionStatus.REJECTED: "❌",
            ContributionStatus.SKEPTICAL: "🤔",
        }.get(result.status, "❓")

        print(f"   {icon} {result.status.value}: {result.reason[:60]}")
        if result.reward > 0:
            print(f"   🎁 Reward: {result.reward}c")
        if result.coherence_delta != 0:
            print(f"   📊 Coherence: {event.coherence:.2%} ({result.coherence_delta:+.2%})")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    status_counts = {}
    for result in processor.contribution_history:
        status = result.status.value
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"\n📊 Contribution Results:")
    for status, count in sorted(status_counts.items()):
        print(f"   {status}: {count}")

    total_rewards = sum(r.reward for r in processor.contribution_history)
    total_claims = sum(r.claims_added for r in processor.contribution_history)

    print(f"\n💰 Economics:")
    print(f"   Total rewards issued: {total_rewards}c")
    print(f"   Total claims added: {total_claims}")
    print(f"   Final coherence: {event.coherence:.2%}")

    accepted = sum(1 for r in processor.contribution_history
                   if r.status in [ContributionStatus.ACCEPTED, ContributionStatus.HIGH_VALUE, ContributionStatus.NOTED])
    total = len(processor.contribution_history)
    print(f"\n📈 Rates:")
    print(f"   Acceptance rate: {accepted}/{total} ({100*accepted/total:.0f}%)")
    print(f"   Reward rate: {sum(1 for r in processor.contribution_history if r.reward > 0)}/{total} "
          f"({100*sum(1 for r in processor.contribution_history if r.reward > 0)/total:.0f}%)")

    return {
        'status_counts': status_counts,
        'total_rewards': total_rewards,
        'total_claims': total_claims,
        'final_coherence': event.coherence,
        'acceptance_rate': accepted / total,
        'reward_rate': sum(1 for r in processor.contribution_history if r.reward > 0) / total
    }


def main():
    results = simulate_contribution_flow()

    print(f"\n💾 Saving results...")
    with open('/app/test_eu/contribution_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n✅ Contribution simulation complete!")


if __name__ == "__main__":
    main()
