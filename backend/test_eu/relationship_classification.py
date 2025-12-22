"""
Relationship Type Classification Between Similar Events

CRITICAL RESEARCH: This experiment addresses the key insight that mass determines
importance, NOT containment. When two events are similar (e.g., "Hong Kong Fire"
and "Lee Condolences about HK Fire"), we need to determine the relationship type:

- CONTAINS: B is a sub-aspect of A (or vice versa)
- SIBLING: Both are aspects of a larger topic (need parent frame)
- CAUSES: A led to / caused B
- RELATES: Associated but distinct

The Earth vs Jupiter analogy: Jupiter's greater mass doesn't make Earth its moon.
Similarly, higher-mass events don't automatically "contain" lower-mass related events.

Run inside container:
    docker exec herenews-app python /app/test_eu/relationship_classification.py
"""

import os
import json
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
import httpx
import psycopg2
import time
from enum import Enum

from load_graph import load_snapshot, GraphSnapshot


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"

PG_HOST = os.environ.get("POSTGRES_HOST", "herenews-postgres")
PG_DB = os.environ.get("POSTGRES_DB", "herenews")
PG_USER = os.environ.get("POSTGRES_USER", "herenews_user")
PG_PASS = os.environ.get("POSTGRES_PASSWORD", "herenews_pass")

SIM_THRESHOLD = 0.70
LLM_THRESHOLD = 0.55
RELATIONSHIP_THRESHOLD = 0.40  # Lower threshold to find related event pairs


class RelationType(Enum):
    CONTAINS = "contains"      # A contains B as sub-aspect
    CONTAINED_BY = "contained_by"  # B contains A as sub-aspect
    SIBLING = "sibling"        # Both aspects of larger topic
    CAUSES = "causes"          # A caused/led to B
    CAUSED_BY = "caused_by"    # B caused/led to A
    RELATES = "relates"        # Associated but distinct
    SAME = "same"              # Same event (should merge)
    UNRELATED = "unrelated"    # Not actually related


def get_pg_connection():
    return psycopg2.connect(host=PG_HOST, database=PG_DB, user=PG_USER, password=PG_PASS)


def load_cached_embeddings() -> Dict[str, List[float]]:
    conn = get_pg_connection()
    cur = conn.cursor()
    cur.execute("SELECT claim_id, embedding FROM core.claim_embeddings")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    embeddings = {}
    for claim_id, emb in rows:
        if isinstance(emb, str):
            emb = [float(x) for x in emb.strip('[]').split(',')]
        embeddings[claim_id] = list(emb)
    return embeddings


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def llm_same_event(text1: str, text2: str) -> bool:
    prompt = f"""Are these claims about the same news story/event? Answer YES or NO.

Claim 1: {text1[:250]}
Claim 2: {text2[:250]}

Same story?"""

    for attempt in range(3):
        try:
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 5,
                    "temperature": 0
                },
                timeout=60.0
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip().upper().startswith("YES")
        except:
            if attempt < 2:
                time.sleep(2)
            else:
                return False


def llm_classify_relationship(event_a_desc: str, event_b_desc: str, mass_a: float, mass_b: float) -> Dict:
    """
    Classify the relationship between two related events.

    Key insight: Mass determines importance, NOT containment.
    We ask the LLM to determine the semantic relationship type.
    """
    prompt = f"""Analyze the relationship between these two news events.

Event A (mass={mass_a:.1f}): {event_a_desc[:400]}

Event B (mass={mass_b:.1f}): {event_b_desc[:400]}

Note: "Mass" indicates importance/volume of coverage, NOT hierarchy.
Higher mass does NOT mean containment.

Determine the relationship type:

1. CONTAINS - Event A's topic fully encompasses Event B as a sub-aspect
   Example: "Hong Kong Fire" contains "Fire Casualties Report"

2. CONTAINED_BY - Event B's topic fully encompasses Event A as a sub-aspect
   Example: "Political Reactions" is contained by "Hong Kong Fire Coverage"

3. SIBLING - Both are distinct aspects of a larger topic (neither contains the other)
   Example: "Fire Deaths" and "Political Condolences" are siblings under "Hong Kong Fire Story"

4. CAUSES - Event A caused or led to Event B
   Example: "Fire Breaks Out" causes "Investigation Launched"

5. CAUSED_BY - Event B caused or led to Event A
   Example: "Safety Violations" caused by "Fire Investigation"

6. RELATES - Associated by shared entity/context but semantically distinct
   Example: "Hong Kong Fire" relates to "Hong Kong Democracy Protests" (same location, different topics)

7. SAME - These describe the same event and should merge

8. UNRELATED - Despite similarity scores, these are actually about different things

Return JSON:
{{
  "relationship": "contains|contained_by|sibling|causes|caused_by|relates|same|unrelated",
  "confidence": 0.0-1.0,
  "parent_topic": "if sibling, what is the parent topic?" or null,
  "reasoning": "brief explanation of why this relationship type"
}}"""

    for attempt in range(3):
        try:
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0
                },
                timeout=60.0
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("```")[1].replace("json", "").strip()
            return json.loads(content)
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                return {"relationship": "error", "confidence": 0, "reasoning": str(e)}


def llm_would_mass_containment_be_wrong(event_a_desc: str, event_b_desc: str, mass_a: float, mass_b: float) -> Dict:
    """
    Specifically test if mass-based containment assumption would be wrong.

    If mass_a > mass_b, the old model would have A contain B.
    We ask: would that be semantically incorrect?
    """
    larger = "A" if mass_a > mass_b else "B"
    smaller = "B" if mass_a > mass_b else "A"
    larger_desc = event_a_desc if mass_a > mass_b else event_b_desc
    smaller_desc = event_b_desc if mass_a > mass_b else event_a_desc

    prompt = f"""A system uses "mass" (importance/volume) to determine event hierarchy.
It assumes: larger mass event CONTAINS smaller mass event.

Event {larger} (larger mass): {larger_desc[:300]}
Event {smaller} (smaller mass): {smaller_desc[:300]}

Question: Would it be SEMANTICALLY WRONG to say "{larger} contains {smaller}"?

Consider:
- Does {larger}'s topic actually encompass {smaller}'s topic?
- Or is {smaller} a distinct story that happens to be related?
- The Earth-Jupiter analogy: Jupiter's mass doesn't make Earth its moon.

Return JSON:
{{
  "mass_containment_would_be_wrong": true/false,
  "reason": "explanation",
  "better_relationship": "contains|sibling|causes|relates"
}}"""

    for attempt in range(3):
        try:
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0
                },
                timeout=60.0
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("```")[1].replace("json", "").strip()
            return json.loads(content)
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                return {"mass_containment_would_be_wrong": None, "reason": str(e)}


@dataclass
class EU:
    id: str
    level: int = 0
    claim_ids: List[str] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    page_ids: Set[str] = field(default_factory=set)
    embedding: Optional[List[float]] = None
    children: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None

    def size(self) -> int:
        return len(self.claim_ids)

    def mass(self) -> float:
        """Mass formula from our research"""
        coherence = 1.0  # Simplified for this experiment
        return self.size() * 0.1 * (0.5 + coherence) * (1 + 0.1 * len(self.page_ids))

    def label(self) -> str:
        return self.texts[0][:60] + "..." if self.texts else "empty"

    def description(self) -> str:
        return " ".join(self.texts[:3])[:500]


class RelationshipClassifier:
    def __init__(self, snapshot: GraphSnapshot):
        self.snapshot = snapshot
        self.eus: Dict[str, EU] = {}
        self.claim_to_eu: Dict[str, str] = {}
        self.sub_counter = 0
        self.event_counter = 0
        self.llm_calls = 0

        # Results tracking
        self.relationship_results: List[Dict] = []
        self.mass_assumption_tests: List[Dict] = []

    def process_claim(self, claim_id: str, text: str, page_id: str, embedding: List[float]) -> str:
        level0_eus = [eu for eu in self.eus.values() if eu.level == 0 and eu.parent_id is None]

        best_eu = None
        best_sim = 0.0

        for eu in level0_eus:
            if eu.embedding:
                sim = cosine_sim(embedding, eu.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_eu = eu

        should_merge = False

        if best_sim >= SIM_THRESHOLD:
            should_merge = True
        elif best_sim >= LLM_THRESHOLD and best_eu:
            self.llm_calls += 1
            should_merge = llm_same_event(text, best_eu.texts[0])

        if should_merge and best_eu:
            best_eu.claim_ids.append(claim_id)
            best_eu.texts.append(text)
            best_eu.page_ids.add(page_id)

            old_emb = np.array(best_eu.embedding)
            new_emb = np.array(embedding)
            n = len(best_eu.claim_ids)
            best_eu.embedding = ((old_emb * (n - 1) + new_emb) / n).tolist()

            self.claim_to_eu[claim_id] = best_eu.id
            return best_eu.id
        else:
            self.sub_counter += 1
            new_eu = EU(
                id=f"sub_{self.sub_counter}",
                level=0,
                claim_ids=[claim_id],
                texts=[text],
                page_ids={page_id},
                embedding=embedding
            )
            self.eus[new_eu.id] = new_eu
            self.claim_to_eu[claim_id] = new_eu.id
            return new_eu.id

    def find_related_pairs(self, min_size: int = 5) -> List[Tuple[EU, EU, float]]:
        """Find pairs of EUs that are related but not identical"""
        candidates = [
            eu for eu in self.eus.values()
            if eu.level == 0 and eu.size() >= min_size
        ]

        print(f"\nFinding related pairs among {len(candidates)} EUs with {min_size}+ claims...")

        pairs = []
        for i, eu1 in enumerate(candidates):
            for eu2 in candidates[i+1:]:
                sim = cosine_sim(eu1.embedding, eu2.embedding)

                # Related but not same event
                # High sim (>0.7) would merge as same event
                # We're interested in 0.4-0.7 range - related but distinct
                if RELATIONSHIP_THRESHOLD <= sim < SIM_THRESHOLD:
                    pairs.append((eu1, eu2, sim))

        # Sort by similarity descending
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs

    def classify_relationships(self, max_pairs: int = 30):
        """Classify relationships between related event pairs"""
        print(f"\n{'='*70}")
        print("RELATIONSHIP CLASSIFICATION")
        print(f"{'='*70}\n")

        pairs = self.find_related_pairs()
        print(f"Found {len(pairs)} related pairs")

        if not pairs:
            print("No related pairs found. Try lowering min_size or threshold.")
            return

        # Analyze top pairs
        for eu1, eu2, sim in pairs[:max_pairs]:
            print(f"\n--- Pair: sim={sim:.3f} ---")
            print(f"A ({eu1.size()} claims, mass={eu1.mass():.1f}): {eu1.label()}")
            print(f"B ({eu2.size()} claims, mass={eu2.mass():.1f}): {eu2.label()}")

            # Classify relationship
            self.llm_calls += 1
            result = llm_classify_relationship(
                eu1.description(),
                eu2.description(),
                eu1.mass(),
                eu2.mass()
            )

            rel_type = result.get('relationship', 'unknown')
            confidence = result.get('confidence', 0)
            reasoning = result.get('reasoning', '')
            parent_topic = result.get('parent_topic')

            print(f"  Relationship: {rel_type} (confidence: {confidence:.0%})")
            print(f"  Reasoning: {reasoning[:80]}...")
            if parent_topic:
                print(f"  Parent topic: {parent_topic}")

            self.relationship_results.append({
                'eu1_id': eu1.id,
                'eu1_label': eu1.label(),
                'eu1_mass': eu1.mass(),
                'eu1_size': eu1.size(),
                'eu2_id': eu2.id,
                'eu2_label': eu2.label(),
                'eu2_mass': eu2.mass(),
                'eu2_size': eu2.size(),
                'similarity': sim,
                'relationship': rel_type,
                'confidence': confidence,
                'reasoning': reasoning,
                'parent_topic': parent_topic
            })

    def test_mass_assumption(self, max_tests: int = 20):
        """Test whether mass-based containment assumption would be wrong"""
        print(f"\n{'='*70}")
        print("TESTING MASS-BASED CONTAINMENT ASSUMPTION")
        print(f"{'='*70}\n")

        print("Question: If we assume higher mass = contains lower mass, how often are we wrong?")
        print("(Like assuming Jupiter contains Earth because it has more mass)\n")

        pairs = self.find_related_pairs()

        wrong_count = 0
        tested = 0

        for eu1, eu2, sim in pairs[:max_tests]:
            # Skip if masses are very similar (ambiguous containment)
            mass_ratio = max(eu1.mass(), eu2.mass()) / min(eu1.mass(), eu2.mass())
            if mass_ratio < 1.5:
                continue

            self.llm_calls += 1
            result = llm_would_mass_containment_be_wrong(
                eu1.description(),
                eu2.description(),
                eu1.mass(),
                eu2.mass()
            )

            tested += 1
            would_be_wrong = result.get('mass_containment_would_be_wrong')
            reason = result.get('reason', '')
            better_rel = result.get('better_relationship', '')

            larger = eu1 if eu1.mass() > eu2.mass() else eu2
            smaller = eu2 if eu1.mass() > eu2.mass() else eu1

            print(f"\n--- Test {tested} ---")
            print(f"Larger (mass={larger.mass():.1f}): {larger.label()}")
            print(f"Smaller (mass={smaller.mass():.1f}): {smaller.label()}")
            print(f"Mass assumption 'larger contains smaller': {'WRONG' if would_be_wrong else 'OK'}")
            print(f"  Reason: {reason[:70]}...")
            print(f"  Better relationship: {better_rel}")

            if would_be_wrong:
                wrong_count += 1

            self.mass_assumption_tests.append({
                'larger_id': larger.id,
                'larger_label': larger.label(),
                'larger_mass': larger.mass(),
                'smaller_id': smaller.id,
                'smaller_label': smaller.label(),
                'smaller_mass': smaller.mass(),
                'similarity': sim,
                'mass_assumption_wrong': would_be_wrong,
                'reason': reason,
                'better_relationship': better_rel
            })

        if tested > 0:
            error_rate = wrong_count / tested
            print(f"\n{'='*50}")
            print(f"MASS ASSUMPTION ERROR RATE: {error_rate:.0%} ({wrong_count}/{tested})")
            print(f"{'='*50}")
            print(f"\nThis means in {error_rate:.0%} of cases, using mass alone")
            print("to determine containment would be semantically incorrect.")


def run_analysis(snapshot: GraphSnapshot):
    print("Loading cached embeddings...")
    cached = load_cached_embeddings()
    print(f"  Found {len(cached)} cached embeddings")

    all_claims = list(snapshot.claims.keys())
    random.seed(42)
    random.shuffle(all_claims)

    classifier = RelationshipClassifier(snapshot)

    # Build sub-events
    print(f"\n{'='*70}")
    print("Building sub-events from claims...")
    print(f"{'='*70}\n")

    for cid in all_claims:
        claim = snapshot.claims[cid]
        classifier.process_claim(cid, claim.text, claim.page_id or "?", cached[cid])

    level0 = [eu for eu in classifier.eus.values() if eu.level == 0]
    print(f"Created {len(level0)} sub-events")

    # Size distribution
    sizes = sorted([eu.size() for eu in level0], reverse=True)[:10]
    print(f"Top 10 sizes: {sizes}")

    # Phase 1: Classify relationships
    classifier.classify_relationships(max_pairs=30)

    # Phase 2: Test mass assumption
    classifier.test_mass_assumption(max_tests=20)

    return classifier


def main():
    print("=" * 70)
    print("RELATIONSHIP TYPE CLASSIFICATION EXPERIMENT")
    print("Testing: Mass determines importance, NOT containment")
    print("=" * 70 + "\n")

    snapshot = load_snapshot()
    print(f"Loaded {len(snapshot.claims)} claims\n")

    classifier = run_analysis(snapshot)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    if classifier.relationship_results:
        rel_counts = {}
        for r in classifier.relationship_results:
            rt = r['relationship']
            rel_counts[rt] = rel_counts.get(rt, 0) + 1

        print("Relationship type distribution:")
        for rt, count in sorted(rel_counts.items(), key=lambda x: -x[1]):
            pct = count / len(classifier.relationship_results) * 100
            print(f"  {rt}: {count} ({pct:.0f}%)")

        # Key finding: SIBLING vs CONTAINS
        sibling_count = rel_counts.get('sibling', 0)
        contains_count = rel_counts.get('contains', 0) + rel_counts.get('contained_by', 0)

        print(f"\nKEY FINDING:")
        print(f"  SIBLING relationships: {sibling_count}")
        print(f"  CONTAINS relationships: {contains_count}")
        print(f"  Ratio: {sibling_count / max(1, contains_count):.1f}x more siblings than containment")

        if sibling_count > contains_count:
            print("\n  -> Most related events are SIBLINGS, not parent-child!")
            print("  -> Mass-based containment assumption would be WRONG most of the time")

    if classifier.mass_assumption_tests:
        wrong = sum(1 for t in classifier.mass_assumption_tests if t['mass_assumption_wrong'])
        total = len(classifier.mass_assumption_tests)
        print(f"\nMass assumption test:")
        print(f"  Would be wrong: {wrong}/{total} ({wrong/total*100:.0f}%)")

    # Save results
    output = {
        'relationship_classifications': classifier.relationship_results,
        'mass_assumption_tests': classifier.mass_assumption_tests,
        'summary': {
            'total_subevents': len([eu for eu in classifier.eus.values() if eu.level == 0]),
            'pairs_classified': len(classifier.relationship_results),
            'mass_tests': len(classifier.mass_assumption_tests),
            'llm_calls': classifier.llm_calls
        }
    }

    output_path = Path("/app/test_eu/results/relationship_classification.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
