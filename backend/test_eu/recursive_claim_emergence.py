"""
Recursive Claim Emergence

The ultimate test: Can higher-order knowledge emerge from ground claims?

Process:
1. Start with ground claims (L0) - raw observations from sources
2. Cluster by semantic similarity + entity overlap
3. Use LLM to synthesize emergent pattern claims (L1)
4. Repeat: cluster L1 claims → emerge L2 claims
5. Continue until no more emergence possible
6. Display the full knowledge hierarchy

This is the scientific method made visible:
  Observations → Patterns → Hypotheses → Theories → Paradigms

Run inside container:
    docker exec herenews-app python /app/test_eu/recursive_claim_emergence.py
"""

import json
import os
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import hashlib

sys.path.insert(0, '/app/backend')

from openai import OpenAI
from load_graph import load_snapshot, GraphSnapshot, ClaimData


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EmergentClaim:
    """A claim at any abstraction level"""
    id: str
    text: str
    level: int  # 0 = ground, 1+ = emergent

    # What supports this claim
    supported_by: List[str] = field(default_factory=list)  # claim IDs
    support_texts: List[str] = field(default_factory=list)  # for display

    # Epistemic status
    entropy: float = 1.0
    n_independent_sources: int = 0

    # Metadata
    entities: List[str] = field(default_factory=list)
    cluster_id: Optional[str] = None


@dataclass
class EmergenceResult:
    """Result of one emergence iteration"""
    level: int
    input_claims: int
    clusters_found: int
    claims_emerged: int
    emerged_claims: List[EmergentClaim]


# =============================================================================
# EMERGENCE ENGINE
# =============================================================================

class RecursiveEmergenceEngine:
    """Engine for recursively emerging higher-order claims"""

    def __init__(self, snapshot: GraphSnapshot):
        self.snapshot = snapshot
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self._embedding_cache = {}

        # All claims by level
        self.claims_by_level: Dict[int, List[EmergentClaim]] = defaultdict(list)

        # Full graph
        self.all_claims: Dict[str, EmergentClaim] = {}

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding with caching"""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key not in self._embedding_cache:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text[:8000]
            )
            self._embedding_cache[cache_key] = response.data[0].embedding
        return self._embedding_cache[cache_key]

    def cosine_sim(self, a: List[float], b: List[float]) -> float:
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # =========================================================================
    # STEP 1: INITIALIZE FROM GROUND CLAIMS
    # =========================================================================

    def initialize_ground_claims(self) -> int:
        """Load ground claims (L0) from snapshot"""
        print("\n" + "=" * 60)
        print("LEVEL 0: GROUND CLAIMS (Observations)")
        print("=" * 60)

        for cid, claim in self.snapshot.claims.items():
            ec = EmergentClaim(
                id=cid,
                text=claim.text,
                level=0,
                supported_by=[],  # Ground claims have no support
                entities=claim.entity_ids,
                n_independent_sources=1,  # Each is one source
            )

            # Entropy based on corroboration in original data
            n_corr = len(claim.corroborated_by_ids)
            n_contra = len(claim.contradicted_by_ids)
            ec.entropy = self._compute_entropy(n_corr, n_contra)

            self.claims_by_level[0].append(ec)
            self.all_claims[cid] = ec

        print(f"  Loaded {len(self.claims_by_level[0])} ground claims")

        # Show sample
        print("\n  Sample ground claims:")
        for ec in self.claims_by_level[0][:5]:
            print(f"    [{ec.entropy:.2f}] {ec.text[:70]}...")

        return len(self.claims_by_level[0])

    def _compute_entropy(self, n_corr: int, n_contra: int, independence: float = 0.35) -> float:
        """Calibrated entropy formula"""
        base = 1.0
        effective_corr = n_corr * independence

        if effective_corr > 0:
            corr_reduction = 0.49 * (effective_corr ** 0.30)
        else:
            corr_reduction = 0.0

        if n_contra > 0:
            contra_addition = 0.27 * (n_contra ** 0.31)
        else:
            contra_addition = 0.0

        return max(0.05, min(0.99, base - corr_reduction + contra_addition))

    # =========================================================================
    # STEP 2: CLUSTER CLAIMS
    # =========================================================================

    def cluster_claims(self, claims: List[EmergentClaim], threshold: float = 0.75) -> List[List[EmergentClaim]]:
        """Cluster claims by semantic similarity"""
        if not claims:
            return []

        # Get embeddings
        embeddings = {}
        for claim in claims:
            embeddings[claim.id] = self.get_embedding(claim.text)

        # Simple greedy clustering
        clusters = []
        assigned = set()

        for claim in claims:
            if claim.id in assigned:
                continue

            # Start new cluster
            cluster = [claim]
            assigned.add(claim.id)

            # Find similar claims
            for other in claims:
                if other.id in assigned:
                    continue

                sim = self.cosine_sim(embeddings[claim.id], embeddings[other.id])

                # Also check entity overlap
                entity_overlap = len(set(claim.entities) & set(other.entities)) > 0

                if sim >= threshold or (sim >= 0.6 and entity_overlap):
                    cluster.append(other)
                    assigned.add(other.id)

            if len(cluster) >= 2:  # Only keep clusters with 2+ claims
                clusters.append(cluster)

        return clusters

    # =========================================================================
    # STEP 3: EMERGE HIGHER-ORDER CLAIMS
    # =========================================================================

    def emerge_from_cluster(self, cluster: List[EmergentClaim], target_level: int) -> Optional[EmergentClaim]:
        """Use LLM to synthesize an emergent claim from a cluster"""

        claim_texts = "\n".join([f"- {c.text}" for c in cluster[:10]])

        prompt = f"""You are analyzing a cluster of related claims to identify a higher-order pattern.

CLAIMS IN CLUSTER:
{claim_texts}

These claims all support some higher-order assertion. What is it?

Rules:
1. The emergent claim should be MORE ABSTRACT than the individual claims
2. It should be SUPPORTED BY all/most claims in the cluster
3. It should capture the PATTERN or THEME they share
4. It should be a single, clear assertion

Abstraction level guide:
- Level 1: Pattern across observations ("Multiple sources report X")
- Level 2: Interpretation/analysis ("The evidence suggests X")
- Level 3: Broader implication ("This indicates a pattern of X")
- Level 4+: Systemic/theoretical claim ("This reflects X")

Current target level: {target_level}

Respond in JSON:
{{
  "emergent_claim": "the higher-order claim that these support",
  "abstraction_type": "PATTERN|INTERPRETATION|IMPLICATION|SYSTEMIC",
  "confidence": 0.0-1.0,
  "key_entities": ["entities involved"],
  "what_would_contradict": "what evidence would undermine this claim"
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            result = json.loads(response.choices[0].message.content)

            # Create emergent claim
            claim_id = f"L{target_level}_{hashlib.md5(result['emergent_claim'].encode()).hexdigest()[:8]}"

            # Compute entropy from supporting claims
            support_entropies = [c.entropy for c in cluster]
            n_supporters = len(cluster)

            # Emergent claim's entropy depends on:
            # 1. How many claims support it
            # 2. Their independence
            # 3. Confidence of emergence
            base_entropy = 0.8 - (result.get('confidence', 0.5) * 0.3)
            support_reduction = 0.15 * np.log1p(n_supporters)
            inherited_uncertainty = np.mean(support_entropies) * 0.2  # Inherit some uncertainty

            entropy = max(0.1, min(0.95, base_entropy - support_reduction + inherited_uncertainty))

            ec = EmergentClaim(
                id=claim_id,
                text=result['emergent_claim'],
                level=target_level,
                supported_by=[c.id for c in cluster],
                support_texts=[c.text[:50] for c in cluster[:5]],
                entropy=entropy,
                n_independent_sources=n_supporters,
                entities=result.get('key_entities', []),
            )

            return ec

        except Exception as e:
            print(f"    Error emerging claim: {e}")
            return None

    # =========================================================================
    # STEP 4: RUN ONE EMERGENCE ITERATION
    # =========================================================================

    def run_emergence_iteration(self, source_level: int) -> EmergenceResult:
        """Run one iteration of emergence from source_level to source_level+1"""

        target_level = source_level + 1
        source_claims = self.claims_by_level[source_level]

        print(f"\n" + "=" * 60)
        print(f"LEVEL {target_level}: EMERGING FROM {len(source_claims)} L{source_level} CLAIMS")
        print("=" * 60)

        if len(source_claims) < 2:
            print("  Not enough claims to cluster")
            return EmergenceResult(
                level=target_level,
                input_claims=len(source_claims),
                clusters_found=0,
                claims_emerged=0,
                emerged_claims=[]
            )

        # Cluster source claims
        print(f"\n  Clustering L{source_level} claims...")
        clusters = self.cluster_claims(source_claims, threshold=0.70)
        print(f"  Found {len(clusters)} clusters")

        # Emerge from each cluster
        emerged = []

        for i, cluster in enumerate(clusters[:20]):  # Limit to 20 clusters
            print(f"\n  Cluster {i+1} ({len(cluster)} claims):")
            for c in cluster[:3]:
                print(f"    - {c.text[:60]}...")
            if len(cluster) > 3:
                print(f"    ... and {len(cluster) - 3} more")

            ec = self.emerge_from_cluster(cluster, target_level)

            if ec:
                print(f"\n  → EMERGED: {ec.text[:70]}...")
                print(f"    Entropy: {ec.entropy:.2f}, Supporters: {len(ec.supported_by)}")

                emerged.append(ec)
                self.claims_by_level[target_level].append(ec)
                self.all_claims[ec.id] = ec

        print(f"\n  Emerged {len(emerged)} L{target_level} claims")

        return EmergenceResult(
            level=target_level,
            input_claims=len(source_claims),
            clusters_found=len(clusters),
            claims_emerged=len(emerged),
            emerged_claims=emerged
        )

    # =========================================================================
    # STEP 5: FULL RECURSIVE EMERGENCE
    # =========================================================================

    def run_full_emergence(self, max_levels: int = 5) -> Dict:
        """Run full recursive emergence until no more claims emerge"""

        results = []

        # Initialize
        self.initialize_ground_claims()

        # Recursively emerge
        for level in range(max_levels):
            result = self.run_emergence_iteration(level)
            results.append(result)

            if result.claims_emerged == 0:
                print(f"\n  No more emergence possible at level {level + 1}")
                break

        return {
            'iterations': results,
            'total_levels': len([r for r in results if r.claims_emerged > 0]) + 1,
            'claims_by_level': {k: len(v) for k, v in self.claims_by_level.items()},
        }

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def display_hierarchy(self):
        """Display the full emerged hierarchy"""

        print("\n" + "=" * 70)
        print("EMERGED KNOWLEDGE HIERARCHY")
        print("=" * 70)

        max_level = max(self.claims_by_level.keys())

        for level in range(max_level, -1, -1):
            claims = self.claims_by_level[level]

            if level == 0:
                level_name = "GROUND (Observations)"
            elif level == 1:
                level_name = "PATTERNS"
            elif level == 2:
                level_name = "INTERPRETATIONS"
            elif level == 3:
                level_name = "IMPLICATIONS"
            else:
                level_name = f"SYSTEMIC (L{level})"

            print(f"\n{'─' * 70}")
            print(f"LEVEL {level}: {level_name} ({len(claims)} claims)")
            print(f"{'─' * 70}")

            # Show claims at this level
            display_claims = claims[:10] if level > 0 else claims[:5]

            for claim in display_claims:
                entropy_bar = "█" * int((1 - claim.entropy) * 10) + "░" * int(claim.entropy * 10)
                print(f"\n  [{entropy_bar}] H={claim.entropy:.2f}")
                print(f"  {claim.text[:80]}{'...' if len(claim.text) > 80 else ''}")

                if claim.supported_by:
                    print(f"  ↑ Supported by {len(claim.supported_by)} L{claim.level - 1} claims")

            if len(claims) > len(display_claims):
                print(f"\n  ... and {len(claims) - len(display_claims)} more L{level} claims")

    def display_sample_chains(self, n_chains: int = 3):
        """Show sample evidence chains from top to bottom"""

        print("\n" + "=" * 70)
        print("SAMPLE EVIDENCE CHAINS (Top → Ground)")
        print("=" * 70)

        max_level = max(self.claims_by_level.keys())
        top_claims = self.claims_by_level[max_level][:n_chains]

        for i, top_claim in enumerate(top_claims):
            print(f"\n{'─' * 70}")
            print(f"CHAIN {i + 1}")
            print(f"{'─' * 70}")

            self._display_chain(top_claim, 0)

    def _display_chain(self, claim: EmergentClaim, indent: int):
        """Recursively display a claim and its support"""
        prefix = "  " * indent

        level_label = f"L{claim.level}"
        print(f"{prefix}[{level_label}] {claim.text[:70 - indent*2]}...")

        if claim.supported_by:
            # Show first few supporters
            for sid in claim.supported_by[:3]:
                if sid in self.all_claims:
                    supporter = self.all_claims[sid]
                    self._display_chain(supporter, indent + 1)

            if len(claim.supported_by) > 3:
                print(f"{prefix}  ... +{len(claim.supported_by) - 3} more")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("RECURSIVE CLAIM EMERGENCE")
    print("Watching knowledge structure itself")
    print("=" * 70)

    # Load data
    snapshot = load_snapshot()
    print(f"\nLoaded: {len(snapshot.claims)} ground claims")

    # Run emergence
    engine = RecursiveEmergenceEngine(snapshot)
    results = engine.run_full_emergence(max_levels=4)

    # Display hierarchy
    engine.display_hierarchy()

    # Display sample chains
    engine.display_sample_chains(n_chains=3)

    # Summary
    print("\n" + "=" * 70)
    print("EMERGENCE SUMMARY")
    print("=" * 70)

    print(f"\nTotal levels emerged: {results['total_levels']}")
    print("\nClaims by level:")
    for level, count in sorted(results['claims_by_level'].items()):
        if level == 0:
            name = "Ground observations"
        elif level == 1:
            name = "Patterns"
        elif level == 2:
            name = "Interpretations"
        elif level == 3:
            name = "Implications"
        else:
            name = f"Systemic (L{level})"
        print(f"  L{level} ({name}): {count} claims")

    # Theoretical reflection
    print(f"""
{'=' * 70}
WHAT THIS DEMONSTRATES
{'=' * 70}

This is the scientific method in action:

  L0: Observations    → Raw claims from sources
  L1: Patterns        → "Multiple sources report X"
  L2: Interpretations → "The evidence suggests X"
  L3: Implications    → "This indicates a pattern of X"
  L4+: Systemic       → "This reflects a broader phenomenon"

Each level is:
  - Emerged from lower levels (not assigned)
  - Supported by evidence (traced to ground)
  - Has entropy (uncertainty that propagates)

This is how all knowledge works:
  Physics: measurement → law → theory
  Medicine: symptom → syndrome → diagnosis
  Law: evidence → argument → verdict
  History: source → interpretation → narrative

We just watched it happen computationally.
""")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'ground_claims': len(snapshot.claims),
        'levels_emerged': results['total_levels'],
        'claims_by_level': results['claims_by_level'],
        'sample_emerged': [
            {
                'level': c.level,
                'text': c.text,
                'entropy': c.entropy,
                'n_supporters': len(c.supported_by),
            }
            for level_claims in engine.claims_by_level.values()
            for c in level_claims[:5]
            if c.level > 0
        ]
    }

    output_path = Path("/app/test_eu/results/recursive_emergence_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
