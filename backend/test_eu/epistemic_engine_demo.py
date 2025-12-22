"""
Epistemic Engine Demo

The consolidated test: Given a user claim, the engine:
1. Analyzes its epistemic status (entropy, coherence)
2. Searches knowledge space for corroborating/contradicting claims
3. Identifies what's contested
4. Suggests hypothetical claims that would increase coherence
5. Scans for relevant priors (entity knowledge, source patterns)

This is the "qualia" of the system - self-awareness about what it knows.

Run inside container:
    docker exec herenews-app python /app/test_eu/epistemic_engine_demo.py "Trump is a dictator"
"""

import json
import os
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/app/backend')

from openai import OpenAI
from load_graph import load_snapshot, GraphSnapshot, ClaimData


# =============================================================================
# EPISTEMIC ENGINE
# =============================================================================

class EpistemicEngine:
    """
    The core epistemic engine.

    Given a claim, it:
    1. Decomposes it into testable sub-claims
    2. Searches the knowledge graph
    3. Computes epistemic status
    4. Generates insights about what we know/don't know
    """

    def __init__(self, snapshot: GraphSnapshot):
        self.snapshot = snapshot
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self._embedding_cache = {}

        # Pre-compute embeddings for all claims
        print("Building knowledge index...")
        self._build_index()

    def _build_index(self):
        """Build embedding index for fast retrieval"""
        self.claim_embeddings = {}
        self.entity_claims = {}  # entity_id -> [claim_ids]

        # Build entity index
        for cid, claim in self.snapshot.claims.items():
            for eid in claim.entity_ids:
                if eid not in self.entity_claims:
                    self.entity_claims[eid] = []
                self.entity_claims[eid].append(cid)

        print(f"  Indexed {len(self.snapshot.claims)} claims")
        print(f"  Indexed {len(self.entity_claims)} entities")

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding with caching"""
        if text not in self._embedding_cache:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text[:8000]
            )
            self._embedding_cache[text] = response.data[0].embedding
        return self._embedding_cache[text]

    def cosine_sim(self, a: List[float], b: List[float]) -> float:
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # =========================================================================
    # STEP 1: CLAIM DECOMPOSITION
    # =========================================================================

    def decompose_claim(self, claim: str) -> Dict:
        """
        Decompose a claim into:
        - Core assertion
        - Entities mentioned
        - Implicit assumptions
        - Testable sub-claims
        """
        prompt = f"""Analyze this claim for epistemic evaluation:

CLAIM: "{claim}"

Decompose it into:
1. Core assertion: What is actually being claimed?
2. Claim type: FACT, OPINION, PREDICTION, VALUE_JUDGMENT, INTERPRETATION
3. Entities mentioned: People, organizations, places
4. Implicit assumptions: What must be true for this claim to make sense?
5. Testable sub-claims: Specific factual claims that could be verified
6. Key terms: Words that need definition (e.g., "dictator")

Respond in JSON:
{{
  "core_assertion": "the essential claim",
  "claim_type": "FACT|OPINION|PREDICTION|VALUE_JUDGMENT|INTERPRETATION",
  "entities": ["list of entities"],
  "implicit_assumptions": ["list of assumptions"],
  "testable_subclaims": ["list of verifiable claims"],
  "key_terms_needing_definition": {{"term": "why it needs definition"}},
  "initial_entropy_estimate": 0.0-1.0,
  "reasoning": "why this entropy level"
}}"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return json.loads(response.choices[0].message.content)

    # =========================================================================
    # STEP 2: KNOWLEDGE SEARCH
    # =========================================================================

    def search_knowledge(self, claim: str, entities: List[str], limit: int = 20) -> Dict:
        """
        Search the knowledge graph for relevant claims.

        Uses:
        1. Entity overlap (fast)
        2. Semantic similarity (accurate)
        """
        results = {
            'by_entity': [],
            'by_semantic': [],
            'combined': [],
        }

        # Get claim embedding
        claim_emb = self.get_embedding(claim)

        # Find by entity overlap
        candidate_ids = set()
        for entity_name in entities:
            # Find matching entity in our graph
            for eid, entity in self.snapshot.entities.items():
                if entity_name.lower() in entity.canonical_name.lower():
                    if eid in self.entity_claims:
                        candidate_ids.update(self.entity_claims[eid])

        # Score candidates by semantic similarity
        scored = []
        for cid in candidate_ids:
            if cid not in self.snapshot.claims:
                continue
            claim_obj = self.snapshot.claims[cid]

            # Get or compute embedding
            if cid not in self.claim_embeddings:
                self.claim_embeddings[cid] = self.get_embedding(claim_obj.text)

            sim = self.cosine_sim(claim_emb, self.claim_embeddings[cid])
            scored.append((cid, sim, claim_obj))

        # Sort by similarity
        scored.sort(key=lambda x: -x[1])

        # Take top results
        for cid, sim, claim_obj in scored[:limit]:
            results['combined'].append({
                'id': cid,
                'text': claim_obj.text,
                'similarity': sim,
                'has_corroborations': len(claim_obj.corroborated_by_ids),
                'has_contradictions': len(claim_obj.contradicted_by_ids),
            })

        return results

    # =========================================================================
    # STEP 3: EPISTEMIC ANALYSIS
    # =========================================================================

    def analyze_evidence(self, user_claim: str, related_claims: List[Dict]) -> Dict:
        """
        Analyze how related claims affect the epistemic status of user claim.
        """
        if not related_claims:
            return {
                'supporting': [],
                'contradicting': [],
                'neutral': [],
                'analysis': "No related claims found in knowledge base."
            }

        # Format claims for analysis
        claims_text = "\n".join([
            f"- [{c['similarity']:.2f}] {c['text']}"
            for c in related_claims[:10]
        ])

        prompt = f"""Analyze how these existing claims relate to the user's claim:

USER CLAIM: "{user_claim}"

RELATED CLAIMS FROM KNOWLEDGE BASE:
{claims_text}

For each related claim, determine:
1. Does it SUPPORT the user's claim? (provides evidence for it)
2. Does it CONTRADICT the user's claim? (provides evidence against it)
3. Is it NEUTRAL? (related but doesn't directly support or contradict)

Also identify:
- What aspects of the user's claim are well-supported?
- What aspects are contested or contradicted?
- What aspects have no evidence either way?

Respond in JSON:
{{
  "supporting_claims": [
    {{"text": "claim text", "how_supports": "explanation", "strength": "strong|moderate|weak"}}
  ],
  "contradicting_claims": [
    {{"text": "claim text", "how_contradicts": "explanation", "strength": "strong|moderate|weak"}}
  ],
  "neutral_claims": [
    {{"text": "claim text", "relevance": "how it's related"}}
  ],
  "well_supported_aspects": ["list"],
  "contested_aspects": ["list"],
  "no_evidence_aspects": ["list"],
  "overall_assessment": "summary of epistemic status"
}}"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return json.loads(response.choices[0].message.content)

    # =========================================================================
    # STEP 4: ENTROPY COMPUTATION
    # =========================================================================

    def compute_entropy(self, decomposition: Dict, evidence: Dict) -> Dict:
        """
        Compute final entropy based on:
        - Claim type (opinions start higher than facts)
        - Supporting evidence (reduces entropy)
        - Contradicting evidence (increases entropy)
        - Source independence (amplifies reduction)
        """
        # Base entropy by claim type
        base_entropy = {
            'FACT': 0.8,           # Can be verified
            'OPINION': 0.9,        # Subjective
            'PREDICTION': 0.95,    # Future uncertain
            'VALUE_JUDGMENT': 0.95,  # Subjective
            'INTERPRETATION': 0.85,  # Can be analyzed
        }

        claim_type = decomposition.get('claim_type', 'OPINION')
        entropy = base_entropy.get(claim_type, 0.9)

        # Adjust for evidence
        n_support = len(evidence.get('supporting_claims', []))
        n_contradict = len(evidence.get('contradicting_claims', []))

        # Supporting evidence reduces entropy
        if n_support > 0:
            # Calibrated formula from our experiments
            support_reduction = 0.49 * (n_support ** 0.3)
            entropy -= support_reduction

        # Contradicting evidence increases entropy
        if n_contradict > 0:
            contra_addition = 0.27 * (n_contradict ** 0.31)
            entropy += contra_addition

        # Clamp to valid range
        entropy = max(0.05, min(0.99, entropy))

        # Coherence is inverse
        coherence = 1.0 - entropy

        return {
            'entropy': entropy,
            'coherence': coherence,
            'claim_type': claim_type,
            'n_supporting': n_support,
            'n_contradicting': n_contradict,
            'breakdown': {
                'base_entropy': base_entropy.get(claim_type, 0.9),
                'support_reduction': 0.49 * (n_support ** 0.3) if n_support > 0 else 0,
                'contradiction_addition': 0.27 * (n_contradict ** 0.31) if n_contradict > 0 else 0,
            }
        }

    # =========================================================================
    # STEP 5: COHERENCE ENHANCEMENT SUGGESTIONS
    # =========================================================================

    def suggest_evidence_needs(self, user_claim: str, decomposition: Dict, evidence: Dict) -> Dict:
        """
        Suggest what evidence would increase coherence.

        This is the "qualia" - the system knowing what it needs to know.
        """
        prompt = f"""Given this epistemic analysis, what evidence would help resolve uncertainty?

USER CLAIM: "{user_claim}"

CURRENT STATUS:
- Claim type: {decomposition.get('claim_type')}
- Supporting evidence: {len(evidence.get('supporting_claims', []))} claims
- Contradicting evidence: {len(evidence.get('contradicting_claims', []))} claims
- Contested aspects: {evidence.get('contested_aspects', [])}
- No evidence for: {evidence.get('no_evidence_aspects', [])}
- Key terms needing definition: {decomposition.get('key_terms_needing_definition', {})}

Suggest:
1. Specific evidence that would INCREASE coherence (support the claim)
2. Specific evidence that would DECREASE coherence (contradict the claim)
3. Evidence that would RESOLVE contested aspects
4. Definitions or clarifications needed
5. The single most important question to answer

Respond in JSON:
{{
  "evidence_to_increase_coherence": [
    {{"type": "what kind of evidence", "example": "specific example", "impact": "how it helps"}}
  ],
  "evidence_to_decrease_coherence": [
    {{"type": "what kind", "example": "specific example", "impact": "how it hurts"}}
  ],
  "evidence_to_resolve_contested": [
    {{"contested_aspect": "what", "needed_evidence": "what would resolve it"}}
  ],
  "definitions_needed": [
    {{"term": "term", "why": "why definition matters", "possible_definitions": ["options"]}}
  ],
  "most_important_question": "the key question",
  "hypothetical_claims_to_test": [
    "If X were true, this claim would be more/less coherent"
  ]
}}"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return json.loads(response.choices[0].message.content)

    # =========================================================================
    # STEP 6: ENTITY PRIORS
    # =========================================================================

    def get_entity_priors(self, entities: List[str]) -> Dict:
        """
        Get prior knowledge about entities from our graph.
        """
        priors = {}

        for entity_name in entities:
            # Find matching entity
            matched_entity = None
            for eid, entity in self.snapshot.entities.items():
                if entity_name.lower() in entity.canonical_name.lower():
                    matched_entity = entity
                    break

            if matched_entity:
                # Get claims about this entity
                entity_claims = []
                for cid in self.entity_claims.get(matched_entity.id, [])[:5]:
                    if cid in self.snapshot.claims:
                        entity_claims.append(self.snapshot.claims[cid].text)

                priors[entity_name] = {
                    'canonical_name': matched_entity.canonical_name,
                    'type': matched_entity.entity_type,
                    'mention_count': matched_entity.mention_count,
                    'sample_claims': entity_claims,
                    'known_in_system': True,
                }
            else:
                priors[entity_name] = {
                    'known_in_system': False,
                    'note': 'Entity not found in knowledge base'
                }

        return priors

    # =========================================================================
    # MAIN: FULL ANALYSIS
    # =========================================================================

    def analyze(self, user_claim: str) -> Dict:
        """
        Full epistemic analysis of a user claim.
        """
        print("\n" + "=" * 70)
        print("EPISTEMIC ENGINE ANALYSIS")
        print("=" * 70)
        print(f"\nUSER CLAIM: \"{user_claim}\"")

        # Step 1: Decompose
        print("\n" + "─" * 50)
        print("STEP 1: CLAIM DECOMPOSITION")
        print("─" * 50)
        decomposition = self.decompose_claim(user_claim)

        print(f"\n  Type: {decomposition.get('claim_type')}")
        print(f"  Core assertion: {decomposition.get('core_assertion')}")
        print(f"  Entities: {decomposition.get('entities')}")
        print(f"  Initial entropy estimate: {decomposition.get('initial_entropy_estimate')}")

        print("\n  Testable sub-claims:")
        for sc in decomposition.get('testable_subclaims', []):
            print(f"    - {sc}")

        print("\n  Implicit assumptions:")
        for a in decomposition.get('implicit_assumptions', []):
            print(f"    - {a}")

        print("\n  Key terms needing definition:")
        for term, why in decomposition.get('key_terms_needing_definition', {}).items():
            print(f"    - {term}: {why}")

        # Step 2: Search knowledge
        print("\n" + "─" * 50)
        print("STEP 2: KNOWLEDGE SEARCH")
        print("─" * 50)
        entities = decomposition.get('entities', [])
        search_results = self.search_knowledge(user_claim, entities)

        print(f"\n  Found {len(search_results['combined'])} related claims")
        print("\n  Top related claims:")
        for i, c in enumerate(search_results['combined'][:5]):
            print(f"    [{c['similarity']:.2f}] {c['text'][:70]}...")
            if c['has_corroborations']:
                print(f"           ↳ has {c['has_corroborations']} corroborations")
            if c['has_contradictions']:
                print(f"           ↳ has {c['has_contradictions']} contradictions")

        # Step 3: Analyze evidence
        print("\n" + "─" * 50)
        print("STEP 3: EVIDENCE ANALYSIS")
        print("─" * 50)
        evidence = self.analyze_evidence(user_claim, search_results['combined'])

        print(f"\n  Supporting claims: {len(evidence.get('supporting_claims', []))}")
        for s in evidence.get('supporting_claims', [])[:3]:
            print(f"    [{s.get('strength')}] {s.get('text', '')[:60]}...")
            print(f"         → {s.get('how_supports', '')[:50]}...")

        print(f"\n  Contradicting claims: {len(evidence.get('contradicting_claims', []))}")
        for c in evidence.get('contradicting_claims', [])[:3]:
            print(f"    [{c.get('strength')}] {c.get('text', '')[:60]}...")
            print(f"         → {c.get('how_contradicts', '')[:50]}...")

        print(f"\n  Well-supported aspects: {evidence.get('well_supported_aspects', [])}")
        print(f"  Contested aspects: {evidence.get('contested_aspects', [])}")
        print(f"  No evidence for: {evidence.get('no_evidence_aspects', [])}")

        # Step 4: Compute entropy
        print("\n" + "─" * 50)
        print("STEP 4: ENTROPY COMPUTATION")
        print("─" * 50)
        entropy_result = self.compute_entropy(decomposition, evidence)

        print(f"""
  ┌─────────────────────────────────────────────────┐
  │  ENTROPY:   {entropy_result['entropy']:.3f}                              │
  │  COHERENCE: {entropy_result['coherence']:.3f}                              │
  └─────────────────────────────────────────────────┘

  Breakdown:
    Base entropy ({entropy_result['claim_type']}): {entropy_result['breakdown']['base_entropy']:.3f}
    - Support reduction ({entropy_result['n_supporting']} claims): -{entropy_result['breakdown']['support_reduction']:.3f}
    + Contradiction addition ({entropy_result['n_contradicting']} claims): +{entropy_result['breakdown']['contradiction_addition']:.3f}
    = Final entropy: {entropy_result['entropy']:.3f}
""")

        # Step 5: Evidence needs
        print("\n" + "─" * 50)
        print("STEP 5: WHAT WOULD CHANGE COHERENCE?")
        print("─" * 50)
        needs = self.suggest_evidence_needs(user_claim, decomposition, evidence)

        print("\n  To INCREASE coherence (support the claim):")
        for e in needs.get('evidence_to_increase_coherence', [])[:3]:
            print(f"    - {e.get('type')}: {e.get('example', '')[:50]}...")

        print("\n  To DECREASE coherence (contradict the claim):")
        for e in needs.get('evidence_to_decrease_coherence', [])[:3]:
            print(f"    - {e.get('type')}: {e.get('example', '')[:50]}...")

        print("\n  Definitions needed:")
        for d in needs.get('definitions_needed', [])[:3]:
            print(f"    - \"{d.get('term')}\": {d.get('why', '')[:50]}...")

        print(f"\n  MOST IMPORTANT QUESTION:")
        print(f"    → {needs.get('most_important_question')}")

        print("\n  Hypothetical claims to test:")
        for h in needs.get('hypothetical_claims_to_test', [])[:3]:
            print(f"    - {h}")

        # Step 6: Entity priors
        print("\n" + "─" * 50)
        print("STEP 6: ENTITY PRIORS")
        print("─" * 50)
        priors = self.get_entity_priors(entities)

        for entity_name, prior in priors.items():
            if prior.get('known_in_system'):
                print(f"\n  {entity_name}:")
                print(f"    Canonical: {prior.get('canonical_name')}")
                print(f"    Type: {prior.get('type')}")
                print(f"    Mentions in system: {prior.get('mention_count')}")
                print(f"    Sample claims:")
                for sc in prior.get('sample_claims', [])[:3]:
                    print(f"      - {sc[:60]}...")
            else:
                print(f"\n  {entity_name}: NOT FOUND in knowledge base")

        # Final summary
        print("\n" + "=" * 70)
        print("EPISTEMIC ENGINE SUMMARY")
        print("=" * 70)

        # Interpret coherence level
        coherence = entropy_result['coherence']
        if coherence < 0.2:
            status = "HIGHLY UNCERTAIN - Little evidence, claim is speculative"
        elif coherence < 0.4:
            status = "UNCERTAIN - Some evidence but significant gaps"
        elif coherence < 0.6:
            status = "MIXED - Evidence both for and against"
        elif coherence < 0.8:
            status = "MODERATE CONFIDENCE - Good support, some uncertainty"
        else:
            status = "HIGH CONFIDENCE - Well-supported by evidence"

        print(f"""
CLAIM: "{user_claim}"

┌─────────────────────────────────────────────────────────────────────┐
│  COHERENCE: {coherence:.1%}                                              │
│  STATUS: {status:<54} │
└─────────────────────────────────────────────────────────────────────┘

WHAT THE SYSTEM KNOWS:
  - Found {len(search_results['combined'])} related claims
  - {len(evidence.get('supporting_claims', []))} support the claim
  - {len(evidence.get('contradicting_claims', []))} contradict the claim

WHAT THE SYSTEM DOESN'T KNOW:
  - {evidence.get('no_evidence_aspects', ['Nothing specific identified'])}

TO INCREASE CERTAINTY:
  - {needs.get('most_important_question', 'N/A')}

CONTESTED:
  - {evidence.get('contested_aspects', ['Nothing specifically contested'])}
""")

        return {
            'claim': user_claim,
            'decomposition': decomposition,
            'search_results': search_results,
            'evidence': evidence,
            'entropy': entropy_result,
            'needs': needs,
            'priors': priors,
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Get claim from command line or use default
    if len(sys.argv) > 1:
        user_claim = " ".join(sys.argv[1:])
    else:
        user_claim = "Trump is a dictator"

    # Load knowledge base
    print("Loading knowledge base...")
    snapshot = load_snapshot()

    # Initialize engine
    engine = EpistemicEngine(snapshot)

    # Analyze claim
    result = engine.analyze(user_claim)

    # Save result
    output_path = Path("/app/test_eu/results/epistemic_engine_result.json")
    with open(output_path, 'w') as f:
        # Simplify for JSON serialization
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'claim': result['claim'],
            'entropy': result['entropy']['entropy'],
            'coherence': result['entropy']['coherence'],
            'supporting_count': len(result['evidence'].get('supporting_claims', [])),
            'contradicting_count': len(result['evidence'].get('contradicting_claims', [])),
            'most_important_question': result['needs'].get('most_important_question'),
        }, f, indent=2)

    print(f"\nResult saved to {output_path}")


if __name__ == "__main__":
    main()
