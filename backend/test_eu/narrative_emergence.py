"""
Narrative Emergence

Enhanced recursive emergence that produces NARRATIVES, not just claims.

Each emergent level is a structured narrative:
- What we know (supported facts)
- What is contested (conflicting claims)
- What we don't know (gaps)
- What would help (needed evidence)
- Key question (the unresolved core)

This is the "Event" as epistemic organism - self-aware about its knowledge state.

Run inside container:
    docker exec herenews-app python /app/test_eu/narrative_emergence.py
"""

import json
import os
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import hashlib

sys.path.insert(0, '/app/backend')

from openai import OpenAI
from load_graph import load_snapshot, GraphSnapshot, ClaimData


# =============================================================================
# NARRATIVE STRUCTURE
# =============================================================================

@dataclass
class EpistemicNarrative:
    """A narrative at any abstraction level - the "Event" as organism"""
    id: str
    level: int

    # Core identity
    title: str
    summary: str

    # Epistemic state (the "qualia")
    what_we_know: List[str] = field(default_factory=list)
    what_is_contested: List[Dict] = field(default_factory=list)  # {claim, alternatives, type}
    what_we_dont_know: List[str] = field(default_factory=list)
    what_would_help: List[str] = field(default_factory=list)
    key_question: str = ""

    # Support structure
    supported_by: List[str] = field(default_factory=list)  # narrative/claim IDs
    entities: List[str] = field(default_factory=list)

    # Metrics
    entropy: float = 1.0
    coherence: float = 0.0
    n_ground_claims: int = 0


# =============================================================================
# NARRATIVE ENGINE
# =============================================================================

class NarrativeEmergenceEngine:
    """Engine that emerges narratives from claims"""

    def __init__(self, snapshot: GraphSnapshot):
        self.snapshot = snapshot
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self._embedding_cache = {}

        self.narratives_by_level: Dict[int, List[EpistemicNarrative]] = defaultdict(list)
        self.all_narratives: Dict[str, EpistemicNarrative] = {}

    def get_embedding(self, text: str) -> List[float]:
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
    # CLUSTERING
    # =========================================================================

    def cluster_claims(self, claims: List, threshold: float = 0.70, use_entity_gravity: bool = True) -> List[List]:
        """Cluster claims/narratives by similarity WITH entity gravity"""
        if len(claims) < 2:
            return []

        # Get embeddings
        embeddings = {}
        for c in claims:
            text = c.text if hasattr(c, 'text') else c.summary
            embeddings[c.id] = self.get_embedding(text)

        # Build entity index for gravity-based clustering
        entity_to_claims = defaultdict(set)
        claim_entities = {}
        for c in claims:
            entities = set(c.entity_ids if hasattr(c, 'entity_ids') else getattr(c, 'entities', []))
            claim_entities[c.id] = entities
            for e in entities:
                entity_to_claims[e].add(c.id)

        # Compute entity specificity (IDF-like): rare entities = high weight
        # Common entities like "Hong Kong", "China" get LOW weight
        total_claims = len(claims)
        entity_specificity = {}
        for entity, claim_set in entity_to_claims.items():
            # IDF: log(total / docs_containing_term)
            # More claims contain entity → lower specificity
            freq = len(claim_set) / total_claims
            if freq > 0.3:  # Entity appears in >30% of claims = too common
                entity_specificity[entity] = 0.1  # Almost no gravitational pull
            elif freq > 0.1:  # 10-30% = somewhat common
                entity_specificity[entity] = 0.3
            else:  # <10% = specific, high pull
                entity_specificity[entity] = 1.0

        # Entity gravity: claims sharing SPECIFIC entities are gravitationally bound
        def compute_gravity(c1_id: str, c2_id: str) -> float:
            """Compute gravitational pull between claims via shared entities"""
            e1, e2 = claim_entities.get(c1_id, set()), claim_entities.get(c2_id, set())
            shared = e1 & e2
            if not shared:
                return 0.0

            # Weight by specificity - specific entities pull harder
            weighted_shared = sum(entity_specificity.get(e, 0.5) for e in shared)
            total = len(e1 | e2)

            # Only high gravity if we share SPECIFIC entities
            return weighted_shared / (total ** 0.5) if total > 0 else 0.0

        # Combined similarity: semantic + entity gravity
        def combined_similarity(c1_id: str, c2_id: str) -> float:
            semantic = self.cosine_sim(embeddings[c1_id], embeddings[c2_id])
            gravity = compute_gravity(c1_id, c2_id) if use_entity_gravity else 0.0

            # Entity gravity BOOSTS semantic similarity significantly
            # If they share entities, even moderate semantic similarity should cluster
            if gravity > 0.3:  # Strong entity overlap
                return max(semantic, 0.5) + gravity * 0.4  # Boost to cluster
            elif gravity > 0.1:  # Some entity overlap
                return semantic + gravity * 0.3
            return semantic

        # Greedy clustering with combined similarity
        clusters = []
        assigned = set()

        # Sort claims by entity connectivity (most connected first = gravitational centers)
        claims_sorted = sorted(claims, key=lambda c: len(claim_entities.get(c.id, set())), reverse=True)

        for claim in claims_sorted:
            if claim.id in assigned:
                continue

            cluster = [claim]
            assigned.add(claim.id)

            for other in claims:
                if other.id in assigned:
                    continue

                sim = combined_similarity(claim.id, other.id)

                if sim >= threshold or sim >= 0.45:  # Lower threshold when gravity helps
                    cluster.append(other)
                    assigned.add(other.id)

            if len(cluster) >= 2:
                clusters.append(cluster)

        return clusters

    # =========================================================================
    # NARRATIVE SYNTHESIS
    # =========================================================================

    def synthesize_narrative(self, cluster: List, target_level: int) -> Optional[EpistemicNarrative]:
        """Synthesize a narrative from a cluster of claims/narratives"""

        # Gather claim texts
        if target_level == 1:
            # L1: synthesizing from ground claims
            claim_texts = [c.text for c in cluster[:15]]
            source_type = "ground claims"
        else:
            # L2+: synthesizing from narratives
            claim_texts = [f"{n.title}: {n.summary}" for n in cluster[:10]]
            source_type = f"L{target_level-1} narratives"

        claims_str = "\n".join([f"- {t[:200]}" for t in claim_texts])

        # Check for contradictions within cluster
        contradictions = self._find_contradictions(cluster)
        contra_str = "\n".join([f"- {c}" for c in contradictions[:5]]) if contradictions else "(none found)"

        prompt = f"""Analyze this cluster of {source_type} and synthesize an epistemic narrative.

SOURCE CLAIMS/NARRATIVES:
{claims_str}

CONTRADICTIONS DETECTED:
{contra_str}

Create a structured narrative that captures:
1. TITLE: Short name for this pattern/event/interpretation
2. SUMMARY: One paragraph synthesis
3. WHAT WE KNOW: List of well-supported facts (cite number of sources)
4. WHAT IS CONTESTED: List of conflicts with alternatives
5. WHAT WE DON'T KNOW: Gaps in knowledge
6. WHAT WOULD HELP: Evidence that would resolve uncertainty
7. KEY QUESTION: The central unresolved question

Level guide:
- L1 (Pattern): "Multiple sources report X" - aggregate observations
- L2 (Interpretation): "The evidence suggests X" - draw conclusions
- L3 (Implication): "This indicates X" - broader meaning
- L4+ (Systemic): "This reflects X" - structural/paradigmatic

Target level: L{target_level}

Respond in JSON:
{{
  "title": "short title",
  "summary": "one paragraph synthesis",
  "what_we_know": ["fact 1 (N sources)", "fact 2 (N sources)"],
  "what_is_contested": [
    {{"claim": "one version", "alternatives": ["other versions"], "type": "NUMBER|CAUSE|ATTRIBUTION"}}
  ],
  "what_we_dont_know": ["gap 1", "gap 2"],
  "what_would_help": ["evidence type 1", "evidence type 2"],
  "key_question": "the central question",
  "confidence": 0.0-1.0,
  "entities": ["key entities"]
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            result = json.loads(response.choices[0].message.content)

            # Create narrative
            narrative_id = f"L{target_level}_{hashlib.md5(result['title'].encode()).hexdigest()[:8]}"

            # Compute entropy
            n_sources = len(cluster)
            n_contested = len(result.get('what_is_contested', []))
            confidence = result.get('confidence', 0.5)

            base_entropy = 0.8 - (confidence * 0.3)
            support_reduction = 0.15 * np.log1p(n_sources)
            contest_addition = 0.1 * n_contested

            entropy = max(0.1, min(0.95, base_entropy - support_reduction + contest_addition))
            coherence = 1.0 - entropy

            # Count ground claims
            if target_level == 1:
                n_ground = len(cluster)
            else:
                n_ground = sum(n.n_ground_claims for n in cluster if hasattr(n, 'n_ground_claims'))

            narrative = EpistemicNarrative(
                id=narrative_id,
                level=target_level,
                title=result['title'],
                summary=result['summary'],
                what_we_know=result.get('what_we_know', []),
                what_is_contested=result.get('what_is_contested', []),
                what_we_dont_know=result.get('what_we_dont_know', []),
                what_would_help=result.get('what_would_help', []),
                key_question=result.get('key_question', ''),
                supported_by=[c.id for c in cluster],
                entities=result.get('entities', []),
                entropy=entropy,
                coherence=coherence,
                n_ground_claims=n_ground,
            )

            return narrative

        except Exception as e:
            print(f"    Error synthesizing narrative: {e}")
            return None

    def _find_contradictions(self, cluster: List) -> List[str]:
        """Find contradictions within a cluster"""
        contradictions = []

        texts = [c.text if hasattr(c, 'text') else c.summary for c in cluster]

        # Look for number disagreements
        import re
        for i, t1 in enumerate(texts):
            nums1 = set(re.findall(r'\b\d+\b', t1))
            for t2 in texts[i+1:]:
                nums2 = set(re.findall(r'\b\d+\b', t2))
                # Different specific numbers for same thing
                if nums1 and nums2 and nums1 != nums2:
                    shared_context = set(t1.lower().split()) & set(t2.lower().split())
                    if len(shared_context) > 5:  # Same topic
                        contradictions.append(f"Numbers differ: {nums1} vs {nums2}")
                        break

        return contradictions[:5]

    # =========================================================================
    # NARRATIVE MERGING (Entity Gravity for Mass Accumulation)
    # =========================================================================

    def _merge_related_narratives(self, narratives: List[EpistemicNarrative]) -> List[EpistemicNarrative]:
        """Merge narratives that are about the same event (share SPECIFIC entities)"""

        if len(narratives) < 2:
            return narratives

        # Build entity overlap matrix with specificity weighting
        narrative_entities = {}
        all_entities_count = defaultdict(int)

        for n in narratives:
            # Collect entities from the narrative and its supported claims
            entities = set(n.entities)
            for claim_id in n.supported_by:
                if claim_id in self.snapshot.claims:
                    claim = self.snapshot.claims[claim_id]
                    entities.update(claim.entity_ids)
            narrative_entities[n.id] = entities
            for e in entities:
                all_entities_count[e] += 1

        # Compute entity specificity across narratives
        total_narratives = len(narratives)
        entity_specificity = {}
        for entity, count in all_entities_count.items():
            freq = count / total_narratives
            if freq > 0.4:  # Appears in >40% of narratives = too common (Hong Kong, China)
                entity_specificity[entity] = 0.0  # NO gravitational pull for merge
            elif freq > 0.2:
                entity_specificity[entity] = 0.2
            else:
                entity_specificity[entity] = 1.0

        # Find narratives that should merge (high SPECIFIC entity overlap)
        merge_groups = []
        merged_ids = set()

        for n1 in narratives:
            if n1.id in merged_ids:
                continue

            group = [n1]
            merged_ids.add(n1.id)

            for n2 in narratives:
                if n2.id in merged_ids:
                    continue

                e1, e2 = narrative_entities[n1.id], narrative_entities[n2.id]
                shared = e1 & e2

                # Weight shared entities by specificity
                weighted_overlap = sum(entity_specificity.get(e, 0.5) for e in shared)
                specific_shared = [e for e in shared if entity_specificity.get(e, 0) > 0.5]

                # Only merge if we share SPECIFIC entities (not just "Hong Kong")
                if weighted_overlap >= 2.0 or len(specific_shared) >= 2:
                    group.append(n2)
                    merged_ids.add(n2.id)
                    print(f"    Merging: '{n2.title[:40]}...' → '{n1.title[:40]}...'")
                    print(f"      (specific shared: {specific_shared[:3]})")

            merge_groups.append(group)

        # Create merged narratives
        merged = []
        for group in merge_groups:
            if len(group) == 1:
                merged.append(group[0])
            else:
                # Synthesize a merged narrative
                merged_narrative = self._synthesize_merged_narrative(group)
                if merged_narrative:
                    merged.append(merged_narrative)
                else:
                    # Fallback: keep the largest
                    merged.append(max(group, key=lambda n: n.n_ground_claims))

        return merged

    def _synthesize_merged_narrative(self, narratives: List[EpistemicNarrative]) -> Optional[EpistemicNarrative]:
        """Create a unified narrative from multiple aspect-narratives about the same event"""

        # Gather all aspects
        all_know = []
        all_contested = []
        all_dont_know = []
        all_would_help = []
        all_questions = []
        total_claims = 0
        all_entities = set()
        all_supported_by = []

        for n in narratives:
            all_know.extend(n.what_we_know)
            all_contested.extend(n.what_is_contested)
            all_dont_know.extend(n.what_we_dont_know)
            all_would_help.extend(n.what_would_help)
            if n.key_question:
                all_questions.append(n.key_question)
            total_claims += n.n_ground_claims
            all_entities.update(n.entities)
            all_supported_by.extend(n.supported_by)

        # Create unified summary via LLM
        aspects_str = "\n".join([f"- {n.title}: {n.summary[:200]}" for n in narratives[:5]])

        prompt = f"""Synthesize these related narratives about the SAME EVENT into one unified narrative.

ASPECT NARRATIVES:
{aspects_str}

COMBINED FACTS KNOWN:
{chr(10).join(['- ' + k[:100] for k in all_know[:10]])}

CONTESTED POINTS:
{chr(10).join(['- ' + str(c)[:100] for c in all_contested[:5]])}

Create a UNIFIED narrative that:
1. Has a clear title for the whole event
2. Synthesizes all aspects into one coherent summary
3. Preserves the most important facts, contests, and gaps

Respond in JSON:
{{
  "title": "unified event title",
  "summary": "comprehensive summary covering all aspects",
  "key_question": "the central unresolved question"
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            result = json.loads(response.choices[0].message.content)

            # Compute unified entropy
            # More claims = lower entropy, contested = higher entropy
            base_entropy = 0.5
            claim_reduction = 0.1 * np.log1p(total_claims)
            contest_addition = 0.05 * len(all_contested)
            entropy = max(0.1, min(0.9, base_entropy - claim_reduction + contest_addition))

            merged = EpistemicNarrative(
                id=f"L1_merged_{hashlib.md5(result['title'].encode()).hexdigest()[:8]}",
                level=1,
                title=result['title'],
                summary=result['summary'],
                what_we_know=list(set(all_know))[:10],  # Dedupe
                what_is_contested=all_contested[:5],
                what_we_dont_know=list(set(all_dont_know))[:5],
                what_would_help=list(set(all_would_help))[:5],
                key_question=result.get('key_question', all_questions[0] if all_questions else ''),
                supported_by=list(set(all_supported_by)),
                entities=list(all_entities),
                entropy=entropy,
                coherence=1.0 - entropy,
                n_ground_claims=total_claims,
            )

            print(f"    → Created merged event: '{merged.title}' ({total_claims} claims, {1-entropy:.0%} coherence)")
            return merged

        except Exception as e:
            print(f"    Error merging narratives: {e}")
            return None

    # =========================================================================
    # EMERGENCE LOOP
    # =========================================================================

    def run_emergence(self, max_levels: int = 4) -> Dict:
        """Run full emergence from ground to highest abstraction"""

        results = []

        # Level 0: Load ground claims
        print("\n" + "=" * 70)
        print("LEVEL 0: GROUND CLAIMS")
        print("=" * 70)

        ground_claims = list(self.snapshot.claims.values())
        print(f"  Loaded {len(ground_claims)} ground claims")

        # Level 1: Emerge from ground claims
        print("\n" + "=" * 70)
        print("LEVEL 1: PATTERN NARRATIVES (Events)")
        print("=" * 70)

        clusters_l1 = self.cluster_claims(ground_claims, threshold=0.65)
        print(f"  Found {len(clusters_l1)} clusters")

        for i, cluster in enumerate(clusters_l1[:15]):  # Limit
            print(f"\n  Cluster {i+1} ({len(cluster)} claims)...")
            narrative = self.synthesize_narrative(cluster, target_level=1)

            if narrative:
                self.narratives_by_level[1].append(narrative)
                self.all_narratives[narrative.id] = narrative
                print(f"    → {narrative.title}")

        print(f"\n  Emerged {len(self.narratives_by_level[1])} L1 narratives")

        # MERGE PHASE: Combine L1 narratives that share ground claims (same event, different aspects)
        print("\n" + "-" * 70)
        print("MERGE PHASE: Combining narratives about the same event")
        print("-" * 70)

        merged_narratives = self._merge_related_narratives(self.narratives_by_level[1])
        self.narratives_by_level[1] = merged_narratives
        print(f"  After merge: {len(merged_narratives)} L1 narratives")

        # Level 2+: Emerge from narratives
        for level in range(2, max_levels + 1):
            prev_level = level - 1
            prev_narratives = self.narratives_by_level[prev_level]

            if len(prev_narratives) < 2:
                print(f"\n  Not enough L{prev_level} narratives to cluster")
                break

            print(f"\n" + "=" * 70)
            print(f"LEVEL {level}: {'INTERPRETATION' if level == 2 else 'IMPLICATION' if level == 3 else 'SYSTEMIC'} NARRATIVES")
            print("=" * 70)

            clusters = self.cluster_claims(prev_narratives, threshold=0.60)
            print(f"  Found {len(clusters)} clusters from {len(prev_narratives)} L{prev_level} narratives")

            for cluster in clusters[:10]:
                narrative = self.synthesize_narrative(cluster, target_level=level)

                if narrative:
                    self.narratives_by_level[level].append(narrative)
                    self.all_narratives[narrative.id] = narrative
                    print(f"\n    → {narrative.title}")
                    print(f"      Entropy: {narrative.entropy:.2f}, Ground claims: {narrative.n_ground_claims}")

            if not self.narratives_by_level[level]:
                break

        return {
            'levels': max(self.narratives_by_level.keys()) if self.narratives_by_level else 0,
            'narratives_by_level': {k: len(v) for k, v in self.narratives_by_level.items()},
        }

    # =========================================================================
    # DISPLAY
    # =========================================================================

    def display_narratives(self):
        """Display all narratives as epistemic organisms"""

        print("\n" + "=" * 70)
        print("EMERGED EPISTEMIC NARRATIVES")
        print("=" * 70)

        max_level = max(self.narratives_by_level.keys()) if self.narratives_by_level else 0

        for level in range(max_level, 0, -1):
            narratives = self.narratives_by_level[level]

            level_names = {1: "EVENTS (Patterns)", 2: "INTERPRETATIONS", 3: "IMPLICATIONS", 4: "SYSTEMIC"}
            level_name = level_names.get(level, f"L{level}")

            print(f"\n{'─' * 70}")
            print(f"LEVEL {level}: {level_name}")
            print(f"{'─' * 70}")

            for narrative in narratives[:5]:
                self._display_narrative(narrative)

    def _display_narrative(self, n: EpistemicNarrative):
        """Display a single narrative as an epistemic organism"""

        coherence_bar = "█" * int(n.coherence * 10) + "░" * int((1 - n.coherence) * 10)

        print(f"""
┌{'─' * 68}┐
│ {n.title[:66]:<66} │
├{'─' * 68}┤
│ Coherence: [{coherence_bar}] {n.coherence:.0%}                          │
│ Ground claims: {n.n_ground_claims:<5} Entropy: {n.entropy:.2f}                          │
├{'─' * 68}┤
│ SUMMARY:                                                           │
│ {n.summary[:66]:<66} │""")

        if len(n.summary) > 66:
            print(f"│ {n.summary[66:132]:<66} │")

        print(f"├{'─' * 68}┤")
        print(f"│ WHAT WE KNOW:                                                      │")
        for fact in n.what_we_know[:3]:
            print(f"│   • {fact[:62]:<62} │")

        if n.what_is_contested:
            print(f"├{'─' * 68}┤")
            print(f"│ WHAT IS CONTESTED:                                                 │")
            for c in n.what_is_contested[:2]:
                claim = c.get('claim', str(c))[:58]
                print(f"│   ⚡ {claim:<62} │")

        if n.what_we_dont_know:
            print(f"├{'─' * 68}┤")
            print(f"│ WHAT WE DON'T KNOW:                                                │")
            for gap in n.what_we_dont_know[:2]:
                print(f"│   ? {gap[:62]:<62} │")

        if n.key_question:
            print(f"├{'─' * 68}┤")
            print(f"│ KEY QUESTION:                                                      │")
            print(f"│   → {n.key_question[:62]:<62} │")

        print(f"└{'─' * 68}┘")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("NARRATIVE EMERGENCE")
    print("Events as Epistemic Organisms")
    print("=" * 70)

    snapshot = load_snapshot()
    print(f"\nLoaded {len(snapshot.claims)} ground claims")

    engine = NarrativeEmergenceEngine(snapshot)
    results = engine.run_emergence(max_levels=3)

    engine.display_narratives()

    # Summary
    print("\n" + "=" * 70)
    print("EMERGENCE SUMMARY")
    print("=" * 70)

    print(f"""
Levels emerged: {results['levels']}

Narratives by level:
{chr(10).join(f"  L{k}: {v} narratives" for k, v in sorted(results['narratives_by_level'].items()))}

Each narrative knows:
  ✓ What it knows (supported facts)
  ✓ What is contested (conflicts to resolve)
  ✓ What it doesn't know (gaps)
  ✓ What would help (needed evidence)
  ✓ Its key question (central uncertainty)

This is the Event as Epistemic Organism -
a self-aware knowledge structure that can
articulate its own uncertainty and needs.
""")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'ground_claims': len(snapshot.claims),
        'results': results,
        'narratives': [
            {
                'id': n.id,
                'level': n.level,
                'title': n.title,
                'summary': n.summary,
                'what_we_know': n.what_we_know,
                'what_is_contested': n.what_is_contested,
                'what_we_dont_know': n.what_we_dont_know,
                'key_question': n.key_question,
                'entropy': n.entropy,
                'coherence': n.coherence,
                'n_ground_claims': n.n_ground_claims,
            }
            for level_narratives in engine.narratives_by_level.values()
            for n in level_narratives
        ]
    }

    output_path = Path("/app/test_eu/results/narrative_emergence_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
