"""
Unified Epistemic Kernel
=========================

Combines the best from experiments:
- belief_kernel.py: Flat belief list, simple pairwise comparison
- relate_updates.py: Structured extraction for numeric/temporal hints
- universal_kernel.py: q1/q2 prompting pattern
- kernel_complete.py: Jaynes entropy calculation
- breathing_event.py: Embedding-based clustering for event emergence

Design: Domain knowledge in prompt, not code.
One LLM call per claim. No topics, no aspects.
Embedding similarity connects beliefs about the same event.
"""

import asyncio
import json
import math
import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum


# Embedding similarity thresholds (from breathing_event.py)
# Note: text-embedding-3-small gives lower scores than expected for paraphrases
# "Fire kills 11" vs "Blaze leaves 11 dead" = 0.66 (should be higher)
SIM_THRESHOLD = 0.65       # High similarity = same event (skip LLM)
WEAK_SIM_THRESHOLD = 0.50  # Weak similarity = potentially same event (LLM with hint)
SURFACE_MERGE_THRESHOLD = 0.55  # Threshold for merging surfaces into parent events


def cosine_sim(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    a, b = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class Relation(Enum):
    """Possible relations between claims."""
    NOVEL = "novel"           # New fact, no existing belief covers it
    CONFIRMS = "confirms"     # Same fact from another source
    REFINES = "refines"       # More specific version
    SUPERSEDES = "supersedes" # Updated value (temporal)
    CONFLICTS = "conflicts"   # Cannot both be true


@dataclass
class Belief:
    """A belief with provenance and evolution history."""
    id: str
    text: str                              # Current belief text
    sources: Set[str] = field(default_factory=set)
    claim_ids: Set[str] = field(default_factory=set)  # Claims supporting this
    superseded: Optional[str] = None       # Previous value if updated
    evolution: List[Tuple[str, str]] = field(default_factory=list)  # (value, source)
    embedding: Optional[List[float]] = None  # For semantic clustering
    question_type: Optional[str] = None      # What question this answers

    @property
    def canonical(self) -> str:
        """Alias for text (backwards compatibility)."""
        return self.text

    def add_source(self, source: str, claim_id: str):
        """Add corroborating source."""
        self.sources.add(source)
        self.claim_ids.add(claim_id)

    def update(self, new_text: str, source: str, claim_id: str):
        """Update belief with new value."""
        self.evolution.append((self.text, list(self.sources)[-1] if self.sources else 'unknown'))
        self.superseded = self.text
        self.text = new_text
        self.sources.add(source)
        self.claim_ids.add(claim_id)

    def entropy(self) -> float:
        """
        Jaynes-aligned uncertainty.
        Single source = high entropy (0.8)
        Multiple sources = lower entropy
        """
        n = len(self.sources)
        if n == 0:
            return 1.0
        if n == 1:
            return 0.80
        if n == 2:
            return 0.50
        if n == 3:
            return 0.35
        return max(0.15, 0.5 - n * 0.05)


@dataclass
class Conflict:
    """Unresolved conflict between claims."""
    belief_id: str
    new_claim: str
    new_source: str
    existing_text: str
    reasoning: str


class EpistemicKernel:
    """
    Minimal epistemic engine.

    State: list of beliefs + list of conflicts
    Operation: process(claim, source) → updates state

    Uses embedding similarity to connect beliefs about the same event.
    """

    def __init__(self, llm_client=None, use_embeddings: bool = True):
        self.llm = llm_client
        self.beliefs: List[Belief] = []
        self.conflicts: List[Conflict] = []
        self.history: List[Dict] = []
        self._claim_counter = 0
        self.use_embeddings = use_embeddings
        # Track efficiency
        self.llm_calls = 0
        self.llm_calls_skipped = 0

    def _next_claim_id(self) -> str:
        self._claim_counter += 1
        return f"c{self._claim_counter:03d}"

    def _next_belief_id(self) -> str:
        return f"b{len(self.beliefs):03d}"

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding vector for text using OpenAI API."""
        if not self.use_embeddings or not self.llm:
            return None
        try:
            response = await self.llm.embeddings.create(
                model="text-embedding-3-small",
                input=text[:8000]  # Truncate to fit context
            )
            return response.data[0].embedding
        except Exception as e:
            # Silently fail - embeddings are enhancement, not critical
            return None

    # =========================================================================
    # STRUCTURED EXTRACTION (from relate_updates.py)
    # =========================================================================

    def _extract_hints(self, text: str) -> Dict:
        """
        Extract structural hints to help LLM comparison.
        From relate_updates.py: numbers, temporal markers.
        """
        text_lower = text.lower()
        hints = {
            'numbers': [],
            'has_update_language': False,
            'numeric_value': None,
        }

        # Extract numbers
        numbers = re.findall(r'\b(\d+)\b', text)
        hints['numbers'] = [int(n) for n in numbers if int(n) > 0]
        if hints['numbers']:
            hints['numeric_value'] = max(hints['numbers'])

        # Update language (from relate_updates.py)
        update_phrases = [
            'rises to', 'risen to', 'increased to', 'now',
            'updated', 'latest', 'climbed to', 'reached',
            'has grown', 'hits', 'surpasses', 'as of'
        ]
        hints['has_update_language'] = any(p in text_lower for p in update_phrases)

        return hints

    # =========================================================================
    # BELIEF FORMATTING (for LLM context)
    # =========================================================================

    def _format_beliefs_for_prompt(self) -> str:
        """Format current beliefs for LLM context."""
        if not self.beliefs:
            return "(no beliefs yet)"

        lines = []
        for i, b in enumerate(self.beliefs):
            src_count = len(b.sources)
            confidence = "CONFIRMED" if src_count >= 3 else "LIKELY" if src_count >= 2 else "REPORTED"
            lines.append(f"[{i}] ({confidence}, {src_count} sources) {b.text}")

        return "\n".join(lines)

    # =========================================================================
    # CORE OPERATION: PROCESS CLAIM
    # =========================================================================

    async def process(self, claim_text: str, source: str) -> Dict:
        """
        Process one claim.

        Uses embedding similarity as pre-filter to reduce LLM calls:
        - High similarity (≥0.70): Skip LLM, directly CONFIRMS
        - Medium similarity (≥0.55): Call LLM with similarity hint
        - Low similarity (<0.55): Full LLM classification

        Returns: {relation, affected_belief, reasoning}
        """
        claim_id = self._next_claim_id()
        hints = self._extract_hints(claim_text)

        # Get embedding for incoming claim FIRST (for pre-filter)
        claim_embedding = await self._get_embedding(claim_text)

        # If no beliefs yet, this is the first
        if not self.beliefs:
            belief = Belief(
                id=self._next_belief_id(),
                text=claim_text,
                sources={source},
                claim_ids={claim_id},
                embedding=claim_embedding
            )
            self.beliefs.append(belief)
            result = {
                'relation': Relation.NOVEL.value,
                'affected_belief': None,
                'reasoning': 'First claim establishes initial belief'
            }
            self.history.append({
                'claim': claim_text,
                'source': source,
                'claim_id': claim_id,
                'result': result
            })
            return result

        # =========================================================================
        # EMBEDDING PRE-FILTER (from breathing_event.py)
        # Find most similar existing belief before calling LLM
        # =========================================================================
        best_sim = 0.0
        best_idx = None

        if claim_embedding:
            for i, belief in enumerate(self.beliefs):
                if belief.embedding:
                    sim = cosine_sim(claim_embedding, belief.embedding)
                    if sim > best_sim:
                        best_sim = sim
                        best_idx = i

        # Decision based on similarity
        if best_sim >= SIM_THRESHOLD:
            # HIGH SIMILARITY: Skip LLM, directly CONFIRMS or SUPERSEDES
            # Check if this is an update (has update language or higher numbers)
            if hints['has_update_language']:
                # Likely an update - still need LLM to classify SUPERSEDES vs CONFIRMS
                self.llm_calls += 1
                result = await self._classify_with_llm(claim_text, source, hints,
                                                       similarity_hint=best_sim,
                                                       similar_belief_idx=best_idx)
            else:
                # Same fact, different source - skip LLM
                self.llm_calls_skipped += 1
                result = {
                    'relation': Relation.CONFIRMS.value,
                    'affected_belief': best_idx,
                    'reasoning': f'High embedding similarity ({best_sim:.2f}) indicates same fact',
                    'normalized_claim': claim_text,
                    'skipped_llm': True,
                    'similarity': best_sim
                }
        elif best_sim >= WEAK_SIM_THRESHOLD:
            # MEDIUM SIMILARITY: Call LLM but with similarity hint
            self.llm_calls += 1
            result = await self._classify_with_llm(claim_text, source, hints,
                                                   similarity_hint=best_sim,
                                                   similar_belief_idx=best_idx)
        else:
            # LOW SIMILARITY: Full LLM classification
            self.llm_calls += 1
            result = await self._classify_with_llm(claim_text, source, hints)

        # Update state based on relation
        relation = Relation(result['relation'])
        affected_idx = result.get('affected_belief')

        if relation == Relation.NOVEL:
            # New fact - add belief with embedding (already computed in pre-filter)
            belief = Belief(
                id=self._next_belief_id(),
                text=claim_text,
                sources={source},
                claim_ids={claim_id},
                embedding=claim_embedding,  # Use pre-computed embedding
                question_type=result.get('question_answered')
            )
            self.beliefs.append(belief)

        elif relation == Relation.CONFIRMS:
            # Same fact - add source to existing belief
            if affected_idx is not None and 0 <= affected_idx < len(self.beliefs):
                self.beliefs[affected_idx].add_source(source, claim_id)

        elif relation == Relation.REFINES:
            # More specific - update belief text
            if affected_idx is not None and 0 <= affected_idx < len(self.beliefs):
                normalized = result.get('normalized_claim', claim_text)
                self.beliefs[affected_idx].update(normalized, source, claim_id)

        elif relation == Relation.SUPERSEDES:
            # Temporal update - replace value
            if affected_idx is not None and 0 <= affected_idx < len(self.beliefs):
                normalized = result.get('normalized_claim', claim_text)
                self.beliefs[affected_idx].update(normalized, source, claim_id)

        elif relation == Relation.CONFLICTS:
            # Cannot resolve - flag for review
            if affected_idx is not None and 0 <= affected_idx < len(self.beliefs):
                self.conflicts.append(Conflict(
                    belief_id=self.beliefs[affected_idx].id,
                    new_claim=claim_text,
                    new_source=source,
                    existing_text=self.beliefs[affected_idx].text,
                    reasoning=result.get('reasoning', '')
                ))

        self.history.append({
            'claim': claim_text,
            'source': source,
            'claim_id': claim_id,
            'result': result
        })

        return result

    async def _classify_with_llm(self, claim: str, source: str, hints: Dict,
                                   similarity_hint: float = None,
                                   similar_belief_idx: int = None) -> Dict:
        """
        Classify relationship using LLM.

        Key: q1/q2 pattern from universal_kernel.py
        Claims only relate if they answer the SAME question.

        If similarity_hint is provided, the LLM is informed that embedding
        similarity suggests a connection to a specific belief.
        """

        # Build similarity context if provided
        similarity_context = ""
        if similarity_hint is not None and similar_belief_idx is not None:
            similar_belief = self.beliefs[similar_belief_idx]
            similarity_context = f"""
EMBEDDING SIMILARITY HINT:
This claim has {similarity_hint:.0%} semantic similarity to belief [{similar_belief_idx}].
High similarity (>70%) suggests they answer the same question.
If the values differ and there's update language, this is likely SUPERSEDES.
If the values are the same, this is likely CONFIRMS.
"""

        # Build prompt with q1/q2 insight
        prompt = f"""Compare a new claim to existing beliefs about a news event.

EXISTING BELIEFS:
{self._format_beliefs_for_prompt()}

NEW CLAIM: "{claim}"
SOURCE: {source}
EXTRACTED: numbers={hints['numbers']}, has_update_language={hints['has_update_language']}
{similarity_context}

STEP 1 - IDENTIFY THE QUESTION:
What specific question does this claim answer?
Examples: "count of deaths", "location of fire", "cause of incident", "response actions"

STEP 2 - FIND MATCHING BELIEF:
Does any existing belief answer the SAME question?
If not → NOVEL (different questions = no relationship)

STEP 3 - IF SAME QUESTION, DETERMINE RELATIONSHIP:
- CONFIRMS: Same question, same value (corroboration)
- REFINES: Same question, more specific (adds detail without changing value)
- SUPERSEDES: Same question, different value WITH update language ("rises to", "now", "latest")
- CONFLICTS: Same question, different value, NO temporal ordering

KEY INSIGHT: "36 dead" vs "128 dead" with "rises to" → SUPERSEDES (death toll updated)
KEY INSIGHT: "36 dead" vs "fire on 14th floor" → NOVEL (death count ≠ location)

Return JSON:
{{
  "question_answered": "what question this claim answers",
  "matching_belief_question": "what question the matching belief answers (or null)",
  "relation": "NOVEL|CONFIRMS|REFINES|SUPERSEDES|CONFLICTS",
  "affected_belief": <index number or null>,
  "reasoning": "one sentence explanation",
  "normalized_claim": "the claim in clear standalone form"
}}"""

        try:
            response = await self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            result = json.loads(response.choices[0].message.content)

            # Normalize relation to lowercase
            rel = result.get('relation', 'NOVEL').upper()
            if rel not in ['NOVEL', 'CONFIRMS', 'REFINES', 'SUPERSEDES', 'CONFLICTS']:
                rel = 'NOVEL'
            result['relation'] = rel.lower()

            return result

        except Exception as e:
            return {
                'relation': 'novel',
                'affected_belief': None,
                'reasoning': f'Error: {e}',
                'normalized_claim': claim
            }

    # =========================================================================
    # METRICS (from kernel_complete.py)
    # =========================================================================

    def entropy(self) -> float:
        """
        Overall epistemic entropy.
        Jaynes-aligned: more corroboration = less uncertainty.
        """
        if not self.beliefs:
            return 1.0

        entropies = [b.entropy() for b in self.beliefs]
        return sum(entropies) / len(entropies)

    def coherence(self) -> float:
        """
        How consistent is the belief state?
        High coherence = few conflicts, lots of corroboration.
        """
        if not self.history:
            return 1.0

        total = len(self.history)
        conflicts = len(self.conflicts)

        # Agreements: CONFIRMS, REFINES, SUPERSEDES
        agreements = sum(1 for h in self.history
                        if h['result'].get('relation') in ('confirms', 'refines', 'supersedes'))

        raw = (agreements - conflicts * 2) / max(total, 1)
        return max(0.0, min(1.0, 0.5 + raw * 0.5))

    # =========================================================================
    # OUTPUT
    # =========================================================================

    def topology(self) -> Dict:
        """
        Return the belief topology as a graph structure.

        Returns:
            nodes: list of beliefs with their properties
            edges: list of (source_belief, target_belief, relation)
            surfaces: connected components (natural event clusters)

        Topology formation:
        - Beliefs are nodes
        - Edges form from: CONFLICTS (explicit), source overlap (implicit same-event)
        - Surfaces emerge as connected components (natural event boundaries)
        """
        from collections import defaultdict

        # Build edges from:
        # 1. CONFLICTS between beliefs
        # 2. Source overlap (same sources = likely same event)
        # 3. Evolution chains (supersession/refinement)
        edges = []
        edge_set = set()  # Avoid duplicates

        # Evolution edges (from history: SUPERSEDES, REFINES)
        for h in self.history:
            rel = h['result'].get('relation', 'novel')
            affected = h['result'].get('affected_belief')

            if rel in ('supersedes', 'refines') and affected is not None:
                # This claim updated belief[affected]
                # The evolution is recorded in the belief
                if affected < len(self.beliefs):
                    b = self.beliefs[affected]
                    if b.superseded:
                        # Find belief that had the old value
                        for j, other in enumerate(self.beliefs):
                            if j != affected and (other.text == b.superseded or
                                                  b.superseded in other.text):
                                key = (min(affected, j), max(affected, j), 'evolves')
                                if key not in edge_set:
                                    edge_set.add(key)
                                    edges.append({
                                        'source': j,
                                        'target': affected,
                                        'relation': 'evolves',
                                        'weight': 0.9
                                    })

        # Conflict edges
        for c in self.conflicts:
            # Find which belief this conflicts with
            for i, b in enumerate(self.beliefs):
                if b.id == c.belief_id:
                    # The new conflicting claim became which belief?
                    for j, other in enumerate(self.beliefs):
                        if c.new_claim in other.text or any(c.new_claim[:30] in cid for cid in other.claim_ids):
                            if i != j:
                                key = (min(i, j), max(i, j), 'conflicts')
                                if key not in edge_set:
                                    edge_set.add(key)
                                    edges.append({
                                        'source': i,
                                        'target': j,
                                        'relation': 'conflicts',
                                        'weight': 1.0
                                    })

        # Source overlap edges (same sources reporting = likely same event)
        for i, b1 in enumerate(self.beliefs):
            for j, b2 in enumerate(self.beliefs):
                if i < j:  # Avoid duplicates
                    shared = b1.sources & b2.sources
                    if shared:
                        key = (i, j, 'same_event')
                        if key not in edge_set:
                            edge_set.add(key)
                            edges.append({
                                'source': i,
                                'target': j,
                                'relation': 'same_event',
                                'weight': len(shared) / max(len(b1.sources), len(b2.sources)),
                                'shared_sources': list(shared)
                            })

        # Embedding similarity edges (semantic clustering - from breathing_event.py)
        # High similarity = same event, even if sources don't overlap
        for i, b1 in enumerate(self.beliefs):
            for j, b2 in enumerate(self.beliefs):
                if i < j and b1.embedding and b2.embedding:
                    sim = cosine_sim(b1.embedding, b2.embedding)
                    if sim >= SIM_THRESHOLD:
                        # Strong similarity - definitely same event
                        key = (i, j, 'semantic')
                        if key not in edge_set:
                            edge_set.add(key)
                            edges.append({
                                'source': i,
                                'target': j,
                                'relation': 'semantic',
                                'weight': sim,
                                'similarity': round(sim, 3)
                            })
                    elif sim >= WEAK_SIM_THRESHOLD:
                        # Weak similarity - potentially same event (weight is lower)
                        key = (i, j, 'semantic_weak')
                        if key not in edge_set:
                            edge_set.add(key)
                            edges.append({
                                'source': i,
                                'target': j,
                                'relation': 'semantic_weak',
                                'weight': sim * 0.5,  # Lower weight for weak similarity
                                'similarity': round(sim, 3)
                            })

        # Build adjacency for connected components
        adj = defaultdict(set)
        for e in edges:
            adj[e['source']].add(e['target'])
            adj[e['target']].add(e['source'])

        # Find connected components (surfaces)
        visited = set()
        surfaces = []

        for i in range(len(self.beliefs)):
            if i not in visited:
                stack = [i]
                component = []
                while stack:
                    n = stack.pop()
                    if n in visited or n >= len(self.beliefs):
                        continue
                    visited.add(n)
                    component.append(n)
                    stack.extend(adj[n] - visited)
                surfaces.append(component)

        # Sort surfaces by size (largest first)
        surfaces.sort(key=len, reverse=True)

        # Compute surface properties for hierarchical emergence
        surface_data = []
        for idx, comp in enumerate(surfaces):
            beliefs_in_surface = [self.beliefs[i] for i in comp if i < len(self.beliefs)]
            sources = set().union(*[b.sources for b in beliefs_in_surface])

            # Centroid embedding (mean of belief embeddings)
            embeddings = [b.embedding for b in beliefs_in_surface if b.embedding]
            centroid = np.mean(embeddings, axis=0).tolist() if embeddings else None

            # Mass calculation (from breathing_event.py)
            # mass = size * 0.1 * (0.5 + coherence) * (1 + 0.1 * sources)
            size = len(comp)
            coherence = 1.0  # TODO: compute from internal conflicts
            mass = size * 0.1 * (0.5 + coherence) * (1 + 0.1 * len(sources))

            surface_data.append({
                'id': idx,
                'beliefs': comp,
                'size': size,
                'total_sources': len(sources),
                'mass': round(mass, 2),
                'coherence': coherence,
                'centroid': centroid,
                'level': 0,  # Base level - sub-events
                'label': beliefs_in_surface[0].text[:50] if beliefs_in_surface else ''
            })

        # Identify potential merges (surfaces that could form higher-order events)
        potential_merges = []
        for i, s1 in enumerate(surface_data):
            for j, s2 in enumerate(surface_data):
                if i < j and s1['centroid'] and s2['centroid']:
                    sim = cosine_sim(s1['centroid'], s2['centroid'])
                    if sim >= SURFACE_MERGE_THRESHOLD:
                        potential_merges.append({
                            'surface_a': i,
                            'surface_b': j,
                            'similarity': round(sim, 3),
                            'combined_mass': round(s1['mass'] + s2['mass'], 2),
                            'label_a': s1['label'][:30],
                            'label_b': s2['label'][:30]
                        })

        return {
            'nodes': [
                {
                    'id': i,
                    'text': b.text,
                    'sources': list(b.sources),
                    'source_count': len(b.sources),
                    'entropy': b.entropy(),
                    'confidence': 'confirmed' if len(b.sources) >= 3 else
                                 'likely' if len(b.sources) >= 2 else 'reported'
                }
                for i, b in enumerate(self.beliefs)
            ],
            'edges': edges,
            'surfaces': surface_data,
            'potential_merges': potential_merges,  # Candidates for hierarchical events
            'stats': {
                'total_beliefs': len(self.beliefs),
                'total_edges': len(edges),
                'connected_surfaces': len([s for s in surfaces if len(s) > 1]),
                'isolated_beliefs': len([s for s in surfaces if len(s) == 1]),
                'total_mass': round(sum(s['mass'] for s in surface_data), 2),
                'merge_candidates': len(potential_merges)
            }
        }

    def summary(self) -> Dict:
        """Current state summary."""
        relations = {}
        for h in self.history:
            rel = h['result'].get('relation', 'unknown')
            relations[rel] = relations.get(rel, 0) + 1

        # Group beliefs by confidence
        confirmed = [b for b in self.beliefs if len(b.sources) >= 3]
        corroborated = [b for b in self.beliefs if len(b.sources) == 2]
        single_source = [b for b in self.beliefs if len(b.sources) == 1]

        # Efficiency stats
        total_potential_llm = len(self.history) - 1  # First claim never calls LLM
        llm_saved_pct = (self.llm_calls_skipped / max(total_potential_llm, 1)) * 100

        return {
            'total_claims': len(self.history),
            'total_beliefs': len(self.beliefs),
            'compression': len(self.history) / max(len(self.beliefs), 1),
            'confirmed': len(confirmed),
            'corroborated': len(corroborated),
            'single_source': len(single_source),
            'conflicts': len(self.conflicts),
            'entropy': self.entropy(),
            'coherence': self.coherence(),
            'relations': relations,
            # Efficiency metrics
            'llm_calls': self.llm_calls,
            'llm_calls_skipped': self.llm_calls_skipped,
            'llm_efficiency': f'{llm_saved_pct:.0f}% saved',
            'beliefs': [
                {
                    'text': b.text,
                    'sources': list(b.sources),
                    'entropy': b.entropy(),
                    'superseded': b.superseded
                }
                for b in self.beliefs
            ],
            'unresolved_conflicts': [
                {
                    'new_claim': c.new_claim,
                    'existing': c.existing_text,
                    'reasoning': c.reasoning
                }
                for c in self.conflicts
            ]
        }

    async def generate_prose(self) -> str:
        """Generate narrative from current beliefs."""
        if not self.beliefs:
            return "Awaiting information..."

        # Group by confidence
        confirmed = [b for b in self.beliefs if len(b.sources) >= 3]
        corroborated = [b for b in self.beliefs if len(b.sources) == 2]
        single_source = [b for b in self.beliefs if len(b.sources) == 1]

        beliefs_text = ""
        if confirmed:
            beliefs_text += "CONFIRMED (3+ sources):\n"
            beliefs_text += "\n".join(f"- {b.text}" for b in confirmed[:5])
            beliefs_text += "\n\n"
        if corroborated:
            beliefs_text += "CORROBORATED (2 sources):\n"
            beliefs_text += "\n".join(f"- {b.text}" for b in corroborated[:5])
            beliefs_text += "\n\n"
        if single_source[:3]:
            beliefs_text += "REPORTED (single source):\n"
            beliefs_text += "\n".join(f"- {b.text}" for b in single_source[:3])

        if self.conflicts:
            beliefs_text += "\n\nUNRESOLVED:\n"
            for c in self.conflicts[:2]:
                beliefs_text += f"- Dispute: {c.reasoning[:60]}...\n"

        prompt = f"""Write a news summary based ONLY on these beliefs:

{beliefs_text}

RULES:
- ONLY include facts from beliefs above
- For CONFIRMED: state as fact
- For CORROBORATED: use hedging ("sources report...")
- For REPORTED: note uncertainty ("one source claims...")
- Maximum 100 words

Return ONLY the prose."""

        try:
            response = await self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"


# =============================================================================
# TEST
# =============================================================================

async def test_kernel():
    """Test with sample claims."""
    import os
    from openai import AsyncOpenAI

    llm = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    kernel = EpistemicKernel(llm_client=llm)

    # Test claims (same event, different sources)
    claims = [
        ("At least 11 people killed in Hong Kong fire", "BBC"),
        ("Fire breaks out at Wang Fuk Court in Tai Po", "SCMP"),
        ("11 confirmed dead in apartment blaze", "Reuters"),
        ("Death toll rises to 36 as rescue continues", "AP"),
        ("Fire originated on 14th floor", "Police"),
        ("36 dead, over 70 injured", "HK Gov"),
        ("Death toll hits 128 as DNA identification proceeds", "HK Gov"),
        ("Fire believed to have started on 15th floor", "Fire Dept"),
    ]

    print("=" * 60)
    print("UNIFIED EPISTEMIC KERNEL TEST")
    print("=" * 60)

    for text, source in claims:
        result = await kernel.process(text, source)
        rel = result['relation']
        symbol = {'novel': '+', 'confirms': '=', 'refines': '↑',
                  'supersedes': '→', 'conflicts': '!'}[rel]
        print(f"  [{symbol}] {rel:10s} | {text[:45]}...")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    summary = kernel.summary()
    print(f"\nClaims: {summary['total_claims']} → Beliefs: {summary['total_beliefs']} "
          f"({summary['compression']:.1f}x compression)")
    print(f"Entropy: {summary['entropy']:.2f}, Coherence: {summary['coherence']:.2%}")

    print(f"\nBeliefs by confidence:")
    print(f"  Confirmed (3+): {summary['confirmed']}")
    print(f"  Corroborated (2): {summary['corroborated']}")
    print(f"  Single source: {summary['single_source']}")
    print(f"  Conflicts: {summary['conflicts']}")

    print(f"\nRelations: {summary['relations']}")

    print(f"\nCurrent beliefs:")
    for b in summary['beliefs'][:10]:
        src_count = len(b['sources'])
        ent = b['entropy']
        marker = "✓" if src_count >= 2 else "○"
        print(f"  {marker} [{src_count}] H={ent:.2f} | {b['text'][:55]}...")

    if summary['unresolved_conflicts']:
        print(f"\nConflicts:")
        for c in summary['unresolved_conflicts'][:3]:
            print(f"  ! {c['reasoning'][:60]}...")

    print("\n" + "=" * 60)
    print("PROSE")
    print("=" * 60)
    prose = await kernel.generate_prose()
    print(prose)

    return kernel


if __name__ == '__main__':
    asyncio.run(test_kernel())
