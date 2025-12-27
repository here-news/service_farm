"""
Epistemic Kernel
=================

Domain-agnostic epistemic engine for claim processing.

Components:
- Topology (from topology.py): Pure mathematical structures
- Classifier: LLM-based relation classification
- Kernel: Orchestration and state management

Philosophy:
- Claims are input; truth emerges from topology
- Embedding pre-filter reduces LLM calls
- q1/q2 pattern: claims relate only if they answer the same question

NO domain-specific logic in this file.
Use HintExtractor for domain-specific enhancements.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Protocol

from .topology import (
    Topology, Node, Edge, Surface, Relation,
    cosine_similarity, validate_jaynes
)


# =============================================================================
# SIMILARITY THRESHOLDS
# =============================================================================

SIM_THRESHOLD = 0.65       # High similarity = skip LLM, directly confirms
WEAK_SIM_THRESHOLD = 0.50  # Medium similarity = LLM with hint


# =============================================================================
# HINT EXTRACTOR PROTOCOL (for domain-specific logic)
# =============================================================================

class HintExtractor(Protocol):
    """
    Protocol for domain-specific hint extraction.

    Implementations can provide:
    - Update language detection
    - Numeric extraction
    - Temporal markers
    - Domain-specific patterns
    """
    def extract(self, text: str) -> Dict:
        """Extract hints from claim text."""
        ...


class DefaultHintExtractor:
    """
    Default hint extractor - no domain assumptions.

    Only extracts universal patterns:
    - Numbers
    - Basic temporal words
    """

    def extract(self, text: str) -> Dict:
        text_lower = text.lower()
        hints = {
            'numbers': [],
            'has_update_language': False,
            'numeric_value': None,
        }

        # Extract numbers (universal)
        numbers = re.findall(r'\b(\d+)\b', text)
        hints['numbers'] = [int(n) for n in numbers if int(n) > 0]
        if hints['numbers']:
            hints['numeric_value'] = max(hints['numbers'])

        # Basic temporal markers (universal, not domain-specific)
        temporal_words = ['now', 'updated', 'latest', 'current', 'new']
        hints['has_update_language'] = any(w in text_lower for w in temporal_words)

        return hints


# =============================================================================
# CONFLICT RECORD
# =============================================================================

@dataclass
class Conflict:
    """Record of unresolved conflict between claims."""
    node_id: str
    new_claim: str
    new_source: str
    existing_text: str
    reasoning: str


# =============================================================================
# EPISTEMIC KERNEL
# =============================================================================

class EpistemicKernel:
    """
    Domain-agnostic epistemic engine.

    State:
    - topology: Hypergeometric belief structure
    - conflicts: Unresolved contradictions
    - history: Claim processing log

    Operations:
    - process(claim, source): Ingest claim, update topology
    - topology(): Get current belief structure

    Configuration:
    - llm_client: OpenAI-compatible async client
    - hint_extractor: Domain-specific pattern extraction
    - prompt_template: Custom LLM prompt (optional)
    """

    def __init__(
        self,
        llm_client=None,
        use_embeddings: bool = True,
        hint_extractor: HintExtractor = None,
        prompt_template: str = None
    ):
        self.llm = llm_client
        self.use_embeddings = use_embeddings
        self.hint_extractor = hint_extractor or DefaultHintExtractor()
        self.prompt_template = prompt_template

        # State
        self.topo = Topology()
        self.conflicts: List[Conflict] = []
        self.history: List[Dict] = []
        self._claim_counter = 0

        # Efficiency tracking
        self.llm_calls = 0
        self.llm_calls_skipped = 0

    # =========================================================================
    # BACKWARDS COMPATIBILITY
    # =========================================================================

    @property
    def beliefs(self) -> List[Node]:
        """Backwards compatibility: access nodes as beliefs."""
        return self.topo.nodes

    # =========================================================================
    # ID GENERATION
    # =========================================================================

    def _next_claim_id(self) -> str:
        self._claim_counter += 1
        return f"c{self._claim_counter:03d}"

    def _next_node_id(self) -> str:
        return f"n{len(self.topo.nodes):03d}"

    # =========================================================================
    # EMBEDDING
    # =========================================================================

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding vector for text."""
        if not self.use_embeddings or not self.llm:
            return None
        try:
            response = await self.llm.embeddings.create(
                model="text-embedding-3-small",
                input=text[:8000]
            )
            return response.data[0].embedding
        except Exception:
            return None

    # =========================================================================
    # CORE OPERATION: PROCESS CLAIM
    # =========================================================================

    async def process(self, claim_text: str, source: str) -> Dict:
        """
        Process one claim.

        Flow:
        1. Get embedding for pre-filter
        2. If no nodes, create first node
        3. Else: use embedding similarity to gate LLM
           - High similarity: skip LLM, directly CONFIRMS
           - Medium similarity: LLM with hint
           - Low similarity: full LLM classification
        4. Update topology based on relation

        Returns: {relation, affected_belief, reasoning, ...}
        """
        claim_id = self._next_claim_id()
        hints = self.hint_extractor.extract(claim_text)

        # Get embedding FIRST (for pre-filter)
        claim_embedding = await self._get_embedding(claim_text)

        # First claim = first node
        if not self.topo.nodes:
            node = Node(
                id=self._next_node_id(),
                text=claim_text,
                sources={source},
                claim_ids={claim_id},
                embedding=claim_embedding
            )
            self.topo.add_node(node)
            result = {
                'relation': Relation.NOVEL.value,
                'affected_belief': None,
                'reasoning': 'First claim establishes initial node'
            }
            self._record_history(claim_text, source, claim_id, result)
            return result

        # =====================================================================
        # EMBEDDING PRE-FILTER
        # =====================================================================
        best_sim = 0.0
        best_idx = None

        if claim_embedding:
            for i, node in enumerate(self.topo.nodes):
                if node.embedding:
                    sim = cosine_similarity(claim_embedding, node.embedding)
                    if sim > best_sim:
                        best_sim = sim
                        best_idx = i

        # Decision based on similarity
        if best_sim >= SIM_THRESHOLD:
            if hints['has_update_language']:
                # Potential update - need LLM
                self.llm_calls += 1
                result = await self._classify_with_llm(
                    claim_text, source, hints,
                    similarity_hint=best_sim,
                    similar_node_idx=best_idx
                )
            else:
                # Same fact - skip LLM
                self.llm_calls_skipped += 1
                result = {
                    'relation': Relation.CONFIRMS.value,
                    'affected_belief': best_idx,
                    'reasoning': f'High embedding similarity ({best_sim:.2f})',
                    'normalized_claim': claim_text,
                    'skipped_llm': True,
                    'similarity': best_sim
                }
        elif best_sim >= WEAK_SIM_THRESHOLD:
            # Medium similarity - LLM with hint
            self.llm_calls += 1
            result = await self._classify_with_llm(
                claim_text, source, hints,
                similarity_hint=best_sim,
                similar_node_idx=best_idx
            )
        else:
            # Low similarity - full LLM
            self.llm_calls += 1
            result = await self._classify_with_llm(claim_text, source, hints)

        # Update topology
        self._apply_relation(result, claim_text, source, claim_id, claim_embedding)
        self._record_history(claim_text, source, claim_id, result)

        return result

    def _apply_relation(
        self,
        result: Dict,
        claim_text: str,
        source: str,
        claim_id: str,
        embedding: Optional[List[float]]
    ):
        """Apply classified relation to topology."""
        relation = Relation(result['relation'])
        affected_idx = result.get('affected_belief')

        if relation == Relation.NOVEL:
            node = Node(
                id=self._next_node_id(),
                text=claim_text,
                sources={source},
                claim_ids={claim_id},
                embedding=embedding,
                metadata={'question_type': result.get('question_answered')}
            )
            self.topo.add_node(node)

        elif relation == Relation.CONFIRMS:
            if affected_idx is not None and 0 <= affected_idx < len(self.topo.nodes):
                self.topo.nodes[affected_idx].add_source(source, claim_id)

        elif relation == Relation.REFINES:
            if affected_idx is not None and 0 <= affected_idx < len(self.topo.nodes):
                normalized = result.get('normalized_claim', claim_text)
                self.topo.nodes[affected_idx].update(normalized, source, claim_id)

        elif relation == Relation.SUPERSEDES:
            if affected_idx is not None and 0 <= affected_idx < len(self.topo.nodes):
                normalized = result.get('normalized_claim', claim_text)
                self.topo.nodes[affected_idx].update(normalized, source, claim_id)

        elif relation == Relation.CONFLICTS:
            if affected_idx is not None and 0 <= affected_idx < len(self.topo.nodes):
                self.conflicts.append(Conflict(
                    node_id=self.topo.nodes[affected_idx].id,
                    new_claim=claim_text,
                    new_source=source,
                    existing_text=self.topo.nodes[affected_idx].text,
                    reasoning=result.get('reasoning', '')
                ))

        # Mark topology dirty
        self.topo._dirty = True

    def _record_history(self, claim: str, source: str, claim_id: str, result: Dict):
        """Record claim processing in history."""
        self.history.append({
            'claim': claim,
            'source': source,
            'claim_id': claim_id,
            'result': result
        })

    # =========================================================================
    # LLM CLASSIFICATION
    # =========================================================================

    def _format_nodes_for_prompt(self) -> str:
        """Format current nodes for LLM context."""
        if not self.topo.nodes:
            return "(no nodes yet)"

        lines = []
        for i, n in enumerate(self.topo.nodes):
            confidence = n.confidence_level().upper()
            lines.append(f"[{i}] ({confidence}, {n.source_count} sources) {n.text}")

        return "\n".join(lines)

    async def _classify_with_llm(
        self,
        claim: str,
        source: str,
        hints: Dict,
        similarity_hint: float = None,
        similar_node_idx: int = None
    ) -> Dict:
        """
        Classify relationship using LLM.

        Key insight: q1/q2 pattern
        Claims only relate if they answer the SAME question.
        """

        # Similarity context
        similarity_context = ""
        if similarity_hint is not None and similar_node_idx is not None:
            similar_node = self.topo.nodes[similar_node_idx]
            similarity_context = f"""
EMBEDDING SIMILARITY HINT:
This claim has {similarity_hint:.0%} semantic similarity to node [{similar_node_idx}].
High similarity (>65%) suggests they answer the same question.
If values differ and there's update language, this is likely SUPERSEDES.
If values are the same, this is likely CONFIRMS.
"""

        # Use custom prompt or default
        if self.prompt_template:
            prompt = self.prompt_template.format(
                nodes=self._format_nodes_for_prompt(),
                claim=claim,
                source=source,
                numbers=hints['numbers'],
                has_update_language=hints['has_update_language'],
                similarity_context=similarity_context
            )
        else:
            prompt = self._default_prompt(claim, source, hints, similarity_context)

        try:
            response = await self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            result = json.loads(response.choices[0].message.content)

            # Normalize relation
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

    def _default_prompt(
        self,
        claim: str,
        source: str,
        hints: Dict,
        similarity_context: str
    ) -> str:
        """Default domain-agnostic classification prompt."""
        return f"""Compare a new claim to existing propositions.

EXISTING PROPOSITIONS:
{self._format_nodes_for_prompt()}

NEW CLAIM: "{claim}"
SOURCE: {source}
EXTRACTED: numbers={hints['numbers']}, has_update_language={hints['has_update_language']}
{similarity_context}

STEP 1 - IDENTIFY THE QUESTION:
What specific question does this claim answer?
Examples: "quantity of X", "location of Y", "cause of Z", "identity of W"

STEP 2 - FIND MATCHING PROPOSITION:
Does any existing proposition answer the SAME question?
If not -> NOVEL (different questions = no relationship)

STEP 3 - IF SAME QUESTION, DETERMINE RELATIONSHIP:
- CONFIRMS: Same question, same answer (corroboration)
- REFINES: Same question, more specific (adds detail)
- SUPERSEDES: Same question, different answer WITH update language
- CONFLICTS: Same question, different answer, NO temporal ordering

Return JSON:
{{
  "question_answered": "what question this claim answers",
  "matching_proposition_question": "what question the matching proposition answers (or null)",
  "relation": "NOVEL|CONFIRMS|REFINES|SUPERSEDES|CONFLICTS",
  "affected_belief": <index number or null>,
  "reasoning": "one sentence explanation",
  "normalized_claim": "the claim in clear standalone form"
}}"""

    # =========================================================================
    # METRICS
    # =========================================================================

    def entropy(self) -> float:
        """Overall epistemic entropy (average across nodes)."""
        return self.topo.total_entropy()

    def coherence(self) -> float:
        """
        How consistent is the state?
        High coherence = few conflicts, lots of corroboration.
        """
        if not self.history:
            return 1.0

        total = len(self.history)
        conflicts = len(self.conflicts)

        agreements = sum(
            1 for h in self.history
            if h['result'].get('relation') in ('confirms', 'refines', 'supersedes')
        )

        raw = (agreements - conflicts * 2) / max(total, 1)
        return max(0.0, min(1.0, 0.5 + raw * 0.5))

    # =========================================================================
    # OUTPUT
    # =========================================================================

    def topology(self) -> Dict:
        """Return the belief topology as a dictionary."""
        # Ensure computed
        self.topo.compute()

        # Get base topology
        result = self.topo.to_dict()

        # Add potential merges (for hierarchical events)
        potential_merges = []
        for i, s1 in enumerate(self.topo.surfaces):
            for j, s2 in enumerate(self.topo.surfaces):
                if i < j and s1.centroid and s2.centroid:
                    sim = cosine_similarity(s1.centroid, s2.centroid)
                    if sim >= Topology.MERGE_THRESHOLD:
                        potential_merges.append({
                            'surface_a': i,
                            'surface_b': j,
                            'similarity': round(sim, 3),
                            'combined_mass': round(s1.mass + s2.mass, 2)
                        })

        result['potential_merges'] = potential_merges
        result['stats']['merge_candidates'] = len(potential_merges)

        return result

    def summary(self) -> Dict:
        """Current state summary."""
        relations = {}
        for h in self.history:
            rel = h['result'].get('relation', 'unknown')
            relations[rel] = relations.get(rel, 0) + 1

        confirmed = [n for n in self.topo.nodes if n.source_count >= 3]
        corroborated = [n for n in self.topo.nodes if n.source_count == 2]
        single_source = [n for n in self.topo.nodes if n.source_count == 1]

        # Efficiency
        total_potential = len(self.history) - 1
        saved_pct = (self.llm_calls_skipped / max(total_potential, 1)) * 100

        return {
            'total_claims': len(self.history),
            'total_beliefs': len(self.topo.nodes),
            'compression': len(self.history) / max(len(self.topo.nodes), 1),
            'confirmed': len(confirmed),
            'corroborated': len(corroborated),
            'single_source': len(single_source),
            'conflicts': len(self.conflicts),
            'entropy': self.entropy(),
            'coherence': self.coherence(),
            'relations': relations,
            'llm_calls': self.llm_calls,
            'llm_calls_skipped': self.llm_calls_skipped,
            'llm_efficiency': f'{saved_pct:.0f}% saved',
            'beliefs': [
                {
                    'text': n.text,
                    'sources': list(n.sources),
                    'entropy': n.entropy(),
                    'plausibility': n.plausibility(),
                    'superseded': n.superseded
                }
                for n in self.topo.nodes
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

    def validate(self) -> Dict:
        """Validate topology against Jaynes principle."""
        return validate_jaynes(self.topo)


# =============================================================================
# BACKWARDS COMPATIBILITY: Belief alias
# =============================================================================

# For code that imports Belief from kernel
Belief = Node


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

async def process_claims(
    claims: List[tuple],
    llm_client,
    hint_extractor: HintExtractor = None
) -> EpistemicKernel:
    """
    Process a list of (text, source) claims.

    Returns: Configured EpistemicKernel with processed claims.
    """
    kernel = EpistemicKernel(
        llm_client=llm_client,
        hint_extractor=hint_extractor
    )

    for text, source in claims:
        await kernel.process(text, source)

    return kernel
