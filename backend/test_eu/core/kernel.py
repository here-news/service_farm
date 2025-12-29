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

Repository Integration:
- Optional ClaimRepository for pgvector similarity search
- Falls back to in-memory O(n) if no repository provided
- See WEAVER_HYPOTHESIS.md for architecture details
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Protocol, Tuple, TYPE_CHECKING

from .topology import (
    Topology, Node, Edge, Surface, Relation,
    cosine_similarity, validate_jaynes
)

if TYPE_CHECKING:
    from repositories.claim_repository import ClaimRepository
    from models.domain.claim import Claim as DomainClaim


# =============================================================================
# SIMILARITY THRESHOLDS
# =============================================================================

SIM_THRESHOLD = 0.85       # Very high similarity = skip LLM (conservative - prevents false CONFIRMS)
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

    Note: Event detection is handled at a higher level (see EventInterpreter).
    The kernel focuses on pure epistemic structure.
    """

    def __init__(
        self,
        llm_client=None,
        use_embeddings: bool = True,
        hint_extractor: HintExtractor = None,
        prompt_template: str = None,
        confidence_threshold: float = 0.5,
        claim_repository: 'ClaimRepository' = None
    ):
        self.llm = llm_client
        self.use_embeddings = use_embeddings
        self.hint_extractor = hint_extractor or DefaultHintExtractor()
        self.prompt_template = prompt_template
        self.confidence_threshold = confidence_threshold  # Min confidence to apply relation

        # Repository for pgvector similarity (optional - falls back to in-memory O(n))
        self.claim_repository = claim_repository

        # State
        self.topo = Topology()
        self.conflicts: List[Conflict] = []
        self.history: List[Dict] = []
        self._claim_counter = 0

        # Map claim_id -> node_idx for repository-backed mode
        self._claim_to_node: Dict[str, int] = {}

        # Efficiency tracking
        self.llm_calls = 0
        self.llm_calls_skipped = 0
        self.pgvector_queries = 0

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
            # Use text-embedding-3-small (1536 dims) to match stored claim embeddings
            # Note: Can switch to text-embedding-3-large (3072 dims) after backfilling
            response = await self.llm.embeddings.create(
                model="text-embedding-3-small",
                input=text[:8000]
            )
            return response.data[0].embedding
        except Exception:
            return None

    # =========================================================================
    # SIMILARITY SEARCH (pgvector or in-memory)
    # =========================================================================

    async def _find_similar(
        self,
        embedding: List[float],
        exclude_claim_id: str = None
    ) -> Tuple[float, Optional[int], Optional[str]]:
        """
        Find most similar existing node.

        Uses pgvector if claim_repository is available, else O(n) in-memory.

        Returns:
            (best_similarity, best_node_idx, best_claim_id)
        """
        if self.claim_repository and embedding:
            # Use pgvector for efficient similarity search
            return await self._find_similar_pgvector(embedding, exclude_claim_id)
        else:
            # Fall back to in-memory O(n) search
            return self._find_similar_inmemory(embedding)

    async def _find_similar_pgvector(
        self,
        embedding: List[float],
        exclude_claim_id: str = None
    ) -> Tuple[float, Optional[int], Optional[str]]:
        """
        pgvector-backed similarity search.

        Queries PostgreSQL for top candidates, returns best match.
        """
        self.pgvector_queries += 1

        exclude_ids = [exclude_claim_id] if exclude_claim_id else []
        results = await self.claim_repository.find_similar(
            embedding=embedding,
            limit=5,
            exclude_claim_ids=exclude_ids
        )

        if not results:
            return 0.0, None, None

        # Find best match that's in our topology
        for r in results:
            claim_id = r['claim_id']
            similarity = r['similarity']

            if claim_id in self._claim_to_node:
                node_idx = self._claim_to_node[claim_id]
                return similarity, node_idx, claim_id

        # Best match not in our topology yet - return it anyway for potential linking
        best = results[0]
        return best['similarity'], None, best['claim_id']

    def _find_similar_inmemory(
        self,
        embedding: List[float]
    ) -> Tuple[float, Optional[int], Optional[str]]:
        """
        In-memory O(n) similarity search.

        Used when no repository is available.
        """
        best_sim = 0.0
        best_idx = None

        if embedding:
            for i, node in enumerate(self.topo.nodes):
                if node.embedding:
                    sim = cosine_similarity(embedding, node.embedding)
                    if sim > best_sim:
                        best_sim = sim
                        best_idx = i

        return best_sim, best_idx, None

    # =========================================================================
    # CORE OPERATION: PROCESS CLAIM
    # =========================================================================

    async def process(
        self,
        claim_text: str,
        source: str,
        embedding: List[float] = None,
        similar_candidates: List[Tuple[int, float]] = None,
        claim_id: str = None
    ) -> Dict:
        """
        Process one claim through the epistemic kernel.

        Args:
            claim_text: The pure claim text
            source: Source identifier (e.g., domain name)
            embedding: Pre-computed embedding from data layer (for storage in node)
            similar_candidates: Pre-computed similarity candidates from pgvector.
                               List of (node_idx, similarity_score) tuples, sorted by similarity desc.
                               If None, falls back to in-memory search (for testing).
            claim_id: Optional claim ID (for tracking). Auto-generated if not provided.

        Flow:
        1. If no nodes, create first node
        2. Use similarity candidates to gate LLM:
           - High similarity + no update language: skip LLM, CONFIRMS
           - Otherwise: LLM classification
        3. Update topology based on relation

        Returns: {relation, affected_belief, reasoning, confidence, ...}
        """
        if claim_id is None:
            claim_id = self._next_claim_id()
        hints = self.hint_extractor.extract(claim_text)

        # First claim = first node
        if not self.topo.nodes:
            node = Node(
                id=self._next_node_id(),
                text=claim_text,
                sources={source},
                claim_ids={claim_id},
                embedding=embedding
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
        # SIMILARITY-BASED GATING
        # =====================================================================
        # Use pre-computed candidates from data layer (pgvector)
        # Fall back to in-memory if no candidates provided (for testing)
        if similar_candidates is not None:
            best_idx, best_sim = similar_candidates[0] if similar_candidates else (None, 0.0)
        elif embedding:
            # Fallback: in-memory O(n) search (for testing without pgvector)
            best_sim, best_idx, _ = self._find_similar_inmemory(embedding)
        else:
            best_sim, best_idx = 0.0, None

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
        self._apply_relation(result, claim_text, source, claim_id, embedding)
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
        """Apply classified relation to topology based on confidence threshold."""
        relation = Relation(result['relation'])
        affected_idx = result.get('affected_belief')

        # Parse confidence (support both numeric and string for robustness)
        confidence = result.get('confidence', 0.5)
        if isinstance(confidence, str):
            confidence = {'HIGH': 0.9, 'MEDIUM': 0.6, 'LOW': 0.3}.get(confidence.upper(), 0.5)
        confidence = float(confidence)

        # Below threshold: treat as NOVEL but store tentative relationship
        if confidence < self.confidence_threshold and relation != Relation.NOVEL:
            result['applied_relation'] = 'novel'
            result['tentative_relation'] = relation.value
            result['tentative_target'] = affected_idx
            result['below_threshold'] = True
            relation = Relation.NOVEL
            affected_idx = None
        else:
            result['applied_relation'] = relation.value
            result['below_threshold'] = False

        if relation == Relation.NOVEL:
            # Build metadata
            node_metadata = {
                'confidence': confidence
            }
            # If downgraded from another relation, store tentative link
            if result.get('tentative_relation'):
                node_metadata['tentative_relation'] = result['tentative_relation']
                node_metadata['tentative_target'] = result['tentative_target']

            node = Node(
                id=self._next_node_id(),
                text=claim_text,
                sources={source},
                claim_ids={claim_id},
                embedding=embedding,
                metadata=node_metadata
            )
            node_idx = len(self.topo.nodes)
            self.topo.add_node(node)

            # Track claim_id -> node_idx for pgvector mode
            self._claim_to_node[claim_id] = node_idx

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

        # Similarity context - warn LLM that embeddings capture topic, not event identity
        similarity_context = ""
        if similarity_hint is not None and similar_node_idx is not None:
            similar_node = self.topo.nodes[similar_node_idx]
            similarity_context = f"""
EMBEDDING CONTEXT:
This claim has {similarity_hint:.0%} semantic similarity to node [{similar_node_idx}].
CAUTION: Embedding similarity captures TOPIC overlap (e.g., "missing persons", "investigation").
It does NOT indicate same real-world event. Two claims about different incidents can have high similarity.
You must determine event identity from EXPLICIT shared entities/locations, not from this similarity score.
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

STEP 1 - EXTRACT ENTITIES FROM NEW CLAIM ONLY:
List entities (people, places, organizations) that are EXPLICITLY mentioned in the NEW CLAIM text.
Do NOT include entities from existing propositions. Only extract what appears in the new claim.
What real-world event does this claim refer to (if identifiable from the claim text)?

STEP 2 - FIND MATCHING PROPOSITION:
Does any existing proposition refer to the SAME SPECIFIC real-world event?
Evidence for same event:
- Shared specific location (e.g., "Wang Fuk Court", not just "Hong Kong")
- Shared specific people (e.g., "John Lee")
- Same specific incident explicitly referenced

STEP 3 - DETERMINE RELATIONSHIP AND CONFIDENCE:
- CONFIRMS: Same claim, different source (corroboration)
- REFINES: Adds detail to same event (more specific)
- SUPERSEDES: Updates a value WITH temporal/update language
- CONFLICTS: Contradicts existing claim about same event
- NOVEL: Different event or unclear if same event

CONFIDENCE (0.0 to 1.0):
- 0.9-1.0: Explicit shared specific entities (e.g., both mention "Wang Fuk Court fire")
- 0.6-0.8: Strong implicit connection (e.g., "John Lee investigation" after fire claims)
- 0.3-0.5: Possible connection, thematic similarity only
- 0.0-0.2: Unlikely same event, generic topic overlap

Return JSON:
{{
  "entities": ["list of entities mentioned in new claim"],
  "event_reference": "what real-world event this refers to",
  "shared_entities": ["entities shared with matching proposition"],
  "same_event": true/false,
  "confidence": <0.0 to 1.0>,
  "relation": "NOVEL|CONFIRMS|REFINES|SUPERSEDES|CONFLICTS",
  "affected_belief": <index number or null>,
  "reasoning": "one sentence explanation"
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
            'pgvector_queries': self.pgvector_queries,
            'similarity_mode': 'pgvector' if self.claim_repository else 'in-memory',
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

    # =========================================================================
    # SURFACE-LEVEL PROSE GENERATION
    # =========================================================================

    async def label_surface(self, surface_idx: int) -> str:
        """
        Generate a short descriptive label for a surface.

        Uses LLM to summarize what the surface is about.
        """
        self.topo.compute()

        if surface_idx >= len(self.topo.surfaces):
            return "Unknown"

        surface = self.topo.surfaces[surface_idx]
        nodes = [self.topo.nodes[i] for i in surface.node_indices]

        if not nodes:
            return "Empty surface"

        # Get sample beliefs from surface
        beliefs_text = "\n".join(f"- {n.text[:100]}" for n in nodes[:5])

        prompt = f"""Based on these beliefs, generate a SHORT (3-6 word) descriptive label:

{beliefs_text}

Return ONLY the label, nothing else. Examples:
- "Hong Kong High-Rise Fire"
- "Jimmy Lai National Security Trial"
- "Do Kwon Crypto Fraud Case"
"""

        try:
            response = await self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=20
            )
            label = response.choices[0].message.content.strip().strip('"')
            surface.label = label
            return label
        except Exception:
            return nodes[0].text[:50] if nodes else "Unknown"

    async def generate_surface_prose(self, surface_idx: int) -> Dict:
        """
        Generate epistemic prose for ONE surface.

        Returns:
            {
                'label': surface label,
                'prose': generated summary,
                'stats': {confirmed, corroborated, uncertain, contested}
            }
        """
        self.topo.compute()

        if surface_idx >= len(self.topo.surfaces):
            return {'label': 'Unknown', 'prose': '', 'stats': {}}

        surface = self.topo.surfaces[surface_idx]
        nodes = [self.topo.nodes[i] for i in surface.node_indices]

        if not nodes:
            return {'label': 'Empty', 'prose': 'No beliefs.', 'stats': {}}

        # Categorize by confidence
        confirmed = [n for n in nodes if n.source_count >= 3]
        corroborated = [n for n in nodes if n.source_count == 2]
        uncertain = [n for n in nodes if n.source_count == 1]

        # Find conflicts for this surface
        node_ids = {n.id for n in nodes}
        contested = [c for c in self.conflicts if c.node_id in node_ids]

        # Build belief text by confidence level
        beliefs_text = ""
        if confirmed:
            beliefs_text += "CONFIRMED (3+ independent sources):\n"
            beliefs_text += "\n".join(f"- {n.text}" for n in confirmed[:5])
            beliefs_text += "\n\n"

        if corroborated:
            beliefs_text += "CORROBORATED (2 sources):\n"
            beliefs_text += "\n".join(f"- {n.text}" for n in corroborated[:5])
            beliefs_text += "\n\n"

        if uncertain[:3]:
            beliefs_text += "REPORTED (single source - unconfirmed):\n"
            beliefs_text += "\n".join(f"- {n.text}" for n in uncertain[:3])
            beliefs_text += "\n\n"

        if contested:
            beliefs_text += "CONTESTED (conflicting reports):\n"
            for c in contested[:2]:
                beliefs_text += f"- Conflict: {c.reasoning[:60]}...\n"

        # Generate prose
        prompt = f"""Write a news summary based ONLY on these beliefs:

{beliefs_text}

RULES:
- ONLY include facts from beliefs above
- For CONFIRMED: state as fact
- For CORROBORATED: use hedging ("sources report...", "according to...")
- For REPORTED: note uncertainty ("one source claims...", "unconfirmed reports suggest...")
- For CONTESTED: note the dispute
- Maximum 100 words
- Be epistemically honest

Return ONLY the prose summary."""

        try:
            response = await self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            prose = response.choices[0].message.content.strip()
        except Exception as e:
            prose = f"Error generating prose: {e}"

        return {
            'label': surface.label or await self.label_surface(surface_idx),
            'prose': prose,
            'stats': {
                'confirmed': len(confirmed),
                'corroborated': len(corroborated),
                'uncertain': len(uncertain),
                'contested': len(contested),
                'total_nodes': len(nodes),
                'entropy': surface.entropy(),
                'sources': surface.total_sources
            }
        }

    async def generate_all_prose(self) -> List[Dict]:
        """
        Generate prose for ALL surfaces.

        Each surface = one event cluster.
        Returns list of {label, prose, stats} for each surface.
        """
        self.topo.compute()

        results = []
        for i, surface in enumerate(self.topo.surfaces):
            result = await self.generate_surface_prose(i)
            result['surface_id'] = i
            result['size'] = surface.size
            results.append(result)

        return results

    async def report(self) -> Dict:
        """
        Generate full epistemic report.

        Returns topology structure + per-surface prose.
        """
        self.topo.compute()

        # Get base topology
        topo_data = self.topology()

        # Generate prose for each surface
        surface_reports = await self.generate_all_prose()

        return {
            'topology': topo_data,
            'surfaces': surface_reports,
            'jaynes': self.validate(),
            'summary': self.summary()
        }


# =============================================================================
# BACKWARDS COMPATIBILITY: Belief alias
# =============================================================================

# For code that imports Belief from kernel
Belief = Node


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def process_claims(
    claims: List[tuple],
    llm_client,
    hint_extractor: HintExtractor = None
) -> EpistemicKernel:
    """
    Process a list of (text, source) or (text, source, embedding) claims.

    For testing/simple use cases without pgvector.

    Returns: Configured EpistemicKernel with processed claims.
    """
    kernel = EpistemicKernel(
        llm_client=llm_client,
        hint_extractor=hint_extractor
    )

    for item in claims:
        if len(item) == 3:
            text, source, embedding = item
        else:
            text, source = item
            embedding = None

        await kernel.process(text, source, embedding=embedding)

    return kernel


async def process_domain_claims(
    claims: List['DomainClaim'],
    llm_client,
    claim_repository: 'ClaimRepository' = None,
    hint_extractor: HintExtractor = None,
    embeddings: Dict[str, List[float]] = None
) -> EpistemicKernel:
    """
    Process a list of Claim domain models.

    Args:
        claims: List of Claim domain models (with entities hydrated)
        llm_client: OpenAI-compatible async client
        claim_repository: Optional - enables pgvector similarity search
        hint_extractor: Optional domain-specific hints
        embeddings: Dict mapping claim_id -> embedding vector (pre-computed)

    Returns: Configured EpistemicKernel with processed claims.

    Architecture:
        Data layer provides:
        - embeddings: Pre-computed from claim text (via OpenAI/etc)
        - similar_candidates: From pgvector query (via claim_repository)

        Kernel does:
        - LLM classification (CONFIRMS, REFINES, etc.)
        - Topology updates
    """
    kernel = EpistemicKernel(
        llm_client=llm_client,
        hint_extractor=hint_extractor
    )
    embeddings = embeddings or {}

    for claim in claims:
        # Get pre-computed embedding from data layer
        embedding = embeddings.get(claim.id)

        # Get similar candidates from pgvector if available
        similar_candidates = None
        if claim_repository and embedding:
            results = await claim_repository.find_similar(
                embedding=embedding,
                limit=5,
                exclude_claim_ids=[claim.id]
            )
            # Convert to (node_idx, similarity) tuples
            # Note: node_idx mapping requires tracking processed claim_ids
            similar_candidates = []
            for r in results:
                if r['claim_id'] in kernel._claim_to_node:
                    node_idx = kernel._claim_to_node[r['claim_id']]
                    similar_candidates.append((node_idx, r['similarity']))

        # Extract source domain
        source = claim.metadata.get('source_name', 'unknown') if claim.metadata else 'unknown'
        if not source or source == 'unknown':
            source = f"source_{claim.page_id[:8]}" if claim.page_id else 'unknown'

        result = await kernel.process(
            claim_text=claim.text,
            source=source,
            embedding=embedding,
            similar_candidates=similar_candidates if similar_candidates else None,
            claim_id=claim.id
        )

        # Track claim_id -> node_idx for subsequent pgvector lookups
        if result.get('applied_relation') == 'novel' or result.get('relation') == 'novel':
            kernel._claim_to_node[claim.id] = len(kernel.topo.nodes) - 1

    return kernel
