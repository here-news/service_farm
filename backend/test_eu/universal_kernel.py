"""
Universal Epistemic Kernel
==========================

Minimal Bayesian belief engine. Domain-agnostic.

One operation: UPDATE(evidence, state) → state'

The topology IS the knowledge:
- Nodes: evidence items (claims at any scale)
- Edges: relations (CONFIRMS, REFINES, SUPERSEDES, CONFLICTS)
- Derived: current beliefs, plausibility, gaps

Based on Jaynes: beliefs are posterior claims, priors for next update.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional, List, Tuple
from openai import AsyncOpenAI
import numpy as np
import json
import math
import re


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

Relation = Literal["NOVEL", "CONFIRMS", "REFINES", "SUPERSEDES", "CONFLICTS", "DIVERGENT"]

@dataclass
class Node:
    """Evidence unit. Could be claim, observation, hypothesis - same structure."""
    id: str
    content: str
    source: str
    time: Optional[datetime] = None
    time_precision: str = "unknown"  # "exact", "hour", "day", "unknown"
    confidence: float = 0.5          # Extraction confidence
    modality: str = "unknown"        # "observation", "reported_speech", etc.
    embedding: Optional[List[float]] = None  # For similarity pre-filtering


@dataclass
class Edge:
    """Relation between evidence."""
    from_id: str      # New evidence
    to_id: str        # Existing evidence it relates to
    relation: Relation
    reasoning: str = ""
    would_resolve: str = ""  # For DIVERGENT: what would clarify the relationship


@dataclass
class State:
    """Epistemic state = graph of evidence + relations."""
    nodes: dict = field(default_factory=dict)  # id -> Node
    edges: list = field(default_factory=list)  # List[Edge]

    def to_dict(self) -> dict:
        return {
            "nodes": {k: {"id": v.id, "content": v.content, "source": v.source}
                      for k, v in self.nodes.items()},
            "edges": [{"from": e.from_id, "to": e.to_id,
                       "relation": e.relation, "reasoning": e.reasoning,
                       "would_resolve": getattr(e, 'would_resolve', '')}
                      for e in self.edges],
            "metrics": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "belief_count": len(self.current_beliefs()),
                "conflict_count": len(self.conflicts()),
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> "State":
        """Restore state from saved topology."""
        state = cls()
        for nid, ndata in data.get("nodes", {}).items():
            state.nodes[nid] = Node(
                id=ndata["id"],
                content=ndata["content"],
                source=ndata.get("source", "unknown")
            )
        for edata in data.get("edges", []):
            state.edges.append(Edge(
                from_id=edata["from"],
                to_id=edata["to"],
                relation=edata["relation"],
                reasoning=edata.get("reasoning", ""),
                would_resolve=edata.get("would_resolve", "")
            ))
        return state

    # =========================================================================
    # DERIVED COMPUTATIONS (pure functions of state)
    # =========================================================================

    def current_beliefs(self) -> list[Node]:
        """Nodes not superseded by anything = current posterior."""
        superseded = {e.to_id for e in self.edges if e.relation == "SUPERSEDES"}
        return [n for n in self.nodes.values() if n.id not in superseded]

    def plausibility(self, node_id: str) -> float:
        """P(node) based on epistemic support. Bayesian with Laplace smoothing.

        Support types (weighted):
        - CONFIRMS: Full positive support (1.0)
        - REFINES: Partial positive support (0.5) - compatible, adds detail
        - CONFLICTS: Full negative support (1.0)
        - DIVERGENT: Partial negative support (0.5) - uncertain discrepancy
        """
        # Count incoming edges by type
        confirms = sum(1 for e in self.edges
                       if e.to_id == node_id and e.relation == "CONFIRMS")
        refines = sum(1 for e in self.edges
                      if e.to_id == node_id and e.relation == "REFINES")
        conflicts = sum(1 for e in self.edges
                        if e.to_id == node_id and e.relation == "CONFLICTS")
        divergent = sum(1 for e in self.edges
                        if e.to_id == node_id and e.relation == "DIVERGENT")

        # Weighted support: confirms + half-weight for refines
        positive = confirms + (refines * 0.5)
        # Weighted doubt: conflicts + half-weight for divergent
        negative = conflicts + (divergent * 0.5)

        # Laplace smoothing: (positive + 1) / (positive + negative + 2)
        return (positive + 1) / (positive + negative + 2)

    def support_count(self, node_id: str) -> int:
        """How many nodes confirm this one."""
        return sum(1 for e in self.edges
                   if e.to_id == node_id and e.relation == "CONFIRMS") + 1

    def conflicts(self) -> list[tuple[str, str, str]]:
        """Unresolved conflict pairs - OPPORTUNITIES for truth discovery."""
        return [(e.from_id, e.to_id, e.reasoning) for e in self.edges
                if e.relation == "CONFLICTS"]

    def divergent_pairs(self) -> list[tuple[str, str, str, str]]:
        """
        Pairs where temporal relationship is UNCLEAR.
        Returns (from_id, to_id, reasoning, would_resolve).
        These are opportunities - more info could clarify if SUPERSEDES or CONFLICTS.
        """
        return [(e.from_id, e.to_id, e.reasoning, e.would_resolve)
                for e in self.edges if e.relation == "DIVERGENT"]

    def opportunities(self) -> dict:
        """
        Gaps and conflicts = opportunities to discover truth.
        These are the most valuable signals in the topology.
        """
        return {
            "conflicts": [
                {
                    "claim_a": self.nodes[c[0]].content if c[0] in self.nodes else c[0],
                    "claim_b": self.nodes[c[1]].content if c[1] in self.nodes else c[1],
                    "source_a": self.nodes[c[0]].source if c[0] in self.nodes else "?",
                    "source_b": self.nodes[c[1]].source if c[1] in self.nodes else "?",
                    "reasoning": c[2] if len(c) > 2 else "",
                    "action": "Investigate which claim is correct"
                }
                for c in self.conflicts()
            ],
            "low_confidence": [
                {
                    "claim": n.content,
                    "source": n.source,
                    "support": self.support_count(n.id),
                    "action": "Seek corroboration from additional sources"
                }
                for n in self.current_beliefs()
                if self.support_count(n.id) == 1
            ]
        }

    def supersession_chain(self, node_id: str) -> list[str]:
        """Get chain of superseded nodes leading to this one."""
        chain = [node_id]
        for e in self.edges:
            if e.from_id == node_id and e.relation == "SUPERSEDES":
                chain = self.supersession_chain(e.to_id) + chain
                break
        return chain

    def entropy(self) -> float:
        """Shannon entropy of plausibility distribution.

        Only considers beliefs with actual evidence (edges).
        Orphan beliefs with no edges have no epistemic status.

        Returns 0-1 where:
        - 0 = complete certainty (one dominant belief, or all beliefs agree)
        - 1 = maximum uncertainty (uniform distribution among contested beliefs)
        """
        # Only consider beliefs that have incoming edges (actual epistemic evidence)
        involved_ids = set()
        for e in self.edges:
            involved_ids.add(e.to_id)  # Beliefs that are targets of edges

        beliefs = [b for b in self.current_beliefs() if b.id in involved_ids]

        if len(beliefs) <= 1:
            return 0.0  # No contested beliefs = complete certainty

        probs = [self.plausibility(b.id) for b in beliefs]
        total = sum(probs)
        if total == 0:
            return 1.0

        probs = [p / total for p in probs]
        raw_entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in probs)
        return raw_entropy / math.log2(len(probs))  # Normalize to 0-1

    def coherence(self) -> float:
        """1 - (uncertainty / total_edges).

        CONFLICTS = full uncertainty (1.0)
        DIVERGENT = partial uncertainty (0.5) - unresolved discrepancy
        """
        if not self.edges:
            return 1.0
        conflict_count = sum(1 for e in self.edges if e.relation == "CONFLICTS")
        divergent_count = sum(1 for e in self.edges if e.relation == "DIVERGENT")
        # DIVERGENT counts at 50% weight - it's uncertainty, not confirmed conflict
        uncertainty = conflict_count + (divergent_count * 0.5)
        return max(0, 1 - (uncertainty / len(self.edges)))

    # =========================================================================
    # EPISTEMIC VALUE METRICS - Distinguishing Good from Bad Uncertainty
    # =========================================================================

    def topic_diversity(self) -> int:
        """
        Count independent topic clusters.
        A claim is "independent" if it has no edges to other claims.
        High diversity = broad coverage = GOOD.
        """
        # Find connected components via edges
        connected = set()
        for e in self.edges:
            connected.add(e.from_id)
            connected.add(e.to_id)

        # Independent = beliefs with no connections
        beliefs = self.current_beliefs()
        independent = [b for b in beliefs if b.id not in connected]

        # Each connected component + independent nodes = topics
        return len(independent) + (1 if connected else 0)

    def investigation_debt(self) -> dict:
        """
        Unresolved conflicts that need investigation.
        This is NOT a negative metric - it's a TODO list for journalism.
        Each conflict is an OPPORTUNITY to discover truth.

        BUT: We distinguish QUALITY of conflicts:
        - GENUINE: Both sides have corroboration → real investigation needed
        - SUSPICIOUS: One side isolated vs consensus → might be disinfo
        - NOISE: Poorly sourced vs well-established → likely disinfo
        """
        conflicts = self.conflicts()
        items = []

        for c in conflicts:
            node_a = self.nodes.get(c[0])
            node_b = self.nodes.get(c[1])

            if not node_a or not node_b:
                continue

            # Assess quality of each side
            support_a = self.support_count(c[0])
            support_b = self.support_count(c[1])

            # Pure epistemic description - no judgment
            # Just describe the support structure
            if support_a >= 2 and support_b >= 2:
                structure = "CONTESTED"  # Both positions have backing
                note = f"Both claims have multiple sources ({support_a} vs {support_b})"
            elif support_a == 1 and support_b >= 2:
                structure = "ASYMMETRIC"
                note = f"Claim A has 1 source, Claim B has {support_b} sources"
            elif support_b == 1 and support_a >= 2:
                structure = "ASYMMETRIC"
                note = f"Claim A has {support_a} sources, Claim B has 1 source"
            else:
                structure = "UNVERIFIED"
                note = "Both claims have single sources"

            # Priority based on topic, not judgment
            priority = "normal"

            # Check if high-stakes topic
            reason = c[2] if len(c) > 2 else ""
            if any(x in reason.lower() for x in ["death", "casualt", "killed", "injur"]):
                priority = "high"  # Always prioritize life-safety conflicts

            items.append({
                "claim_a": node_a.content,
                "claim_b": node_b.content,
                "source_a": node_a.source,
                "source_b": node_b.source,
                "support_a": support_a,
                "support_b": support_b,
                "structure": structure,  # Descriptive, not judgmental
                "note": note,
                "reason": reason,
                "priority": priority,
            })

        # Sort by total support (most-supported conflicts first)
        items.sort(key=lambda x: -(x["support_a"] + x["support_b"]))

        return {
            "count": len(items),
            "contested": sum(1 for i in items if i["structure"] == "CONTESTED"),
            "asymmetric": sum(1 for i in items if i["structure"] == "ASYMMETRIC"),
            "unverified": sum(1 for i in items if i["structure"] == "UNVERIFIED"),
            "items": items
        }

    def corroboration_depth(self) -> float:
        """
        Average confirmation count per belief.
        High = well-sourced claims = GOOD.
        """
        beliefs = self.current_beliefs()
        if not beliefs:
            return 0.0
        total_support = sum(self.support_count(b.id) for b in beliefs)
        return total_support / len(beliefs)

    def contribution_value(self, node_id: str) -> dict:
        """
        Calculate the epistemic value of a specific claim's contribution.

        All contributions have value - they're just different kinds:
        - NOVEL: Discovery value - expands knowledge scope
        - CONFIRMS: Corroboration value - increases confidence
        - REFINES: Precision value - improves clarity
        - SUPERSEDES: Currency value - updates stale info
        - CONFLICTS: Investigation value - reveals truth gaps

        BUT: Conflict value depends on QUALITY:
        - Surfacing genuine conflicts (both sides corroborated) = HIGH value
        - Surfacing suspicious conflicts (isolated vs consensus) = UNCERTAIN value
        - Creating noise = LOW or NEGATIVE value

        Returns a dict with contribution type and value score.
        """
        edges_from = [e for e in self.edges if e.from_id == node_id]
        edges_to = [e for e in self.edges if e.to_id == node_id]

        # Count what this node does TO others
        confirms = sum(1 for e in edges_from if e.relation == "CONFIRMS")
        supersedes = sum(1 for e in edges_from if e.relation == "SUPERSEDES")
        conflicts = sum(1 for e in edges_from if e.relation == "CONFLICTS")
        refines = sum(1 for e in edges_from if e.relation == "REFINES")

        # Count what others do TO this node (incoming confirmation)
        confirmed_by = sum(1 for e in edges_to if e.relation == "CONFIRMS")

        # THIS node's support level
        my_support = self.support_count(node_id)

        # Determine primary contribution type
        if conflicts > 0:
            # Surfacing conflicts - but value depends on quality
            # Check: am I well-supported, or am I an isolated claim creating noise?
            conflict_edges = [e for e in edges_from if e.relation == "CONFLICTS"]

            genuine_conflicts = 0
            suspicious_conflicts = 0
            for ce in conflict_edges:
                other_support = self.support_count(ce.to_id)
                if my_support >= 2 and other_support >= 2:
                    genuine_conflicts += 1  # Both sides have backup
                elif my_support == 1 and other_support >= 2:
                    suspicious_conflicts += 1  # I'm isolated vs consensus
                else:
                    genuine_conflicts += 0.5  # Unclear

            # Pure description - no judgment about "suspicious" or "disinfo"
            # Just describe the epistemic position
            primary = "CONFLICT"
            value = conflicts * 1.5  # All conflicts have value - they reveal contested territory

            # Describe the epistemic situation factually
            if my_support >= 2:
                note = f"Surfaced {conflicts} conflict(s) (this claim has {my_support} sources)"
            else:
                note = f"Surfaced {conflicts} conflict(s) (this claim has 1 source)"

        elif supersedes > 0:
            primary = "CURRENCY"
            value = supersedes * 1.5
            note = f"Updated {supersedes} outdated claim(s)"
        elif confirms > 0:
            primary = "CORROBORATION"
            value = confirms * 1.0
            note = f"Confirmed {confirms} existing claim(s)"
        elif refines > 0:
            primary = "PRECISION"
            value = refines * 0.8
            note = f"Refined {refines} claim(s) with more detail"
        elif not edges_from:
            primary = "DISCOVERY"
            value = 1.0  # Novel topic
            note = "Introduced new topic or first claim on existing topic"
        else:
            primary = "NOVEL"
            value = 0.5
            note = "Added without relation to existing claims"

        # Bonus for being corroborated by others
        credibility_bonus = confirmed_by * 0.3

        return {
            "node_id": node_id,
            "primary_type": primary,
            "base_value": value,
            "credibility_bonus": credibility_bonus,
            "total_value": value + credibility_bonus,
            "my_support": my_support,
            "note": note,
            "details": {
                "confirms": confirms,
                "supersedes": supersedes,
                "conflicts": conflicts,
                "refines": refines,
                "confirmed_by": confirmed_by
            }
        }

    def epistemic_health(self) -> dict:
        """
        Comprehensive epistemic health report.
        Distinguishes between healthy and unhealthy uncertainty.
        """
        beliefs = self.current_beliefs()
        n_claims = len(self.nodes)
        n_beliefs = len(beliefs)

        # Good metrics
        diversity = self.topic_diversity()
        corroboration = self.corroboration_depth()
        compression = n_claims / max(n_beliefs, 1)

        # Neutral metrics (work to be done)
        debt = self.investigation_debt()

        # Calculate health score (0-100)
        # High compression is good (many claims → few beliefs = consolidation)
        # High corroboration is good (multi-source verification)
        # Low conflict ratio is good (but conflicts aren't BAD, just work to do)
        conflict_ratio = debt["count"] / max(n_beliefs, 1)

        # Score: starts at 100, adjusts for various factors
        # -10 per unresolved conflict (not punishment, just "work remaining")
        # +5 per topic (breadth is good)
        # +10 per average confirmation (corroboration is great)
        score = 100
        score -= debt["count"] * 5  # Work remaining
        score += min(diversity * 3, 20)  # Topic breadth, capped
        score += min((corroboration - 1) * 10, 30)  # Corroboration bonus

        return {
            "score": max(0, min(100, score)),
            "interpretation": self._interpret_health(score),
            "good": {
                "topic_diversity": diversity,
                "corroboration_depth": round(corroboration, 2),
                "compression_ratio": round(compression, 1),
                "multi_source_claims": sum(1 for b in beliefs if self.support_count(b.id) > 1)
            },
            "work_remaining": {
                "investigation_debt": debt["count"],
                "single_source_claims": sum(1 for b in beliefs if self.support_count(b.id) == 1),
                "high_priority_conflicts": len([i for i in debt["items"] if i.get("priority") == "high"])
            }
        }

    def _interpret_health(self, score: int) -> str:
        if score >= 80:
            return "Excellent - well-corroborated, low investigation debt"
        elif score >= 60:
            return "Good - reasonable corroboration, some investigation needed"
        elif score >= 40:
            return "Fair - needs more corroboration or conflict resolution"
        else:
            return "Developing - significant investigation debt or low corroboration"


# =============================================================================
# THE KERNEL - ONE OPERATION
# =============================================================================

class UniversalKernel:
    """
    Universal epistemic engine.

    One operation: update(evidence, state) → state'
    """

    CLASSIFY_PROMPT = """Compare two claims about the same event.

CLAIM A (new):
  Content: {new_content}
  Source: {new_source}
  Time: {new_time} (precision: {new_precision})

CLAIM B (existing):
  Content: {existing_content}
  Source: {existing_source}
  Time: {existing_time} (precision: {existing_precision})

STEP 1: Identify the METRIC each claim measures:
- Death toll (count of dead)
- Injury count
- Fire origin/cause
- Evacuation count
- Response resources (trucks, ambulances)
- Government response
- Personal stories
- Other specific metric

If metrics are DIFFERENT → Return NOVEL immediately.

STEP 2: If SAME metric, choose relationship:

NOVEL - Different metrics entirely (deaths vs injuries, cause vs response)
CONFIRMS - Same metric, same values, agreement
REFINES - Adds detail/precision to same metric (compatible)
SUPERSEDES - Later update with NEW value (REQUIRES "now", "risen to", later timestamp)
CONFLICTS - Same metric, incompatible values, SAME time reference
DIVERGENT - SAME metric, DIFFERENT values, time order UNCLEAR

CRITICAL EXAMPLES:

NOVEL (different metrics - no edge):
- "128 dead" vs "76 injured" → NOVEL (death toll ≠ injury count)
- "fire started on scaffolding" vs "128 dead" → NOVEL (cause ≠ casualties)
- "Xi urged action" vs "36 dead" → NOVEL (political response ≠ casualties)

DIVERGENT (BOTH claims give COUNTS for same metric, different values):
- "36 dead" vs "128 dead" → DIVERGENT (both are death COUNTS)
- "76 injured" vs "50 injured" → DIVERGENT (both are injury COUNTS)
- "200 fire trucks" vs "128 fire trucks" → DIVERGENT (both are resource COUNTS)

REFINES (specific detail about individuals/items within an aggregate):
- "firefighter is among the dead" vs "100+ dead" → REFINES (specific person ⊂ aggregate)
- "elderly residents were trapped" vs "36 dead" → REFINES (demographic detail)
- "128 dead" vs "over 100 dead" → REFINES (precise vs approximate)
- "deadliest fire in decades" vs "100+ dead" → REFINES (characterization + count)

KEY RULE: If one claim identifies a SPECIFIC person/group and the other gives a COUNT,
that's REFINES (the specific is part of the count), NOT DIVERGENT.

Respond with JSON:
{{"relation": "CONFIRMS|REFINES|SUPERSEDES|CONFLICTS|DIVERGENT|NOVEL", "reasoning": "First state the metric of each claim, then explain", "would_resolve": "what additional info would clarify (for DIVERGENT, otherwise empty)"}}"""

    async def update(
        self,
        evidence: Node,
        state: State,
        llm: AsyncOpenAI,
        model: str = "gpt-4o-mini"
    ) -> tuple[State, list[Edge]]:
        """
        Add evidence to state, return updated state + new edges.

        This is THE operation. Everything else derives from state.

        Optimization: Only compare to CURRENT beliefs (not superseded ones).
        This makes it O(n × b) where b << n.
        """
        # Add node
        state.nodes[evidence.id] = evidence

        # Only compare to current beliefs (not superseded)
        current = state.current_beliefs()
        new_edges = []

        for existing in current:
            if existing.id == evidence.id:
                continue

            # Classify relation via LLM
            edge = await self._classify(evidence, existing, llm, model)
            if edge and edge.relation != "NOVEL":
                state.edges.append(edge)
                new_edges.append(edge)

        return state, new_edges

    async def consolidate(
        self,
        state: State,
        llm: AsyncOpenAI,
        model: str = "gpt-4o-mini"
    ) -> tuple[State, list[Edge]]:
        """
        Final pass: compare beliefs that likely share topics.

        Smart filtering: only compare claims that both contain numbers,
        since these are likely about evolving metrics (death toll, injuries, etc.)
        """
        import re
        beliefs = state.current_beliefs()
        new_edges = []

        # Track which pairs already have edges
        existing_pairs = {(e.from_id, e.to_id) for e in state.edges}
        existing_pairs |= {(e.to_id, e.from_id) for e in state.edges}

        # Identify numeric claims (likely evolving metrics)
        def has_number(text):
            return bool(re.search(r'\d+', text))

        numeric_beliefs = [b for b in beliefs if has_number(b.content)]
        print(f"\n  Consolidating: {len(numeric_beliefs)} numeric claims (from {len(beliefs)} beliefs)")

        comparisons = 0
        for i, a in enumerate(numeric_beliefs):
            for b in numeric_beliefs[i+1:]:
                # Skip if already compared
                if (a.id, b.id) in existing_pairs:
                    continue

                comparisons += 1
                # Compare and potentially add edge
                edge = await self._classify(a, b, llm, model)
                if edge and edge.relation in ("SUPERSEDES", "CONFLICTS", "DIVERGENT"):
                    state.edges.append(edge)
                    new_edges.append(edge)
                    print(f"    → {edge.relation}: '{a.content[:30]}...' vs '{b.content[:30]}...'")

        print(f"  Consolidation: {comparisons} comparisons → {len(new_edges)} new edges")
        return state, new_edges

    async def _classify(
        self,
        new: Node,
        existing: Node,
        llm: AsyncOpenAI,
        model: str
    ) -> Optional[Edge]:
        """Classify relationship between two evidence nodes."""

        # Format time for display
        def fmt_time(t, precision):
            if t is None:
                return "unknown"
            return str(t)

        prompt = self.CLASSIFY_PROMPT.format(
            new_content=new.content,
            new_source=new.source,
            new_time=fmt_time(new.time, new.time_precision),
            new_precision=new.time_precision,
            existing_content=existing.content,
            existing_source=existing.source,
            existing_time=fmt_time(existing.time, existing.time_precision),
            existing_precision=existing.time_precision
        )

        try:
            response = await llm.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200
            )

            text = response.choices[0].message.content.strip()

            # Parse JSON response
            if "{" in text:
                json_str = text[text.find("{"):text.rfind("}")+1]
                data = json.loads(json_str)
                relation = data.get("relation", "NOVEL").upper()
                reasoning = data.get("reasoning", "")
                would_resolve = data.get("would_resolve", "")

                if relation in ("CONFIRMS", "REFINES", "SUPERSEDES", "CONFLICTS", "DIVERGENT"):
                    return Edge(
                        from_id=new.id,
                        to_id=existing.id,
                        relation=relation,
                        reasoning=reasoning,
                        would_resolve=would_resolve
                    )

            return None

        except Exception as e:
            print(f"Classification error: {e}")
            return None

    # =========================================================================
    # BATCH MODE - Production-style optimizations
    # =========================================================================

    async def generate_embeddings(
        self,
        nodes: List[Node],
        llm: AsyncOpenAI
    ) -> None:
        """Generate embeddings for nodes that don't have them."""
        needs_embedding = [n for n in nodes if n.embedding is None]
        if not needs_embedding:
            return

        texts = [n.content for n in needs_embedding]
        response = await llm.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        for node, data in zip(needs_embedding, response.data):
            node.embedding = data.embedding

    def cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a, b = np.array(v1), np.array(v2)
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def find_similar_pairs(
        self,
        nodes: List[Node],
        threshold: float = 0.4
    ) -> List[Tuple[Node, Node, float]]:
        """Find pairs with similarity above threshold."""
        pairs = []
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i+1:]:
                if n1.embedding and n2.embedding:
                    sim = self.cosine_similarity(n1.embedding, n2.embedding)
                    if sim > threshold:
                        pairs.append((n1, n2, sim))
        return pairs

    BATCH_CLASSIFY_PROMPT = """For each pair, identify WHAT QUESTION each claim answers.

Two claims relate only if they answer the SAME question with different values.
Different questions = NOVEL (no epistemic relation).

For each pair:
1. q1 = question claim1 answers (e.g. "count of X", "cause of Y", "time of Z")
2. q2 = question claim2 answers
3. If q1 ≠ q2 → NOVEL
4. If q1 = q2:
   - Similar values → CONFIRMS
   - One more specific → REFINES
   - Temporal update → SUPERSEDES
   - Different values, no timing → DIVERGENT

Example: "50 apples picked" vs "50 oranges shipped"
q1 = "count of apples picked", q2 = "count of oranges shipped"
q1 ≠ q2 → NOVEL (the number 50 is coincidence)

PAIRS:
{pairs_text}

Return JSON: {{"results": [{{"pair": 1, "q1": "...", "q2": "...", "same_q": true/false, "relation": "NOVEL|CONFIRMS|REFINES|SUPERSEDES|CONFLICTS|DIVERGENT"}}]}}"""

    async def _batch_classify(
        self,
        pairs: List[Tuple[Node, Node, float]],
        llm: AsyncOpenAI,
        model: str = "gpt-4o-mini",
        batch_size: int = 20
    ) -> List[Optional[Edge]]:
        """Classify multiple pairs in one LLM call."""
        if not pairs:
            return []

        all_edges = []

        # Process in batches
        for batch_start in range(0, len(pairs), batch_size):
            batch = pairs[batch_start:batch_start + batch_size]

            # Build pairs text
            pairs_text = []
            for i, (n1, n2, sim) in enumerate(batch):
                pairs_text.append(f"""
Pair {i+1} (sim={sim:.2f}):
  A: {n1.content[:150]}
  B: {n2.content[:150]}""")

            prompt = self.BATCH_CLASSIFY_PROMPT.format(pairs_text="\n".join(pairs_text))

            try:
                response = await llm.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )

                text = response.choices[0].message.content.strip()
                data = json.loads(text)
                results = data.get("results", data.get("pairs", []))
                # Handle case where LLM returns array directly
                if isinstance(data, list):
                    results = data

                # Map results back to edges
                batch_edges = [None] * len(batch)
                for item in results:
                    idx = item.get("pair", 0) - 1
                    if 0 <= idx < len(batch):
                        relation = item.get("relation", "NOVEL").upper()
                        same_q = item.get("same_q", item.get("same_metric", True))
                        q1 = item.get("q1", item.get("type1", "")).lower()
                        q2 = item.get("q2", item.get("type2", "")).lower()

                        # VALIDATION: If LLM says questions differ, MUST be NOVEL
                        if not same_q:
                            continue  # Different questions = NOVEL (no edge)

                        # HARD VALIDATION: If q1/q2 strings are too different, reject
                        # This catches cases where LLM says same_q=true but questions differ
                        if q1 and q2:
                            from rapidfuzz import fuzz
                            q_sim = fuzz.token_sort_ratio(q1, q2)
                            if q_sim < 70:  # Questions must be >70% similar
                                continue  # Questions too different = NOVEL

                        if relation in ("CONFIRMS", "REFINES", "SUPERSEDES", "CONFLICTS", "DIVERGENT"):
                            n1, n2, _ = batch[idx]
                            batch_edges[idx] = Edge(
                                from_id=n1.id,
                                to_id=n2.id,
                                relation=relation,
                                reasoning=f"[{q1}] vs [{q2}]",
                                would_resolve=item.get("would_resolve", "")
                            )

                all_edges.extend(batch_edges)

            except Exception as e:
                print(f"Batch classification error: {e}")
                all_edges.extend([None] * len(batch))

        return all_edges

    async def update_batch(
        self,
        nodes: List[Node],
        state: State,
        llm: AsyncOpenAI,
        model: str = "gpt-4o-mini",
        sim_threshold: float = 0.4,
        auto_confirm_threshold: float = 0.85
    ) -> Tuple[State, List[Edge]]:
        """
        Batch update: add all nodes and classify relationships efficiently.

        Uses production-style optimizations:
        1. Generate embeddings for all nodes at once
        2. Pre-filter pairs by similarity (only compare sim > threshold)
        3. Auto-classify very high similarity as CONFIRMS
        4. Batch LLM calls (20 pairs per call)
        """
        if not nodes:
            return state, []

        # Add all nodes to state
        for node in nodes:
            state.nodes[node.id] = node

        # Generate embeddings
        all_nodes = list(state.nodes.values())
        print(f"  📊 Generating embeddings for {len(all_nodes)} nodes...")
        await self.generate_embeddings(all_nodes, llm)

        # Find similar pairs
        print(f"  🔍 Finding similar pairs (threshold={sim_threshold})...")
        similar_pairs = self.find_similar_pairs(all_nodes, sim_threshold)
        print(f"  📋 Found {len(similar_pairs)} pairs to classify")

        # Filter out already-edged pairs
        existing_pairs = {(e.from_id, e.to_id) for e in state.edges}
        existing_pairs |= {(e.to_id, e.from_id) for e in state.edges}
        similar_pairs = [(n1, n2, sim) for n1, n2, sim in similar_pairs
                        if (n1.id, n2.id) not in existing_pairs]

        new_edges = []

        # Auto-classify very high similarity
        auto_confirmed = []
        needs_llm = []
        for n1, n2, sim in similar_pairs:
            if sim > auto_confirm_threshold:
                edge = Edge(from_id=n1.id, to_id=n2.id, relation="CONFIRMS",
                           reasoning=f"Auto-confirmed: {sim:.2f} similarity")
                auto_confirmed.append(edge)
            else:
                needs_llm.append((n1, n2, sim))

        if auto_confirmed:
            print(f"  ✓ Auto-confirmed {len(auto_confirmed)} high-similarity pairs")
            for edge in auto_confirmed:
                state.edges.append(edge)
                new_edges.append(edge)

        # Batch classify remaining pairs
        if needs_llm:
            print(f"  🤖 LLM classifying {len(needs_llm)} pairs in batches of 20...")
            llm_edges = await self._batch_classify(needs_llm, llm, model)
            for edge in llm_edges:
                if edge:
                    state.edges.append(edge)
                    new_edges.append(edge)
            print(f"  ✅ Created {sum(1 for e in llm_edges if e)} edges from LLM")

        return state, new_edges


# =============================================================================
# TEST
# =============================================================================

async def test_kernel(num_claims: int = 20, resume: bool = False):
    """Test with real claims from HK Fire event.

    Args:
        num_claims: Total claims to process (including already processed if resuming)
        resume: If True, load existing topology and continue from where we left off
    """
    import os

    llm = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    kernel = UniversalKernel()

    # Resume from existing topology or start fresh
    existing_ids = set()
    if resume:
        try:
            with open("/app/test_eu/topology_state.json") as f:
                saved = json.load(f)
            state = State.from_dict(saved)
            existing_ids = set(state.nodes.keys())
            print(f"[RESUME] Loaded {len(existing_ids)} existing claims")
        except FileNotFoundError:
            print("[RESUME] No existing topology, starting fresh")
            state = State()
    else:
        state = State()

    # Load real claims
    try:
        with open("/tmp/hk_event.json") as f:
            data = json.load(f)
        claims_data = data.get("claims", [])[:num_claims]
    except FileNotFoundError:
        # Fallback sample claims
        claims_data = [
            {"id": "cl_001", "text": "At least 11 people killed in Hong Kong high-rise fire", "source_name": "BBC"},
            {"id": "cl_002", "text": "Fire breaks out in Wang Fuk Court, Tai Po district", "source_name": "SCMP"},
            {"id": "cl_003", "text": "Death toll rises to 17 as rescue efforts continue", "source_name": "Reuters"},
            {"id": "cl_004", "text": "11 confirmed dead, dozens injured", "source_name": "Al Jazeera"},
            {"id": "cl_005", "text": "36 confirmed dead in Hong Kong apartment fire", "source_name": "AP"},
            {"id": "cl_006", "text": "128 confirmed dead as DNA identification proceeds", "source_name": "HK Gov"},
            {"id": "cl_007", "text": "Fire originated on 14th floor", "source_name": "Police"},
            {"id": "cl_008", "text": "Fire believed to have started on 15th floor", "source_name": "Fire Dept"},
            {"id": "cl_009", "text": "76 people injured in the blaze", "source_name": "Hospital"},
            {"id": "cl_010", "text": "Over 50 injured, many in critical condition", "source_name": "Reuters"},
        ][:num_claims]

    # Filter out already processed claims if resuming
    if resume:
        new_claims = [c for c in claims_data if c.get("id", f"cl_{claims_data.index(c):03d}") not in existing_ids]
        print(f"[RESUME] {len(new_claims)} new claims to process")
        claims_data = new_claims

    total_claims = len(existing_ids) + len(claims_data)
    print(f"Universal Kernel Test - {total_claims} total claims ({len(existing_ids)} existing + {len(claims_data)} new)")
    print("=" * 70)
    print("Gaps and conflicts are OPPORTUNITIES to discover truth")
    print("=" * 70)

    import time
    timings = []
    history = []  # Track epistemic evolution

    # t=0: No claims yet = no knowledge = max entropy, no coherence
    prev_entropy = None  # Undefined until first claim
    prev_coherence = None  # Undefined until first claim
    prev_beliefs = 0

    for i, c in enumerate(claims_data):
        cid = c.get("id", f"cl_{i:03d}")
        text = c.get("text", "")
        source = c.get("source_name", c.get("source", "unknown"))

        # Extract temporal metadata from claim
        # Be honest: if the extraction pipeline didn't tell us the precision, it's unknown
        event_time = None
        time_precision = c.get("time_precision", "unknown")  # Must come from extraction
        if c.get("event_time"):
            try:
                from dateutil import parser as date_parser
                event_time = date_parser.parse(c["event_time"])
            except:
                pass

        confidence = c.get("confidence", 0.5)
        modality = c.get("modality", "unknown")

        node = Node(
            id=cid,
            content=text,
            source=source,
            time=event_time,
            time_precision=time_precision,
            confidence=confidence,
            modality=modality
        )

        t0 = time.time()
        state, new_edges = await kernel.update(node, state, llm)
        elapsed_ms = (time.time() - t0) * 1000
        timings.append(elapsed_ms)

        # Show progress with symbols, timing, and epistemic metrics
        symbols = {"CONFIRMS": "=", "REFINES": "↑", "SUPERSEDES": "→", "CONFLICTS": "!", "DIVERGENT": "?", "NOVEL": "+"}
        rel_symbols = [symbols.get(e.relation, "?") for e in new_edges] or ["+"]

        # Epistemic metrics after this update
        n_beliefs = len(state.current_beliefs())
        entropy = state.entropy()
        coherence = state.coherence()

        # Calculate deltas (first claim is baseline)
        if prev_entropy is None:
            d_entropy = 0  # First claim establishes baseline
            d_coherence = 0
        else:
            d_entropy = entropy - prev_entropy
            d_coherence = coherence - prev_coherence
        d_beliefs = n_beliefs - prev_beliefs

        # Track history for analysis
        history.append({
            "claim": i + 1,
            "relations": [e.relation for e in new_edges],
            "beliefs": n_beliefs,
            "entropy": entropy,
            "coherence": coherence,
            "d_entropy": d_entropy,
            "d_coherence": d_coherence,
        })

        # Show delta indicators (first claim shows * for baseline)
        if i == 0:
            h_arrow = "*"  # Baseline established
            c_arrow = "*"
        else:
            h_arrow = "↓" if d_entropy < -0.01 else ("↑" if d_entropy > 0.01 else "·")
            c_arrow = "↑" if d_coherence > 0.01 else ("↓" if d_coherence < -0.01 else "·")

        print(f"[{i+1:3d}] {''.join(rel_symbols):6s} {elapsed_ms:4.0f}ms | "
              f"b={n_beliefs:2d} H={entropy:.2f}{h_arrow} C={coherence:.0%}{c_arrow}")

        prev_entropy = entropy
        prev_coherence = coherence
        prev_beliefs = n_beliefs

    # Consolidation pass - catch missed supersessions/conflicts
    print("\n" + "-" * 70)
    state, consolidation_edges = await kernel.consolidate(state, llm)

    print("\n" + "=" * 70)
    print("TOPOLOGY RESULTS")
    print("=" * 70)

    # Current beliefs (sorted by support)
    beliefs = sorted(state.current_beliefs(),
                     key=lambda b: state.support_count(b.id), reverse=True)
    print(f"\n[CURRENT BELIEFS] ({len(beliefs)} from {len(state.nodes)} claims)")
    print("-" * 70)
    for b in beliefs[:15]:  # Top 15
        s = state.support_count(b.id)
        p = state.plausibility(b.id)
        src = b.source[:12]
        print(f"  x{s} [{b.id}] ({src}) {b.content[:55]}...")

    if len(beliefs) > 15:
        print(f"  ... and {len(beliefs) - 15} more")

    # OPPORTUNITIES (the valuable part!)
    print(f"\n[OPPORTUNITIES FOR TRUTH] - These need investigation")
    print("-" * 70)

    opps = state.opportunities()

    if opps["conflicts"]:
        print(f"\n  CONFLICTS ({len(opps['conflicts'])}) - Contradictory claims:")
        for c in opps["conflicts"][:5]:
            print(f"    ! \"{c['claim_a'][:40]}...\" ({c['source_a']})")
            print(f"      vs \"{c['claim_b'][:40]}...\" ({c['source_b']})")
            print(f"      -> {c['action']}")
            print()

    if opps["low_confidence"]:
        print(f"\n  LOW CONFIDENCE ({len(opps['low_confidence'])}) - Single source, needs corroboration:")
        for lc in opps["low_confidence"][:5]:
            print(f"    ? \"{lc['claim'][:50]}...\" ({lc['source']})")
            print(f"      -> {lc['action']}")

    # Metrics
    print(f"\n[METRICS]")
    print("-" * 70)
    print(f"  Claims processed: {len(state.nodes)}")
    print(f"  Current beliefs:  {len(beliefs)}")
    print(f"  Compression:      {len(state.nodes)/max(len(beliefs),1):.1f}x")
    print(f"  Edges:            {len(state.edges)}")
    print(f"  Entropy:          {state.entropy():.2f}")
    print(f"  Coherence:        {state.coherence():.2%}")

    # Timing stats
    if timings:
        avg_ms = sum(timings) / len(timings)
        total_s = sum(timings) / 1000
        print(f"\n  Timing:")
        print(f"    Total:          {total_s:.1f}s")
        print(f"    Avg per claim:  {avg_ms:.0f}ms")
        print(f"    Min/Max:        {min(timings):.0f}ms / {max(timings):.0f}ms")

    # NEW: Epistemic Health Report (nuanced metrics)
    print(f"\n[EPISTEMIC HEALTH]")
    print("-" * 70)
    health = state.epistemic_health()
    print(f"  Overall Score: {health['score']}/100 - {health['interpretation']}")
    print(f"\n  Strengths:")
    print(f"    Topic diversity:      {health['good']['topic_diversity']} independent topics")
    print(f"    Corroboration depth:  {health['good']['corroboration_depth']:.2f} avg confirmations/belief")
    print(f"    Compression:          {health['good']['compression_ratio']}x (claims → beliefs)")
    print(f"    Multi-source claims:  {health['good']['multi_source_claims']}")
    # Get detailed conflict structure (descriptive, not judgmental)
    debt = state.investigation_debt()
    print(f"\n  Epistemic Landscape:")
    print(f"    Conflicts:            {debt['count']}")
    if debt['count'] > 0:
        print(f"      - CONTESTED:        {debt['contested']} (both sides have multiple sources)")
        print(f"      - ASYMMETRIC:       {debt['asymmetric']} (one side has more sources)")
        print(f"      - UNVERIFIED:       {debt['unverified']} (both sides single-source)")
    print(f"    Single-source claims: {health['work_remaining']['single_source_claims']}")

    # Contributor Value Analysis
    if history:
        print(f"\n[CONTRIBUTOR VALUE ANALYSIS]")
        print("-" * 70)
        print("  All contributions have value - different kinds of epistemic work:\n")

        # Calculate contribution value for each claim
        # Use the actual node IDs from claims_data, not assumed format
        contrib_values = []
        node_ids = list(state.nodes.keys())
        for i, h in enumerate(history):
            if i < len(node_ids):
                node_id = node_ids[i]
                val = state.contribution_value(node_id)
                val["claim_idx"] = h["claim"]
                val["node_id"] = node_id
                contrib_values.append(val)

        # Sort by total value
        contrib_values.sort(key=lambda x: -x["total_value"])

        # Show top contributors by value type
        by_type = {}
        for v in contrib_values:
            t = v["primary_type"]
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(v)

        # Display contribution types (purely descriptive)
        value_order = ["CONFLICT", "CURRENCY", "CORROBORATION", "PRECISION", "DISCOVERY"]
        for vtype in value_order:
            if vtype in by_type:
                items = by_type[vtype][:2]  # Top 2 per type
                emoji = {
                    "CONFLICT": "⚡",
                    "CURRENCY": "↻",
                    "CORROBORATION": "✓",
                    "PRECISION": "🎯",
                    "DISCOVERY": "+"
                }.get(vtype, "·")
                desc = {
                    "CONFLICT": "Revealed contested territory",
                    "CURRENCY": "Updated outdated information",
                    "CORROBORATION": "Confirmed existing claims",
                    "PRECISION": "Refined claims with detail",
                    "DISCOVERY": "Introduced new topics"
                }.get(vtype, "")
                print(f"  {emoji} {vtype} - {desc}")
                for v in items:
                    cid = v["claim_idx"]
                    support = v.get("my_support", 1)
                    print(f"      Claim {cid:2d}: {v['note']} (support: x{support})")

        # Summary: All contributors matter
        print(f"\n  Key Insight: Someone who surfaces a CONFLICT isn't 'hurting' the system.")
        print(f"  They're doing investigative journalism - revealing where truth is contested.")

    # Edge breakdown
    print(f"\n  Relations:")
    for rel in ["CONFIRMS", "REFINES", "SUPERSEDES", "CONFLICTS", "DIVERGENT"]:
        count = sum(1 for e in state.edges if e.relation == rel)
        pct = count / max(len(state.edges), 1) * 100
        print(f"    {rel:12s}: {count:3d} ({pct:5.1f}%)")

    # Show DIVERGENT pairs - these need more information to resolve
    divergent = state.divergent_pairs()
    if divergent:
        print(f"\n[DIVERGENT PAIRS] ({len(divergent)}) - Temporal relationship unclear")
        print("-" * 70)
        print("  These could be updates OR conflicts - more information would clarify:\n")
        for d in divergent[:5]:
            from_node = state.nodes.get(d[0])
            to_node = state.nodes.get(d[1])
            if from_node and to_node:
                print(f"  ? \"{from_node.content[:45]}...\"")
                print(f"    vs \"{to_node.content[:45]}...\"")
                if d[3]:  # would_resolve
                    print(f"    → Would resolve: {d[3]}")
                print()

    # Save topology
    with open("/app/test_eu/topology_state.json", "w") as f:
        json.dump(state.to_dict(), f, indent=2, default=str)
    print(f"\n  Topology saved to /app/test_eu/topology_state.json")

    return state


async def test_kernel_batch(num_claims: int = 50):
    """
    Test with batch mode - uses production-style optimizations.

    Much faster than sequential mode:
    - Embedding pre-filtering reduces comparisons
    - Batch LLM calls (20 pairs per call)
    - Auto-confirm high similarity pairs
    """
    import os
    import time

    llm = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    kernel = UniversalKernel()
    state = State()

    # Load claims
    try:
        with open("/tmp/hk_event.json") as f:
            data = json.load(f)
        claims_data = data.get("claims", [])[:num_claims]
    except FileNotFoundError:
        claims_data = [
            {"id": "cl_001", "text": "At least 11 people killed in Hong Kong high-rise fire", "source_name": "BBC"},
            {"id": "cl_002", "text": "Fire breaks out in Wang Fuk Court, Tai Po district", "source_name": "SCMP"},
            {"id": "cl_003", "text": "Death toll rises to 17 as rescue efforts continue", "source_name": "Reuters"},
            {"id": "cl_004", "text": "11 confirmed dead, dozens injured", "source_name": "Al Jazeera"},
            {"id": "cl_005", "text": "36 confirmed dead in Hong Kong apartment fire", "source_name": "AP"},
        ][:num_claims]

    print(f"Universal Kernel BATCH Test - {len(claims_data)} claims")
    print("=" * 70)
    print("Using production-style optimizations: embeddings + batch LLM")
    print("=" * 70)

    # Convert to nodes
    nodes = []
    for i, c in enumerate(claims_data):
        cid = c.get("id", f"cl_{i:03d}")
        node = Node(
            id=cid,
            content=c.get("text", ""),
            source=c.get("source_name", c.get("source", "unknown")),
            time=None,
            time_precision=c.get("time_precision", "unknown"),
            confidence=c.get("confidence", 0.5),
            modality=c.get("modality", "unknown")
        )
        nodes.append(node)

    # Run batch update
    t0 = time.time()
    state, edges = await kernel.update_batch(nodes, state, llm)
    elapsed = time.time() - t0

    print(f"\n" + "=" * 70)
    print("BATCH RESULTS")
    print("=" * 70)

    # Metrics
    beliefs = state.current_beliefs()
    print(f"\n[METRICS]")
    print("-" * 70)
    print(f"  Claims processed: {len(state.nodes)}")
    print(f"  Current beliefs:  {len(beliefs)}")
    print(f"  Edges created:    {len(edges)}")
    print(f"  Total edges:      {len(state.edges)}")
    print(f"  Entropy:          {state.entropy():.2f}")
    print(f"  Coherence:        {state.coherence():.2%}")
    print(f"\n  Time: {elapsed:.1f}s ({elapsed/len(nodes)*1000:.0f}ms per claim)")

    # Edge breakdown
    print(f"\n  Relations:")
    for rel in ["CONFIRMS", "REFINES", "SUPERSEDES", "CONFLICTS", "DIVERGENT"]:
        count = sum(1 for e in state.edges if e.relation == rel)
        pct = count / max(len(state.edges), 1) * 100
        print(f"    {rel:12s}: {count:3d} ({pct:5.1f}%)")

    # Conflicts
    conflicts = state.conflicts()
    if conflicts:
        print(f"\n[CONFLICTS] ({len(conflicts)})")
        for c in conflicts[:3]:
            n1 = state.nodes.get(c[0])
            n2 = state.nodes.get(c[1])
            if n1 and n2:
                print(f"  ! \"{n1.content[:40]}...\" vs \"{n2.content[:40]}...\"")

    # Divergent
    divergent = state.divergent_pairs()
    if divergent:
        print(f"\n[DIVERGENT] ({len(divergent)})")
        for d in divergent[:3]:
            n1 = state.nodes.get(d[0])
            n2 = state.nodes.get(d[1])
            if n1 and n2:
                print(f"  ? \"{n1.content[:40]}...\" vs \"{n2.content[:40]}...\"")

    # Save
    with open("/app/test_eu/topology_batch.json", "w") as f:
        json.dump(state.to_dict(), f, indent=2, default=str)
    print(f"\n  Topology saved to /app/test_eu/topology_batch.json")

    return state


if __name__ == "__main__":
    import asyncio
    import sys

    # Parse args: python universal_kernel.py [num_claims] [--resume] [--batch]
    args = sys.argv[1:]
    resume = "--resume" in args
    batch = "--batch" in args
    args = [a for a in args if not a.startswith("--")]
    num = int(args[0]) if args else 50

    if batch:
        asyncio.run(test_kernel_batch(num))
    else:
        asyncio.run(test_kernel(num, resume=resume))
