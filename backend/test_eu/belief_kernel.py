"""
Belief Kernel - The Minimal Epistemic Engine
=============================================

Core idea: Put domain knowledge in the prompt, not in code.

NO topics. NO clustering. NO thresholds. NO adhoc rules.
Just: what do we believe, and how does new info change it?

The bet: LLM reasoning > engineered heuristics
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from openai import AsyncOpenAI


@dataclass
class Belief:
    """A single belief with provenance"""
    text: str
    sources: List[str] = field(default_factory=list)
    confidence: float = 1.0  # Increases with corroboration
    supersedes: Optional[str] = None  # What this replaced
    updated_count: int = 0  # How many times this was updated


@dataclass
class BeliefKernel:
    """
    The simplest epistemic engine.

    Maintains beliefs in natural language.
    One LLM call per claim to determine relationship.
    Domain knowledge lives in the prompt, not code.
    """
    beliefs: List[Belief] = field(default_factory=list)
    conflicts: List[Dict] = field(default_factory=list)
    history: List[Dict] = field(default_factory=list)

    # Stats
    llm_calls: int = 0

    def _format_beliefs(self) -> str:
        """Format beliefs for prompt - limit to most relevant"""
        if not self.beliefs:
            return "(no beliefs yet)"

        # Show all beliefs (could limit for very long lists)
        lines = []
        for i, b in enumerate(self.beliefs):
            src_count = len(b.sources)
            lines.append(f"[{i}] {b.text} ({src_count} sources)")
        return "\n".join(lines)

    async def process(self, claim: str, source: str, llm) -> Dict:
        """
        Process one claim. One LLM call. That's it.

        Returns the relationship and updates internal state.
        """
        prompt = f"""You are reasoning about news claims.

CURRENT BELIEFS:
{self._format_beliefs()}

NEW CLAIM: "{claim}"
SOURCE: {source}

How does this new claim relate to our current beliefs?

COMPATIBLE - New fact that can be true alongside existing beliefs → ADD it
REDUNDANT - Already expressed by an existing belief → SKIP (note which)
REFINES - More specific version of existing belief → REPLACE with more specific
SUPERSEDES - Updated information that replaces old → REPLACE
CONFLICTS - Cannot be true if an existing belief is true → FLAG for review

KEY INSIGHT for news: Death tolls, injury counts, and missing persons counts
typically INCREASE over time as more information comes in. A higher number
SUPERSEDES a lower number - it's an update, not a conflict.

Examples:
- "36 dead" then "128 dead" → SUPERSEDES (count updated)
- "fire in Hong Kong" then "fire in Tai Po, Hong Kong" → REFINES (more specific)
- "100 trucks" then "200 trucks" from different sources → CONFLICTS (same event, different facts)

Think step by step:
1. What is this claim asserting?
2. Does any existing belief cover the same fact?
3. If yes, is the new claim more specific, an update, or contradictory?

Return JSON:
{{
  "relation": "COMPATIBLE|REDUNDANT|REFINES|SUPERSEDES|CONFLICTS",
  "affected_belief_index": <number or null if COMPATIBLE>,
  "reasoning": "<one sentence explaining the relationship>",
  "normalized_claim": "<the claim in clear, standalone form>"
}}"""

        try:
            self.llm_calls += 1
            response = await llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            result = json.loads(response.choices[0].message.content)
        except Exception as e:
            result = {
                "relation": "COMPATIBLE",
                "affected_belief_index": None,
                "reasoning": f"Error: {e}",
                "normalized_claim": claim
            }

        # Update beliefs based on relationship
        relation = result.get("relation", "COMPATIBLE")
        affected_idx = result.get("affected_belief_index")
        normalized = result.get("normalized_claim", claim)

        if relation == "COMPATIBLE":
            # New fact - add it
            self.beliefs.append(Belief(text=normalized, sources=[source]))

        elif relation == "REDUNDANT":
            # Already known - just add source
            if affected_idx is not None and 0 <= affected_idx < len(self.beliefs):
                self.beliefs[affected_idx].sources.append(source)

        elif relation == "REFINES":
            # More specific - replace
            if affected_idx is not None and 0 <= affected_idx < len(self.beliefs):
                old_text = self.beliefs[affected_idx].text
                old_sources = self.beliefs[affected_idx].sources
                self.beliefs[affected_idx] = Belief(
                    text=normalized,
                    sources=old_sources + [source],
                    supersedes=old_text
                )

        elif relation == "SUPERSEDES":
            # Update - replace with note
            if affected_idx is not None and 0 <= affected_idx < len(self.beliefs):
                old_text = self.beliefs[affected_idx].text
                old_sources = self.beliefs[affected_idx].sources
                self.beliefs[affected_idx] = Belief(
                    text=normalized,
                    sources=old_sources + [source],
                    supersedes=old_text
                )

        elif relation == "CONFLICTS":
            # Cannot resolve - flag it
            self.conflicts.append({
                "new_claim": normalized,
                "source": source,
                "conflicts_with": affected_idx,
                "existing_belief": self.beliefs[affected_idx].text if affected_idx is not None and affected_idx < len(self.beliefs) else None,
                "reasoning": result.get("reasoning", "")
            })

        # Log
        self.history.append({
            "claim": claim,
            "source": source,
            "result": result
        })

        return result

    def summary(self) -> Dict:
        """Get current state summary"""
        relations = {}
        for h in self.history:
            rel = h["result"].get("relation", "UNKNOWN")
            relations[rel] = relations.get(rel, 0) + 1

        return {
            "beliefs": len(self.beliefs),
            "conflicts": len(self.conflicts),
            "relations": relations,
            "current_beliefs": [b.text for b in self.beliefs],
            "unresolved_conflicts": self.conflicts
        }

    def compute_coherence(self) -> float:
        """
        Coherence = ratio of corroborating claims to total claims
        High coherence = sources agree with each other
        Low coherence = many new facts, few corroborations
        """
        if not self.history:
            return 0.0

        total = len(self.history)
        # Claims that agree with existing beliefs
        agreements = sum(1 for h in self.history
                        if h["result"].get("relation") in ("REDUNDANT", "SUPERSEDES", "REFINES"))
        # Conflicts heavily penalize coherence
        conflicts = sum(1 for h in self.history
                       if h["result"].get("relation") == "CONFLICTS")

        # Coherence = (agreements - conflicts) / total, scaled to 0-1
        raw = (agreements - conflicts * 2) / max(total, 1)
        return max(0.0, min(1.0, 0.5 + raw * 0.5))

    def compute_entropy(self) -> float:
        """
        Entropy = uncertainty based on source diversity
        Low entropy = most beliefs corroborated by multiple sources
        High entropy = most beliefs from single source, or conflicts exist
        """
        if not self.beliefs:
            return 1.0

        # Count beliefs by source count
        single_source = sum(1 for b in self.beliefs if len(b.sources) == 1)
        multi_source = sum(1 for b in self.beliefs if len(b.sources) >= 2)
        high_conf = sum(1 for b in self.beliefs if len(b.sources) >= 3)

        total_beliefs = len(self.beliefs)
        conflicts = len(self.conflicts)

        # Base entropy from source distribution
        # More multi-source beliefs = lower entropy
        if total_beliefs == 0:
            return 1.0

        source_score = (high_conf * 2 + multi_source) / (total_beliefs * 2)
        conflict_penalty = min(conflicts * 0.15, 0.5)  # Max 0.5 penalty

        entropy = 0.8 - source_score * 0.7 + conflict_penalty
        return max(0.05, min(0.95, entropy))

    async def generate_prose(self, llm) -> str:
        """
        Generate narrative prose from current beliefs.
        The kernel's prose is grounded in what we actually believe.
        """
        if not self.beliefs:
            return "Awaiting information..."

        # Group beliefs by confidence (source count)
        high_conf = [b for b in self.beliefs if len(b.sources) >= 3]
        medium_conf = [b for b in self.beliefs if len(b.sources) == 2]
        low_conf = [b for b in self.beliefs if len(b.sources) == 1]

        # Format for LLM
        beliefs_text = ""
        if high_conf:
            beliefs_text += "CONFIRMED (3+ sources):\n"
            beliefs_text += "\n".join(f"- {b.text}" for b in high_conf[:5])
            beliefs_text += "\n\n"
        if medium_conf:
            beliefs_text += "CORROBORATED (2 sources):\n"
            beliefs_text += "\n".join(f"- {b.text}" for b in medium_conf[:5])
            beliefs_text += "\n\n"
        if low_conf[:5]:
            beliefs_text += "REPORTED (1 source):\n"
            beliefs_text += "\n".join(f"- {b.text}" for b in low_conf[:5])

        # Note conflicts
        if self.conflicts:
            beliefs_text += "\n\nUNRESOLVED CONFLICTS:\n"
            for c in self.conflicts[:3]:
                beliefs_text += f"- {c['new_claim'][:80]}...\n"

        prompt = f"""Write a news summary based ONLY on these verified beliefs. Do NOT invent any details.

{beliefs_text}

STRICT RULES:
- ONLY include facts that appear in the beliefs above
- DO NOT add causes, locations, or details not explicitly stated
- DO NOT speculate about investigations, responses, or future actions
- For CONFIRMED: state as fact ("At least X...")
- For CORROBORATED: use hedging ("According to sources...")
- For REPORTED: note uncertainty ("One source reports...")
- If beliefs are sparse, keep the summary SHORT
- Maximum 100 words

FORBIDDEN: Do not mention electrical issues, investigations, safety regulations,
or any other details unless they appear word-for-word in the beliefs above.

Return ONLY the prose summary."""

        try:
            self.llm_calls += 1
            response = await llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating narrative: {e}"

    def to_api_response(self) -> Dict:
        """Format for API response (compatible with frontend)"""
        return {
            "id": "kernel",
            "title": self._generate_title(),
            "phase": self._get_phase(),
            "entropy": self.compute_entropy(),
            "coherence": self.compute_coherence(),
            "claim_count": len(self.history),
            "belief_count": len(self.beliefs),
            "beliefs": [
                {
                    "text": b.text,
                    "sources": b.sources,
                    "source_count": len(b.sources),
                    "supersedes": b.supersedes
                }
                for b in self.beliefs
            ],
            "conflicts": self.conflicts,
            "relations": self.summary()["relations"],
            "llm_calls": self.llm_calls
        }

    def _generate_title(self) -> str:
        """Generate title from beliefs"""
        if not self.beliefs:
            return "Developing Story"

        # Look for key facts
        for b in self.beliefs:
            text = b.text.lower()
            if "killed" in text or "dead" in text or "died" in text:
                # Extract number and location if possible
                return b.text[:60] + "..."

        # Fallback to first belief
        return self.beliefs[0].text[:50] + "..."

    def _get_phase(self) -> str:
        """Determine phase based on state"""
        if not self.beliefs:
            return "emerging"

        coherence = self.compute_coherence()
        entropy = self.compute_entropy()

        if len(self.conflicts) > 3:
            return "contested"
        elif coherence > 0.7 and entropy < 0.3:
            return "stable"
        elif coherence > 0.5:
            return "converging"
        else:
            return "emerging"


async def test_kernel(num_claims: int = None, verbose: bool = True):
    """Test the kernel on HK Fire claims"""
    print("=" * 60)
    print("BELIEF KERNEL TEST")
    print("=" * 60)
    print("\nCore idea: Domain knowledge in prompt, not code.")

    # Load claims
    with open('/tmp/hk_event.json', 'r') as f:
        data = json.load(f)

    all_claims = data.get('claims', [])
    claims = all_claims[:num_claims] if num_claims else all_claims
    print(f"\nProcessing {len(claims)} claims (of {len(all_claims)} total)...")

    # Initialize
    llm = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    kernel = BeliefKernel()

    # Process each claim
    for i, claim in enumerate(claims):
        text = claim['text']
        source = claim.get('source_name', 'unknown')

        result = await kernel.process(text, source, llm)
        rel = result.get('relation', '?')

        # Progress indicator
        symbol = {
            'COMPATIBLE': '+',
            'REDUNDANT': '=',
            'REFINES': '↑',
            'SUPERSEDES': '→',
            'CONFLICTS': '!'
        }.get(rel, '?')

        if verbose:
            print(f"  [{i+1:3d}] {symbol} {rel:11s} | {text[:50]}...")
        else:
            print(symbol, end='', flush=True)

        if (i + 1) % 20 == 0:
            s = kernel.summary()
            if verbose:
                print(f"        --- {s['beliefs']} beliefs, {s['conflicts']} conflicts ---")
            else:
                print(f" [{s['beliefs']}b/{s['conflicts']}c]", end='', flush=True)

    if not verbose:
        print()  # newline

    # Final summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    summary = kernel.summary()

    print(f"\nRelation counts:")
    for rel, count in sorted(summary['relations'].items()):
        pct = count / len(claims) * 100
        bar = '█' * int(pct / 5)
        print(f"  {rel:11s}: {count:3d} ({pct:5.1f}%) {bar}")

    print(f"\n  Total claims:  {len(claims)}")
    print(f"  Final beliefs: {len(summary['current_beliefs'])}")
    print(f"  Compression:   {len(claims) / max(len(summary['current_beliefs']), 1):.1f}x")

    # Show high-confidence beliefs (multiple sources)
    print(f"\nHigh-confidence beliefs (2+ sources):")
    multi_source = [b for b in kernel.beliefs if len(b.sources) >= 2]
    for b in multi_source[:10]:
        print(f"  [{len(b.sources)} src] {b.text[:70]}...")

    # Show beliefs that were updated
    print(f"\nBeliefs that evolved (superseded earlier values):")
    updated = [b for b in kernel.beliefs if b.supersedes]
    for b in updated[:5]:
        print(f"  NOW: {b.text[:60]}...")
        print(f"  WAS: {b.supersedes[:60]}...")

    print(f"\nUnresolved conflicts ({len(summary['unresolved_conflicts'])}):")
    for conflict in summary['unresolved_conflicts'][:5]:
        print(f"  ! {conflict['new_claim'][:60]}...")
        if conflict['existing_belief']:
            print(f"    vs: {conflict['existing_belief'][:60]}...")

    # Metrics
    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)

    total = len(claims)
    rels = summary['relations']

    # Common sense score: how well does it match human judgment?
    # Conflicts should be rare (true disagreements)
    # Redundant + Supersedes + Refines = recognized relationships
    recognized = rels.get('REDUNDANT', 0) + rels.get('SUPERSEDES', 0) + rels.get('REFINES', 0)
    conflict_rate = rels.get('CONFLICTS', 0) / total * 100

    print(f"""
  Claims processed:    {total}
  LLM calls:           {kernel.llm_calls}

  New facts added:     {rels.get('COMPATIBLE', 0)} ({rels.get('COMPATIBLE', 0)/total*100:.1f}%)
  Relationships found: {recognized} ({recognized/total*100:.1f}%)
    - Redundant:       {rels.get('REDUNDANT', 0)}
    - Supersedes:      {rels.get('SUPERSEDES', 0)}
    - Refines:         {rels.get('REFINES', 0)}

  Conflicts:           {rels.get('CONFLICTS', 0)} ({conflict_rate:.1f}%)

  Belief compression:  {total} claims → {len(summary['current_beliefs'])} beliefs ({len(claims)/max(len(summary['current_beliefs']),1):.1f}x)
""")

    return kernel, summary


if __name__ == '__main__':
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    num = int(args[0]) if args else None
    verbose = '--quiet' not in sys.argv
    asyncio.run(test_kernel(num_claims=num, verbose=verbose))
