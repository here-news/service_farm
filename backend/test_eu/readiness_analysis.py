"""
Event Readiness Analysis - Epistemic & Semantic Assessment

For each emerged event, assess:

EPISTEMIC READINESS (Is it trustworthy?)
- Source diversity: Claims from multiple independent pages
- Corroboration density: Internal claim support
- Contradiction resolution: Are tensions resolved?
- Authority signals: Institutional sources, official statements

SEMANTIC READINESS (Is it complete?)
- Narrative completeness: Does it have 5W1H (Who, What, When, Where, Why, How)?
- Causal structure: Are causes and effects identified?
- Temporal coherence: Is timeline clear?
- Entity clarity: Are actors well-defined?

Run inside container:
    docker exec herenews-app python /app/test_eu/readiness_analysis.py
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
from datetime import datetime

from load_graph import load_snapshot, GraphSnapshot


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"

PG_HOST = os.environ.get("POSTGRES_HOST", "herenews-postgres")
PG_DB = os.environ.get("POSTGRES_DB", "herenews")
PG_USER = os.environ.get("POSTGRES_USER", "herenews_user")
PG_PASS = os.environ.get("POSTGRES_PASSWORD", "herenews_pass")

SIM_THRESHOLD = 0.70
LLM_THRESHOLD = 0.55
EVENT_MERGE_THRESHOLD = 0.60


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


def llm_call(prompt: str, max_tokens: int = 500) -> str:
    for attempt in range(3):
        try:
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0
                },
                timeout=90.0
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except:
            if attempt < 2:
                time.sleep(2)
            else:
                return ""


def llm_same_event(text1: str, text2: str) -> bool:
    prompt = f"""Are these claims about the same news story/event? Answer YES or NO.
Claim 1: {text1[:250]}
Claim 2: {text2[:250]}
Same story?"""
    return llm_call(prompt, 5).upper().startswith("YES")


def llm_same_story(summary1: str, summary2: str) -> bool:
    prompt = f"""Are these two descriptions part of the SAME broader news event/story? Answer YES or NO.
Description 1: {summary1[:300]}
Description 2: {summary2[:300]}
Same broader story?"""
    return llm_call(prompt, 5).upper().startswith("YES")


def llm_assess_5w1h(texts: List[str]) -> Dict:
    """Assess if event has Who, What, When, Where, Why, How"""
    combined = "\n".join([f"- {t[:150]}" for t in texts[:10]])
    prompt = f"""Analyze these claims from a news event and assess the 5W1H coverage:

Claims:
{combined}

For each element, rate coverage as: CLEAR (explicitly stated), PARTIAL (implied/incomplete), MISSING (not mentioned)

Return JSON:
{{
  "who": {{"status": "CLEAR|PARTIAL|MISSING", "value": "who is involved"}},
  "what": {{"status": "CLEAR|PARTIAL|MISSING", "value": "what happened"}},
  "when": {{"status": "CLEAR|PARTIAL|MISSING", "value": "when it happened"}},
  "where": {{"status": "CLEAR|PARTIAL|MISSING", "value": "where it happened"}},
  "why": {{"status": "CLEAR|PARTIAL|MISSING", "value": "why it happened"}},
  "how": {{"status": "CLEAR|PARTIAL|MISSING", "value": "how it happened"}}
}}"""

    result = llm_call(prompt, 400)
    try:
        if result.startswith("```"):
            result = result.split("```")[1].replace("json", "").strip()
        return json.loads(result)
    except:
        return {}


def llm_assess_narrative(texts: List[str]) -> Dict:
    """Assess narrative structure and completeness"""
    combined = "\n".join([f"- {t[:150]}" for t in texts[:10]])
    prompt = f"""Analyze these claims from a news event for narrative completeness:

Claims:
{combined}

Assess:
1. Is there a clear beginning/trigger event?
2. Is there development/progression?
3. Is there current status/outcome?
4. Are there open questions/unresolved elements?
5. What is the narrative arc stage? (EMERGING, DEVELOPING, CLIMAX, RESOLVING, CONCLUDED)

Return JSON:
{{
  "has_beginning": true/false,
  "has_development": true/false,
  "has_outcome": true/false,
  "open_questions": ["list of unresolved questions"],
  "narrative_stage": "EMERGING|DEVELOPING|CLIMAX|RESOLVING|CONCLUDED",
  "completeness_score": 0.0-1.0,
  "summary": "one sentence summary of the event"
}}"""

    result = llm_call(prompt, 400)
    try:
        if result.startswith("```"):
            result = result.split("```")[1].replace("json", "").strip()
        return json.loads(result)
    except:
        return {}


def llm_assess_epistemic_quality(texts: List[str], sources: int, corr: int, contra: int) -> Dict:
    """Assess epistemic quality - how trustworthy is this event?"""
    combined = "\n".join([f"- {t[:150]}" for t in texts[:8]])
    prompt = f"""Analyze the epistemic quality of this news event:

Claims:
{combined}

Statistics:
- Sources: {sources} different pages
- Corroborations: {corr} claims support each other
- Contradictions: {contra} claims conflict

Assess:
1. Are claims from diverse, independent sources?
2. Are there official/institutional sources?
3. Are there eyewitness accounts?
4. Is there speculation vs confirmed facts?
5. What is the overall confidence level?

Return JSON:
{{
  "source_independence": "HIGH|MEDIUM|LOW",
  "has_official_sources": true/false,
  "has_eyewitness": true/false,
  "speculation_ratio": 0.0-1.0,
  "confidence_level": "HIGH|MEDIUM|LOW|UNCERTAIN",
  "epistemic_concerns": ["list of concerns"],
  "readiness": "READY|NEEDS_VERIFICATION|PREMATURE"
}}"""

    result = llm_call(prompt, 400)
    try:
        if result.startswith("```"):
            result = result.split("```")[1].replace("json", "").strip()
        return json.loads(result)
    except:
        return {}


@dataclass
class EU:
    id: str
    level: int = 0
    claim_ids: List[str] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    page_ids: Set[str] = field(default_factory=set)
    embedding: Optional[List[float]] = None
    internal_corr: int = 0
    internal_contra: int = 0
    children: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None

    def size(self) -> int:
        return len(self.claim_ids)

    def coherence(self) -> float:
        total = self.internal_corr + self.internal_contra
        return self.internal_corr / total if total > 0 else 1.0

    def tension(self) -> float:
        total = self.internal_corr + self.internal_contra
        return self.internal_contra / total if total > 0 else 0.0

    def mass(self) -> float:
        return self.size() * 0.1 * (0.5 + self.coherence()) * (1 + 0.1 * len(self.page_ids))

    def state(self) -> str:
        return "ACTIVE" if self.tension() > 0.1 else "STABLE"

    def label(self) -> str:
        return self.texts[0][:60] + "..." if self.texts else "empty"


class ReadinessAnalyzer:
    def __init__(self, snapshot: GraphSnapshot):
        self.snapshot = snapshot
        self.eus: Dict[str, EU] = {}
        self.claim_to_eu: Dict[str, str] = {}
        self.sub_counter = 0
        self.event_counter = 0
        self.llm_calls = 0

    def build_hierarchy(self, cached: Dict[str, List[float]]):
        """Build EU hierarchy from claims"""
        all_claims = list(self.snapshot.claims.keys())
        random.seed(42)
        random.shuffle(all_claims)

        # Phase 1: Claims → Sub-events
        for cid in all_claims:
            claim = self.snapshot.claims[cid]
            self._process_claim(cid, claim.text, claim.page_id or "?", cached[cid])

        # Phase 2: Sub-events → Events
        self._merge_into_events(min_size=3)

    def _process_claim(self, claim_id: str, text: str, page_id: str, embedding: List[float]):
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

            claim = self.snapshot.claims.get(claim_id)
            if claim:
                for corr_id in claim.corroborates_ids:
                    if corr_id in best_eu.claim_ids:
                        best_eu.internal_corr += 1
                for contra_id in claim.contradicts_ids:
                    if contra_id in best_eu.claim_ids:
                        best_eu.internal_contra += 1

            self.claim_to_eu[claim_id] = best_eu.id
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

    def _merge_into_events(self, min_size: int = 3):
        candidates = [
            eu for eu in self.eus.values()
            if eu.level == 0 and eu.size() >= min_size and eu.parent_id is None
        ]

        if len(candidates) < 2:
            return

        candidates.sort(key=lambda x: x.size(), reverse=True)
        used = set()

        for eu in candidates:
            if eu.id in used:
                continue

            group = [eu]
            used.add(eu.id)

            for other in candidates:
                if other.id in used:
                    continue

                sim = cosine_sim(eu.embedding, other.embedding)

                if sim >= EVENT_MERGE_THRESHOLD:
                    self.llm_calls += 1
                    if llm_same_story(eu.texts[0], other.texts[0]):
                        group.append(other)
                        used.add(other.id)

            if len(group) > 1:
                self.event_counter += 1
                event = EU(
                    id=f"event_{self.event_counter}",
                    level=1,
                    claim_ids=[],
                    texts=[],
                    page_ids=set(),
                    children=[sub.id for sub in group]
                )

                all_embs = []
                for sub in group:
                    event.claim_ids.extend(sub.claim_ids)
                    event.texts.extend(sub.texts[:3])
                    event.page_ids |= sub.page_ids
                    event.internal_corr += sub.internal_corr
                    event.internal_contra += sub.internal_contra
                    if sub.embedding:
                        all_embs.append(sub.embedding)
                    sub.parent_id = event.id

                event.embedding = np.mean(all_embs, axis=0).tolist()
                self.eus[event.id] = event

    def analyze_event(self, eu: EU) -> Dict:
        """Full readiness analysis for an event"""
        print(f"\n  Analyzing: {eu.label()[:50]}")

        # Basic metrics
        metrics = {
            'id': eu.id,
            'size': eu.size(),
            'sources': len(eu.page_ids),
            'corroborations': eu.internal_corr,
            'contradictions': eu.internal_contra,
            'coherence': round(eu.coherence(), 3),
            'tension': round(eu.tension(), 3),
            'mass': round(eu.mass(), 2),
            'state': eu.state()
        }

        # 5W1H Assessment
        print("    → Assessing 5W1H...")
        self.llm_calls += 1
        w5h1 = llm_assess_5w1h(eu.texts)

        # Narrative Assessment
        print("    → Assessing narrative...")
        self.llm_calls += 1
        narrative = llm_assess_narrative(eu.texts)

        # Epistemic Assessment
        print("    → Assessing epistemic quality...")
        self.llm_calls += 1
        epistemic = llm_assess_epistemic_quality(
            eu.texts,
            len(eu.page_ids),
            eu.internal_corr,
            eu.internal_contra
        )

        # Compute readiness scores
        semantic_score = self._compute_semantic_score(w5h1, narrative)
        epistemic_score = self._compute_epistemic_score(epistemic, metrics)

        return {
            'metrics': metrics,
            '5w1h': w5h1,
            'narrative': narrative,
            'epistemic': epistemic,
            'scores': {
                'semantic_readiness': round(semantic_score, 2),
                'epistemic_readiness': round(epistemic_score, 2),
                'overall_readiness': round((semantic_score + epistemic_score) / 2, 2)
            },
            'recommendation': self._get_recommendation(semantic_score, epistemic_score, narrative, epistemic)
        }

    def _compute_semantic_score(self, w5h1: Dict, narrative: Dict) -> float:
        """Compute semantic readiness score (0-1)"""
        score = 0.0

        # 5W1H coverage (50% of semantic score)
        if w5h1:
            clear_count = sum(1 for k, v in w5h1.items() if isinstance(v, dict) and v.get('status') == 'CLEAR')
            partial_count = sum(1 for k, v in w5h1.items() if isinstance(v, dict) and v.get('status') == 'PARTIAL')
            w5h1_score = (clear_count * 1.0 + partial_count * 0.5) / 6
            score += w5h1_score * 0.5

        # Narrative completeness (50% of semantic score)
        if narrative:
            narrative_score = narrative.get('completeness_score', 0)
            score += narrative_score * 0.5

        return score

    def _compute_epistemic_score(self, epistemic: Dict, metrics: Dict) -> float:
        """Compute epistemic readiness score (0-1)"""
        score = 0.0

        # Source diversity (25%)
        source_count = metrics.get('sources', 1)
        source_score = min(source_count / 5, 1.0)  # 5+ sources = max
        score += source_score * 0.25

        # Coherence (25%)
        coherence = metrics.get('coherence', 0.5)
        score += coherence * 0.25

        # Epistemic quality assessment (50%)
        if epistemic:
            independence = {'HIGH': 1.0, 'MEDIUM': 0.6, 'LOW': 0.3}.get(
                epistemic.get('source_independence', 'LOW'), 0.3)
            confidence = {'HIGH': 1.0, 'MEDIUM': 0.6, 'LOW': 0.3, 'UNCERTAIN': 0.1}.get(
                epistemic.get('confidence_level', 'UNCERTAIN'), 0.1)
            speculation = 1.0 - epistemic.get('speculation_ratio', 0.5)

            epistemic_avg = (independence + confidence + speculation) / 3
            score += epistemic_avg * 0.5

        return score

    def _get_recommendation(self, semantic: float, epistemic: float, narrative: Dict, epistemic_data: Dict) -> Dict:
        """Generate actionable recommendation"""
        overall = (semantic + epistemic) / 2

        if overall >= 0.7:
            status = "READY_TO_PUBLISH"
            action = "Event is well-formed and trustworthy. Ready for publication."
        elif overall >= 0.5:
            status = "NEEDS_CURATION"
            action = "Event has gaps. Needs editorial review before publication."
        elif overall >= 0.3:
            status = "DEVELOPING"
            action = "Event is still emerging. Monitor for more claims."
        else:
            status = "PREMATURE"
            action = "Insufficient information. Too early to surface."

        gaps = []
        if semantic < 0.5:
            gaps.append("Missing key narrative elements (5W1H)")
        if epistemic < 0.5:
            gaps.append("Epistemic concerns (source diversity, verification)")
        if narrative and narrative.get('open_questions'):
            gaps.extend([f"Unresolved: {q}" for q in narrative.get('open_questions', [])[:3]])
        if epistemic_data and epistemic_data.get('epistemic_concerns'):
            gaps.extend(epistemic_data.get('epistemic_concerns', [])[:3])

        return {
            'status': status,
            'action': action,
            'gaps': gaps,
            'narrative_stage': narrative.get('narrative_stage', 'UNKNOWN') if narrative else 'UNKNOWN'
        }

    def get_top_events(self, n: int = 10) -> List[EU]:
        """Get top events by mass"""
        events = [eu for eu in self.eus.values() if eu.level == 1]
        large_subs = [eu for eu in self.eus.values() if eu.level == 0 and eu.size() >= 10 and eu.parent_id is None]
        all_top = events + large_subs
        return sorted(all_top, key=lambda x: x.mass(), reverse=True)[:n]


def run_analysis(snapshot: GraphSnapshot):
    print("Loading cached embeddings...")
    cached = load_cached_embeddings()
    print(f"  Found {len(cached)} cached embeddings")

    analyzer = ReadinessAnalyzer(snapshot)

    print("\nBuilding hierarchy...")
    analyzer.build_hierarchy(cached)

    level0 = [eu for eu in analyzer.eus.values() if eu.level == 0]
    level1 = [eu for eu in analyzer.eus.values() if eu.level == 1]
    print(f"  Sub-events: {len(level0)}, Events: {len(level1)}")

    # Analyze top events
    print(f"\n{'='*70}")
    print("READINESS ANALYSIS - Top Events")
    print(f"{'='*70}")

    top_events = analyzer.get_top_events(10)
    results = []

    for eu in top_events:
        result = analyzer.analyze_event(eu)
        results.append(result)

    # Print results
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}\n")

    for r in results:
        m = r['metrics']
        s = r['scores']
        rec = r['recommendation']

        status_icon = {
            'READY_TO_PUBLISH': '✅',
            'NEEDS_CURATION': '🟡',
            'DEVELOPING': '🔵',
            'PREMATURE': '⚪'
        }.get(rec['status'], '❓')

        print(f"{status_icon} [{m['id']}] {rec['status']}")
        print(f"   Size: {m['size']} claims, {m['sources']} sources")
        print(f"   Coherence: {m['coherence']:.0%}, Tension: {m['tension']:.0%}")
        print(f"   Semantic: {s['semantic_readiness']:.0%}, Epistemic: {s['epistemic_readiness']:.0%}")
        print(f"   Overall: {s['overall_readiness']:.0%}")

        narrative = r.get('narrative', {})
        if narrative:
            print(f"   Stage: {narrative.get('narrative_stage', '?')}")
            if narrative.get('summary'):
                print(f"   Summary: {narrative.get('summary', '')[:70]}...")

        if rec.get('gaps'):
            print(f"   Gaps: {', '.join(rec['gaps'][:2])}")
        print()

    # Print 5W1H analysis for top 3
    print(f"\n{'='*70}")
    print("5W1H COVERAGE (Top 3 Events)")
    print(f"{'='*70}\n")

    for r in results[:3]:
        m = r['metrics']
        w = r.get('5w1h', {})
        if w:
            print(f"[{m['id']}] {m['size']} claims")
            for element in ['who', 'what', 'when', 'where', 'why', 'how']:
                if element in w and isinstance(w[element], dict):
                    status = w[element].get('status', '?')
                    value = w[element].get('value', '')[:40]
                    icon = {'CLEAR': '✓', 'PARTIAL': '~', 'MISSING': '✗'}.get(status, '?')
                    print(f"   {icon} {element.upper()}: {value}")
            print()

    return analyzer, results


def main():
    print("=" * 70)
    print("Event Readiness Analysis")
    print("=" * 70 + "\n")

    snapshot = load_snapshot()
    print(f"Loaded {len(snapshot.claims)} claims\n")

    analyzer, results = run_analysis(snapshot)

    # Summary statistics
    print(f"{'='*70}")
    print("PUBLICATION READINESS SUMMARY")
    print(f"{'='*70}\n")

    ready = [r for r in results if r['recommendation']['status'] == 'READY_TO_PUBLISH']
    needs_curation = [r for r in results if r['recommendation']['status'] == 'NEEDS_CURATION']
    developing = [r for r in results if r['recommendation']['status'] == 'DEVELOPING']
    premature = [r for r in results if r['recommendation']['status'] == 'PREMATURE']

    print(f"✅ Ready to publish: {len(ready)}")
    print(f"🟡 Needs curation: {len(needs_curation)}")
    print(f"🔵 Still developing: {len(developing)}")
    print(f"⚪ Premature: {len(premature)}")

    # Save
    output = {
        'summary': {
            'ready': len(ready),
            'needs_curation': len(needs_curation),
            'developing': len(developing),
            'premature': len(premature),
            'llm_calls': analyzer.llm_calls
        },
        'events': results
    }

    output_path = Path("/app/test_eu/results/readiness_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
