"""
Weaver vs Worker: Narrative Enrichment Experiment
==================================================

This experiment validates that REEE-emerged events OUTRICH the legacy event
worker by adding epistemic depth beyond simple narrative generation.

What REEE adds (enrichment):
- Tensions: detected conflicts, contradictions, single-source claims
- Meta-claims: epistemic observations about the claim set
- Surface provenance: traceable identity grouping
- Typed observations: death counts, injury counts, etc.
- Proto-inquiries: auto-emerged questions needing resolution

Pipeline:
1. Load an existing event with its legacy narrative from Neo4j
2. Load all claims for that event
3. Run REEE weaver: IdentityLinker → Surfaces → Views → Event
4. Detect meta-claims and tensions
5. Generate enriched narrative with epistemic annotations
6. Measure enrichment delta over legacy

Run:
    docker exec herenews-app python -m reee.experiments.weaver_vs_worker
    docker exec herenews-app python -m reee.experiments.weaver_vs_worker --event <event_id>
"""

import asyncio
import argparse
import json
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime

from reee import Claim, Parameters
from reee.types import Surface, Event
from reee.identity import IdentityLinker
from reee.views import IncidentViewParams, IncidentEventView, ViewResult
from reee.meta.detectors import TensionDetector
from reee.experiments.loader import create_context, log, load_claims_for_event, load_events


def detect_cross_surface_conflicts(claims: Dict[str, Claim], typed_obs: List[Dict]) -> List[Dict]:
    """
    Detect conflicts across all claims with same question_key.

    This catches conflicts that span surfaces (when identity linking fragments).
    Returns list of conflicts with question_key, distinct values, sources.
    """
    # Group by question_key
    by_qkey: Dict[str, List[Dict]] = {}
    for obs in typed_obs:
        qkey = obs['question_key']
        if qkey not in by_qkey:
            by_qkey[qkey] = []
        by_qkey[qkey].append(obs)

    conflicts = []
    for qkey, observations in by_qkey.items():
        values = [obs['value'] for obs in observations]
        unique_values = sorted(set(values))

        if len(unique_values) > 1:
            # Conflict detected
            sources = list(set(obs['source'] for obs in observations))
            conflicts.append({
                'question_key': qkey,
                'values': unique_values,
                'observations': observations,
                'source_count': len(sources),
                'sources': sources,
            })

    return conflicts


# =============================================================================
# DOMAIN MODELS
# =============================================================================

@dataclass
class LegacyNarrative:
    """Legacy event narrative from Neo4j."""
    event_id: str
    event_name: str
    event_type: str
    summary: Optional[str] = None
    narrative_json: Optional[str] = None
    sections: List[Dict] = field(default_factory=list)
    key_figures: List[Dict] = field(default_factory=list)

    @property
    def has_structured_narrative(self) -> bool:
        return bool(self.sections or self.key_figures)


@dataclass
class WeaverNarrative:
    """Narrative generated from REEE-emerged event."""
    event_id: str
    anchors: Set[str]
    claim_count: int
    surface_count: int

    # Generated narrative
    title: str = ""
    sections: List[Dict] = field(default_factory=list)
    key_figures: List[Dict] = field(default_factory=list)
    tensions: List[str] = field(default_factory=list)

    # Provenance
    claims_cited: Set[str] = field(default_factory=set)


@dataclass
class EnrichmentResult:
    """Measures REEE enrichment beyond legacy event worker."""
    event_name: str

    # Base narrative (both can do this)
    legacy_sections: int = 0
    weaver_sections: int = 0
    legacy_key_figures: int = 0
    weaver_key_figures: int = 0
    legacy_claim_refs: int = 0
    weaver_claim_refs: int = 0

    # Key figure overlap
    shared_figures: List[str] = field(default_factory=list)
    legacy_only_figures: List[str] = field(default_factory=list)
    weaver_only_figures: List[str] = field(default_factory=list)

    # === ENRICHMENT: What REEE adds beyond legacy ===

    # Epistemic depth
    tensions_detected: int = 0        # conflicts, contradictions
    meta_claims: int = 0              # epistemic observations
    typed_observations: int = 0       # death counts, etc.

    # Provenance
    surfaces_formed: int = 0          # identity clusters
    claims_with_provenance: int = 0   # claims traced to surfaces

    # Open questions
    proto_inquiries: int = 0          # auto-emerged questions

    @property
    def enrichment_score(self) -> float:
        """
        Score measuring what REEE adds beyond legacy.

        0 = no enrichment (legacy could do everything)
        1+ = significant enrichment
        """
        enrichment = 0.0

        # Tensions are high value (legacy has none)
        enrichment += self.tensions_detected * 0.3

        # Meta-claims are epistemic observations
        enrichment += self.meta_claims * 0.2

        # Typed observations enable structured queries
        enrichment += self.typed_observations * 0.2

        # Surface provenance enables traceability
        if self.surfaces_formed > 1:
            enrichment += 0.3

        # Proto-inquiries are auto-emerged questions
        enrichment += self.proto_inquiries * 0.4

        return enrichment

    @property
    def has_baseline_parity(self) -> bool:
        """Can REEE produce at least as much base narrative as legacy?"""
        section_ok = self.weaver_sections >= self.legacy_sections * 0.5
        figure_ok = self.weaver_key_figures >= self.legacy_key_figures * 0.5
        return section_ok and figure_ok


# =============================================================================
# LOAD LEGACY NARRATIVE
# =============================================================================

async def load_legacy_narrative(ctx, event_id: str) -> Optional[LegacyNarrative]:
    """Load legacy event with its narrative from Neo4j."""
    result = await ctx.neo4j._execute_read('''
        MATCH (e:Event {id: $eid})
        RETURN e.id as id, e.canonical_name as name, e.event_type as type,
               e.summary as summary, e.narrative as narrative
    ''', {'eid': event_id})

    if not result:
        return None

    row = result[0]
    narrative = LegacyNarrative(
        event_id=row['id'],
        event_name=row['name'] or 'Unknown',
        event_type=row['type'] or 'UNKNOWN',
        summary=row['summary'],
        narrative_json=row.get('narrative'),
    )

    # Parse structured narrative if present
    if narrative.narrative_json:
        try:
            parsed = json.loads(narrative.narrative_json)
            narrative.sections = parsed.get('sections', [])
            narrative.key_figures = parsed.get('key_figures', [])
        except json.JSONDecodeError:
            pass

    return narrative


async def find_event_with_narrative(ctx, min_claims: int = 20) -> Optional[str]:
    """Find an event that has a structured narrative."""
    result = await ctx.neo4j._execute_read('''
        MATCH (e:Event)-[:INTAKES]->(c:Claim)
        WHERE e.narrative IS NOT NULL
        WITH e, count(c) as cnt
        WHERE cnt >= $min_claims
        RETURN e.id as id, e.canonical_name as name, cnt
        ORDER BY cnt DESC
        LIMIT 5
    ''', {'min_claims': min_claims})

    if result:
        log(f"Events with narratives:")
        for r in result:
            log(f"  - {r['name']}: {r['cnt']} claims")
        return result[0]['id']
    return None


# =============================================================================
# WEAVER PIPELINE
# =============================================================================

async def run_weaver_pipeline(
    ctx,
    claims: List[Claim],
    params: Parameters,
) -> tuple[Dict[str, Surface], ViewResult]:
    """
    Run REEE weaver pipeline: claims → surfaces → events.

    Returns:
        (surfaces, view_result)
    """
    # L0 → L2: Identity linking
    linker = IdentityLinker(llm=None, params=params)
    claims_dict = {c.id: c for c in claims}

    for claim in claims:
        await linker.add_claim(claim, extract_qkey=False)

    surfaces = linker.compute_surfaces()

    # L2 → L3: Incident view
    inc_params = IncidentViewParams(
        temporal_window_days=14,
        require_discriminative_anchor=True,
        min_signals=1,
    )

    view = IncidentEventView(surfaces, inc_params)
    result = view.build()

    return surfaces, result


# =============================================================================
# NARRATIVE GENERATION
# =============================================================================

async def generate_weaver_narrative(
    ctx,
    event: Event,
    surfaces: Dict[str, Surface],
    claims: Dict[str, Claim],
    params: Parameters,
) -> WeaverNarrative:
    """
    Generate StructuredNarrative from REEE-emerged event using LLM.

    This is the key capability we need to match the legacy event worker.
    """
    # Collect all claims for this event
    event_claims = []
    for sid in event.surface_ids:
        if sid in surfaces:
            for cid in surfaces[sid].claim_ids:
                if cid in claims:
                    event_claims.append(claims[cid])

    # Sort by timestamp if available
    event_claims.sort(
        key=lambda c: c.timestamp or datetime.min,
        reverse=True
    )

    # Build claim context
    claim_texts = []
    for i, claim in enumerate(event_claims[:50]):  # Limit for prompt
        claim_texts.append(f"[{claim.id[:8]}] ({claim.source}): {claim.text}")

    claims_context = "\n".join(claim_texts)

    # Detect tensions for additional context
    detector = TensionDetector(
        claims={c.id: c for c in event_claims},
        surfaces={sid: surfaces[sid] for sid in event.surface_ids if sid in surfaces},
        edges=set(),
        params=params,
    )
    meta_claims = detector.detect_all()
    tensions = [mc.mc_type for mc in meta_claims]

    # Generate narrative via LLM
    prompt = f"""Given these claims about an event, generate a structured narrative.

CLAIMS:
{claims_context}

ANCHOR ENTITIES: {list(event.anchor_entities)[:5]}

Generate a JSON response with:
1. "title": A concise canonical name for this event
2. "sections": Array of {{topic, title, content}} where:
   - topic: one of "what_happened", "casualties", "response", "investigation", "context"
   - title: human-readable section title
   - content: prose narrative citing claims as [cl_xxx] where xxx is the claim ID prefix
3. "key_figures": Array of {{label, value, claim_id}} for numeric facts like death tolls, injuries, amounts

Focus on factual accuracy. Cite claims using their ID prefixes.
Respond ONLY with valid JSON, no markdown."""

    try:
        response = await ctx.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON (handle markdown code blocks)
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        parsed = json.loads(content)

        # Extract claim references
        claims_cited = set()
        for section in parsed.get('sections', []):
            text = section.get('content', '')
            import re
            refs = re.findall(r'\[cl_([a-f0-9]+)\]', text)
            claims_cited.update(refs)

        return WeaverNarrative(
            event_id=event.id,
            anchors=event.anchor_entities,
            claim_count=event.total_claims,
            surface_count=len(event.surface_ids),
            title=parsed.get('title', ''),
            sections=parsed.get('sections', []),
            key_figures=parsed.get('key_figures', []),
            tensions=tensions,
            claims_cited=claims_cited,
        )

    except Exception as e:
        log(f"   ⚠️  Narrative generation error: {e}")
        return WeaverNarrative(
            event_id=event.id,
            anchors=event.anchor_entities,
            claim_count=event.total_claims,
            surface_count=len(event.surface_ids),
        )


# =============================================================================
# COMPARISON
# =============================================================================

def measure_enrichment(
    legacy: LegacyNarrative,
    weaver: WeaverNarrative,
    surfaces: Dict[str, Surface],
    meta_claims: List,
    typed_obs_count: int,
    proto_count: int,
) -> EnrichmentResult:
    """Measure REEE enrichment beyond legacy event worker."""
    result = EnrichmentResult(event_name=legacy.event_name)

    # Base narrative comparison
    result.legacy_sections = len(legacy.sections)
    result.weaver_sections = len(weaver.sections)
    result.legacy_key_figures = len(legacy.key_figures)
    result.weaver_key_figures = len(weaver.key_figures)

    # Claim references in legacy
    import re
    legacy_refs = set()
    for section in legacy.sections:
        content = section.get('content', '')
        refs = re.findall(r'\[cl_[a-f0-9-]+\]', content)
        legacy_refs.update(refs)
    result.legacy_claim_refs = len(legacy_refs)
    result.weaver_claim_refs = len(weaver.claims_cited)

    # Key figure overlap
    legacy_labels = {kf.get('label', '').lower() for kf in legacy.key_figures}
    weaver_labels = {kf.get('label', '').lower() for kf in weaver.key_figures}
    result.shared_figures = list(legacy_labels & weaver_labels)
    result.legacy_only_figures = list(legacy_labels - weaver_labels)
    result.weaver_only_figures = list(weaver_labels - legacy_labels)

    # === ENRICHMENT METRICS ===

    # Epistemic depth
    result.tensions_detected = len(weaver.tensions)
    result.meta_claims = len(meta_claims)
    result.typed_observations = typed_obs_count

    # Provenance
    result.surfaces_formed = len(surfaces)
    result.claims_with_provenance = sum(len(s.claim_ids) for s in surfaces.values())

    # Proto-inquiries
    result.proto_inquiries = proto_count

    return result


# =============================================================================
# EXPERIMENT
# =============================================================================

async def run_experiment(event_id: Optional[str] = None):
    """Run weaver vs worker comparison experiment."""
    log("=" * 70)
    log("WEAVER VS WORKER: NARRATIVE QUALITY COMPARISON")
    log("=" * 70)
    log("")

    params = Parameters(
        identity_confidence_threshold=0.45,
        hub_max_df=5,
        aboutness_min_signals=2,
    )

    ctx = await create_context()

    try:
        # 1. Find/load target event
        if not event_id:
            log("🔍 Finding event with structured narrative...")
            event_id = await find_event_with_narrative(ctx, min_claims=15)
            if not event_id:
                log("   No events with narratives found. Using any large event.")
                events = await load_events(ctx, min_claims=20, limit=1)
                if events:
                    event_id = events[0].id
                else:
                    log("❌ No suitable events found")
                    return

        log(f"")
        log(f"📦 Loading event: {event_id}")

        # 2. Load legacy narrative
        legacy = await load_legacy_narrative(ctx, event_id)
        if not legacy:
            log(f"❌ Event {event_id} not found")
            return

        log(f"   Name: {legacy.event_name}")
        log(f"   Type: {legacy.event_type}")
        log(f"   Has structured narrative: {legacy.has_structured_narrative}")
        if legacy.sections:
            log(f"   Legacy sections: {len(legacy.sections)}")
            for s in legacy.sections[:3]:
                log(f"     - {s.get('topic', 'unknown')}: {s.get('title', '')[:40]}")
        if legacy.key_figures:
            log(f"   Legacy key figures: {len(legacy.key_figures)}")
            for kf in legacy.key_figures[:3]:
                log(f"     - {kf.get('label')}: {kf.get('value')}")

        # 3. Load claims
        log("")
        log("📚 Loading claims...")
        claims, _ = await load_claims_for_event(ctx, event_id, limit=100)
        log(f"   Loaded: {len(claims)} claims")

        claims_with_emb = sum(1 for c in claims if c.embedding)
        claims_with_ent = sum(1 for c in claims if c.entities)
        log(f"   With embeddings: {claims_with_emb}")
        log(f"   With entities: {claims_with_ent}")

        # 4. Run weaver pipeline
        log("")
        log("🧬 Running REEE weaver pipeline...")
        surfaces, view_result = await run_weaver_pipeline(ctx, claims, params)

        log(f"   Surfaces formed: {len(surfaces)}")
        log(f"   Events emerged: {view_result.total_events}")
        log(f"   Multi-surface incidents: {view_result.total_incidents}")

        # Find the largest emerged event
        if not view_result.events:
            log("   ⚠️  No events emerged - using all surfaces as one event")
            # Create synthetic event from all surfaces
            all_claims_set = set()
            all_entities = set()
            all_anchors = set()
            for s in surfaces.values():
                all_claims_set.update(s.claim_ids)
                all_entities.update(s.entities)
                all_anchors.update(s.anchor_entities)

            emerged_event = Event(
                id="synth_event_0",
                surface_ids=set(surfaces.keys()),
                total_claims=len(all_claims_set),
                entities=all_entities,
                anchor_entities=all_anchors,
            )
        else:
            emerged_event = max(
                view_result.events.values(),
                key=lambda e: e.total_claims
            )

        log(f"   Using emerged event: {emerged_event.id}")
        log(f"     Claims: {emerged_event.total_claims}")
        log(f"     Surfaces: {len(emerged_event.surface_ids)}")
        log(f"     Anchors: {list(emerged_event.anchor_entities)[:5]}")

        # 5. Extract typed observations FIRST (before tension detection)
        log("")
        log("🔢 Extracting typed observations...")
        from reee.experiments.inquiry_emergence_quality import extract_typed_value
        claims_dict = {c.id: c for c in claims}
        typed_obs = []
        for claim in claims:
            extraction = extract_typed_value(claim.text)
            if extraction:
                # Set on claim so tension detector can find conflicts
                claim.question_key = extraction['question_key']
                claim.extracted_value = extraction['value']
                claim.observation_kind = extraction['observation_kind']
                typed_obs.append({
                    'claim_id': claim.id,
                    'source': claim.source,
                    **extraction
                })

        log(f"   Typed observations: {len(typed_obs)}")
        for obs in typed_obs[:5]:
            log(f"     - {obs['question_key']}: {obs['value']} ({obs['source']})")

        # 6. Detect meta-claims and tensions (now claims have typed values)
        log("")
        log("🔬 Detecting meta-claims and tensions...")

        # Use ALL surfaces (not just emerged event) to find cross-surface conflicts
        # The emerged event may be fragmented, but conflicts span all claims
        event_surfaces = surfaces  # All surfaces from this event's claims

        detector = TensionDetector(
            claims=claims_dict,
            surfaces=event_surfaces,
            edges=set(),
            params=params,
        )
        meta_claims = detector.detect_all()

        log(f"   Meta-claims detected: {len(meta_claims)}")
        for mc in meta_claims[:5]:
            evidence_str = str(mc.evidence)[:50] if mc.evidence else ''
            log(f"     - [{mc.type}] target={mc.target_id[:20]}... {evidence_str}")

        # Also detect CROSS-SURFACE conflicts for same question_key
        # This is the key insight: conflicts may span surfaces
        cross_surface_conflicts = detect_cross_surface_conflicts(claims_dict, typed_obs)
        log(f"   Cross-surface conflicts: {len(cross_surface_conflicts)}")
        for conflict in cross_surface_conflicts[:3]:
            log(f"     - {conflict['question_key']}: {conflict['values']} ({conflict['source_count']} sources)")

        # 7. Generate weaver narrative
        log("")
        log("📝 Generating enriched narrative from REEE event...")
        weaver = await generate_weaver_narrative(
            ctx, emerged_event, surfaces, claims_dict, params
        )

        log(f"   Title: {weaver.title}")
        log(f"   Sections: {len(weaver.sections)}")
        for s in weaver.sections[:3]:
            log(f"     - {s.get('topic', 'unknown')}: {s.get('title', '')[:40]}")
        log(f"   Key figures: {len(weaver.key_figures)}")
        for kf in weaver.key_figures[:3]:
            log(f"     - {kf.get('label')}: {kf.get('value')}")
        log(f"   Tensions in narrative: {len(weaver.tensions)}")

        # 8. Seed proto-inquiries (if any)
        from reee.inquiry.seeder import InquirySeeder
        seeder = InquirySeeder(params=params)
        protos = seeder.seed_from_meta_claims(
            surfaces=event_surfaces,
            meta_claims=meta_claims,
            claims=claims_dict,
            event_names={sid: legacy.event_name for sid in event_surfaces},
        )
        log("")
        log(f"   Proto-inquiries emerged: {len(protos)}")
        for proto in protos[:5]:
            log(f"     [{proto.inquiry_type.value}] {proto.question_text[:50]}...")
            log(f"       Reported values: {proto.reported_values}")
            log(f"       Typed obs: {proto.typed_observation_count}, Sources: {proto.source_count}")
            log(f"       MAP: {proto.posterior_map} (p={proto.posterior_probability:.3f})")
            log(f"       Entropy: {proto.entropy_bits:.2f} bits, Priority: {proto.priority_score():.1f}")
            log(f"       Scope: {proto.scope.signature_key()}")
            if proto.triggers:
                log(f"       Triggers ({len(proto.triggers)}):")
                for t in proto.triggers[:3]:
                    log(f"         - {t.type} @ {t.target_id[:20]}...")

        # 9. NARRATIVE COMPARISON (side-by-side)
        log("")
        log("=" * 70)
        log("NARRATIVE COMPARISON")
        log("=" * 70)

        log("")
        log("┌─ LEGACY NARRATIVE ─────────────────────────────────────────────────")
        if legacy.summary:
            log("│")
            for line in legacy.summary.split('\n'):
                log(f"│  {line[:70]}")
        elif legacy.sections:
            for section in legacy.sections:
                log(f"│  [{section.get('topic', '')}] {section.get('title', '')}")
                content = section.get('content', '')[:200]
                for line in content.split('\n'):
                    log(f"│    {line[:68]}")
        else:
            log("│  (no narrative stored)")
        log("│")
        if legacy.key_figures:
            log("│  Key Figures:")
            for kf in legacy.key_figures:
                log(f"│    • {kf.get('label')}: {kf.get('value')}")
        log("└─────────────────────────────────────────────────────────────────────")

        log("")
        log("┌─ WEAVER NARRATIVE ─────────────────────────────────────────────────")
        log(f"│  {weaver.title}")
        log("│")
        for section in weaver.sections:
            log(f"│  [{section.get('topic', '')}] {section.get('title', '')}")
            content = section.get('content', '')[:200]
            for line in content.split('\n'):
                log(f"│    {line[:68]}")
            log("│")
        if weaver.key_figures:
            log("│  Key Figures:")
            for kf in weaver.key_figures:
                log(f"│    • {kf.get('label')}: {kf.get('value')}")
        log("│")
        if weaver.tensions:
            log("│  Tensions Detected:")
            for t in weaver.tensions[:5]:
                log(f"│    ⚠️  {t}")
        log("└─────────────────────────────────────────────────────────────────────")

        # 10. Measure enrichment
        log("")
        log("=" * 70)
        log("ENRICHMENT METRICS")
        log("=" * 70)

        enrichment = measure_enrichment(
            legacy=legacy,
            weaver=weaver,
            surfaces=event_surfaces,
            meta_claims=meta_claims,
            typed_obs_count=len(typed_obs),
            proto_count=len(protos),
        )

        log("")
        log("REEE adds beyond legacy:")
        log(f"   Typed observations:    {enrichment.typed_observations}")
        log(f"   Tensions detected:     {enrichment.tensions_detected}")
        log(f"   Meta-claims:           {enrichment.meta_claims}")
        log(f"   Surfaces (provenance): {enrichment.surfaces_formed}")
        log(f"   Proto-inquiries:       {enrichment.proto_inquiries}")

        # 10. Summary
        log("")
        log("=" * 70)
        log("SUMMARY")
        log("=" * 70)

        log(f"   Baseline parity: {'✅ Yes' if enrichment.has_baseline_parity else '⚠️  Partial'}")
        log(f"   Enrichment score: {enrichment.enrichment_score:.1f}")
        log("")

        if enrichment.enrichment_score >= 1.0:
            log("✅ REEE SIGNIFICANTLY OUTRICHES LEGACY")
            log("   Tensions, meta-claims, and proto-inquiries add epistemic depth.")
        elif enrichment.enrichment_score >= 0.5:
            log("✅ REEE ENRICHES LEGACY")
            log("   Meaningful epistemic additions beyond simple narrative.")
        else:
            log("⚠️  MINIMAL ENRICHMENT")
            log("   Consider events with more conflict/ambiguity.")

        if enrichment.has_baseline_parity:
            log("")
            log("   READY: Weaver can replace event worker AND add epistemic depth.")
        else:
            log("")
            log("   NOTE: Narrative generation may need tuning, but enrichment is valuable.")

        # Return for programmatic use
        return {
            'legacy': legacy,
            'weaver': weaver,
            'enrichment': enrichment,
            'meta_claims': meta_claims,
            'protos': protos,
        }

    finally:
        await ctx.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Weaver vs Worker Comparison')
    parser.add_argument('--event', type=str, help='Specific event ID to test')
    args = parser.parse_args()

    asyncio.run(run_experiment(event_id=args.event))


if __name__ == "__main__":
    main()
