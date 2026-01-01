"""
Proto-Inquiry Emergence Experiment
==================================

Tests whether the weaver can naturally yield valid inquiries
without user input by:

1. Loading claims from a real event (Hong Kong fire)
2. Forming surfaces using IdentityLinker
3. Detecting meta-claims (tensions)
4. Seeding proto-inquiries from surfaces + meta-claims

Expected outcome:
- Surfaces form around typed variables (death_count, injury_count)
- Meta-claims detect conflicts/entropy
- Proto-inquiries emerge with clear questions

Run:
    docker exec herenews-app python -m reee.experiments.proto_inquiry_emergence
"""

import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
import re

from reee import Claim, Parameters
from reee.types import Surface, MetaClaim
from reee.identity import IdentityLinker
from reee.meta.detectors import TensionDetector
from reee.inquiry.seeder import InquirySeeder, ProtoInquiry
from reee.experiments.loader import (
    create_context, load_events, load_claims_for_event, log
)


# =============================================================================
# VALUE EXTRACTION (for typed surfaces)
# =============================================================================

def extract_death_count(text: str) -> Optional[tuple]:
    """
    Extract death count from claim text.

    Returns: (value, observation_kind) tuple or None
    - observation_kind: 'point' (default for all extractions)

    NOTE: We intentionally treat ALL extractions as POINT observations.
    The observation_kind distinction (lower_bound, update) only matters
    when we have TIMESTAMPS to establish temporal ordering.

    Without timestamps, "at least 128" vs "at least 160" are just
    competing claims that should both contribute to the posterior.
    """
    text_lower = text.lower()

    # All patterns extract as POINT - timestamps determine semantics
    patterns = [
        r'at\s+least\s+(\d+)',  # "at least 160"
        r'(\d+)\s+(?:people\s+)?(?:were\s+)?killed',
        r'(\d+)\s*(?:people\s+)?(?:died|dead)',
        r'death\s+(?:toll|count)\s+(?:rose|risen|hits?)?\s*(?:to\s+)?(\d+)',
        r'killed\s+(\d+)',
        r'(\d+)\s*fatalities',
        r'(\d+)\s+out\s+of\s+.*?dead',
        r'(?:the\s+)?(\d+)\s+dead',
    ]

    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                value = int(match.group(1))
                if 1 <= value <= 1000:
                    return (value, 'point')
            except ValueError:
                continue

    return None


def extract_injury_count(text: str) -> Optional[int]:
    """Extract injury count from claim text."""
    patterns = [
        r'(\d+)\s*(?:people\s+)?(?:injured|hurt|wounded)',
        r'(?:injury\s+count|injuries)[:\s]+(\d+)',
        r'injuring\s+(\d+)',
        r'(\d+)\s+(?:were\s+)?injured',
    ]

    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return None


def has_update_language(text: str) -> bool:
    """Check if text contains update language."""
    update_patterns = [
        r'revised', r'updated', r'now stands at', r'risen to',
        r'increased to', r'climbed to', r'death toll (?:has )?risen',
    ]
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in update_patterns)


def enrich_claims_with_values(claims: List[Claim]) -> List[Claim]:
    """Add extracted values to claims."""
    for claim in claims:
        # Try death count
        result = extract_death_count(claim.text)
        if result is not None:
            value, obs_kind = result
            claim.question_key = 'death_count'
            claim.extracted_value = value
            claim.observation_kind = obs_kind  # 'point', 'lower_bound', or 'update'
            claim.has_update_language = (obs_kind == 'update')
            claim.is_monotonic = True
            continue

        # Try injury count
        injury = extract_injury_count(claim.text)
        if injury is not None:
            claim.question_key = 'injury_count'
            claim.extracted_value = injury
            claim.observation_kind = 'point'
            claim.has_update_language = has_update_language(claim.text)
            claim.is_monotonic = True

    return claims


# =============================================================================
# EXPERIMENT
# =============================================================================

async def run_experiment():
    """
    Run the proto-inquiry emergence experiment.

    Steps:
    1. Load real claims from Neo4j
    2. Enrich with extracted values
    3. Run Engine to form surfaces
    4. Detect meta-claims
    5. Seed proto-inquiries
    6. Analyze results
    """
    log("=" * 70)
    log("Proto-Inquiry Emergence Experiment")
    log("=" * 70)
    log("")

    # 1. Load claims
    log("📚 Loading claims from database...")
    ctx = await create_context()

    try:
        events = await load_events(ctx, min_claims=15, limit=1)
        if not events:
            log("❌ No events found with sufficient claims")
            return

        event = events[0]
        log(f"   Event: {event.name}")
        log(f"   Claims in DB: {event.claim_count}")

        claims_list, claim_to_event = await load_claims_for_event(
            ctx, event.id, limit=80  # More claims for better coverage
        )
        log(f"   Loaded: {len(claims_list)} claims")

        # 2. Enrich with extracted values
        log("")
        log("🔍 Extracting typed values from claims...")
        claims_list = enrich_claims_with_values(claims_list)

        typed_claims = [c for c in claims_list if c.question_key]
        log(f"   Typed claims: {len(typed_claims)}")

        by_qkey = {}
        for c in typed_claims:
            by_qkey.setdefault(c.question_key, []).append(c)

        for qkey, qclaims in by_qkey.items():
            values = [c.extracted_value for c in qclaims]
            log(f"   - {qkey}: {len(qclaims)} claims, values: {sorted(set(values))}")

        # 3. Run IdentityLinker to form surfaces
        log("")
        log("🧬 Running IdentityLinker to form surfaces...")
        params = Parameters(
            identity_confidence_threshold=0.45,
            high_entropy_threshold=0.5,
        )

        linker = IdentityLinker(llm=None, params=params)

        # Add claims to linker (skip question_key extraction since we already enriched)
        claims_dict = {c.id: c for c in claims_list}
        for claim in claims_list:
            await linker.add_claim(claim, extract_qkey=False)

        # Compute surfaces from identity edges
        surfaces = linker.compute_surfaces()

        log(f"   Surfaces formed: {len(surfaces)}")
        log(f"   Identity edges: {len(linker.edges)}")

        # Show surfaces
        for sid, surface in list(surfaces.items())[:5]:
            surface_claims = [claims_dict[cid] for cid in surface.claim_ids if cid in claims_dict]
            sources = {c.source for c in surface_claims}
            log(f"   - {sid[:12]}: {len(surface.claim_ids)} claims, {len(sources)} sources")

        # 4. Detect meta-claims
        log("")
        log("🔬 Detecting tensions (meta-claims)...")
        detector = TensionDetector(
            claims=claims_dict,
            surfaces=surfaces,
            edges=linker.edges,
            params=params,
        )
        meta_claims = detector.detect_all()

        log(f"   Meta-claims detected: {len(meta_claims)}")

        by_type = {}
        for mc in meta_claims:
            by_type.setdefault(mc.type, []).append(mc)

        for mc_type, mcs in by_type.items():
            log(f"   - {mc_type}: {len(mcs)}")

        # 5. Seed proto-inquiries
        log("")
        log("🌱 Seeding proto-inquiries...")
        seeder = InquirySeeder(params=params)

        protos = seeder.seed_from_meta_claims(
            surfaces=surfaces,
            meta_claims=meta_claims,
            claims=claims_dict,
            event_names={sid: event.name for sid in surfaces},
        )

        log(f"   Proto-inquiries emerged: {len(protos)}")

        # 6. Analyze results
        log("")
        log("=" * 70)
        log("EMERGED PROTO-INQUIRIES")
        log("=" * 70)

        for i, proto in enumerate(protos[:5], 1):
            log("")
            log(f"[{i}] {proto.question_text}")
            log(f"    Type: {proto.inquiry_type.value}")
            log(f"    Schema: {proto.schema_type.value}")
            log(f"    Variable: {proto.target_variable}")
            log("")
            log(f"    EPISTEMIC ACCOUNTING:")
            log(f"      Typed observations: {proto.typed_observation_count} (feed posterior)")
            log(f"      Supporting claims: {proto.supporting_claim_count} (context only)")
            log(f"      Sources: {proto.source_count}")
            log("")
            log(f"    OBSERVATIONS (detailed):")
            for j, obs in enumerate(proto.observations, 1):
                obs_value = obs.bound_value if obs.kind.value == 'lower_bound' else obs.map_value
                is_update = obs.signals.get('is_update', False)
                ts = obs.timestamp if obs.timestamp else 'none'
                conf = obs.extraction_confidence
                log(f"      [{j}] value={obs_value}, kind={obs.kind.value}, is_update={is_update}, ts={ts}, conf={conf:.2f}, src={obs.source}")
            log("")
            log(f"    BELIEF STATE BOUNDS:")
            if proto.belief_state:
                log(f"      hard_lower_bound: {proto.belief_state._hard_lower_bound}")
                log(f"      soft_lower_bound: {proto.belief_state._soft_lower_bound}")
            else:
                log(f"      (no belief state)")
            log("")
            log(f"    REPORTED VALUES (actual claims):")
            if proto.reported_values:
                unique_values = sorted(set(proto.reported_values))
                log(f"      Values: {unique_values}")
                if len(unique_values) > 1:
                    log(f"      ⚠️  CONFLICT DETECTED: {len(unique_values)} distinct values")
            else:
                log(f"      (no typed observations)")
            log("")
            log(f"    POSTERIOR (Bayesian inference):")
            log(f"      MAP estimate: {proto.posterior_map}")
            log(f"      Confidence: {proto.posterior_probability:.1%}")
            log(f"      Entropy: {proto.entropy_bits:.2f} bits")
            log("")
            log(f"    Credible interval (noise spillover, not claimed values):")
            for h in proto.hypotheses[:5]:
                log(f"      - {h['value']}: {h['probability']:.1%}")
            log("")
            log(f"    Priority Score: {proto.priority_score():.1f}")

        # 7. Summary
        log("")
        log("=" * 70)
        log("SUMMARY")
        log("=" * 70)
        log(f"   Claims loaded: {len(claims_list)}")
        log(f"   Typed claims: {len(typed_claims)}")
        log(f"   Surfaces formed: {len(surfaces)}")
        log(f"   Meta-claims: {len(meta_claims)}")
        log(f"   Proto-inquiries: {len(protos)}")
        log("")

        if protos:
            log("✅ Proto-inquiries emerged naturally from weaving!")
            log("   The system can seed inquiries without user input.")
        else:
            log("⚠️  No proto-inquiries emerged.")
            log("   This could mean:")
            log("   - No typed surfaces (need question_key extraction)")
            log("   - No tensions detected (low entropy, no conflicts)")
            log("   - All claims agree (no investigation needed)")

    finally:
        await ctx.close()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    asyncio.run(run_experiment())
