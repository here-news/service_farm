"""
Inquiry Emergence Quality Test
==============================

Tests the quality of automatically emerged inquiries across multiple events
and previews webapp seeding.

This experiment:
1. Loads multiple events from Neo4j
2. Runs proto-inquiry emergence on each
3. Previews what webapp inquiries would be created
4. Evaluates quality metrics

Run:
    docker exec herenews-app python -m reee.experiments.inquiry_emergence_quality
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import re

from reee import Claim, Parameters
from reee.types import Surface, MetaClaim
from reee.identity import IdentityLinker
from reee.meta.detectors import TensionDetector
from reee.inquiry.seeder import InquirySeeder, ProtoInquiry
from reee.inquiry.webapp_seeder import WebappInquirySeeder, preview_proto_as_inquiry
from reee.experiments.loader import (
    create_context, load_events, load_claims_for_event, log
)


# =============================================================================
# VALUE EXTRACTION
# =============================================================================

def extract_typed_value(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract typed values from claim text.

    Returns dict with: question_key, value, observation_kind
    """
    text_lower = text.lower()

    # Death count patterns
    death_patterns = [
        (r'at\s+least\s+(\d+)(?:\s+(?:people\s+)?(?:died|dead|killed))', 'point'),
        (r'(\d+)\s+(?:people\s+)?(?:were\s+)?killed', 'point'),
        (r'(\d+)\s*(?:people\s+)?(?:died|dead)', 'point'),
        (r'death\s+(?:toll|count)\s+(?:rose|risen|hits?)?\s*(?:to\s+)?(\d+)', 'point'),
        (r'killed\s+(\d+)', 'point'),
        (r'(\d+)\s*fatalities', 'point'),
    ]

    for pattern, obs_kind in death_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                value = int(match.group(1))
                if 1 <= value <= 10000:
                    return {
                        'question_key': 'death_count',
                        'value': value,
                        'observation_kind': obs_kind,
                    }
            except ValueError:
                continue

    # Injury count patterns
    injury_patterns = [
        r'(\d+)\s*(?:people\s+)?(?:injured|hurt|wounded)',
        r'injuring\s+(\d+)',
    ]

    for pattern in injury_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                value = int(match.group(1))
                if 1 <= value <= 10000:
                    return {
                        'question_key': 'injury_count',
                        'value': value,
                        'observation_kind': 'point',
                    }
            except ValueError:
                continue

    # Sentence/prison patterns
    sentence_patterns = [
        r'sentenced\s+to\s+(\d+)\s+years?',
        r'(\d+)[\s-]*years?\s+(?:in\s+)?prison',
        r'prison\s+(?:sentence|term)\s+(?:of\s+)?(\d+)\s+years?',
    ]

    for pattern in sentence_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                value = int(match.group(1))
                if 1 <= value <= 200:
                    return {
                        'question_key': 'sentence_years',
                        'value': value,
                        'observation_kind': 'point',
                    }
            except ValueError:
                continue

    # Monetary amounts (millions)
    money_patterns = [
        r'\$(\d+(?:\.\d+)?)\s*(?:million|m\b)',
        r'(\d+(?:\.\d+)?)\s*million\s*dollars?',
    ]

    for pattern in money_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                value = float(match.group(1))
                if 0.1 <= value <= 10000:
                    return {
                        'question_key': 'amount_millions',
                        'value': value,
                        'observation_kind': 'point',
                    }
            except ValueError:
                continue

    return None


def enrich_claims_with_values(claims: List[Claim]) -> List[Claim]:
    """Add extracted values to claims."""
    for claim in claims:
        extraction = extract_typed_value(claim.text)
        if extraction:
            claim.question_key = extraction['question_key']
            claim.extracted_value = extraction['value']
            claim.observation_kind = extraction['observation_kind']
            claim.is_monotonic = extraction['question_key'] in ('death_count', 'injury_count')
    return claims


# =============================================================================
# QUALITY METRICS
# =============================================================================

@dataclass
class EventQualityReport:
    """Quality report for a single event."""
    event_id: str
    event_name: str
    claims_loaded: int = 0
    typed_claims: int = 0
    surfaces_formed: int = 0
    meta_claims: int = 0
    protos_emerged: int = 0

    # Quality flags
    has_conflict: bool = False
    has_typed_question: bool = False
    has_multi_source: bool = False

    # Proto details
    protos: List[Dict] = field(default_factory=list)
    webapp_previews: List[Dict] = field(default_factory=list)

    @property
    def quality_score(self) -> float:
        """0-100 quality score."""
        score = 0.0

        # Base score from emergence
        if self.protos_emerged > 0:
            score += 30

        # Typed questions are high quality
        if self.has_typed_question:
            score += 25

        # Conflicts are interesting
        if self.has_conflict:
            score += 25

        # Multi-source corroboration
        if self.has_multi_source:
            score += 20

        return min(score, 100)


@dataclass
class QualityTestResult:
    """Overall quality test result."""
    events_tested: int = 0
    events_with_protos: int = 0
    total_protos: int = 0

    reports: List[EventQualityReport] = field(default_factory=list)

    @property
    def emergence_rate(self) -> float:
        if self.events_tested == 0:
            return 0.0
        return self.events_with_protos / self.events_tested

    @property
    def avg_quality(self) -> float:
        if not self.reports:
            return 0.0
        return sum(r.quality_score for r in self.reports) / len(self.reports)


# =============================================================================
# EXPERIMENT
# =============================================================================

async def test_event(
    ctx,
    event,
    params: Parameters,
    seeder: InquirySeeder,
    webapp_seeder: WebappInquirySeeder,
) -> EventQualityReport:
    """Test proto-inquiry emergence on a single event."""
    report = EventQualityReport(
        event_id=event.id,
        event_name=event.name,
    )

    try:
        # Load claims
        claims_list, _ = await load_claims_for_event(ctx, event.id, limit=80)
        report.claims_loaded = len(claims_list)

        if len(claims_list) < 5:
            return report

        # Enrich with typed values
        claims_list = enrich_claims_with_values(claims_list)
        typed_claims = [c for c in claims_list if c.question_key]
        report.typed_claims = len(typed_claims)

        # Form surfaces
        linker = IdentityLinker(llm=None, params=params)
        claims_dict = {c.id: c for c in claims_list}

        for claim in claims_list:
            await linker.add_claim(claim, extract_qkey=False)

        surfaces = linker.compute_surfaces()
        report.surfaces_formed = len(surfaces)

        # Detect meta-claims
        detector = TensionDetector(
            claims=claims_dict,
            surfaces=surfaces,
            edges=linker.edges,
            params=params,
        )
        meta_claims = detector.detect_all()
        report.meta_claims = len(meta_claims)

        # Seed proto-inquiries
        protos = seeder.seed_from_meta_claims(
            surfaces=surfaces,
            meta_claims=meta_claims,
            claims=claims_dict,
            event_names={sid: event.name for sid in surfaces},
        )
        report.protos_emerged = len(protos)

        # Analyze protos
        for proto in protos:
            proto_info = {
                'question': proto.question_text,
                'type': proto.inquiry_type.value,
                'variable': proto.target_variable,
                'map': proto.posterior_map,
                'confidence': proto.posterior_probability,
                'typed_obs': proto.typed_observation_count,
                'sources': proto.source_count,
                'reported_values': proto.reported_values,
                'priority': proto.priority_score(),
            }
            report.protos.append(proto_info)

            # Check quality flags
            if proto.target_variable in ('death_count', 'injury_count', 'sentence_years'):
                report.has_typed_question = True

            if len(set(proto.reported_values)) > 1:
                report.has_conflict = True

            if proto.source_count >= 2:
                report.has_multi_source = True

            # Preview webapp inquiry
            preview = webapp_seeder.preview_inquiry(proto)
            report.webapp_previews.append(preview)

    except Exception as e:
        log(f"   ⚠️  Error testing {event.name}: {e}")

    return report


async def run_quality_test():
    """Run comprehensive quality test across multiple events."""
    log("=" * 70)
    log("INQUIRY EMERGENCE QUALITY TEST")
    log("=" * 70)
    log("")

    params = Parameters(
        identity_confidence_threshold=0.45,
        high_entropy_threshold=0.5,
    )

    seeder = InquirySeeder(params=params)
    webapp_seeder = WebappInquirySeeder(min_priority_score=20.0)

    result = QualityTestResult()

    ctx = await create_context()

    try:
        # Load multiple events
        log("📚 Loading events from database...")
        events = await load_events(ctx, min_claims=10, limit=10)
        log(f"   Found {len(events)} events with >= 10 claims")
        log("")

        # Test each event
        for i, event in enumerate(events, 1):
            log(f"[{i}/{len(events)}] Testing: {event.name[:50]}...")

            report = await test_event(ctx, event, params, seeder, webapp_seeder)
            result.reports.append(report)
            result.events_tested += 1

            if report.protos_emerged > 0:
                result.events_with_protos += 1
                result.total_protos += report.protos_emerged
                log(f"         ✅ {report.protos_emerged} proto(s), quality={report.quality_score:.0f}")
            else:
                log(f"         ○  No proto-inquiries (typed={report.typed_claims})")

        # Summary
        log("")
        log("=" * 70)
        log("SUMMARY")
        log("=" * 70)
        log(f"   Events tested: {result.events_tested}")
        log(f"   Events with proto-inquiries: {result.events_with_protos} ({result.emergence_rate:.0%})")
        log(f"   Total proto-inquiries: {result.total_protos}")
        log(f"   Average quality score: {result.avg_quality:.0f}/100")
        log("")

        # Detailed results
        log("=" * 70)
        log("EMERGED INQUIRIES (by quality)")
        log("=" * 70)
        log("")

        # Sort by quality
        reports_with_protos = [r for r in result.reports if r.protos_emerged > 0]
        reports_with_protos.sort(key=lambda r: r.quality_score, reverse=True)

        for report in reports_with_protos:
            log(f"📌 {report.event_name}")
            log(f"   Quality: {report.quality_score:.0f}/100")
            log(f"   Claims: {report.claims_loaded}, Typed: {report.typed_claims}")
            log("")

            for proto in report.protos:
                log(f"   ❓ {proto['question']}")
                log(f"      Type: {proto['type']}, Variable: {proto['variable']}")
                log(f"      MAP: {proto['map']} ({proto['confidence']:.0%} conf)")
                log(f"      Observations: {proto['typed_obs']} from {proto['sources']} sources")

                if proto['reported_values']:
                    unique = sorted(set(proto['reported_values']))
                    log(f"      Reported values: {unique}")
                    if len(unique) > 1:
                        log(f"      ⚠️  CONFLICT: {len(unique)} distinct values")

                log(f"      Priority: {proto['priority']:.1f}")
                log("")

            # Show webapp preview
            if report.webapp_previews:
                preview = report.webapp_previews[0]
                log(f"   📋 Webapp Preview:")
                log(f"      Title: {preview['inquiry']['title']}")
                log(f"      Schema: {preview['inquiry']['schema_type']}")
                log(f"      Tasks: {len(preview['tasks'])}")
                for task in preview['tasks']:
                    log(f"        - [{task['type']}] {task['description'][:50]}")
                log("")

            log("-" * 50)

        # Quality analysis
        log("")
        log("=" * 70)
        log("QUALITY ANALYSIS")
        log("=" * 70)
        log("")

        with_conflict = sum(1 for r in reports_with_protos if r.has_conflict)
        with_typed = sum(1 for r in reports_with_protos if r.has_typed_question)
        with_multi = sum(1 for r in reports_with_protos if r.has_multi_source)

        log(f"   Proto-inquiries with conflicts: {with_conflict}/{len(reports_with_protos)}")
        log(f"   Proto-inquiries with typed variables: {with_typed}/{len(reports_with_protos)}")
        log(f"   Proto-inquiries with multi-source: {with_multi}/{len(reports_with_protos)}")
        log("")

        if result.emergence_rate >= 0.5:
            log("✅ GOOD: Proto-inquiries emerge naturally from most events")
        elif result.emergence_rate >= 0.2:
            log("⚠️  MODERATE: Proto-inquiries emerge from some events")
        else:
            log("❌ POOR: Proto-inquiries rarely emerge")

        if result.avg_quality >= 60:
            log("✅ GOOD: Emerged inquiries are high quality")
        elif result.avg_quality >= 40:
            log("⚠️  MODERATE: Emerged inquiries have moderate quality")
        else:
            log("❌ POOR: Emerged inquiries need improvement")

        log("")

        # Return for programmatic use
        return result

    finally:
        await ctx.close()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    asyncio.run(run_quality_test())
