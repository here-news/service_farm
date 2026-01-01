#!/usr/bin/env python3
"""
Forecast Inquiry Experiment
===========================

Tests REEE's ability to handle predictive inquiries with:
1. Boolean event with deadline (e.g., "Will GPT-5 release before July 2025?")
2. Indicator-based evidence (hiring signals, API changes, official statements)
3. Deadline-based resolution (adjudication when time passes)

Key concepts:
- ForecastDomain: Boolean event with deadline
- Indicator surfaces: Separate evidence types that shift odds
- Resolution rules: Time-based adjudication at deadline
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reee.inquiry.engine import InquiryEngine
from reee.inquiry.types import (
    InquirySchema, RigorLevel, InquiryStatus,
    ContributionType, TaskType
)
from reee.typed_belief import ObservationKind


def log(msg: str):
    print(msg, flush=True)


def section(title: str):
    log(f"\n{'='*70}")
    log(f"  {title}")
    log(f"{'='*70}\n")


# =============================================================================
# FORECAST INDICATOR TYPES
# =============================================================================

class IndicatorType(Enum):
    """Types of evidence that shift forecast odds."""
    OFFICIAL_STATEMENT = "official_statement"    # Company/person statement
    HIRING_SIGNAL = "hiring_signal"              # Job postings, team growth
    PRODUCT_SIGNAL = "product_signal"            # API changes, docs updates
    MEDIA_REPORT = "media_report"                # Journalist reports
    INSIDER_LEAK = "insider_leak"                # Anonymous sources
    TIMELINE_MENTION = "timeline_mention"        # Explicit date mentions
    DENIAL = "denial"                            # Official denial


@dataclass
class IndicatorLikelihood:
    """Likelihood model for how indicator affects forecast."""
    indicator_type: IndicatorType
    # P(indicator | event will happen) / P(indicator | event won't happen)
    likelihood_ratio_positive: float  # If indicator suggests YES
    likelihood_ratio_negative: float  # If indicator suggests NO

    @classmethod
    def defaults(cls) -> Dict[IndicatorType, 'IndicatorLikelihood']:
        """Default likelihood ratios for each indicator type."""
        return {
            IndicatorType.OFFICIAL_STATEMENT: cls(
                IndicatorType.OFFICIAL_STATEMENT,
                likelihood_ratio_positive=5.0,   # Strong if confirms
                likelihood_ratio_negative=0.3,   # Strong if denies
            ),
            IndicatorType.HIRING_SIGNAL: cls(
                IndicatorType.HIRING_SIGNAL,
                likelihood_ratio_positive=1.5,   # Weak positive
                likelihood_ratio_negative=0.8,   # Weak negative
            ),
            IndicatorType.PRODUCT_SIGNAL: cls(
                IndicatorType.PRODUCT_SIGNAL,
                likelihood_ratio_positive=2.0,   # Moderate positive
                likelihood_ratio_negative=0.6,   # Moderate negative
            ),
            IndicatorType.MEDIA_REPORT: cls(
                IndicatorType.MEDIA_REPORT,
                likelihood_ratio_positive=1.8,   # Moderate
                likelihood_ratio_negative=0.7,
            ),
            IndicatorType.INSIDER_LEAK: cls(
                IndicatorType.INSIDER_LEAK,
                likelihood_ratio_positive=2.5,   # Higher uncertainty
                likelihood_ratio_negative=0.5,
            ),
            IndicatorType.TIMELINE_MENTION: cls(
                IndicatorType.TIMELINE_MENTION,
                likelihood_ratio_positive=3.0,   # Explicit dates are strong
                likelihood_ratio_negative=0.4,
            ),
            IndicatorType.DENIAL: cls(
                IndicatorType.DENIAL,
                likelihood_ratio_positive=0.3,   # Official denial is strong negative
                likelihood_ratio_negative=3.0,
            ),
        }


@dataclass
class ForecastInquiry:
    """Extended inquiry type for forecasts."""
    base_inquiry_id: str
    event_description: str  # "GPT-5 releases"
    deadline: datetime      # By when?

    # Resolution
    resolution_sources: List[str] = field(default_factory=list)  # URLs to check
    resolved: bool = False
    resolution_value: Optional[bool] = None
    resolution_time: Optional[datetime] = None

    # Indicator model
    indicator_likelihoods: Dict[IndicatorType, IndicatorLikelihood] = field(
        default_factory=IndicatorLikelihood.defaults
    )

    # Evidence trail
    indicators: List[Dict] = field(default_factory=list)


# =============================================================================
# FORECAST ENGINE WRAPPER
# =============================================================================

class ForecastEngine:
    """
    Wraps InquiryEngine with forecast-specific logic.

    Key additions:
    - Indicator-based evidence with likelihood model
    - Deadline-based resolution tasks
    - Adjudication at deadline
    """

    def __init__(self):
        self.engine = InquiryEngine()
        self.forecasts: Dict[str, ForecastInquiry] = {}

    def create_forecast(
        self,
        event_description: str,
        deadline: datetime,
        resolution_sources: List[str],
        created_by: str,
        initial_stake: float = 0.0,
        prior_probability: float = 0.5,
    ) -> ForecastInquiry:
        """Create a forecast inquiry."""

        # Create base inquiry with boolean schema
        schema = InquirySchema(
            schema_type="boolean",
            rigor=RigorLevel.B
        )

        title = f"Will {event_description} before {deadline.strftime('%Y-%m-%d')}?"

        base_inquiry = self.engine.create_inquiry(
            title=title,
            description=f"Forecast: {event_description}. Deadline: {deadline.isoformat()}",
            schema=schema,
            created_by=created_by,
            scope_keywords=[event_description.split()[0]],  # First word as keyword
            initial_stake=initial_stake,
        )

        # Create forecast wrapper
        forecast = ForecastInquiry(
            base_inquiry_id=base_inquiry.id,
            event_description=event_description,
            deadline=deadline,
            resolution_sources=resolution_sources,
        )

        self.forecasts[base_inquiry.id] = forecast

        return forecast

    async def add_indicator(
        self,
        forecast_id: str,
        user_id: str,
        indicator_type: IndicatorType,
        text: str,
        source_url: str,
        direction: str = "positive",  # "positive" or "negative"
    ) -> Dict:
        """Add an indicator to a forecast."""

        forecast = self.forecasts.get(forecast_id)
        if not forecast:
            raise ValueError(f"Forecast {forecast_id} not found")

        # Get likelihood model
        likelihoods = forecast.indicator_likelihoods.get(indicator_type)
        if not likelihoods:
            likelihoods = IndicatorLikelihood(indicator_type, 1.5, 0.7)

        # Compute implied value
        if direction == "positive":
            # Evidence suggests event WILL happen
            lr = likelihoods.likelihood_ratio_positive
            value = "true"
        else:
            # Evidence suggests event WON'T happen
            lr = likelihoods.likelihood_ratio_negative
            value = "false"

        # Add as contribution
        contrib = await self.engine.add_contribution(
            inquiry_id=forecast_id,
            user_id=user_id,
            contribution_type=ContributionType.EVIDENCE,
            text=f"[{indicator_type.value}] {text}",
            source_url=source_url,
            extracted_value=value,
            observation_kind="point"
        )

        # Record indicator
        indicator_record = {
            "type": indicator_type.value,
            "direction": direction,
            "likelihood_ratio": lr,
            "text": text,
            "source": source_url,
            "contribution_id": contrib.id,
            "timestamp": datetime.utcnow().isoformat(),
        }
        forecast.indicators.append(indicator_record)

        return indicator_record

    def check_resolution(self, forecast_id: str) -> Dict:
        """Check if forecast can be resolved."""

        forecast = self.forecasts.get(forecast_id)
        if not forecast:
            return {"error": "Forecast not found"}

        now = datetime.utcnow()
        inquiry = self.engine.get_inquiry(forecast_id)
        trace = self.engine.get_trace(forecast_id)

        result = {
            "forecast_id": forecast_id,
            "event": forecast.event_description,
            "deadline": forecast.deadline.isoformat(),
            "time_remaining": str(forecast.deadline - now) if now < forecast.deadline else "PAST",
            "current_probability": trace['belief_state']['map_probability'],
            "current_map": trace['belief_state']['map'],
            "indicators_count": len(forecast.indicators),
        }

        if now >= forecast.deadline:
            result["resolution_due"] = True
            result["resolution_task"] = {
                "type": "resolution_check_due",
                "description": f"Verify if {forecast.event_description} occurred by {forecast.deadline}",
                "sources_to_check": forecast.resolution_sources,
            }
        else:
            result["resolution_due"] = False

        return result

    async def resolve_forecast(
        self,
        forecast_id: str,
        outcome: bool,
        evidence_url: str,
        resolved_by: str,
    ) -> Dict:
        """Manually resolve a forecast at/after deadline."""

        forecast = self.forecasts.get(forecast_id)
        if not forecast:
            raise ValueError(f"Forecast {forecast_id} not found")

        # Add final resolution evidence
        await self.engine.add_contribution(
            inquiry_id=forecast_id,
            user_id=resolved_by,
            contribution_type=ContributionType.EVIDENCE,
            text=f"RESOLUTION: {forecast.event_description} {'occurred' if outcome else 'did not occur'} by deadline",
            source_url=evidence_url,
            extracted_value="true" if outcome else "false",
            observation_kind="point"
        )

        # Mark resolved
        forecast.resolved = True
        forecast.resolution_value = outcome
        forecast.resolution_time = datetime.utcnow()

        return {
            "resolved": True,
            "outcome": outcome,
            "resolution_time": forecast.resolution_time.isoformat(),
        }


# =============================================================================
# EXPERIMENT: GPT-5 RELEASE FORECAST
# =============================================================================

async def run_gpt5_forecast_experiment():
    """Simulate a forecast inquiry about GPT-5 release."""

    section("FORECAST EXPERIMENT: GPT-5 Release")

    engine = ForecastEngine()

    # Create forecast
    log("  Creating forecast inquiry...")
    forecast = engine.create_forecast(
        event_description="GPT-5 releases publicly",
        deadline=datetime(2025, 7, 1),
        resolution_sources=[
            "https://openai.com/blog",
            "https://platform.openai.com/docs/models",
            "https://openai.com/gpt-5",
        ],
        created_by="market_maker",
        initial_stake=500.0,
        prior_probability=0.5,
    )

    log(f"  Forecast ID: {forecast.base_inquiry_id}")
    log(f"  Event: {forecast.event_description}")
    log(f"  Deadline: {forecast.deadline}")

    # Add indicators
    section("Adding Indicators")

    indicators = [
        (IndicatorType.HIRING_SIGNAL, "positive",
         "OpenAI posts 50+ roles for 'post-training evaluation'",
         "https://openai.com/careers"),

        (IndicatorType.MEDIA_REPORT, "positive",
         "The Information reports GPT-5 training completed in Q4 2024",
         "https://theinformation.com/gpt5-training"),

        (IndicatorType.INSIDER_LEAK, "positive",
         "Anonymous source claims internal demo scheduled for March",
         "https://twitter.com/leak_account"),

        (IndicatorType.OFFICIAL_STATEMENT, "negative",
         "Sam Altman: 'We're focused on o1-pro and reasoning, not GPT-5 naming'",
         "https://x.com/sama/status/12345"),

        (IndicatorType.PRODUCT_SIGNAL, "positive",
         "API docs show new 'gpt-5-preview' model ID in staging",
         "https://platform.openai.com/docs"),

        (IndicatorType.TIMELINE_MENTION, "positive",
         "OpenAI investor letter mentions 'major model launch H1 2025'",
         "https://reuters.com/openai-investors"),
    ]

    for i, (itype, direction, text, url) in enumerate(indicators):
        log(f"\n  [{i+1}] Adding {itype.value} ({direction})")
        log(f"      \"{text[:60]}...\"")

        result = await engine.add_indicator(
            forecast_id=forecast.base_inquiry_id,
            user_id=f"analyst_{i}",
            indicator_type=itype,
            text=text,
            source_url=url,
            direction=direction,
        )

        # Check current state
        trace = engine.engine.get_trace(forecast.base_inquiry_id)
        log(f"      → P(release): {trace['belief_state']['map_probability']*100:.1f}%")

    # Final state
    section("Final Forecast State")

    trace = engine.engine.get_trace(forecast.base_inquiry_id)
    log(f"  MAP: {trace['belief_state']['map']}")
    log(f"  P(MAP): {trace['belief_state']['map_probability']*100:.1f}%")
    log(f"  Indicators processed: {len(forecast.indicators)}")

    log("\n  Indicator breakdown:")
    positive = sum(1 for i in forecast.indicators if i['direction'] == 'positive')
    negative = sum(1 for i in forecast.indicators if i['direction'] == 'negative')
    log(f"    Positive signals: {positive}")
    log(f"    Negative signals: {negative}")

    # Resolution check
    section("Resolution Check")

    resolution = engine.check_resolution(forecast.base_inquiry_id)
    log(f"  Deadline: {resolution['deadline']}")
    log(f"  Time remaining: {resolution['time_remaining']}")
    log(f"  Resolution due: {resolution['resolution_due']}")
    log(f"  Current probability: {resolution['current_probability']*100:.1f}%")

    if resolution.get('resolution_task'):
        log(f"\n  Resolution task:")
        log(f"    {resolution['resolution_task']['description']}")
        log(f"    Sources to check: {resolution['resolution_task']['sources_to_check']}")

    # Simulate resolution
    section("Simulating Resolution (at deadline)")

    # Pretend deadline has passed and we're resolving
    log("  Scenario: Deadline passed, GPT-5 was released...")

    result = await engine.resolve_forecast(
        forecast_id=forecast.base_inquiry_id,
        outcome=True,
        evidence_url="https://openai.com/blog/gpt-5",
        resolved_by="resolver",
    )

    log(f"  Resolution: {'TRUE' if result['outcome'] else 'FALSE'}")
    log(f"  Resolution time: {result['resolution_time']}")

    trace = engine.engine.get_trace(forecast.base_inquiry_id)
    log(f"  Final P(true): {trace['belief_state']['map_probability']*100:.1f}%")

    return forecast, trace


# =============================================================================
# EXPERIMENT: REPORT-TRUTH VS WORLD-TRUTH
# =============================================================================

async def run_report_vs_world_experiment():
    """Test distinction between 'did they report it?' vs 'is it true?'."""

    section("EXPERIMENT: Report-Truth vs World-Truth")

    engine = InquiryEngine()

    # Inquiry 1: Did Reuters report X?
    log("  Creating report-truth inquiry...")

    report_schema = InquirySchema(
        schema_type="boolean",
        rigor=RigorLevel.A  # High rigor - just checking if report exists
    )

    report_inquiry = engine.create_inquiry(
        title="Did Reuters report that Company X is under investigation?",
        description="Checking if Reuters published this claim",
        schema=report_schema,
        created_by="fact_checker",
        scope_entities=["Reuters", "Company X"],
        initial_stake=50.0,
    )

    # Inquiry 2: Is X actually under investigation?
    log("  Creating world-truth inquiry...")

    world_schema = InquirySchema(
        schema_type="boolean",
        rigor=RigorLevel.B  # Medium rigor - harder to verify
    )

    world_inquiry = engine.create_inquiry(
        title="Is Company X actually under federal investigation?",
        description="Verifying the underlying claim",
        schema=world_schema,
        created_by="fact_checker",
        scope_entities=["Company X", "DOJ", "FBI"],
        initial_stake=100.0,
    )

    # Add evidence to report-truth (easy to verify)
    log("\n  Adding evidence to report-truth inquiry...")

    await engine.add_contribution(
        inquiry_id=report_inquiry.id,
        user_id="verifier1",
        contribution_type=ContributionType.EVIDENCE,
        text="Found Reuters article dated Jan 15: 'Company X faces federal probe'",
        source_url="https://reuters.com/company-x-probe",
        extracted_value="true",
        observation_kind="point"
    )

    await engine.add_contribution(
        inquiry_id=report_inquiry.id,
        user_id="verifier2",
        contribution_type=ContributionType.EVIDENCE,
        text="Reuters wire confirms article published",
        source_url="https://reuters.com/wire/12345",
        extracted_value="true",
        observation_kind="point"
    )

    report_trace = engine.get_trace(report_inquiry.id)
    log(f"  Report-truth P(true): {report_trace['belief_state']['map_probability']*100:.1f}%")

    # Add evidence to world-truth (harder)
    log("\n  Adding evidence to world-truth inquiry...")

    await engine.add_contribution(
        inquiry_id=world_inquiry.id,
        user_id="analyst1",
        contribution_type=ContributionType.EVIDENCE,
        text="Reuters reports federal investigation",
        source_url="https://reuters.com/company-x-probe",
        extracted_value="true",
        observation_kind="point"
    )

    await engine.add_contribution(
        inquiry_id=world_inquiry.id,
        user_id="analyst2",
        contribution_type=ContributionType.EVIDENCE,
        text="Company X denies any investigation",
        source_url="https://companyx.com/press/denial",
        extracted_value="false",
        observation_kind="point"
    )

    await engine.add_contribution(
        inquiry_id=world_inquiry.id,
        user_id="analyst3",
        contribution_type=ContributionType.ATTRIBUTION,
        text="Reuters citing anonymous source only, no official confirmation",
        source_url="https://reuters.com/company-x-probe"
    )

    world_trace = engine.get_trace(world_inquiry.id)
    log(f"  World-truth P(true): {world_trace['belief_state']['map_probability']*100:.1f}%")

    # Compare
    section("Comparison")

    log(f"  Report-truth (did Reuters report it?):")
    log(f"    MAP: {report_trace['belief_state']['map']}")
    log(f"    P(MAP): {report_trace['belief_state']['map_probability']*100:.1f}%")
    log(f"    Status: {'RESOLVABLE' if report_trace['belief_state']['map_probability'] >= 0.95 else 'UNCERTAIN'}")

    log(f"\n  World-truth (is it actually true?):")
    log(f"    MAP: {world_trace['belief_state']['map']}")
    log(f"    P(MAP): {world_trace['belief_state']['map_probability']*100:.1f}%")
    log(f"    Status: {'RESOLVABLE' if world_trace['belief_state']['map_probability'] >= 0.95 else 'UNCERTAIN'}")

    log("\n  Key insight:")
    log("    Report-truth can be highly certain (article exists)")
    log("    World-truth remains uncertain (conflicting evidence)")

    return report_inquiry, world_inquiry


# =============================================================================
# MAIN
# =============================================================================

async def main():
    section("FORECAST INQUIRY EXPERIMENTS")

    log("  Running forecast experiments to validate:")
    log("    1. Indicator-based evidence")
    log("    2. Likelihood model for different signal types")
    log("    3. Deadline-based resolution")
    log("    4. Report-truth vs world-truth distinction")

    # Run GPT-5 forecast
    await run_gpt5_forecast_experiment()

    # Run report vs world truth
    await run_report_vs_world_experiment()

    section("EXPERIMENT SUMMARY")

    log("""
  FORECAST INQUIRY CAPABILITIES:

  ✓ Boolean forecast with deadline
  ✓ Indicator types (hiring, product, statement, leak)
  ✓ Likelihood ratios for signal strength
  ✓ Resolution task generation at deadline
  ✓ Explicit adjudication (not epistemic inference)

  ✓ Report-truth vs world-truth separation
  ✓ Different rigor levels for different certainty

  PENDING IMPLEMENTATION:

  ○ Automatic resolution checking (background job)
  ○ Resolution source verification (crawl check)
  ○ Calibration loop (track resolved forecasts for source reliability)
  ○ Composite forecasts (multiple deadlines)
    """)


if __name__ == "__main__":
    asyncio.run(main())
