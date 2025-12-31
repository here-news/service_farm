"""
Inquiry Engine
==============

Orchestrates REEE for user-facing inquiries.

This is the application layer that:
1. Scopes REEE to specific inquiries
2. Converts contributions to L0 claims
3. Runs typed belief inference per surface
4. Generates tasks from meta-claims
5. Tracks resolution criteria
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from openai import AsyncOpenAI

from .types import (
    Inquiry, InquirySchema, InquiryStatus, RigorLevel,
    Contribution, ContributionType,
    InquiryTask, TaskType
)
from ..types import Claim, Surface, MetaClaim
from ..typed_belief import (
    TypedBeliefState, CountDomain, CountDomainConfig,
    CategoricalDomain, Observation, ObservationKind, UniformNoise
)


class InquiryEngine:
    """
    Wraps REEE for inquiry workflow.

    Does NOT replace REEE - it uses REEE as the epistemic substrate.
    """

    def __init__(self, llm: 'AsyncOpenAI' = None):
        self.llm = llm

        # Inquiries by ID
        self.inquiries: Dict[str, Inquiry] = {}

        # Contributions by inquiry ID
        self.contributions: Dict[str, List[Contribution]] = {}

        # Tasks by inquiry ID
        self.tasks: Dict[str, List[InquiryTask]] = {}

        # Belief states by inquiry ID (one per inquiry)
        self.belief_states: Dict[str, TypedBeliefState] = {}

        # Claims created from contributions (for audit)
        self.claims: Dict[str, Claim] = {}

    # =========================================================================
    # INQUIRY LIFECYCLE
    # =========================================================================

    def create_inquiry(
        self,
        title: str,
        schema: InquirySchema,
        created_by: str,
        description: str = "",
        scope_entities: List[str] = None,
        scope_keywords: List[str] = None,
        initial_stake: float = 0.0
    ) -> Inquiry:
        """Create a new inquiry."""
        inquiry = Inquiry(
            title=title,
            description=description,
            schema=schema,
            created_by=created_by,
            scope_entities=set(scope_entities or []),
            scope_keywords=scope_keywords or [],
        )

        if initial_stake > 0:
            inquiry.stakes[created_by] = initial_stake
            inquiry.total_stake = initial_stake

        # Initialize belief state based on schema
        self.belief_states[inquiry.id] = self._create_belief_state(schema)

        # Initialize collections
        self.inquiries[inquiry.id] = inquiry
        self.contributions[inquiry.id] = []
        self.tasks[inquiry.id] = []

        return inquiry

    def _create_belief_state(self, schema: InquirySchema) -> TypedBeliefState:
        """Create typed belief state for schema."""
        if schema.schema_type == "monotone_count":
            scales = {
                "small": [("small", 10.0, 0.6), ("medium", 50.0, 0.3), ("large", 200.0, 0.1)],
                "medium": [("small", 20.0, 0.3), ("medium", 80.0, 0.4), ("large", 200.0, 0.3)],
                "large": [("small", 50.0, 0.2), ("medium", 150.0, 0.4), ("large", 500.0, 0.4)],
            }
            config = CountDomainConfig(
                max_count=schema.count_max,
                scales=scales.get(schema.count_scale, scales["medium"]),
                monotone=schema.count_monotone,
            )
            return TypedBeliefState(
                domain=CountDomain(config),
                noise_model=UniformNoise(delta=3.0)
            )

        elif schema.schema_type == "categorical":
            return TypedBeliefState(
                domain=CategoricalDomain(categories=schema.categories),
                noise_model=UniformNoise(delta=0.2)
            )

        elif schema.schema_type == "boolean":
            return TypedBeliefState(
                domain=CategoricalDomain(categories=["true", "false"]),
                noise_model=UniformNoise(delta=0.2)
            )

        else:
            # Custom/other: boolean fallback
            return TypedBeliefState(
                domain=CategoricalDomain(categories=["true", "false"]),
                noise_model=UniformNoise(delta=0.2)
            )

    def get_inquiry(self, inquiry_id: str) -> Optional[Inquiry]:
        """Get inquiry by ID."""
        return self.inquiries.get(inquiry_id)

    def list_inquiries(
        self,
        status: InquiryStatus = None,
        limit: int = 50,
        order_by: str = "stake"  # stake, created, updated
    ) -> List[Inquiry]:
        """List inquiries with optional filtering."""
        result = list(self.inquiries.values())

        if status:
            result = [i for i in result if i.status == status]

        if order_by == "stake":
            result.sort(key=lambda i: -i.total_stake)
        elif order_by == "created":
            result.sort(key=lambda i: i.created_at, reverse=True)
        elif order_by == "updated":
            result.sort(key=lambda i: i.updated_at, reverse=True)

        return result[:limit]

    # =========================================================================
    # CONTRIBUTIONS
    # =========================================================================

    async def add_contribution(
        self,
        inquiry_id: str,
        user_id: str,
        contribution_type: ContributionType,
        text: str,
        source_url: str = None,
        source_name: str = None,
        extracted_value: Any = None,
        observation_kind: str = "point"
    ) -> Contribution:
        """
        Add a user contribution to an inquiry.

        This:
        1. Creates the contribution record
        2. Creates L0 claim(s)
        3. Converts to observation
        4. Updates belief state
        5. Recomputes inquiry summary
        """
        inquiry = self.inquiries.get(inquiry_id)
        if not inquiry:
            raise ValueError(f"Inquiry {inquiry_id} not found")

        if inquiry.status != InquiryStatus.OPEN:
            raise ValueError(f"Inquiry {inquiry_id} is not open")

        # Create contribution
        contribution = Contribution(
            inquiry_id=inquiry_id,
            user_id=user_id,
            type=contribution_type,
            text=text,
            source_url=source_url,
            source_name=source_name or self._extract_domain(source_url),
            extracted_value=extracted_value,
            observation_kind=observation_kind,
        )

        # Create L0 claim
        claim = Claim(
            id=f"claim_{uuid.uuid4().hex[:8]}",
            text=text,
            source=contribution.source_name or "user_submission",
            page_id=source_url,
            timestamp=datetime.utcnow(),
        )
        self.claims[claim.id] = claim
        contribution.claim_ids.append(claim.id)

        # Convert to observation and update belief
        if extracted_value is not None:
            obs = self._make_observation(
                inquiry.schema,
                extracted_value,
                observation_kind,
                contribution.source_name
            )

            if obs.kind != ObservationKind.NONE:
                state = self.belief_states[inquiry_id]
                old_map_prob = state.map_probability

                state.add_observation(obs)

                # Compute impact
                contribution.posterior_impact = abs(state.map_probability - old_map_prob)

        contribution.processed = True
        self.contributions[inquiry_id].append(contribution)

        # Update inquiry summary
        self._update_inquiry_summary(inquiry)

        # Check for new tasks
        self._check_for_tasks(inquiry)

        return contribution

    def _make_observation(
        self,
        schema: InquirySchema,
        value: Any,
        kind: str,
        source: str
    ) -> Observation:
        """Convert extracted value to typed observation."""
        if schema.schema_type == "monotone_count":
            if kind == "lower_bound":
                return Observation.lower_bound(
                    int(value),
                    max_count=schema.count_max,
                    source=source
                )
            elif kind == "interval" and isinstance(value, (list, tuple)):
                return Observation.interval(
                    int(value[0]), int(value[1]),
                    source=source
                )
            elif kind == "approximate":
                return Observation.approximate(
                    int(value),
                    source=source
                )
            else:
                return Observation.point(
                    int(value),
                    source=source
                )

        elif schema.schema_type in ("categorical", "boolean"):
            val_str = str(value).lower()
            return Observation.point(val_str, source=source)

        else:
            return Observation.none()

    def _extract_domain(self, url: str) -> Optional[str]:
        """Extract domain from URL."""
        if not url:
            return None
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return None

    def _update_inquiry_summary(self, inquiry: Inquiry) -> None:
        """Update inquiry's belief summary from state."""
        state = self.belief_states.get(inquiry.id)
        if not state:
            return

        inquiry.posterior_map = state.map_value
        inquiry.posterior_probability = state.map_probability
        inquiry.entropy_bits = state.entropy()
        inquiry.normalized_entropy = state.normalized_entropy()
        inquiry.credible_interval = state.credible_interval(0.95)
        inquiry.contribution_count = len(self.contributions.get(inquiry.id, []))
        inquiry.updated_at = datetime.utcnow()

        # Track stability for resolution
        if inquiry.posterior_probability >= 0.95:
            if inquiry.stable_since is None:
                inquiry.stable_since = datetime.utcnow()
        else:
            inquiry.stable_since = None

        # Check resolution
        if inquiry.is_resolvable() and inquiry.status == InquiryStatus.OPEN:
            inquiry.status = InquiryStatus.RESOLVED
            inquiry.resolved_at = datetime.utcnow()

    # =========================================================================
    # TASKS
    # =========================================================================

    def _check_for_tasks(self, inquiry: Inquiry) -> None:
        """Generate tasks based on epistemic state."""
        state = self.belief_states.get(inquiry.id)
        if not state:
            return

        existing_types = {t.type for t in self.tasks.get(inquiry.id, []) if not t.completed}

        # Single source only
        if len(state.observations) == 1 and TaskType.SINGLE_SOURCE_ONLY not in existing_types:
            task = InquiryTask(
                inquiry_id=inquiry.id,
                type=TaskType.SINGLE_SOURCE_ONLY,
                description="Only one source. Add corroborating evidence from another source.",
                bounty=inquiry.total_stake * 0.1,
            )
            self.tasks[inquiry.id].append(task)
            inquiry.blocking_tasks.append(task.id)

        # High entropy
        if state.normalized_entropy() > 0.7 and TaskType.HIGH_ENTROPY not in existing_types:
            task = InquiryTask(
                inquiry_id=inquiry.id,
                type=TaskType.HIGH_ENTROPY,
                description="High uncertainty. Add definitive evidence to resolve.",
                bounty=inquiry.total_stake * 0.15,
            )
            self.tasks[inquiry.id].append(task)

        inquiry.open_tasks = len([t for t in self.tasks.get(inquiry.id, []) if not t.completed])

    def get_tasks(self, inquiry_id: str, include_completed: bool = False) -> List[InquiryTask]:
        """Get tasks for an inquiry."""
        tasks = self.tasks.get(inquiry_id, [])
        if not include_completed:
            tasks = [t for t in tasks if not t.completed]
        return tasks

    def complete_task(
        self,
        inquiry_id: str,
        task_id: str,
        completed_by: str,
        contribution_id: str
    ) -> Optional[InquiryTask]:
        """Mark a task as completed."""
        tasks = self.tasks.get(inquiry_id, [])
        for task in tasks:
            if task.id == task_id:
                task.completed = True
                task.completed_at = datetime.utcnow()

                # Remove from blocking
                inquiry = self.inquiries.get(inquiry_id)
                if inquiry and task.id in inquiry.blocking_tasks:
                    inquiry.blocking_tasks.remove(task.id)
                    self._update_inquiry_summary(inquiry)

                return task
        return None

    # =========================================================================
    # STAKES
    # =========================================================================

    def add_stake(self, inquiry_id: str, user_id: str, amount: float) -> float:
        """Add stake to an inquiry. Returns new total."""
        inquiry = self.inquiries.get(inquiry_id)
        if not inquiry:
            raise ValueError(f"Inquiry {inquiry_id} not found")

        inquiry.stakes[user_id] = inquiry.stakes.get(user_id, 0) + amount
        inquiry.total_stake = sum(inquiry.stakes.values())
        inquiry.updated_at = datetime.utcnow()

        return inquiry.total_stake

    # =========================================================================
    # EPISTEMIC TRACE
    # =========================================================================

    def get_trace(self, inquiry_id: str) -> Dict:
        """
        Get full epistemic trace for an inquiry.

        This is the audit trail showing exactly why the posterior is what it is.
        """
        inquiry = self.inquiries.get(inquiry_id)
        if not inquiry:
            return {}

        state = self.belief_states.get(inquiry_id)
        contributions = self.contributions.get(inquiry_id, [])
        tasks = self.tasks.get(inquiry_id, [])

        # Build trace
        trace = {
            "inquiry": inquiry.summary(),

            "observations": [
                {
                    "kind": obs.kind.value,
                    "value_distribution": dict(obs.value_distribution),
                    "source": obs.source,
                    "confidence": obs.extraction_confidence,
                }
                for obs in (state.observations if state else [])
            ],

            "contributions": [
                {
                    "id": c.id,
                    "type": c.type.value,
                    "text": c.text[:100],
                    "source": c.source_name,
                    "extracted_value": c.extracted_value,
                    "impact": round(c.posterior_impact, 3),
                    "created_at": c.created_at.isoformat(),
                }
                for c in contributions
            ],

            "belief_state": {
                "map": state.map_value if state else None,
                "map_probability": round(state.map_probability, 3) if state else 0,
                "entropy_bits": round(state.entropy(), 2) if state else 0,
                "normalized_entropy": round(state.normalized_entropy(), 3) if state else 0,
                "observation_count": len(state.observations) if state else 0,
                "log_scores": [round(s, 2) for s in state._log_scores] if state else [],
                "total_log_score": round(state.total_log_score(), 2) if state else 0,
            },

            "tasks": [
                {
                    "id": t.id,
                    "type": t.type.value,
                    "description": t.description,
                    "bounty": t.bounty,
                    "completed": t.completed,
                }
                for t in tasks
            ],

            "resolution": {
                "status": inquiry.status.value,
                "resolvable": inquiry.is_resolvable(),
                "stable_since": inquiry.stable_since.isoformat() if inquiry.stable_since else None,
                "blocking_tasks": inquiry.blocking_tasks,
            }
        }

        # Add posterior distribution for count domains
        if state and hasattr(state.domain, 'config'):
            posterior = state.compute_posterior()
            top_10 = sorted(posterior.items(), key=lambda x: -x[1])[:10]
            trace["posterior_top_10"] = [
                {"value": v, "probability": round(p, 4)}
                for v, p in top_10
            ]

        return trace
