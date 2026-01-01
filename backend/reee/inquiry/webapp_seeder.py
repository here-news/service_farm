"""
Webapp Inquiry Seeder: Bridge ProtoInquiry → Webapp Inquiry
============================================================

This module bridges REEE's automatic inquiry emergence with the webapp's
inquiry system. It converts ProtoInquiry objects (system-generated from
weaving) into webapp Inquiry objects that can be persisted and displayed.

Key Responsibilities:
1. Convert ProtoInquiry → Inquiry (type mapping, field mapping)
2. Create InquiryTasks from meta-claims/tensions
3. Initialize belief state from proto-inquiry observations
4. Optionally persist to database via InquiryRepository

Usage:
    seeder = WebappInquirySeeder(inquiry_repo)
    results = await seeder.seed_from_protos(proto_inquiries)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
import uuid
import asyncpg

from .seeder import ProtoInquiry, ProtoInquiryType, SchemaType as ProtoSchemaType

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE MAPPINGS
# =============================================================================

# Map REEE schema types to webapp schema types
SCHEMA_TYPE_MAP = {
    ProtoSchemaType.MONOTONE_COUNT: "monotone_count",
    ProtoSchemaType.CATEGORICAL: "categorical",
    ProtoSchemaType.BOOLEAN: "boolean",
    ProtoSchemaType.REPORT_TRUTH: "report_truth",
    ProtoSchemaType.QUOTE_AUTHENTICITY: "quote_authenticity",
}

# Map proto-inquiry types to task types
TENSION_TO_TASK_TYPE = {
    "typed_value_conflict": "unresolved_conflict",
    "unresolved_conflict": "unresolved_conflict",
    "single_source_only": "single_source_only",
    "high_entropy_surface": "high_entropy",
    "needs_primary_source": "need_primary_source",
}

# Task descriptions by type
TASK_DESCRIPTIONS = {
    "unresolved_conflict": "Resolve conflicting values from different sources",
    "single_source_only": "Find additional sources to corroborate this claim",
    "high_entropy": "Reduce uncertainty by finding authoritative sources",
    "need_primary_source": "Find primary source documentation",
}


# =============================================================================
# SEEDING RESULT
# =============================================================================

@dataclass
class SeedingResult:
    """Result of seeding operation."""
    proto_id: str
    inquiry_id: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    tasks_created: int = 0

    # Provenance
    surface_id: Optional[str] = None
    meta_claim_ids: List[str] = field(default_factory=list)


@dataclass
class BatchSeedingResult:
    """Result of batch seeding operation."""
    total_protos: int = 0
    seeded: int = 0
    failed: int = 0
    skipped: int = 0
    results: List[SeedingResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_protos == 0:
            return 0.0
        return self.seeded / self.total_protos


# =============================================================================
# WEBAPP INQUIRY SEEDER
# =============================================================================

class WebappInquirySeeder:
    """
    Seeds webapp inquiries from REEE proto-inquiries.

    This is the bridge between automatic inquiry emergence (REEE weaver)
    and user-facing inquiries (webapp).

    Modes:
    1. Preview mode (dry_run=True): Returns what would be created without persisting
    2. Persist mode (dry_run=False): Actually creates inquiries in database
    """

    def __init__(
        self,
        db_pool: Optional[asyncpg.Pool] = None,
        min_priority_score: float = 30.0,
        default_rigor_level: str = "B",
        auto_create_tasks: bool = True,
    ):
        """
        Initialize seeder.

        Args:
            db_pool: PostgreSQL connection pool (None for preview mode)
            min_priority_score: Minimum score to seed (filter low-quality)
            default_rigor_level: Default rigor badge (A/B/C)
            auto_create_tasks: Whether to create tasks from tensions
        """
        self.db_pool = db_pool
        self.min_priority_score = min_priority_score
        self.default_rigor_level = default_rigor_level
        self.auto_create_tasks = auto_create_tasks

        # Track seeded to avoid duplicates
        self._seeded_proto_ids: set = set()

    async def seed_from_protos(
        self,
        proto_inquiries: List[ProtoInquiry],
        dry_run: bool = False,
        created_by: str = "system",
    ) -> BatchSeedingResult:
        """
        Seed webapp inquiries from a batch of proto-inquiries.

        Args:
            proto_inquiries: List of proto-inquiries from REEE
            dry_run: If True, preview only (no database writes)
            created_by: User ID for attribution (default: system)

        Returns:
            BatchSeedingResult with details of what was seeded
        """
        result = BatchSeedingResult(total_protos=len(proto_inquiries))

        for proto in proto_inquiries:
            # Skip if already seeded
            if proto.id in self._seeded_proto_ids:
                result.skipped += 1
                continue

            # Skip if below priority threshold
            priority = proto.priority_score()
            if priority < self.min_priority_score:
                result.skipped += 1
                result.results.append(SeedingResult(
                    proto_id=proto.id,
                    success=False,
                    error=f"Priority score {priority:.1f} below threshold {self.min_priority_score}"
                ))
                continue

            # Convert and seed
            try:
                seed_result = await self._seed_single(
                    proto,
                    dry_run=dry_run,
                    created_by=created_by
                )
                result.results.append(seed_result)

                if seed_result.success:
                    result.seeded += 1
                    self._seeded_proto_ids.add(proto.id)
                else:
                    result.failed += 1

            except Exception as e:
                logger.error(f"Error seeding proto {proto.id}: {e}")
                result.failed += 1
                result.results.append(SeedingResult(
                    proto_id=proto.id,
                    success=False,
                    error=str(e)
                ))

        return result

    async def _seed_single(
        self,
        proto: ProtoInquiry,
        dry_run: bool,
        created_by: str,
    ) -> SeedingResult:
        """Seed a single proto-inquiry."""
        # Build inquiry data
        inquiry_data = self._proto_to_inquiry_data(proto)

        if dry_run:
            # Preview mode - return what would be created
            return SeedingResult(
                proto_id=proto.id,
                inquiry_id=f"inq_preview_{uuid.uuid4().hex[:8]}",
                success=True,
                surface_id=proto.surface_id,
                meta_claim_ids=proto.meta_claim_ids,
                tasks_created=len(self._extract_tasks(proto)),
            )

        if not self.db_pool:
            return SeedingResult(
                proto_id=proto.id,
                success=False,
                error="No database pool configured"
            )

        # Persist to database
        async with self.db_pool.acquire() as conn:
            # Create inquiry
            inquiry_id = f"inq_{uuid.uuid4().hex[:8]}"

            await conn.execute("""
                INSERT INTO inquiries (
                    id, title, description, status, rigor_level,
                    schema_type, schema_config, scope_entities, scope_keywords,
                    scope_time_start, scope_time_end,
                    posterior_map, posterior_prob, entropy_bits, normalized_entropy,
                    created_by, created_at, updated_at
                )
                VALUES ($1, $2, $3, 'open', $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, now(), now())
            """,
                inquiry_id,
                inquiry_data["title"],
                inquiry_data["description"],
                inquiry_data["rigor_level"],
                inquiry_data["schema_type"],
                inquiry_data["schema_config"],
                inquiry_data["scope_entities"],
                inquiry_data["scope_keywords"],
                inquiry_data["scope_time_start"],
                inquiry_data["scope_time_end"],
                inquiry_data["posterior_map"],
                inquiry_data["posterior_prob"],
                inquiry_data["entropy_bits"],
                inquiry_data["normalized_entropy"],
                created_by,
            )

            # Create tasks from tensions
            tasks_created = 0
            if self.auto_create_tasks:
                tasks = self._extract_tasks(proto)
                for task in tasks:
                    task_id = f"task_{uuid.uuid4().hex[:8]}"
                    await conn.execute("""
                        INSERT INTO inquiry_tasks (
                            id, inquiry_id, task_type, description, bounty, meta_claim_id
                        )
                        VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                        task_id,
                        inquiry_id,
                        task["type"],
                        task["description"],
                        task.get("bounty", 0),
                        task.get("meta_claim_id"),
                    )
                    tasks_created += 1

            logger.info(f"Seeded inquiry {inquiry_id} from proto {proto.id}")

            return SeedingResult(
                proto_id=proto.id,
                inquiry_id=inquiry_id,
                success=True,
                surface_id=proto.surface_id,
                meta_claim_ids=proto.meta_claim_ids,
                tasks_created=tasks_created,
            )

    def _proto_to_inquiry_data(self, proto: ProtoInquiry) -> Dict[str, Any]:
        """Convert proto-inquiry to inquiry creation data."""
        import json

        # Map schema type
        schema_type = SCHEMA_TYPE_MAP.get(
            proto.schema_type,
            proto.schema_type.value if hasattr(proto.schema_type, 'value') else str(proto.schema_type)
        )

        # Build schema config
        schema_config = {
            "schema_type": schema_type,
        }

        if proto.schema_type == ProtoSchemaType.MONOTONE_COUNT:
            schema_config["count_max"] = 500
            schema_config["count_monotone"] = True
        elif proto.schema_type == ProtoSchemaType.CATEGORICAL and proto.hypotheses:
            schema_config["categories"] = [h["value"] for h in proto.hypotheses if "value" in h]

        # Build description
        description = self._build_description(proto)

        return {
            "title": proto.question_text,
            "description": description,
            "rigor_level": self.default_rigor_level,
            "schema_type": schema_type,
            "schema_config": json.dumps(schema_config),
            "scope_entities": list(proto.scope.anchor_entities)[:10],
            "scope_keywords": proto.scope.keywords[:10] if proto.scope.keywords else [],
            "scope_time_start": proto.scope.time_start,
            "scope_time_end": proto.scope.time_end,
            "posterior_map": json.dumps(proto.posterior_map) if proto.posterior_map else None,
            "posterior_prob": proto.posterior_probability,
            "entropy_bits": proto.entropy_bits,
            "normalized_entropy": proto.normalized_entropy,
        }

    def _build_description(self, proto: ProtoInquiry) -> str:
        """Build inquiry description from proto."""
        parts = []

        # Add inquiry type context
        type_desc = {
            ProtoInquiryType.VALUE_RESOLUTION: "Seeking to determine the value of a typed variable.",
            ProtoInquiryType.CONFLICT_RESOLUTION: "Multiple sources report conflicting values.",
            ProtoInquiryType.CORROBORATION: "Single source claim needs corroboration.",
            ProtoInquiryType.SCOPE_CLARIFICATION: "Clarifying whether events are the same or distinct.",
            ProtoInquiryType.TRUTH_VERIFICATION: "Verifying whether a reported claim is accurate.",
        }
        parts.append(type_desc.get(proto.inquiry_type, ""))

        # Add conflict info if present
        if len(proto.reported_values) > 1 and len(set(proto.reported_values)) > 1:
            unique_values = sorted(set(proto.reported_values))
            parts.append(f"Reported values: {', '.join(str(v) for v in unique_values)}")

        # Add source info
        if proto.source_count > 0:
            parts.append(f"Based on {proto.typed_observation_count} typed observations from {proto.source_count} sources.")

        return " ".join(filter(None, parts))

    def _extract_tasks(self, proto: ProtoInquiry) -> List[Dict]:
        """Extract tasks from proto-inquiry tensions."""
        tasks = []
        seen_types = set()

        for i, tension in enumerate(proto.tensions):
            # Parse tension type
            tension_type = tension.split(":")[0].strip() if ":" in tension else tension

            # Map to task type
            task_type = TENSION_TO_TASK_TYPE.get(tension_type)
            if not task_type or task_type in seen_types:
                continue

            seen_types.add(task_type)

            tasks.append({
                "type": task_type,
                "description": TASK_DESCRIPTIONS.get(task_type, tension),
                "bounty": 0,  # System-seeded tasks have no bounty initially
                "meta_claim_id": proto.meta_claim_ids[i] if i < len(proto.meta_claim_ids) else None,
            })

        return tasks

    def preview_inquiry(self, proto: ProtoInquiry) -> Dict:
        """
        Preview what an inquiry would look like without persisting.

        Useful for testing and debugging the emergence pipeline.
        """
        inquiry_data = self._proto_to_inquiry_data(proto)
        tasks = self._extract_tasks(proto)

        return {
            "proto_id": proto.id,
            "inquiry": {
                **inquiry_data,
                "id": f"inq_preview_{proto.id[:8]}",
                "status": "open",
            },
            "tasks": tasks,
            "provenance": {
                "surface_id": proto.surface_id,
                "meta_claim_ids": proto.meta_claim_ids,
                "priority_score": proto.priority_score(),
                "reported_values": proto.reported_values,
                "typed_observation_count": proto.typed_observation_count,
            }
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def preview_proto_as_inquiry(proto: ProtoInquiry) -> Dict:
    """Preview a proto-inquiry as a webapp inquiry without database."""
    seeder = WebappInquirySeeder()
    return seeder.preview_inquiry(proto)


async def seed_protos_to_webapp(
    proto_inquiries: List[ProtoInquiry],
    db_pool: asyncpg.Pool,
    min_priority: float = 30.0,
    dry_run: bool = False,
) -> BatchSeedingResult:
    """
    Seed proto-inquiries to webapp database.

    Convenience function for the common case.

    Args:
        proto_inquiries: Proto-inquiries from REEE weaver
        db_pool: Database connection pool
        min_priority: Minimum priority score to seed
        dry_run: If True, preview only

    Returns:
        BatchSeedingResult with seeding outcomes
    """
    seeder = WebappInquirySeeder(
        db_pool=db_pool,
        min_priority_score=min_priority,
    )
    return await seeder.seed_from_protos(proto_inquiries, dry_run=dry_run)
