"""
ShadowRunner - Run kernel in shadow mode alongside legacy weaver.

This is the Phase 3/4 validation tool:
1. Loads claim from DB → ClaimEvidence
2. Loads snapshot → PartitionSnapshot
3. Runs kernel → TopologyDelta
4. Compares with legacy weaver output
5. Logs differences (does NOT persist kernel output)

Once confidence is high, switch to kernel output.
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from services.neo4j_service import Neo4jService

from ..contracts.evidence import ClaimEvidence
from ..contracts.state import PartitionSnapshot
from ..contracts.delta import TopologyDelta
from ..topo import TopologyKernel, KernelParams, compute_scope_id

from .snapshot_loader import SnapshotLoader
from .evidence_builder import EvidenceBuilder
from .delta_applier import ShadowDeltaApplier

logger = logging.getLogger(__name__)


class ShadowRunner:
    """Run kernel in shadow mode for validation.

    Usage:
        runner = ShadowRunner(neo4j)
        results = await runner.validate_claim(claim_id)
        summary = runner.get_summary()
    """

    def __init__(
        self,
        neo4j: Neo4jService,
        kernel_params: KernelParams = KernelParams(),
    ):
        self.neo4j = neo4j
        self.kernel = TopologyKernel(kernel_params)
        self.snapshot_loader = SnapshotLoader(neo4j)
        self.evidence_builder = EvidenceBuilder(neo4j)
        self.shadow_applier = ShadowDeltaApplier(neo4j)

        # Validation stats
        self.processed = 0
        self.errors = 0
        self.results: List[Dict[str, Any]] = []

    async def validate_claim(
        self,
        claim_id: str,
        legacy_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Validate kernel output for a single claim.

        Args:
            claim_id: Claim to process
            legacy_result: Optional legacy weaver output for comparison

        Returns:
            Validation result including delta and diff
        """
        try:
            # 1. Build evidence
            evidence = await self.evidence_builder.build_from_claim_id(claim_id)
            if not evidence:
                return {"error": f"Claim not found: {claim_id}"}

            # 2. Compute scope
            scope_id = compute_scope_id(evidence.anchors)

            # 3. Load snapshot
            snapshot = await self.snapshot_loader.load_for_claim(
                scope_id=scope_id,
                surface_time=evidence.time,
                question_key=evidence.question_key,
            )

            # 4. Run kernel
            delta = self.kernel.process_evidence(snapshot, evidence)

            # 5. Compare with legacy (if provided)
            diff = None
            if legacy_result:
                diff = await self.shadow_applier.apply_shadow(delta, legacy_result)

            result = {
                "claim_id": claim_id,
                "scope_id": scope_id,
                "delta_summary": delta.to_summary(),
                "surfaces_created": len([
                    t for t in delta.decision_traces
                    if t.decision_type == "surface_key"
                ]),
                "incident_routing": [
                    {"outcome": t.outcome, "target": t.target_id}
                    for t in delta.decision_traces
                    if t.decision_type == "incident_membership"
                ],
                "signals": [
                    {"type": s.signal_type.value, "severity": s.severity.value}
                    for s in delta.signals
                ],
                "diff": diff,
                "timestamp": datetime.utcnow().isoformat(),
            }

            self.processed += 1
            self.results.append(result)

            return result

        except Exception as e:
            self.errors += 1
            logger.error(f"Error validating claim {claim_id}: {e}")
            return {"error": str(e), "claim_id": claim_id}

    async def validate_batch(
        self,
        claim_ids: List[str],
        legacy_results: Optional[Dict[str, Dict]] = None,
    ) -> List[Dict[str, Any]]:
        """Validate kernel output for multiple claims.

        Args:
            claim_ids: Claims to process
            legacy_results: Optional dict of claim_id -> legacy result

        Returns:
            List of validation results
        """
        results = []
        for claim_id in claim_ids:
            legacy = legacy_results.get(claim_id) if legacy_results else None
            result = await self.validate_claim(claim_id, legacy)
            results.append(result)

        return results

    async def validate_page(
        self,
        page_id: str,
        legacy_results: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Any]:
        """Validate kernel output for all claims on a page.

        Args:
            page_id: Page to process
            legacy_results: Optional dict of claim_id -> legacy result

        Returns:
            Page-level validation summary
        """
        # Get all claims for page
        claims = await self.evidence_builder.build_for_page(page_id)

        if not claims:
            return {"error": f"No claims found for page: {page_id}"}

        results = []
        for evidence in claims:
            # Compute scope
            scope_id = compute_scope_id(evidence.anchors)

            # Load snapshot
            snapshot = await self.snapshot_loader.load_for_claim(
                scope_id=scope_id,
                surface_time=evidence.time,
            )

            # Run kernel
            delta = self.kernel.process_evidence(snapshot, evidence)

            # Compare if legacy available
            diff = None
            if legacy_results and evidence.claim_id in legacy_results:
                diff = await self.shadow_applier.apply_shadow(
                    delta, legacy_results[evidence.claim_id]
                )

            results.append({
                "claim_id": evidence.claim_id,
                "scope_id": scope_id,
                "delta_summary": delta.to_summary(),
                "diff": diff,
            })

        return {
            "page_id": page_id,
            "claims_processed": len(results),
            "results": results,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get validation run summary."""
        diff_summary = self.shadow_applier.get_diff_summary()

        return {
            "processed": self.processed,
            "errors": self.errors,
            "success_rate": (self.processed - self.errors) / self.processed if self.processed > 0 else 0.0,
            "diff_summary": diff_summary,
        }


async def run_shadow_validation(
    neo4j: Neo4jService,
    claim_ids: List[str],
    limit: int = 100,
) -> Dict[str, Any]:
    """Convenience function to run shadow validation.

    Args:
        neo4j: Neo4j service
        claim_ids: Claims to validate
        limit: Max claims to process

    Returns:
        Validation summary
    """
    runner = ShadowRunner(neo4j)

    # Process limited batch
    batch = claim_ids[:limit]
    await runner.validate_batch(batch)

    return runner.get_summary()
