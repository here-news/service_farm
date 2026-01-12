#!/usr/bin/env python3
"""
Kernel Validation Script - Run kernel against real claims.

Usage:
    python -m reee.tests.scripts.validate_kernel --limit 50
    python -m reee.tests.scripts.validate_kernel --claim-id cl_xxx
    python -m reee.tests.scripts.validate_kernel --page-id pg_xxx
"""

import asyncio
import argparse
import logging
import json
from datetime import datetime
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


async def get_sample_claims(neo4j, limit: int = 50) -> List[str]:
    """Get sample claim IDs for validation."""
    results = await neo4j._execute_read("""
        MATCH (c:Claim)
        WHERE c.text IS NOT NULL
        RETURN c.id as id
        ORDER BY c.created_at DESC
        LIMIT $limit
    """, {'limit': limit})
    return [r['id'] for r in results]


async def validate_single_claim(
    claim_id: str,
    neo4j,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Validate kernel output for a single claim."""
    from reee.weaver import EvidenceBuilder, SnapshotLoader
    from reee.topo import TopologyKernel, compute_scope_id
    from reee.explain import format_trace, TraceStyle

    evidence_builder = EvidenceBuilder(neo4j)
    snapshot_loader = SnapshotLoader(neo4j)
    kernel = TopologyKernel()

    # Build evidence
    evidence = await evidence_builder.build_from_claim_id(claim_id)
    if not evidence:
        return {"error": f"Claim not found: {claim_id}"}

    # Compute scope
    scope_id = compute_scope_id(evidence.anchors)

    # Load snapshot
    snapshot = await snapshot_loader.load_for_claim(
        scope_id=scope_id,
        surface_time=evidence.time,
    )

    # Run kernel
    delta = kernel.process_evidence(snapshot, evidence)

    # Format results
    result = {
        "claim_id": claim_id,
        "text_preview": evidence.text[:100] + "..." if len(evidence.text) > 100 else evidence.text,
        "scope_id": scope_id,
        "entities": list(evidence.entities)[:5],
        "anchors": list(evidence.anchors),
        "surfaces_in_snapshot": len(snapshot.surfaces),
        "incidents_in_snapshot": len(snapshot.incidents),
        "delta": delta.to_summary(),
    }

    # Add trace details if verbose
    if verbose:
        result["traces"] = []
        for trace in delta.decision_traces:
            result["traces"].append({
                "type": trace.decision_type,
                "outcome": trace.outcome,
                "rules": list(trace.rules_fired),
                "formatted": format_trace(trace, TraceStyle.SHORT),
            })

        result["signals"] = [
            {"type": s.signal_type.value, "hint": s.resolution_hint}
            for s in delta.signals
        ]

    return result


async def run_validation(
    neo4j,
    claim_ids: List[str] = None,
    page_id: str = None,
    limit: int = 50,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run kernel validation."""
    from reee.weaver import EvidenceBuilder

    # Get claims to validate
    if claim_ids:
        pass  # Use provided
    elif page_id:
        evidence_builder = EvidenceBuilder(neo4j)
        evidences = await evidence_builder.build_for_page(page_id)
        claim_ids = [e.claim_id for e in evidences]
        logger.info(f"Found {len(claim_ids)} claims on page {page_id}")
    else:
        claim_ids = await get_sample_claims(neo4j, limit)
        logger.info(f"Loaded {len(claim_ids)} sample claims")

    # Validate each claim
    results = []
    errors = 0

    for i, claim_id in enumerate(claim_ids):
        try:
            result = await validate_single_claim(claim_id, neo4j, verbose)
            results.append(result)

            if "error" in result:
                errors += 1
                logger.warning(f"[{i+1}/{len(claim_ids)}] Error: {result['error']}")
            else:
                # Summary log
                delta = result["delta"]
                logger.info(
                    f"[{i+1}/{len(claim_ids)}] {claim_id}: "
                    f"scope={result['scope_id'][:20]}... "
                    f"surfaces={delta['surfaces_upserted']} "
                    f"incidents={delta['incidents_upserted']} "
                    f"signals={delta['signals']}"
                )
        except Exception as e:
            errors += 1
            logger.error(f"[{i+1}/{len(claim_ids)}] Exception for {claim_id}: {e}")
            results.append({"claim_id": claim_id, "error": str(e)})

    # Aggregate stats
    successful = [r for r in results if "error" not in r]

    # Count outcomes
    surface_outcomes = {}
    routing_outcomes = {}
    signal_types = {}

    for r in successful:
        delta = r.get("delta", {})
        for sig_type in delta.get("signal_types", []):
            signal_types[sig_type] = signal_types.get(sig_type, 0) + 1

    summary = {
        "total": len(claim_ids),
        "successful": len(successful),
        "errors": errors,
        "signal_types": signal_types,
        "timestamp": datetime.utcnow().isoformat(),
    }

    return {
        "summary": summary,
        "results": results if verbose else results[:10],  # Limit output unless verbose
    }


async def main():
    parser = argparse.ArgumentParser(description="Validate kernel against real claims")
    parser.add_argument("--limit", type=int, default=50, help="Number of claims to validate")
    parser.add_argument("--claim-id", type=str, help="Specific claim ID to validate")
    parser.add_argument("--page-id", type=str, help="Validate all claims from a page")
    parser.add_argument("--verbose", "-v", action="store_true", help="Include trace details")
    parser.add_argument("--output", "-o", type=str, help="Output file (JSON)")
    args = parser.parse_args()

    # Initialize services
    import os
    import asyncpg
    from services.neo4j_service import Neo4jService

    neo4j = Neo4jService(
        uri=os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", ""),
    )
    await neo4j.connect()

    try:
        # Run validation
        claim_ids = [args.claim_id] if args.claim_id else None

        result = await run_validation(
            neo4j=neo4j,
            claim_ids=claim_ids,
            page_id=args.page_id,
            limit=args.limit,
            verbose=args.verbose,
        )

        # Output
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Results written to {args.output}")
        else:
            print("\n" + "="*60)
            print("VALIDATION SUMMARY")
            print("="*60)
            print(json.dumps(result["summary"], indent=2))

            if args.verbose and result.get("results"):
                print("\n" + "="*60)
                print("SAMPLE RESULTS")
                print("="*60)
                for r in result["results"][:5]:
                    print(json.dumps(r, indent=2, default=str))
                    print("-"*40)

    finally:
        await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
