"""
Belief State Demo: Jaynes on Real Claims
=========================================

Demonstrates the new BeliefState model on real claims from the database.

Usage:
    docker exec herenews-app python -m reee.experiments.belief_state_demo
"""

import asyncio
import os
from collections import defaultdict

import asyncpg
from pgvector.asyncpg import register_vector

from reee import Engine, Claim, Parameters, Relation
from reee.belief_state import BeliefState
from reee.experiments.loader import create_context, load_claims_for_event, load_events, log


async def demo_belief_states():
    """Run belief state demo on real claims."""
    ctx = await create_context()

    try:
        # Load events with substantial claims
        events = await load_events(ctx, min_claims=20, limit=2)

        if not events:
            log("No events found with enough claims")
            return

        log("=" * 70)
        log("BELIEF STATE DEMO: Jaynes on Real Claims")
        log("=" * 70)

        for event in events:
            log(f"\n{'='*70}")
            log(f"EVENT: {event.name}")
            log(f"{'='*70}")

            # Load claims for this event
            claims, _ = await load_claims_for_event(ctx, event.id, limit=30)
            log(f"Loaded {len(claims)} claims\n")

            # Run REEE engine to get surfaces
            engine = Engine(params=Parameters(aboutness_min_signals=1))

            for claim in claims:
                await engine.add_claim(claim)

            surfaces = engine.compute_surfaces()
            log(f"Surfaces formed: {len(surfaces)}")

            # For each surface, build a belief state
            for surface in surfaces[:5]:  # Show first 5 surfaces
                log(f"\n--- Surface {surface.id} ({len(surface.claim_ids)} claims) ---")

                # Get the claims in this surface
                surface_claims = [engine.claims[cid] for cid in surface.claim_ids if cid in engine.claims]

                if not surface_claims:
                    continue

                # Show claims
                for c in surface_claims[:3]:
                    log(f"  [{c.source}] {c.text[:80]}...")

                if len(surface_claims) > 3:
                    log(f"  ... and {len(surface_claims) - 3} more")

                # Build belief state
                # For demo, use claim text as "value" (in real use, extract numeric/categorical values)
                state = BeliefState()

                # Group by semantic similarity to find "values"
                # For now, just use sources as observations of "this surface's proposition"
                for claim in surface_claims:
                    # Use a simplified value - first 50 chars as proxy
                    value = claim.text[:50] if claim.text else "unknown"
                    state.add_observation(
                        value=value,
                        source=claim.source,
                        claim_id=claim.id
                    )

                # Show belief state summary
                summary = state.summary()
                log(f"\n  Belief State:")
                log(f"    Sources: {summary['n_sources']}")
                log(f"    Values: {summary['n_values']}")
                log(f"    Entropy: {summary['entropy_bits']:.2f} bits")
                log(f"    Normalized Entropy: {summary['normalized_entropy']:.2f}")
                log(f"    Confidence: {summary['confidence']}")
                log(f"    Has Conflict: {summary['has_conflict']}")

            log("\n")

        # Now demo with a synthetic numeric scenario
        log("=" * 70)
        log("SYNTHETIC DEMO: Death Toll Scenario")
        log("=" * 70)

        state = BeliefState()

        log("\nStep 1: Initial report - '2 killed'")
        state.add_observation(value=2, source="reuters.com", claim_id="c1")
        log(f"  MAP: {state.map_value}, Entropy: {state.entropy():.3f} bits")

        log("\nStep 2: Corroboration - '2 killed' from another source")
        state.add_observation(value=2, source="bbc.com", claim_id="c2")
        log(f"  MAP: {state.map_value}, Entropy: {state.entropy():.3f} bits")

        log("\nStep 3: Update - 'death toll rises to 3'")
        state.add_observation(
            value=3,
            source="afp.com",
            claim_id="c3",
            is_update=True,
            relation_to_existing="supersedes",
            related_value=2
        )
        posterior = state.compute_posterior()
        post_str = ", ".join(f"{k}: {v:.2f}" for k, v in sorted(posterior.items()))
        log(f"  Posterior: {{{post_str}}}")
        log(f"  MAP: {state.map_value}, Entropy: {state.entropy():.3f} bits")

        log("\nStep 4: Conflict - '4 killed' from outlier source")
        state.add_observation(
            value=4,
            source="tabloid.com",
            claim_id="c4",
            relation_to_existing="conflicts",
            related_value=3
        )
        posterior = state.compute_posterior()
        post_str = ", ".join(f"{k}: {v:.2f}" for k, v in sorted(posterior.items()))
        log(f"  Posterior: {{{post_str}}}")
        log(f"  MAP: {state.map_value}, Entropy: {state.entropy():.3f} bits")
        log(f"  Confidence: {state.confidence_level()}")

        log("\nStep 5: More corroboration for '3'")
        state.add_observation(value=3, source="ap.com", claim_id="c5")
        state.add_observation(value=3, source="guardian.com", claim_id="c6")
        posterior = state.compute_posterior()
        post_str = ", ".join(f"{k}: {v:.2f}" for k, v in sorted(posterior.items()))
        log(f"  Posterior: {{{post_str}}}")
        log(f"  MAP: {state.map_value}, Entropy: {state.entropy():.3f} bits")
        log(f"  Confidence: {state.confidence_level()}")
        log(f"  95% Credible Set: {state.credible_set(0.95)}")

        log("\n" + "=" * 70)
        log("DEMO COMPLETE")
        log("=" * 70)

    finally:
        await ctx.close()


if __name__ == "__main__":
    asyncio.run(demo_belief_states())
