"""
Emergence Scale Experiment
==========================

Tests REEE emergence at varying scales using real claims from the database.

Usage:
    docker exec herenews-app python -m reee.experiments.emergence_scale 20
"""

import asyncio
import sys

from reee import Engine, Parameters
from reee.experiments.loader import (
    create_context, load_multi_event_claims, log
)


async def run_experiment(claims_per_event: int = 10):
    """Run emergence experiment at scale."""
    ctx = await create_context()

    try:
        # Load claims from 3 events
        all_claims, claim_to_event, events = await load_multi_event_claims(
            ctx,
            claims_per_event=claims_per_event,
            num_events=3,
            min_claims=15
        )

        log('=' * 60)
        log(f'EMERGENCE TEST: {claims_per_event} claims/event')
        log('=' * 60)
        log('\nEvents:')
        for ev in events:
            log(f'  {ev.name}: {ev.claim_count} total claims')

        log(f'\nLoaded {len(all_claims)} claims total')

        # Use tuned parameters
        params = Parameters(
            aboutness_min_signals=1,
            aboutness_threshold=0.33,
            hub_max_df=5
        )

        engine = Engine(llm=ctx.llm, params=params)

        # Add claims and track progress
        log('\n--- Adding Claims (Identity Detection) ---')
        identity_count = 0
        for i, claim in enumerate(all_claims):
            result = await engine.add_claim(claim)
            if result['relations']:
                identity_count += len(result['relations'])
                for r in result['relations']:
                    ev1 = claim_to_event.get(claim.id, '?')[:12]
                    ev2 = claim_to_event.get(r['other_id'], '?')[:12]
                    cross = ' CROSS' if ev1 != ev2 else ''
                    log(f'  [{i+1}] {claim.id[:8]}...→{r["other_id"][:8]}...: {r["relation"]} ({r["confidence"]:.0%}) [{ev1}↔{ev2}]{cross}')

            if (i + 1) % 5 == 0:
                log(f'  ... processed {i+1}/{len(all_claims)} claims, {identity_count} identity edges')

        log(f'\nTotal identity edges: {identity_count}')

        # Compute surfaces
        surfaces = engine.compute_surfaces()
        log(f'\n--- Surfaces (L2): {len(surfaces)} ---')

        # Analyze surface composition
        surface_stats = {'pure': 0, 'mixed': 0}
        for s in surfaces:
            events_in = set(claim_to_event.get(cid, '?') for cid in s.claim_ids)
            if len(events_in) == 1:
                surface_stats['pure'] += 1
            else:
                surface_stats['mixed'] += 1
                log(f'  MIXED {s.id}: {len(s.claim_ids)} claims from {events_in}')

        log(f'\nSurface purity: {surface_stats["pure"]}/{len(surfaces)} ({surface_stats["pure"]/len(surfaces)*100:.0f}%)')

        # Show surface distribution by event
        event_surfaces = {}
        for s in surfaces:
            events_in = list(set(claim_to_event.get(cid, '?') for cid in s.claim_ids))
            ev = events_in[0] if len(events_in) == 1 else 'mixed'
            event_surfaces[ev] = event_surfaces.get(ev, 0) + 1

        log('\nSurfaces per event:')
        for ev, cnt in sorted(event_surfaces.items()):
            log(f'  {ev}: {cnt} surfaces')

        # Compute aboutness
        aboutness = engine.compute_surface_aboutness()
        log(f'\n--- Aboutness Edges: {len(aboutness)} ---')

        # Analyze aboutness edges
        within_event = 0
        cross_event = 0
        for s1, s2, score, _ in aboutness:
            ev1 = list(set(claim_to_event.get(cid, '?') for cid in engine.surfaces[s1].claim_ids))[0]
            ev2 = list(set(claim_to_event.get(cid, '?') for cid in engine.surfaces[s2].claim_ids))[0]
            if ev1 == ev2:
                within_event += 1
            else:
                cross_event += 1
                log(f'  CROSS: {s1}↔{s2} ({score:.2f}) [{ev1[:12]}↔{ev2[:12]}]')

        log(f'\nWithin-event edges: {within_event}')
        log(f'Cross-event edges: {cross_event}')

        # Compute events
        events_out = engine.compute_events()
        log(f'\n--- Events (L3): {len(events_out)} ---')

        # Analyze events
        event_pure = 0
        event_mixed = 0
        for e in events_out:
            gt_events = set()
            total_claims = 0
            for sid in e.surface_ids:
                s = engine.surfaces[sid]
                total_claims += len(s.claim_ids)
                for cid in s.claim_ids:
                    gt_events.add(claim_to_event.get(cid, '?'))

            if len(gt_events) == 1:
                event_pure += 1
                status = 'PURE'
            else:
                event_mixed += 1
                status = 'MIXED'

            log(f'  {e.id}: {len(e.surface_ids)} surfaces, {total_claims} claims → {[g[:15] for g in gt_events]} [{status}]')

        # Summary
        log('\n' + '=' * 60)
        log('SUMMARY')
        log('=' * 60)
        log(f'Claims: {len(all_claims)} ({claims_per_event}/event × 3 events)')
        log(f'Identity edges: {identity_count}')
        log(f'Surfaces: {len(surfaces)} (purity: {surface_stats["pure"]}/{len(surfaces)})')
        log(f'Aboutness edges: {len(aboutness)} (within: {within_event}, cross: {cross_event})')
        log(f'Events: {len(events_out)} vs GT=3 (purity: {event_pure}/{len(events_out)})')
        log(f'Fragmentation: {len(events_out)/3:.2f}x')

        # Detect tensions
        meta_claims = engine.detect_tensions()
        log(f'\nMeta-claims: {len(meta_claims)}')
        tension_types = {}
        for mc in meta_claims:
            tension_types[mc.type] = tension_types.get(mc.type, 0) + 1
        for t, cnt in sorted(tension_types.items(), key=lambda x: -x[1]):
            log(f'  {t}: {cnt}')

    finally:
        await ctx.close()


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    asyncio.run(run_experiment(n))
