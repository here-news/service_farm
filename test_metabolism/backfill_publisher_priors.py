"""
Backfill Publisher Source Priors

One-time script to populate source_type and base_prior
on existing publisher entities using Wikidata P31 when available.
"""
import asyncio
import sys
import os

sys.path.insert(0, '/app')

from services.neo4j_service import Neo4jService
from services.wikidata_client import WikidataClient
from services.source_classification import (
    classify_source_by_domain, classify_source_from_wikidata, compute_base_prior
)


async def fetch_p31(wikidata_client, qid):
    """Fetch P31 (instance of) values from Wikidata using WikidataClient session."""
    if not qid:
        return []

    try:
        await wikidata_client._ensure_session()

        params = {
            'action': 'wbgetentities',
            'ids': qid,
            'format': 'json',
            'props': 'claims'
        }

        async with wikidata_client.session.get(
            wikidata_client.api_url,
            params=params,
            timeout=10
        ) as resp:
            if resp.status != 200:
                return []

            data = await resp.json()
            entity = data.get('entities', {}).get(qid, {})
            claims = entity.get('claims', {})

            p31_qids = []
            for claim in claims.get('P31', []):
                mainsnak = claim.get('mainsnak', {})
                datavalue = mainsnak.get('datavalue', {})
                if datavalue.get('type') == 'wikibase-entityid':
                    p31_qids.append(datavalue['value']['id'])

            return p31_qids
    except Exception as e:
        print(f"      Error fetching P31 for {qid}: {e}")
        return []


async def main():
    print("=" * 80)
    print("📰 Backfill Publisher Source Priors (with Wikidata P31)")
    print("=" * 80)

    neo4j = Neo4jService()
    await neo4j.connect()

    # Find all publishers (update even those with priors to use Wikidata)
    print("\n📋 Finding publishers...")
    publishers = await neo4j._execute_read("""
        MATCH (pub:Entity {is_publisher: true})
        RETURN pub.id as id, pub.canonical_name as name, pub.domain as domain,
               pub.wikidata_qid as qid, pub.source_type as current_type
    """)

    if not publishers:
        print("   No publishers found!")
        await neo4j.close()
        return

    print(f"   Found {len(publishers)} publishers")

    # Update each publisher
    updated = 0
    wikidata_classified = 0
    domain_classified = 0

    wikidata_client = WikidataClient()

    for pub in publishers:
        domain = pub['domain'] or ''
        name = pub['name'] or ''
        qid = pub['qid']
        current_type = pub['current_type']

        # Try Wikidata P31 first
        source_type = None
        method = "domain"

        if qid:
            p31_qids = await fetch_p31(wikidata_client, qid)
            if p31_qids:
                source_type = classify_source_from_wikidata(p31_qids)
                if source_type:
                    method = f"P31:{p31_qids[0]}"
                    wikidata_classified += 1

        # Fall back to domain
        if not source_type:
            source_type, has_byline = classify_source_by_domain(domain, name)
            domain_classified += 1
        else:
            has_byline = source_type not in ('official', 'aggregator')

        base_prior = compute_base_prior(source_type, has_byline)

        # Only update if different or missing
        if current_type != source_type:
            await neo4j._execute_write("""
                MATCH (e:Entity {id: $id})
                SET e.source_type = $source_type,
                    e.base_prior = $base_prior,
                    e.updated_at = datetime()
            """, {
                'id': pub['id'],
                'source_type': source_type,
                'base_prior': base_prior
            })
            updated += 1
            status = "UPDATED"
        else:
            status = "unchanged"

        print(f"   [{base_prior:.2f}] {source_type:12s} - {name:30s} ({method}) {status}")

    await wikidata_client.close()

    print(f"\n✅ Updated {updated} publishers")
    print(f"   Wikidata P31: {wikidata_classified}")
    print(f"   Domain fallback: {domain_classified}")

    # Verify
    print("\n📊 Final distribution:")
    stats = await neo4j._execute_read("""
        MATCH (pub:Entity {is_publisher: true})
        WITH pub.source_type as source_type, count(*) as count
        RETURN source_type, count
        ORDER BY count DESC
    """)

    for s in stats:
        print(f"   {s['source_type'] or 'NULL':12s}: {s['count']} publishers")

    await neo4j.close()
    print("\n✅ Backfill complete!")


if __name__ == "__main__":
    asyncio.run(main())
