#!/usr/bin/env python3
"""
Create synthetic claims for Hong Kong fire event to test metabolism
"""
import asyncio
from services.neo4j_service import Neo4jService
from utils.id_generator import generate_page_id, generate_claim_id, generate_entity_id
from datetime import datetime

# Sample claims from Hong Kong fire articles
HONG_KONG_FIRE_CLAIMS = [
    # Casualties
    "At least 11 people were killed in the fire",
    "13 people died in the fire",
    "More than 30 people were injured",
    "36 people were injured according to authorities",
    "44 people were hospitalized",

    # Location
    "The fire occurred in Tai Po, Hong Kong",
    "The blaze engulfed multiple high-rise apartment towers",
    "The fire started in a flat on the 7th floor",

    # Timeline
    "The fire broke out on Wednesday",
    "Emergency services received the first call around 4pm",
    "The fire alarm went off at 3:40pm",
    "Firefighters battled the blaze for several hours",

    # Response
    "More than 170 firefighters responded to the scene",
    "120 firefighters were deployed",
    "Police evacuated residents from surrounding buildings",
    "Emergency services rescued dozens of trapped residents",

    # Cause
    "The fire is believed to have started in a flat",
    "Authorities are investigating the cause",
    "The building was undergoing renovation",

    # Impact
    "Residents reported thick black smoke",
    "The fire spread rapidly through multiple floors",
    "Windows were blown out by the intensity of the fire",
]

async def main():
    neo4j = Neo4jService()
    await neo4j.connect()

    print("🔥 Creating synthetic Hong Kong fire claims for event metabolism testing\n")

    # Create pages
    urls = [
        "https://bbc.com/news/live/c2emg1kj1klt",
        "https://nypost.com/2025/11/26/world-news/hong-kong-fire",
        "https://dw.com/en/hong-kong-fire-death-toll-rises",
    ]

    page_ids = []
    for url in urls:
        page_id = generate_page_id()
        page_ids.append(page_id)

        await neo4j._execute_write("""
            CREATE (p:Page {
                id: $page_id,
                url: $url,
                status: 'knowledge_complete',
                created_at: datetime()
            })
        """, {'page_id': page_id, 'url': url})

        print(f"✅ Created page {page_id}")

    # Create claims (distribute across pages)
    claim_count = 0
    for i, claim_text in enumerate(HONG_KONG_FIRE_CLAIMS):
        page_id = page_ids[i % len(page_ids)]  # Round-robin across pages
        claim_id = generate_claim_id()

        await neo4j._execute_write("""
            MATCH (p:Page {id: $page_id})
            CREATE (c:Claim {
                id: $claim_id,
                text: $text,
                confidence: 0.85,
                modality: 'factual',
                created_at: datetime()
            })
            CREATE (p)-[:EXTRACTED]->(c)
        """, {'page_id': page_id, 'claim_id': claim_id, 'text': claim_text})

        claim_count += 1

    print(f"\n✅ Created {claim_count} claims across {len(page_ids)} pages")

    # Create some key entities
    entities = [
        ("Tai Po", "LOCATION"),
        ("Hong Kong", "LOCATION"),
        ("Emergency Services", "ORGANIZATION"),
    ]

    for entity_name, entity_type in entities:
        entity_id = generate_entity_id()

        await neo4j._execute_write("""
            CREATE (e:Entity {
                id: $entity_id,
                canonical_name: $name,
                entity_type: $type,
                confidence: 0.9,
                mention_count: 1,
                created_at: datetime()
            })
        """, {'entity_id': entity_id, 'name': entity_name, 'type': entity_type})

    print(f"✅ Created {len(entities)} entities")

    print(f"\n🚀 Ready to test event metabolism!")
    print(f"📋 Next: Queue these pages to event worker")

    await neo4j.close()

if __name__ == '__main__':
    asyncio.run(main())
