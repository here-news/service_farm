"""
Test Claim-Based Enrichment with Hong Kong Fire Case

Validates the new architecture:
- Claims extraction and routing
- Master overview accumulation
- Timeline milestone generation
"""
import asyncio
import json
from datetime import datetime
from workers.claim_based_enrichment import ClaimBasedEnricher
from openai import AsyncOpenAI
import os


async def test_claim_based_hong_kong_fire():
    """Test claim-based enrichment with Hong Kong fire pages"""

    print("=" * 80)
    print("CLAIM-BASED ENRICHMENT TEST - Hong Kong Fire")
    print("=" * 80)

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    enricher = ClaimBasedEnricher(client)

    # Initialize event state
    event_state = {
        'event_id': 'hk-fire-2025'
    }

    # Test pages in chronological order
    pages = [
        {
            'id': 'hk-fire-1',
            'title': 'Dozens killed in Hong Kong apartment fire',
            'pub_time': datetime(2025, 11, 26, 4, 0),
            'content_text': '''
            A devastating fire swept through a residential building in Hong Kong early Tuesday morning,
            killing at least 44 people and injuring dozens more. The blaze broke out around 3:30 AM at
            Wang Fuk Court in the Sham Shui Po district.

            Firefighters battled the flames for several hours before bringing the fire under control.
            The Hong Kong Fire Services Department deployed over 200 firefighters and 40 vehicles to the scene.
            '''
        },
        {
            'id': 'hk-fire-2',
            'title': 'Hong Kong fire: Death toll rises to 44 as investigators examine cause',
            'pub_time': datetime(2025, 11, 26, 14, 30),
            'content_text': '''
            Authorities confirmed Wednesday that 44 people died in the Wang Fuk Court fire, making it
            one of Hong Kong's deadliest residential fires in decades. Another 28 people were hospitalized
            with injuries ranging from smoke inhalation to severe burns.

            Investigators are examining whether faulty wiring or combustible materials in external bamboo
            scaffolding contributed to the rapid spread of flames. The building was undergoing renovation
            work at the time of the fire.
            '''
        },
        {
            'id': 'hk-fire-3',
            'title': 'Hong Kong Police Charge 3 with Manslaughter in Fatal Fire Case',
            'pub_time': datetime(2025, 11, 29, 10, 30),
            'content_text': '''
            Hong Kong authorities announced Thursday that three contractors have been charged with manslaughter
            in connection with the Wang Fuk Court fire that killed 44 people on November 26.

            The suspects, aged 35 to 52, were arrested immediately after the fire and have been in custody since.
            Prosecutors allege the men installed bamboo scaffolding without proper fire-resistant netting,
            violating building safety codes.

            Lead investigator Inspector Wong stated: "Our investigation shows clear negligence. The scaffolding
            acted as a highway for the fire to spread between buildings."
            '''
        },
        {
            'id': 'hk-fire-4',
            'title': 'Hong Kong Government Bans Bamboo Scaffolding on High-Rises Following Fire',
            'pub_time': datetime(2025, 12, 1, 14, 0),
            'content_text': '''
            In a televised address Monday, Chief Executive John Lee unveiled emergency regulations banning the use
            of bamboo scaffolding on high-rise buildings taller than 15 stories.

            "The Wang Fuk Court fire exposed a fundamental vulnerability in our building safety practices," Lee said.
            "While bamboo scaffolding is a Hong Kong tradition, we cannot allow tradition to compromise safety."

            The new rules, effective immediately, require metal scaffolding with fire-resistant coverings for all
            high-rises and mandatory sprinkler systems on external scaffolding.
            '''
        }
    ]

    # Process each page
    for i, page in enumerate(pages, 1):
        print(f"\n{'─' * 80}")
        print(f"[{i}/{len(pages)}] Processing: {page['title']}")
        print(f"Date: {page['pub_time']}")
        print('─' * 80)

        result = await enricher.enrich_event_with_page(event_state, page)

        print(f"\n✅ Results:")
        print(f"   Claims extracted: {result['claims_extracted']}")
        print(f"   Sections updated: {', '.join(result['sections_updated'])}")
        print(f"   Milestones added: {result['milestones_added']}")

    # Display final state
    print("\n\n" + "=" * 80)
    print("FINAL EVENT STATE")
    print("=" * 80)

    print(f"\n📊 Event Metadata:")
    print(f"   Total artifacts: {event_state['artifact_count']}")
    print(f"   Total sections: {len(event_state['sections'])}")
    print(f"   Timeline entries: {len(event_state['timeline'])}")

    print(f"\n📖 Master Overview:")
    print("─" * 80)
    overview_preview = event_state['overview'][:500] + "..." if len(event_state['overview']) > 500 else event_state['overview']
    print(overview_preview)

    print(f"\n\n📑 Sections:")
    for key, section in event_state['sections'].items():
        print(f"\n{section['name']} ({key})")
        print(f"  Type: {section['semantic_type']}")
        print(f"  Pages: {section['page_count']}")
        print(f"  Claims: {len(section['claims'])}")

        # Show sample claims
        if section['claims']:
            print(f"  Sample claims:")
            for claim in section['claims'][:3]:
                print(f"    • [{claim['modality']}] {claim['text']} (conf: {claim['confidence']:.2f})")

    print(f"\n\n📅 Timeline:")
    for milestone in event_state['timeline']:
        print(f"  {milestone['date']} - {milestone['title']}")
        print(f"    Type: {milestone['type']}, Severity: {milestone['severity']}")

    # Save to file
    # Serialize for JSON
    serialized_state = event_state.copy()

    # Convert datetime objects in timeline
    for milestone in serialized_state['timeline']:
        if isinstance(milestone.get('date'), datetime):
            milestone['date'] = milestone['date'].isoformat()

    output_path = '/tmp/claim_based_result.json'
    with open(output_path, 'w') as f:
        json.dump(serialized_state, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_claim_based_hong_kong_fire())
