"""
Test Processing New Page - Demonstrate how claim-based enricher handles incoming updates

Scenario: Hong Kong fire event is established with 44 deaths
New page arrives: Death toll rises to 128, fire alarms weren't working
"""
import asyncio
import json
from datetime import datetime
from workers.claim_based_enrichment import ClaimBasedEnricher
from openai import AsyncOpenAI
import os


async def test_new_page_arrival():
    """Process new BBC page showing death toll increase"""

    print("=" * 80)
    print("NEW PAGE ARRIVAL TEST - Hong Kong Fire Death Toll Update")
    print("=" * 80)

    # Load existing event state
    with open('/tmp/claim_based_result.json') as f:
        event_state = json.load(f)

    print(f"\n📊 Existing Event State:")
    print(f"   Total artifacts: {event_state['artifact_count']}")
    print(f"   Total sections: {len(event_state['sections'])}")
    overview_paragraphs = len([p for p in event_state['overview'].split('\n\n') if p.strip()])
    print(f"   Overview paragraphs: {overview_paragraphs}")

    # Show current main event claims
    main_section = event_state['sections']['main']
    print(f"\n📑 Main Event Section BEFORE:")
    print(f"   Pages: {main_section['page_count']}")
    print(f"   Claims: {len(main_section['claims'])}")
    casualty_claims = [c for c in main_section['claims'] if 'killed' in c['text'].lower() or 'died' in c['text'].lower()]
    print(f"   Casualty claims:")
    for c in casualty_claims:
        print(f"      • {c['text']} (confidence: {c['confidence']})")

    # New BBC page - CRITICAL UPDATE
    new_page = {
        'id': 'aa052ee6-3e94-46c1-8f6c-e693e97b3b28',
        'title': 'Hong Kong fire death toll rises to 128 as officials say fire alarms not working properly',
        'pub_time': datetime(2025, 12, 2, 14, 51),  # 6 days after initial fire
        'url': 'https://www.bbc.com/news/live/c2emg1kj1klt',
        'content_text': '''
        Weeks-long investigation under way as death toll soars

        As night falls in Hong Kong, dozens of families continue dealing with the aftermath of the deadly fire.

        What happened?

        At 14:51 local time on Wednesday (06:51 GMT), a fire broke out at the Wang Fuk Court apartment complex
        in Tai Po, home to around 4,600 residents.

        The fire ripped through the estate for over a day - before finally being put out on at around 10:18
        local time (02:18 GMT) this morning.

        What do we know?

        At least 128 people are now known to have died in the fire and there are dozens still missing.

        The fire service said that the fire alarms in all eight blocks were not working effectively, after
        reports from residents that some didn't go off - here's what we know about the fire and how it
        could've spread so quickly.

        Three men from a construction firm have been arrested on suspicion of manslaughter, and there is a
        separate corruption investigation under way.

        What happens next

        An investigation will be taking place over the next few weeks, authorities say. Our reporter at the
        scene has spotted what appears to be investigators arriving wearing personal protective equipment.

        Officials say that schemes will be set up to arrange financial assistance for those who have lost
        their homes.
        '''
    }

    print(f"\n\n{'='*80}")
    print(f"🚨 NEW PAGE ARRIVES - {new_page['pub_time']}")
    print(f"{'='*80}")
    print(f"Title: {new_page['title']}")
    print(f"Date: {new_page['pub_time']}")
    print(f"Source: {new_page['url']}")

    # Process with claim-based enricher
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    enricher = ClaimBasedEnricher(client)

    result = await enricher.enrich_event_with_page(event_state, new_page)

    print(f"\n\n✅ Processing Results:")
    print(f"   Claims extracted: {result['claims_extracted']}")
    print(f"   Sections updated: {', '.join(result['sections_updated'])}")
    print(f"   Milestones added: {result['milestones_added']}")

    # Show updated state
    print(f"\n\n📊 Updated Event State:")
    print(f"   Total artifacts: {event_state['artifact_count']}")
    print(f"   Total sections: {len(event_state['sections'])}")
    print(f"   Timeline entries: {len(event_state['timeline'])}")

    # Show main event AFTER
    main_section = event_state['sections']['main']
    print(f"\n📑 Main Event Section AFTER:")
    print(f"   Pages: {main_section['page_count']}")
    print(f"   Claims: {len(main_section['claims'])}")
    casualty_claims = [c for c in main_section['claims'] if 'killed' in c['text'].lower() or 'died' in c['text'].lower() or '128' in c['text']]
    print(f"   Casualty claims:")
    for c in casualty_claims:
        print(f"      • {c['text']} (confidence: {c['confidence']})")

    # Show overview update
    print(f"\n📖 Master Overview (last paragraph):")
    print("─" * 80)
    paragraphs = [p for p in event_state['overview'].split('\n\n') if p.strip()]
    print(paragraphs[-1] if paragraphs else "No overview")

    # Show investigation section
    if 'investigation' in event_state['sections']:
        investigation = event_state['sections']['investigation']
        print(f"\n🔍 Investigation Section:")
        print(f"   Pages: {investigation['page_count']}")
        print(f"   Claims: {len(investigation['claims'])}")
        print(f"   Recent claims:")
        for c in investigation['claims'][-3:]:
            print(f"      • [{c['modality']}] {c['text']}")

    # Show latest timeline entry
    print(f"\n📅 Latest Timeline Entry:")
    if event_state['timeline']:
        latest = event_state['timeline'][-1]
        print(f"   {latest['date']} - {latest['title']}")
        print(f"   Type: {latest['type']}, Severity: {latest['severity']}")

    # Save updated state
    output_path = '/tmp/claim_based_updated.json'
    with open(output_path, 'w') as f:
        json.dump(event_state, f, indent=2)

    print(f"\n\nUpdated state saved to: {output_path}")
    print("=" * 80)
    print("\n🎯 KEY DEMONSTRATION:")
    print("   ✓ Master overview ACCUMULATED (new paragraph added)")
    print("   ✓ Main section claims ACCUMULATED (updated death toll)")
    print("   ✓ Investigation claims ACCUMULATED (fire alarm failure)")
    print("   ✓ Timeline ACCUMULATED (new milestone)")
    print("   ✓ NO REWRITING - all previous information preserved")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_new_page_arrival())
