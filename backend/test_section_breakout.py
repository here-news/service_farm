"""
Test Section Breakout - Demonstrate promotion with divergent pages

This adds hypothetical pages that SHOULD trigger section promotion
"""
import asyncio
import json
from datetime import datetime, timedelta
from workers.section_aware_enrichment import SectionAwareEnricher
from openai import AsyncOpenAI
import os

async def test_breakout():
    """Add investigation and policy response pages to trigger promotion"""

    # Load existing event state
    with open('/tmp/section_growth_result.json') as f:
        event_state = json.load(f)

    print("="*80)
    print("SECTION BREAKOUT TEST")
    print("Adding temporally/semantically divergent pages...")
    print("="*80)

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    enricher = SectionAwareEnricher(client)

    # Hypothetical page 1: Investigation article (3 days later)
    investigation_page = {
        'id': 'test-investigation-1',
        'title': 'Hong Kong Police Charge 3 with Manslaughter in Fatal Fire Case',
        'description': 'Three men have been formally charged with manslaughter following the deadly Wang Fuk Court fire. Investigators revealed bamboo scaffolding contractors failed to follow safety protocols.',
        'content_text': '''
        Hong Kong authorities announced Thursday that three contractors have been charged with manslaughter
        in connection with the Wang Fuk Court fire that killed 44 people on November 26.

        The suspects, aged 35 to 52, were arrested immediately after the fire and have been in custody since.
        Prosecutors allege the men installed bamboo scaffolding without proper fire-resistant netting,
        violating building safety codes.

        Lead investigator Inspector Wong stated: "Our investigation shows clear negligence. The scaffolding
        acted as a highway for the fire to spread between buildings. These men prioritized cost over safety."

        The case will proceed to Hong Kong High Court next month. If convicted, each defendant faces up to
        life imprisonment under Hong Kong's manslaughter statutes.

        Separately, the Buildings Department has launched an audit of all scaffolding installations citywide,
        inspecting over 2,000 construction sites for similar violations.
        ''',
        'url': 'https://example.com/hk-fire-charges',
        'pub_time': datetime(2025, 11, 29, 10, 30),  # 3 days later
        'created_at': datetime(2025, 11, 29, 10, 30)
    }

    # Hypothetical page 2: Policy response (5 days later)
    policy_page = {
        'id': 'test-policy-1',
        'title': 'Hong Kong Government Bans Bamboo Scaffolding on High-Rises Following Fire',
        'description': 'Chief Executive John Lee announced sweeping building code reforms, banning traditional bamboo scaffolding on buildings over 15 stories in response to the Wang Fuk Court tragedy.',
        'content_text': '''
        In a televised address Monday, Chief Executive John Lee unveiled emergency regulations banning the use
        of bamboo scaffolding on high-rise buildings taller than 15 stories.

        "The Wang Fuk Court fire exposed a fundamental vulnerability in our building safety practices," Lee said.
        "While bamboo scaffolding is a Hong Kong tradition, we cannot allow tradition to compromise safety."

        The new rules, effective immediately, require:
        - Metal scaffolding with fire-resistant coverings for all high-rises
        - Mandatory sprinkler systems on external scaffolding
        - Weekly safety inspections by certified engineers
        - Criminal liability for contractors who violate protocols

        The construction industry protested the measures, arguing bamboo scaffolding is cheaper and faster to install.
        However, public support for the reforms is overwhelming, with a recent poll showing 87% approval.

        Legislative Council is expected to pass permanent legislation codifying these rules by year's end.
        ''',
        'url': 'https://example.com/hk-scaffolding-ban',
        'pub_time': datetime(2025, 12, 1, 14, 0),  # 5 days later
        'created_at': datetime(2025, 12, 1, 14, 0)
    }

    print("\n" + "─"*80)
    print("[6/7] INVESTIGATION PAGE (3 days later)")
    print("─"*80)
    print(f"Title: {investigation_page['title']}")
    print(f"Time: {investigation_page['pub_time']}")

    result1 = await enricher.enrich_event_with_page(event_state, investigation_page)

    print(f"\n✅ Page enriched {result1['multi_section_count']} section(s):")
    for section_result in result1['sections_enriched']:
        print(f"   • {section_result['section_name']} ({section_result['section_key']}): {section_result['aspect']}")
        if 'promotion_score' in section_result:
            score = section_result['promotion_score']
            if isinstance(score, dict) and 'signals' in score and score.get('total', 0) > 0:
                print(f"     📊 Promotion Score: {score['total']:.3f} (T:{score['signals']['temporal_gap']:.2f} E:{score['signals']['entity_divergence']:.2f} S:{score['signals']['semantic_shift']:.2f} D:{score['signals']['page_density']:.2f})")

                if score['total'] >= 0.6:
                    print(f"     🔴 PROMOTABLE - Should become separate event!")
                elif score['total'] >= 0.45:
                    print(f"     🟡 REVIEW - Consider for promotion")
                else:
                    print(f"     🟢 STABLE")

    print("\n" + "─"*80)
    print("[7/7] POLICY RESPONSE PAGE (5 days later)")
    print("─"*80)
    print(f"Title: {policy_page['title']}")
    print(f"Time: {policy_page['pub_time']}")

    result2 = await enricher.enrich_event_with_page(event_state, policy_page)

    print(f"\n✅ Page enriched {result2['multi_section_count']} section(s):")
    for section_result in result2['sections_enriched']:
        print(f"   • {section_result['section_name']} ({section_result['section_key']}): {section_result['aspect']}")
        if 'promotion_score' in section_result:
            score = section_result['promotion_score']
            if isinstance(score, dict) and 'signals' in score and score.get('total', 0) > 0:
                print(f"     📊 Promotion Score: {score['total']:.3f} (T:{score['signals']['temporal_gap']:.2f} E:{score['signals']['entity_divergence']:.2f} S:{score['signals']['semantic_shift']:.2f} D:{score['signals']['page_density']:.2f})")

                if score['total'] >= 0.6:
                    print(f"     🔴 PROMOTABLE - Should become separate event!")
                elif score['total'] >= 0.45:
                    print(f"     🟡 REVIEW - Consider for promotion")
                else:
                    print(f"     🟢 STABLE")

    # Final summary
    print("\n\n" + "="*80)
    print("FINAL EVENT STATE - WITH BREAKOUT SECTIONS")
    print("="*80)
    print(f"Total artifacts: {event_state['artifact_count']}")
    print(f"Total sections: {len(event_state['sections'])}\n")

    for key, section in event_state['sections'].items():
        ps_data = section.get('promotion_score', 0.0)
        ps = ps_data if isinstance(ps_data, (int, float)) else ps_data.get('total', 0.0) if isinstance(ps_data, dict) else 0.0

        status = "🔴 PROMOTABLE" if ps >= 0.6 else "🟡 REVIEW" if ps >= 0.45 else "🟢 STABLE" if ps > 0 else ""

        print(f"{section['name']} ({key})")
        print(f"  Pages: {section['page_count']}")
        if key != 'main' and ps > 0:
            print(f"  Score: {ps:.3f} {status}")
        print()

    # Save updated state
    with open('/tmp/section_breakout_result.json', 'w') as f:
        json.dump(event_state, f, indent=2, default=str)

    print("Updated state saved to: /tmp/section_breakout_result.json")
    print("="*80)

if __name__ == '__main__':
    asyncio.run(test_breakout())
