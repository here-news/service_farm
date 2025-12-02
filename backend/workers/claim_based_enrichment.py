"""
Claim-Based Event Enrichment - Following Louvre demo architecture

Philosophy:
- Master overview: Cumulative narrative that grows (doesn't rewrite)
- Sections: Collections of claims with confidence scores
- Claims: Atomic facts extracted from pages, routed to sections
- Timeline: Key milestones extracted from pages

Architecture (like Louvre demo):
Event:
  - overview: Master narrative (synthesized, cumulative)
  - sections: { main: {claims: [...], ontology: {...}}, investigation: {...} }
  - timeline: [...milestones...]
"""
import asyncio
import json
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClaimBasedEnricher:
    """
    Enriches events using claim extraction and routing

    Process:
    1. Extract claims from page
    2. Route claims to sections
    3. Update section claims (accumulate, don't rewrite)
    4. Update master overview (append developments)
    5. Extract timeline milestones
    """

    def __init__(self, openai_client: AsyncOpenAI):
        self.client = openai_client

    async def enrich_event_with_page(
        self,
        event_state: Dict,
        page: Dict
    ) -> Dict:
        """
        Claim-based enrichment: event(n) + page → event(n+1)

        Steps:
        1. Extract claims from page
        2. Route claims to sections (multi-section)
        3. Update sections with new claims
        4. Update master overview
        5. Extract timeline milestones
        """
        # Initialize if first page
        if 'sections' not in event_state:
            event_state['sections'] = {
                'main': self._create_empty_section('Main Event', 'primary')
            }
            event_state['overview'] = ''
            event_state['timeline'] = []
            event_state['artifact_count'] = 0

        logger.info(f"\n{'='*80}")
        logger.info(f"Processing page: {page.get('title', 'Untitled')}")
        logger.info(f"{'='*80}")

        # Step 1: Extract claims from page
        claims = await self._extract_claims_from_page(page)
        logger.info(f"   📋 Extracted {len(claims)} claims from page")

        # Step 2: Route claims to sections
        claim_routing = await self._route_claims_to_sections(event_state, claims, page)
        logger.info(f"   📌 Claims routed to {len(claim_routing)} section(s)")

        # Step 3: Update sections with claims
        for section_key, section_claims in claim_routing.items():
            if section_key not in event_state['sections']:
                # Create new section
                section_info = await self._get_section_info(section_key)
                logger.info(f"   🆕 Creating section: {section_info['name']}")
                event_state['sections'][section_key] = self._create_empty_section(
                    section_info['name'],
                    section_info['semantic_type']
                )

            section = event_state['sections'][section_key]

            # Add claims to section
            for claim in section_claims:
                section['claims'].append({
                    'text': claim['text'],
                    'confidence': claim['confidence'],
                    'modality': claim['modality'],
                    'source_id': str(page['id']),
                    'source_title': page.get('title', ''),
                    'timestamp': datetime.utcnow().isoformat()
                })

            section['page_ids'].append(str(page['id']))
            section['page_count'] = len(section['page_ids'])
            section['updated_at'] = datetime.utcnow().isoformat()

            logger.info(f"   ➜ {section['name']}: +{len(section_claims)} claims (total: {len(section['claims'])})")

        # Step 4: Extract timeline milestones
        milestones = await self._extract_timeline_milestones(page, claims)
        for milestone in milestones:
            event_state['timeline'].append(milestone)
            logger.info(f"   📅 Timeline: {milestone['title']}")

        # Step 5: Update master overview (cumulative)
        event_state['overview'] = await self._update_master_overview(
            event_state,
            page,
            claims
        )

        # Update event-level metadata
        event_state['artifact_count'] += 1

        return {
            'claims_extracted': len(claims),
            'sections_updated': list(claim_routing.keys()),
            'milestones_added': len(milestones),
            'page_id': str(page['id'])
        }

    async def _extract_claims_from_page(self, page: Dict) -> List[Dict]:
        """
        Extract atomic claims from page content

        Returns:
        [
            {
                'text': '44 people killed in fire',
                'confidence': 0.95,
                'modality': 'FACT' | 'ALLEGED' | 'ESTIMATED',
                'category': 'casualties' | 'timeline' | 'location' | etc
            }
        ]
        """
        page_title = page.get('title', 'Untitled')
        page_content = page.get('content_text', '')[:2000]

        prompt = f"""Extract atomic factual claims from this news article.

Article: {page_title}
Content: {page_content}

Extract SPECIFIC, VERIFIABLE claims. Each claim should be:
- Atomic (one fact per claim)
- Specific (with numbers, names, locations)
- Verifiable from the article

Categorize each claim:
- casualties: Deaths, injuries, victims
- timeline: When events occurred
- location: Where events occurred
- response: Emergency response, rescue efforts
- investigation: Investigation findings, arrests
- legal: Charges, trials, legal proceedings
- policy: Government policy, regulations
- cause: Root causes, contributing factors

Assign modality:
- FACT: Confirmed by authorities/multiple sources
- ALLEGED: Claimed but not confirmed
- ESTIMATED: Approximate numbers/assessments

Return JSON array:
[
  {{
    "text": "44 people killed in apartment fire",
    "confidence": 0.95,
    "modality": "FACT",
    "category": "casualties"
  }},
  {{
    "text": "Fire occurred at Wang Fuk Court in Hong Kong",
    "confidence": 0.98,
    "modality": "FACT",
    "category": "location"
  }}
]
"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You extract atomic claims from news articles. Return valid JSON array only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()

            claims = json.loads(content)

            # Validate
            if not isinstance(claims, list):
                claims = [claims]

            return claims

        except Exception as e:
            logger.error(f"Claim extraction error: {e}")
            # Fallback: extract basic claims from title
            return [{
                'text': page_title,
                'confidence': 0.7,
                'modality': 'FACT',
                'category': 'general'
            }]

    async def _route_claims_to_sections(
        self,
        event_state: Dict,
        claims: List[Dict],
        page: Dict
    ) -> Dict[str, List[Dict]]:
        """
        Route extracted claims to appropriate sections

        Returns:
        {
            'main': [claim1, claim2],
            'investigation': [claim3],
            'policy_response': [claim4]
        }
        """
        # Prepare section summary
        sections_summary = []
        for key, section in event_state['sections'].items():
            sections_summary.append(
                f"- {key}: {section['name']} ({section['semantic_type']}, {section['page_count']} pages, {len(section['claims'])} claims)"
            )

        sections_text = "\n".join(sections_summary) if sections_summary else "None (first page)"

        # Format claims for routing
        claims_text = "\n".join([
            f"{i+1}. [{c['category']}] {c['text']} (confidence: {c['confidence']}, modality: {c['modality']})"
            for i, c in enumerate(claims)
        ])

        prompt = f"""Route these claims to appropriate event sections.

EXISTING SECTIONS:
{sections_text}

CLAIMS TO ROUTE:
{claims_text}

Rules:
1. Route core event facts (casualties, location, initial timeline) to "main"
2. Route investigation/arrests to "investigation" or "legal_action"
3. Route policy changes to "policy_response"
4. Create new sections for distinct phases (use keys like: investigation, legal_action, policy_response, aftermath)
5. Claims can go to MULTIPLE sections if relevant

Return JSON object mapping section_key → claim indices:
{{
  "main": [1, 2, 4],
  "investigation": [3, 5],
  "policy_response": [6]
}}

If creating new section, use standard keys: investigation, legal_action, policy_response, aftermath, memorial
"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You route claims to event sections. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()

            routing = json.loads(content)

            # Convert indices to claims
            result = {}
            for section_key, indices in routing.items():
                result[section_key] = [claims[i-1] for i in indices if 0 < i <= len(claims)]

            # Ensure at least main section
            if not result:
                result['main'] = claims

            return result

        except Exception as e:
            logger.error(f"Claim routing error: {e}")
            # Default: all claims to main
            return {'main': claims}

    async def _extract_timeline_milestones(
        self,
        page: Dict,
        claims: List[Dict]
    ) -> List[Dict]:
        """
        Extract timeline milestones from page

        Returns:
        [
            {
                'date': '2025-11-26T03:30:00',
                'title': 'Fire breaks out at Wang Fuk Court',
                'description': 'Fire started in apartment building...',
                'type': 'event' | 'development' | 'announcement',
                'severity': 'critical' | 'high' | 'medium'
            }
        ]
        """
        # Get timeline-related claims
        timeline_claims = [c for c in claims if c['category'] in ['timeline', 'casualties', 'response', 'investigation', 'legal']]

        if not timeline_claims:
            return []

        page_title = page.get('title', '')
        pub_time = page.get('pub_time', page.get('created_at'))

        # Simple heuristic: major developments become milestones
        if any(kw in page_title.lower() for kw in ['killed', 'dead', 'fire', 'arrest', 'charge', 'ban', 'regulation']):
            milestone_type = 'event'
            severity = 'critical'

            if 'arrest' in page_title.lower() or 'charge' in page_title.lower():
                milestone_type = 'development'
                severity = 'high'
            elif 'ban' in page_title.lower() or 'policy' in page_title.lower():
                milestone_type = 'announcement'
                severity = 'medium'

            return [{
                'date': pub_time.isoformat() if isinstance(pub_time, datetime) else pub_time,
                'title': page_title,
                'description': timeline_claims[0]['text'] if timeline_claims else '',
                'type': milestone_type,
                'severity': severity
            }]

        return []

    async def _update_master_overview(
        self,
        event_state: Dict,
        page: Dict,
        claims: List[Dict]
    ) -> str:
        """
        Update master overview with new developments (cumulative, doesn't rewrite)

        Strategy:
        - If overview is empty: Generate initial overview
        - If overview exists: Append new developments
        """
        current_overview = event_state.get('overview', '')

        # Prepare context
        page_title = page.get('title', '')
        page_date = page.get('pub_time', page.get('created_at'))

        # Get key claims
        key_claims = [c for c in claims if c['confidence'] > 0.8][:5]
        claims_text = "\n".join([f"- {c['text']}" for c in key_claims])

        if not current_overview:
            # Generate initial overview
            prompt = f"""Generate a comprehensive master overview for this breaking event.

First article: {page_title}
Date: {page_date}

Key facts:
{claims_text}

Write a 2-3 paragraph overview that:
1. Opens with what happened (core event)
2. Includes key details (casualties, location, timeline)
3. Sets up context for ongoing developments

Write in journalistic style, past tense. Focus on facts.
"""
        else:
            # Append new development
            prompt = f"""Update the event overview with new developments.

CURRENT OVERVIEW:
{current_overview}

NEW DEVELOPMENT:
Article: {page_title}
Date: {page_date}
Key claims:
{claims_text}

Instructions:
1. If this is truly NEW information, add a paragraph about this development
2. If it's CORROBORATION of existing facts, DON'T add (overview stays same)
3. If it's a MAJOR development (arrests, policy changes), add prominently
4. Maintain chronological flow

Return the FULL updated overview (or same overview if no update needed).
"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You maintain event overviews. Write clearly and factually."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Overview update error: {e}")
            return current_overview  # Keep existing overview on error

    async def _get_section_info(self, section_key: str) -> Dict:
        """Get section name and semantic type from key"""
        section_map = {
            'main': {'name': 'Main Event', 'semantic_type': 'primary'},
            'investigation': {'name': 'Investigation', 'semantic_type': 'investigation'},
            'legal_action': {'name': 'Legal Action', 'semantic_type': 'legal_action'},
            'policy_response': {'name': 'Policy Response', 'semantic_type': 'policy_response'},
            'aftermath': {'name': 'Aftermath', 'semantic_type': 'aftermath'},
            'memorial': {'name': 'Memorial', 'semantic_type': 'memorial'},
            'response': {'name': 'Emergency Response', 'semantic_type': 'response'}
        }

        return section_map.get(section_key, {
            'name': section_key.replace('_', ' ').title(),
            'semantic_type': section_key
        })

    def _create_empty_section(self, name: str, semantic_type: str) -> Dict:
        """Create empty section structure"""
        return {
            'name': name,
            'semantic_type': semantic_type,
            'claims': [],  # List of atomic claims
            'page_ids': [],
            'page_count': 0,
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat()
        }
