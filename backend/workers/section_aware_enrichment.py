"""
Section-Aware Event Enrichment - Events grow through sections

Philosophy:
- Events have sections that grow independently
- Sections can be promoted to separate events
- LLM detects section boundaries naturally
- Each section has its own ontology and promotion score

Architecture:
event = {
    "sections": {
        "main": {...},  # Always present
        "investigation": {...},  # Detected phases
        "rescue_operations": {...}
    }
}
"""
import asyncio
import json
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from openai import AsyncOpenAI
from workers.holistic_enrichment import FactBelief, HolisticEventEnricher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SectionAwareEnricher:
    """
    Enriches events with section awareness

    Routes pages to appropriate sections and calculates promotion scores
    """

    def __init__(self, openai_client: AsyncOpenAI):
        self.client = openai_client
        self.base_enricher = HolisticEventEnricher(openai_client)

    async def enrich_event_with_page(
        self,
        event_state: Dict,
        page: Dict
    ) -> Dict:
        """
        Multi-section enrichment: event(n) + page → event(n+1)

        Steps:
        1. Extract all aspects from page holistically
        2. Route each aspect to appropriate section(s)
        3. Enrich ALL applicable sections
        4. Update promotion scores
        """
        # Initialize sections if first page
        if 'sections' not in event_state:
            event_state['sections'] = {
                'main': self._create_empty_section('Main Event', 'primary')
            }
            event_state['artifact_count'] = 0
            event_state['enrichment_timeline'] = []

        # Detect ALL sections this page contributes to
        section_routing = await self._detect_multi_section_routing(event_state, page)

        logger.info(f"   📌 Page contributes to {len(section_routing)} section(s)")

        enrichment_results = []

        # Enrich each applicable section
        for routing in section_routing:
            section_key = routing['section_key']

            # Create new section if needed
            if routing['action'] == 'create_section':
                logger.info(f"   🆕 Creating section: {routing['section_name']}")
                event_state['sections'][section_key] = self._create_empty_section(
                    routing['section_name'],
                    routing['semantic_type']
                )

            section = event_state['sections'][section_key]
            logger.info(f"   ➜ Enriching: {section['name']} ({routing['aspect']})")

            # Enrich the section using base enricher with filtered content
            # Wrap section in event_state format
            # Deserialize FactBelief objects from dicts if loading from JSON
            ontology = section.get('ontology', {})
            deserialized_ontology = self._deserialize_ontology(ontology)

            section_wrapper = {
                'event_id': event_state['event_id'],
                'ontology': deserialized_ontology,
                'artifact_count': section.get('page_count', 0),
                'enrichment_timeline': section.get('enrichment_timeline', [])
            }

            enrichment_result = await self.base_enricher.enrich_event_with_page(section_wrapper, page)

            # Update section with enrichment results (section_wrapper is modified in place)
            section['ontology'] = section_wrapper['ontology']
            section['enrichment_timeline'] = section_wrapper['enrichment_timeline']
            section['page_ids'].append(str(page['id']))
            section['page_count'] = len(section['page_ids'])
            section['updated_at'] = datetime.utcnow().isoformat()

            # Calculate promotion score for this section
            if section_key != 'main':
                promotion_score = await self._calculate_promotion_score(
                    event_state,
                    section_key,
                    section
                )
                section['promotion_score'] = promotion_score
                section['promotion_signals'] = promotion_score['signals']

                logger.info(f"   📊 Promotion score: {promotion_score['total']:.2f}")
                if promotion_score['total'] >= 0.6:
                    logger.warning(f"   🔴 Section '{section['name']}' is PROMOTABLE!")
                elif promotion_score['total'] >= 0.45:
                    logger.info(f"   🟡 Section '{section['name']}' in REVIEW zone")

            enrichment_results.append({
                'section_key': section_key,
                'section_name': section['name'],
                'aspect': routing['aspect'],
                'promotion_score': section.get('promotion_score', {'total': 0.0})
            })

        # Update event-level counters (only once per page, not per section)
        event_state['artifact_count'] += 1

        # Record page contribution to all sections
        for routing in section_routing:
            event_state['enrichment_timeline'].append({
                'page_id': str(page['id']),
                'section': routing['section_key'],
                'timestamp': datetime.utcnow().isoformat(),
                'page_time': page.get('pub_time', page.get('created_at'))
            })

        return {
            'sections_enriched': enrichment_results,
            'page_id': str(page['id']),
            'multi_section_count': len(section_routing)
        }

    async def _detect_multi_section_routing(
        self,
        event_state: Dict,
        page: Dict
    ) -> List[Dict]:
        """
        Detect ALL sections this page contributes to (multi-section routing)

        Returns list of routing decisions:
        [
            {
                "action": "enrich_section" | "create_section",
                "section_key": "main" | "investigation" | ...,
                "section_name": "Main Event" | "Criminal Investigation",
                "semantic_type": "primary" | "investigation" | ...,
                "aspect": "casualties, timeline" | "arrests, charges",
                "rationale": "..."
            },
            ...
        ]
        """
        # Prepare section summary
        sections_summary = []
        for key, section in event_state['sections'].items():
            sections_summary.append(
                f"- {key}: {section['name']} ({section['semantic_type']}, {section['page_count']} pages)"
            )

        sections_text = "\n".join(sections_summary) if sections_summary else "None (this is the first page)"

        # Get page content
        page_title = page.get('title', 'Untitled')
        page_desc = page.get('description', '')[:500]
        page_content = page.get('content_text', '')[:1000]

        prompt = f"""Analyze this news article and determine ALL sections it contributes information to.

A SINGLE article often contains information for MULTIPLE sections. For example:
- "Fire kills 44, police arrest 3" enriches BOTH "Emergency Response" AND "Legal Action"
- "Investigation finds violations, new regulations passed" enriches BOTH "Investigation" AND "Policy Response"

EXISTING SECTIONS:
{sections_text}

NEW PAGE:
Title: {page_title}
Description: {page_desc}
Content: {page_content}

Analyze what information this page contains. For each distinct aspect, determine which section(s) it belongs to.

Available semantic types:
- primary: Core event narrative (what happened)
- response: Emergency response, rescue, evacuation
- investigation: Criminal or official investigation
- aftermath: Long-term consequences, rebuilding
- legal_action: Lawsuits, charges, trials, arrests
- policy_response: Government policy changes, regulations
- memorial: Commemorations, remembrance
- related_incident: Similar or related event

Rules:
1. ONLY include "main" section if page contains CORE event facts (casualties, locations, immediate timeline of the primary incident)
2. Do NOT route to "main" for: policy responses, investigations, legal proceedings, aftermath, or events days/weeks later
3. Add specialized sections for distinct aspects (arrests → legal_action, policy → policy_response)
4. Create new section if page discusses a clearly separate phase/aspect
5. Return ALL applicable sections, not just one

Examples:
- "Fire kills 44 at apartment building" → main (core event), response (rescue efforts)
- "Police charge 3 with manslaughter" → legal_action ONLY (not main - this is days later)
- "Government bans bamboo scaffolding" → policy_response ONLY (not main - this is policy, not the fire itself)
- "Investigation reveals safety violations" → investigation ONLY (not main - this is investigation findings)

Return ONLY valid JSON array:
[
  {{
    "action": "enrich_section",
    "section_key": "main",
    "section_name": "Main Event",
    "semantic_type": "primary",
    "aspect": "casualties, timeline, locations",
    "rationale": "Core fire details"
  }},
  {{
    "action": "create_section",
    "section_key": "legal_action",
    "section_name": "Legal Action",
    "semantic_type": "legal_action",
    "aspect": "arrests, charges",
    "rationale": "Article discusses arrests"
  }}
]
"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You route pages to ALL applicable event sections. Return valid JSON array only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON if wrapped in markdown
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()

            routings = json.loads(content)

            # Validate it's a list
            if not isinstance(routings, list):
                routings = [routings]

            # Sanitize section keys
            for routing in routings:
                if routing['action'] == 'create_section':
                    section_key = routing.get('section_key', routing.get('section_name', 'new_section'))
                    section_key = re.sub(r'[^a-z0-9_]', '_', section_key.lower())
                    routing['section_key'] = section_key

            # Always ensure at least one section (default to main)
            if not routings:
                routings = [{
                    "action": "enrich_section",
                    "section_key": "main",
                    "section_name": "Main Event",
                    "semantic_type": "primary",
                    "aspect": "general information",
                    "rationale": "Default routing to main section"
                }]

            return routings

        except Exception as e:
            logger.error(f"Multi-section routing LLM error: {e}")
            # Default: route to main only
            return [{
                "action": "enrich_section",
                "section_key": "main",
                "section_name": "Main Event",
                "semantic_type": "primary",
                "aspect": "general information",
                "rationale": f"Error in routing, defaulting to main: {e}"
            }]

    async def _detect_section_routing(
        self,
        event_state: Dict,
        page: Dict
    ) -> Dict:
        """
        LLM decides: enrich existing section or create new?

        Returns:
        {
            "action": "enrich_section" | "create_section",
            "section_key": "main" | "investigation" | ...,
            "section_name": "Main Event" | "Criminal Investigation",
            "semantic_type": "primary" | "investigation" | "response" | ...,
            "rationale": "..."
        }
        """
        # Prepare section summary
        sections_summary = []
        for key, section in event_state['sections'].items():
            sections_summary.append(
                f"- {key}: {section['name']} ({section['semantic_type']}, {section['page_count']} pages)"
            )

        sections_text = "\n".join(sections_summary) if sections_summary else "None (this is the first page)"

        # Get page preview
        page_title = page.get('title', 'Untitled')
        page_desc = page.get('description', '')[:300]

        prompt = f"""You are routing a new page to the appropriate section of an event.

EXISTING SECTIONS:
{sections_text}

NEW PAGE:
Title: {page_title}
Description: {page_desc}

Does this page:
A) Enrich an existing section (which one?)
B) Create a new section (what name? what semantic type?)

Semantic types available:
- primary: Main event narrative
- response: Emergency response, rescue, evacuation
- investigation: Criminal or official investigation
- aftermath: Long-term consequences, rebuilding
- legal_action: Lawsuits, charges, trials
- policy_response: Government policy changes
- memorial: Commemorations, remembrance
- related_incident: Similar or related event

Return ONLY valid JSON:
{{
  "action": "enrich_section" or "create_section",
  "section_key": "main" or "investigation" or "new_section_key" (lowercase, underscores),
  "section_name": "Main Event" or "Criminal Investigation" (human-readable),
  "semantic_type": "primary" or "investigation" etc,
  "rationale": "Brief explanation"
}}
"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You route pages to event sections. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON if wrapped in markdown
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()

            decision = json.loads(content)

            # Validate and sanitize section_key
            if decision['action'] == 'create_section':
                # Sanitize section key
                section_key = decision.get('section_key', decision.get('section_name', 'new_section'))
                section_key = re.sub(r'[^a-z0-9_]', '_', section_key.lower())
                decision['section_key'] = section_key

            return decision

        except Exception as e:
            logger.error(f"Section routing LLM error: {e}")
            # Default: route to main
            return {
                "action": "enrich_section",
                "section_key": "main",
                "section_name": "Main Event",
                "semantic_type": "primary",
                "rationale": f"Error in routing, defaulting to main: {e}"
            }

    async def _calculate_promotion_score(
        self,
        event_state: Dict,
        section_key: str,
        section: Dict
    ) -> Dict:
        """
        Calculate promotion score for a section

        Returns:
        {
            "total": 0.75,
            "signals": {
                "temporal_gap": 0.6,
                "entity_divergence": 0.8,
                "semantic_shift": 0.9,
                "page_density": 0.7,
                "human_weight": 0.0
            }
        }
        """
        # Compare against the section with most pages (excluding current section)
        other_sections = {k: v for k, v in event_state['sections'].items() if k != section_key}
        if other_sections:
            main_section = max(
                other_sections.values(),
                key=lambda s: s.get('page_count', 0)
            )
        else:
            main_section = event_state['sections'].get('main', {})

        signals = {}

        # 1. Temporal Gap (0.0-1.0)
        signals['temporal_gap'] = self._calculate_temporal_gap(section, main_section)

        # 2. Entity Divergence (0.0-1.0)
        signals['entity_divergence'] = self._calculate_entity_divergence(section, main_section)

        # 3. Semantic Shift (0.0-1.0)
        signals['semantic_shift'] = await self._calculate_semantic_shift(section, main_section)

        # 4. Page Density (0.0-1.0)
        signals['page_density'] = self._calculate_page_density(section, event_state)

        # 5. Human Weight (-0.3 to +0.3)
        signals['human_weight'] = section.get('human_weight', 0.0)

        # Weighted sum
        total = (
            signals['temporal_gap'] * 0.25 +
            signals['entity_divergence'] * 0.25 +
            signals['semantic_shift'] * 0.20 +
            signals['page_density'] * 0.20 +
            signals['human_weight'] * 0.10
        )

        return {
            'total': round(total, 3),
            'signals': signals
        }

    def _calculate_temporal_gap(self, section: Dict, main_section: Dict) -> float:
        """
        Calculate temporal gap signal (0.0-1.0)

        Fault-tolerant: tries multiple sources, degrades gracefully
        """
        def extract_times(timeline_entries):
            """Extract timestamps from timeline with multiple fallbacks"""
            times = []
            for entry in timeline_entries:
                # Try 1: page_time (actual publication time)
                ts = entry.get('page_time')
                if not ts:
                    # Try 2: timestamp (processing time)
                    ts = entry.get('timestamp')

                if ts:
                    try:
                        if isinstance(ts, str):
                            # Handle ISO format and timezone variants
                            ts_clean = ts.replace('Z', '+00:00')
                            times.append(datetime.fromisoformat(ts_clean))
                        elif isinstance(ts, datetime):
                            times.append(ts)
                    except (ValueError, AttributeError):
                        continue
            return times

        try:
            # Get times from enrichment timeline
            section_times = extract_times(section.get('enrichment_timeline', []))
            main_times = extract_times(main_section.get('enrichment_timeline', []))

            # Fallback 1: Try ontology timeline
            if not section_times:
                ontology_start = section.get('ontology', {}).get('timeline', {}).get('start')
                if ontology_start:
                    if isinstance(ontology_start, dict):
                        val = ontology_start.get('current_value')
                    else:
                        val = ontology_start
                    if val:
                        try:
                            if isinstance(val, str):
                                section_times = [datetime.fromisoformat(val.replace('Z', '+00:00'))]
                            elif isinstance(val, datetime):
                                section_times = [val]
                        except:
                            pass

            if not main_times:
                ontology_start = main_section.get('ontology', {}).get('timeline', {}).get('start')
                if ontology_start:
                    if isinstance(ontology_start, dict):
                        val = ontology_start.get('current_value')
                    else:
                        val = ontology_start
                    if val:
                        try:
                            if isinstance(val, str):
                                main_times = [datetime.fromisoformat(val.replace('Z', '+00:00'))]
                            elif isinstance(val, datetime):
                                main_times = [val]
                        except:
                            pass

            # Fallback 2: Use created_at from sections themselves
            if not section_times and section.get('created_at'):
                try:
                    section_times = [datetime.fromisoformat(section['created_at'].replace('Z', '+00:00'))]
                except:
                    pass

            if not main_times and main_section.get('created_at'):
                try:
                    main_times = [datetime.fromisoformat(main_section['created_at'].replace('Z', '+00:00'))]
                except:
                    pass

            # If still no times, return moderate default
            if not section_times or not main_times:
                return 0.4  # Moderate default when data missing

            # Calculate gap between earliest section page and latest main page
            section_start = min(section_times)
            main_latest = max(main_times)

            gap_hours = abs((section_start - main_latest).total_seconds() / 3600)

            # Tiered scoring
            if gap_hours < 6:
                return 0.0      # Same timeframe
            elif gap_hours < 24:
                return 0.3      # Same day
            elif gap_hours < 48:
                return 0.6      # Next day
            elif gap_hours < 168:  # < 1 week
                return 0.8
            else:
                return 0.9      # Week+ later

        except Exception as e:
            logger.debug(f"Temporal gap error (tolerating): {e}")
            return 0.4  # Safe default

    def _calculate_entity_divergence(self, section: Dict, main_section: Dict) -> float:
        """Calculate entity divergence (Jaccard distance)"""
        # TODO: Extract entities from sections
        # For now, return moderate divergence
        return 0.5

    async def _calculate_semantic_shift(self, section: Dict, main_section: Dict) -> float:
        """
        LLM-based semantic shift detection

        Fault-tolerant: uses fallbacks and keyword heuristics
        """
        section_summary = section.get('ontology', {}).get('story', {}).get('description', '')
        main_summary = main_section.get('ontology', {}).get('story', {}).get('description', '')

        # Fallback 1: If no summaries, use semantic types
        if not section_summary or not main_summary:
            section_type = section.get('semantic_type', 'unknown')
            main_type = main_section.get('semantic_type', 'primary')

            # Heuristic: certain types are naturally divergent
            divergent_types = {'investigation', 'legal_action', 'policy_response', 'aftermath'}
            if section_type in divergent_types:
                return 0.7  # Assume high divergence for these types
            return 0.4  # Moderate default

        # Try LLM classification
        try:
            prompt = f"""Rate semantic shift: 0.0=same event, 0.5=related phase, 1.0=different event.

Main: {main_summary[:250]}
Section: {section_summary[:250]}

Number only:"""

            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Return only a number 0.0-1.0"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )

            score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.debug(f"Semantic shift error (tolerating): {e}")

            # Fallback 2: Keyword heuristic
            investigation_keywords = ['arrest', 'charge', 'investigation', 'police', 'suspect', 'prosecutor']
            policy_keywords = ['ban', 'regulation', 'law', 'government', 'policy', 'legislation']

            section_lower = section_summary.lower()

            if any(kw in section_lower for kw in investigation_keywords):
                return 0.7
            elif any(kw in section_lower for kw in policy_keywords):
                return 0.8

            return 0.5  # Safe default

    def _calculate_page_density(self, section: Dict, event_state: Dict) -> float:
        """Calculate page density signal"""
        section_pages = section.get('page_count', 0)
        total_pages = event_state.get('artifact_count', 1)

        if total_pages == 0:
            return 0.0

        ratio = section_pages / total_pages

        if section_pages == 1:
            return 0.2
        elif section_pages <= 3:
            return 0.5
        elif section_pages <= 5:
            return 0.7
        else:
            return 0.9

    def _deserialize_ontology(self, ontology: Dict) -> Dict:
        """
        Reconstruct FactBelief objects from serialized dicts

        When loading from JSON, FactBelief objects become dicts.
        This reconstructs them so the base enricher can update them.
        """
        if not ontology:
            return {}

        deserialized = {}
        for aspect_name, aspect_data in ontology.items():
            if not isinstance(aspect_data, dict):
                deserialized[aspect_name] = aspect_data
                continue

            deserialized[aspect_name] = {}
            for key, belief_data in aspect_data.items():
                # Check if this is a serialized FactBelief (has timeline, current_value, etc.)
                if isinstance(belief_data, dict) and 'timeline' in belief_data:
                    # Reconstruct FactBelief from dict
                    fact_belief = FactBelief(
                        fact_type=belief_data.get('fact_type', aspect_name),
                        subtype=belief_data.get('subtype', key),
                        initial_value=belief_data.get('current_value'),
                        initial_confidence=belief_data.get('current_confidence', 0.7),
                        source='reconstructed'
                    )
                    # Restore full timeline
                    fact_belief.timeline = belief_data.get('timeline', [])
                    fact_belief.contradictions = belief_data.get('contradictions', [])
                    fact_belief.resolution_history = belief_data.get('resolution_history', [])

                    deserialized[aspect_name][key] = fact_belief
                else:
                    # Not a FactBelief, keep as-is
                    deserialized[aspect_name][key] = belief_data

        return deserialized

    def _create_empty_section(self, name: str, semantic_type: str) -> Dict:
        """Create empty section structure"""
        return {
            'name': name,
            'semantic_type': semantic_type,
            'start': None,
            'end': None,
            'summary': '',
            'ontology': {},
            'page_ids': [],
            'page_count': 0,
            'promotion_score': 0.0,
            'promotion_signals': {},
            'human_weight': 0.0,
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat(),
            'enrichment_timeline': []
        }

    def compute_event_coherence(self, event_state: Dict) -> float:
        """Compute overall event coherence from sections"""
        if 'sections' not in event_state:
            return 0.0

        coherences = []
        for section in event_state['sections'].values():
            section_coherence = self._compute_section_coherence(section)
            coherences.append(section_coherence)

        return sum(coherences) / len(coherences) if coherences else 0.0

    def _compute_section_coherence(self, section: Dict) -> float:
        """Compute coherence for a single section"""
        ontology = section.get('ontology', {})
        if not ontology:
            return 0.0

        confidences = []
        multi_source_count = 0
        total_facts = 0

        for aspect_name, aspect_data in ontology.items():
            if aspect_name in ('story', 'narrative'):
                continue

            if isinstance(aspect_data, dict):
                for key, belief in aspect_data.items():
                    if isinstance(belief, FactBelief):
                        confidences.append(belief.current_confidence)
                        total_facts += 1
                        if len(belief.timeline) > 1:
                            multi_source_count += 1

        if not confidences:
            return 0.5

        avg_confidence = sum(confidences) / len(confidences)
        corroboration_rate = multi_source_count / total_facts if total_facts > 0 else 0.0

        coherence = (avg_confidence * 0.7) + (corroboration_rate * 0.3)
        return round(coherence, 3)
