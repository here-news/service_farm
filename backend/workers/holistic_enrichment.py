"""
Holistic Event Enrichment - event(n) + page → event(n+1)

Philosophy:
- Claims are NOT necessary - work directly with page content
- LLM sees full context (title + description + content)
- Single holistic enrichment per page
- Extracts: corroboration, novel info, contradictions
- Updates entire event ontology incrementally

Vision:
- event(0): First artifact creates draft ontology (low coherence)
- event(n+1): Each artifact enriches ontology, increases coherence
- Entities from semantic_worker add to event surface
"""
import asyncio
import asyncpg
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from openai import AsyncOpenAI
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactBelief:
    """
    Represents our belief about a fact that evolves over time

    NOT a single resolved value, but a timeline with confidence evolution
    """
    def __init__(self, fact_type: str, subtype: str, initial_value: Any, initial_confidence: float = 0.7, source: str = "initial"):
        self.fact_type = fact_type
        self.subtype = subtype

        # Timeline of belief updates
        self.timeline = [{
            "value": initial_value,
            "confidence": initial_confidence,
            "source": source,
            "timestamp": datetime.utcnow().isoformat(),
            "update_type": "initial"
        }]

        # Current belief (most recent/confident)
        self.current_value = initial_value
        self.current_confidence = initial_confidence

        # Track contradictions
        self.contradictions = []
        self.resolution_history = []

    def update_with_new_fact(self, new_value: Any, new_confidence: float, source: str) -> Dict:
        """
        Update belief with new fact - incremental contradiction resolution

        Returns resolution decision for transparency
        """
        # CASE 1: Unanimous - same value
        if new_value == self.current_value:
            # Consensus strengthens confidence
            old_confidence = self.current_confidence
            self.current_confidence = min(self.current_confidence * 1.15, 0.98)

            self.timeline.append({
                "value": new_value,
                "confidence": self.current_confidence,
                "source": source,
                "timestamp": datetime.utcnow().isoformat(),
                "update_type": "consensus"
            })

            resolution = {
                "type": "consensus",
                "action": "confidence_boost",
                "old_confidence": old_confidence,
                "new_confidence": self.current_confidence,
                "rationale": f"Multiple sources agree on {new_value}"
            }

            self.resolution_history.append(resolution)
            return resolution

        # CASE 2: Temporal evolution - increasing pattern
        if isinstance(new_value, (int, float)) and isinstance(self.current_value, (int, float)):
            if new_value > self.current_value:
                # Likely evolution (death toll rising)
                old_value = self.current_value
                self.current_value = new_value
                self.current_confidence = new_confidence

                self.timeline.append({
                    "value": new_value,
                    "confidence": new_confidence,
                    "source": source,
                    "timestamp": datetime.utcnow().isoformat(),
                    "update_type": "evolution"
                })

                resolution = {
                    "type": "temporal_evolution",
                    "action": "update_value",
                    "old_value": old_value,
                    "new_value": new_value,
                    "rationale": f"Value evolved from {old_value} to {new_value}"
                }

                self.resolution_history.append(resolution)
                return resolution

        # CASE 3: Bayesian override - higher confidence
        if new_confidence > self.current_confidence * 1.1:
            old_value = self.current_value
            old_confidence = self.current_confidence

            # Store old belief as potential contradiction
            self.contradictions.append({
                "value": self.current_value,
                "confidence": self.current_confidence,
                "superseded_by": source,
                "reason": "higher_confidence_source"
            })

            self.current_value = new_value
            self.current_confidence = new_confidence * 0.9  # Slight penalty for contradiction

            self.timeline.append({
                "value": new_value,
                "confidence": self.current_confidence,
                "source": source,
                "timestamp": datetime.utcnow().isoformat(),
                "update_type": "override"
            })

            resolution = {
                "type": "bayesian_override",
                "action": "override_value",
                "old_value": old_value,
                "new_value": new_value,
                "old_confidence": old_confidence,
                "new_confidence": self.current_confidence,
                "rationale": f"Higher confidence source ({new_confidence:.2f} > {old_confidence:.2f})"
            }

            self.resolution_history.append(resolution)
            return resolution

        # CASE 4: Unresolved contradiction - flag but keep current belief
        self.contradictions.append({
            "value": new_value,
            "confidence": new_confidence,
            "source": source,
            "flagged": True,
            "reason": "contradicts_current_belief"
        })

        self.timeline.append({
            "value": new_value,
            "confidence": new_confidence,
            "source": source,
            "timestamp": datetime.utcnow().isoformat(),
            "update_type": "contradiction_flagged"
        })

        resolution = {
            "type": "unresolved_contradiction",
            "action": "keep_current",
            "current_value": self.current_value,
            "conflicting_value": new_value,
            "rationale": f"Contradiction: {self.current_value} vs {new_value} - keeping current (higher confidence)"
        }

        self.resolution_history.append(resolution)
        return resolution

    def to_dict(self) -> Dict:
        """Export belief state"""
        return {
            "fact_type": self.fact_type,
            "subtype": self.subtype,
            "current_value": self.current_value,
            "current_confidence": self.current_confidence,
            "timeline": self.timeline,
            "contradictions": self.contradictions,
            "resolution_history": self.resolution_history,
            "update_count": len(self.timeline)
        }


class HolisticEventEnricher:
    """
    Holistic event enrichment: event(n) + page → event(n+1)

    NOT claim-based - works directly with page content
    """

    def __init__(self, openai_client: AsyncOpenAI):
        self.client = openai_client

    async def enrich_event_with_page(
        self,
        event_state: Dict,
        page: Dict
    ) -> Dict:
        """
        Single-shot enrichment: event(n) + page → event(n+1)

        Args:
            event_state: Current event ontology state
            page: New artifact with title, description, content

        Returns:
            Enrichment result with corroboration/novel/contradictions
        """
        # Prepare page content (truncate if too long)
        page_context = self._prepare_page_context(page)

        # Prepare current event state summary
        event_context = self._prepare_event_context(event_state)

        # LLM extraction with structured output
        enrichment = await self._extract_holistic_update(event_context, page_context, page)

        if not enrichment:
            return {"error": "LLM extraction failed"}

        # Apply updates to event state
        update_summary = self._apply_updates_to_ontology(event_state, enrichment, page)

        return {
            "enrichment": enrichment,
            "update_summary": update_summary,
            "page_url": page.get('url'),
            "page_timestamp": page.get('created_at')
        }

    def _prepare_page_context(self, page: Dict) -> str:
        """Prepare page content for LLM (with truncation if needed)"""
        title = page.get('title', 'Untitled')
        description = page.get('description', '')
        content_text = page.get('content_text', '')
        pub_time = page.get('pub_time', 'unknown')

        # Truncate content if too long (keep first 3000 chars)
        if len(content_text) > 3000:
            content_text = content_text[:3000] + "... [truncated]"

        return f"""
TITLE: {title}
DESCRIPTION: {description}
PUB_TIME: {pub_time}
CONTENT:
{content_text}
""".strip()

    def _prepare_event_context(self, event_state: Dict) -> str:
        """Prepare current event state for LLM"""
        if not event_state.get('ontology'):
            return "CURRENT EVENT STATE: Empty (first artifact)"

        ontology = event_state['ontology']
        lines = ["CURRENT EVENT STATE:"]

        # Story (if exists)
        if 'story' in ontology:
            story = ontology['story']
            lines.append("\nCurrent Story:")
            lines.append(f"  {story.get('description', 'N/A')}")

        # Casualties
        if 'casualties' in ontology:
            lines.append("\nCasualties:")
            for subtype, belief in ontology['casualties'].items():
                if isinstance(belief, FactBelief):
                    lines.append(f"  {subtype}: {belief.current_value} (confidence: {belief.current_confidence:.2f})")

        # Timeline
        if 'timeline' in ontology:
            lines.append("\nTimeline:")
            for key, belief in ontology['timeline'].items():
                if isinstance(belief, FactBelief):
                    lines.append(f"  {key}: {belief.current_value} (confidence: {belief.current_confidence:.2f})")

        # Locations
        if 'locations' in ontology:
            lines.append("\nLocations:")
            for key, belief in ontology['locations'].items():
                if isinstance(belief, FactBelief):
                    lines.append(f"  {key}: {belief.current_value} (confidence: {belief.current_confidence:.2f})")

        # Narrative
        if 'narrative' in ontology:
            lines.append("\nNarrative:")
            for key, val in ontology['narrative'].items():
                if val:
                    lines.append(f"  {key}: {val}")

        return "\n".join(lines)

    async def _extract_holistic_update(
        self,
        event_context: str,
        page_context: str,
        page: Dict
    ) -> Optional[Dict]:
        """
        LLM extracts holistic update from page given current event state

        Returns structured update with corroboration/novel/contradictions + synthesized story
        """
        prompt = f"""You are enriching an event ontology with a new page artifact.

{event_context}

NEW PAGE ARTIFACT:
{page_context}

Your task: Extract structured update to the event ontology + synthesize descriptive story.

Return ONLY valid JSON with this structure:
{{
  "story": {{
    "description": "2-3 sentence narrative synthesizing the full event story so far",
    "who": ["Key participants/entities involved"],
    "when": {{"start": "2025-11-26T14:51:00", "end": null, "precision": "exact"}},
    "where": ["Locations involved"],
    "what": "What happened (1-2 sentences)",
    "why": "Causal factors/reasons",
    "how": "Mechanism/process of what happened"
  }},
  "casualties": {{
    "deaths": {{"value": 44, "confidence": 0.9, "evidence": "title states 44 killed"}},
    "injured": {{"value": null, "confidence": 0.0, "evidence": "not mentioned"}},
    "missing": {{"value": 279, "confidence": 0.85, "evidence": "description says 279 missing"}}
  }},
  "timeline": {{
    "start": {{"value": "2025-11-26T14:51:00", "confidence": 0.9, "evidence": "fire broke out at 2:51 p.m."}},
    "milestones": [
      {{"event": "upgraded to level 5", "time": "2025-11-26T18:22:00", "confidence": 0.9}}
    ]
  }},
  "locations": {{
    "primary": {{"value": "Wang Fuk Court", "confidence": 0.95, "evidence": "mentioned multiple times"}},
    "district": {{"value": "Tai Po", "confidence": 1.0, "evidence": "explicitly stated"}}
  }},
  "response": {{
    "evacuations": {{"value": 700, "confidence": 0.85, "evidence": "700 people evacuated"}},
    "arrests": {{"value": 3, "confidence": 0.9, "evidence": "3 arrested for manslaughter"}},
    "firefighter_casualties": {{"value": 1, "confidence": 0.9, "evidence": "one firefighter died"}}
  }},
  "corroboration": [
    "Confirms fire location in Tai Po district",
    "Confirms evacuations around 700 people"
  ],
  "novel": [
    "Death toll increased from 40 to 44",
    "Bamboo scaffolding identified as fire spread mechanism"
  ],
  "contradictions": [
    {{"aspect": "deaths", "old": 40, "new": 44, "resolution": "temporal_evolution"}}
  ]
}}

IMPORTANT:
- Story should be a holistic narrative combining current state + new page
- Extract ONLY explicitly stated facts (no speculation)
- Use null for unknown values
- Confidence 0.9+ for explicit statements, 0.7-0.8 for implied, 0.5-0.6 for uncertain
- Evidence should cite specific phrases from the page
- Identify corroboration (confirms existing), novel (new info), contradictions (conflicts)
"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a factual event enrichment assistant. Extract only verifiable facts from news articles. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()

            enrichment = json.loads(content)
            return enrichment

        except Exception as e:
            logger.error(f"Holistic extraction error: {e}")
            return None

    def _milestones_match(self, m1: Dict, m2: Dict) -> bool:
        """
        Check if two milestones describe the same event

        Uses fuzzy matching on event description
        """
        from difflib import SequenceMatcher

        # Extract event descriptions
        e1 = m1.get('event', '').lower()
        e2 = m2.get('event', '').lower()

        # Check similarity ratio
        similarity = SequenceMatcher(None, e1, e2).ratio()

        # If very similar text, they're the same event
        return similarity > 0.8

    def _apply_updates_to_ontology(
        self,
        event_state: Dict,
        enrichment: Dict,
        page: Dict
    ) -> Dict:
        """
        Apply LLM-extracted updates to event ontology + story

        Returns summary of what was updated
        """
        if 'ontology' not in event_state:
            event_state['ontology'] = {}

        ontology = event_state['ontology']
        source = page.get('url', 'unknown')
        updates = []

        # Update story (synthesized narrative)
        if 'story' in enrichment:
            ontology['story'] = enrichment['story']
            updates.append(f"📖 story updated")

        # Update casualties
        if 'casualties' in enrichment:
            if 'casualties' not in ontology:
                ontology['casualties'] = {}

            for subtype, data in enrichment['casualties'].items():
                if data['value'] is not None:
                    if subtype not in ontology['casualties']:
                        # Initialize new fact
                        ontology['casualties'][subtype] = FactBelief(
                            'casualties', subtype, data['value'], data['confidence'], source
                        )
                        updates.append(f"🆕 casualties.{subtype} = {data['value']}")
                    else:
                        # Update existing belief
                        belief = ontology['casualties'][subtype]
                        resolution = belief.update_with_new_fact(
                            data['value'], data['confidence'], source
                        )

                        emoji = {
                            "consensus": "✓",
                            "temporal_evolution": "📈",
                            "bayesian_override": "🔄",
                            "unresolved_contradiction": "⚠️"
                        }.get(resolution['type'], "•")

                        updates.append(f"{emoji} casualties.{subtype}: {resolution['rationale']}")

        # Update timeline
        if 'timeline' in enrichment:
            if 'timeline' not in ontology:
                ontology['timeline'] = {}

            for key, data in enrichment['timeline'].items():
                if key == 'milestones':
                    # Handle milestones list with smart resolution
                    if 'milestones' not in ontology['timeline']:
                        ontology['timeline']['milestones'] = []

                    # Resolve each new milestone against existing ones
                    for new_milestone in data:
                        merged = False

                        # Check if this milestone matches an existing one (same event description)
                        for i, existing in enumerate(ontology['timeline']['milestones']):
                            if self._milestones_match(new_milestone, existing):
                                # Same event, resolve temporal contradiction
                                # Keep the more precise/confident time
                                if new_milestone.get('confidence', 0.7) > existing.get('confidence', 0.7):
                                    ontology['timeline']['milestones'][i] = new_milestone
                                    updates.append(f"🔄 timeline.milestone resolved: {new_milestone['event']}")
                                merged = True
                                break

                        if not merged:
                            # Truly new milestone
                            ontology['timeline']['milestones'].append(new_milestone)
                            updates.append(f"🆕 timeline.milestone: {new_milestone['event']}")
                elif data['value'] is not None:
                    if key not in ontology['timeline']:
                        ontology['timeline'][key] = FactBelief(
                            'timeline', key, data['value'], data['confidence'], source
                        )
                        updates.append(f"🆕 timeline.{key} = {data['value']}")
                    else:
                        belief = ontology['timeline'][key]
                        resolution = belief.update_with_new_fact(
                            data['value'], data['confidence'], source
                        )
                        updates.append(f"• timeline.{key}: {resolution['rationale']}")

        # Update locations
        if 'locations' in enrichment:
            if 'locations' not in ontology:
                ontology['locations'] = {}

            for key, data in enrichment['locations'].items():
                if data['value'] is not None:
                    if key not in ontology['locations']:
                        ontology['locations'][key] = FactBelief(
                            'locations', key, data['value'], data['confidence'], source
                        )
                        updates.append(f"🆕 locations.{key} = {data['value']}")
                    else:
                        belief = ontology['locations'][key]
                        resolution = belief.update_with_new_fact(
                            data['value'], data['confidence'], source
                        )
                        updates.append(f"• locations.{key}: {resolution['rationale']}")

        # Update response
        if 'response' in enrichment:
            if 'response' not in ontology:
                ontology['response'] = {}

            for key, data in enrichment['response'].items():
                if data['value'] is not None:
                    if key not in ontology['response']:
                        ontology['response'][key] = FactBelief(
                            'response', key, data['value'], data['confidence'], source
                        )
                        updates.append(f"🆕 response.{key} = {data['value']}")
                    else:
                        belief = ontology['response'][key]
                        resolution = belief.update_with_new_fact(
                            data['value'], data['confidence'], source
                        )
                        updates.append(f"• response.{key}: {resolution['rationale']}")

        return {
            "updates": updates,
            "corroboration": enrichment.get('corroboration', []),
            "novel": enrichment.get('novel', []),
            "contradictions": enrichment.get('contradictions', [])
        }

    async def enrich_event_incrementally(
        self,
        event_id: str,
        pages: List[Dict]
    ) -> Dict:
        """
        Process pages incrementally: event(0) → event(1) → ... → event(n)

        Args:
            event_id: Event UUID
            pages: List of page artifacts in chronological order

        Returns:
            Final event state with full ontology
        """
        # Initialize event state
        event_state = {
            "event_id": event_id,
            "ontology": {},
            "artifact_count": 0,
            "enrichment_timeline": []
        }

        logger.info(f"\n{'='*80}")
        logger.info(f"HOLISTIC ENRICHMENT: Processing {len(pages)} pages")
        logger.info(f"{'='*80}\n")

        for i, page in enumerate(pages, 1):
            logger.info(f"📄 Page {i}/{len(pages)}: {page.get('title', 'Untitled')}")
            logger.info(f"   URL: {page.get('url', 'unknown')}")
            logger.info(f"   Created: {page.get('created_at')}\n")

            # Holistic enrichment
            result = await self.enrich_event_with_page(event_state, page)

            if 'error' in result:
                logger.warning(f"   ⚠️ {result['error']}")
                continue

            # Log updates
            update_summary = result['update_summary']

            if update_summary['corroboration']:
                logger.info(f"   ✓ CORROBORATION:")
                for item in update_summary['corroboration']:
                    logger.info(f"      - {item}")

            if update_summary['novel']:
                logger.info(f"   ✨ NOVEL:")
                for item in update_summary['novel']:
                    logger.info(f"      - {item}")

            if update_summary['contradictions']:
                logger.info(f"   ⚠️  CONTRADICTIONS:")
                for item in update_summary['contradictions']:
                    logger.info(f"      - {item}")

            logger.info(f"\n   Updates:")
            for update in update_summary['updates']:
                logger.info(f"      {update}")

            # Store in timeline
            event_state['enrichment_timeline'].append({
                "artifact_index": i,
                "page_url": result['page_url'],
                "page_timestamp": result['page_timestamp'],
                "corroboration_count": len(update_summary['corroboration']),
                "novel_count": len(update_summary['novel']),
                "contradiction_count": len(update_summary['contradictions']),
                "updates": update_summary['updates']
            })

            event_state['artifact_count'] = i
            logger.info("")

        # Compute final coherence
        event_state['coherence'] = self._compute_coherence(event_state)

        return event_state

    def _compute_coherence(self, event_state: Dict) -> float:
        """
        Compute event coherence based on ontology state

        Factors:
        - Average confidence across FactBeliefs
        - Corroboration rate (how many facts confirmed by multiple sources)
        - Completeness (how many aspects filled in)
        """
        ontology = event_state.get('ontology', {})
        if not ontology:
            return 0.0

        confidences = []
        multi_source_count = 0
        total_facts = 0

        for aspect_name, aspect_data in ontology.items():
            if aspect_name in ('story', 'narrative'):
                continue  # Skip story/narrative (descriptive, not factual)

            if isinstance(aspect_data, dict):
                for key, belief in aspect_data.items():
                    if isinstance(belief, FactBelief):
                        confidences.append(belief.current_confidence)
                        total_facts += 1
                        if len(belief.timeline) > 1:
                            multi_source_count += 1

        if not confidences:
            return 0.0

        avg_confidence = sum(confidences) / len(confidences)
        corroboration_rate = multi_source_count / total_facts if total_facts > 0 else 0.0

        # Weighted combination
        coherence = (avg_confidence * 0.7) + (corroboration_rate * 0.3)

        return round(coherence, 3)

    def generate_big_picture(self, event_state: Dict) -> str:
        """
        Generate human-readable big picture summary
        """
        output = []
        output.append("\n" + "="*80)
        output.append("BIG PICTURE: Event Ontology")
        output.append("="*80 + "\n")

        output.append(f"Artifacts processed: {event_state['artifact_count']}")
        output.append(f"Coherence: {event_state.get('coherence', 0.0):.2%}\n")

        ontology = event_state.get('ontology', {})

        # Story (synthesized narrative)
        if 'story' in ontology:
            story = ontology['story']
            output.append("\n📖 STORY")
            output.append("-" * 80)
            output.append(f"  {story.get('description', 'N/A')}\n")
            if story.get('who'):
                output.append(f"  WHO: {', '.join(story['who'][:5])}")
            if story.get('when'):
                when = story['when']
                output.append(f"  WHEN: {when.get('start', 'unknown')} to {when.get('end', 'ongoing')} ({when.get('precision', 'unknown')} precision)")
            if story.get('where'):
                output.append(f"  WHERE: {', '.join(story['where'])}")
            if story.get('what'):
                output.append(f"  WHAT: {story['what']}")
            if story.get('why'):
                output.append(f"  WHY: {story['why']}")
            if story.get('how'):
                output.append(f"  HOW: {story['how']}")

        # Casualties
        if 'casualties' in ontology:
            output.append("\n📊 CASUALTIES")
            output.append("-" * 80)
            for subtype, belief in ontology['casualties'].items():
                output.append(f"  {subtype}: {belief.current_value} (confidence: {belief.current_confidence:.2%}, updates: {len(belief.timeline)})")
                if len(belief.timeline) > 1:
                    values = [t['value'] for t in belief.timeline]
                    output.append(f"    Evolution: {' → '.join(map(str, values))}")

        # Timeline
        if 'timeline' in ontology:
            output.append("\n⏰ TIMELINE")
            output.append("-" * 80)
            for key, belief in ontology['timeline'].items():
                if key == 'milestones':
                    output.append(f"  milestones: {len(belief)} events")
                    for milestone in belief[:3]:  # Show first 3
                        output.append(f"    - {milestone.get('event')} at {milestone.get('time')}")
                elif isinstance(belief, FactBelief):
                    output.append(f"  {key}: {belief.current_value} (confidence: {belief.current_confidence:.2%})")

        # Locations
        if 'locations' in ontology:
            output.append("\n📍 LOCATIONS")
            output.append("-" * 80)
            for key, belief in ontology['locations'].items():
                output.append(f"  {key}: {belief.current_value} (confidence: {belief.current_confidence:.2%})")

        # Response
        if 'response' in ontology:
            output.append("\n🚒 RESPONSE")
            output.append("-" * 80)
            for key, belief in ontology['response'].items():
                output.append(f"  {key}: {belief.current_value} (confidence: {belief.current_confidence:.2%})")

        output.append("\n" + "="*80 + "\n")

        return "\n".join(output)


async def test_holistic_enrichment(event_id: str):
    """Test holistic enrichment on Hong Kong fire event"""
    import os

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    enricher = HolisticEventEnricher(client)

    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        user=os.getenv('POSTGRES_USER', 'herenews_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
        database=os.getenv('POSTGRES_DB', 'herenews'),
        min_size=1,
        max_size=2
    )

    async with db_pool.acquire() as conn:
        # Get event
        event = await conn.fetchrow("""
            SELECT id, title FROM core.events WHERE id = $1
        """, event_id)

        print(f"\nTesting holistic enrichment: {event['title']}\n")

        # Get pages in chronological order
        pages_raw = await conn.fetch("""
            SELECT
                p.id, p.url, p.title, p.description, p.content_text,
                p.pub_time, p.created_at
            FROM core.pages p
            JOIN core.page_events pe ON p.id = pe.page_id
            WHERE pe.event_id = $1
            ORDER BY p.created_at
        """, event_id)

        pages = [dict(p) for p in pages_raw]

        # Process incrementally
        event_state = await enricher.enrich_event_incrementally(event_id, pages)

        # Generate big picture
        big_picture = enricher.generate_big_picture(event_state)
        print(big_picture)

        # Store in enriched_json
        def serialize_for_json(obj):
            if isinstance(obj, FactBelief):
                return obj.to_dict()
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        enriched_json_str = json.dumps({
            "holistic_enrichment": {
                "ontology": {
                    aspect_name: {
                        key: serialize_for_json(val)
                        for key, val in aspect_data.items()
                    } if isinstance(aspect_data, dict) else aspect_data
                    for aspect_name, aspect_data in event_state['ontology'].items()
                },
                "coherence": event_state['coherence'],
                "artifact_count": event_state['artifact_count'],
                "enrichment_timeline": event_state['enrichment_timeline'],
                "enriched_at": datetime.utcnow().isoformat()
            }
        }, default=str)

        await conn.execute("""
            UPDATE core.events
            SET enriched_json = $2, coherence = $3
            WHERE id = $1
        """, event_id, enriched_json_str, event_state['coherence'])

        print("✅ Stored holistic enrichment in enriched_json\n")

    await db_pool.close()


if __name__ == '__main__':
    import sys
    event_id = sys.argv[1] if len(sys.argv) > 1 else '0c6bc931-94ea-4e14-9fc9-5c5ed3ebeb2a'
    asyncio.run(test_holistic_enrichment(event_id))
