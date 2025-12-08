"""
Streamlined Semantic Analyzer - Extract atomic claims with entities from cleaned content
"""
import asyncio
from typing import Dict, Any, List, Optional
from openai import OpenAI
import os
import json
from datetime import datetime
import hashlib
import time
import logging

logger = logging.getLogger(__name__)

class EnhancedSemanticAnalyzer:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _with_retry(self, fn, *args, **kwargs):
        """Execute function with retry logic for API failures"""
        for i in range(2):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if i == 0 and any(t in str(e) for t in ["429", "Rate limit", "timeout", "500", "502", "503"]):
                    time.sleep(1.2)
                    continue
                raise

    async def extract_enhanced_claims(self, page_meta: Dict[str, Any],
                                    page_text: List[Dict[str, str]],
                                    url: str, fetch_time: str = None,
                                    lang: str = "en", about_text: str = None) -> Dict[str, Any]:
        """
        Extract atomic, attributable claims with entities from content

        Args:
            page_meta: {title, byline, pub_time, site}
            page_text: [{selector, text}] - content blocks
            url: canonical URL
            fetch_time: when content was fetched
            lang: language code
            about_text: Concise summary for story matching (optional)

        Returns:
            {
                "claims": [claim objects with WHO/WHERE/WHEN],
                "excluded_claims": [claims that failed premise checks],
                "entities": {people, locations, organizations, time_references},
                "gist": "one sentence summary",
                "confidence": float,
                "about_embedding": [1536-dim embedding vector],
                "token_usage": {prompt_tokens, completion_tokens, total_tokens}
            }
        """
        try:
            if not page_text:
                return self._empty_extraction("No page content provided")

            # Prepare content
            full_text = self._prepare_content(page_meta, page_text)

            # Extract claims with entities
            claims_response = await self._extract_atomic_claims_with_ner(
                full_text, page_meta, url, lang
            )

            # Apply premise scoring (no filtering - score instead of exclude)
            # Defensive: ensure claims is a list, not None
            all_claims = claims_response.get('claims', []) or []
            if not isinstance(all_claims, list):
                logger.warning(f"⚠️  Claims response malformed: {type(all_claims)}, expected list")
                all_claims = []

            logger.info(f"🔍 LLM returned {len(all_claims)} raw claims before premise filtering")

            admitted_claims = []
            excluded_claims = []

            for claim in all_claims:
                # Score claim based on premise checks (instead of binary pass/fail)
                adjusted_confidence, failed_checks, verification_needed = self._score_claim_with_premises(claim)

                # Update claim with adjusted confidence and verification flags
                claim['confidence'] = adjusted_confidence
                claim['failed_checks'] = failed_checks
                claim['verification_needed'] = verification_needed

                # Only exclude if confidence drops below 0.3 (very low quality)
                if adjusted_confidence >= 0.3:
                    admitted_claims.append(claim)
                else:
                    claim['excluded_reason'] = f"Confidence too low ({adjusted_confidence:.2f}): {', '.join(failed_checks)}"
                    excluded_claims.append(claim)

            # Extract entities only from admitted claims (for backwards compatibility)
            entities = self._extract_entities_from_claims(admitted_claims)

            # Get entity descriptions from gpt-4o (new format: {"PERSON:Name": "description", ...})
            entity_descriptions = claims_response.get('entities', {})
            # Filter to only keep description entries (not the old format lists)
            if isinstance(entity_descriptions, dict):
                entity_descriptions = {k: v for k, v in entity_descriptions.items()
                                       if isinstance(v, str) and ':' in k}

            # Extract primary event time from claims
            primary_event_time = self._extract_primary_event_time(admitted_claims, page_meta)

            print(f"✅ Claims: {len(admitted_claims)} admitted, {len(excluded_claims)} excluded")
            if primary_event_time:
                print(f"📅 Primary event time: {primary_event_time.get('time')} (precision: {primary_event_time.get('precision')})")

            # Generate embedding for story matching
            about_embedding = []
            if about_text:
                about_embedding = await self.generate_about_embedding(about_text)

            # Log entity descriptions for debugging
            logger.info(f"📝 Entity descriptions from gpt-4o: {list(entity_descriptions.keys()) if entity_descriptions else 'none'}")

            # Extract entity relationships from LLM response
            entity_relationships = claims_response.get('entity_relationships', [])
            if entity_relationships:
                logger.info(f"🔗 Entity relationships extracted: {len(entity_relationships)}")
                for rel in entity_relationships[:3]:  # Log first 3
                    logger.info(f"   {rel.get('subject')} --[{rel.get('predicate')}]--> {rel.get('object')}")

            return {
                "claims": admitted_claims,
                "excluded_claims": excluded_claims,
                "entities": entities,  # OLD format for backward compatibility
                "entity_descriptions": entity_descriptions,  # NEW format: {"PERSON:Name": "description"}
                "entity_relationships": entity_relationships,  # NEW: hierarchical entity relationships
                "gist": claims_response.get('gist', 'No summary available'),
                "confidence": claims_response.get('overall_confidence', 0.5),
                "notes_unsupported": claims_response.get('notes_unsupported', []),
                "about_embedding": about_embedding,
                "primary_event_time": primary_event_time,
                "token_usage": claims_response.get('token_usage', {})
            }

        except Exception as e:
            print(f"❌ Claims extraction failed: {e}")
            return self._empty_extraction(f"Extraction failed: {str(e)}")

    def _prepare_content(self, page_meta: Dict[str, Any],
                        page_text: List[Dict[str, str]]) -> str:
        """Prepare content for LLM analysis"""
        content_parts = []

        # Add metadata
        if page_meta.get('title'):
            content_parts.append(f"Title: {page_meta['title']}")
        if page_meta.get('byline'):
            content_parts.append(f"Author: {page_meta['byline']}")
        if page_meta.get('pub_time'):
            content_parts.append(f"Published: {page_meta['pub_time']}")
        if page_meta.get('site'):
            content_parts.append(f"Source: {page_meta['site']}")

        content_parts.append("---")

        # Add text content
        for text_block in page_text[:20]:  # Limit to first 20 blocks
            text = text_block.get('text', '').strip()
            if text and len(text) > 20:
                content_parts.append(text)

        return "\n".join(content_parts)

    async def _extract_atomic_claims_with_ner(self, content: str,
                                            page_meta: Dict[str, Any],
                                            url: str, lang: str = "en") -> Dict[str, Any]:
        """Extract atomic claims with named entities using LLM"""

        # Log content being sent to LLM
        logger.info(f"📝 Sending {len(content)} chars to LLM. Preview: {content[:200]}...")

        system_prompt = f"""You are a fact extractor for structured journalism. Extract atomic, verifiable claims that meet minimum journalistic standards.

Source language: {lang}
**CRITICAL: Extract ALL claims in English, regardless of source language.**
- Translate claim text to English
- Keep entity names as-is (don't translate proper nouns)
- Use English entity type prefixes (PERSON, ORG, GPE, LOCATION)
- Preserve original meaning and attribution

Extract ALL factual claims from the article - don't filter at extraction time. The system will
assess quality later. Include:
- Direct observations (facts stated in article): "36 people died", "fire broke out at 2:51pm"
- Reported speech (attributed statements): "Official said..."
- Allegations (criminal/fault claims): "Accused of..."

Guidelines for extraction (not hard filters):
1. **Attribution**: Prefer claims with named sources, but include unattributed facts too
2. **Temporal context**: Include event time when available
3. **Modality clarity**: Distinguish observations vs. reported speech vs. allegations
4. **Evidence signal**: Note if there are quotes, documents, photos, statements
5. **Hedging filter**: Include hedged claims but mark appropriately
6. **Controversy guard**: Include all claims - quality scoring happens later

Extract up to 15 claims per article. Prioritize important facts and statements.

For each claim:
- TEXT: Full claim with attribution **IN ENGLISH** (for reported_speech, include "X said Y" in the text!)
  Examples:
  • observation: "Biden signed the infrastructure bill"
  • reported_speech: "White House spokesperson said Trump has legal authority" ← Include attribution!
  • allegation: "Prosecutors accused Smith of fraud"

  **TRANSLATE to English if source is not English!**
  Example (Russian → English): "Politico утверждает..." → "Politico claims that..."

  CRITICAL: Don't extract just the CONTENT of quotes - include WHO said it in the TEXT field!

- WHO: SPECIFIC named people/organizations (PERSON:/ORG: prefix) - ONLY if article provides specific names
  ⚠️  Use FULL SPECIFIC names, NOT generic terms!
  ✅ CORRECT: "ORG:Israel Defense Forces", "ORG:Fire Services Department", "PERSON:John Lee Ka-chiu"
  ❌ WRONG: "ORG:Military", "ORG:Government", "ORG:Officials", "ORG:Police", "ORG:Authorities", "ORG:Firefighters"

  **FOR reported_speech/opinion: The speaker/source MUST be in WHO!**
  ✅ "Jane Smith says X" → WHO = ["PERSON:Jane Smith"]
  ✅ "Activist Tom Wong said..." → WHO = ["PERSON:Tom Wong"]
  ✅ "Critics argue..." + article names specific critic → WHO = ["PERSON:CriticName"]

  **CRITICAL RULE: If article only says generic terms like "authorities", "officials", "police", "firefighters" WITHOUT naming the specific organization, LEAVE WHO EMPTY or use ONLY the named person if given**

  Examples:
  ✅ "Authorities said..." + article mentions "Hong Kong Fire Services Department" → "ORG:Hong Kong Fire Services Department"
  ✅ "Fire chief John Lee said..." → "PERSON:John Lee"
  ❌ "Officials confirmed..." + no org name in article → WHO = [] (empty, don't extract!)
  ❌ "The BBC reported..." → WHO = [] (news orgs are NEWS SOURCES, not speakers to extract)

- WHERE: Specific places at ALL granularities (GPE:/LOCATION: prefix)
  ⚠️  Extract ALL location levels: buildings, streets, venues, neighborhoods, cities, regions, countries
  ✅ CORRECT: "LOCATION:White House", "LOCATION:Main Street", "LOCATION:Tiananmen Square", "GPE:Beijing", "GPE:New York"
  ❌ WRONG: Extract only country/city and ignore specific buildings or streets

  **Example: "Fire at Empire State Building in New York" → ["LOCATION:Empire State Building", "GPE:New York"]**

- WHEN: Event time with precision level matching temporal granularity

  **PRECISION LEVELS:**
  - "hour" = Specific time with hours/minutes: "at 2:51 p.m.", "at 14:51", "at noon"
  - "day" = Date only, no specific time: "on November 26", "Wednesday", "yesterday"
  - "month" = Month/year only: "in November 2025", "last month"
  - "year" = Year only: "in 2023", "last year"
  - "approximate" = Vague timeframe: "recently", "this week", "around Tuesday"

  **TIMEZONE INFERENCE:**
  - Infer timezone from article location context (e.g., Hong Kong → +08:00, New York → -05:00/-04:00)
  - If article explicitly mentions timezone (e.g., "2 p.m. EST"), extract it
  - Default to UTC (+00:00) if location/timezone unclear

  **PRIORITY ORDER FOR TEMPORAL EXTRACTION:**

  1. **EXPLICIT TIME IN CLAIM TEXT** (highest priority)
     If claim mentions specific time: "at 2:51 p.m.", "Wednesday at 14:51", "on Sept 26 at noon"
     → Extract that date/time with precision="hour"

     Examples:
     - "Fire broke out at 2:51 p.m." (article published 2025-11-26 09:39)
       → WHEN = {{date: "2025-11-26", time: "14:51:00", precision: "hour"}}

     - "Meeting scheduled for Wednesday at 3pm" (article published 2025-11-26)
       → WHEN = {{date: "2025-11-26", time: "15:00:00", precision: "hour"}}

  2. **EXPLICIT DATE (no specific time)** (second priority)
     If claim mentions date without specific time: "on November 26", "on Sept 26, 2019"
     → Extract date with precision="day"

     Examples:
     - "Fire broke out at Wang Fuk Court on November 26" (article published 2025-11-26)
       → WHEN = {{date: "2025-11-26", time: null, precision: "day"}}

     - "JPMorgan agreed to pay $290 million in 2023"
       → WHEN = {{date: "2023", time: null, precision: "year"}}

  3. **RECENT/PRESENT ACTION** (third priority)
     If claim has EXPLICIT present tense or "today" marker: "today X happened", "announced today"
     → Use article publication date

     Examples (article published 2025-10-31):
     - "According to newly unsealed records, JPMorgan reported suspicious activity in 2019"
       → WHEN = 2025-10-31 (records NEWLY unsealed today, NOT 2019)

     - "Judge today ordered the unsealing of documents"
       → WHEN = 2025-10-31 (explicitly says "today")

  4. **NO TEMPORAL INFO** (fallback)
     If claim has NO temporal markers at all
     → Leave WHEN empty/null (DO NOT guess or use pub_time!)

  **CRITICAL TIMESTAMP EXTRACTION RULES:**
  • ONLY extract times EXPLICITLY stated in the article text
  • If NO time is mentioned for the claim, leave WHEN field empty/null
  • DO NOT use pub_time as default - only use it if the claim text says "today", "yesterday", "recently" referring to publication date
  • DO NOT guess or infer times not explicitly stated

  **MODALITY-SPECIFIC GUIDANCE:**
  • observation: When the fact/event occurred (extract ONLY if text says "at 2:51 p.m.", "on Nov 26", etc.)
  • reported_speech: When the STATEMENT was made (extract ONLY if text says "said on Tuesday", "announced yesterday", etc.)
  • allegation: When the accusation was made (extract ONLY if explicitly stated)
  • opinion: When opinion was expressed (extract ONLY if explicitly stated)

  **Example (article published Dec 4 about Nov 26 fire):**
  ✅ "Fire broke out at 2:51 p.m. on November 26" → WHEN = {{date: "2025-11-26", time: "14:51:00", precision: "hour"}}
  ✅ "Fire broke out on November 26" → WHEN = {{date: "2025-11-26", time: null, precision: "day"}}
  ✅ "159 people confirmed dead in the fire" → WHEN = null (no time mentioned in claim)
  ❌ WRONG: Using Dec 4 pub_time for claim about Nov 26 event

  **⚠️ For breaking news (fires, accidents, attacks), extract specific times like "at 2:51 p.m." or "on November 26" from the text!**
- MODALITY: Choose ONE (4 types only):

  1. **observation** - Objectively verifiable fact or event
     Examples: "Biden signed the bill", "Fire destroyed building", "Biden is president"
     Use when: Direct factual statement, not attributed to someone saying it

  2. **reported_speech** - Someone said/stated/claimed something
     Examples: "Spokesperson said X", "Report states Y", "Source claims Z"
     Use when: ANY quote, statement, or claim attributed to a source
     CRITICAL: If someone SAID it → reported_speech (even if they said a fact!)
     Example: "Spokesperson said Biden is president" → reported_speech, NOT observation

  3. **allegation** - Unproven accusation (legal/criminal context)
     Examples: "Accused of fraud", "Allegedly murdered"
     Requires: Formal attribution (court, police, prosecutor)

  4. **opinion** - Subjective view, prediction, value judgment
     Examples: "This is a disaster", "Policy will fail"
     Use when: Editorial, speculation, subjective statement

  RULE: Attribution beats content! If someone said a fact → reported_speech, not observation.

- EVIDENCE: Artifacts referenced (document, video, photo, statement, testimony)
- CONFIDENCE: 0.0-1.0 based on evidence strength"""

        user_prompt = f"""Extract atomic claims from this content:

METADATA:
Title: {page_meta.get('title', 'Unknown')}
Site: {page_meta.get('site', 'Unknown')}
Published: {page_meta.get('pub_time', 'Unknown')}

CONTENT:
{content}

Return JSON:
{{
    "claims": [
        {{
            "text": "Full claim with attribution IN ENGLISH - e.g., 'Spokesperson said X' for reported_speech (NOT just 'X')",
            "text_original": "Original language text (only if source is not English, otherwise omit)",
            "who": ["PERSON:Name", "ORG:Organization"],
            "where": ["GPE:Location", "LOCATION:Place"],
            "when": {{
                "date": "YYYY-MM-DD or YYYY or null",
                "time": "HH:MM:SS or null",
                "precision": "hour|day|month|year|approximate",
                "timezone": "+HH:MM or -HH:MM (infer from location, e.g., Hong Kong=+08:00, NYC=-05:00)",
                "event_time": "descriptive timestamp",
                "temporal_context": "timing description from text"
            }},
            "modality": "observation|reported_speech|allegation|opinion",
            "evidence_references": ["statement", "document", "photo"],
            "confidence": 0.85
        }}
    ],
    "entities": {{
        "PERSON:Name": "Location + role/occupation who key_action (e.g., 'Hong Kong activist who campaigns for democracy')",
        "ORG:Name": "Location + type that function (e.g., 'Hong Kong government department overseeing building safety')"
    }},
    "entity_relationships": [
        {{
            "subject": "TYPE:Entity1",
            "predicate": "RELATIONSHIP_TYPE",
            "object": "TYPE:Entity2"
        }}
    ],
    "gist": "One sentence summary IN ENGLISH",
    "overall_confidence": 0.8,
    "notes_unsupported": ["Weak statements not meeting standards"]
}}

ENTITIES FIELD: For each unique entity in who/where fields, provide a concise description (max 15 words):
- PERSON: "[Location] [role] who [action]" - e.g., "Hong Kong activist who campaigns for democracy"
- ORG: "[Location] [type] that [function]" - e.g., "US tech company developing AI"
- GPE/LOCATION: "[Type] in [parent location]" - e.g., "District in northern Hong Kong"
IMPORTANT: Always include the entity's location/country when known or inferable from article context.

ENTITY_RELATIONSHIPS FIELD: Extract relationships between entities mentioned in the article.
Relationship types:
- PART_OF: Physical containment (block in building, room in floor, floor in building, district in city)
  Examples: "LOCATION:Block 6" PART_OF "LOCATION:Wang Fuk Court", "LOCATION:16th Floor" PART_OF "LOCATION:Block 7"
- LOCATED_IN: Geographic containment (building in city, neighborhood in district, city in country)
  Examples: "LOCATION:Wang Fuk Court" LOCATED_IN "GPE:Sha Tin", "GPE:Sha Tin" LOCATED_IN "GPE:Hong Kong"
- WORKS_FOR: Employment relationship (person works for organization)
  Examples: "PERSON:John Lee" WORKS_FOR "ORG:Hong Kong Government"
- MEMBER_OF: Membership (person is member of organization/group)
  Examples: "PERSON:Jane Chan" MEMBER_OF "ORG:Democratic Party"
- AFFILIATED_WITH: Other organizational affiliation

Extract relationships even if implied (not explicitly stated).
For buildings with blocks/floors/units, infer PART_OF relationships from context.
Use FULL entity references with TYPE: prefix (e.g., "LOCATION:Block 6", not just "Block 6").

Only include claims passing ALL 6 criteria above. Use notes_unsupported for interesting but weak statements."""

        try:
            response = await asyncio.to_thread(
                self._with_retry,
                self.openai_client.chat.completions.create,
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=4000  # Increased from 2000 to prevent JSON truncation
            )

            raw_response = response.choices[0].message.content
            logger.info(f"📥 Raw OpenAI response ({len(raw_response)} chars): {raw_response[:500]}...")

            result = json.loads(raw_response)

            # Debug: Check result immediately after parsing
            claims_from_json = result.get('claims', [])
            logger.info(f"🔬 Immediately after JSON parse: claims type={type(claims_from_json)}, count={len(claims_from_json) if isinstance(claims_from_json, list) else 'N/A'}")

            # Log what LLM returned for debugging
            logger.info(f"🔍 LLM returned: claims type={type(result.get('claims'))}, count={len(result.get('claims', []) or [])}")

            # Defensive: ensure claims is a list
            if result.get('claims') is None:
                logger.warning("⚠️  LLM returned null claims, replacing with empty list")
                result['claims'] = []
            elif not isinstance(result.get('claims'), list):
                logger.warning(f"⚠️  LLM returned non-list claims: {type(result.get('claims'))}, replacing with empty list")
                result['claims'] = []

            # Extract token usage
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "model": "gpt-4o"
            }

            # Normalize claims
            result = self._normalize_claims(result, url)
            result['token_usage'] = token_usage

            # Debug: Check result after normalization
            logger.info(f"🔬 After normalization: claims count={len(result.get('claims', []))}")

            return result

        except Exception as e:
            logger.error(f"❌ LLM extraction failed: {e}", exc_info=True)
            return {
                "claims": [],
                "gist": "Extraction failed",
                "overall_confidence": 0.1,
                "notes_unsupported": [f"Error: {str(e)}"],
                "token_usage": {}
            }

    def _normalize_claims(self, result: Dict[str, Any], url: str) -> Dict[str, Any]:
        """Normalize LLM output with deterministic IDs and calibrated confidence"""
        claims = result.get('claims', []) or []
        if not isinstance(claims, list):
            logger.warning(f"⚠️  _normalize_claims received non-list: {type(claims)}")
            claims = []

        current_time_iso = datetime.now().isoformat()

        for claim in claims:
            # Generate deterministic ID
            claim_text = claim.get('text', '')
            claim['id'] = self._claim_id(claim_text, url)

            # Normalize temporal info
            when_info = claim.get('when', {})
            if when_info and isinstance(when_info, dict):
                claim['when'] = self._normalize_when(when_info, current_time_iso)

            # Ensure confidence is valid
            confidence = claim.get('confidence', 0.7)
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                confidence = 0.7
            claim['confidence'] = round(min(max(float(confidence), 0.0), 0.99), 3)

        # Update result with cleaned claims
        result['claims'] = claims

        # Add metadata
        result['extraction_timestamp'] = current_time_iso
        result['source_url'] = url

        return result

    def _claim_id(self, text: str, url: str, version: str = "1.0") -> str:
        """Generate deterministic claim ID"""
        h = hashlib.sha256(f"{version}|{url}|{text}".encode()).hexdigest()[:16]
        return f"clm_{h}"

    def _normalize_when(self, when: dict, reported_time_iso: str) -> dict:
        """Normalize temporal information to consistent format"""
        normalized = {
            "date": when.get("date") or None,
            "time": when.get("time") or None,
            "precision": when.get("precision") or "approximate",
            "timezone": when.get("timezone") or None,
            "event_time": when.get("event_time") or None,
            "reported_time": reported_time_iso,
            "temporal_context": when.get("temporal_context") or None
        }

        # Create ISO timestamp if both date and time exist
        # SIMPLE FIX: Use timezone if provided, default to UTC
        # TODO(GH issue): Long-term epistemic concern - timezone inference accuracy
        if normalized["date"] and normalized["time"]:
            timezone = normalized["timezone"] or "+00:00"  # Default to UTC
            # Convert +HH:MM format to ISO format
            normalized["event_time_iso"] = f'{normalized["date"]}T{normalized["time"]}{timezone}'

        return normalized

    def _score_claim_with_premises(self, claim: Dict[str, Any]) -> tuple[float, List[str], bool]:
        """
        Score claim based on journalistic standards (confidence adjustment instead of filtering)

        Strategy: Start with LLM confidence, apply penalties for failed checks
        - No information loss: Include claim even if it fails checks
        - Lower confidence for missing attribution/evidence
        - Flag for user verification if standards not met

        6 checks with penalties:
        1. Attribution (-0.15): Named source missing
        2. Temporal context (-0.10): Event time missing
        3. Modality clarity (-0.15): Not specified
        4. Evidence signal (-0.10): No artifacts referenced
        5. Hedging filter (-0.20): Vague unattributed language
        6. Controversy guard (-0.25): Criminal claims without proper source

        Returns: (adjusted_confidence, failed_checks, verification_needed)
        """
        text = claim.get('text', '').lower()
        who = claim.get('who', []) or []
        when = claim.get('when', {}) or {}
        modality = claim.get('modality', '') or ''
        evidence = claim.get('evidence_references', []) or []
        base_confidence = claim.get('confidence', 0.7) or 0.7

        failed_checks = []
        penalty = 0.0

        # Check 1: Attribution - named source preferred
        if not who or all(":" not in w for w in who):
            failed_checks.append("attribution")
            penalty += 0.15

        # Check 2: Temporal context - event time preferred
        if not when or not when.get('date'):
            failed_checks.append("temporal")
            penalty += 0.10

        # Check 3: Modality clarity - must be specified
        valid_modalities = ['observation', 'reported_speech', 'allegation', 'opinion']
        if not modality or modality not in valid_modalities:
            failed_checks.append("modality")
            penalty += 0.15

        # Check 4: Evidence signal - artifacts preferred
        if not evidence or len(evidence) == 0:
            failed_checks.append("evidence")
            penalty += 0.10

        # Check 5: Hedging filter - vague language penalized
        hedging = ["reportedly", "allegedly", "might have", "may have", "rumored", "unconfirmed"]
        if any(h in text for h in hedging):
            failed_checks.append("hedging")
            penalty += 0.20

        # Check 6: Controversy guard - criminal claims need proper sourcing
        criminal_terms = ["murdered", "killed", "assassinated", "raped", "trafficked", "kidnapped"]
        if any(term in text for term in criminal_terms):
            if modality not in ['observation', 'allegation']:
                failed_checks.append("controversy")
                penalty += 0.25

        # Apply penalty to base confidence
        adjusted_confidence = max(0.0, base_confidence - penalty)

        # Flag for verification if 2+ checks failed or adjusted confidence < 0.6
        verification_needed = len(failed_checks) >= 2 or adjusted_confidence < 0.6

        return adjusted_confidence, failed_checks, verification_needed

    def _passes_claim_premises(self, claim: Dict[str, Any]) -> tuple[bool, str]:
        """
        DEPRECATED: Use _score_claim_with_premises instead

        Kept for backward compatibility
        """
        adjusted_confidence, failed_checks, _ = self._score_claim_with_premises(claim)
        if adjusted_confidence >= 0.65:
            return True, ""
        else:
            return False, f"Confidence too low: {', '.join(failed_checks)}"

    def _is_proper_noun(self, text: str) -> bool:
        """
        Check if text is likely a proper noun using linguistic features.

        Proper nouns (accept):
        - Multiple words with capitals: "Hong Kong Fire Services Department"
        - Single word with capital and multiple syllables: "Beijing", "Microsoft"

        Common nouns/generic terms (reject):
        - Lowercase words
        - Single short words without capitals
        - Words that are all uppercase (likely abbreviations without context)
        """
        words = text.split()

        # At least one word should start with capital
        has_capital = any(w[0].isupper() for w in words if w)
        if not has_capital:
            return False

        # Multi-word names are generally good (Hong Kong, Fire Services Department)
        if len(words) >= 2:
            return True

        # Single word: check it's substantial (not "Ng", "Wu", "Li")
        if len(words) == 1:
            # Need at least 3 characters for single-word entities
            if len(text) < 3:
                return False
            # All caps without context is suspicious (BBC, CNN as participants)
            if text.isupper() and len(text) <= 4:
                return False
            return True

        return False

    def _extract_entities_from_claims(self, claims: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Extract and categorize entities from claims"""
        entities = {
            "people": set(),
            "organizations": set(),
            "locations": set(),
            "time_references": set()
        }

        for claim in claims:
            if not isinstance(claim, dict):
                continue

            # Extract from WHO field with linguistic validation
            for who in claim.get('who', []):
                if who.startswith('PERSON:'):
                    name = who[7:].strip()
                    if self._is_proper_noun(name):
                        entities['people'].add(name)
                elif who.startswith('ORG:'):
                    name = who[4:].strip()
                    if self._is_proper_noun(name):
                        entities['organizations'].add(name)

            # Extract from WHERE field with validation
            for where in claim.get('where', []):
                location = where.split(':', 1)[-1] if ':' in where else where
                location = location.strip()
                # Locations need to be specific named places
                if self._is_proper_noun(location):
                    entities['locations'].add(location)

            # Extract from WHEN field
            when_info = claim.get('when', {})
            if isinstance(when_info, dict):
                if when_info.get('date'):
                    entities['time_references'].add(when_info['date'])
                if when_info.get('event_time'):
                    entities['time_references'].add(when_info['event_time'])
                if when_info.get('temporal_context'):
                    entities['time_references'].add(when_info['temporal_context'])

        # Convert sets to lists
        return {k: list(v) for k, v in entities.items()}

    def _extract_primary_event_time(self, claims: List[Dict[str, Any]], page_meta: Dict[str, Any]) -> Optional[Dict]:
        """
        Extract the primary event time from claims with precision metadata

        Strategy:
        1. Find earliest event_time from claims (the root event)
        2. Detect time precision from temporal_context
        3. Return {time: ISO string, precision: 'hour'|'day'|'month'|'year'}

        This distinguishes between:
        - Event time: When the actual event occurred
        - Publish time: When the article was published

        Time precision affects story matching:
        - hour/day: Breaking news, tight matching (within hours/days)
        - month: Recent events, medium matching (within weeks)
        - year: Historical events, loose matching (within months)

        Example: Investigation piece published in 2024 about 2020 scandal
        -> {time: '2020-03-15T00:00:00', precision: 'month'}
        """
        from dateutil import parser

        # Parse pub_time first for filtering
        pub_time = page_meta.get('pub_time')
        pub_dt = None
        if pub_time:
            try:
                pub_dt = parser.parse(pub_time) if isinstance(pub_time, str) else pub_time
            except Exception:
                pass

        event_times = []

        for claim in claims:
            if not isinstance(claim, dict):
                continue

            when_info = claim.get('when', {})
            if isinstance(when_info, dict):
                event_time_str = when_info.get('event_time')
                temporal_context = when_info.get('temporal_context', '')

                if event_time_str:
                    try:
                        # Parse to datetime
                        event_dt = parser.parse(event_time_str)

                        # FILTER: If event_time is >6 months before pub_time,
                        # it's likely a background reference, not the current event
                        # (e.g., "Comey was indicted in 2023" in a 2025 article about current dismissal hearing)
                        if pub_dt:
                            days_diff = (pub_dt - event_dt).days
                            if days_diff > 180:  # 6 months
                                print(f"⏰ Filtering old event_time: {event_time_str} ({days_diff} days before pub, likely background)")
                                continue

                        # Detect precision from temporal context or date format
                        precision = self._detect_time_precision(temporal_context, event_time_str)

                        event_times.append({
                            'time': event_dt,
                            'precision': precision,
                            'context': temporal_context
                        })
                    except Exception:
                        pass

        # Find earliest RECENT event time (after filtering)
        if event_times:
            earliest = min(event_times, key=lambda x: x['time'])
            return {
                'time': earliest['time'].isoformat(),
                'precision': earliest['precision'],
                'context': earliest['context']
            }

        # Fallback: use publish_time from page_meta (current event)
        if pub_dt:
            return {
                'time': pub_dt.isoformat(),
                'precision': 'day',
                'context': 'publication date (current event)'
            }

        return None

    def _detect_time_precision(self, temporal_context: str, event_time_str: str) -> str:
        """
        Detect time precision from context or date string

        Returns: 'hour' | 'day' | 'month' | 'year'
        """
        context_lower = temporal_context.lower() if temporal_context else ''

        # High precision indicators (hour/day)
        if any(word in context_lower for word in ['today', 'yesterday', 'this morning', 'tonight', 'hour', 'minute', 'am', 'pm']):
            return 'hour'

        # Day precision indicators
        if any(word in context_lower for word in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'last week', 'this week']):
            return 'day'

        # Month precision indicators
        if any(word in context_lower for word in ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'last month', 'this month', 'early', 'late', 'mid-']):
            return 'month'

        # Year precision indicators
        if any(word in context_lower for word in ['year', 'decade', 'century', 'ago']):
            return 'year'

        # Check date string format
        if 'T' in event_time_str and ':' in event_time_str:
            return 'hour'  # Has time component
        elif event_time_str.count('-') == 2:
            return 'day'  # Full date (YYYY-MM-DD)
        elif event_time_str.count('-') == 1:
            return 'month'  # Year-month (YYYY-MM)
        else:
            return 'year'  # Year only

    async def generate_about_embedding(self, about_text: str) -> List[float]:
        """
        Generate embedding for story matching

        Args:
            about_text: Concise 1-2 sentence summary from cleaning stage

        Returns:
            List of 1536 floats (embedding vector)
        """
        if not about_text or len(about_text) < 10:
            print("⚠️  About text too short for embedding, returning empty")
            return []

        try:
            response = self._with_retry(
                self.openai_client.embeddings.create,
                model="text-embedding-3-small",
                input=about_text
            )
            embedding = response.data[0].embedding
            print(f"✅ Generated embedding for about text ({len(embedding)} dims)")
            return embedding
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            return []

    async def generate_dual_embeddings(
        self,
        about_text: str,
        claims: List[Dict],
        entities: Dict[str, List[str]],
        full_content: str
    ) -> Dict[str, List[float]]:
        """
        Generate dual embeddings for diversity-preserving story matching

        Event embedding: Captures WHO/WHAT/WHERE/WHEN (factual events)
        Frame embedding: Captures full perspective/angle/framing

        Args:
            about_text: 1-2 sentence summary
            claims: List of extracted claims with WHO/WHERE/WHEN
            entities: Dict of categorized entities
            full_content: Full article text for frame embedding

        Returns:
            {
                "event_embedding": [1536 floats],
                "frame_embedding": [1536 floats],
                "event_text": "text used for event embedding",
                "frame_text": "text used for frame embedding"
            }
        """
        # Build event-focused text (who/what/where/when)
        event_text = self._build_event_text(about_text, claims, entities)

        # Build frame text (full perspective, first 2000 chars)
        frame_text = full_content[:2000] if full_content else about_text

        # Log what's being embedded for debugging
        print(f"📝 Event text ({len(event_text)} chars): {event_text[:200]}...")
        print(f"📝 Frame text ({len(frame_text)} chars): {frame_text[:200]}...")

        # Generate both embeddings
        try:
            event_embedding = await self.generate_about_embedding(event_text)
            frame_embedding = await self.generate_about_embedding(frame_text)

            print(f"✅ Generated dual embeddings: event={len(event_embedding)} dims, frame={len(frame_embedding)} dims")

            return {
                "event_embedding": event_embedding,
                "frame_embedding": frame_embedding,
                "event_text": event_text,
                "frame_text": frame_text
            }
        except Exception as e:
            print(f"❌ Dual embedding generation failed: {e}")
            return {
                "event_embedding": [],
                "frame_embedding": [],
                "event_text": event_text,
                "frame_text": frame_text
            }

    def _build_event_text(
        self,
        about_text: str,
        claims: List[Dict],
        entities: Dict[str, List[str]]
    ) -> str:
        """
        Build event-focused text for embedding (REVISED 2025-10-16 v3)

        CORE INSIGHT: Use claims directly, not synthesized entity strings!

        Claims already capture WHO/WHAT/WHEN in natural language.
        Embeddings handle semantic similarity - they understand that:
        - "US Mint will release Steve Jobs coin" ≈ "Steve Jobs to be honored on coin"

        We don't need perfect string matching - that's what embeddings are FOR.

        Example for "Steve Jobs coin":
        - Input: Top 3 claim texts about the coin
        - Output: Combined claim texts
        - Embedding model handles the similarity matching
        - Result: 0.85+ similarity across all 4 articles (MERGE - correct!)
        """
        # Approach: Gist + Top 3 claims
        # - Gist provides focused summary (high similarity across articles)
        # - Top 3 claims add key ontological details (WHO/WHAT/WHEN)
        # - Tested with embeddings: min=0.844, avg=0.884 similarity (BEST!)

        if not about_text:
            about_text = "No content"

        # Start with gist
        event_parts = [about_text]

        # Add top 3 claims if available
        if claims:
            for claim in claims[:3]:
                if isinstance(claim, dict):
                    claim_text = claim.get('text', '').strip()
                    if claim_text:
                        event_parts.append(claim_text)

        combined = " ".join(event_parts)
        num_claims = len(event_parts) - 1
        print(f"📝 Event text: gist ({len(about_text)} chars) + {num_claims} claim(s) = {len(combined)} total chars")
        print(f"   {combined[:150]}...")
        return combined

    def _extract_key_noun(self, about_text: str) -> str:
        """
        Extract THE key noun from about_text (one word only)

        This captures WHAT type of thing the event is about
        without the variance of descriptors/adjectives

        Example:
        - Input: "The U.S. Mint is set to release a commemorative $1 coin..."
        - Output: "coin" (NOT "commemorative coin")
        """
        text_lower = about_text.lower()

        # Key nouns (event types) in priority order
        nouns = [
            'coin', 'bill', 'law', 'policy', 'program', 'investigation',
            'trial', 'election', 'agreement', 'deal', 'report', 'study',
            'project', 'initiative', 'plan', 'proposal', 'ruling', 'verdict',
            'appointment', 'resignation', 'merger', 'acquisition', 'launch',
            'attack', 'strike', 'protest', 'scandal', 'crisis'
        ]

        for noun in nouns:
            if noun in text_lower:
                return noun

        return ""

    def _extract_action_keywords(self, about_text: str) -> str:
        """
        DEPRECATED: Use _extract_key_noun instead for v2 event matching

        Extract core action keywords from about_text (not full text)

        Focus on action verbs + key nouns to capture WHAT happened
        without the variance of full sentences

        Example:
        - Input: "The U.S. Mint is set to release a commemorative $1 coin..."
        - Output: "commemorative coin release"
        """
        # Common action patterns (simplified for now)
        # This can be enhanced with NLP if needed
        keywords = []

        text_lower = about_text.lower()

        # Extract key action verbs
        actions = ['release', 'announce', 'unveil', 'launch', 'introduce',
                   'seize', 'arrest', 'appoint', 'elect', 'approve', 'reject']
        for action in actions:
            if action in text_lower:
                keywords.append(action)
                break

        # Extract key nouns (event types)
        nouns = ['coin', 'bill', 'law', 'policy', 'program', 'investigation',
                 'trial', 'election', 'agreement', 'deal', 'report', 'study']
        for noun in nouns:
            if noun in text_lower:
                keywords.append(noun)

        # Extract descriptors
        descriptors = ['commemorative', 'innovation', 'emergency', 'historic',
                       'controversial', 'unprecedented']
        for desc in descriptors:
            if desc in text_lower:
                keywords.insert(0, desc)  # Descriptors go first

        return " ".join(keywords[:4]) if keywords else ""  # Limit to 4 keywords

    def _empty_extraction(self, reason: str) -> Dict[str, Any]:
        """Return empty extraction result"""
        return {
            "claims": [],
            "excluded_claims": [],
            "entities": {
                "people": [],
                "organizations": [],
                "locations": [],
                "time_references": []
            },
            "entity_descriptions": {},  # NEW format: {"PERSON:Name": "description"}
            "entity_relationships": [],  # NEW: hierarchical entity relationships
            "gist": "No content extracted",
            "confidence": 0.0,
            "notes_unsupported": [reason],
            "token_usage": {}
        }

    # =========================================================================
    # NEW: Mention-based extraction for Knowledge Worker pipeline
    # =========================================================================

    async def extract_with_mentions(
        self,
        page_meta: Dict[str, Any],
        page_text: List[Dict[str, str]],
        url: str,
        lang: str = "en"
    ) -> 'ExtractionResult':
        """
        Extract claims with mentions (not entities) for the Knowledge Worker pipeline.

        This is the new extraction method that outputs:
        - Mentions: Raw entity references with context (ephemeral)
        - Claims: Reference mentions by local ID
        - MentionRelationships: Structural relationships between mentions

        The output is passed to IdentificationService for entity resolution.

        Args:
            page_meta: {title, byline, pub_time, site}
            page_text: [{selector, text}] - content blocks
            url: canonical URL
            lang: language code

        Returns:
            ExtractionResult with mentions, claims, and relationships
        """
        from models.mention import Mention, MentionRelationship, ExtractionResult

        try:
            if not page_text:
                return ExtractionResult(gist="No content provided", confidence=0.0)

            # Prepare content
            full_text = self._prepare_content(page_meta, page_text)

            # Extract with mention-based prompt
            result = await self._extract_mentions_llm(full_text, page_meta, url, lang)

            return result

        except Exception as e:
            logger.error(f"❌ Mention extraction failed: {e}", exc_info=True)
            return ExtractionResult(
                gist=f"Extraction failed: {str(e)}",
                confidence=0.0
            )

    async def _extract_mentions_llm(
        self,
        content: str,
        page_meta: Dict[str, Any],
        url: str,
        lang: str = "en"
    ) -> 'ExtractionResult':
        """
        LLM call for mention-based extraction.

        Key difference from old approach:
        - LLM outputs mentions with local IDs (m1, m2, etc.)
        - Claims reference mentions by ID, not by "TYPE:Name"
        - Context is preserved for each mention (helps identification)
        """
        from models.mention import Mention, MentionRelationship, ExtractionResult

        system_prompt = f"""You are a fact extractor for structured journalism. Extract atomic claims with entity mentions.

Source language: {lang}
**CRITICAL: Extract ALL claims in English, regardless of source language.**

Your task:
1. Identify all entity MENTIONS in the text (people, organizations, locations)
2. Extract claims that reference these mentions
3. Identify structural relationships between mentions (PART_OF, LOCATED_IN, etc.)

For each MENTION:
- Assign a unique ID (m1, m2, m3...)
- Capture the surface form (exact text)
- Identify the type hint (PERSON, ORGANIZATION, LOCATION)
- Extract surrounding context (the sentence/phrase where it appears)

For each CLAIM:
- Reference mentions by their IDs (not by name strings)
- Include temporal information when available
- Classify modality (observation, reported_speech, allegation, opinion)

For RELATIONSHIPS between mentions:
- PART_OF: Physical containment (block in building, floor in building)
- LOCATED_IN: Geographic containment (building in city, city in country)
- WORKS_FOR: Employment (person works for organization)
- MEMBER_OF: Membership (person in group/party)
- AFFILIATED_WITH: Other affiliation

Extract up to 15 claims. Prioritize important facts."""

        user_prompt = f"""Extract mentions and claims from this content:

METADATA:
Title: {page_meta.get('title', 'Unknown')}
Site: {page_meta.get('site', 'Unknown')}
Published: {page_meta.get('pub_time', 'Unknown')}

CONTENT:
{content}

Return JSON:
{{
    "mentions": [
        {{
            "id": "m1",
            "surface_form": "Building A",
            "type_hint": "LOCATION",
            "context": "the fire spread to Building A, also known as the East Tower",
            "description": "Residential tower in the Greenview housing complex",
            "aliases": ["East Tower"]
        }},
        {{
            "id": "m2",
            "surface_form": "Greenview Estate",
            "type_hint": "LOCATION",
            "context": "Greenview Estate, a public housing complex in the northern district",
            "description": "Public housing estate with multiple residential towers",
            "aliases": []
        }},
        {{
            "id": "m3",
            "surface_form": "John Smith",
            "type_hint": "PERSON",
            "context": "rescue worker John Smith, a 15-year veteran of the department",
            "description": "Senior rescue worker with 15 years experience",
            "aliases": []
        }}
    ],
    "claims": [
        {{
            "text": "The fire spread to Building A of Greenview Estate",
            "who": [],
            "where": ["m1", "m2"],
            "when": {{
                "date": "2025-01-15",
                "time": "14:30:00",
                "precision": "hour",
                "timezone": "+00:00"
            }},
            "modality": "observation",
            "evidence_references": [],
            "confidence": 0.9
        }}
    ],
    "mention_relationships": [
        {{"subject": "m1", "predicate": "PART_OF", "object": "m2"}}
    ],
    "gist": "One sentence summary IN ENGLISH",
    "overall_confidence": 0.8,
    "extraction_quality": 0.9
}}

EXTRACTION_QUALITY (0.0-1.0):
Rate the quality of extracted mentions:
- 1.0: All mentions are proper named entities (specific people, organizations, places)
- 0.7: Most mentions are proper names, a few generic terms
- 0.5: Mix of proper names and generic descriptions
- 0.3: Mostly generic terms ("authorities", "residents", "the building")
- 0.0: No proper named entities found, only generic descriptions

Only extract PROPER NAMED ENTITIES, not generic terms like:
- "authorities", "officials", "police" (use specific dept name if available)
- "residents", "victims", "survivors", "families"
- "the building", "apartment building", "first floor"
- Days of week, generic times ("Friday", "Wednesday afternoon")
- Numbers as entities ("128 people", "thousands")

MENTION FIELDS:
- id: Unique identifier (m1, m2, m3...)
- surface_form: Primary name as it appears in text
- type_hint: PERSON, ORGANIZATION, or LOCATION
- context: Verbatim sentence/phrase from article containing this mention
- description: What/who this entity is (role, type, key characteristic)
- aliases: Other names for the SAME entity found in the text
  Example: "Building A, also known as East Tower" → surface_form="Building A", aliases=["East Tower"]

IMPORTANT:
- When text says "X, also known as Y" or "X (called Y)" → put Y in aliases, not as separate mention
- Each unique real-world entity should have ONE mention with all its aliases
- Claims reference mentions by ID in who/where arrays
- Include ALL locations at every granularity (building, district, city, country)
- Extract PART_OF relationships (unit in building, building in complex)
- Extract LOCATED_IN relationships (complex in district, district in city)"""

        try:
            response = await asyncio.to_thread(
                self._with_retry,
                self.openai_client.chat.completions.create,
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=4000
            )

            raw_response = response.choices[0].message.content
            logger.info(f"📥 Mention extraction response ({len(raw_response)} chars)")

            result = json.loads(raw_response)

            # Convert to ExtractionResult
            mentions = []
            for m in result.get('mentions', []):
                mentions.append(Mention(
                    id=m.get('id', ''),
                    surface_form=m.get('surface_form', ''),
                    type_hint=m.get('type_hint', 'UNKNOWN'),
                    context=m.get('context', ''),
                    description=m.get('description', ''),
                    aliases=m.get('aliases', [])
                ))

            relationships = []
            for r in result.get('mention_relationships', []):
                relationships.append(MentionRelationship(
                    subject_id=r.get('subject', ''),
                    predicate=r.get('predicate', ''),
                    object_id=r.get('object', '')
                ))

            # Claims stay as dicts for now (will be converted to Claim models later)
            claims = result.get('claims', [])

            # Link claims to mentions
            for i, claim in enumerate(claims):
                for m in mentions:
                    if m.id in claim.get('who', []) or m.id in claim.get('where', []):
                        m.claim_indices.append(i)

            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "model": "gpt-4o"
            }

            logger.info(f"✅ Extracted {len(mentions)} mentions, {len(claims)} claims, {len(relationships)} relationships")

            extraction_quality = result.get('extraction_quality', 0.5)
            logger.info(f"📊 Extraction quality: {extraction_quality}")

            return ExtractionResult(
                mentions=mentions,
                claims=claims,
                mention_relationships=relationships,
                gist=result.get('gist', ''),
                confidence=result.get('overall_confidence', 0.5),
                extraction_quality=extraction_quality,
                token_usage=token_usage
            )

        except Exception as e:
            logger.error(f"❌ Mention LLM extraction failed: {e}", exc_info=True)
            return ExtractionResult(
                gist=f"Extraction failed: {str(e)}",
                confidence=0.0
            )


# Global instance
semantic_analyzer = EnhancedSemanticAnalyzer()
