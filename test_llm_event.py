#!/usr/bin/env python3
import asyncio
import json
from openai import AsyncOpenAI

async def test():
    client = AsyncOpenAI()

    claim_texts = [
        "Comey is seeking the dismissal of the DOJ's false statements case against him",
        "Prosecutors urged the federal judge to reject Comey's claim of vindictive prosecution",
        "Comey was indicted for making a false statement to Congress"
    ]

    entities = [
        {"canonical_name": "James Comey", "entity_type": "PERSON"},
        {"canonical_name": "DOJ", "entity_type": "ORGANIZATION"},
        {"canonical_name": "Congress", "entity_type": "ORGANIZATION"}
    ]

    entity_info = [f"{e['canonical_name']} ({e['entity_type']})" for e in entities]

    prompt = f"""Synthesize an event from these claims.

Claims:
{chr(10).join(f"- {claim}" for claim in claim_texts)}

Entities involved:
{chr(10).join(f"- {e}" for e in entity_info)}

Create a structured event description with:
1. Concise title (< 100 chars) describing what happened
2. Summary (2-3 sentences) explaining the event
3. Event type (legal_proceeding, investigation, incident, policy_announcement, political_event, etc.)
4. Location (if mentioned)
5. Timeframe (if mentioned, ISO format)

Return JSON:
{{
  "title": "DOJ prosecution of James Comey for false statements",
  "summary": "James Comey faces federal charges...",
  "event_type": "legal_proceeding",
  "location": "United States",
  "event_start": "2025-01-15T00:00:00Z"
}}

Be precise and factual. Extract actual event, not meta-commentary."""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an event synthesis system. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        print("Success:")
        print(response.choices[0].message.content)

        # Try to parse it
        result = json.loads(response.choices[0].message.content)
        print("\nParsed JSON:")
        print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
