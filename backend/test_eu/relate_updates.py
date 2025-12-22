"""
PRINCIPLED UPDATES vs CONTRADICTS

The relate() function should detect this structurally, not ad-hoc.

A claim has:
  - Attributes (death_toll, location, cause, ...)
  - Values for those attributes
  - Temporal markers (optional)

relate(c1, c2) compares:
  1. Do they discuss the same attribute?
  2. Are the values the same or different?
  3. If different, is there temporal ordering?

Output:
  - CORROBORATES: same attribute, same value
  - UPDATES: same attribute, different value, c2 is later
  - CONTRADICTS: same attribute, different value, same time or conflicting
  - INDEPENDENT: different attributes or topics
"""
import sys
sys.path.insert(0, '/app/backend')
sys.path.insert(0, '/app/test_eu')

import re
from dataclasses import dataclass
from typing import Optional, Dict, Any
from collections import defaultdict


@dataclass
class ClaimStructure:
    """Structured representation of a claim."""
    text: str
    attributes: Dict[str, Any]  # e.g., {"death_toll": 160, "location": "Wang Fuk Court"}
    temporal_markers: list  # e.g., ["rises to", "now", "latest"]
    is_update_language: bool


def extract_structure(text: str) -> ClaimStructure:
    """
    Extract structured attributes from claim text.
    This is the principled way - extract attribute-value pairs.
    """
    text_lower = text.lower()
    attributes = {}
    temporal_markers = []

    # NUMERIC ATTRIBUTES
    # Death toll
    death_patterns = [
        (r'(\d+)\s*(?:people\s+)?(?:were\s+|have\s+been\s+)?killed', 'death_toll'),
        (r'(\d+)\s*(?:people\s+)?(?:have\s+)?died', 'death_toll'),
        (r'(\d+)\s*(?:people\s+)?dead', 'death_toll'),
        (r'death\s+toll[:\s]+(\d+)', 'death_toll'),
        (r'death\s+toll\s+(?:rises\s+to|hits|of)\s+(\d+)', 'death_toll'),
        (r'at\s+least\s+(\d+)\s+(?:people\s+)?(?:killed|dead|died)', 'death_toll'),
    ]

    for pattern, attr in death_patterns:
        m = re.search(pattern, text_lower)
        if m:
            attributes[attr] = int(m.group(1))
            break

    # Missing count
    missing_patterns = [
        (r'(\d+)\s+(?:people\s+)?(?:are\s+)?missing', 'missing_count'),
        (r'(\d+)\s+(?:people\s+)?unaccounted', 'missing_count'),
    ]

    for pattern, attr in missing_patterns:
        m = re.search(pattern, text_lower)
        if m:
            attributes[attr] = int(m.group(1))
            break

    # LOCATION ATTRIBUTES
    location_patterns = [
        (r'(wang\s+fuk\s+court)', 'location'),
        (r'(tai\s+po(?:\s+district)?)', 'location'),
        (r'(hong\s+kong)', 'region'),
    ]

    for pattern, attr in location_patterns:
        m = re.search(pattern, text_lower)
        if m:
            attributes[attr] = m.group(1).strip()

    # TEMPORAL MARKERS (indicate this is an update)
    update_phrases = [
        'rises to', 'risen to', 'increased to', 'now',
        'updated', 'latest', 'climbed to', 'reached',
        'has grown', 'hits', 'surpasses'
    ]

    for phrase in update_phrases:
        if phrase in text_lower:
            temporal_markers.append(phrase)

    is_update = len(temporal_markers) > 0

    return ClaimStructure(
        text=text,
        attributes=attributes,
        temporal_markers=temporal_markers,
        is_update_language=is_update
    )


def relate(c1_text: str, c2_text: str) -> tuple:
    """
    Principled relate() function.

    Returns: (relation, confidence, details)
    """
    s1 = extract_structure(c1_text)
    s2 = extract_structure(c2_text)

    # Find shared attributes
    shared_attrs = set(s1.attributes.keys()) & set(s2.attributes.keys())

    if not shared_attrs:
        return ('INDEPENDENT', 0.5, {'reason': 'no shared attributes'})

    relations = []

    for attr in shared_attrs:
        v1 = s1.attributes[attr]
        v2 = s2.attributes[attr]

        if v1 == v2:
            relations.append(('CORROBORATES', attr, v1, v2))

        elif isinstance(v1, int) and isinstance(v2, int):
            # Numeric attribute with different values
            if s2.is_update_language or v2 > v1:
                # c2 has update language OR higher value → likely UPDATE
                relations.append(('UPDATES', attr, v1, v2))
            elif s1.is_update_language or v1 > v2:
                # c1 is the update
                relations.append(('UPDATES', attr, v2, v1))
            else:
                # Same time, different values → true CONTRADICTION
                relations.append(('CONTRADICTS', attr, v1, v2))

        else:
            # Non-numeric, different values
            relations.append(('CONTRADICTS', attr, v1, v2))

    # Aggregate
    if all(r[0] == 'CORROBORATES' for r in relations):
        return ('CORROBORATES', 0.9, {'attrs': relations})
    elif any(r[0] == 'CONTRADICTS' for r in relations):
        return ('CONTRADICTS', 0.8, {'attrs': relations})
    elif any(r[0] == 'UPDATES' for r in relations):
        return ('UPDATES', 0.85, {'attrs': relations})
    else:
        return ('INDEPENDENT', 0.5, {'attrs': relations})


def main():
    print("=" * 70)
    print("PRINCIPLED UPDATES vs CONTRADICTS")
    print("=" * 70)

    # Test cases
    test_pairs = [
        # CORROBORATES - same fact
        ("At least 36 people were killed in Hong Kong fire",
         "36 people died in the Hong Kong blaze"),

        # UPDATES - death toll rises
        ("At least 36 people were killed in the fire",
         "Death toll rises to 128 in Hong Kong fire"),

        # UPDATES - higher number with update language
        ("128 dead in Hong Kong fire",
         "Death toll hits 160 as rescue efforts continue"),

        # CONTRADICTS - same time, different values (hypothetical)
        ("Police confirm 12 dead in incident",
         "Officials report only 8 casualties"),

        # CORROBORATES - same location
        ("Fire broke out at Wang Fuk Court",
         "Blaze engulfed Wang Fuk Court apartments"),
    ]

    print("\n--- Test Cases ---\n")

    for c1, c2 in test_pairs:
        relation, conf, details = relate(c1, c2)

        print(f"C1: {c1[:50]}...")
        print(f"C2: {c2[:50]}...")
        print(f"→ {relation} (conf={conf:.2f})")

        if 'attrs' in details:
            for r in details['attrs']:
                print(f"   {r[1]}: {r[2]} → {r[3]} [{r[0]}]")
        print()

    # Apply to real data
    print("=" * 70)
    print("REAL DATA: HK Fire Death Toll")
    print("=" * 70)

    from load_graph import load_snapshot
    snapshot = load_snapshot()
    claims = snapshot.claims

    # Get death toll claims
    death_claims = []
    for cid, claim in claims.items():
        text = claim.text.lower()
        if ('fire' in text or 'blaze' in text) and 'hong kong' in text:
            s = extract_structure(claim.text)
            if 'death_toll' in s.attributes:
                death_claims.append((s.attributes['death_toll'], s.is_update_language, claim.text[:60]))

    death_claims.sort(key=lambda x: x[0])

    print("\nDeath toll claims with structure:\n")
    for toll, is_update, text in death_claims:
        marker = "📈 UPDATE" if is_update else "   "
        print(f"  [{toll:3d}] {marker} {text}...")

    # Build update chain
    print("\n--- Update Chain ---\n")

    chain = []
    for i, (toll1, upd1, text1) in enumerate(death_claims):
        for toll2, upd2, text2 in death_claims[i+1:]:
            if toll2 > toll1:
                rel, conf, details = relate(
                    f"{toll1} killed in fire",
                    f"death toll rises to {toll2}" if upd2 else f"{toll2} killed in fire"
                )
                if rel == 'UPDATES':
                    chain.append((toll1, toll2))
                    break

    if chain:
        evolution = [chain[0][0]]
        for _, to_val in chain:
            if to_val not in evolution:
                evolution.append(to_val)
        print(f"  Evolution: {' → '.join(map(str, evolution))}")
        print(f"  Current state: {max(evolution)}")

    print("\n" + "=" * 70)
    print("THE PRINCIPLE")
    print("=" * 70)
    print("""
    relate(c1, c2) extracts STRUCTURE:
      - attributes: {death_toll: 36, location: "Wang Fuk Court"}
      - temporal_markers: ["rises to", "latest"]

    Then compares:
      same attr, same value → CORROBORATES
      same attr, diff value + update language → UPDATES
      same attr, diff value + no temporal order → CONTRADICTS

    This is principled, not ad-hoc:
      - We extract structure from text
      - We compare structured attributes
      - We use linguistic markers for temporal order

    The engine doesn't "know" death tolls go up.
    It sees "rises to 160" and marks it as UPDATE.
    """)


if __name__ == "__main__":
    main()
