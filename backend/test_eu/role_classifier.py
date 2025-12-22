"""
Experiment 2: Role Classification

Can we infer EU roles from existing claim data?

Roles to detect:
- OCCURRENCE: "X happened" (past tense, factual statement)
- ASSERTION: "X claims/says Y" (attributed statement)
- DENIAL: "X denies Y" (negation by named party)
- CONFIRMATION: "X confirms Y" (official validation)
- UPDATE: Claims with UPDATES relationship
- CONTRADICTION: Claims with CONTRADICTS relationship

Run inside container:
    docker exec herenews-app python /app/test_eu/role_classifier.py
"""

import json
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple
from pathlib import Path

from load_graph import load_snapshot, GraphSnapshot, ClaimData


class EURole(Enum):
    OCCURRENCE = "occurrence"       # Something happened
    ASSERTION = "assertion"         # Someone claims something
    DENIAL = "denial"               # Someone denies something
    CONFIRMATION = "confirmation"   # Official confirmation
    UPDATE = "update"               # Updates previous info
    CONTRADICTION = "contradiction" # Contradicts existing claim
    INSTITUTIONAL = "institutional" # Official/institutional action
    UNKNOWN = "unknown"


# Pattern matching for role detection
DENIAL_PATTERNS = [
    r'\bdenies?\b',
    r'\bdenied\b',
    r'\brefutes?\b',
    r'\brefuted\b',
    r'\brejects?\b',
    r'\brejected\b',
    r'\bdismisses?\b',
    r'\bdismissed\b',
    r'\bnot guilty\b',
    r'\bpleaded not guilty\b',
]

ASSERTION_PATTERNS = [
    r'\bclaims?\b',
    r'\bclaimed\b',
    r'\balleges?\b',
    r'\balleged\b',
    r'\bsays?\b',
    r'\bsaid\b',
    r'\bstates?\b',
    r'\bstated\b',
    r'\baccording to\b',
    r'\breports?\b',
    r'\breported\b',
    r'\baccuses?\b',
    r'\baccused\b',
]

CONFIRMATION_PATTERNS = [
    r'\bconfirms?\b',
    r'\bconfirmed\b',
    r'\bverifies?\b',
    r'\bverified\b',
    r'\backnowledges?\b',
    r'\backnowledged\b',
    r'\badmits?\b',
    r'\badmitted\b',
]

INSTITUTIONAL_PATTERNS = [
    r'\bsentenced\b',
    r'\bruled\b',
    r'\bordered\b',
    r'\bfiled\b',
    r'\bissued\b',
    r'\bimposed\b',
    r'\bsanctioned\b',
    r'\bcharged\b',
    r'\bindicted\b',
    r'\bconvicted\b',
    r'\bacquitted\b',
    r'\bappealed\b',
]


def classify_role(claim: ClaimData, snapshot: GraphSnapshot) -> Tuple[EURole, str]:
    """
    Classify claim into EU role based on text patterns and graph relationships.
    Returns (role, reasoning)
    """
    text = claim.text.lower()

    # Check graph relationships first (most reliable)
    if claim.updates_ids:
        return EURole.UPDATE, f"Has UPDATES relationship to {len(claim.updates_ids)} claims"

    if claim.contradicts_ids:
        return EURole.CONTRADICTION, f"Has CONTRADICTS relationship to {len(claim.contradicts_ids)} claims"

    # Check text patterns
    for pattern in DENIAL_PATTERNS:
        if re.search(pattern, text):
            return EURole.DENIAL, f"Matched denial pattern: {pattern}"

    for pattern in CONFIRMATION_PATTERNS:
        if re.search(pattern, text):
            return EURole.CONFIRMATION, f"Matched confirmation pattern: {pattern}"

    for pattern in INSTITUTIONAL_PATTERNS:
        if re.search(pattern, text):
            return EURole.INSTITUTIONAL, f"Matched institutional pattern: {pattern}"

    for pattern in ASSERTION_PATTERNS:
        if re.search(pattern, text):
            return EURole.ASSERTION, f"Matched assertion pattern: {pattern}"

    # Default to occurrence (factual statement)
    return EURole.OCCURRENCE, "No attribution patterns detected"


def classify_all_claims(snapshot: GraphSnapshot) -> Dict[str, Tuple[EURole, str]]:
    """Classify all claims into EU roles"""
    classifications = {}
    for cid, claim in snapshot.claims.items():
        role, reason = classify_role(claim, snapshot)
        classifications[cid] = (role, reason)
    return classifications


def analyze_role_distribution(
    classifications: Dict[str, Tuple[EURole, str]],
    snapshot: GraphSnapshot
) -> Dict:
    """Analyze distribution of roles"""
    results = {
        'total_claims': len(classifications),
        'role_counts': {},
        'role_percentages': {},
        'examples_by_role': {},
        'role_by_event': {},
    }

    # Count roles
    role_counter = Counter(role for role, _ in classifications.values())
    results['role_counts'] = {role.value: count for role, count in role_counter.items()}
    results['role_percentages'] = {
        role.value: count / len(classifications) * 100
        for role, count in role_counter.items()
    }

    # Examples by role
    for role in EURole:
        examples = []
        for cid, (r, reason) in classifications.items():
            if r == role and len(examples) < 5:
                claim = snapshot.claims.get(cid)
                if claim:
                    examples.append({
                        'id': cid,
                        'text': claim.text[:100],
                        'reason': reason
                    })
        results['examples_by_role'][role.value] = examples

    # Role distribution by event
    for eid, event in snapshot.events.items():
        event_roles = Counter()
        for cid in event.claim_ids:
            if cid in classifications:
                role, _ = classifications[cid]
                event_roles[role.value] += 1
        results['role_by_event'][event.canonical_name] = dict(event_roles)

    return results


def analyze_role_patterns(
    classifications: Dict[str, Tuple[EURole, str]],
    snapshot: GraphSnapshot
) -> Dict:
    """Analyze patterns in how roles relate to each other"""
    results = {
        'corroboration_patterns': [],
        'contradiction_patterns': [],
        'role_sequences': [],
    }

    # What roles corroborate what roles?
    corr_pairs = Counter()
    for cid, (role, _) in classifications.items():
        claim = snapshot.claims.get(cid)
        if claim:
            for target_id in claim.corroborates_ids:
                if target_id in classifications:
                    target_role, _ = classifications[target_id]
                    corr_pairs[(role.value, target_role.value)] += 1

    results['corroboration_patterns'] = [
        {'source': src, 'target': tgt, 'count': count}
        for (src, tgt), count in corr_pairs.most_common(10)
    ]

    # What roles contradict what roles?
    contra_pairs = Counter()
    for cid, (role, _) in classifications.items():
        claim = snapshot.claims.get(cid)
        if claim:
            for target_id in claim.contradicts_ids:
                if target_id in classifications:
                    target_role, _ = classifications[target_id]
                    contra_pairs[(role.value, target_role.value)] += 1

    results['contradiction_patterns'] = [
        {'source': src, 'target': tgt, 'count': count}
        for (src, tgt), count in contra_pairs.most_common(10)
    ]

    return results


def main():
    print("=" * 60)
    print("EU Experiment 2: Role Classification")
    print("=" * 60)

    # Load data
    snapshot = load_snapshot()
    print(f"\nLoaded: {len(snapshot.claims)} claims")

    # Classify all claims
    print("Classifying claims...")
    classifications = classify_all_claims(snapshot)

    # Analyze distribution
    print("Analyzing distribution...")
    distribution = analyze_role_distribution(classifications, snapshot)

    # Analyze patterns
    print("Analyzing patterns...")
    patterns = analyze_role_patterns(classifications, snapshot)

    # Print results
    print("\n" + "=" * 60)
    print("Role Distribution")
    print("=" * 60)
    for role, count in sorted(distribution['role_counts'].items(), key=lambda x: x[1], reverse=True):
        pct = distribution['role_percentages'][role]
        bar = "█" * int(pct / 2)
        print(f"  {role:15} {count:4} ({pct:5.1f}%) {bar}")

    print("\n" + "=" * 60)
    print("Examples by Role")
    print("=" * 60)
    for role, examples in distribution['examples_by_role'].items():
        if examples:
            print(f"\n{role.upper()}:")
            for ex in examples[:3]:
                print(f"  - {ex['text']}...")
                print(f"    ({ex['reason']})")

    print("\n" + "=" * 60)
    print("Role Distribution by Event")
    print("=" * 60)
    for event_name, roles in distribution['role_by_event'].items():
        total = sum(roles.values())
        print(f"\n{event_name} ({total} claims):")
        for role, count in sorted(roles.items(), key=lambda x: x[1], reverse=True):
            pct = count / total * 100
            print(f"  {role}: {count} ({pct:.0f}%)")

    print("\n" + "=" * 60)
    print("Corroboration Patterns (what roles support what)")
    print("=" * 60)
    for pattern in patterns['corroboration_patterns'][:5]:
        print(f"  {pattern['source']} -> corroborates -> {pattern['target']}: {pattern['count']}")

    print("\n" + "=" * 60)
    print("Contradiction Patterns (what roles conflict)")
    print("=" * 60)
    for pattern in patterns['contradiction_patterns'][:5]:
        print(f"  {pattern['source']} -> contradicts -> {pattern['target']}: {pattern['count']}")

    # Save results
    output = {
        'distribution': distribution,
        'patterns': patterns
    }
    output_path = Path("/app/test_eu/results/role_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
