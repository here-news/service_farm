"""
Relation Extraction Experiment v2
=================================

PROBLEM:
Current entity_relationships are structural links without provenance.
They can't be distinguished from hallucinated connections.

SOLUTION:
Treat relations as CLAIMS with:
1. Evidence (which sentence/quote asserts this?)
2. Confidence (how explicitly stated?)
3. Provenance (derived from claim X)

This allows relations to flow through the same constraint ledger as facts.

STAGE CONTRACTS (What Each Stage May/Must Not Do)
=================================================

KnowledgeWorker (L0 Extraction)
-------------------------------
SHOULD DO:
- Emit: Claim, Entity, and RelationClaim when explicitly stated
- Emit roles (who/where/org) and timestamps as constraints
- Canonicalize entities (Wikidata) as identity assistance
- Tag semantic outputs with confidence and input evidence

MUST NOT:
- Merge events or decide "same incident"
- Use Wikidata to force event membership
- Use semantic similarity to assert identity

Weaver L2 Surfaces (Identity)
-----------------------------
SHOULD DO:
- Only "same referent / same question_key" clustering
- Keep conflicts inside the same surface

MUST NOT:
- Use aboutness or entity relations to form surfaces
- Use semantic similarity to merge

Incident View L3 (Tight Membrane)
---------------------------------
SHOULD DO:
- Bind surfaces using time + higher-order motifs + bridge resistance
- Reject percolation bridges (context incompatibility)

MUST NOT:
- Use LLM-only similarity to create core merges
- Use Wikidata relations to merge incidents

Case View L4 (Looser Membrane)
------------------------------
SHOULD DO:
- Use relation backbone + local hubness + core/periphery
- This is where DoKwon↔Terraform relations matter

MUST NOT:
- Collapse unrelated cases due to one generic relation

NEW PROMPT DESIGN
=================

The key change: Relations are claims with evidence.

Instead of:
```json
"entity_relationships": [
    {"subject": "PERSON:Do Kwon", "predicate": "FOUNDED", "object": "ORG:Terraform Labs"}
]
```

We want:
```json
"relation_claims": [
    {
        "id": "rel_abc123",
        "subject": "PERSON:Do Kwon",
        "predicate": "FOUNDED",
        "object": "ORG:Terraform Labs",
        "evidence_text": "Do Kwon founded Terraform Labs in 2018",
        "evidence_claim_id": "clm_xyz789",
        "source_sentence": "The South Korean entrepreneur founded Terraform Labs...",
        "confidence": 0.92,
        "modality": "observation",
        "temporal_scope": "2018"
    }
]
```

ANTI-TRAP RULE
==============

Core edges require >= 2 independent signals, and at least one must be non-LLM.
LLM-only edges are periphery-only until corroborated.

Relation claims from LLM extraction:
- ARE semantic outputs (tagged with confidence)
- CAN attach periphery to cores
- CANNOT form cores by themselves
- MUST be corroborated by structural evidence for core merges

IMPLEMENTATION
==============
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import json


class RelationType(Enum):
    """Valid relation types with semantic meaning."""
    # Organizational
    FOUNDED = "founded"         # Person founded Org
    LEADS = "leads"             # Person leads Org
    WORKS_FOR = "works_for"     # Person works for Org
    MEMBER_OF = "member_of"     # Person/Org is member of Org

    # Geographic
    LOCATED_IN = "located_in"   # Entity is in Location
    PART_OF = "part_of"         # Location is part of Location

    # Ownership/Control
    OWNS = "owns"               # Entity owns Entity
    CONTROLS = "controls"       # Entity controls Entity
    SUBSIDIARY_OF = "subsidiary_of"  # Org is subsidiary of Org

    # Temporal/Causal
    PRECEDED_BY = "preceded_by"     # Event preceded by Event
    CAUSED_BY = "caused_by"         # Event caused by Entity/Event
    RESULTED_IN = "resulted_in"     # Entity/Event resulted in Entity/Event

    # Adversarial
    ACCUSED_OF = "accused_of"       # Person accused of Act
    SUED_BY = "sued_by"             # Entity sued by Entity
    INVESTIGATED_BY = "investigated_by"  # Entity investigated by Entity


@dataclass
class RelationClaim:
    """
    A relation as a first-class claim with provenance.

    This is NOT a graph edge - it's an observation about a relationship
    that must be supported by evidence.
    """
    id: str
    subject_id: str          # Entity ID (not name)
    subject_name: str        # Human-readable name
    predicate: RelationType
    object_id: str           # Entity ID (not name)
    object_name: str         # Human-readable name

    # Provenance (REQUIRED)
    evidence_text: str       # The sentence that asserts this relation
    source_claim_id: Optional[str] = None  # Claim this was derived from
    page_id: Optional[str] = None

    # Confidence and modality
    confidence: float = 0.5
    modality: str = "observation"  # observation, reported_speech, allegation

    # Temporal scope (when was this relation active?)
    temporal_scope: Optional[str] = None  # "2018", "2018-2022", "present"

    # Constraint type for ledger
    constraint_type: str = "semantic"  # semantic, structural, typed

    def to_constraint_entry(self) -> Dict:
        """Convert to constraint ledger entry format."""
        return {
            "type": "relation_claim",
            "constraint_type": self.constraint_type,
            "subject_id": self.subject_id,
            "predicate": self.predicate.value,
            "object_id": self.object_id,
            "evidence": self.evidence_text,
            "confidence": self.confidence,
            "modality": self.modality,
            "temporal_scope": self.temporal_scope,
            "provenance": {
                "source_claim_id": self.source_claim_id,
                "page_id": self.page_id
            }
        }


# =============================================================================
# ENHANCED EXTRACTION PROMPT FRAGMENT
# =============================================================================

RELATION_EXTRACTION_PROMPT = """
RELATION CLAIMS: Extract relationships AS CLAIMS with evidence.

For each relationship:
1. EVIDENCE_TEXT: The exact sentence that asserts this relationship (REQUIRED)
2. PREDICATE: The relationship type (from list below)
3. CONFIDENCE: How explicitly stated (0.0-1.0)
4. MODALITY: How it's expressed (observation, reported_speech, allegation)
5. TEMPORAL_SCOPE: When the relationship was/is active (if known)

PREDICATE TYPES:
- FOUNDED: Person created/started Org → "Do Kwon founded Terraform Labs"
- LEADS: Person leads/runs Org → "CEO Elon Musk runs Tesla"
- WORKS_FOR: Person employed by Org → "Smith works for Apple"
- OWNS: Entity owns Entity → "Musk owns Twitter"
- CONTROLS: Entity controls Entity → "Parent company controls subsidiary"
- SUBSIDIARY_OF: Org is subsidiary of Org
- ACCUSED_OF: Person accused of Act → "Smith accused of fraud"
- SUED_BY: Entity sued by Entity
- INVESTIGATED_BY: Entity investigated by Entity

EXTRACTION RULES:
1. ONLY extract if there's explicit text evidence (no inference!)
2. Include the exact sentence in evidence_text
3. If relationship is implied but not stated, set confidence < 0.5
4. For allegations, use modality="allegation"

EXAMPLE:
Article: "Do Kwon, who founded Terraform Labs in 2018, faces fraud charges."

```json
"relation_claims": [
    {
        "subject": "PERSON:Do Kwon",
        "predicate": "FOUNDED",
        "object": "ORG:Terraform Labs",
        "evidence_text": "Do Kwon, who founded Terraform Labs in 2018",
        "confidence": 0.95,
        "modality": "observation",
        "temporal_scope": "2018"
    },
    {
        "subject": "PERSON:Do Kwon",
        "predicate": "ACCUSED_OF",
        "object": "ACT:fraud",
        "evidence_text": "faces fraud charges",
        "confidence": 0.9,
        "modality": "allegation",
        "temporal_scope": "present"
    }
]
```
"""


# =============================================================================
# VALIDATION: Check if relation_claims flow correctly through constraint ledger
# =============================================================================

def validate_relation_claim(rel: RelationClaim) -> Tuple[bool, List[str]]:
    """
    Validate that a relation claim has required provenance.

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []

    if not rel.evidence_text:
        issues.append("Missing evidence_text (REQUIRED)")

    if rel.confidence > 0.9 and not rel.source_claim_id:
        issues.append("High confidence but no source_claim_id")

    if rel.modality == "allegation" and rel.confidence > 0.7:
        issues.append("Allegations should have confidence <= 0.7")

    if rel.predicate in (RelationType.FOUNDED, RelationType.LEADS):
        if not rel.temporal_scope:
            issues.append(f"{rel.predicate.value} should have temporal_scope")

    return len(issues) == 0, issues


def can_form_core(constraints: List[Dict]) -> Tuple[bool, str]:
    """
    Check if constraints can form a core edge.

    ANTI-TRAP RULE:
    Core edges require >= 2 independent signals, and at least one must be non-LLM.

    Args:
        constraints: List of constraint entries for a potential edge

    Returns:
        (can_form_core, reason)
    """
    if len(constraints) < 2:
        return False, f"Only {len(constraints)} constraints (need >= 2)"

    semantic_only = all(c.get("constraint_type") == "semantic" for c in constraints)
    if semantic_only:
        return False, "All constraints are semantic (need >= 1 non-semantic)"

    non_semantic = [c for c in constraints if c.get("constraint_type") != "semantic"]
    return True, f"Valid core: {len(non_semantic)} non-semantic + {len(constraints) - len(non_semantic)} semantic"


# =============================================================================
# TEST CASES
# =============================================================================

def test_relation_extraction():
    """Test that relation claims have proper provenance."""

    # Good: Explicit evidence
    good_rel = RelationClaim(
        id="rel_001",
        subject_id="ent_doKwon",
        subject_name="Do Kwon",
        predicate=RelationType.FOUNDED,
        object_id="ent_terraform",
        object_name="Terraform Labs",
        evidence_text="Do Kwon, who founded Terraform Labs in 2018",
        source_claim_id="clm_xyz",
        page_id="page_abc",
        confidence=0.95,
        modality="observation",
        temporal_scope="2018"
    )
    is_valid, issues = validate_relation_claim(good_rel)
    assert is_valid, f"Good relation should be valid: {issues}"
    print(f"✓ Good relation claim: {good_rel.subject_name} --[{good_rel.predicate.value}]--> {good_rel.object_name}")

    # Bad: No evidence
    bad_rel = RelationClaim(
        id="rel_002",
        subject_id="ent_doKwon",
        subject_name="Do Kwon",
        predicate=RelationType.OWNS,
        object_id="ent_luna",
        object_name="Luna",
        evidence_text="",  # Missing!
        confidence=0.8
    )
    is_valid, issues = validate_relation_claim(bad_rel)
    assert not is_valid, "Bad relation should be invalid"
    print(f"✓ Bad relation caught: {issues}")

    # Anti-trap rule test
    semantic_only = [
        {"constraint_type": "semantic", "predicate": "FOUNDED"},
        {"constraint_type": "semantic", "predicate": "LEADS"}
    ]
    can_core, reason = can_form_core(semantic_only)
    assert not can_core, "Semantic-only should not form core"
    print(f"✓ Semantic-only blocked: {reason}")

    mixed = [
        {"constraint_type": "semantic", "predicate": "FOUNDED"},
        {"constraint_type": "structural", "predicate": "co_anchor_motif"}
    ]
    can_core, reason = can_form_core(mixed)
    assert can_core, "Mixed should form core"
    print(f"✓ Mixed constraints allowed: {reason}")

    print("\n✓ All relation extraction tests passed")


if __name__ == "__main__":
    test_relation_extraction()
