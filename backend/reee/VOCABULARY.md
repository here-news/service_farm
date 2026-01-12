# Epistemic Vocabulary and Membrane Contract

This document defines the canonical vocabulary for the epistemic unit architecture,
grounded in production evidence from the Wang Fuk Court fire postmortem (2025-01).

## Layer Hierarchy

```
L0: Claim       Atomic observation with provenance
L2: Surface     Question-keyed proposition bundle
L3: Incident    Atomic "something happened" (time-bounded)
L4: Story       Composed narrative with membrane
L5: Lens        Entity profile (navigation, not narrative)
```

## L0: Claim

**Definition**: Atomic, immutable observation extracted from a source.

**Identity**: `claim_id` (content hash or UUID)

**Properties**:
- `text`: The extracted assertion
- `source`: Publisher/URL provenance
- `entities`: Named entities mentioned
- `anchor_entities`: High-salience entities (subjects, not bystanders)
- `question_key`: The proposition type (e.g., "fire_death_count", "fire_cause")
- `extracted_value`: Typed value if applicable
- `timestamp`: Extraction or publication time

**Invariant**: Claims are append-only. Corrections create new claims with SUPERSEDES edge.


## L2: Surface

**Definition**: Bundle of claims answering the same question (same `question_key`).

**Identity**: `(scope_id, question_key)` - NOT embedding similarity

**Formation**: Claims with identical `question_key` → same Surface

**Properties**:
- `claim_ids`: Member claims
- `entropy`: Posterior uncertainty over the typed variable (Jaynes)
- `sources`: Distinct sources providing observations
- `anchor_entities`: Union of claim anchors

**Invariant**: Conflicts stay INSIDE one surface (same question, different answers).
Embedding similarity is evidence for L3 linking, not L2 identity.


## L3: Incident

**Definition**: Atomic "something happened" - a time-bounded real-world occurrence.

**Identity**: `scope_signature = hash(anchor_motif, time_bin)`

**Formation**: Surfaces → Incident via:
1. Shared anchor motif (k≥2 entity set)
2. Temporal gate (≤7 day window)
3. Discriminative anchor (not a hub)

**Properties**:
- `surface_ids`: Member surfaces
- `anchor_entities`: The defining motif
- `companion_entities`: Other mentioned entities
- `time_window`: (start, end) of the incident

**Invariant**: Incidents are episodic, not ongoing. A trial spanning months
has many incidents (each hearing), not one mega-incident.


## L4: Story

**Definition**: Composed narrative object with explicit membrane contract.
**NOT** a clustering output—Story is assembled from a spine, a mode, and incidents
that pass the membrane contract.

**Composition** (explicit components, not discovered clusters):
```
Story = {
    focal: {
        primary:    EntityID,           # Main focal entity (non-hub)
        co_spines:  Set[EntityID],      # Usually empty; size 1-2 when truly dyadic
        kind:       "single" | "dyad" | "set",
        evidence:   Set[Constraint],    # Why these are focal (not just mentioned)
    },
    mode:            TemporalMode,      # Gap-clustered time window (>30 days = new mode)
    core_incidents:  Set[Incident],     # Core-A ∪ Core-B members
    periphery_ids:   Set[Incident_ID],  # Not persisted as members
    facet_surfaces:  Set[Surface],      # Question-keyed propositions attached
    inquiries:       Set[Inquiry],      # Open questions / gap tasks
}
```

**Identity**: `story_id = hash(sorted(focal.primary ∪ co_spines), mode_id)`
Titles never define identity.

**Focal Set Kinds**:
- **Single** (most common): Star-shaped story around one entity (Wang Fuk Court fire, Jimmy Lai trial)
- **Dyad**: Interaction-centric story where the relationship itself is focal (Do Kwon ↔ Terraform Labs, Trump ↔ Letitia James, Israel ↔ Hamas)
- **Set**: Rare; explicit group as focal (e.g., "TIME Person of the Year candidates" only if structural evidence supports the set)

**When to allow 2+ spines** (without reintroducing percolation):
Require direct, repeatable, non-semantic evidence that spines are jointly focal:
1. **Repeated co-anchor**: k=2 recurring pair in incidents within same mode, OR
2. **High pair discriminativeness**: PMI/lift over incident co-occurrence, OR
3. **Structural motif**: Hyperedge support showing they co-occur as a unit

**Never promote**: Shared hub (Hong Kong, United States) into co-spines.
If evidence is insufficient → keep two single-spine stories + cross-story link ("related", "overlaps").

**Components**:
- **Focal set**: Primary + optional co-spines (earned by structural evidence, not semantic vibes)
- **Mode**: Temporal window (gap-based clustering, >30 days = new mode)
- **Membrane**: The contract determining incident membership

### Membrane Contract

For `Story(focal=F, mode=M)`:

| Level | Criterion | Example |
|-------|-----------|---------|
| **Core-A (single)** | `focal.primary ∈ incident.anchors` | Wang Fuk Court is anchor |
| **Core-A (dyad)** | Both spines are anchors OR dyad motif present | Trump AND Letitia James both anchors |
| **Core-B** | Structural warrant exists | "Tai Po blaze" with geo+time witness |
| **Periphery** | Semantic-only or chain-only | Never persisted as member |
| **Reject** | No warrant, hub-only connection | Jimmy Lai trial incident |

**Core-A Invariant (single)**: Automatic membership when primary spine is anchor.

**Core-A Invariant (dyad/set)**: Requires explicit dyad evidence—either both/all spines are anchors,
or the dyad motif itself is present in the incident. Single-spine mention is insufficient.

**Core-B Invariant**: Requires **≥2 StructuralWitnesses**, at least one of which must be
geo or event-type (not time-only). Time witness is required when timestamp evidence is available.

Witness types:
- **Time witness**: Strong overlap with story mode (within N days). Required when available.
- **Geo witness**: Location containment (Tai Po District ⊃ Wang Fuk Court)
- **Event-type witness**: Same event kind surface in same mode

Hub entities CANNOT provide witnesses. A witness from "Hong Kong" (hub) does not count.

**Periphery Invariant**: NEVER becomes core. Not counted for completeness.
May be shown as "related" but not "part of story".

**Reject Invariant**: Must not attach even as periphery. These are either:
- Unrelated (Jimmy Lai trial shared only hub anchor "John Lee")
- Separate storylines (Wong Kwok-ngon sedition is related-but-distinct)

### What Hubs Cannot Do

Hub entities (high frequency + high context entropy) CANNOT:
- Be a story spine
- Provide Core-B warrants
- Create core edges via chain transitivity

Hubs CAN:
- Exist as L5 Lenses (users want "Hong Kong" page)
- Be mentioned in stories without defining them


## L5: Lens (Entity Profile)

**Definition**: All incidents mentioning an entity, for navigation.
Lenses are **navigation artifacts**, not narrative containers.

**Identity**: `entity_id`

**Properties**:
- `incident_ids`: All incidents where entity appears (unbounded)
- `story_ids`: Stories where entity is spine or participant
- `aliases`: Alternative names/spellings
- `entity_type`: person, location, organization, event

**Invariant**: Lenses have **NO membrane**. They are **allowed to be hubby and unbounded**.
A "Hong Kong" lens with 10,000 incidents is fine—it's navigation, not a story.

**Use case**: Entity pages, search results, cross-story navigation.
NOT for homepage story feeds (use Stories with membranes for that).


## Structural vs Semantic Constraints

**CRITICAL POLICY: Semantic constraints are proposals; they cannot be structural witnesses.**

Constraints in the ledger are typed by `kind`:

| Kind | Category | Can Witness Core-B? |
|------|----------|---------------------|
| `time` | Structural | Yes (but alone is insufficient) |
| `geo` | Structural | Yes |
| `event_type` | Structural | Yes |
| `motif` | Structural | Yes |
| `context` | Structural | Yes |
| `embedding` | Semantic | **No** |
| `llm_proposal` | Semantic | **No** |
| `title_similarity` | Semantic | **No** |

Semantic constraints CAN:
- Generate Core-B candidates (periphery until confirmed)
- Attach periphery members
- Propose warrants that trigger extraction tasks

Semantic constraints CANNOT:
- Provide structural witnesses for Core-B promotion
- Create or merge Story cores
- Override membership decisions

This is the anti-trap rule: no amount of semantic evidence alone can force core membership.


## Structural Witnesses (for Core-B)

Core-B membership requires explicit, auditable warrants:

```python
@dataclass
class StructuralWitness:
    witness_type: Literal["time", "geo", "event_type"]
    source: str  # Which claim/entity/constraint produced this
    confidence: float
    evidence: Dict[str, Any]
```

### Time Witness
```python
TimeWitness(
    witness_type="time",
    source="incident.time_window",
    confidence=0.9,
    evidence={
        "incident_time": "2025-11-26",
        "story_mode_center": "2025-11-26",
        "delta_days": 0,
        "threshold_days": 7,
    }
)
```

### Geo Witness
```python
GeoWitness(
    witness_type="geo",
    source="entity_enricher.containment",
    confidence=0.85,
    evidence={
        "incident_location": "Tai Po District",
        "spine_location": "Wang Fuk Court",
        "containment": "Tai Po District ⊃ Wang Fuk Court estate",
    }
)
```

### Event-Type Witness
```python
EventTypeWitness(
    witness_type="event_type",
    source="surface.question_key",
    confidence=0.8,
    evidence={
        "incident_event_type": "fire_alarm_level",
        "story_event_type": "building_fire",
        "compatibility": "fire_alarm → building_fire",
    }
)
```


## WFC Leak Postmortem

Production evidence from Wang Fuk Court fire case (2025-01):

### Before (37% membrane leak)
- 33 incidents in "WFC Deadly Fire" case
- 21 with WFC as anchor (Core-A)
- 12 without WFC anchor (leaked)

### Leak Classification

| Incident | Anchors | Verdict | Reason |
|----------|---------|---------|--------|
| in_93tfl25p | Jimmy Lai, Apple Daily, ... | **REJECT** | Unrelated. Only shared hub (John Lee) |
| in_rra2ia4i | Wong Kwok-ngon, West Kowloon Courts | **RELATED_STORY** | Separate storyline—fire-triggered sedition charges. Cross-story edge optional. |
| in_uw1jfl93 | John Lee, Tai Po | **Core-B candidate** | Time+Geo witness possible |
| in_97702qrd | Chris Tang, Tai Po | **Core-B candidate** | Time+Geo+EventType witness |
| in_t573bwg6 | Tai Po Baptist Church | **Core-B candidate** | Time+Geo witness |
| in_bbcxv4qo | HK Fire Services | **Core-B candidate** | Time+EventType witness |

Note: RELATED_STORY is distinct from REJECT. Related storylines can have optional
cross-story edges for navigation (e.g., "see also: Wong Kwok-ngon sedition case")
but are NOT core members of the WFC fire story.

### Root Causes

1. **Chain percolation**: Jimmy Lai incident connected only via hub (John Lee)
2. **Missing witnesses**: Extraction didn't yield geo/event-type cues
3. **No membrane enforcement**: All semantically-similar incidents became core

### Correct Behavior

```python
def classify_incident(incident, story) -> MembershipLevel:
    # Core-A: spine is anchor
    if story.spine in incident.anchor_entities:
        return MembershipLevel.CORE_A

    # Core-B: has structural witness
    witnesses = find_structural_witnesses(incident, story)
    if len(witnesses) > 0 and not is_hub_only_connection(incident, story):
        return MembershipLevel.CORE_B

    # Check for reject conditions
    if shares_only_hub_anchors(incident, story):
        return MembershipLevel.REJECT

    if is_separate_storyline(incident, story):
        return MembershipLevel.REJECT  # Or RELATED_STORY

    # Periphery: semantic-only, not persisted
    return MembershipLevel.PERIPHERY
```


## Meta-Claims for Membrane Health

```python
MetaClaimType = [
    # Membrane diagnostics
    "membrane_leak_high",           # Core has >X% incidents without spine anchor
    "missing_structural_witness",   # Core-B candidate lacks witnesses
    "hub_only_connection",          # Incident connected only via hub anchors
    "separate_storyline_attached",  # Related but distinct story leaked in

    # Core-B witness blocking (self-awareness about witness scarcity)
    "core_b_blocked_missing_witnesses",  # Incident was periphery due to < 2 witnesses
                                         # Args: story_id, incident_id, missing=[geo|event_type|time]

    # Extraction gaps (why witnesses are missing)
    "missing_geo_extraction",       # Location cues in text, not extracted
    "missing_event_type",           # Event type cues present, not typed
    "missing_time_evidence",        # No timestamp on incident
]

# Example: When an incident could have been Core-B but wasn't
# MetaClaim(
#     type="core_b_blocked_missing_witnesses",
#     story_id="wfc_fire_2025_11",
#     incident_id="in_97702qrd",
#     evidence={"witnesses_found": ["time"], "witnesses_missing": ["geo", "event_type"]}
# )
```


## Architecture: Separation of Concerns

```
┌─────────────────────────────────────────────────────────────────┐
│                    Entity Enricher (L0.5)                       │
│  Purpose: Calibration (improves inputs)                         │
│  - Alias resolution                                             │
│  - Entity types                                                 │
│  - Disambiguation (sense IDs)                                   │
│  - Geo containment chains                                       │
│                                                                 │
│  Interface to Story Builder: READ-ONLY                          │
│  - entity_aliases(id) → Set[str]                               │
│  - entity_type(id) → str                                       │
│  - geo_contains(loc1, loc2) → bool                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Story Builder (L4)                           │
│  Purpose: Membrane layer (forms bounded narratives)             │
│  - Computes Core-A from anchor overlap                         │
│  - Computes Core-B from structural witnesses                   │
│  - Rejects hub-only connections                                │
│  - Emits meta-claims when witnesses missing                    │
│                                                                 │
│  NEVER: Promotes based on semantic similarity alone            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Semantic Inspector (Service)                 │
│  Purpose: Proposal layer (adds candidates + explanations)       │
│  - Proposes canonical titles (presentation only)               │
│  - Proposes "same episode" hypotheses → SemanticConstraint     │
│  - Identifies missing anchors/facets → gap tasks               │
│                                                                 │
│  NEVER: Creates core merges                                    │
│  NEVER: Overrides spine/mode identity                          │
│  NEVER: Replaces missing structural signals                    │
└─────────────────────────────────────────────────────────────────┘
```


## LLM Role in Membranization (Bounded Semantic Tightening)

The LLM acts as a **semantic witness generator**, not a merger. It proposes warrants
that enter the constraint ledger; it never directly creates Core memberships.

### The Anti-Trap Rule

**Semantic-only evidence cannot create or merge Story cores.**

Semantic evidence CAN:
1. Attach periphery candidates
2. Propose Core-B warrants (which stay periphery until structural witness exists)
3. Propose tasks to obtain missing structural witnesses

### LLM Warrant Types

When called, LLM must output explicit warrant tokens, not vibes:

```python
class SemanticWarrant:
    warrant_type: Literal[
        "same_event_reference",    # "Tai Po blaze" refers to Wang Fuk Court fire
        "alias_or_part_of",        # Tai Po housing estate ↔ Wang Fuk Court
        "event_type_match",        # alarm level 5, blaze, fatal fire
        "time_compatibility",      # explicit or implied date match
        "dyad_interaction",        # A↔B is interaction-centric, not co-mention
    ]
    source_text: str               # The text that triggered this warrant
    confidence: float
    extraction_task: Optional[str] # If warrant needs structural confirmation
```

### Membership Decision Protocol

**(A) Incident → Story membership** (bounded diff, not whole graph)

Input: Story focal set + mode window + 1 incident summary
Output: `{Core-B_candidate, Periphery_candidate, Reject}` + explicit warrants

Decision flow:
```
1. LLM produces warrants for incident
2. Kernel checks: does warrant enable structural witness extraction?
   - YES → create extraction task, incident stays PERIPHERY until confirmed
   - NO → incident stays PERIPHERY (semantic-only)
3. Once extraction task completes with structural witness:
   - Kernel re-evaluates → may promote to Core-B
```

**(B) Multi-spine promotion** (dyad/set focal)

Input: candidate focal set {A,B} + co-occurrence incidents + negative examples (A without B)
Output: `PROMOTE_CO_SPINE` only if LLM can justify "dyad-ness" (interaction-centric)

This prevents "Hong Kong + John Lee" becoming co-spines for everything.

### When to Call LLM (Trigger Policy)

Only call LLM when a meta-claim indicates the system is blocked:

| Meta-Claim | LLM Task |
|------------|----------|
| `core_b_blocked_missing_witnesses` | Extract geo/event-type from text |
| `question_key_too_generic` | Propose more specific question_key |
| `story_fragmented` | Check for same_event_reference warrants |
| `missing_spine_anchor` | Check for alias/part-of warrants |

**No meta-claim → No LLM call.** Keep it deterministic when possible.

### High-ROI Semantic Tasks

Based on WFC postmortem, LLM is most valuable for:

1. **Alias/same-event resolution**: "Tai Po blaze" ↔ "Wang Fuk Court fire" → enables Core-B
2. **Event-type detection**: fire, election, indictment → improves facet expectations
3. **Entity disambiguation**: homonyms → reduces false bridges early
4. **Question_key normalization**: stop producing mega-surfaces via over-generic keys

### Anti-Percolation Tests (LLM-specific)

```python
class TestLLMAntiPercolation:
    def test_hub_anchor_no_merge(self):
        """Two stories share hub anchor; LLM must NOT merge cores."""
        story_wfc = make_story(focal="Wang Fuk Court")
        story_jimmy = make_story(focal="Jimmy Lai")
        shared_hub = "John Lee"  # Hub entity

        # LLM cannot use shared hub to merge these stories
        warrant = llm_propose_warrant(story_wfc, story_jimmy, shared_hub)
        assert warrant is None or warrant.warrant_type != "same_event_reference"

    def test_core_b_stays_periphery_without_structural(self):
        """Core-B candidate with semantic warrant only → stays periphery."""
        incident = make_incident(anchors={"Tai Po", "Chris Tang"})
        story = make_story(focal="Wang Fuk Court")

        # LLM proposes "same_event_reference" warrant
        warrant = SemanticWarrant(
            warrant_type="same_event_reference",
            source_text="Tai Po blaze victims",
            confidence=0.9,
            extraction_task="extract_geo_containment"
        )

        # Without structural witness: still periphery
        membership = classify_with_warrant(incident, story, warrant)
        assert membership == MembershipLevel.PERIPHERY

        # After extraction task succeeds with geo witness: Core-B
        add_witness(incident, GeoWitness("Tai Po ⊃ Wang Fuk Court"))
        membership = classify_with_warrant(incident, story, warrant)
        assert membership == MembershipLevel.CORE_B

    def test_related_storyline_not_merged(self):
        """Wong Kwok-ngon saga must be RELATED_STORY, not merged."""
        incident = make_incident(
            anchors={"Wong Kwok-ngon", "YouTube"},
            claims=["charged with sedition over fire videos"]
        )
        story = make_story(focal="Wang Fuk Court")

        warrant = llm_propose_warrant(incident, story)
        # Even if LLM sees "fire videos", it must not merge
        membership = classify_with_warrant(incident, story, warrant)
        assert membership in [MembershipLevel.RELATED_STORY, MembershipLevel.REJECT]
```


## Tests for Membrane Contract

```python
class TestMembrane:
    def test_core_a_automatic(self):
        """Spine in anchors → Core-A without additional evidence."""
        incident = make_incident(anchors={"Wang Fuk Court", "John Lee"})
        story = make_story(spine="Wang Fuk Court")
        assert classify_incident(incident, story) == MembershipLevel.CORE_A

    def test_core_b_requires_witness(self):
        """Spine not in anchors → Core-B only with structural witness."""
        incident = make_incident(anchors={"Tai Po", "Chris Tang"})
        story = make_story(spine="Wang Fuk Court")

        # Without witness: periphery
        assert classify_incident(incident, story) == MembershipLevel.PERIPHERY

        # With geo witness: Core-B
        add_witness(incident, GeoWitness("Tai Po ⊃ Wang Fuk Court"))
        assert classify_incident(incident, story) == MembershipLevel.CORE_B

    def test_hub_only_rejected(self):
        """Connection via hub anchors only → Reject."""
        incident = make_incident(anchors={"John Lee", "Hong Kong"})  # Both hubs
        story = make_story(spine="Wang Fuk Court")
        assert classify_incident(incident, story) == MembershipLevel.REJECT

    def test_unrelated_rejected(self):
        """Completely unrelated incident → Reject."""
        incident = make_incident(
            anchors={"Jimmy Lai", "Apple Daily", "CCP"},
            claims=["Lai sought downfall of Communist Party"]
        )
        story = make_story(spine="Wang Fuk Court")
        assert classify_incident(incident, story) == MembershipLevel.REJECT

    def test_separate_storyline_rejected(self):
        """Related but distinct storyline → Reject (or RELATED_STORY)."""
        incident = make_incident(
            anchors={"Wong Kwok-ngon", "YouTube"},
            claims=["charged with sedition over videos"]
        )
        story = make_story(spine="Wang Fuk Court")
        # Wong Kwok-ngon saga is about fire fallout but is its own story
        assert classify_incident(incident, story) in [
            MembershipLevel.REJECT,
            MembershipLevel.RELATED_STORY
        ]

    def test_no_leak_core_invariant(self):
        """Every core incident must be Core-A or Core-B with warrant."""
        story = build_story_from_incidents(incidents)
        for inc_id in story.core_incident_ids:
            membership = story.get_membership(inc_id)
            assert membership.level in [MembershipLevel.CORE_A, MembershipLevel.CORE_B]
            if membership.level == MembershipLevel.CORE_B:
                assert len(membership.witnesses) > 0
```


## Summary

| Concept | Definition | Identity |
|---------|------------|----------|
| Claim | Atomic observation | content hash |
| Surface | Question-keyed proposition | (scope_id, question_key) |
| Incident | Time-bounded occurrence | hash(anchor_motif, time_bin) |
| Story | Composed narrative with membrane | hash(sorted(focal_set), mode_id) |
| Lens | Entity profile for navigation | entity_id |

| Membership | Criterion | Persisted |
|------------|-----------|-----------|
| Core-A (single) | focal.primary ∈ anchors | Yes |
| Core-A (dyad) | both spines ∈ anchors OR dyad motif present | Yes |
| Core-B | ≥2 structural witnesses (one geo/event-type) | Yes |
| Periphery | semantic-only | No |
| Reject | hub-only or unrelated | No |
| Related_Story | separate storyline | Cross-link only |

Key invariant: **Core membership must not be created by shared hub anchors
or chain transitivity.** This is the membrane contract.


## Semantics Role Decision Table

Semantic similarity is **evidence, not definition**. It clusters by topic/framing/style,
which is exactly how you get mega-clusters and contamination.

| Context | Semantics Role | Allowed Operations |
|---------|---------------|-------------------|
| **Lens/Topic views** | Dominant | Cluster by embedding, group by topic |
| **Story core membership** | Insufficient | Candidate generation only; requires structural witness |
| **Story periphery** | Allowed | Attach periphery candidates, propose warrants |
| **Related-story links** | Allowed | Cross-story edges, "see also" navigation |
| **Titles/summaries** | Presentation only | Never defines identity |

### Safe Semantics Uses (High ROI)

1. **Candidate retrieval**: Top-k likely story/incident matches → then apply membrane deterministically
2. **Core-B warrants**: LLM proposes tokens → kernel accepts only with structural support
3. **Question_key normalization**: Reduce mega-surfaces by making keys less generic
4. **Entity disambiguation**: Prevent homonym bridges
5. **Titles/summaries**: Presentation only, never identity

### The Principled Compromise

```
Semantic (embedding/LLM) MAY propose: "incident I likely belongs to story S because..."
Core membership REQUIRES: at least one non-semantic witness (time/geo/event-type/motif)
                          OR explicit resolver artifact from Rigor A/B inquiry
Semantic-only edges: PERIPHERY only, never merge two cores
```


## Metrics

```python
# Core leak rate: % of core incidents that lack spine as anchor
core_leak_rate = (
    len([i for i in story.core_incident_ids if story.spine not in i.anchor_entities])
    / len(story.core_incident_ids)
)
# Target: core_leak_rate == 0.0 (all core incidents are Core-A or valid Core-B)

# Core-B ratio: % of core that is Core-B (vs Core-A)
core_b_ratio = len(story.core_b_ids) / len(story.core_incident_ids)
# Higher means story relies more on witnesses; lower means spine is well-extracted

# Witness scarcity: how often Core-B candidates are blocked
witness_scarcity = len(meta_claims["core_b_blocked_missing_witnesses"]) / total_candidate_count
```
