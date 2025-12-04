# Universal Event Formation Design - Discovery-Based Approach

## Core Principle

**Two-Layer Architecture:**
1. **Abstract Model Layer**: Discover patterns from data (universal, domain-agnostic)
2. **Natural Language Layer**: Translate patterns to human-readable names (domain-specific)

```
┌─────────────────────────────────────────────────┐
│  Natural Language Layer (Domain-Specific)       │
│  "Fire Breakout" | "Campaign Period" | "Deal"  │
└─────────────────────────────────────────────────┘
                      ↑
                  Translation
                      ↓
┌─────────────────────────────────────────────────┐
│  Abstract Model Layer (Universal)               │
│  Temporal Segment | Semantic Cluster | Network │
└─────────────────────────────────────────────────┘
```

---

## Layer 1: Abstract Model (Universal)

### Core Abstractions

**No assumptions except:**
1. Events have temporal bounds
2. Events involve entities (participants)
3. Events are described by claims (observations)
4. Structure emerges from data patterns

### Event Scaffold Discovery

```python
class EventScaffold:
    """
    Universal event structure discovered from first page
    """
    def __init__(self, first_page):
        self.id = generate_id()
        self.temporal_bounds = extract_temporal_bounds(first_page)
        self.participants = extract_entities(first_page)
        self.observations = extract_claims(first_page)

        # Discovered structure (not prescribed)
        self.temporal_segments = self._discover_temporal_segments()
        self.semantic_clusters = self._discover_semantic_clusters()
        self.coherence = self._compute_coherence()

    def _discover_temporal_segments(self):
        """
        Discover natural temporal boundaries from claims

        Algorithm:
        1. Extract all temporal markers from claims
        2. Cluster by temporal proximity
        3. Return temporal segments

        Returns:
            [TemporalSegment(t_start, t_end, claims), ...]
        """
        temporal_markers = []
        for claim in self.observations:
            markers = extract_temporal_markers(claim)
            # e.g., "at 17:59", "this morning", "days later"
            temporal_markers.append({
                'claim': claim,
                'time': parse_temporal_marker(markers),
                'precision': assess_precision(markers)
            })

        # Cluster by temporal proximity
        segments = temporal_clustering(
            temporal_markers,
            method="density_based",  # DBSCAN or similar
            threshold=infer_from_domain()  # Learn from data
        )

        return segments

    def _discover_semantic_clusters(self):
        """
        Discover natural semantic groupings from claim embeddings

        Algorithm:
        1. Get claim embeddings
        2. Cluster by semantic similarity
        3. Return semantic clusters

        Returns:
            [SemanticCluster(centroid, claims, keywords), ...]
        """
        embeddings = [claim.embedding for claim in self.observations]

        # Cluster semantically
        clusters = semantic_clustering(
            embeddings,
            method="hierarchical",  # Or DBSCAN
            min_cluster_size=2
        )

        # Extract keywords for each cluster
        for cluster in clusters:
            cluster.keywords = extract_salient_keywords(cluster.claims)
            cluster.topic_vector = compute_centroid(cluster.embeddings)

        return clusters

    def _compute_coherence(self):
        """
        Coherence = How well do temporal and semantic structures align?

        High coherence: Temporal segments match semantic clusters
        Low coherence: Mixed topics within time periods (complex event)
        """
        alignment_score = 0.0

        for temporal_seg in self.temporal_segments:
            for semantic_cluster in self.semantic_clusters:
                overlap = compute_overlap(temporal_seg.claims, semantic_cluster.claims)
                alignment_score += overlap

        # Normalize by total claims
        coherence = alignment_score / len(self.observations)

        return coherence
```

### Structure Discovery Patterns

**Pattern 1: Single Temporal Segment + Single Semantic Cluster**
```python
temporal_segments = [Segment(t0, t1, 5 claims)]
semantic_clusters = [Cluster(topic_A, 5 claims)]
coherence = 1.0  # Perfect alignment

→ Interpretation: Simple, focused event (MICRO)
→ Example: "Breaking: Fire breaks out at Wang Fuk Court"
```

**Pattern 2: Multiple Temporal Segments + Single Semantic Cluster**
```python
temporal_segments = [
    Segment(t0, t1, 3 claims),
    Segment(t2, t3, 2 claims)
]
semantic_clusters = [Cluster(topic_A, 5 claims)]
coherence = 1.0  # Still coherent, same topic evolving

→ Interpretation: Sequential development (MESO)
→ Example: "Fire breaks out → Fire spreads → Firefighters respond"
```

**Pattern 3: Single Temporal Segment + Multiple Semantic Clusters**
```python
temporal_segments = [Segment(t0, t1, 8 claims)]
semantic_clusters = [
    Cluster(topic_A, 3 claims),  # Fire incident
    Cluster(topic_B, 3 claims),  # Casualties
    Cluster(topic_C, 2 claims)   # Official response
]
coherence = 0.4  # Low - multiple topics at once

→ Interpretation: Comprehensive reporting of complex event (MESO/MACRO)
→ Example: First page covers fire + casualties + response simultaneously
```

**Pattern 4: Multiple Temporal Segments + Multiple Semantic Clusters (Mixed)**
```python
temporal_segments = [
    Segment(Nov 26 17:00-18:00, 4 claims),
    Segment(Nov 26 18:00-23:00, 3 claims),
    Segment(Nov 27+, 3 claims)
]
semantic_clusters = [
    Cluster(fire_incident, 3 claims),
    Cluster(emergency_response, 4 claims),
    Cluster(casualties, 3 claims)
]
coherence = 0.6  # Moderate - some overlap, some separation

→ Interpretation: Multi-phase event with temporal evolution (MESO)
→ Example: Fire → Response → Casualties reported over time
```

### Umbrella Decision (Universal Rule)

```python
def should_create_umbrella(scaffold):
    """
    Universal heuristic: Create umbrella if structure is complex

    Indicators of complexity:
    1. Multiple semantic clusters (≥3)
    2. Low coherence (<0.6)
    3. Wide temporal span (>12 hours from first page)
    """
    indicators = {
        'multi_topic': len(scaffold.semantic_clusters) >= 3,
        'low_coherence': scaffold.coherence < 0.6,
        'wide_temporal': scaffold.temporal_span.hours > 12,
        'many_claims': len(scaffold.observations) >= 8
    }

    # If 2+ indicators, create umbrella
    complexity_score = sum(indicators.values())

    if complexity_score >= 2:
        return True, "UMBRELLA_EVENT"
    else:
        return False, "SIMPLE_EVENT"
```

---

## Layer 2: Natural Language Translation

### Domain Pattern Matching

**For each discovered segment/cluster, translate to natural language:**

```python
class DomainTranslator:
    """
    Translate abstract patterns to domain-specific natural language
    """

    def __init__(self):
        # Domain patterns learned from examples or predefined
        self.domain_patterns = self._load_domain_patterns()

    def translate_cluster_to_phase_name(
        self,
        semantic_cluster: SemanticCluster,
        temporal_segment: TemporalSegment,
        event_type: str
    ) -> str:
        """
        Translate abstract cluster to natural phase name

        Uses:
        1. Cluster keywords
        2. Temporal position (early/mid/late)
        3. Event type (fire, election, scandal, etc.)
        4. Domain patterns
        """

        # Extract characteristics
        keywords = semantic_cluster.keywords
        temporal_position = self._assess_temporal_position(temporal_segment)

        # Match to domain pattern
        if event_type == "FIRE" or event_type == "DISASTER":
            return self._translate_disaster_phase(keywords, temporal_position)

        elif event_type == "ELECTION":
            return self._translate_election_phase(keywords, temporal_position)

        elif event_type == "SCANDAL":
            return self._translate_scandal_phase(keywords, temporal_position)

        else:
            # Generic fallback
            return self._generic_phase_name(keywords, temporal_position)

    def _translate_disaster_phase(self, keywords, position):
        """
        Fire/Disaster domain translation
        """
        # Keyword matching for disaster events
        if any(kw in keywords for kw in ['broke out', 'started', 'ignited', 'occurred']):
            return "Fire Breakout"

        elif any(kw in keywords for kw in ['firefighters', 'rescue', 'evacuate', 'battle']):
            return "Emergency Response"

        elif any(kw in keywords for kw in ['dead', 'casualties', 'injured', 'toll']):
            return "Casualty Assessment"

        elif any(kw in keywords for kw in ['investigation', 'probe', 'arrested', 'charged']):
            return "Investigation"

        elif any(kw in keywords for kw in ['reform', 'regulations', 'inquiry', 'government']):
            return "Political Response"

        else:
            # Temporal fallback
            if position == "early":
                return "Initial Incident"
            elif position == "middle":
                return "Ongoing Response"
            else:
                return "Aftermath"

    def _translate_election_phase(self, keywords, position):
        """
        Election domain translation
        """
        if any(kw in keywords for kw in ['campaign', 'rally', 'debate']):
            return "Campaign Period"

        elif any(kw in keywords for kw in ['vote', 'ballot', 'polls']):
            return "Voting Day"

        elif any(kw in keywords for kw in ['results', 'winner', 'declared']):
            return "Results Announcement"

        elif any(kw in keywords for kw in ['transition', 'inauguration', 'sworn in']):
            return "Transition Period"

        else:
            return f"Election Phase ({position})"

    def _translate_scandal_phase(self, keywords, position):
        """
        Scandal domain translation
        """
        if any(kw in keywords for kw in ['revealed', 'leaked', 'exposed']):
            return "Initial Revelation"

        elif any(kw in keywords for kw in ['denied', 'statement', 'response']):
            return "Official Response"

        elif any(kw in keywords for kw in ['investigation', 'probe', 'inquiry']):
            return "Investigation"

        elif any(kw in keywords for kw in ['resigned', 'stepped down', 'consequences']):
            return "Consequences"

        else:
            return f"Scandal Development ({position})"

    def _generic_phase_name(self, keywords, position):
        """
        Generic fallback using top keywords
        """
        # Use top 2-3 keywords
        top_keywords = keywords[:3]
        name = " ".join(top_keywords).title()

        # Add temporal qualifier
        if position == "early":
            return f"Initial {name}"
        elif position == "late":
            return f"Late {name}"
        else:
            return name

    def _assess_temporal_position(self, segment):
        """
        Is this segment early, middle, or late in event timeline?
        """
        # Compare to event bounds
        if segment.start < event.earliest_time + timedelta(hours=6):
            return "early"
        elif segment.end > event.latest_time - timedelta(hours=6):
            return "late"
        else:
            return "middle"
```

### Example: Hong Kong Fire Translation

**Abstract Discovery (Layer 1):**
```python
# First page processed
scaffold = EventScaffold(first_page)

# Discovered:
temporal_segments = [
    Segment(t=2025-11-26 17:59-18:30, claims=[c1, c2]),  # Earliest
    Segment(t=2025-11-26 18:30-20:00, claims=[c3, c4, c5]),  # Middle
    Segment(t=2025-11-26 20:00+, claims=[c6])  # Late
]

semantic_clusters = [
    Cluster(keywords=['blaze', 'engulfs', 'fire', 'started'], claims=[c1, c2]),
    Cluster(keywords=['firefighters', 'rescue', 'evacuate'], claims=[c3, c4]),
    Cluster(keywords=['dead', 'casualties', 'toll'], claims=[c5, c6])
]

coherence = 0.55  # Moderate
complexity_score = 3  # Multi-topic + low coherence + many claims
→ Decision: CREATE UMBRELLA
```

**Natural Language Translation (Layer 2):**
```python
# Detect event type
event_type = infer_event_type(semantic_clusters)
# → "FIRE" (based on keywords)

translator = DomainTranslator()

# Translate each cluster to phase name
phases = []
for cluster, segment in zip(semantic_clusters, temporal_segments):
    phase_name = translator.translate_cluster_to_phase_name(
        cluster, segment, event_type="FIRE"
    )
    phases.append(phase_name)

# Result:
phases = [
    "Fire Breakout",       # keywords: blaze, engulfs, started
    "Emergency Response",  # keywords: firefighters, rescue
    "Casualty Assessment"  # keywords: dead, casualties, toll
]
```

**Neo4j Graph (combines both layers):**
```cypher
// Abstract layer stored in properties
(e:Event {
    id: "...",
    canonical_name: "2025 Hong Kong Tai Po Fire",
    coherence: 0.55,
    complexity_score: 3,
    temporal_span_hours: 8,
    semantic_cluster_count: 3
})

// Natural language layer in phase names
(e)-[:HAS_PHASE {sequence: 1}]->(p1:Phase {
    name: "Fire Breakout",  // Natural language (Layer 2)
    cluster_keywords: ["blaze", "engulfs", "fire"],  // Abstract (Layer 1)
    temporal_start: datetime("2025-11-26T17:59"),
    temporal_end: datetime("2025-11-26T18:30")
})

(e)-[:HAS_PHASE {sequence: 2}]->(p2:Phase {
    name: "Emergency Response",
    cluster_keywords: ["firefighters", "rescue", "evacuate"],
    temporal_start: datetime("2025-11-26T18:30"),
    temporal_end: datetime("2025-11-26T20:00")
})
```

---

## Universal Matching Algorithm

**No domain-specific logic in matching:**

```python
def find_candidate_events(new_page):
    """
    Universal event matching (works for any domain)
    """
    candidates = query_neo4j("""
        MATCH (e:Event)
        WHERE e.status IN ['provisional', 'emerging', 'stable']
        RETURN e
    """)

    scores = []
    for event in candidates:
        score = compute_match_score(
            semantic_similarity=cosine(new_page.embedding, event.centroid),
            entity_overlap=jaccard(new_page.entities, event.entities),
            temporal_proximity=temporal_score(new_page.time, event.bounds),
            location_overlap=location_score(new_page.locations, event.locations)
        )
        scores.append((event, score))

    return sorted(scores, key=lambda x: x[1], reverse=True)
```

**Phase assignment also universal:**

```python
def find_best_phase(event, new_page_clusters):
    """
    Match new semantic clusters to existing phases
    """
    new_clusters = discover_semantic_clusters(new_page.claims)

    best_matches = []
    for new_cluster in new_clusters:
        # Find most similar existing phase
        for phase in event.phases:
            similarity = cosine(
                new_cluster.topic_vector,
                phase.cluster_keywords_embedding
            )
            best_matches.append((phase, similarity))

    if max(best_matches, key=lambda x: x[1])[1] > 0.7:
        # Strong match → attach to existing phase
        return best_matches[0][0]
    else:
        # No match → create new phase
        return create_new_phase(new_cluster, event)
```

---

## Key Benefits

**1. Universal:**
- No hardcoded phase types
- Works for any event domain
- Discovers structure from data

**2. Natural Language:**
- Translates patterns to readable names
- Domain-aware labeling
- Human-friendly output

**3. Learnable:**
- Domain patterns can be learned from examples
- System improves with more data
- Can add new domains without changing core

**4. Traceable:**
- Abstract model preserved in graph properties
- Can always see WHY a phase was named something
- Debugging and refinement possible

---

## Implementation Strategy

### Phase 1: Universal Core
```python
class UniversalEventWorker:
    def process_page(self, page):
        # 1. Discover abstract structure
        scaffold = discover_scaffold(page)

        # 2. Universal matching
        candidates = find_candidates(page)

        # 3. Universal decision
        if should_create_event(candidates, threshold):
            event = create_event(scaffold)
        else:
            attach_to_event(best_candidate, page)
```

### Phase 2: Domain Translation
```python
class DomainAwarePresentation:
    def present_event(self, event):
        # Translate abstract model to natural language
        event_type = infer_type(event)
        translator = get_translator(event_type)

        readable_phases = [
            translator.translate(phase)
            for phase in event.phases
        ]

        return {
            'title': event.canonical_name,
            'phases': readable_phases,
            'abstract_model': event.raw_structure  # For debugging
        }
```

---

## Success: Same Abstract Model, Different Translations

**Event: Hong Kong Fire**
- Abstract: 3 temporal segments, 3 semantic clusters, coherence 0.55
- Translation: "Fire Breakout → Emergency Response → Casualty Assessment"

**Event: US Election**
- Abstract: 4 temporal segments, 4 semantic clusters, coherence 0.75
- Translation: "Campaign Period → Voting Day → Results → Transition"

**Event: Corporate Scandal**
- Abstract: 3 temporal segments, 3 semantic clusters, coherence 0.45
- Translation: "Initial Revelation → Official Response → Investigation"

**Same underlying machinery, different surface presentation!**

Is this the right direction?
