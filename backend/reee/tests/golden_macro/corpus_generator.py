"""
Macro Corpus Generator
======================

Generates ~1000 claims covering all 8 archetypes for comprehensive kernel validation.

Archetypes:
1. Star Story (WFC) - 150 claims: spine + rotating companions
2. Dyad Story (Do Kwon + Terraform) - 80 claims: multi-spine promotion
3. Hub Adversary - 200 claims: hub entity pressure
4. Homonym Adversary - 60 claims: disambiguation test
5. Scope Pollution - 100 claims: surface isolation
6. Time Missingness (50%) - 120 claims: conservative blocking
7. Typed Conflicts - 90 claims: Jaynes posterior, conflicts
8. Related Storyline - 100 claims: RELATED_STORY link

Total: ~900 claims (can be adjusted)

Usage:
    from corpus_generator import MacroCorpusGenerator

    generator = MacroCorpusGenerator(seed=42)
    corpus = generator.generate()
    generator.save_corpus(corpus, "corpus.json")
"""

import hashlib
import json
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Set, Optional


@dataclass
class GeneratedClaim:
    """A generated claim for the corpus."""
    id: str
    text: str
    publisher: str
    reported_time: str
    event_time: Optional[str]
    question_key: str
    anchor_entities: List[str]
    scope_id: str
    archetype: str  # Which archetype this claim belongs to

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GeneratedIncident:
    """A generated incident for the corpus."""
    id: str
    description: str
    anchor_entities: List[str]
    companion_entities: List[str]
    time_start: str
    time_end: str
    claim_ids: List[str]
    archetype: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GeneratedEntity:
    """A generated entity for the corpus."""
    id: str
    name: str
    entity_type: str
    role: str  # spine, hub, companion, etc.
    archetype: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CorpusManifest:
    """Manifest describing the generated corpus."""
    corpus_id: str
    seed: int
    generated_at: str
    total_claims: int
    total_incidents: int
    total_entities: int
    archetypes: Dict[str, Dict[str, Any]]
    expected_invariants: List[str]
    quantitative_bounds: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GeneratedCorpus:
    """The complete generated corpus."""
    manifest: CorpusManifest
    entities: List[GeneratedEntity]
    claims: List[GeneratedClaim]
    incidents: List[GeneratedIncident]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'manifest': self.manifest.to_dict(),
            'entities': [e.to_dict() for e in self.entities],
            'claims': [c.to_dict() for c in self.claims],
            'incidents': [i.to_dict() for i in self.incidents],
        }


class MacroCorpusGenerator:
    """
    Generates a macro corpus covering all 8 archetypes.

    Uses deterministic generation with a seed for reproducibility.
    """

    PUBLISHERS = ['reuters', 'ap', 'bbc', 'guardian', 'nyt', 'ft', 'scmp', 'cnn', 'dw', 'afp']

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.base_time = datetime(2024, 1, 1, 0, 0, 0)

        # Counters for deterministic IDs
        self.claim_counter = 0
        self.incident_counter = 0
        self.entity_counter = 0

    def _hash_id(self, prefix: str, content: str) -> str:
        """Generate deterministic hash-based ID."""
        h = hashlib.sha256(f"{self.seed}:{content}".encode()).hexdigest()[:12]
        return f"{prefix}_{h}"

    def _next_claim_id(self) -> str:
        self.claim_counter += 1
        return f"claim_{self.claim_counter:04d}"

    def _next_incident_id(self) -> str:
        self.incident_counter += 1
        return f"inc_{self.incident_counter:04d}"

    def _next_entity_id(self) -> str:
        self.entity_counter += 1
        return f"ent_{self.entity_counter:04d}"

    def _random_publisher(self) -> str:
        return self.rng.choice(self.PUBLISHERS)

    def _time_offset(self, days: int, hours: int = 0) -> datetime:
        return self.base_time + timedelta(days=days, hours=hours)

    def _iso_time(self, dt: datetime) -> str:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # =========================================================================
    # ARCHETYPE GENERATORS
    # =========================================================================

    def _generate_star_story(self) -> tuple[List[GeneratedEntity], List[GeneratedClaim], List[GeneratedIncident]]:
        """
        Archetype 1: Star Story (WFC pattern)

        Spine entity appears in all incidents with rotating companions.
        ~150 claims, ~15 incidents
        """
        entities = []
        claims = []
        incidents = []

        # Spine entity
        spine = GeneratedEntity(
            id=self._next_entity_id(),
            name="Global Relief Foundation",
            entity_type="organization",
            role="spine",
            archetype="star_story",
        )
        entities.append(spine)

        # Companion entities (each appears in 1-2 incidents only)
        companions = []
        companion_names = [
            ("African Union", "organization"),
            ("Southeast Asian Nations", "organization"),
            ("European Commission", "organization"),
            ("United Nations", "organization"),
            ("World Bank", "organization"),
            ("Red Cross", "organization"),
            ("Bill Gates Foundation", "organization"),
            ("USAID", "organization"),
        ]
        for name, etype in companion_names:
            ent = GeneratedEntity(
                id=self._next_entity_id(),
                name=name,
                entity_type=etype,
                role="companion",
                archetype="star_story",
            )
            entities.append(ent)
            companions.append(ent)

        # Generate incidents: spine + 1-2 companions each
        question_keys = ["partnership", "funding", "expansion", "policy", "event", "announcement"]

        for i in range(15):
            day = i * 2  # Spread over 30 days

            # Pick 1-2 companions for this incident
            num_companions = self.rng.randint(1, 2)
            incident_companions = self.rng.sample(companions, num_companions)

            anchor_entities = [spine.name] + [c.name for c in incident_companions]
            qk = self.rng.choice(question_keys)
            scope_id = f"scope_grf_{qk}_{i}"

            # Generate 8-12 claims per incident
            incident_claims = []
            num_claims = self.rng.randint(8, 12)

            for j in range(num_claims):
                claim = GeneratedClaim(
                    id=self._next_claim_id(),
                    text=f"{spine.name} {qk} with {', '.join(c.name for c in incident_companions)} - report {j+1}",
                    publisher=self._random_publisher(),
                    reported_time=self._iso_time(self._time_offset(day, j)),
                    event_time=self._iso_time(self._time_offset(day, 0)),
                    question_key=qk,
                    anchor_entities=anchor_entities,
                    scope_id=scope_id,
                    archetype="star_story",
                )
                claims.append(claim)
                incident_claims.append(claim.id)

            incident = GeneratedIncident(
                id=self._next_incident_id(),
                description=f"{spine.name} {qk} with {', '.join(c.name for c in incident_companions)}",
                anchor_entities=anchor_entities,
                companion_entities=[],
                time_start=self._iso_time(self._time_offset(day, 0)),
                time_end=self._iso_time(self._time_offset(day, 12)),
                claim_ids=incident_claims,
                archetype="star_story",
            )
            incidents.append(incident)

        return entities, claims, incidents

    def _generate_dyad_story(self) -> tuple[List[GeneratedEntity], List[GeneratedClaim], List[GeneratedIncident]]:
        """
        Archetype 2: Dyad Story (Do Kwon + Terraform pattern)

        Two entities that always appear together (PMI > 2).
        ~80 claims, ~8 incidents
        """
        entities = []
        claims = []
        incidents = []

        # Dyad entities
        entity1 = GeneratedEntity(
            id=self._next_entity_id(),
            name="Marcus Chen",
            entity_type="person",
            role="dyad_spine",
            archetype="dyad_story",
        )
        entity2 = GeneratedEntity(
            id=self._next_entity_id(),
            name="NovaTech Labs",
            entity_type="organization",
            role="dyad_spine",
            archetype="dyad_story",
        )
        entities.extend([entity1, entity2])

        # Generate incidents: both entities always together
        question_keys = ["trial", "investigation", "settlement", "testimony", "ruling", "appeal"]

        for i in range(8):
            day = 40 + i * 3  # Start after star story
            qk = self.rng.choice(question_keys)
            scope_id = f"scope_dyad_{qk}_{i}"

            anchor_entities = [entity1.name, entity2.name]

            # Generate 8-12 claims per incident
            incident_claims = []
            num_claims = self.rng.randint(8, 12)

            for j in range(num_claims):
                claim = GeneratedClaim(
                    id=self._next_claim_id(),
                    text=f"{entity1.name} and {entity2.name} {qk} - update {j+1}",
                    publisher=self._random_publisher(),
                    reported_time=self._iso_time(self._time_offset(day, j)),
                    event_time=self._iso_time(self._time_offset(day, 0)),
                    question_key=qk,
                    anchor_entities=anchor_entities,
                    scope_id=scope_id,
                    archetype="dyad_story",
                )
                claims.append(claim)
                incident_claims.append(claim.id)

            incident = GeneratedIncident(
                id=self._next_incident_id(),
                description=f"{entity1.name} and {entity2.name} {qk}",
                anchor_entities=anchor_entities,
                companion_entities=[],
                time_start=self._iso_time(self._time_offset(day, 0)),
                time_end=self._iso_time(self._time_offset(day, 12)),
                claim_ids=incident_claims,
                archetype="dyad_story",
            )
            incidents.append(incident)

        return entities, claims, incidents

    def _generate_hub_adversary(self) -> tuple[List[GeneratedEntity], List[GeneratedClaim], List[GeneratedIncident]]:
        """
        Archetype 3: Hub Adversary

        Hub entity (appears in 30%+ of corpus) must not define stories.
        ~200 claims across ~20 incidents, hub in 15+ of them
        """
        entities = []
        claims = []
        incidents = []

        # Hub entity
        hub = GeneratedEntity(
            id=self._next_entity_id(),
            name="Pacific Region",
            entity_type="location",
            role="hub",
            archetype="hub_adversary",
        )
        entities.append(hub)

        # Non-hub spines (each forms their own story)
        spine_names = [
            ("Tech Summit 2024", "event"),
            ("Climate Alliance", "organization"),
            ("Trade Ministry", "organization"),
        ]
        spines = []
        for name, etype in spine_names:
            ent = GeneratedEntity(
                id=self._next_entity_id(),
                name=name,
                entity_type=etype,
                role="spine",
                archetype="hub_adversary",
            )
            entities.append(ent)
            spines.append(ent)

        # Generate incidents for each spine (with hub present in most)
        for spine_idx, spine in enumerate(spines):
            for i in range(6):  # 6 incidents per spine
                day = 70 + spine_idx * 20 + i * 3
                qk = self.rng.choice(["announcement", "meeting", "policy", "event"])
                scope_id = f"scope_hub_{spine.name}_{qk}_{i}"

                # Hub present in 80% of incidents
                include_hub = self.rng.random() < 0.8
                anchor_entities = [spine.name]
                if include_hub:
                    anchor_entities.append(hub.name)

                incident_claims = []
                num_claims = self.rng.randint(8, 12)

                for j in range(num_claims):
                    claim = GeneratedClaim(
                        id=self._next_claim_id(),
                        text=f"{spine.name} {qk} {'in ' + hub.name if include_hub else ''} - report {j+1}",
                        publisher=self._random_publisher(),
                        reported_time=self._iso_time(self._time_offset(day, j)),
                        event_time=self._iso_time(self._time_offset(day, 0)),
                        question_key=qk,
                        anchor_entities=anchor_entities,
                        scope_id=scope_id,
                        archetype="hub_adversary",
                    )
                    claims.append(claim)
                    incident_claims.append(claim.id)

                incident = GeneratedIncident(
                    id=self._next_incident_id(),
                    description=f"{spine.name} {qk}",
                    anchor_entities=anchor_entities,
                    companion_entities=[],
                    time_start=self._iso_time(self._time_offset(day, 0)),
                    time_end=self._iso_time(self._time_offset(day, 12)),
                    claim_ids=incident_claims,
                    archetype="hub_adversary",
                )
                incidents.append(incident)

        # Add some hub-only incidents (should NOT form stories)
        for i in range(2):
            day = 130 + i * 5
            qk = "general_news"
            scope_id = f"scope_hub_only_{i}"

            incident_claims = []
            for j in range(5):
                claim = GeneratedClaim(
                    id=self._next_claim_id(),
                    text=f"{hub.name} general update - report {j+1}",
                    publisher=self._random_publisher(),
                    reported_time=self._iso_time(self._time_offset(day, j)),
                    event_time=self._iso_time(self._time_offset(day, 0)),
                    question_key=qk,
                    anchor_entities=[hub.name],
                    scope_id=scope_id,
                    archetype="hub_adversary",
                )
                claims.append(claim)
                incident_claims.append(claim.id)

            incident = GeneratedIncident(
                id=self._next_incident_id(),
                description=f"{hub.name} general news",
                anchor_entities=[hub.name],
                companion_entities=[],
                time_start=self._iso_time(self._time_offset(day, 0)),
                time_end=self._iso_time(self._time_offset(day, 12)),
                claim_ids=incident_claims,
                archetype="hub_adversary",
            )
            incidents.append(incident)

        return entities, claims, incidents

    def _generate_homonym_adversary(self) -> tuple[List[GeneratedEntity], List[GeneratedClaim], List[GeneratedIncident]]:
        """
        Archetype 4: Homonym Adversary

        Same name, different entities - must not merge.
        ~60 claims, ~6 incidents
        """
        entities = []
        claims = []
        incidents = []

        # Two entities with same display name but different contexts
        entity1 = GeneratedEntity(
            id=self._next_entity_id(),
            name="Phoenix",
            entity_type="organization",
            role="homonym_1",
            archetype="homonym_adversary",
        )
        entity1_context = GeneratedEntity(
            id=self._next_entity_id(),
            name="Aerospace Industry",
            entity_type="industry",
            role="context_1",
            archetype="homonym_adversary",
        )

        entity2 = GeneratedEntity(
            id=self._next_entity_id(),
            name="Phoenix",
            entity_type="sports_team",
            role="homonym_2",
            archetype="homonym_adversary",
        )
        entity2_context = GeneratedEntity(
            id=self._next_entity_id(),
            name="Basketball League",
            entity_type="organization",
            role="context_2",
            archetype="homonym_adversary",
        )

        entities.extend([entity1, entity1_context, entity2, entity2_context])

        # Generate incidents for Phoenix (aerospace) - should NOT merge with Phoenix (sports)
        for i in range(3):
            day = 140 + i * 5
            qk = "contract"
            scope_id = f"scope_phoenix_aero_{i}"

            anchor_entities = ["Phoenix", "Aerospace Industry"]

            incident_claims = []
            for j in range(10):
                claim = GeneratedClaim(
                    id=self._next_claim_id(),
                    text=f"Phoenix aerospace company {qk} - report {j+1}",
                    publisher=self._random_publisher(),
                    reported_time=self._iso_time(self._time_offset(day, j)),
                    event_time=self._iso_time(self._time_offset(day, 0)),
                    question_key=qk,
                    anchor_entities=anchor_entities,
                    scope_id=scope_id,
                    archetype="homonym_adversary",
                )
                claims.append(claim)
                incident_claims.append(claim.id)

            incident = GeneratedIncident(
                id=self._next_incident_id(),
                description=f"Phoenix aerospace {qk}",
                anchor_entities=anchor_entities,
                companion_entities=[],
                time_start=self._iso_time(self._time_offset(day, 0)),
                time_end=self._iso_time(self._time_offset(day, 12)),
                claim_ids=incident_claims,
                archetype="homonym_adversary",
            )
            incidents.append(incident)

        # Generate incidents for Phoenix (sports)
        for i in range(3):
            day = 155 + i * 5
            qk = "game"
            scope_id = f"scope_phoenix_sports_{i}"

            anchor_entities = ["Phoenix", "Basketball League"]

            incident_claims = []
            for j in range(10):
                claim = GeneratedClaim(
                    id=self._next_claim_id(),
                    text=f"Phoenix basketball team {qk} - report {j+1}",
                    publisher=self._random_publisher(),
                    reported_time=self._iso_time(self._time_offset(day, j)),
                    event_time=self._iso_time(self._time_offset(day, 0)),
                    question_key=qk,
                    anchor_entities=anchor_entities,
                    scope_id=scope_id,
                    archetype="homonym_adversary",
                )
                claims.append(claim)
                incident_claims.append(claim.id)

            incident = GeneratedIncident(
                id=self._next_incident_id(),
                description=f"Phoenix sports {qk}",
                anchor_entities=anchor_entities,
                companion_entities=[],
                time_start=self._iso_time(self._time_offset(day, 0)),
                time_end=self._iso_time(self._time_offset(day, 12)),
                claim_ids=incident_claims,
                archetype="homonym_adversary",
            )
            incidents.append(incident)

        return entities, claims, incidents

    def _generate_scope_pollution(self) -> tuple[List[GeneratedEntity], List[GeneratedClaim], List[GeneratedIncident]]:
        """
        Archetype 5: Scope Pollution

        Same question_key in different scopes must produce separate surfaces.
        ~100 claims, ~10 incidents
        """
        entities = []
        claims = []
        incidents = []

        # Different organizations with same question types
        orgs = [
            ("Federal Reserve", "financial"),
            ("European Central Bank", "financial"),
            ("Bank of Japan", "financial"),
            ("People's Bank of China", "financial"),
            ("Bank of England", "financial"),
        ]

        for org_name, sector in orgs:
            ent = GeneratedEntity(
                id=self._next_entity_id(),
                name=org_name,
                entity_type="organization",
                role="scope_anchor",
                archetype="scope_pollution",
            )
            entities.append(ent)

        # Generate incidents - each org has "interest_rate" question but different scopes
        for i, (org_name, sector) in enumerate(orgs):
            for j in range(2):  # 2 incidents per org
                day = 170 + i * 10 + j * 3
                qk = "interest_rate"  # Same question_key!
                scope_id = f"scope_{org_name.replace(' ', '_')}_{qk}"  # Different scopes

                anchor_entities = [org_name]

                incident_claims = []
                for k in range(10):
                    claim = GeneratedClaim(
                        id=self._next_claim_id(),
                        text=f"{org_name} {qk} decision - report {k+1}",
                        publisher=self._random_publisher(),
                        reported_time=self._iso_time(self._time_offset(day, k)),
                        event_time=self._iso_time(self._time_offset(day, 0)),
                        question_key=qk,
                        anchor_entities=anchor_entities,
                        scope_id=scope_id,
                        archetype="scope_pollution",
                    )
                    claims.append(claim)
                    incident_claims.append(claim.id)

                incident = GeneratedIncident(
                    id=self._next_incident_id(),
                    description=f"{org_name} {qk} announcement",
                    anchor_entities=anchor_entities,
                    companion_entities=[],
                    time_start=self._iso_time(self._time_offset(day, 0)),
                    time_end=self._iso_time(self._time_offset(day, 12)),
                    claim_ids=incident_claims,
                    archetype="scope_pollution",
                )
                incidents.append(incident)

        return entities, claims, incidents

    def _generate_time_missingness(self) -> tuple[List[GeneratedEntity], List[GeneratedClaim], List[GeneratedIncident]]:
        """
        Archetype 6: Time Missingness (50%)

        Claims with missing event_time - tests conservative blocking.
        ~120 claims, ~12 incidents
        """
        entities = []
        claims = []
        incidents = []

        spine = GeneratedEntity(
            id=self._next_entity_id(),
            name="International Summit",
            entity_type="event",
            role="spine",
            archetype="time_missingness",
        )
        entities.append(spine)

        companions = [
            ("World Leaders Forum", "organization"),
            ("Climate Committee", "organization"),
            ("Security Council", "organization"),
        ]
        for name, etype in companions:
            ent = GeneratedEntity(
                id=self._next_entity_id(),
                name=name,
                entity_type=etype,
                role="companion",
                archetype="time_missingness",
            )
            entities.append(ent)

        # Generate incidents with 50% time missingness
        for i in range(12):
            day = 220 + i * 3
            qk = self.rng.choice(["session", "resolution", "statement", "agreement"])
            scope_id = f"scope_summit_{qk}_{i}"

            companion = companions[i % len(companions)]
            anchor_entities = [spine.name, companion[0]]

            incident_claims = []
            for j in range(10):
                # 50% have missing event_time
                has_event_time = self.rng.random() > 0.5

                claim = GeneratedClaim(
                    id=self._next_claim_id(),
                    text=f"{spine.name} {qk} with {companion[0]} - report {j+1}",
                    publisher=self._random_publisher(),
                    reported_time=self._iso_time(self._time_offset(day, j)),
                    event_time=self._iso_time(self._time_offset(day, 0)) if has_event_time else None,
                    question_key=qk,
                    anchor_entities=anchor_entities,
                    scope_id=scope_id,
                    archetype="time_missingness",
                )
                claims.append(claim)
                incident_claims.append(claim.id)

            incident = GeneratedIncident(
                id=self._next_incident_id(),
                description=f"{spine.name} {qk}",
                anchor_entities=anchor_entities,
                companion_entities=[],
                time_start=self._iso_time(self._time_offset(day, 0)),
                time_end=self._iso_time(self._time_offset(day, 12)),
                claim_ids=incident_claims,
                archetype="time_missingness",
            )
            incidents.append(incident)

        return entities, claims, incidents

    def _generate_typed_conflicts(self) -> tuple[List[GeneratedEntity], List[GeneratedClaim], List[GeneratedIncident]]:
        """
        Archetype 7: Typed Conflicts

        Conflicting values for same typed question (e.g., death toll).
        Tests Jaynes posterior computation.
        ~90 claims, ~9 incidents
        """
        entities = []
        claims = []
        incidents = []

        spine = GeneratedEntity(
            id=self._next_entity_id(),
            name="Industrial Accident",
            entity_type="event",
            role="spine",
            archetype="typed_conflicts",
        )
        location = GeneratedEntity(
            id=self._next_entity_id(),
            name="Factory District",
            entity_type="location",
            role="companion",
            archetype="typed_conflicts",
        )
        entities.extend([spine, location])

        # Generate incidents with conflicting typed values
        for i in range(9):
            day = 260 + i * 3
            qk = "casualty_count"  # Typed question
            scope_id = f"scope_accident_{qk}_{i}"

            anchor_entities = [spine.name, location.name]

            incident_claims = []
            # Generate conflicting counts from different sources
            base_count = 10 + i * 5
            variations = [base_count - 2, base_count, base_count + 3, base_count + 1, base_count - 1]

            for j, count in enumerate(variations):
                claim = GeneratedClaim(
                    id=self._next_claim_id(),
                    text=f"{spine.name}: {count} casualties reported",
                    publisher=self._random_publisher(),
                    reported_time=self._iso_time(self._time_offset(day, j)),
                    event_time=self._iso_time(self._time_offset(day, 0)),
                    question_key=qk,
                    anchor_entities=anchor_entities,
                    scope_id=scope_id,
                    archetype="typed_conflicts",
                )
                claims.append(claim)
                incident_claims.append(claim.id)

            # Add some non-conflicting claims
            for j in range(5):
                claim = GeneratedClaim(
                    id=self._next_claim_id(),
                    text=f"{spine.name} response ongoing - update {j+1}",
                    publisher=self._random_publisher(),
                    reported_time=self._iso_time(self._time_offset(day, 5 + j)),
                    event_time=self._iso_time(self._time_offset(day, 0)),
                    question_key="response_status",
                    anchor_entities=anchor_entities,
                    scope_id=scope_id,
                    archetype="typed_conflicts",
                )
                claims.append(claim)
                incident_claims.append(claim.id)

            incident = GeneratedIncident(
                id=self._next_incident_id(),
                description=f"{spine.name} in {location.name}",
                anchor_entities=anchor_entities,
                companion_entities=[],
                time_start=self._iso_time(self._time_offset(day, 0)),
                time_end=self._iso_time(self._time_offset(day, 12)),
                claim_ids=incident_claims,
                archetype="typed_conflicts",
            )
            incidents.append(incident)

        return entities, claims, incidents

    def _generate_related_storyline(self) -> tuple[List[GeneratedEntity], List[GeneratedClaim], List[GeneratedIncident]]:
        """
        Archetype 8: Related Storyline

        Two distinct stories that are related but should NOT merge.
        Tests RELATED_STORY link vs membership.
        ~100 claims, ~10 incidents
        """
        entities = []
        claims = []
        incidents = []

        # Story A: Election
        election = GeneratedEntity(
            id=self._next_entity_id(),
            name="National Election 2024",
            entity_type="event",
            role="spine_a",
            archetype="related_storyline",
        )
        candidate1 = GeneratedEntity(
            id=self._next_entity_id(),
            name="Candidate Alpha",
            entity_type="person",
            role="companion_a",
            archetype="related_storyline",
        )

        # Story B: Policy (related to election but separate)
        policy = GeneratedEntity(
            id=self._next_entity_id(),
            name="Economic Policy Reform",
            entity_type="policy",
            role="spine_b",
            archetype="related_storyline",
        )
        ministry = GeneratedEntity(
            id=self._next_entity_id(),
            name="Finance Ministry",
            entity_type="organization",
            role="companion_b",
            archetype="related_storyline",
        )

        entities.extend([election, candidate1, policy, ministry])

        # Generate Story A incidents (election)
        for i in range(5):
            day = 290 + i * 3
            qk = self.rng.choice(["polling", "campaign", "debate", "rally"])
            scope_id = f"scope_election_{qk}_{i}"

            anchor_entities = [election.name, candidate1.name]

            incident_claims = []
            for j in range(10):
                claim = GeneratedClaim(
                    id=self._next_claim_id(),
                    text=f"{election.name}: {candidate1.name} {qk} - report {j+1}",
                    publisher=self._random_publisher(),
                    reported_time=self._iso_time(self._time_offset(day, j)),
                    event_time=self._iso_time(self._time_offset(day, 0)),
                    question_key=qk,
                    anchor_entities=anchor_entities,
                    scope_id=scope_id,
                    archetype="related_storyline",
                )
                claims.append(claim)
                incident_claims.append(claim.id)

            incident = GeneratedIncident(
                id=self._next_incident_id(),
                description=f"{election.name} {qk}",
                anchor_entities=anchor_entities,
                companion_entities=[],
                time_start=self._iso_time(self._time_offset(day, 0)),
                time_end=self._iso_time(self._time_offset(day, 12)),
                claim_ids=incident_claims,
                archetype="related_storyline",
            )
            incidents.append(incident)

        # Generate Story B incidents (policy)
        for i in range(5):
            day = 305 + i * 3
            qk = self.rng.choice(["announcement", "legislation", "budget", "review"])
            scope_id = f"scope_policy_{qk}_{i}"

            anchor_entities = [policy.name, ministry.name]

            incident_claims = []
            for j in range(10):
                claim = GeneratedClaim(
                    id=self._next_claim_id(),
                    text=f"{policy.name}: {ministry.name} {qk} - report {j+1}",
                    publisher=self._random_publisher(),
                    reported_time=self._iso_time(self._time_offset(day, j)),
                    event_time=self._iso_time(self._time_offset(day, 0)),
                    question_key=qk,
                    anchor_entities=anchor_entities,
                    scope_id=scope_id,
                    archetype="related_storyline",
                )
                claims.append(claim)
                incident_claims.append(claim.id)

            incident = GeneratedIncident(
                id=self._next_incident_id(),
                description=f"{policy.name} {qk}",
                anchor_entities=anchor_entities,
                companion_entities=[],
                time_start=self._iso_time(self._time_offset(day, 0)),
                time_end=self._iso_time(self._time_offset(day, 12)),
                claim_ids=incident_claims,
                archetype="related_storyline",
            )
            incidents.append(incident)

        return entities, claims, incidents

    # =========================================================================
    # MAIN GENERATION
    # =========================================================================

    def generate(self) -> GeneratedCorpus:
        """Generate the complete macro corpus."""
        all_entities = []
        all_claims = []
        all_incidents = []
        archetype_stats = {}

        # Generate each archetype
        generators = [
            ("star_story", self._generate_star_story),
            ("dyad_story", self._generate_dyad_story),
            ("hub_adversary", self._generate_hub_adversary),
            ("homonym_adversary", self._generate_homonym_adversary),
            ("scope_pollution", self._generate_scope_pollution),
            ("time_missingness", self._generate_time_missingness),
            ("typed_conflicts", self._generate_typed_conflicts),
            ("related_storyline", self._generate_related_storyline),
        ]

        for archetype_name, generator in generators:
            entities, claims, incidents = generator()
            all_entities.extend(entities)
            all_claims.extend(claims)
            all_incidents.extend(incidents)

            archetype_stats[archetype_name] = {
                "entities": len(entities),
                "claims": len(claims),
                "incidents": len(incidents),
            }

        # Create manifest
        manifest = CorpusManifest(
            corpus_id=f"macro_corpus_seed{self.seed}",
            seed=self.seed,
            generated_at=datetime.utcnow().isoformat() + "Z",
            total_claims=len(all_claims),
            total_incidents=len(all_incidents),
            total_entities=len(all_entities),
            archetypes=archetype_stats,
            expected_invariants=[
                "no_semantic_only_core",
                "no_hub_story_definition",
                "scoped_surface_isolation",
                "no_chain_percolation",
            ],
            quantitative_bounds={
                "stories_range": [40, 80],
                "periphery_rate": [0.05, 0.25],
                "witness_scarcity": 0.40,
                "max_case_size": 50,
            },
        )

        return GeneratedCorpus(
            manifest=manifest,
            entities=all_entities,
            claims=all_claims,
            incidents=all_incidents,
        )

    def save_corpus(self, corpus: GeneratedCorpus, path: str):
        """Save corpus to JSON file."""
        with open(path, 'w') as f:
            json.dump(corpus.to_dict(), f, indent=2)

    def load_corpus(self, path: str) -> GeneratedCorpus:
        """Load corpus from JSON file."""
        with open(path) as f:
            data = json.load(f)

        manifest = CorpusManifest(**data['manifest'])
        entities = [GeneratedEntity(**e) for e in data['entities']]
        claims = [GeneratedClaim(**c) for c in data['claims']]
        incidents = [GeneratedIncident(**i) for i in data['incidents']]

        return GeneratedCorpus(
            manifest=manifest,
            entities=entities,
            claims=claims,
            incidents=incidents,
        )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    output_path = sys.argv[2] if len(sys.argv) > 2 else "corpus.json"

    print(f"Generating macro corpus with seed={seed}...")
    generator = MacroCorpusGenerator(seed=seed)
    corpus = generator.generate()

    print(f"\nGenerated corpus:")
    print(f"  Total claims: {corpus.manifest.total_claims}")
    print(f"  Total incidents: {corpus.manifest.total_incidents}")
    print(f"  Total entities: {corpus.manifest.total_entities}")
    print(f"\nArchetypes:")
    for name, stats in corpus.manifest.archetypes.items():
        print(f"  {name}: {stats['claims']} claims, {stats['incidents']} incidents")

    generator.save_corpus(corpus, output_path)
    print(f"\nSaved to {output_path}")
