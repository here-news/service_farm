"""
JIMMY LAI CASE - Streaming Epistemic Simulation

Watch the narrative emerge claim by claim:
1. Claims stream in with full provenance
2. Event state updates (mass, heat, coherence)
3. Narrative evolves at milestones
4. Gaps detected → Quests emitted

This demonstrates UEE's epistemic narrative capability.
"""
import sys
sys.path.insert(0, '/app/backend')
sys.path.insert(0, '/app/test_eu')

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Set, Optional, Tuple
import re
import numpy as np
from load_graph import load_snapshot


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Voice:
    """A source that speaks in the narrative"""
    name: str
    type: str  # 'wire', 'official', 'international', 'local', 'hk_gov', 'uk_gov', 'us_gov', 'ngo'
    credibility: float
    claims: List[str] = field(default_factory=list)

@dataclass
class StreamingClaim:
    """A claim with full provenance"""
    id: str
    text: str
    source_name: str
    source_type: str
    source_url: str
    timestamp: Optional[datetime]
    entities: List[str]
    embedding: Optional[List[float]] = None

@dataclass
class Quest:
    """A gap that needs filling"""
    type: str  # 'missing_source', 'needs_verification', 'contested', 'missing_perspective'
    title: str
    description: str
    ask_text: str
    bounty: int
    related_claims: List[str] = field(default_factory=list)

@dataclass
class NarrativeState:
    """Current state of the epistemic narrative"""
    claims: List[StreamingClaim] = field(default_factory=list)
    voices: Dict[str, Voice] = field(default_factory=dict)

    # Jaynesian quantities
    mass: float = 0.0
    heat: float = 0.0
    coherence: float = 1.0

    # Tracking
    corroborations: List[Tuple[str, str]] = field(default_factory=list)
    contradictions: List[Tuple[str, str, str]] = field(default_factory=list)  # (claim1, claim2, note)

    # Gaps
    quests: List[Quest] = field(default_factory=list)

    # Milestones
    milestones: List[dict] = field(default_factory=list)


# =============================================================================
# SOURCE CLASSIFICATION
# =============================================================================

SOURCE_CONFIG = {
    # Wire services
    'reuters.com': ('Reuters', 'wire', 0.88),
    'apnews.com': ('AP', 'wire', 0.88),
    'afp.com': ('AFP', 'wire', 0.85),

    # International media
    'bbc.com': ('BBC', 'international', 0.90),
    'theguardian.com': ('Guardian', 'international', 0.85),
    'nytimes.com': ('NYT', 'international', 0.85),
    'washingtonpost.com': ('WaPo', 'international', 0.82),
    'dw.com': ('DW', 'international', 0.85),
    'aljazeera.com': ('Al Jazeera', 'international', 0.80),
    'cnn.com': ('CNN', 'international', 0.78),
    'theatlantic.com': ('Atlantic', 'international', 0.82),

    # HK/China media
    'scmp.com': ('SCMP', 'local', 0.80),
    'hongkongfp.com': ('HKFP', 'local', 0.75),
    'rthk.hk': ('RTHK', 'local', 0.78),

    # Government sources
    'gov.hk': ('HK Government', 'hk_gov', 0.70),  # official but potentially biased
    'gov.uk': ('UK Government', 'uk_gov', 0.85),
    'state.gov': ('US State Dept', 'us_gov', 0.82),

    # NGOs
    'amnesty.org': ('Amnesty', 'ngo', 0.80),
    'hrw.org': ('HRW', 'ngo', 0.80),
    'rsf.org': ('RSF', 'ngo', 0.80),

    # Other
    'christianitytoday.com': ('Christianity Today', 'religious', 0.70),
    'foxnews.com': ('Fox News', 'partisan_us', 0.60),
    'nypost.com': ('NY Post', 'tabloid', 0.55),
}

def classify_source(url: str) -> Tuple[str, str, float]:
    """Classify source from URL"""
    url_lower = url.lower()
    for domain, (name, stype, cred) in SOURCE_CONFIG.items():
        if domain in url_lower:
            return name, stype, cred

    # Extract domain as fallback
    try:
        domain = url.split('/')[2].replace('www.', '')
        name = domain.split('.')[0].upper()
        return name, 'unknown', 0.50
    except:
        return 'Unknown', 'unknown', 0.50


# =============================================================================
# STREAMING ENGINE
# =============================================================================

class JimmyLaiStreamingEngine:
    """Stream claims and build epistemic narrative"""

    def __init__(self):
        self.state = NarrativeState()
        self.claim_count = 0
        self.last_coherence = 1.0

    def stream_claim(self, claim: StreamingClaim) -> Dict:
        """Process one claim and return state update"""
        self.claim_count += 1

        # Add to state
        self.state.claims.append(claim)

        # Update voice registry
        if claim.source_name not in self.state.voices:
            self.state.voices[claim.source_name] = Voice(
                name=claim.source_name,
                type=claim.source_type,
                credibility=self._get_credibility(claim.source_name),
                claims=[]
            )
        self.state.voices[claim.source_name].claims.append(claim.id)

        # Update Jaynesian quantities
        cred = self._get_credibility(claim.source_name)
        self.state.mass += cred
        self.state.heat += 1.0  # Simplified - real system uses recency decay

        # Check for corroboration/contradiction
        relations = self._check_relations(claim)

        # Update coherence
        self._update_coherence()

        # Check for milestone (coherence leap)
        milestone = self._check_milestone()

        # Detect gaps
        new_quests = self._detect_gaps()

        return {
            'claim_num': self.claim_count,
            'claim': claim,
            'mass': self.state.mass,
            'heat': self.state.heat,
            'coherence': self.state.coherence,
            'voices': len(self.state.voices),
            'relations': relations,
            'milestone': milestone,
            'new_quests': new_quests
        }

    def _get_credibility(self, source_name: str) -> float:
        """Get source credibility"""
        for domain, (name, _, cred) in SOURCE_CONFIG.items():
            if name == source_name:
                return cred
        return 0.50

    def _check_relations(self, new_claim: StreamingClaim) -> Dict:
        """Check relations between new claim and existing claims"""
        relations = {'corroborates': [], 'contradicts': [], 'updates': []}

        new_text = new_claim.text.lower()

        for existing in self.state.claims[:-1]:  # Exclude the new claim
            existing_text = existing.text.lower()

            # Simple word overlap for corroboration
            new_words = set(new_text.split())
            existing_words = set(existing_text.split())
            overlap = len(new_words & existing_words) / max(len(new_words | existing_words), 1)

            if overlap > 0.4:
                # Check for contradiction indicators
                if self._is_contradiction(new_text, existing_text):
                    relations['contradicts'].append(existing.id)
                    self.state.contradictions.append((new_claim.id, existing.id, 'semantic'))
                else:
                    relations['corroborates'].append(existing.id)
                    self.state.corroborations.append((new_claim.id, existing.id))

        return relations

    def _is_contradiction(self, text1: str, text2: str) -> bool:
        """Simple contradiction detection"""
        # Look for opposing sentiment about key topics
        contradiction_pairs = [
            ('fair trial', 'unfair'),
            ('independent judiciary', 'political prosecution'),
            ('national security', 'press freedom'),
            ('legitimate prosecution', 'persecution'),
            ('guilty', 'innocent'),
        ]

        for pos, neg in contradiction_pairs:
            if (pos in text1 and neg in text2) or (neg in text1 and pos in text2):
                return True
        return False

    def _update_coherence(self):
        """Update coherence score"""
        total_relations = len(self.state.corroborations) + len(self.state.contradictions)
        if total_relations > 0:
            self.state.coherence = len(self.state.corroborations) / total_relations
        else:
            self.state.coherence = 1.0

    def _check_milestone(self) -> Optional[Dict]:
        """Check if we hit a milestone (coherence leap or significant event)"""
        milestone = None

        # Check for significant claim counts
        if self.claim_count in [5, 15, 30, 50, 75, 100]:
            milestone = {
                'type': 'claim_count',
                'count': self.claim_count,
                'mass': self.state.mass,
                'coherence': self.state.coherence,
                'voices': len(self.state.voices)
            }
            self.state.milestones.append(milestone)

        # Check for coherence change
        coherence_delta = abs(self.state.coherence - self.last_coherence)
        if coherence_delta > 0.1:
            milestone = {
                'type': 'coherence_shift',
                'old': self.last_coherence,
                'new': self.state.coherence,
                'direction': 'up' if self.state.coherence > self.last_coherence else 'down'
            }
            self.state.milestones.append(milestone)

        self.last_coherence = self.state.coherence
        return milestone

    def _detect_gaps(self) -> List[Quest]:
        """Detect epistemic gaps and generate quests"""
        quests = []

        # Only check at certain intervals
        if self.claim_count % 10 != 0:
            return quests

        # Check for missing source types
        source_types = {v.type for v in self.state.voices.values()}

        # Need official HK government response?
        if 'hk_gov' not in source_types and self.claim_count >= 10:
            quests.append(Quest(
                type='missing_source',
                title='Official HK Government statement needed',
                description='No direct statement from Hong Kong government',
                ask_text='Anyone have official gov.hk statement on Jimmy Lai case?',
                bounty=15
            ))

        # Need UK government response?
        if 'uk_gov' not in source_types and self.claim_count >= 15:
            quests.append(Quest(
                type='missing_source',
                title='UK Government position needed',
                description='Jimmy Lai is UK citizen - what is official UK response?',
                ask_text='Official UK government statement on Lai case?',
                bounty=12
            ))

        # Need human rights org perspective?
        if 'ngo' not in source_types and self.claim_count >= 20:
            quests.append(Quest(
                type='missing_perspective',
                title='Human rights assessment needed',
                description='No Amnesty/HRW/RSF analysis in narrative',
                ask_text='Human rights organization analysis of Lai trial?',
                bounty=10
            ))

        # If we have contradictions, ask for resolution
        if len(self.state.contradictions) > 2:
            quests.append(Quest(
                type='contested',
                title='Resolve: Fair trial or political persecution?',
                description=f'{len(self.state.contradictions)} contradicting claims detected',
                ask_text='Evidence for trial fairness or unfairness?',
                bounty=20,
                related_claims=[c[0] for c in self.state.contradictions[-3:]]
            ))

        # Add new quests (avoid duplicates)
        existing_titles = {q.title for q in self.state.quests}
        for q in quests:
            if q.title not in existing_titles:
                self.state.quests.append(q)

        return [q for q in quests if q.title not in existing_titles]

    def generate_narrative(self) -> str:
        """Generate current epistemic narrative"""
        if not self.state.claims:
            return "No claims yet."

        # Group claims by theme
        themes = defaultdict(list)
        for claim in self.state.claims:
            text = claim.text.lower()
            if 'sentence' in text or 'year' in text or 'prison' in text:
                themes['sentencing'].append(claim)
            elif 'trial' in text or 'court' in text:
                themes['trial'].append(claim)
            elif 'uk' in text or 'britain' in text or 'starmer' in text:
                themes['uk_response'].append(claim)
            elif 'us' in text or 'america' in text or 'trump' in text:
                themes['us_response'].append(claim)
            elif 'press freedom' in text or 'journalism' in text:
                themes['press_freedom'].append(claim)
            elif 'national security' in text:
                themes['national_security'].append(claim)
            else:
                themes['general'].append(claim)

        narrative = []

        # Sentencing
        if themes['sentencing']:
            narrative.append("**SENTENCING**")
            for claim in themes['sentencing'][:3]:
                narrative.append(f"  📰 {claim.source_name}: \"{claim.text[:100]}...\"")

        # Trial proceedings
        if themes['trial']:
            narrative.append("\n**TRIAL PROCEEDINGS**")
            for claim in themes['trial'][:3]:
                narrative.append(f"  ⚖️ {claim.source_name}: \"{claim.text[:100]}...\"")

        # International response
        if themes['uk_response'] or themes['us_response']:
            narrative.append("\n**INTERNATIONAL RESPONSE**")
            for claim in (themes['uk_response'] + themes['us_response'])[:4]:
                icon = '🇬🇧' if 'uk' in claim.text.lower() else '🇺🇸'
                narrative.append(f"  {icon} {claim.source_name}: \"{claim.text[:100]}...\"")

        # Press freedom angle
        if themes['press_freedom']:
            narrative.append("\n**PRESS FREEDOM DEBATE**")
            for claim in themes['press_freedom'][:2]:
                narrative.append(f"  📝 {claim.source_name}: \"{claim.text[:100]}...\"")

        # HK Government position (if any)
        hk_gov_claims = [c for c in self.state.claims if c.source_type == 'hk_gov']
        if hk_gov_claims:
            narrative.append("\n**HONG KONG GOVERNMENT POSITION**")
            for claim in hk_gov_claims[:2]:
                narrative.append(f"  🏛️ {claim.source_name}: \"{claim.text[:100]}...\"")
        else:
            narrative.append("\n**HONG KONG GOVERNMENT POSITION**")
            narrative.append("  ❓ [No direct official statement - contribute?]")

        return '\n'.join(narrative)


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def run_simulation():
    print("=" * 80)
    print("JIMMY LAI CASE - Streaming Epistemic Simulation")
    print("=" * 80)

    # Load data
    snapshot = load_snapshot()

    # Filter to Jimmy Lai claims
    lai_claims = []
    for c in snapshot.claims.values():
        text = c.text.lower()
        if 'lai' in text or 'apple daily' in text:
            page = snapshot.pages.get(c.page_id)
            if page:
                source_name, source_type, cred = classify_source(page.url)

                lai_claims.append(StreamingClaim(
                    id=c.id,
                    text=c.text,
                    source_name=source_name,
                    source_type=source_type,
                    source_url=page.url,
                    timestamp=getattr(page, 'pub_time', None),
                    entities=list(c.entity_ids) if c.entity_ids else [],
                    embedding=None
                ))

    print(f"\nFound {len(lai_claims)} Jimmy Lai claims")
    print(f"From {len(set(c.source_name for c in lai_claims))} distinct sources")

    # Initialize engine
    engine = JimmyLaiStreamingEngine()

    # Stream claims
    print("\n" + "=" * 80)
    print("STREAMING CLAIMS...")
    print("=" * 80)

    milestone_narratives = []

    for i, claim in enumerate(lai_claims):
        result = engine.stream_claim(claim)

        # Print every 10th claim
        if i % 10 == 0:
            print(f"\n[Claim {result['claim_num']:3}] {claim.source_name:15} | mass={result['mass']:.1f} | coherence={result['coherence']:.2f}")
            print(f"           \"{claim.text[:70]}...\"")

            if result['relations']['corroborates']:
                print(f"           ✓ Corroborates {len(result['relations']['corroborates'])} existing claims")
            if result['relations']['contradicts']:
                print(f"           ⚠️ Contradicts {len(result['relations']['contradicts'])} existing claims")

        # Print milestones
        if result['milestone']:
            m = result['milestone']
            print(f"\n{'='*60}")
            print(f"📍 MILESTONE at claim {result['claim_num']}")
            if m['type'] == 'claim_count':
                print(f"   Voices: {m['voices']} | Mass: {m['mass']:.1f} | Coherence: {m['coherence']:.2f}")
            elif m['type'] == 'coherence_shift':
                direction = "📈" if m['direction'] == 'up' else "📉"
                print(f"   {direction} Coherence shift: {m['old']:.2f} → {m['new']:.2f}")

            # Save narrative at milestone
            narrative = engine.generate_narrative()
            milestone_narratives.append({
                'claim_num': result['claim_num'],
                'narrative': narrative,
                'quests': len(engine.state.quests)
            })
            print(f"{'='*60}")

        # Print new quests
        if result['new_quests']:
            print(f"\n🎯 NEW QUESTS DETECTED:")
            for q in result['new_quests']:
                print(f"   [{q.type}] {q.title} (+{q.bounty} bounty)")
                print(f"   → \"{q.ask_text}\"")

    # ==========================================================================
    # FINAL STATE
    # ==========================================================================
    print("\n" + "=" * 80)
    print("FINAL STATE")
    print("=" * 80)

    print(f"""
    Claims processed: {len(engine.state.claims)}
    Distinct voices:  {len(engine.state.voices)}
    Mass:             {engine.state.mass:.1f}
    Coherence:        {engine.state.coherence:.2f}
    Corroborations:   {len(engine.state.corroborations)}
    Contradictions:   {len(engine.state.contradictions)}
    Open quests:      {len(engine.state.quests)}
    """)

    # Voice breakdown
    print("VOICE BREAKDOWN:")
    for name, voice in sorted(engine.state.voices.items(), key=lambda x: -len(x[1].claims)):
        print(f"  {name:20} [{voice.type:12}] - {len(voice.claims):3} claims (cred={voice.credibility:.2f})")

    # ==========================================================================
    # OPEN QUESTS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("OPEN QUESTS (Contribution Opportunities)")
    print("=" * 80)

    for q in engine.state.quests:
        print(f"""
    ┌{'─'*70}┐
    │ {q.type.upper():68} │
    │ {q.title:68} │
    │ {'-'*68} │
    │ {q.description[:68]:68} │
    │                                                                      │
    │ → "{q.ask_text[:64]}"    │
    │                                                        +{q.bounty:2} bounty │
    └{'─'*70}┘""")

    # ==========================================================================
    # FINAL NARRATIVE
    # ==========================================================================
    print("\n" + "=" * 80)
    print("FINAL EPISTEMIC NARRATIVE")
    print("=" * 80)

    print(engine.generate_narrative())

    # ==========================================================================
    # NARRATIVE EVOLUTION
    # ==========================================================================
    print("\n" + "=" * 80)
    print("NARRATIVE EVOLUTION AT MILESTONES")
    print("=" * 80)

    for mn in milestone_narratives[:3]:  # Show first 3 milestones
        print(f"\n--- At claim {mn['claim_num']} (quests: {mn['quests']}) ---")
        # Print abbreviated narrative
        lines = mn['narrative'].split('\n')[:8]
        for line in lines:
            print(line)
        if len(mn['narrative'].split('\n')) > 8:
            print("  ...")


if __name__ == "__main__":
    run_simulation()
