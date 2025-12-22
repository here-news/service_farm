"""
JIMMY LAI CASE - Refined Epistemic Simulation

Enhanced with:
1. Better contradiction detection (HK Gov vs NGO perspective)
2. Claim attribution in narrative
3. More granular quest generation
4. Epistemic tension tracking
"""
import sys
sys.path.insert(0, '/app/backend')
sys.path.insert(0, '/app/test_eu')

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Set, Optional, Tuple
import re
from load_graph import load_snapshot


# =============================================================================
# EPISTEMIC POLES - The key perspectives in this case
# =============================================================================

EPISTEMIC_POLES = {
    'pro_prosecution': {
        'signals': ['fair trial', 'national security', 'rule of law', 'legitimate prosecution',
                   'foreign interference', 'collusion', 'sedition', 'guilty'],
        'sources': ['hk_gov', 'china_state'],
        'description': 'HK/China government position: legitimate prosecution under NSL'
    },
    'pro_lai': {
        'signals': ['political persecution', 'press freedom', 'arbitrary detention',
                   'unfair trial', 'prisoner of conscience', 'politically motivated',
                   'human rights', 'independent journalism'],
        'sources': ['ngo', 'uk_gov', 'us_gov', 'international'],
        'description': 'International/NGO position: political persecution of journalist'
    }
}


# =============================================================================
# SOURCE CLASSIFICATION (Enhanced)
# =============================================================================

SOURCE_CONFIG = {
    # Wire services
    'reuters.com': ('Reuters', 'wire', 0.88),
    'apnews.com': ('AP', 'wire', 0.88),
    'afp.com': ('AFP', 'wire', 0.85),

    # International
    'bbc.com': ('BBC', 'international', 0.90),
    'theguardian.com': ('Guardian', 'international', 0.85),
    'dw.com': ('DW', 'international', 0.85),
    'cnn.com': ('CNN', 'international', 0.78),
    'theatlantic.com': ('Atlantic', 'international', 0.82),
    'independent.co.uk': ('Independent', 'international', 0.78),
    'telegraph.co.uk': ('Telegraph', 'international', 0.75),
    'thetimes.co.uk': ('Times', 'international', 0.80),
    'politico.com': ('Politico', 'international', 0.78),
    'cnbc.com': ('CNBC', 'international', 0.75),
    'newsweek.com': ('Newsweek', 'international', 0.70),

    # HK/Asia
    'scmp.com': ('SCMP', 'local', 0.80),
    'hongkongfp.com': ('HKFP', 'local', 0.75),
    'rthk.hk': ('RTHK', 'local', 0.78),
    'thestandard.com.hk': ('The Standard', 'local', 0.70),

    # NGOs
    'rsf.org': ('RSF', 'ngo', 0.80),
    'amnesty.org': ('Amnesty', 'ngo', 0.80),
    'hrw.org': ('HRW', 'ngo', 0.80),

    # Government
    'gov.hk': ('HK Government', 'hk_gov', 0.65),
    'gov.uk': ('UK Government', 'uk_gov', 0.85),
    'state.gov': ('US State Dept', 'us_gov', 0.82),

    # Religious
    'christianitytoday.com': ('Christianity Today', 'religious', 0.70),
    'ncregister.com': ('NC Register', 'religious', 0.65),
}

def classify_source(url: str) -> Tuple[str, str, float]:
    url_lower = url.lower()
    for domain, (name, stype, cred) in SOURCE_CONFIG.items():
        if domain in url_lower:
            return name, stype, cred
    # Fallback
    try:
        domain = url.split('/')[2].replace('www.', '')
        name = domain.split('.')[0].upper()
        return name, 'unknown', 0.50
    except:
        return 'Unknown', 'unknown', 0.50


# =============================================================================
# EPISTEMIC ENGINE
# =============================================================================

@dataclass
class EpistemicClaim:
    id: str
    text: str
    source_name: str
    source_type: str
    source_url: str
    pole: Optional[str] = None  # 'pro_prosecution', 'pro_lai', or None
    pole_confidence: float = 0.0

@dataclass
class EpistemicTension:
    """A detected tension between poles"""
    claim_a_id: str
    claim_b_id: str
    pole_a: str
    pole_b: str
    description: str

@dataclass
class Quest:
    type: str
    title: str
    description: str
    ask_text: str
    bounty: int
    pole: Optional[str] = None  # Which pole needs more evidence

@dataclass
class NarrativeState:
    claims: List[EpistemicClaim] = field(default_factory=list)
    mass: float = 0.0

    # Pole tracking
    pole_mass: Dict[str, float] = field(default_factory=lambda: {'pro_prosecution': 0, 'pro_lai': 0, 'neutral': 0})
    pole_claims: Dict[str, List[str]] = field(default_factory=lambda: {'pro_prosecution': [], 'pro_lai': [], 'neutral': []})

    # Tensions
    tensions: List[EpistemicTension] = field(default_factory=list)

    # Quests
    quests: List[Quest] = field(default_factory=list)


class EpistemicEngine:
    def __init__(self):
        self.state = NarrativeState()

    def classify_pole(self, text: str, source_type: str) -> Tuple[Optional[str], float]:
        """Classify which epistemic pole a claim aligns with"""
        text_lower = text.lower()

        pro_prosecution_score = 0
        pro_lai_score = 0

        for signal in EPISTEMIC_POLES['pro_prosecution']['signals']:
            if signal in text_lower:
                pro_prosecution_score += 1

        for signal in EPISTEMIC_POLES['pro_lai']['signals']:
            if signal in text_lower:
                pro_lai_score += 1

        # Also consider source type
        if source_type in ['hk_gov', 'china_state']:
            pro_prosecution_score += 2
        elif source_type in ['ngo', 'uk_gov', 'us_gov']:
            pro_lai_score += 1

        if pro_prosecution_score > pro_lai_score and pro_prosecution_score >= 2:
            return 'pro_prosecution', pro_prosecution_score / (pro_prosecution_score + pro_lai_score + 1)
        elif pro_lai_score > pro_prosecution_score and pro_lai_score >= 2:
            return 'pro_lai', pro_lai_score / (pro_prosecution_score + pro_lai_score + 1)
        else:
            return None, 0.0

    def process_claim(self, claim: EpistemicClaim) -> Dict:
        """Process a claim and return analysis"""
        # Classify pole
        pole, confidence = self.classify_pole(claim.text, claim.source_type)
        claim.pole = pole
        claim.pole_confidence = confidence

        # Add to state
        self.state.claims.append(claim)

        # Get credibility
        cred = 0.5
        for domain, (name, _, c) in SOURCE_CONFIG.items():
            if name == claim.source_name:
                cred = c
                break

        self.state.mass += cred

        # Update pole tracking
        if pole:
            self.state.pole_mass[pole] += cred
            self.state.pole_claims[pole].append(claim.id)
        else:
            self.state.pole_mass['neutral'] += cred
            self.state.pole_claims['neutral'].append(claim.id)

        # Check for tensions
        new_tensions = self._detect_tensions(claim)

        return {
            'pole': pole,
            'confidence': confidence,
            'tensions_found': len(new_tensions)
        }

    def _detect_tensions(self, new_claim: EpistemicClaim) -> List[EpistemicTension]:
        """Detect tensions between this claim and existing claims from opposite pole"""
        tensions = []

        if not new_claim.pole:
            return tensions

        opposite_pole = 'pro_lai' if new_claim.pole == 'pro_prosecution' else 'pro_prosecution'

        for existing in self.state.claims[:-1]:
            if existing.pole == opposite_pole:
                tension = EpistemicTension(
                    claim_a_id=new_claim.id,
                    claim_b_id=existing.id,
                    pole_a=new_claim.pole,
                    pole_b=existing.pole,
                    description=f"{new_claim.source_name} ({new_claim.pole}) vs {existing.source_name} ({existing.pole})"
                )
                tensions.append(tension)
                self.state.tensions.append(tension)

        return tensions

    def detect_quests(self) -> List[Quest]:
        """Detect quests based on epistemic gaps"""
        quests = []

        # Calculate pole imbalance
        pro_p = self.state.pole_mass['pro_prosecution']
        pro_l = self.state.pole_mass['pro_lai']

        # If heavily imbalanced, ask for other side
        if pro_l > pro_p * 3 and pro_p < 10:
            quests.append(Quest(
                type='balance',
                title='HK/China Government perspective needed',
                description=f'Current narrative heavily weighted toward pro-Lai sources (mass ratio: {pro_l:.0f}:{pro_p:.0f})',
                ask_text='Official HK government or Chinese state media statement on Lai prosecution?',
                bounty=20,
                pole='pro_prosecution'
            ))
        elif pro_p > pro_l * 3 and pro_l < 10:
            quests.append(Quest(
                type='balance',
                title='Human rights perspective needed',
                description=f'Current narrative heavily weighted toward prosecution sources (mass ratio: {pro_p:.0f}:{pro_l:.0f})',
                ask_text='NGO or international assessment of trial fairness?',
                bounty=20,
                pole='pro_lai'
            ))

        # Check for missing source types
        source_types = {c.source_type for c in self.state.claims}

        if 'hk_gov' not in source_types:
            quests.append(Quest(
                type='missing_source',
                title='Direct HK Government statement',
                description='No direct gov.hk source in narrative',
                ask_text='Link to official HK government statement on Jimmy Lai?',
                bounty=15,
                pole='pro_prosecution'
            ))

        if 'ngo' not in source_types:
            quests.append(Quest(
                type='missing_source',
                title='Human rights organization analysis',
                description='No Amnesty/HRW/RSF assessment',
                ask_text='Link to human rights org analysis of Lai case?',
                bounty=12,
                pole='pro_lai'
            ))

        if 'uk_gov' not in source_types:
            quests.append(Quest(
                type='missing_source',
                title='UK Government position (Lai is UK citizen)',
                description='No official UK government statement',
                ask_text='Official UK gov statement on British citizen Jimmy Lai?',
                bounty=12,
                pole='pro_lai'
            ))

        # If we have tensions, ask for resolution evidence
        if len(self.state.tensions) > 5:
            quests.append(Quest(
                type='resolution',
                title='Evidence for trial fairness assessment',
                description=f'{len(self.state.tensions)} epistemic tensions detected between sources',
                ask_text='Court documents, UN reports, or independent legal analysis?',
                bounty=25,
                pole=None
            ))

        self.state.quests = quests
        return quests

    def generate_narrative(self) -> str:
        """Generate multi-perspective narrative"""
        lines = []

        lines.append("=" * 70)
        lines.append("JIMMY LAI CASE - Epistemic Narrative")
        lines.append("=" * 70)

        # Summary stats
        lines.append(f"\nEPISTEMIC BALANCE:")
        lines.append(f"  Pro-prosecution mass: {self.state.pole_mass['pro_prosecution']:.1f}")
        lines.append(f"  Pro-Lai mass:         {self.state.pole_mass['pro_lai']:.1f}")
        lines.append(f"  Neutral:              {self.state.pole_mass['neutral']:.1f}")
        lines.append(f"  Tensions detected:    {len(self.state.tensions)}")

        # Group claims by pole
        prosecution_claims = [c for c in self.state.claims if c.pole == 'pro_prosecution']
        lai_claims = [c for c in self.state.claims if c.pole == 'pro_lai']
        neutral_claims = [c for c in self.state.claims if c.pole is None]

        # Prosecution perspective
        lines.append(f"\n{'─' * 70}")
        lines.append("🏛️ HONG KONG/CHINA GOVERNMENT POSITION")
        lines.append(f"{'─' * 70}")
        if prosecution_claims:
            for c in prosecution_claims[:5]:
                lines.append(f"\n  [{c.source_name}] \"{c.text[:100]}...\"")
        else:
            lines.append("\n  ❓ No direct government sources yet")
            lines.append("  → QUEST: Official gov.hk statement needed (+15 bounty)")

        # Pro-Lai perspective
        lines.append(f"\n{'─' * 70}")
        lines.append("📝 PRESS FREEDOM / HUMAN RIGHTS POSITION")
        lines.append(f"{'─' * 70}")
        if lai_claims:
            for c in lai_claims[:5]:
                lines.append(f"\n  [{c.source_name}] \"{c.text[:100]}...\"")
        else:
            lines.append("\n  ❓ No NGO/international sources yet")
            lines.append("  → QUEST: Human rights analysis needed (+12 bounty)")

        # Key tensions
        if self.state.tensions:
            lines.append(f"\n{'─' * 70}")
            lines.append("⚡ EPISTEMIC TENSIONS (Key Disagreements)")
            lines.append(f"{'─' * 70}")

            # Show unique tensions
            seen = set()
            for t in self.state.tensions[:5]:
                key = (t.pole_a, t.pole_b)
                if key not in seen:
                    seen.add(key)
                    claim_a = next((c for c in self.state.claims if c.id == t.claim_a_id), None)
                    claim_b = next((c for c in self.state.claims if c.id == t.claim_b_id), None)
                    if claim_a and claim_b:
                        lines.append(f"\n  TENSION: {claim_a.source_name} vs {claim_b.source_name}")
                        lines.append(f"    A: \"{claim_a.text[:70]}...\"")
                        lines.append(f"    B: \"{claim_b.text[:70]}...\"")

        # International response
        intl_claims = [c for c in self.state.claims if 'uk' in c.text.lower() or 'starmer' in c.text.lower()
                       or 'trump' in c.text.lower() or 'us' in c.text.lower()]
        if intl_claims:
            lines.append(f"\n{'─' * 70}")
            lines.append("🌍 INTERNATIONAL RESPONSE")
            lines.append(f"{'─' * 70}")
            for c in intl_claims[:4]:
                lines.append(f"\n  [{c.source_name}] \"{c.text[:100]}...\"")

        return '\n'.join(lines)


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def run_simulation():
    print("=" * 80)
    print("JIMMY LAI - Refined Epistemic Simulation")
    print("=" * 80)

    snapshot = load_snapshot()

    # Filter claims
    lai_claims = []
    for c in snapshot.claims.values():
        text = c.text.lower()
        if 'lai' in text or 'apple daily' in text:
            page = snapshot.pages.get(c.page_id)
            if page:
                source_name, source_type, _ = classify_source(page.url)
                lai_claims.append(EpistemicClaim(
                    id=c.id,
                    text=c.text,
                    source_name=source_name,
                    source_type=source_type,
                    source_url=page.url
                ))

    print(f"\nFound {len(lai_claims)} Jimmy Lai claims")

    engine = EpistemicEngine()

    # Process all claims
    print("\n" + "=" * 80)
    print("PROCESSING CLAIMS...")
    print("=" * 80)

    for i, claim in enumerate(lai_claims):
        result = engine.process_claim(claim)

        # Print significant claims
        if result['pole'] and result['confidence'] > 0.3:
            print(f"\n[{i+1:3}] {claim.source_name:15} → {result['pole']:15} (conf={result['confidence']:.2f})")
            print(f"      \"{claim.text[:65]}...\"")
            if result['tensions_found'] > 0:
                print(f"      ⚡ Created {result['tensions_found']} tension(s) with opposite pole")

    # Detect quests
    quests = engine.detect_quests()

    # Print final narrative
    print("\n" + engine.generate_narrative())

    # Print quests
    print("\n" + "=" * 70)
    print("🎯 OPEN QUESTS (Contribution Opportunities)")
    print("=" * 70)

    for q in quests:
        pole_label = f" [{q.pole}]" if q.pole else ""
        print(f"""
┌{'─' * 66}┐
│ {q.type.upper()}{pole_label:50} │
│ {q.title:64} │
├{'─' * 66}┤
│ {q.description[:64]:64} │
│                                                                  │
│ → "{q.ask_text[:60]:60}" │
│                                                    +{q.bounty:2} bounty │
└{'─' * 66}┘""")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    Total claims:          {len(engine.state.claims)}
    Total mass:            {engine.state.mass:.1f}

    Pro-prosecution:       {len(engine.state.pole_claims['pro_prosecution']):3} claims, mass={engine.state.pole_mass['pro_prosecution']:.1f}
    Pro-Lai:               {len(engine.state.pole_claims['pro_lai']):3} claims, mass={engine.state.pole_mass['pro_lai']:.1f}
    Neutral:               {len(engine.state.pole_claims['neutral']):3} claims, mass={engine.state.pole_mass['neutral']:.1f}

    Epistemic tensions:    {len(engine.state.tensions)}
    Open quests:           {len(quests)}

    BALANCE ASSESSMENT:
    """)

    pro_p = engine.state.pole_mass['pro_prosecution']
    pro_l = engine.state.pole_mass['pro_lai']

    if pro_l > pro_p * 2:
        print("    ⚠️  IMBALANCED: Narrative heavily favors pro-Lai perspective")
        print("        Need more government/prosecution sources for balance")
    elif pro_p > pro_l * 2:
        print("    ⚠️  IMBALANCED: Narrative heavily favors prosecution perspective")
        print("        Need more NGO/international sources for balance")
    else:
        print("    ✓  BALANCED: Multiple perspectives represented")


if __name__ == "__main__":
    run_simulation()
