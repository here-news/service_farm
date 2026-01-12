"""
Emergence Engine: Streamlined Orchestrator
===========================================

This is the new streamlined engine that delegates to specialized modules.
It replaces the monolithic epistemic_unit.py.

Architecture:
  L0 -> L2: Identity linking (identity/linker.py)
  L2 -> L3: Aboutness scoring (aboutness/scorer.py)
  Meta: Tension detection (meta/detectors.py)
  Interpretation: LLM synthesis (interpretation.py)

INVARIANTS enforced:
  1. L0 is append-only (claims never deleted)
  2. Parameters are versioned (all changes tracked)
  3. Identity/Aboutness separation (L2 uses identity, L3 uses aboutness)
  4. Meta-claims emitted for tension detection
"""

from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI

from reee.types import (
    Claim, Surface, Event, Parameters, MetaClaim, Relation
)
from reee.identity import IdentityLinker
from reee.aboutness import compute_aboutness_edges, compute_events_from_aboutness
from reee.meta import detect_tensions, get_unresolved, count_by_type, resolve_meta_claim
from reee.interpretation import interpret_all


class Engine:
    """
    Orchestrates emergence across levels using specialized modules.

    Clean separation of concerns:
    - IdentityLinker: L0 -> L2 (claim relationships, surface formation)
    - AboutnessScorer: L2 -> L3 (surface relationships, event clustering)
    - TensionDetector: Meta-claim generation
    - Interpretation: LLM synthesis
    """

    def __init__(self, llm: 'AsyncOpenAI' = None, params: Parameters = None):
        self.llm = llm
        self.params = params or Parameters()

        # Delegate identity linking to IdentityLinker
        self._linker = IdentityLinker(llm=llm, params=self.params)

        # L2 Surfaces and L3 Events (computed)
        self.surfaces: Dict[str, Surface] = {}
        self.events: Dict[str, Event] = {}

        # Aboutness edges (surface-to-surface)
        self.surface_aboutness: List[tuple] = []

        # Meta-claims
        self.meta_claims: List[MetaClaim] = []

    # =========================================================================
    # L0: Claims (delegated to IdentityLinker)
    # =========================================================================

    @property
    def claims(self) -> Dict[str, Claim]:
        """Access claims from the identity linker."""
        return self._linker.claims

    @property
    def claim_edges(self) -> List[tuple]:
        """Access identity edges from the linker."""
        return self._linker.edges

    @property
    def question_index(self) -> Dict[str, List[str]]:
        """Access question key index from the linker."""
        return self._linker.question_index

    async def add_claim(
        self,
        claim: Claim,
        extract_question_key: bool = True
    ) -> Dict:
        """
        Add claim and compute relationships to existing claims.

        Delegated to IdentityLinker.
        """
        return await self._linker.add_claim(claim, extract_question_key)

    # =========================================================================
    # L2: Surfaces
    # =========================================================================

    def compute_surfaces(self) -> List[Surface]:
        """
        Compute surfaces from identity edges (connected components).

        Delegated to IdentityLinker.
        """
        self.surfaces = self._linker.compute_surfaces()
        return list(self.surfaces.values())

    # =========================================================================
    # L3: Events
    # =========================================================================

    def compute_surface_aboutness(self) -> List[tuple]:
        """
        Compute soft aboutness edges between surfaces.

        Uses the aboutness module.
        """
        if not self.surfaces:
            self.compute_surfaces()

        self.surface_aboutness = compute_aboutness_edges(
            self.surfaces,
            self.params
        )
        return self.surface_aboutness

    def compute_events(self) -> List[Event]:
        """
        Cluster surfaces into events based on aboutness edges.

        Uses the aboutness module.
        """
        if not self.surface_aboutness:
            self.compute_surface_aboutness()

        self.events = compute_events_from_aboutness(
            self.surfaces,
            self.surface_aboutness,
            self.params
        )
        return list(self.events.values())

    # =========================================================================
    # Meta-Claims
    # =========================================================================

    def detect_tensions(self) -> List[MetaClaim]:
        """
        Detect tensions in the epistemic state.

        Uses the meta module.
        """
        new_meta_claims = detect_tensions(
            self.claims,
            self.surfaces,
            self.claim_edges,
            self.params
        )
        self.meta_claims.extend(new_meta_claims)
        return new_meta_claims

    def get_unresolved_meta_claims(self) -> List[MetaClaim]:
        """Return meta-claims that haven't been resolved."""
        return get_unresolved(self.meta_claims)

    def resolve_meta_claim(
        self,
        meta_claim_id: str,
        resolution: str,
        actor: str = "system"
    ) -> Optional[MetaClaim]:
        """Mark a meta-claim as resolved."""
        return resolve_meta_claim(
            self.meta_claims,
            meta_claim_id,
            resolution,
            actor
        )

    # =========================================================================
    # Interpretation
    # =========================================================================

    async def interpret_all(self) -> None:
        """Generate semantic interpretation for all surfaces and events."""
        if not self.llm:
            return

        await interpret_all(
            list(self.claims.values()),
            list(self.surfaces.values()),
            list(self.events.values()),
            self.llm
        )

    # =========================================================================
    # Output
    # =========================================================================

    def summary(self) -> Dict:
        """Summary of current state."""
        return {
            'claims': len(self.claims),
            'claim_edges': len(self.claim_edges),
            'surfaces': len(self.surfaces),
            'surface_aboutness_edges': len(self.surface_aboutness),
            'events': len(self.events),

            'question_index': {
                'buckets': len(self.question_index),
                'claims_with_key': sum(len(v) for v in self.question_index.values()),
                'largest_bucket': max((len(v) for v in self.question_index.values()), default=0),
                'keys': list(self.question_index.keys())[:10]
            },

            'params': {
                'version': self.params.version,
                'identity_confidence_threshold': self.params.identity_confidence_threshold,
                'hub_max_df': self.params.hub_max_df,
                'aboutness_min_signals': self.params.aboutness_min_signals,
                'aboutness_threshold': self.params.aboutness_threshold,
                'high_entropy_threshold': self.params.high_entropy_threshold,
                'changes': len(self.params.changes)
            },

            'meta_claims': {
                'total': len(self.meta_claims),
                'unresolved': len(self.get_unresolved_meta_claims()),
                'by_type': count_by_type(self.meta_claims)
            },

            'surfaces_detail': [
                {
                    'id': s.id,
                    'title': s.canonical_title,
                    'claims': len(s.claim_ids),
                    'sources': len(s.sources),
                    'entropy': round(s.entropy, 3),
                    'about_links': len(s.about_links)
                }
                for s in self.surfaces.values()
            ],
            'events_detail': [
                {
                    'id': e.id,
                    'title': e.canonical_title,
                    'surfaces': len(e.surface_ids),
                    'total_claims': e.total_claims,
                    'sources': e.total_sources
                }
                for e in self.events.values()
            ]
        }


# Backward compatibility alias
EmergenceEngine = Engine
