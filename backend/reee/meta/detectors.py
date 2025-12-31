"""
Tension Detection for Meta-Claims
==================================

Meta-claims are observations ABOUT the epistemic state, not truth claims.
They may trigger:
- ParameterChange (adjust thresholds)
- New L0 claims (verification, corroboration)
- Task generation (bounties, investigations)

INVARIANT 6: Meta-claims are never injected as world-claims.
"""

from typing import Dict, List, Tuple, Optional

from ..types import (
    Claim, Surface, Relation, Parameters, MetaClaim
)


class TensionDetector:
    """
    Detects tensions in the epistemic state and emits meta-claims.
    """

    def __init__(
        self,
        claims: Dict[str, Claim],
        surfaces: Dict[str, Surface],
        edges: List[Tuple[str, str, Relation, float]],
        params: Parameters = None
    ):
        self.claims = claims
        self.surfaces = surfaces
        self.edges = edges
        self.params = params or Parameters()

    def detect_all(self) -> List[MetaClaim]:
        """
        Detect all tensions and return meta-claims.

        Tension types:
        - high_entropy_surface: Surface has high semantic dispersion
        - single_source_only: Claim has only one source
        - unresolved_conflict: CONFLICTS edge without resolution
        """
        meta_claims = []

        # Detect high-entropy surfaces
        meta_claims.extend(self._detect_high_entropy())

        # Detect single-source claims
        meta_claims.extend(self._detect_single_source())

        # Detect unresolved conflicts
        meta_claims.extend(self._detect_conflicts())

        return meta_claims

    def _detect_high_entropy(self) -> List[MetaClaim]:
        """Detect surfaces with high semantic dispersion."""
        meta_claims = []
        threshold = self.params.high_entropy_threshold

        for surface in self.surfaces.values():
            if surface.entropy > threshold:
                mc = MetaClaim(
                    type="high_entropy_surface",
                    target_id=surface.id,
                    target_type="surface",
                    evidence={
                        'entropy': surface.entropy,
                        'threshold': threshold,
                        'claim_count': len(surface.claim_ids),
                        'sources': list(surface.sources)
                    },
                    params_version=self.params.version
                )
                meta_claims.append(mc)

        return meta_claims

    def _detect_single_source(self) -> List[MetaClaim]:
        """Detect claims with only one source (needs corroboration)."""
        meta_claims = []

        for surface in self.surfaces.values():
            if len(surface.sources) == 1:
                mc = MetaClaim(
                    type="single_source_only",
                    target_id=surface.id,
                    target_type="surface",
                    evidence={
                        'source': list(surface.sources)[0],
                        'claim_count': len(surface.claim_ids)
                    },
                    params_version=self.params.version
                )
                meta_claims.append(mc)

        return meta_claims

    def _detect_conflicts(self) -> List[MetaClaim]:
        """Detect unresolved conflicts between claims."""
        meta_claims = []

        for c1, c2, rel, conf in self.edges:
            if rel == Relation.CONFLICTS:
                mc = MetaClaim(
                    type="unresolved_conflict",
                    target_id=f"{c1}:{c2}",
                    target_type="claim_pair",
                    evidence={
                        'claim_1': c1,
                        'claim_2': c2,
                        'confidence': conf,
                        'claim_1_text': self.claims[c1].text if c1 in self.claims else None,
                        'claim_2_text': self.claims[c2].text if c2 in self.claims else None
                    },
                    params_version=self.params.version
                )
                meta_claims.append(mc)

        return meta_claims


def detect_tensions(
    claims: Dict[str, Claim],
    surfaces: Dict[str, Surface],
    edges: List[Tuple[str, str, Relation, float]],
    params: Parameters = None
) -> List[MetaClaim]:
    """
    Convenience function to detect tensions.

    Returns list of meta-claims.
    """
    detector = TensionDetector(claims, surfaces, edges, params)
    return detector.detect_all()


def resolve_meta_claim(
    meta_claims: List[MetaClaim],
    meta_claim_id: str,
    resolution: str,
    actor: str = "system"
) -> Optional[MetaClaim]:
    """
    Mark a meta-claim as resolved.

    Args:
        meta_claims: List of meta-claims to search
        meta_claim_id: ID of the meta-claim to resolve
        resolution: How it was resolved
        actor: Who/what resolved it

    Returns:
        The resolved meta-claim, or None if not found
    """
    for mc in meta_claims:
        if mc.id == meta_claim_id:
            mc.resolved = True
            mc.resolution = f"{resolution} by {actor}"
            return mc
    return None


def get_unresolved(meta_claims: List[MetaClaim]) -> List[MetaClaim]:
    """Return meta-claims that haven't been resolved."""
    return [mc for mc in meta_claims if not mc.resolved]


def count_by_type(meta_claims: List[MetaClaim]) -> Dict[str, int]:
    """Count meta-claims by type."""
    counts: Dict[str, int] = {}
    for mc in meta_claims:
        counts[mc.type] = counts.get(mc.type, 0) + 1
    return counts
