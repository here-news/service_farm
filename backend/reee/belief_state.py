"""
Belief State: Jaynes-Aligned Posterior Inference
=================================================

This module implements proper Bayesian belief states for REEE surfaces.

Jaynes' Program:
- Probability as extended logic
- Maximum entropy for honest uncertainty
- Bayes' rule for updating with evidence

A BeliefState represents the epistemic state of a proposition (L2 Surface):
- What values/hypotheses are possible?
- What is the posterior probability distribution?
- What is the epistemic entropy (true uncertainty)?

This is NOT the heuristic source-count entropy in topology.py.
This IS the posterior entropy over proposition values.

Example:
    Surface: death_count@brown_shooting
    Claims: ["2 killed", "3 killed", "2 killed", "death toll rises to 3"]

    BeliefState:
        hypotheses: {2: 0.35, 3: 0.60, other: 0.05}
        map_value: 3
        entropy: 0.72 bits
        confidence: "moderate" (not unanimous)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict


# =============================================================================
# VALUE TYPES (what propositions can assert)
# =============================================================================

@dataclass
class PropositionValue:
    """A value that a proposition can take."""
    value: Any                    # The actual value (number, string, etc.)
    source: str                   # Which source reported this
    claim_id: str                 # Which claim asserted this
    timestamp: Optional[str] = None
    is_update: bool = False       # "rises to", "now", "updated" language
    confidence: float = 1.0       # Source reliability weight


# =============================================================================
# BELIEF STATE
# =============================================================================

@dataclass
class BeliefState:
    """
    Bayesian belief state for a proposition.

    Computes posterior distribution over possible values based on:
    - Prior (uniform or weakly informative)
    - Likelihoods from each claim (weighted by source reliability)
    - Relation types (CONFIRMS increases weight, CONFLICTS spreads mass)

    Jaynes alignment:
    - Maximum entropy prior (uniform over observed values)
    - Multiplicative Bayes update
    - Shannon entropy as epistemic uncertainty
    """

    # Observed values from claims
    observations: List[PropositionValue] = field(default_factory=list)

    # Source reliability weights (source -> weight, default 1.0)
    source_weights: Dict[str, float] = field(default_factory=dict)

    # Conflict pairs: (value1, value2) that are mutually exclusive
    conflicts: Set[Tuple[Any, Any]] = field(default_factory=set)

    # Supersession chain: value -> newer_value (temporal ordering)
    supersessions: Dict[Any, Any] = field(default_factory=dict)

    # Cached posterior (recomputed on demand)
    _posterior: Optional[Dict[Any, float]] = field(default=None, repr=False)

    def add_observation(
        self,
        value: Any,
        source: str,
        claim_id: str,
        timestamp: Optional[str] = None,
        is_update: bool = False,
        relation_to_existing: Optional[str] = None,  # "confirms", "conflicts", "supersedes"
        related_value: Optional[Any] = None
    ):
        """
        Add a new observation and invalidate cache.

        Args:
            value: The observed value
            source: Source domain
            claim_id: Claim ID
            timestamp: Optional timestamp
            is_update: Whether claim has update language
            relation_to_existing: How this relates to existing values
            related_value: The value it relates to
        """
        weight = self.source_weights.get(source, 1.0)

        obs = PropositionValue(
            value=value,
            source=source,
            claim_id=claim_id,
            timestamp=timestamp,
            is_update=is_update,
            confidence=weight
        )
        self.observations.append(obs)

        # Track conflicts
        if relation_to_existing == "conflicts" and related_value is not None:
            self.conflicts.add((min(value, related_value), max(value, related_value)))

        # Track supersessions
        if relation_to_existing == "supersedes" and related_value is not None:
            self.supersessions[related_value] = value

        # Invalidate cache
        self._posterior = None

    def compute_posterior(self) -> Dict[Any, float]:
        """
        Compute posterior distribution over values.

        Uses multiplicative Bayesian update:
        P(v | observations) ∝ P(v) * ∏ P(obs_i | v)

        Where:
        - P(v) = uniform prior over observed values
        - P(obs | v) = 1.0 if obs.value == v, small ε otherwise

        Returns:
            Dict mapping values to posterior probabilities (sum to 1)
        """
        if self._posterior is not None:
            return self._posterior

        if not self.observations:
            return {}

        # Collect unique values
        values = set(obs.value for obs in self.observations)

        # Uniform prior (max entropy)
        prior = {v: 1.0 / len(values) for v in values}

        # Likelihood contributions
        # Each observation contributes evidence for its value
        log_likelihood = defaultdict(float)

        for obs in self.observations:
            for v in values:
                if obs.value == v:
                    # Observation supports this value
                    # Weight by source reliability
                    log_likelihood[v] += math.log(0.9) * obs.confidence
                else:
                    # Observation doesn't support this value
                    log_likelihood[v] += math.log(0.1) * obs.confidence

        # Apply supersession (temporal update model)
        # If value A is superseded by B, shift mass toward B
        for old_val, new_val in self.supersessions.items():
            if old_val in log_likelihood and new_val in log_likelihood:
                # Boost newer value, penalize older
                log_likelihood[new_val] += 0.5  # temporal bonus
                log_likelihood[old_val] -= 0.3  # temporal penalty

        # Convert to posterior
        # P(v) ∝ prior(v) * exp(log_likelihood[v])
        unnorm = {}
        max_ll = max(log_likelihood.values()) if log_likelihood else 0

        for v in values:
            # Subtract max for numerical stability
            unnorm[v] = prior[v] * math.exp(log_likelihood[v] - max_ll)

        # Normalize
        total = sum(unnorm.values())
        if total == 0:
            # Fallback to uniform
            posterior = {v: 1.0 / len(values) for v in values}
        else:
            posterior = {v: p / total for v, p in unnorm.items()}

        self._posterior = posterior
        return posterior

    def entropy(self) -> float:
        """
        Shannon entropy of the posterior distribution.

        H = -Σ p(v) * log2(p(v))

        Returns:
            Entropy in bits. 0 = certain, higher = more uncertain.
        """
        posterior = self.compute_posterior()
        if not posterior:
            return 0.0

        h = 0.0
        for p in posterior.values():
            if p > 0:
                h -= p * math.log2(p)

        return h

    def max_entropy(self) -> float:
        """Maximum possible entropy (uniform distribution)."""
        n = len(set(obs.value for obs in self.observations))
        if n <= 1:
            return 0.0
        return math.log2(n)

    def normalized_entropy(self) -> float:
        """Entropy normalized to [0, 1] range."""
        max_h = self.max_entropy()
        if max_h == 0:
            return 0.0
        return self.entropy() / max_h

    @property
    def map_value(self) -> Optional[Any]:
        """Maximum a posteriori value (most probable)."""
        posterior = self.compute_posterior()
        if not posterior:
            return None
        return max(posterior.items(), key=lambda x: x[1])[0]

    @property
    def map_probability(self) -> float:
        """Probability of the MAP value."""
        posterior = self.compute_posterior()
        if not posterior:
            return 0.0
        return max(posterior.values())

    def credible_set(self, threshold: float = 0.95) -> Set[Any]:
        """
        Minimum credible set containing threshold probability mass.

        Args:
            threshold: Desired probability mass (default 0.95)

        Returns:
            Set of values in the credible set
        """
        posterior = self.compute_posterior()
        if not posterior:
            return set()

        # Sort by probability descending
        sorted_values = sorted(posterior.items(), key=lambda x: -x[1])

        credible = set()
        mass = 0.0
        for v, p in sorted_values:
            credible.add(v)
            mass += p
            if mass >= threshold:
                break

        return credible

    def confidence_level(self) -> str:
        """
        Human-readable confidence level.

        Based on MAP probability and source diversity.
        """
        map_prob = self.map_probability
        n_sources = len(set(obs.source for obs in self.observations))

        if map_prob >= 0.9 and n_sources >= 3:
            return "confirmed"
        elif map_prob >= 0.7 and n_sources >= 2:
            return "corroborated"
        elif map_prob >= 0.5:
            return "reported"
        else:
            return "contested"

    def has_conflict(self) -> bool:
        """Whether there are conflicting values."""
        return len(self.conflicts) > 0 or len(set(obs.value for obs in self.observations)) > 1

    def summary(self) -> Dict:
        """Summary of the belief state."""
        posterior = self.compute_posterior()
        return {
            "map_value": self.map_value,
            "map_probability": round(self.map_probability, 3),
            "entropy_bits": round(self.entropy(), 3),
            "normalized_entropy": round(self.normalized_entropy(), 3),
            "confidence": self.confidence_level(),
            "n_observations": len(self.observations),
            "n_sources": len(set(obs.source for obs in self.observations)),
            "n_values": len(posterior),
            "has_conflict": self.has_conflict(),
            "posterior": {str(k): round(v, 3) for k, v in sorted(posterior.items(), key=lambda x: -x[1])[:5]}
        }


# =============================================================================
# BELIEF STATE BUILDER (from Surface claims)
# =============================================================================

def build_belief_state(
    claims: List[Dict],
    relations: List[Tuple[str, str, str, float]],
    source_weights: Optional[Dict[str, float]] = None
) -> BeliefState:
    """
    Build a BeliefState from a list of claims and their relations.

    Args:
        claims: List of claim dicts with 'id', 'value', 'source', 'text'
        relations: List of (claim1_id, claim2_id, relation_type, confidence)
        source_weights: Optional source reliability weights

    Returns:
        BeliefState with observations and relation metadata
    """
    state = BeliefState(source_weights=source_weights or {})

    # Build claim_id -> value mapping
    claim_values = {c['id']: c.get('value', c.get('text', '')) for c in claims}

    # Build relation lookup
    claim_relations = defaultdict(list)
    for c1, c2, rel, conf in relations:
        claim_relations[c2].append((c1, rel, claim_values.get(c1)))

    # Add observations
    for claim in claims:
        claim_id = claim['id']
        value = claim.get('value', claim.get('text', ''))
        source = claim.get('source', 'unknown')
        is_update = claim.get('is_update', False)

        # Check if this claim has relations to existing
        rel_to_existing = None
        related_value = None

        for other_id, rel, other_val in claim_relations.get(claim_id, []):
            rel_to_existing = rel.lower() if isinstance(rel, str) else rel.value.lower()
            related_value = other_val
            break

        state.add_observation(
            value=value,
            source=source,
            claim_id=claim_id,
            is_update=is_update,
            relation_to_existing=rel_to_existing,
            related_value=related_value
        )

    return state
