"""
Typed Belief State: Pure Jaynes Inference Kernel
=================================================

This module is a clean, abstract inference engine with NO worldly assumptions.
All domain-specific knowledge (source types, scale guesses, extraction) is
injected via providers or parameters.

Design:
- ValueDomain: abstract prior + likelihood interface
- Observation: value distribution + provenance (no source_type requirement)
- TypedBeliefState: online Bayesian filtering

What this module does NOT contain:
- Hardcoded source types (reliable, tabloid, etc.)
- Hardcoded scale parameters (μ=5, 50, 500)
- Extraction logic (regex, LLM)
- Update language interpretation

All such decisions are pushed to:
- CountDomainConfig (passed in)
- NoiseModel (injectable)
- Extractors (separate module)
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable


# =============================================================================
# OBSERVATION KIND (point vs bound vs interval)
# =============================================================================

class ObservationKind(Enum):
    """
    What kind of observation was extracted.

    POINT: "13 dead" → X = 13
    LOWER_BOUND: "at least 13 dead" → X >= 13
    UPPER_BOUND: "up to 13 dead" → X <= 13
    INTERVAL: "10-15 dead" → 10 <= X <= 15
    APPROXIMATE: "dozens dead" → X ≈ some range
    NONE: couldn't extract a value → skip
    """
    POINT = "point"
    LOWER_BOUND = "lower_bound"
    UPPER_BOUND = "upper_bound"
    INTERVAL = "interval"
    APPROXIMATE = "approximate"
    NONE = "none"


# =============================================================================
# OBSERVATIONS (extractor output - no worldly assumptions)
# =============================================================================

@dataclass
class Observation:
    """
    An observation from an extractor.

    The extractor outputs:
    - kind: what type of observation (point, bound, interval)
    - value_distribution: distribution over values
    - For bounds: value is the bound, distribution encodes the constraint

    This is the ONLY input to inference - all extraction decisions happen
    before this point.
    """
    # What kind of observation
    kind: ObservationKind = ObservationKind.POINT

    # Distribution over values: {value: probability}
    # For POINT: peaked at extracted value
    # For LOWER_BOUND: uniform over [value, max] or decaying from value
    # For INTERVAL: uniform over [low, high]
    value_distribution: Dict[Any, float] = field(default_factory=dict)

    # For bounds/intervals: the bound value(s)
    bound_value: Optional[int] = None  # For LOWER_BOUND, UPPER_BOUND
    interval_low: Optional[int] = None  # For INTERVAL
    interval_high: Optional[int] = None

    # Provenance (for audit, not for inference unless noise_model uses it)
    source: str = ""
    claim_id: str = ""
    timestamp: Optional[float] = None

    # Extraction confidence (affects likelihood strength)
    extraction_confidence: float = 1.0

    # Soft constraint signals (NOT hard bounds by default)
    signals: Dict[str, Any] = field(default_factory=dict)
    # e.g., {"is_update": True, "direction": "increase"}

    @property
    def map_value(self) -> Any:
        """Most likely value from the distribution."""
        if not self.value_distribution:
            return self.bound_value  # For bounds, return the bound itself
        return max(self.value_distribution.items(), key=lambda x: x[1])[0]

    @property
    def confidence(self) -> float:
        """Probability mass on MAP value."""
        if not self.value_distribution:
            return 0.0
        return max(self.value_distribution.values())

    @classmethod
    def point(cls, value: Any, confidence: float = 0.85, source: str = "", **kwargs) -> 'Observation':
        """Create a point observation: X = value.

        Works for both numeric values (with neighbor spread) and categorical values.
        """
        dist = {value: confidence}
        remainder = 1.0 - confidence

        # For numeric values, spread remainder to neighbors
        if isinstance(value, (int, float)):
            if value > 0:
                dist[value - 1] = remainder / 2
            dist[value + 1] = remainder / 2 if value > 0 else remainder
        # For categorical values, just use the peaked distribution
        # (The CategoricalDomain's confusion matrix handles noise)

        return cls(
            kind=ObservationKind.POINT,
            value_distribution=dist,
            source=source,
            extraction_confidence=confidence,
            **kwargs
        )

    @classmethod
    def lower_bound(cls, value: int, max_count: int = 500, source: str = "", **kwargs) -> 'Observation':
        """
        Create a lower bound observation: X >= value.

        Uses exponentially decaying distribution above the bound.
        """
        dist = {}
        # Exponential decay from bound upward: P(x) ∝ exp(-λ(x-value))
        # λ chosen so 95% mass within 2x the bound
        lam = 2.0 / max(value, 1)  # Decay rate
        for x in range(value, min(value * 3, max_count) + 1):
            dist[x] = math.exp(-lam * (x - value))

        # Normalize
        total = sum(dist.values())
        dist = {x: p/total for x, p in dist.items()}

        return cls(
            kind=ObservationKind.LOWER_BOUND,
            value_distribution=dist,
            bound_value=value,
            source=source,
            **kwargs
        )

    @classmethod
    def interval(cls, low: int, high: int, source: str = "", **kwargs) -> 'Observation':
        """Create an interval observation: low <= X <= high."""
        dist = {x: 1.0 / (high - low + 1) for x in range(low, high + 1)}
        return cls(
            kind=ObservationKind.INTERVAL,
            value_distribution=dist,
            interval_low=low,
            interval_high=high,
            source=source,
            **kwargs
        )

    @classmethod
    def approximate(cls, center: int, spread: int = None, source: str = "", **kwargs) -> 'Observation':
        """
        Create an approximate observation: X ≈ center.

        "dozens" → center=36, spread=24 (12-60)
        "hundreds" → center=150, spread=100 (50-250)
        """
        if spread is None:
            spread = max(center // 2, 5)

        low = max(0, center - spread)
        high = center + spread

        # Triangular distribution peaked at center
        dist = {}
        for x in range(low, high + 1):
            dist[x] = 1.0 - abs(x - center) / (spread + 1)

        total = sum(dist.values())
        dist = {x: p/total for x, p in dist.items()}

        return cls(
            kind=ObservationKind.APPROXIMATE,
            value_distribution=dist,
            bound_value=center,
            source=source,
            **kwargs
        )

    @classmethod
    def none(cls) -> 'Observation':
        """Create a null observation (extraction failed)."""
        return cls(kind=ObservationKind.NONE, value_distribution={})


# =============================================================================
# NOISE MODEL (injectable - replaces hardcoded source types)
# =============================================================================

class NoiseModel(ABC):
    """
    Abstract noise model: P(observed | true).

    Implementations can be:
    - UniformNoise: single δ for everyone
    - SourceNoise: per-source δ from calibration
    - LearnedNoise: hierarchical model updated online
    """

    @abstractmethod
    def delta(self, obs: Observation) -> float:
        """Return δ = E|Y-X| for this observation."""
        pass


class UniformNoise(NoiseModel):
    """Single δ for all sources - honest uncertainty."""

    def __init__(self, delta: float = 2.0):
        """
        Args:
            delta: Expected absolute error E|Y-X| for any source.
                   Default 2.0 = weak assumption of moderate noise.
        """
        self._delta = delta

    def delta(self, obs: Observation) -> float:
        return self._delta


class CalibratedNoise(NoiseModel):
    """Per-source δ from calibration data."""

    def __init__(
        self,
        source_deltas: Dict[str, float],
        default_delta: float = 2.0
    ):
        """
        Args:
            source_deltas: {source_domain: calibrated_delta}
            default_delta: Fallback for unknown sources
        """
        self._deltas = source_deltas
        self._default = default_delta

    def delta(self, obs: Observation) -> float:
        return self._deltas.get(obs.source, self._default)


# =============================================================================
# VALUE DOMAINS (abstract - no hardcoded parameters)
# =============================================================================

class ValueDomain(ABC):
    """Abstract base for typed value domains."""

    @abstractmethod
    def prior(self, x: Any) -> float:
        """Prior probability P(X=x)."""
        pass

    @abstractmethod
    def support(self) -> List[Any]:
        """Enumerable support for exact inference."""
        pass

    @property
    @abstractmethod
    def is_monotone(self) -> bool:
        """Whether X_t >= X_{t-1} constraint applies."""
        pass


@dataclass
class CountDomainConfig:
    """Configuration for CountDomain - all parameters explicit."""
    max_count: int = 500

    # Scale mixture: list of (name, μ, weight)
    # μ = E[X] for this scale; weight = mixture probability
    scales: List[Tuple[str, float, float]] = field(default_factory=lambda: [
        ("small", 5.0, 1/3),
        ("medium", 50.0, 1/3),
        ("large", 500.0, 1/3),
    ])

    # Monotone constraint settings
    monotone: bool = True
    # Only raise hard bound if extraction_confidence >= this threshold
    hard_bound_confidence_threshold: float = 0.95


class CountDomain(ValueDomain):
    """
    Domain for count variables (death_count, injury_count).

    Prior: Mixture of geometrics with configurable scales.
    Likelihood: Discrete Laplace with δ from NoiseModel.

    No hardcoded assumptions - all via CountDomainConfig.
    """

    def __init__(self, config: Optional[CountDomainConfig] = None):
        self.config = config or CountDomainConfig()

        # Precompute geometric parameters: p = 1/(1+μ) gives E[X] = μ
        self._scale_params = []
        total_w = sum(w for _, _, w in self.config.scales)
        for name, mu, weight in self.config.scales:
            p = 1.0 / (1.0 + mu)
            self._scale_params.append((name, p, weight / total_w))

        # Cache prior
        self._prior_cache: Optional[Dict[int, float]] = None

    def _compute_prior(self) -> Dict[int, float]:
        """Compute mixture prior over all values."""
        if self._prior_cache is not None:
            return self._prior_cache

        prior = {}
        for x in range(self.config.max_count + 1):
            p_x = 0.0
            for name, p, weight in self._scale_params:
                # Geometric PMF: P(X=x) = (1-p)^x * p
                p_x += weight * ((1 - p) ** x) * p
            prior[x] = p_x

        # Normalize
        total = sum(prior.values())
        self._prior_cache = {x: p/total for x, p in prior.items()}
        return self._prior_cache

    def prior(self, x: int) -> float:
        if x < 0 or x > self.config.max_count:
            return 0.0
        return self._compute_prior().get(x, 0.0)

    def support(self) -> List[int]:
        return list(range(self.config.max_count + 1))

    @property
    def is_monotone(self) -> bool:
        return self.config.monotone


class CategoricalDomain(ValueDomain):
    """
    Domain for categorical variables (legal_status, verdict).

    Prior: Configurable (default uniform).
    Likelihood: Configurable confusion matrix (default symmetric noise).
    """

    def __init__(
        self,
        categories: List[str],
        prior_weights: Optional[Dict[str, float]] = None,
        confusion_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.categories = categories
        self._n = len(categories)

        # Prior (default uniform)
        if prior_weights:
            total = sum(prior_weights.values())
            self._prior = {c: prior_weights.get(c, 0) / total for c in categories}
        else:
            self._prior = {c: 1.0 / self._n for c in categories}

        # Confusion matrix (default: symmetric noise)
        if confusion_matrix:
            self._confusion = confusion_matrix
        else:
            self._confusion = self._default_confusion(noise_rate=0.1)

    def _default_confusion(self, noise_rate: float) -> Dict[str, Dict[str, float]]:
        """Symmetric confusion: (1-noise_rate) correct, noise_rate spread."""
        matrix = {}
        for true_val in self.categories:
            matrix[true_val] = {}
            for obs_val in self.categories:
                if obs_val == true_val:
                    matrix[true_val][obs_val] = 1.0 - noise_rate
                else:
                    matrix[true_val][obs_val] = noise_rate / (self._n - 1) if self._n > 1 else 0
        return matrix

    def prior(self, x: str) -> float:
        return self._prior.get(x, 0.0)

    def likelihood(self, observed: str, true_val: str) -> float:
        """P(observed | true_val) from confusion matrix."""
        if true_val not in self._confusion:
            return 1.0 / self._n
        return self._confusion[true_val].get(observed, 0.0)

    def support(self) -> List[str]:
        return self.categories

    @property
    def is_monotone(self) -> bool:
        return False


# =============================================================================
# TYPED BELIEF STATE (pure inference)
# =============================================================================

@dataclass
class TypedBeliefState:
    """
    Bayesian belief state over a typed domain.

    Pure inference - no worldly assumptions. All domain knowledge
    comes from the injected domain and noise_model.
    """

    domain: ValueDomain
    noise_model: NoiseModel = field(default_factory=lambda: UniformNoise(2.0))
    observations: List[Observation] = field(default_factory=list)

    # For monotone domains: soft lower bound on X_t
    # Only becomes hard when high-confidence update is observed
    _soft_lower_bound: int = 0
    _hard_lower_bound: int = 0

    # Log scores for each observation (for calibration/learning)
    _log_scores: List[float] = field(default_factory=list)

    # Cached posterior
    _posterior: Optional[Dict[Any, float]] = field(default=None, repr=False)

    def add_observation(self, obs: Observation) -> float:
        """
        Add observation and update posterior.

        Handles ObservationKind properly:
        - POINT: standard likelihood update
        - LOWER_BOUND: enforces hard constraint for monotone domains
        - INTERVAL/APPROXIMATE: uses broad distribution
        - NONE: skipped entirely

        Returns:
            Log predictive score: log P(obs | previous_observations)
            (Used for calibration/learning noise model)
        """
        # Skip null observations
        if obs.kind == ObservationKind.NONE:
            return 0.0

        # Compute log score BEFORE adding observation
        log_score = self._compute_log_score(obs)
        self._log_scores.append(log_score)

        self.observations.append(obs)

        # Update bounds for monotone domains
        if self.domain.is_monotone:
            self._update_monotone_bounds(obs)

        self._posterior = None  # Invalidate cache
        return log_score

    def _compute_log_score(self, obs: Observation) -> float:
        """Log predictive probability of observation given current state."""
        if not self.observations:
            # First observation: score against prior
            posterior = {x: self.domain.prior(x) for x in self.domain.support()}
        else:
            posterior = self.compute_posterior()

        # Marginalize: P(obs) = Σ_x P(obs|x) P(x)
        delta = self.noise_model.delta(obs)
        lam = 1.0 / delta

        p_obs = 0.0
        for x, p_x in posterior.items():
            if p_x <= 0:
                continue
            for y, p_y in obs.value_distribution.items():
                if p_y <= 0:
                    continue
                # Laplace likelihood for counts, confusion matrix for categorical
                if isinstance(self.domain, CountDomain):
                    lik = math.exp(-lam * abs(y - x))
                elif isinstance(self.domain, CategoricalDomain):
                    lik = self.domain.likelihood(y, x)
                else:
                    lik = 1.0 if y == x else 0.1
                p_obs += p_y * lik * p_x

        return math.log(p_obs) if p_obs > 0 else float('-inf')

    def _update_monotone_bounds(self, obs: Observation) -> None:
        """
        Update bounds based on observation kind and signals.

        LOWER_BOUND observations ("at least X") always set hard bound.
        POINT observations also set hard bound for monotone counts (deaths confirmed).
        """
        # Get the relevant value
        if obs.kind == ObservationKind.LOWER_BOUND:
            val = obs.bound_value
        elif obs.kind == ObservationKind.POINT:
            map_val = obs.map_value
            if map_val is None or not isinstance(map_val, (int, float)):
                return
            val = int(map_val)
        else:
            map_val = obs.map_value
            if map_val is None or not isinstance(map_val, (int, float)):
                return
            val = int(map_val)
            # For approximate/interval, only update soft bound
            self._soft_lower_bound = max(self._soft_lower_bound, val)
            return

        if val is None:
            return

        # Always update soft bound with MAP/bound value
        self._soft_lower_bound = max(self._soft_lower_bound, val)

        # LOWER_BOUND observations are hard constraints by definition
        if obs.kind == ObservationKind.LOWER_BOUND:
            self._hard_lower_bound = max(self._hard_lower_bound, val)
            return

        # For POINT observations in monotone count domains:
        # A confirmed count is a hard floor (you can't have fewer deaths once confirmed)
        if obs.kind == ObservationKind.POINT and isinstance(self.domain, CountDomain):
            self._hard_lower_bound = max(self._hard_lower_bound, val)
            return

        # For other domains: only update hard bound if explicitly marked as update
        is_update = obs.signals.get("is_update", False)
        if isinstance(self.domain, CountDomain):
            threshold = self.domain.config.hard_bound_confidence_threshold
        else:
            threshold = 0.95

        if is_update and obs.extraction_confidence >= threshold:
            self._hard_lower_bound = max(self._hard_lower_bound, val)

    def compute_posterior(self) -> Dict[Any, float]:
        """Compute posterior P(X | observations)."""
        if self._posterior is not None:
            return self._posterior

        support = self.domain.support()

        # Start with log prior
        log_posterior = {}
        for x in support:
            p = self.domain.prior(x)
            log_posterior[x] = math.log(p) if p > 0 else float('-inf')

        # Apply hard monotone constraint (soft bound used for display only)
        if self.domain.is_monotone:
            for x in support:
                if x < self._hard_lower_bound:
                    log_posterior[x] = float('-inf')

        # Multiply by likelihoods
        for obs in self.observations:
            delta = self.noise_model.delta(obs)
            lam = 1.0 / delta

            for x in support:
                if log_posterior[x] == float('-inf'):
                    continue

                # Marginalize over observation's value distribution
                lik = 0.0
                for y, p_y in obs.value_distribution.items():
                    if p_y <= 0:
                        continue

                    if isinstance(self.domain, CountDomain):
                        # Discrete Laplace
                        lik += p_y * math.exp(-lam * abs(y - x))
                    elif isinstance(self.domain, CategoricalDomain):
                        # Confusion matrix
                        lik += p_y * self.domain.likelihood(y, x)
                    else:
                        # Fallback: exact match
                        lik += p_y * (1.0 if y == x else 0.1)

                # Scale by extraction confidence
                lik = lik ** obs.extraction_confidence

                if lik > 0:
                    log_posterior[x] += math.log(lik)
                else:
                    log_posterior[x] = float('-inf')

        # Convert to probabilities
        max_log = max(log_posterior.values())
        if max_log == float('-inf'):
            # All zero - uniform over valid support
            valid = [x for x in support if not (self.domain.is_monotone and x < self._hard_lower_bound)]
            self._posterior = {x: 1.0/len(valid) for x in valid} if valid else {}
            return self._posterior

        unnorm = {x: math.exp(lp - max_log) for x, lp in log_posterior.items() if lp > float('-inf')}
        total = sum(unnorm.values())
        self._posterior = {x: p/total for x, p in unnorm.items()}
        return self._posterior

    def entropy(self) -> float:
        """Shannon entropy of posterior in bits."""
        posterior = self.compute_posterior()
        if not posterior:
            return 0.0
        h = 0.0
        for p in posterior.values():
            if p > 0:
                h -= p * math.log2(p)
        return h

    def max_entropy(self) -> float:
        """Maximum possible entropy."""
        if self.domain.is_monotone:
            n = len([x for x in self.domain.support() if x >= self._hard_lower_bound])
        else:
            n = len(self.domain.support())
        return math.log2(n) if n > 1 else 0.0

    def normalized_entropy(self) -> float:
        """Entropy normalized to [0, 1]."""
        max_h = self.max_entropy()
        return self.entropy() / max_h if max_h > 0 else 0.0

    @property
    def map_value(self) -> Optional[Any]:
        """Maximum a posteriori value."""
        posterior = self.compute_posterior()
        if not posterior:
            return None
        return max(posterior.items(), key=lambda x: x[1])[0]

    @property
    def map_probability(self) -> float:
        """Probability of MAP value."""
        posterior = self.compute_posterior()
        return max(posterior.values()) if posterior else 0.0

    def credible_interval(self, mass: float = 0.95) -> Tuple[Any, Any]:
        """Credible interval containing given probability mass."""
        posterior = self.compute_posterior()
        if not posterior:
            return (None, None)

        if self.domain.is_monotone:
            sorted_items = sorted(posterior.items(), key=lambda x: x[0])
        else:
            sorted_items = sorted(posterior.items(), key=lambda x: -x[1])

        cumsum = 0.0
        included = []
        for x, p in sorted_items:
            included.append(x)
            cumsum += p
            if cumsum >= mass:
                break

        if self.domain.is_monotone and included:
            return (min(included), max(included))
        return tuple(included)

    def total_log_score(self) -> float:
        """Sum of log scores for all observations (for model comparison)."""
        return sum(self._log_scores)

    def summary(self) -> Dict:
        """Summary of belief state."""
        return {
            "map_value": self.map_value,
            "map_probability": round(self.map_probability, 4),
            "entropy_bits": round(self.entropy(), 4),
            "normalized_entropy": round(self.normalized_entropy(), 4),
            "n_observations": len(self.observations),
            "hard_lower_bound": self._hard_lower_bound if self.domain.is_monotone else None,
            "soft_lower_bound": self._soft_lower_bound if self.domain.is_monotone else None,
            "credible_95": self.credible_interval(0.95),
            "total_log_score": round(self.total_log_score(), 4),
        }


# =============================================================================
# FACTORY FUNCTIONS (convenience, no hidden assumptions)
# =============================================================================

def count_belief(
    max_count: int = 500,
    scales: Optional[List[Tuple[str, float, float]]] = None,
    noise_delta: float = 2.0,
    monotone: bool = True,
) -> TypedBeliefState:
    """
    Create a belief state for count variables.

    Args:
        max_count: Maximum possible count
        scales: List of (name, μ, weight) for scale mixture prior
        noise_delta: Default E|Y-X| for all sources
        monotone: Whether to enforce X_t >= X_{t-1}
    """
    config = CountDomainConfig(
        max_count=max_count,
        scales=scales or [("small", 5.0, 1/3), ("medium", 50.0, 1/3), ("large", 500.0, 1/3)],
        monotone=monotone,
    )
    return TypedBeliefState(
        domain=CountDomain(config),
        noise_model=UniformNoise(noise_delta),
    )


def categorical_belief(
    categories: List[str],
    prior_weights: Optional[Dict[str, float]] = None,
    noise_rate: float = 0.1,
) -> TypedBeliefState:
    """
    Create a belief state for categorical variables.

    Args:
        categories: List of possible values
        prior_weights: Prior probabilities (default uniform)
        noise_rate: Symmetric confusion rate
    """
    domain = CategoricalDomain(
        categories=categories,
        prior_weights=prior_weights,
    )
    # For categorical, noise is handled by confusion matrix, not NoiseModel
    return TypedBeliefState(domain=domain)
