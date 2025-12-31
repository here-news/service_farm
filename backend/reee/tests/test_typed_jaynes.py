"""
Typed Jaynes Law Tests: Pure Inference Kernel
==============================================

Tests for TypedBeliefState with explicit constraints.
All worldly assumptions (source types, scales) are injected, not hardcoded.

Jaynes laws tested:
1. Corroboration decreases entropy
2. Conflict increases entropy
3. Noise model affects influence (higher δ = weaker evidence)
4. Monotone constraint (soft vs hard bounds)
5. Update signals with high confidence raise hard bound
6. Log scores for calibration

These tests validate inference behavior, not worldly assumptions.
"""

import pytest
from reee.typed_belief import (
    TypedBeliefState, CountDomain, CategoricalDomain, CountDomainConfig,
    Observation, UniformNoise, CalibratedNoise,
    count_belief, categorical_belief
)


# =============================================================================
# HELPERS
# =============================================================================

def obs(
    value: int,
    confidence: float = 0.9,
    source: str = "test.com",
    is_update: bool = False,
    extraction_confidence: float = 1.0
) -> Observation:
    """Create a count observation with peaked distribution."""
    dist = {value: confidence}
    remainder = 1.0 - confidence
    if value > 0:
        dist[value - 1] = remainder / 2
    dist[value + 1] = remainder / 2 if value > 0 else remainder
    return Observation(
        value_distribution=dist,
        source=source,
        extraction_confidence=extraction_confidence,
        signals={"is_update": is_update} if is_update else {},
    )


# =============================================================================
# LAW 1: CORROBORATION DECREASES ENTROPY
# =============================================================================

class TestJaynesLaw1_CorroborationDecreasesEntropy:
    """Adding confirming evidence must decrease entropy."""

    def test_two_sources_lower_entropy_than_one(self):
        """Two sources agreeing have lower entropy than one."""
        state1 = count_belief()
        state1.add_observation(obs(5))
        h1 = state1.entropy()

        state2 = count_belief()
        state2.add_observation(obs(5))
        state2.add_observation(obs(5))
        h2 = state2.entropy()

        assert h2 < h1, f"Corroboration should decrease entropy: {h2} not < {h1}"

    def test_three_sources_lower_than_two(self):
        """Three sources have lower entropy than two."""
        state2 = count_belief()
        state2.add_observation(obs(10))
        state2.add_observation(obs(10))
        h2 = state2.entropy()

        state3 = count_belief()
        state3.add_observation(obs(10))
        state3.add_observation(obs(10))
        state3.add_observation(obs(10))
        h3 = state3.entropy()

        assert h3 < h2

    def test_map_probability_increases_with_corroboration(self):
        """MAP probability increases with corroboration."""
        state = count_belief()
        state.add_observation(obs(7))
        p1 = state.map_probability

        state.add_observation(obs(7))
        p2 = state.map_probability

        assert p2 > p1


# =============================================================================
# LAW 2: CONFLICT INCREASES ENTROPY
# =============================================================================

class TestJaynesLaw2_ConflictIncreasesEntropy:
    """Conflicting evidence increases entropy."""

    def test_conflict_higher_entropy_than_agreement(self):
        """Conflicting reports have higher entropy than agreement."""
        state_agree = count_belief()
        state_agree.add_observation(obs(5))
        state_agree.add_observation(obs(5))
        h_agree = state_agree.entropy()

        state_conflict = count_belief()
        state_conflict.add_observation(obs(5))
        state_conflict.add_observation(obs(10))
        h_conflict = state_conflict.entropy()

        assert h_conflict > h_agree

    def test_larger_conflict_higher_entropy(self):
        """Larger disagreement means more entropy."""
        state_small = count_belief()
        state_small.add_observation(obs(5))
        state_small.add_observation(obs(6))  # Small disagreement
        h_small = state_small.entropy()

        state_large = count_belief()
        state_large.add_observation(obs(5))
        state_large.add_observation(obs(15))  # Large disagreement
        h_large = state_large.entropy()

        assert h_large > h_small


# =============================================================================
# LAW 3: NOISE MODEL AFFECTS INFLUENCE
# =============================================================================

class TestJaynesLaw3_NoiseModelAffectsInfluence:
    """Higher δ = weaker evidence (flatter likelihood)."""

    def test_low_delta_concentrates_faster(self):
        """Lower δ (sharper likelihood) concentrates posterior faster."""
        # Low noise (δ=1)
        state_low = count_belief(noise_delta=1.0)
        state_low.add_observation(obs(5))
        h_low = state_low.entropy()

        # High noise (δ=5)
        state_high = count_belief(noise_delta=5.0)
        state_high.add_observation(obs(5))
        h_high = state_high.entropy()

        # Lower noise should give lower entropy (more concentrated)
        assert h_low < h_high

    def test_calibrated_noise_per_source(self):
        """CalibratedNoise gives different weight per source."""
        noise = CalibratedNoise(
            source_deltas={"trusted.com": 1.0, "noisy.com": 10.0},
            default_delta=2.0
        )

        config = CountDomainConfig(max_count=100)
        state = TypedBeliefState(
            domain=CountDomain(config),
            noise_model=noise
        )

        # Trusted source says 5
        state.add_observation(obs(5, source="trusted.com"))
        # Noisy source says 50
        state.add_observation(obs(50, source="noisy.com"))

        # Trusted source should dominate
        assert state.map_value < 20, f"Trusted should dominate: MAP={state.map_value}"


# =============================================================================
# LAW 4: MONOTONE CONSTRAINT (SOFT VS HARD)
# =============================================================================

class TestJaynesLaw4_MonotoneConstraint:
    """Monotone constraint with soft and hard bounds."""

    def test_soft_bound_tracked(self):
        """Soft bound tracks maximum observed value."""
        state = count_belief()
        state.add_observation(obs(5))
        assert state._soft_lower_bound == 5

        state.add_observation(obs(10))
        assert state._soft_lower_bound == 10

    def test_hard_bound_only_with_high_confidence_update(self):
        """Hard bound only raised with is_update=True AND high confidence."""
        state = count_belief()

        # Regular observation doesn't raise hard bound
        state.add_observation(obs(5, is_update=False, extraction_confidence=1.0))
        assert state._hard_lower_bound == 0

        # Update with low confidence doesn't raise hard bound
        state.add_observation(obs(10, is_update=True, extraction_confidence=0.5))
        assert state._hard_lower_bound == 0

        # Update with high confidence raises hard bound
        state.add_observation(obs(15, is_update=True, extraction_confidence=0.98))
        assert state._hard_lower_bound == 15

    def test_hard_bound_excludes_lower_values(self):
        """Values below hard bound have zero probability."""
        state = count_belief()
        state.add_observation(obs(10, is_update=True, extraction_confidence=0.99))

        posterior = state.compute_posterior()
        for x in range(10):
            assert posterior.get(x, 0) == 0


# =============================================================================
# LAW 5: LOG SCORES FOR CALIBRATION
# =============================================================================

class TestJaynesLaw5_LogScores:
    """Log scores track predictive performance."""

    def test_log_scores_recorded(self):
        """Each observation gets a log score."""
        state = count_belief()
        state.add_observation(obs(5))
        state.add_observation(obs(5))
        state.add_observation(obs(6))

        assert len(state._log_scores) == 3

    def test_confirming_observation_higher_score(self):
        """Observation consistent with posterior gets higher score."""
        state = count_belief()
        state.add_observation(obs(5))
        state.add_observation(obs(5))

        # Add confirming observation
        score_confirm = state.add_observation(obs(5))

        # Reset and add conflicting
        state2 = count_belief()
        state2.add_observation(obs(5))
        state2.add_observation(obs(5))
        score_conflict = state2.add_observation(obs(50))

        assert score_confirm > score_conflict

    def test_total_log_score_for_model_comparison(self):
        """Total log score can compare models."""
        state = count_belief()
        state.add_observation(obs(5))
        state.add_observation(obs(5))

        total = state.total_log_score()
        assert total < 0  # Log probabilities are negative


# =============================================================================
# CATEGORICAL DOMAIN TESTS
# =============================================================================

class TestCategoricalDomain:
    """Tests for categorical domain."""

    def test_uniform_prior(self):
        """Prior is uniform by default."""
        state = categorical_belief(["a", "b", "c"])
        posterior = state.compute_posterior()
        probs = list(posterior.values())
        assert max(probs) - min(probs) < 0.01

    def test_observation_shifts_posterior(self):
        """Observation shifts posterior toward observed value."""
        state = categorical_belief(["guilty", "not_guilty"])
        state.add_observation(Observation(
            value_distribution={"guilty": 0.9, "not_guilty": 0.1}
        ))
        assert state.map_value == "guilty"

    def test_custom_prior(self):
        """Custom prior weights are respected."""
        state = categorical_belief(
            ["a", "b", "c"],
            prior_weights={"a": 10, "b": 1, "c": 1}
        )
        # Without observations, MAP should be "a"
        assert state.map_value == "a"


# =============================================================================
# INVARIANT CHECKS
# =============================================================================

class TestInvariants:
    """Invariants that must hold for any valid inference."""

    def test_posterior_sums_to_one(self):
        """Posterior must sum to 1."""
        state = count_belief()
        state.add_observation(obs(5))
        state.add_observation(obs(7))

        posterior = state.compute_posterior()
        total = sum(posterior.values())
        assert abs(total - 1.0) < 1e-10

    def test_entropy_non_negative(self):
        """Entropy must be non-negative."""
        state = count_belief()
        state.add_observation(obs(5))
        assert state.entropy() >= 0

    def test_normalized_entropy_in_01(self):
        """Normalized entropy must be in [0, 1]."""
        state = count_belief()
        state.add_observation(obs(5))
        state.add_observation(obs(20))
        h_norm = state.normalized_entropy()
        assert 0 <= h_norm <= 1

    def test_map_probability_in_01(self):
        """MAP probability must be in (0, 1]."""
        state = count_belief()
        state.add_observation(obs(5))
        p = state.map_probability
        assert 0 < p <= 1


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestConfiguration:
    """Test that all configuration is explicit."""

    def test_custom_scales(self):
        """Custom scale mixture can be injected."""
        # Single scale: all small events (μ=3)
        state = count_belief(
            scales=[("tiny", 3.0, 1.0)],
            max_count=50
        )
        # Prior should be peaked near 0-3
        assert state.domain.prior(0) > state.domain.prior(20)

    def test_non_monotone_domain(self):
        """Monotone constraint can be disabled."""
        state = count_belief(monotone=False)

        # Even with "update", no hard bound
        state.add_observation(obs(10, is_update=True, extraction_confidence=0.99))
        assert state._hard_lower_bound == 0

    def test_configurable_hard_bound_threshold(self):
        """Hard bound threshold is configurable."""
        config = CountDomainConfig(
            max_count=100,
            hard_bound_confidence_threshold=0.5  # Lower threshold
        )
        state = TypedBeliefState(domain=CountDomain(config))

        # Now even 0.6 confidence should raise hard bound
        state.add_observation(obs(10, is_update=True, extraction_confidence=0.6))
        assert state._hard_lower_bound == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
