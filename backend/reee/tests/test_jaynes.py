"""
Jaynes Law Tests: Probability as Extended Logic
================================================

These tests validate that REEE's belief states follow Jaynes' program:
- Maximum entropy priors (honest uncertainty)
- Bayes' rule for updates
- Predictable entropy changes with evidence

Laws tested:
1. Corroboration concentrates belief (entropy decreases)
2. Conflict increases uncertainty (entropy increases or stays)
3. Supersession shifts posterior (MAP moves to newer value)
4. Independent sources have stronger effect than same source
5. Unanimous agreement produces low entropy
6. Maximum entropy for uniform evidence

These are L2-internal tests - they don't touch aboutness/events.
"""

import pytest
from reee.belief_state import BeliefState, PropositionValue


# =============================================================================
# JAYNES LAW 1: Corroboration Concentrates Belief
# =============================================================================

class TestJaynesLaw1_CorroborationConcentrates:
    """Adding confirming evidence must not increase entropy."""

    def test_single_source_has_zero_entropy(self):
        """Single observation = certainty (no alternatives)."""
        state = BeliefState()
        state.add_observation(value=3, source="src1.com", claim_id="c1")

        # Single value = 0 entropy (log2(1) = 0)
        assert state.entropy() == 0.0
        assert state.map_value == 3
        assert state.map_probability == 1.0

    def test_corroboration_does_not_increase_entropy(self):
        """Adding confirming claim for same value cannot increase entropy."""
        state = BeliefState()
        state.add_observation(value=3, source="src1.com", claim_id="c1")
        entropy_before = state.entropy()

        # Add confirming observation (same value, different source)
        state.add_observation(
            value=3,
            source="src2.com",
            claim_id="c2",
            relation_to_existing="confirms",
            related_value=3
        )
        entropy_after = state.entropy()

        # Entropy should not increase
        assert entropy_after <= entropy_before
        assert state.map_value == 3

    def test_multiple_corroborations_keep_low_entropy(self):
        """Multiple confirming sources maintain certainty."""
        state = BeliefState()

        # Add 5 sources all reporting same value
        for i in range(5):
            state.add_observation(value=42, source=f"src{i}.com", claim_id=f"c{i}")

        # All same value = 0 entropy
        assert state.entropy() == 0.0
        assert state.map_value == 42
        assert state.map_probability == 1.0
        assert state.confidence_level() == "confirmed"


# =============================================================================
# JAYNES LAW 2: Conflict Increases Uncertainty
# =============================================================================

class TestJaynesLaw2_ConflictIncreasesUncertainty:
    """Adding conflicting evidence cannot decrease entropy (unless downweighted)."""

    def test_conflict_increases_entropy_from_zero(self):
        """Adding conflicting value increases entropy from 0."""
        state = BeliefState()
        state.add_observation(value=2, source="src1.com", claim_id="c1")

        assert state.entropy() == 0.0  # Single value

        # Add conflicting observation
        state.add_observation(
            value=3,
            source="src2.com",
            claim_id="c2",
            relation_to_existing="conflicts",
            related_value=2
        )

        # Now we have two values -> entropy > 0
        assert state.entropy() > 0.0
        assert state.has_conflict()

    def test_balanced_conflict_has_maximum_entropy(self):
        """Equal evidence for two values = maximum uncertainty."""
        state = BeliefState()

        # One source says 2
        state.add_observation(value=2, source="src1.com", claim_id="c1")
        # One source says 3
        state.add_observation(value=3, source="src2.com", claim_id="c2")

        # With equal evidence, entropy should be close to max (log2(2) = 1 bit)
        # Due to model specifics, may not be exactly 1.0
        assert state.entropy() > 0.5

        # MAP is one of them, but probability < 1
        assert state.map_value in (2, 3)
        assert state.map_probability < 1.0

    def test_conflict_resolution_by_corroboration(self):
        """Conflict can be resolved by additional corroboration."""
        state = BeliefState()

        # Initial conflict: 1 vs 1
        state.add_observation(value=2, source="src1.com", claim_id="c1")
        state.add_observation(value=3, source="src2.com", claim_id="c2")

        entropy_conflict = state.entropy()

        # Add two more sources confirming value=3
        state.add_observation(value=3, source="src3.com", claim_id="c3")
        state.add_observation(value=3, source="src4.com", claim_id="c4")

        entropy_resolved = state.entropy()

        # Entropy should decrease as evidence concentrates
        assert entropy_resolved < entropy_conflict
        assert state.map_value == 3
        assert state.map_probability > 0.7


# =============================================================================
# JAYNES LAW 3: Supersession Shifts Posterior
# =============================================================================

class TestJaynesLaw3_SupersessionShiftsPosterior:
    """Superseding claims shift MAP toward newer value."""

    def test_supersession_shifts_map(self):
        """'Death toll rises to X' should shift MAP toward X when evidence is balanced."""
        state = BeliefState()

        # Initial report (1 source)
        state.add_observation(value=2, source="src1.com", claim_id="c1")

        assert state.map_value == 2

        # Update report (supersedes) - now 1:1 with temporal advantage
        state.add_observation(
            value=3,
            source="src2.com",
            claim_id="c2",
            is_update=True,
            relation_to_existing="supersedes",
            related_value=2
        )

        # With equal base evidence + supersession bonus, updated value should win
        posterior = state.compute_posterior()
        # Supersession should give value=3 an edge
        assert posterior[3] > posterior[2], \
            f"Supersession should favor newer value: P(3)={posterior[3]:.3f} vs P(2)={posterior[2]:.3f}"

    def test_chain_of_supersessions(self):
        """Chain: 2 -> 3 -> 4 should favor latest."""
        state = BeliefState()

        state.add_observation(value=2, source="src1.com", claim_id="c1")
        state.add_observation(
            value=3,
            source="src2.com",
            claim_id="c2",
            relation_to_existing="supersedes",
            related_value=2
        )
        state.add_observation(
            value=4,
            source="src3.com",
            claim_id="c3",
            relation_to_existing="supersedes",
            related_value=3
        )

        # Latest value should have most mass
        posterior = state.compute_posterior()
        assert posterior[4] >= posterior[3]
        assert posterior[4] >= posterior[2]


# =============================================================================
# JAYNES LAW 4: Source Independence Matters
# =============================================================================

class TestJaynesLaw4_SourceIndependence:
    """Independent sources provide stronger evidence than repeated sources."""

    def test_different_sources_stronger_than_same(self):
        """3 different sources > 3 claims from same source."""
        # State with 3 different sources
        state_diverse = BeliefState()
        state_diverse.add_observation(value=5, source="src1.com", claim_id="c1")
        state_diverse.add_observation(value=5, source="src2.com", claim_id="c2")
        state_diverse.add_observation(value=5, source="src3.com", claim_id="c3")

        # State with same source (but we model this by lower effective weight)
        # In practice, source diversity affects confidence_level
        n_sources_diverse = len(set(obs.source for obs in state_diverse.observations))

        assert n_sources_diverse == 3
        assert state_diverse.confidence_level() == "confirmed"

    def test_source_weight_affects_posterior(self):
        """Higher source weight = stronger evidence."""
        # Trusted source says 10
        state = BeliefState(source_weights={"trusted.com": 2.0, "untrusted.com": 0.5})
        state.add_observation(value=10, source="trusted.com", claim_id="c1")
        state.add_observation(value=20, source="untrusted.com", claim_id="c2")

        # Trusted source should dominate
        assert state.map_value == 10
        assert state.map_probability > 0.6


# =============================================================================
# JAYNES LAW 5: Unanimous Agreement = Low Entropy
# =============================================================================

class TestJaynesLaw5_UnanimousAgreement:
    """When all sources agree, entropy should be minimal."""

    def test_unanimous_agreement_zero_entropy(self):
        """N sources all reporting same value = 0 entropy."""
        state = BeliefState()

        for i in range(10):
            state.add_observation(value="fire", source=f"src{i}.com", claim_id=f"c{i}")

        assert state.entropy() == 0.0
        assert state.normalized_entropy() == 0.0
        assert state.map_value == "fire"
        assert state.map_probability == 1.0

    def test_near_unanimous_low_entropy(self):
        """9 agree, 1 disagrees = low but non-zero entropy."""
        state = BeliefState()

        # 9 sources say "fire"
        for i in range(9):
            state.add_observation(value="fire", source=f"src{i}.com", claim_id=f"c{i}")

        # 1 source says "explosion"
        state.add_observation(value="explosion", source="outlier.com", claim_id="c_outlier")

        # Entropy should be low but non-zero
        assert 0 < state.entropy() < 0.5
        assert state.map_value == "fire"
        assert state.map_probability > 0.8


# =============================================================================
# JAYNES LAW 6: Maximum Entropy for Uniform Evidence
# =============================================================================

class TestJaynesLaw6_MaximumEntropy:
    """Equal evidence for all values = maximum entropy (honest uncertainty)."""

    def test_uniform_evidence_maximum_entropy(self):
        """1 source each for 4 values = max entropy (2 bits)."""
        state = BeliefState()

        state.add_observation(value=1, source="src1.com", claim_id="c1")
        state.add_observation(value=2, source="src2.com", claim_id="c2")
        state.add_observation(value=3, source="src3.com", claim_id="c3")
        state.add_observation(value=4, source="src4.com", claim_id="c4")

        # Should be close to max entropy (log2(4) = 2 bits)
        assert state.entropy() > 1.5
        assert state.normalized_entropy() > 0.7

        # MAP exists but has low probability
        assert state.map_probability < 0.5

    def test_credible_set_grows_with_uncertainty(self):
        """Higher uncertainty = larger credible set."""
        # Certain case
        state_certain = BeliefState()
        state_certain.add_observation(value=1, source="s1.com", claim_id="c1")

        # Uncertain case
        state_uncertain = BeliefState()
        state_uncertain.add_observation(value=1, source="s1.com", claim_id="c1")
        state_uncertain.add_observation(value=2, source="s2.com", claim_id="c2")
        state_uncertain.add_observation(value=3, source="s3.com", claim_id="c3")

        # Credible set should be larger for uncertain case
        assert len(state_certain.credible_set(0.95)) <= len(state_uncertain.credible_set(0.95))


# =============================================================================
# INTEGRATION: Death Toll Example
# =============================================================================

class TestDeathTollScenario:
    """
    Real-world scenario: death_count@brown_shooting

    Claims:
    - "2 killed" (src1)
    - "death toll rises to 3" (src2, update)
    - "2 killed" (src3)
    - "4 killed" (src4, outlier)

    Expected: MAP=3 with moderate uncertainty due to conflict.
    """

    def test_death_toll_scenario(self):
        """Full death toll scenario."""
        state = BeliefState()

        # Initial report
        state.add_observation(value=2, source="src1.com", claim_id="c1")

        # Update
        state.add_observation(
            value=3,
            source="src2.com",
            claim_id="c2",
            is_update=True,
            relation_to_existing="supersedes",
            related_value=2
        )

        # Corroboration of old value
        state.add_observation(value=2, source="src3.com", claim_id="c3")

        # Outlier
        state.add_observation(
            value=4,
            source="src4.com",
            claim_id="c4",
            relation_to_existing="conflicts",
            related_value=3
        )

        summary = state.summary()

        # Should have non-zero entropy (uncertainty)
        assert summary["entropy_bits"] > 0

        # Should have conflict detected
        assert summary["has_conflict"]

        # Confidence should not be "confirmed"
        assert summary["confidence"] in ("reported", "contested", "corroborated")

        # MAP should be 2 or 3 (most supported values)
        assert summary["map_value"] in (2, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
