"""
Test: Core Types
================

Tests the core data types (Claim, Surface, Event, Parameters, etc.)
"""

import pytest
from datetime import datetime

from reee import (
    Claim, Surface, Event, Parameters, MetaClaim,
    Relation, Association, MembershipLevel,
    ParameterChange, AboutnessLink
)


class TestClaim:
    """Claim dataclass tests."""

    def test_claim_creation_minimal(self):
        """Test creating a claim with minimal fields."""
        claim = Claim(
            id='test1',
            text='Test claim',
            source='test.com'
        )
        assert claim.id == 'test1'
        assert claim.text == 'Test claim'
        assert claim.source == 'test.com'

    def test_claim_creation_full(self):
        """Test creating a claim with all fields."""
        claim = Claim(
            id='test2',
            text='Full claim',
            source='full.com',
            entities={'entity1', 'entity2'},
            anchor_entities={'entity1'},
            embedding=[0.1, 0.2, 0.3],
            timestamp=datetime.now(),
            question_key='test_question'
        )
        assert len(claim.entities) == 2
        assert len(claim.anchor_entities) == 1
        assert len(claim.embedding) == 3

    def test_claim_default_entities(self):
        """Test that entities default to empty sets."""
        claim = Claim(id='test', text='text', source='src')
        assert claim.entities == set()
        assert claim.anchor_entities == set()


class TestSurface:
    """Surface dataclass tests."""

    def test_surface_creation(self):
        """Test creating a surface."""
        surface = Surface(
            id='S001',
            claim_ids=['c1', 'c2'],
            entities={'entity1'},
            anchor_entities={'entity1'},
            sources={'src1', 'src2'}
        )
        assert surface.id == 'S001'
        assert len(surface.claim_ids) == 2
        assert len(surface.sources) == 2

    def test_surface_defaults(self):
        """Test surface default values."""
        surface = Surface(id='S002')
        assert surface.claim_ids == set()  # Set, not list
        assert surface.entropy == 0.0
        assert surface.canonical_title is None


class TestEvent:
    """Event dataclass tests."""

    def test_event_creation(self):
        """Test creating an event."""
        event = Event(
            id='E001',
            surface_ids=['S001', 'S002'],
            anchor_entities={'anchor1'}
        )
        assert event.id == 'E001'
        assert len(event.surface_ids) == 2

    def test_event_defaults(self):
        """Test event default values."""
        event = Event(id='E002')
        assert event.surface_ids == set()  # Set, not list
        assert event.total_claims == 0
        assert event.total_sources == 0


class TestParameters:
    """Parameters dataclass tests."""

    def test_parameters_defaults(self):
        """Test default parameter values."""
        params = Parameters()
        assert params.version == 1
        assert params.identity_confidence_threshold == 0.5  # Actual default
        assert params.hub_max_df == 5
        assert params.aboutness_min_signals == 2
        assert params.changes == []

    def test_parameters_custom(self):
        """Test custom parameter values."""
        params = Parameters(
            identity_confidence_threshold=0.8,
            aboutness_threshold=0.3
        )
        assert params.identity_confidence_threshold == 0.8
        assert params.aboutness_threshold == 0.3

    def test_parameter_change_tracking(self):
        """Test parameter change tracking."""
        change = ParameterChange(
            parameter='threshold',  # Correct field name
            old_value=0.2,
            new_value=0.3,
            rationale='tuning',  # Correct field name
            actor='system'
        )
        params = Parameters(changes=[change])
        assert len(params.changes) == 1
        assert params.changes[0].parameter == 'threshold'


class TestMetaClaim:
    """MetaClaim dataclass tests."""

    def test_meta_claim_creation(self):
        """Test creating a meta-claim."""
        mc = MetaClaim(
            type='high_entropy_surface',
            target_id='S001',
            target_type='surface',
            evidence={'entropy': 0.9}
        )
        assert mc.type == 'high_entropy_surface'
        assert mc.target_id == 'S001'
        assert mc.resolved is False

    def test_meta_claim_id_generated(self):
        """Test that meta-claim ID is auto-generated."""
        mc = MetaClaim(type='test', target_id='t1', target_type='test')
        assert mc.id is not None
        assert mc.id.startswith('mc_')


class TestEnums:
    """Enum tests."""

    def test_relation_values(self):
        """Test Relation enum values."""
        assert Relation.CONFIRMS is not None
        assert Relation.REFINES is not None
        assert Relation.SUPERSEDES is not None
        assert Relation.CONFLICTS is not None
        assert Relation.UNRELATED is not None  # Correct enum value

    def test_association_values(self):
        """Test Association enum values."""
        assert Association.SAME is not None  # Correct enum values
        assert Association.RELATED is not None
        assert Association.DISTINCT is not None

    def test_membership_level_values(self):
        """Test MembershipLevel enum values."""
        assert MembershipLevel.CORE is not None
        assert MembershipLevel.PERIPHERY is not None  # Correct enum value
        assert MembershipLevel.QUARANTINE is not None  # Correct enum value


class TestAboutnessLink:
    """AboutnessLink dataclass tests."""

    def test_aboutness_link_creation(self):
        """Test creating an aboutness link."""
        link = AboutnessLink(
            target_id='S002',
            score=0.75,
            evidence={'shared_entities': ['entity1']}
        )
        assert link.target_id == 'S002'
        assert link.score == 0.75
        assert 'shared_entities' in link.evidence


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
