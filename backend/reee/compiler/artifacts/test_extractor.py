"""
Tests for Artifact Extractor

These tests verify the extractor correctly classifies entities:
1. Wang Fuk Court → SPECIFIC_PLACE (referent)
2. Hong Kong, Tai Po → BROAD_LOCATION (context)
3. Donald Trump → PERSON (referent)
4. Ambiguous entities → InquirySeed + context (conservative)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
import json

from reee.compiler.artifacts.extractor import (
    ReferentType,
    EntityClassification,
    InquirySeed,
    ExtractionResult,
    extract_artifact,
    _resolve_entity_id,
    _make_conservative_result,
)
from reee.compiler.membrane import ReferentRole


# =============================================================================
# Mock LLM Client
# =============================================================================

def make_mock_llm(response: dict):
    """Create a mock LLM client that returns the given response."""
    mock = AsyncMock()
    mock.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(response)))]
    )
    return mock


# =============================================================================
# Test 1: Specific places become PLACE referents
# =============================================================================

class TestSpecificPlaceExtraction:
    """Wang Fuk Court should be extracted as SPECIFIC_PLACE referent."""

    @pytest.mark.asyncio
    async def test_wfc_as_specific_place(self):
        """Wang Fuk Court is classified as SPECIFIC_PLACE → PLACE referent."""
        llm_response = {
            "classifications": [
                {
                    "entity_id": "Wang Fuk Court",
                    "entity_name": "Wang Fuk Court",
                    "classification": "SPECIFIC_PLACE",
                    "confidence": 0.95,
                    "reasoning": "A specific building complex in Tai Po"
                },
                {
                    "entity_id": "Tai Po",
                    "entity_name": "Tai Po",
                    "classification": "BROAD_LOCATION",
                    "confidence": 0.9,
                    "reasoning": "A district in Hong Kong, geographic context"
                },
            ],
            "overall_confidence": 0.9
        }
        mock_llm = make_mock_llm(llm_response)

        result = await extract_artifact(
            incident_id="test_incident_1",
            title="Fire at Wang Fuk Court in Tai Po",
            anchor_entities={"Wang Fuk Court", "Tai Po"},
            entity_lookup={},
            llm_client=mock_llm,
        )

        # Wang Fuk Court should be a referent
        assert len(result.artifact.referents) == 1
        wfc_ref = list(result.artifact.referents)[0]
        assert "wang_fuk_court" in wfc_ref.entity_id.lower()
        assert wfc_ref.role == ReferentRole.PLACE

        # Tai Po should be context
        assert any("tai_po" in c.lower() for c in result.artifact.contexts)

        # No inquiries needed
        assert len(result.inquiries) == 0

    @pytest.mark.asyncio
    async def test_building_specific_place(self):
        """Other specific buildings should also be PLACE referents."""
        llm_response = {
            "classifications": [
                {
                    "entity_id": "White House",
                    "entity_name": "White House",
                    "classification": "SPECIFIC_PLACE",
                    "confidence": 0.98,
                    "reasoning": "Specific building, presidential residence"
                },
            ],
            "overall_confidence": 0.95
        }
        mock_llm = make_mock_llm(llm_response)

        result = await extract_artifact(
            incident_id="test_incident_2",
            title="Event at White House",
            anchor_entities={"White House"},
            entity_lookup={},
            llm_client=mock_llm,
        )

        assert len(result.artifact.referents) == 1
        ref = list(result.artifact.referents)[0]
        assert ref.role == ReferentRole.PLACE


# =============================================================================
# Test 2: Broad locations become context
# =============================================================================

class TestBroadLocationExtraction:
    """Hong Kong, Tai Po, China should be BROAD_LOCATION → context."""

    @pytest.mark.asyncio
    async def test_city_as_context(self):
        """Hong Kong should be context, not referent."""
        llm_response = {
            "classifications": [
                {
                    "entity_id": "Hong Kong",
                    "entity_name": "Hong Kong",
                    "classification": "BROAD_LOCATION",
                    "confidence": 0.95,
                    "reasoning": "A city/region, not a specific venue"
                },
            ],
            "overall_confidence": 0.9
        }
        mock_llm = make_mock_llm(llm_response)

        result = await extract_artifact(
            incident_id="test_incident_3",
            title="Incident in Hong Kong",
            anchor_entities={"Hong Kong"},
            entity_lookup={},
            llm_client=mock_llm,
        )

        # No referents
        assert len(result.artifact.referents) == 0

        # Hong Kong is context
        assert any("hong_kong" in c.lower() for c in result.artifact.contexts)

    @pytest.mark.asyncio
    async def test_country_as_context(self):
        """Countries should always be context."""
        llm_response = {
            "classifications": [
                {
                    "entity_id": "China",
                    "entity_name": "China",
                    "classification": "BROAD_LOCATION",
                    "confidence": 0.99,
                    "reasoning": "A country, geographic context only"
                },
            ],
            "overall_confidence": 0.95
        }
        mock_llm = make_mock_llm(llm_response)

        result = await extract_artifact(
            incident_id="test_incident_4",
            title="News from China",
            anchor_entities={"China"},
            entity_lookup={},
            llm_client=mock_llm,
        )

        assert len(result.artifact.referents) == 0
        assert len(result.artifact.contexts) == 1


# =============================================================================
# Test 3: Persons become PERSON referents
# =============================================================================

class TestPersonExtraction:
    """Named individuals should be PERSON referents."""

    @pytest.mark.asyncio
    async def test_person_as_referent(self):
        """Donald Trump should be PERSON referent."""
        llm_response = {
            "classifications": [
                {
                    "entity_id": "Donald Trump",
                    "entity_name": "Donald Trump",
                    "classification": "PERSON",
                    "confidence": 0.99,
                    "reasoning": "Named individual"
                },
            ],
            "overall_confidence": 0.95
        }
        mock_llm = make_mock_llm(llm_response)

        result = await extract_artifact(
            incident_id="test_incident_5",
            title="Statement by Donald Trump",
            anchor_entities={"Donald Trump"},
            entity_lookup={},
            llm_client=mock_llm,
        )

        assert len(result.artifact.referents) == 1
        ref = list(result.artifact.referents)[0]
        assert ref.role == ReferentRole.PERSON


# =============================================================================
# Test 4: Objects become OBJECT referents
# =============================================================================

class TestObjectExtraction:
    """Specific artifacts should be OBJECT referents."""

    @pytest.mark.asyncio
    async def test_vehicle_as_referent(self):
        """Flight MH370 should be OBJECT referent."""
        llm_response = {
            "classifications": [
                {
                    "entity_id": "Flight MH370",
                    "entity_name": "Flight MH370",
                    "classification": "OBJECT",
                    "confidence": 0.98,
                    "reasoning": "Specific aircraft, central to incident"
                },
            ],
            "overall_confidence": 0.95
        }
        mock_llm = make_mock_llm(llm_response)

        result = await extract_artifact(
            incident_id="test_incident_6",
            title="Search for Flight MH370",
            anchor_entities={"Flight MH370"},
            entity_lookup={},
            llm_client=mock_llm,
        )

        assert len(result.artifact.referents) == 1
        ref = list(result.artifact.referents)[0]
        assert ref.role == ReferentRole.OBJECT


# =============================================================================
# Test 5: Ambiguous entities → InquirySeed + context
# =============================================================================

class TestAmbiguousExtraction:
    """Ambiguous entities should generate InquirySeed and go to context."""

    @pytest.mark.asyncio
    async def test_ambiguous_generates_inquiry(self):
        """AMBIGUOUS classification should create InquirySeed."""
        llm_response = {
            "classifications": [
                {
                    "entity_id": "The building",
                    "entity_name": "The building",
                    "classification": "AMBIGUOUS",
                    "confidence": 0.4,
                    "reasoning": "Unclear which building is referenced"
                },
            ],
            "overall_confidence": 0.5
        }
        mock_llm = make_mock_llm(llm_response)

        result = await extract_artifact(
            incident_id="test_incident_7",
            title="Fire at the building",
            anchor_entities={"The building"},
            entity_lookup={},
            llm_client=mock_llm,
        )

        # No referents (conservative)
        assert len(result.artifact.referents) == 0

        # In context (conservative)
        assert len(result.artifact.contexts) == 1

        # InquirySeed generated
        assert len(result.inquiries) == 1
        inquiry = list(result.inquiries)[0]
        assert inquiry.incident_id == "test_incident_7"

    @pytest.mark.asyncio
    async def test_low_confidence_generates_inquiry(self):
        """Low confidence (<0.6) should generate InquirySeed even if not AMBIGUOUS."""
        llm_response = {
            "classifications": [
                {
                    "entity_id": "Some location",
                    "entity_name": "Some location",
                    "classification": "SPECIFIC_PLACE",
                    "confidence": 0.4,  # Below threshold
                    "reasoning": "Might be specific but unsure"
                },
            ],
            "overall_confidence": 0.5
        }
        mock_llm = make_mock_llm(llm_response)

        result = await extract_artifact(
            incident_id="test_incident_8",
            title="Event at some location",
            anchor_entities={"Some location"},
            entity_lookup={},
            llm_client=mock_llm,
        )

        # Low confidence → context (conservative)
        assert len(result.artifact.referents) == 0
        assert len(result.artifact.contexts) == 1

        # InquirySeed generated
        assert len(result.inquiries) == 1


# =============================================================================
# Test 6: LLM failure → conservative result
# =============================================================================

class TestLLMFailure:
    """LLM failure should produce conservative result."""

    @pytest.mark.asyncio
    async def test_llm_error_conservative(self):
        """LLM error should put all entities in context with low confidence."""
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.side_effect = Exception("API Error")

        result = await extract_artifact(
            incident_id="test_incident_9",
            title="Some incident",
            anchor_entities={"Entity A", "Entity B"},
            entity_lookup={},
            llm_client=mock_llm,
        )

        # No referents
        assert len(result.artifact.referents) == 0

        # All in context
        assert len(result.artifact.contexts) == 2

        # Low confidence (triggers DEFER in membrane)
        assert result.artifact.confidence == 0.3


# =============================================================================
# Test 7: Entity ID resolution
# =============================================================================

class TestEntityIDResolution:
    """Entity IDs should be resolved from lookup or normalized."""

    def test_resolve_from_lookup(self):
        """Should use canonical ID from lookup."""
        lookup = {
            "en_123": {"canonical_name": "Wang Fuk Court", "name": "WFC"},
        }
        eid = _resolve_entity_id("Wang Fuk Court", {"Wang Fuk Court"}, lookup)
        assert eid == "en_123"

    def test_resolve_fallback(self):
        """Should use normalized name as fallback."""
        eid = _resolve_entity_id("Wang Fuk Court", {"Wang Fuk Court"}, {})
        assert eid == "name:wang_fuk_court"


# =============================================================================
# Test 8: Mixed entity types
# =============================================================================

class TestMixedExtraction:
    """Multiple entity types in one incident."""

    @pytest.mark.asyncio
    async def test_mixed_entities(self):
        """Incident with place, person, and location."""
        llm_response = {
            "classifications": [
                {
                    "entity_id": "Wang Fuk Court",
                    "entity_name": "Wang Fuk Court",
                    "classification": "SPECIFIC_PLACE",
                    "confidence": 0.95,
                    "reasoning": "Specific building"
                },
                {
                    "entity_id": "John Smith",
                    "entity_name": "John Smith",
                    "classification": "PERSON",
                    "confidence": 0.9,
                    "reasoning": "Named individual"
                },
                {
                    "entity_id": "Hong Kong",
                    "entity_name": "Hong Kong",
                    "classification": "BROAD_LOCATION",
                    "confidence": 0.95,
                    "reasoning": "City/region context"
                },
            ],
            "overall_confidence": 0.9
        }
        mock_llm = make_mock_llm(llm_response)

        result = await extract_artifact(
            incident_id="test_incident_10",
            title="John Smith at Wang Fuk Court, Hong Kong",
            anchor_entities={"Wang Fuk Court", "John Smith", "Hong Kong"},
            entity_lookup={},
            llm_client=mock_llm,
        )

        # 2 referents: place + person
        assert len(result.artifact.referents) == 2
        roles = {r.role for r in result.artifact.referents}
        assert ReferentRole.PLACE in roles
        assert ReferentRole.PERSON in roles

        # 1 context: Hong Kong
        assert len(result.artifact.contexts) == 1

        # No inquiries
        assert len(result.inquiries) == 0


# =============================================================================
# Test 9: Prompt hash for reproducibility
# =============================================================================

class TestReproducibility:
    """Extraction should include prompt hash for reproducibility."""

    @pytest.mark.asyncio
    async def test_prompt_hash_present(self):
        """Result should include prompt hash."""
        llm_response = {
            "classifications": [],
            "overall_confidence": 0.5
        }
        mock_llm = make_mock_llm(llm_response)

        result = await extract_artifact(
            incident_id="test_incident_11",
            title="Test",
            anchor_entities=set(),
            entity_lookup={},
            llm_client=mock_llm,
        )

        assert result.prompt_hash
        assert len(result.prompt_hash) == 16  # SHA256 truncated
        assert result.model_version == "gpt-4o-mini"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
