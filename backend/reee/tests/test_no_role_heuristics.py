"""
Test: No Role Heuristics in Production Code

This test ensures that principled_weaver.py contains NO heuristic-based
role detection patterns. Role labeling must come from:
1. LLM artifacts (via label_incidents_roles_batch)
2. Persisted artifacts (future: from Neo4j)

If this test fails, you are trying to ship heuristics into production.
Move heuristics to relational_experiment.py for baselines only.
"""

import re
from pathlib import Path


# Patterns that indicate heuristic role detection
FORBIDDEN_HEURISTIC_PATTERNS = [
    # Keyword-based role detection
    r"any\s*\(\s*kw\s+in\s+ent",  # any(kw in ent for kw in [...])
    r"for\s+kw\s+in\s+\[.*\].*if\s+kw\s+in",  # for kw in ['court', ...] if kw in

    # Hard-coded geo/facility lists for role assignment
    r"'court'.*'building'.*'tower'",  # facility keyword list
    r"'hong kong'.*'china'.*'taiwan'",  # country keyword list
    r"'district'.*'province'.*'region'",  # administrative keyword list

    # Direct role assignment based on string matching
    r"EntityRole\.\w+\s*#.*heuristic",  # Comment indicating heuristic
    r"role\s*=\s*EntityRole\.\w+.*if.*in\s+ent",  # role = EntityRole.X if "..." in ent
]

# Allowed exception: CaseRoleMode.LEGACY_HEURISTIC for emergency debugging
# This enum value is allowed to exist, but NOT to be used as default


def test_no_heuristic_role_detection_in_principled_weaver():
    """
    Verify principled_weaver.py contains NO heuristic role detection.

    If you're seeing this failure, you added keyword-based role detection.
    Remove it and use LLM artifacts instead (label_incidents_roles_batch).
    """
    weaver_path = Path(__file__).parent.parent.parent / "workers" / "principled_weaver.py"

    assert weaver_path.exists(), f"principled_weaver.py not found at {weaver_path}"

    content = weaver_path.read_text()

    # Remove comments and docstrings for cleaner pattern matching
    # (we want to catch actual code, not documentation)

    violations = []
    for pattern in FORBIDDEN_HEURISTIC_PATTERNS:
        matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
        if matches:
            violations.append((pattern, matches))

    if violations:
        msg = "HEURISTIC ROLE DETECTION FOUND in principled_weaver.py:\n"
        for pattern, matches in violations:
            msg += f"\n  Pattern: {pattern}\n  Matches: {matches[:3]}\n"
        msg += "\nFix: Remove heuristics. Use LLM artifacts only (label_incidents_roles_batch)."
        msg += "\nIf you need baselines, put them in relational_experiment.py."

        assert False, msg


def test_artifact_only_is_default_mode():
    """
    Verify that ARTIFACT_ONLY is the default mode, not LEGACY_HEURISTIC.
    """
    weaver_path = Path(__file__).parent.parent.parent / "workers" / "principled_weaver.py"
    content = weaver_path.read_text()

    # Check that CaseRoleMode exists
    assert "class CaseRoleMode" in content, "CaseRoleMode enum missing"
    assert "ARTIFACT_ONLY" in content, "ARTIFACT_ONLY mode missing"
    assert "LEGACY_HEURISTIC" in content, "LEGACY_HEURISTIC mode missing"

    # Check that default is ARTIFACT_ONLY, not LEGACY_HEURISTIC
    # This catches: mode: CaseRoleMode = CaseRoleMode.LEGACY_HEURISTIC
    if "mode: CaseRoleMode = CaseRoleMode.LEGACY_HEURISTIC" in content:
        assert False, "Default mode is LEGACY_HEURISTIC - must be ARTIFACT_ONLY"
    if "CaseRoleMode.LEGACY_HEURISTIC  # default" in content.lower():
        assert False, "LEGACY_HEURISTIC marked as default - must be ARTIFACT_ONLY"


def test_no_llm_degrades_to_empty_artifacts():
    """
    Verify that when LLM is unavailable, build_role_artifacts returns empty dict.

    This ensures incidents without artifacts cannot contribute spine edges,
    implementing the DEFER degradation mode.
    """
    weaver_path = Path(__file__).parent.parent.parent / "workers" / "principled_weaver.py"
    content = weaver_path.read_text()

    # Check that build_role_artifacts has the correct fallback
    if "if not llm_client:" not in content:
        assert False, "build_role_artifacts missing LLM client check"

    # The function should return {} when no LLM
    # This is the DEFER behavior
    assert "return {}" in content, "Missing empty dict return for no-LLM case"


if __name__ == "__main__":
    test_no_heuristic_role_detection_in_principled_weaver()
    test_artifact_only_is_default_mode()
    test_no_llm_degrades_to_empty_artifacts()
    print("All heuristic-blocking tests passed!")
