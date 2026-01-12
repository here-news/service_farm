"""
Test No Deprecated Imports
==========================

INVARIANT: Workers must not import deprecated REEE symbols.

This prevents new dependencies from forming on legacy code
scheduled for removal.

Deprecated symbols (DO NOT USE):
- Engine, EmergenceEngine
- AboutnessScorer, compute_aboutness_edges
- ClaimExtractor, ClaimComparator
- EpistemicKernel, Belief
- interpret_all, interpret_surface, interpret_event
- PrincipledCaseBuilder (use StoryBuilder)
- EntityCase (use CompleteStory)
"""

import ast
import pytest
from pathlib import Path
from typing import Set, List


# Root paths
BACKEND_ROOT = Path(__file__).parent.parent.parent
WORKERS_DIR = BACKEND_ROOT / "workers"


# Deprecated symbols - workers must NOT import these
DEPRECATED_SYMBOLS = {
    # Engine (use builders)
    "Engine",
    "EmergenceEngine",

    # Aboutness (use PrincipledSurfaceBuilder)
    "AboutnessScorer",
    "compute_aboutness_edges",
    "compute_events_from_aboutness",

    # Extractors (legacy)
    "ClaimExtractor",
    "ExtractedClaim",
    "ClaimComparator",

    # Old belief kernel (use TypedBeliefState)
    "EpistemicKernel",
    "Belief",

    # Legacy interpretation
    "interpret_all",
    "interpret_surface",
    "interpret_event",

    # Old case builder (use StoryBuilder)
    "PrincipledCaseBuilder",
    "CaseBuilderResult",
    "EntityCase",
}


# Deprecated module paths
DEPRECATED_MODULES = {
    "reee.engine",
    "reee.aboutness",
    "reee.extractor",
    "reee.comparator",
    "reee.kernel",
    "reee.interpretation",
    "reee.builders.case_builder",
}


def get_reee_imports(filepath: Path) -> tuple[Set[str], Set[str]]:
    """
    Extract REEE imports from a Python file.

    Returns:
        (symbols, modules) - set of symbol names and module paths imported
    """
    try:
        with open(filepath) as f:
            tree = ast.parse(f.read())
    except SyntaxError:
        return set(), set()

    symbols = set()
    modules = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("reee"):
                modules.add(node.module)
                for alias in node.names:
                    symbols.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("reee"):
                    modules.add(alias.name)

    return symbols, modules


def check_deprecated_usage(filepath: Path) -> List[str]:
    """Check if a file uses deprecated REEE symbols."""
    symbols, modules = get_reee_imports(filepath)

    violations = []

    # Check for deprecated symbols
    for sym in symbols & DEPRECATED_SYMBOLS:
        violations.append(f"imports deprecated symbol: {sym}")

    # Check for deprecated module paths
    for mod in modules & DEPRECATED_MODULES:
        violations.append(f"imports deprecated module: {mod}")

    return violations


class TestNoDeprecatedImports:
    """
    Ensures workers don't import deprecated REEE symbols.
    """

    def test_workers_no_deprecated_imports(self):
        """No worker should import deprecated REEE symbols."""
        if not WORKERS_DIR.exists():
            pytest.skip("workers/ directory not found")

        all_violations = {}

        for py_file in WORKERS_DIR.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            violations = check_deprecated_usage(py_file)
            if violations:
                all_violations[py_file.name] = violations

        if all_violations:
            msg = "Workers importing deprecated REEE symbols:\n"
            for filename, viols in all_violations.items():
                msg += f"\n  {filename}:\n"
                for v in viols:
                    msg += f"    - {v}\n"
            pytest.fail(msg)

    def test_canonical_worker_clean(self):
        """canonical_worker.py should not use deprecated imports."""
        filepath = WORKERS_DIR / "canonical_worker.py"

        if not filepath.exists():
            pytest.skip("canonical_worker.py not found")

        violations = check_deprecated_usage(filepath)

        # No transitional imports should remain after migration to StoryBuilder
        if violations:
            pytest.fail(f"canonical_worker.py has deprecated imports: {violations}")


class TestDeprecationEnforcement:
    """
    Tests that deprecation boundaries are enforced.
    """

    def test_deprecated_symbols_documented(self):
        """All deprecated symbols should be listed."""
        # Just verify the set is non-empty and reasonable
        assert len(DEPRECATED_SYMBOLS) >= 10, \
            "DEPRECATED_SYMBOLS should list all deprecated exports"

    def test_deprecated_modules_documented(self):
        """All deprecated modules should be listed."""
        assert len(DEPRECATED_MODULES) >= 5, \
            "DEPRECATED_MODULES should list all deprecated module paths"


# =============================================================================
# DIRECT RUN SUPPORT
# =============================================================================

if __name__ == "__main__":
    print("Checking for deprecated REEE imports in workers...")
    print(f"Workers dir: {WORKERS_DIR}")
    print(f"Deprecated symbols: {len(DEPRECATED_SYMBOLS)}")
    print("-" * 50)

    if WORKERS_DIR.exists():
        for py_file in WORKERS_DIR.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            violations = check_deprecated_usage(py_file)
            status = "✗ FAIL" if violations else "✓ OK"
            print(f"  {status} {py_file.name}")
            for v in violations:
                print(f"       - {v}")
    else:
        print("  workers/ not found")

    print("-" * 50)
    print("Run with pytest for full validation")
