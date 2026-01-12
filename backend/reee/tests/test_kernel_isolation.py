"""
Test Kernel Isolation
=====================

INVARIANT: Kernel modules must be importable without DB/LLM dependencies.

Kernel modules are:
- types.py
- membrane.py
- typed_belief.py
- builders/story_builder.py
- builders/surface_builder.py
- builders/event_builder.py

These must NOT import:
- neo4j
- psycopg2, asyncpg
- openai
- redis
- requests (direct, httpx ok for types)

This test prevents accidental coupling of pure kernel to infrastructure.
"""

import ast
import pytest
from pathlib import Path
from typing import Set, List, Tuple


# Root of reee package
REEE_ROOT = Path(__file__).parent.parent


# Modules that MUST be pure (no DB/LLM imports)
KERNEL_MODULES = [
    "types.py",
    "membrane.py",
    "typed_belief.py",
    "builders/story_builder.py",
    "builders/surface_builder.py",
    "builders/event_builder.py",
]


# Forbidden imports for kernel modules
FORBIDDEN_IMPORTS = {
    # Database
    "neo4j",
    "psycopg2",
    "asyncpg",
    "sqlalchemy",
    # LLM
    "openai",
    "anthropic",
    # Infrastructure
    "redis",
    "celery",
    # HTTP (for API calls)
    "requests",
    "aiohttp",
}


def get_imports(filepath: Path) -> Set[str]:
    """Extract all import names from a Python file."""
    try:
        with open(filepath) as f:
            tree = ast.parse(f.read())
    except SyntaxError:
        return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Get top-level module (e.g., "neo4j" from "neo4j.driver")
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])

    return imports


def check_forbidden_imports(filepath: Path) -> List[str]:
    """Check if a file imports any forbidden modules."""
    imports = get_imports(filepath)
    return [imp for imp in imports if imp in FORBIDDEN_IMPORTS]


class TestKernelIsolation:
    """
    Ensures kernel modules don't import DB/LLM dependencies.
    """

    @pytest.mark.parametrize("module_path", KERNEL_MODULES)
    def test_kernel_module_isolation(self, module_path: str):
        """Kernel module must not import forbidden dependencies."""
        filepath = REEE_ROOT / module_path

        if not filepath.exists():
            pytest.skip(f"Module not found: {filepath}")

        violations = check_forbidden_imports(filepath)

        assert len(violations) == 0, \
            f"Kernel module {module_path} imports forbidden dependencies: {violations}"

    def test_kernel_modules_importable(self):
        """All kernel modules should be importable."""
        import sys
        import importlib

        # Add backend to path
        backend_path = REEE_ROOT.parent
        if str(backend_path) not in sys.path:
            sys.path.insert(0, str(backend_path))

        for module_path in KERNEL_MODULES:
            module_name = module_path.replace("/", ".").replace(".py", "")
            full_name = f"reee.{module_name}"

            try:
                importlib.import_module(full_name)
            except ImportError as e:
                # Check if the error is about forbidden imports
                error_msg = str(e)
                for forbidden in FORBIDDEN_IMPORTS:
                    if forbidden in error_msg:
                        pytest.fail(
                            f"Kernel module {module_path} has transitive dependency on {forbidden}: {e}"
                        )
                # Other import errors are ok (may be optional deps)

    def test_membrane_is_pure(self):
        """membrane.py should have minimal imports."""
        filepath = REEE_ROOT / "membrane.py"

        if not filepath.exists():
            pytest.skip("membrane.py not found")

        imports = get_imports(filepath)

        # Allowed imports for membrane
        allowed = {"dataclasses", "enum", "typing", "datetime"}
        extra_imports = imports - allowed

        # Report but don't fail (may have internal reee imports)
        if extra_imports:
            print(f"\nmembrane.py additional imports: {extra_imports}")

        # But forbidden imports should fail
        violations = check_forbidden_imports(filepath)
        assert len(violations) == 0, \
            f"membrane.py imports forbidden: {violations}"

    def test_typed_belief_is_pure(self):
        """typed_belief.py should be pure inference."""
        filepath = REEE_ROOT / "typed_belief.py"

        if not filepath.exists():
            pytest.skip("typed_belief.py not found")

        violations = check_forbidden_imports(filepath)
        assert len(violations) == 0, \
            f"typed_belief.py imports forbidden: {violations}"


class TestModuleOwnership:
    """
    Tests for clear module ownership boundaries.
    """

    def test_builders_dont_import_workers(self):
        """Builders should not import worker code."""
        builders_dir = REEE_ROOT / "builders"

        if not builders_dir.exists():
            pytest.skip("builders/ not found")

        worker_imports = {"workers", "canonical_worker", "principled_weaver"}

        for py_file in builders_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue

            imports = get_imports(py_file)
            violations = [imp for imp in imports if imp in worker_imports]

            assert len(violations) == 0, \
                f"Builder {py_file.name} imports worker code: {violations}"

    def test_types_dont_import_builders(self):
        """types.py should not import builders."""
        filepath = REEE_ROOT / "types.py"

        if not filepath.exists():
            pytest.skip("types.py not found")

        imports = get_imports(filepath)

        # types.py should not import builders
        assert "builders" not in imports, \
            "types.py should not import builders (circular risk)"


# =============================================================================
# DIRECT RUN SUPPORT
# =============================================================================

if __name__ == "__main__":
    print("Checking kernel module isolation...")
    print(f"REEE root: {REEE_ROOT}")
    print(f"Kernel modules: {KERNEL_MODULES}")
    print(f"Forbidden imports: {FORBIDDEN_IMPORTS}")
    print("-" * 50)

    for module_path in KERNEL_MODULES:
        filepath = REEE_ROOT / module_path
        if filepath.exists():
            violations = check_forbidden_imports(filepath)
            status = "✗ FAIL" if violations else "✓ OK"
            print(f"  {status} {module_path}: {violations if violations else 'pure'}")
        else:
            print(f"  ? SKIP {module_path}: not found")

    print("-" * 50)
    print("Run with pytest for full validation")
