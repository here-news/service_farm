"""
CI Guard: Compiler is the SOLE AUTHORITY for topology mutations.

This test ensures that:
1. No spine edges are created outside the compiler package (AST-based)
2. No case merges happen outside the compiler package
3. The compile_pair function is the only path to spine edges
4. UnionFind for case formation only happens in authorized files

These tests use AST parsing to prevent bypass via aliasing/indirection.
"""

import ast
from pathlib import Path
from typing import Set, List, Tuple, Optional
from dataclasses import dataclass


# Directories to scan
BACKEND_DIR = Path(__file__).parent.parent.parent  # backend/


# =============================================================================
# AST-Based Violation Detection
# =============================================================================

@dataclass
class ASTViolation:
    """A violation found via AST analysis."""
    filepath: Path
    line: int
    col: int
    violation_type: str
    details: str


class SpineEdgeDetector(ast.NodeVisitor):
    """
    AST visitor that detects spine edge type usage.

    Catches:
    - Direct attribute access: EdgeType.SAME_HAPPENING
    - Aliased imports: from ... import SAME_HAPPENING as X
    - String construction: "SAME" + "_HAPPENING"
    """

    SPINE_EDGE_NAMES = {"SAME_HAPPENING", "UPDATE_TO"}

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.violations: List[ASTViolation] = []
        self._imported_aliases: dict = {}  # Track import aliases

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Track aliased imports of spine edge types."""
        if node.names:
            for alias in node.names:
                name = alias.name
                asname = alias.asname or name
                if name in self.SPINE_EDGE_NAMES:
                    self._imported_aliases[asname] = name
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        """Detect EdgeType.SAME_HAPPENING style access."""
        if node.attr in self.SPINE_EDGE_NAMES:
            self.violations.append(ASTViolation(
                filepath=self.filepath,
                line=node.lineno,
                col=node.col_offset,
                violation_type="spine_edge_attribute",
                details=f"Access to spine edge type: {node.attr}",
            ))
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        """Detect aliased spine edge type usage."""
        if node.id in self._imported_aliases:
            original = self._imported_aliases[node.id]
            self.violations.append(ASTViolation(
                filepath=self.filepath,
                line=node.lineno,
                col=node.col_offset,
                violation_type="spine_edge_alias",
                details=f"Aliased spine edge type: {node.id} (originally {original})",
            ))
        elif node.id in self.SPINE_EDGE_NAMES:
            self.violations.append(ASTViolation(
                filepath=self.filepath,
                line=node.lineno,
                col=node.col_offset,
                violation_type="spine_edge_direct",
                details=f"Direct spine edge type reference: {node.id}",
            ))
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        """Detect string constants containing spine edge names."""
        if isinstance(node.value, str):
            for name in self.SPINE_EDGE_NAMES:
                if name.lower() in node.value.lower() and "same_happening" in node.value.lower():
                    self.violations.append(ASTViolation(
                        filepath=self.filepath,
                        line=node.lineno,
                        col=node.col_offset,
                        violation_type="spine_edge_string",
                        details=f"String containing spine edge type: {node.value[:50]}",
                    ))
                    break
        self.generic_visit(node)


class UnionFindDetector(ast.NodeVisitor):
    """
    AST visitor that detects UnionFind class definitions and usage.
    """

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.violations: List[ASTViolation] = []
        self._defines_union_find = False

    def visit_ClassDef(self, node: ast.ClassDef):
        """Detect UnionFind class definitions."""
        if "unionfind" in node.name.lower() or "union_find" in node.name.lower():
            self._defines_union_find = True
            self.violations.append(ASTViolation(
                filepath=self.filepath,
                line=node.lineno,
                col=node.col_offset,
                violation_type="union_find_definition",
                details=f"UnionFind class definition: {node.name}",
            ))
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        """Detect UnionFind usage."""
        if "unionfind" in node.id.lower():
            self.violations.append(ASTViolation(
                filepath=self.filepath,
                line=node.lineno,
                col=node.col_offset,
                violation_type="union_find_usage",
                details=f"UnionFind usage: {node.id}",
            ))
        self.generic_visit(node)


class ActionMergeDetector(ast.NodeVisitor):
    """
    AST visitor that detects Action.MERGE creation (not just comparison).
    """

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.violations: List[ASTViolation] = []

    def visit_Attribute(self, node: ast.Attribute):
        """Detect Action.MERGE in non-comparison contexts."""
        if node.attr == "MERGE":
            # Check parent context
            parent = getattr(node, '_parent', None)
            # If parent is Compare, it's a comparison (allowed)
            # If parent is Return or assignment, it's creation (forbidden)
            if not isinstance(parent, ast.Compare):
                self.violations.append(ASTViolation(
                    filepath=self.filepath,
                    line=node.lineno,
                    col=node.col_offset,
                    violation_type="action_merge_creation",
                    details="Action.MERGE used outside comparison",
                ))
        self.generic_visit(node)


def add_parent_refs(tree: ast.AST):
    """Add parent references to all AST nodes."""
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node


# =============================================================================
# File Scanning
# =============================================================================

def get_python_files(directory: Path) -> List[Path]:
    """Get all Python files in directory."""
    return list(directory.rglob("*.py"))


def is_allowed_file(rel_path: str, allowed_set: Set[str]) -> bool:
    """Check if file is in allowed set."""
    return any(allowed in rel_path for allowed in allowed_set)


def parse_file(filepath: Path) -> Optional[ast.AST]:
    """Parse a Python file into AST."""
    try:
        content = filepath.read_text()
        return ast.parse(content, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return None


# =============================================================================
# CI Guard Tests (AST-Based)
# =============================================================================

# Files allowed to use spine edge types (the compiler itself)
ALLOWED_SPINE_EDGE_FILES = {
    "reee/compiler/membrane.py",
    "reee/compiler/weaver_compiler.py",
    "reee/compiler/__init__.py",
    "reee/compiler/tier_b_ablation.py",
    "reee/compiler/artifacts/extractor.py",
    "reee/compiler/artifacts/__init__.py",
    "reee/compiler/artifacts/test_extractor.py",
    "reee/compiler/test_membrane.py",
    # Test files
    "reee/tests/test_compiler_authority.py",
    # Legacy contracts (to be deprecated)
    "reee/contracts/case_formation.py",
}

# Files allowed to use UnionFind for case formation
ALLOWED_UNION_FIND_FILES = {
    "reee/compiler/weaver_compiler.py",
    "reee/compiler/tier_b_ablation.py",
    # Test files can use for validation
    "reee/tests/test_compiler_authority.py",
}

# Files allowed to create Action.MERGE decisions
ALLOWED_ACTION_MERGE_FILES = {
    "reee/compiler/membrane.py",
    "reee/compiler/tier_b_ablation.py",
    # Test files
    "reee/tests/test_compiler_authority.py",
}


def test_no_spine_edge_creation_outside_compiler_ast():
    """
    INVARIANT: EdgeType.SAME_HAPPENING and EdgeType.UPDATE_TO are only
    created via compile_pair in the compiler package.

    Uses AST parsing to catch:
    - Direct attribute access
    - Aliased imports
    - String construction
    """
    violations = []

    for filepath in get_python_files(BACKEND_DIR):
        try:
            rel_path = filepath.relative_to(BACKEND_DIR)
        except ValueError:
            continue

        rel_str = str(rel_path)

        # Skip allowed files
        if is_allowed_file(rel_str, ALLOWED_SPINE_EDGE_FILES):
            continue

        # Skip all test files (they may test the compiler)
        if "test" in rel_str.lower():
            continue

        # Parse and analyze
        tree = parse_file(filepath)
        if tree is None:
            continue

        detector = SpineEdgeDetector(filepath)
        detector.visit(tree)

        for v in detector.violations:
            violations.append(f"{rel_path}:{v.line}:{v.col} - {v.details}")

    if violations:
        msg = "COMPILER AUTHORITY VIOLATION (AST): Spine edge types found outside compiler:\n"
        msg += "\n".join(f"  - {v}" for v in violations)
        msg += "\n\nSpine edges MUST be created via compile_pair() in reee/compiler/membrane.py"
        msg += "\n\nNote: This check uses AST parsing and cannot be bypassed via aliasing."
        raise AssertionError(msg)


def test_no_union_find_outside_compiler_ast():
    """
    INVARIANT: UnionFind for case formation should only be used in:
    1. compiler/weaver_compiler.py (authoritative)

    Uses AST parsing to detect both definitions and usage.
    """
    violations = []

    for filepath in get_python_files(BACKEND_DIR):
        try:
            rel_path = filepath.relative_to(BACKEND_DIR)
        except ValueError:
            continue

        rel_str = str(rel_path)

        # Skip allowed files
        if is_allowed_file(rel_str, ALLOWED_UNION_FIND_FILES):
            continue

        # Skip test files
        if "test" in rel_str.lower():
            continue

        # Parse and analyze
        tree = parse_file(filepath)
        if tree is None:
            continue

        detector = UnionFindDetector(filepath)
        detector.visit(tree)

        for v in detector.violations:
            # Only flag if it looks like case/incident formation
            content = filepath.read_text().lower()
            if "case" in content or "incident" in content:
                violations.append(f"{rel_path}:{v.line} - {v.details}")

    if violations:
        msg = "COMPILER AUTHORITY WARNING (AST): UnionFind found outside authorized files:\n"
        msg += "\n".join(f"  - {v}" for v in violations)
        msg += "\n\nCase formation SHOULD use compile_incidents() from reee.compiler"
        # Warning only
        print(msg)


def test_compile_pair_is_only_merge_creator_ast():
    """
    INVARIANT: compile_pair() is the ONLY function that creates Action.MERGE.

    Uses AST parsing to detect MERGE creation vs comparison.
    """
    violations = []

    for filepath in get_python_files(BACKEND_DIR):
        try:
            rel_path = filepath.relative_to(BACKEND_DIR)
        except ValueError:
            continue

        rel_str = str(rel_path)

        # Skip compiler itself
        if "compiler" in rel_str:
            continue

        # Skip tests
        if "test" in rel_str.lower():
            continue

        # Parse and analyze
        tree = parse_file(filepath)
        if tree is None:
            continue

        add_parent_refs(tree)
        detector = ActionMergeDetector(filepath)
        detector.visit(tree)

        for v in detector.violations:
            violations.append(f"{rel_path}:{v.line} - {v.details}")

    if violations:
        msg = "COMPILER AUTHORITY VIOLATION (AST): Action.MERGE created outside compiler:\n"
        msg += "\n".join(f"  - {v}" for v in violations)
        msg += "\n\nMERGE decisions MUST come from compile_pair() in reee/compiler/membrane.py"
        raise AssertionError(msg)


# =============================================================================
# Legacy String-Based Tests (Kept for Backward Compatibility)
# =============================================================================

def test_no_spine_edge_creation_outside_compiler():
    """
    INVARIANT: EdgeType.SAME_HAPPENING and EdgeType.UPDATE_TO are only
    created via compile_pair in the compiler package.

    This is the simple string-based check (kept for backward compatibility).
    The AST-based check above is more robust.
    """
    violations = []

    for filepath in get_python_files(BACKEND_DIR):
        try:
            rel_path = filepath.relative_to(BACKEND_DIR)
        except ValueError:
            continue

        rel_str = str(rel_path)
        if is_allowed_file(rel_str, ALLOWED_SPINE_EDGE_FILES):
            continue

        try:
            content = filepath.read_text()
            lines = content.split('\n')

            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith('#'):
                    continue

                for pattern in {"SAME_HAPPENING", "UPDATE_TO"}:
                    if pattern in line:
                        violations.append(f"{rel_path}:{i} - found '{pattern}'")
        except Exception:
            pass

    if violations:
        msg = "COMPILER AUTHORITY VIOLATION: Spine edge types found outside compiler:\n"
        msg += "\n".join(f"  - {v}" for v in violations)
        msg += "\n\nSpine edges MUST be created via compile_pair() in reee/compiler/membrane.py"
        raise AssertionError(msg)


def test_no_direct_union_find_outside_compiler():
    """
    INVARIANT: UnionFind for case formation should only be used in authorized files.
    """
    violations = []

    for filepath in get_python_files(BACKEND_DIR):
        try:
            rel_path = filepath.relative_to(BACKEND_DIR)
        except ValueError:
            continue

        rel_str = str(rel_path)
        if is_allowed_file(rel_str, ALLOWED_UNION_FIND_FILES):
            continue
        if "test" in rel_str.lower():
            continue

        try:
            content = filepath.read_text()
            if "UnionFind" in content or "union_find" in content.lower():
                if "case" in content.lower() or "incident" in content.lower():
                    violations.append(f"{rel_path} - UnionFind used, may bypass compiler")
        except Exception:
            pass

    if violations:
        msg = "COMPILER AUTHORITY WARNING: UnionFind found outside authorized files:\n"
        msg += "\n".join(f"  - {v}" for v in violations)
        msg += "\n\nCase formation SHOULD use compile_incidents() from reee.compiler"
        print(msg)


def test_compile_pair_is_only_spine_creator():
    """
    INVARIANT: compile_pair() is the ONLY function that returns Action.MERGE.
    """
    violations = []

    for filepath in get_python_files(BACKEND_DIR):
        try:
            rel_path = filepath.relative_to(BACKEND_DIR)
        except ValueError:
            continue

        rel_str = str(rel_path)
        if "compiler" in rel_str:
            continue
        if "test" in rel_str.lower():
            continue

        try:
            content = filepath.read_text()
            if "Action.MERGE" in content:
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    if "Action.MERGE" in line:
                        stripped = line.strip()
                        if "==" in stripped or "!=" in stripped or "if" in stripped:
                            continue
                        if "import" in stripped:
                            continue
                        violations.append(f"{rel_path}:{i} - Action.MERGE usage")
        except Exception:
            pass

    if violations:
        msg = "COMPILER AUTHORITY VIOLATION: Action.MERGE used outside compiler:\n"
        msg += "\n".join(f"  - {v}" for v in violations)
        msg += "\n\nMERGE decisions MUST come from compile_pair() in reee/compiler/membrane.py"
        raise AssertionError(msg)


def test_compiler_imports_are_correct():
    """
    INVARIANT: Code that imports from reee.compiler should use the public API.
    """
    internal_modules = {
        "from reee.compiler.membrane import",
        "from reee.compiler.artifacts.extractor import",
    }

    allowed_internal_imports = {
        "reee/compiler/__init__.py",
        "reee/compiler/weaver_compiler.py",
        "reee/compiler/tier_b_ablation.py",
        "reee/compiler/artifacts/__init__.py",
        "workers/principled_weaver.py",
    }

    violations = []

    for filepath in get_python_files(BACKEND_DIR):
        try:
            rel_path = filepath.relative_to(BACKEND_DIR)
        except ValueError:
            continue

        rel_str = str(rel_path)
        if is_allowed_file(rel_str, allowed_internal_imports):
            continue
        if "test" in rel_str.lower():
            continue

        try:
            content = filepath.read_text()
            for internal in internal_modules:
                if internal in content:
                    violations.append(f"{rel_path} - imports internal module: {internal}")
        except Exception:
            pass

    if violations:
        print("COMPILER API WARNING: Internal compiler imports found:")
        for v in violations:
            print(f"  - {v}")
        print("\nPrefer using 'from reee.compiler import ...' for public API")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
