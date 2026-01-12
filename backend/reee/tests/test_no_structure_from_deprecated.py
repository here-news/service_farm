"""
Test: Kernel modules must not import from deprecated modules.

This test enforces the architectural boundary that core kernel modules
(types.py, membrane.py, typed_belief.py, builders/*) must not depend
on deprecated modules. This prevents deprecated code from "leaking back"
into the stable core.

KERNEL MODULES (must be clean):
- reee/types.py
- reee/membrane.py
- reee/typed_belief.py
- reee/builders/surface_builder.py
- reee/builders/event_builder.py
- reee/builders/story_builder.py

DEPRECATED MODULES (must not be imported by kernel):
- reee/engine.py
- reee/kernel.py
- reee/comparator.py
- reee/extractor.py
- reee/interpretation.py
- reee/aboutness/
- reee/builders/case_builder.py
- reee/deprecated/*
"""

import ast
import pytest
from pathlib import Path

# Paths
REEE_DIR = Path(__file__).parent.parent

# Kernel modules that must stay clean
KERNEL_MODULES = [
    REEE_DIR / "types.py",
    REEE_DIR / "membrane.py",
    REEE_DIR / "typed_belief.py",
    REEE_DIR / "builders" / "surface_builder.py",
    REEE_DIR / "builders" / "event_builder.py",
    REEE_DIR / "builders" / "story_builder.py",
]

# Deprecated module names (as they appear in imports)
DEPRECATED_MODULE_NAMES = {
    "reee.engine",
    "reee.kernel", 
    "reee.comparator",
    "reee.extractor",
    "reee.interpretation",
    "reee.aboutness",
    "reee.builders.case_builder",
    "reee.deprecated",
}

# Deprecated symbols that should not be imported
DEPRECATED_SYMBOLS = {
    # From engine.py
    "Engine", "EmergenceEngine",
    # From kernel.py
    "EpistemicKernel", "Belief",
    # From comparator.py
    "ClaimComparator", "ComparisonRelation",
    # From extractor.py
    "ClaimExtractor", "ExtractedClaim",
    # From interpretation.py
    "interpret_all", "interpret_surface", "interpret_event",
    # From aboutness/
    "AboutnessScorer", "compute_aboutness_edges", "compute_events_from_aboutness",
    # From case_builder.py
    "EntityCase", "PrincipledCaseBuilder",
}


def get_imports_from_file(filepath: Path, top_level_only: bool = True) -> list:
    """
    Extract import statements from a Python file.

    Args:
        filepath: Path to Python file
        top_level_only: If True, only return module-level imports (not inside functions/methods).
                       This allows deprecated imports inside deprecated methods (transitional pattern).
    """
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read(), filename=str(filepath))
    except SyntaxError:
        return []

    imports = []

    if top_level_only:
        # Only check module body, not inside functions/classes
        nodes_to_check = tree.body
    else:
        nodes_to_check = list(ast.walk(tree))

    for node in nodes_to_check:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    'type': 'import',
                    'module': alias.name,
                    'name': alias.asname or alias.name,
                    'lineno': node.lineno,
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                imports.append({
                    'type': 'from',
                    'module': module,
                    'name': alias.name,
                    'lineno': node.lineno,
                })
        elif isinstance(node, ast.ClassDef) and top_level_only:
            # Check class-level imports (but not method-level)
            for class_node in node.body:
                if isinstance(class_node, ast.Import):
                    for alias in class_node.names:
                        imports.append({
                            'type': 'import',
                            'module': alias.name,
                            'name': alias.asname or alias.name,
                            'lineno': class_node.lineno,
                        })
                elif isinstance(class_node, ast.ImportFrom):
                    module = class_node.module or ''
                    for alias in class_node.names:
                        imports.append({
                            'type': 'from',
                            'module': module,
                            'name': alias.name,
                            'lineno': class_node.lineno,
                        })

    return imports


def check_deprecated_imports(filepath: Path) -> list:
    """Check if a file imports from deprecated modules."""
    violations = []
    imports = get_imports_from_file(filepath)
    
    for imp in imports:
        module = imp['module']
        name = imp['name']
        
        # Check for direct deprecated module imports
        for deprecated in DEPRECATED_MODULE_NAMES:
            if module == deprecated or module.startswith(deprecated + "."):
                violations.append({
                    'file': filepath.name,
                    'line': imp['lineno'],
                    'issue': f"imports from deprecated module '{module}'",
                })
                break
        
        # Check for deprecated symbols (even from reee itself)
        if name in DEPRECATED_SYMBOLS:
            violations.append({
                'file': filepath.name,
                'line': imp['lineno'],
                'issue': f"imports deprecated symbol '{name}'",
            })
    
    return violations


class TestKernelNoDeprecatedImports:
    """Ensure kernel modules don't import from deprecated modules."""
    
    def test_types_no_deprecated(self):
        """types.py must not import from deprecated modules."""
        filepath = REEE_DIR / "types.py"
        if not filepath.exists():
            pytest.skip(f"{filepath} not found")
        
        violations = check_deprecated_imports(filepath)
        if violations:
            msg = "\n".join(f"  {v['file']}:{v['line']}: {v['issue']}" for v in violations)
            pytest.fail(f"types.py imports from deprecated modules:\n{msg}")
    
    def test_membrane_no_deprecated(self):
        """membrane.py must not import from deprecated modules."""
        filepath = REEE_DIR / "membrane.py"
        if not filepath.exists():
            pytest.skip(f"{filepath} not found")
        
        violations = check_deprecated_imports(filepath)
        if violations:
            msg = "\n".join(f"  {v['file']}:{v['line']}: {v['issue']}" for v in violations)
            pytest.fail(f"membrane.py imports from deprecated modules:\n{msg}")
    
    def test_typed_belief_no_deprecated(self):
        """typed_belief.py must not import from deprecated modules."""
        filepath = REEE_DIR / "typed_belief.py"
        if not filepath.exists():
            pytest.skip(f"{filepath} not found")
        
        violations = check_deprecated_imports(filepath)
        if violations:
            msg = "\n".join(f"  {v['file']}:{v['line']}: {v['issue']}" for v in violations)
            pytest.fail(f"typed_belief.py imports from deprecated modules:\n{msg}")
    
    def test_surface_builder_no_deprecated(self):
        """surface_builder.py must not import from deprecated modules."""
        filepath = REEE_DIR / "builders" / "surface_builder.py"
        if not filepath.exists():
            pytest.skip(f"{filepath} not found")
        
        violations = check_deprecated_imports(filepath)
        if violations:
            msg = "\n".join(f"  {v['file']}:{v['line']}: {v['issue']}" for v in violations)
            pytest.fail(f"surface_builder.py imports from deprecated modules:\n{msg}")
    
    def test_event_builder_no_deprecated(self):
        """event_builder.py must not import from deprecated modules."""
        filepath = REEE_DIR / "builders" / "event_builder.py"
        if not filepath.exists():
            pytest.skip(f"{filepath} not found")
        
        violations = check_deprecated_imports(filepath)
        if violations:
            msg = "\n".join(f"  {v['file']}:{v['line']}: {v['issue']}" for v in violations)
            pytest.fail(f"event_builder.py imports from deprecated modules:\n{msg}")
    
    def test_story_builder_no_deprecated(self):
        """story_builder.py must not import from deprecated modules."""
        filepath = REEE_DIR / "builders" / "story_builder.py"
        if not filepath.exists():
            pytest.skip(f"{filepath} not found")
        
        violations = check_deprecated_imports(filepath)
        if violations:
            msg = "\n".join(f"  {v['file']}:{v['line']}: {v['issue']}" for v in violations)
            pytest.fail(f"story_builder.py imports from deprecated modules:\n{msg}")
    
    def test_all_kernel_modules_exist(self):
        """Verify all kernel modules exist."""
        missing = []
        for path in KERNEL_MODULES:
            if not path.exists():
                missing.append(str(path.relative_to(REEE_DIR)))
        
        if missing:
            pytest.fail(f"Missing kernel modules: {missing}")


class TestDeprecatedModulesDocumented:
    """Ensure deprecated modules are properly documented."""
    
    def test_deprecated_modules_in_relic(self):
        """All deprecated modules should be listed in RELIC.md."""
        relic_path = REEE_DIR / "deprecated" / "RELIC.md"
        if not relic_path.exists():
            pytest.skip("RELIC.md not found")
        
        content = relic_path.read_text()
        
        # Check each deprecated module is mentioned
        expected_mentions = [
            "case_builder",
            "engine",
            "kernel", 
            "comparator",
            "extractor",
            "interpretation",
            "aboutness",
        ]
        
        missing = [m for m in expected_mentions if m not in content]
        if missing:
            pytest.fail(f"RELIC.md missing documentation for: {missing}")
