"""
Invariant test: the default scan path must never import XAI dependencies.

Two complementary layers:
  1. Subprocess isolation — fresh interpreter guarantees clean sys.modules
     regardless of pytest test-ordering or xdist parallelism.
  2. AST static analysis — catches lazy/conditional imports that the runtime
     test would miss if the code path isn't exercised during the test run.

Covered modules (the "default path"):
  Harpocrates.core.detector  — scan logic
  Harpocrates.core.scanner   — file-tree orchestration
  Harpocrates.mcp.server     — MCP tool surface

Forbidden imports (anywhere except Harpocrates/ml/explain.py):
  xgboost                  — XAI TreeSHAP dependency
  Harpocrates.ml.explain   — the opt-in XAI module itself
  shap                     — banned everywhere per CLAUDE.md
  lime                     — banned everywhere per CLAUDE.md
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_FORBIDDEN_MODULES = {"xgboost", "Harpocrates.ml.explain", "shap", "lime"}

# Directories whose source files must never import forbidden modules.
# explain.py itself is excluded — it IS allowed to lazily import xgboost.
_HOT_PATH_DIRS = [
    _REPO_ROOT / "Harpocrates" / "core",
    _REPO_ROOT / "Harpocrates" / "mcp",
]
_EXPLAIN_PATH = _REPO_ROOT / "Harpocrates" / "ml" / "explain.py"


# ---------------------------------------------------------------------------
# Layer 1: subprocess runtime isolation
# ---------------------------------------------------------------------------

def test_default_scan_path_does_not_import_xai_runtime() -> None:
    """
    Spawn a fresh Python interpreter — guaranteed clean sys.modules.

    Using sys.modules snapshot diffing within a pytest session is unreliable:
    if a prior test loaded xgboost, it would already be in sys.modules before
    the snapshot is taken, and the diff would miss the leak.
    """
    check_code = (
        "import sys; "
        "import Harpocrates.core.detector; "
        "import Harpocrates.core.scanner; "
        "import Harpocrates.mcp.server; "
        "forbidden = {'xgboost', 'shap', 'lime', 'Harpocrates.ml.explain'}; "
        "leaked = forbidden & set(sys.modules); "
        "assert not leaked, f'Hot path leaked XAI imports: {leaked}'"
    )
    result = subprocess.run(
        [sys.executable, "-c", check_code],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"Hot path imported XAI deps in a fresh interpreter.\n"
        f"stderr: {result.stderr.strip()}\n"
        f"stdout: {result.stdout.strip()}"
    )


# ---------------------------------------------------------------------------
# Layer 2: AST static analysis
# ---------------------------------------------------------------------------

def _forbidden_imports_in_file(path: Path) -> list[str]:
    """
    Return list of forbidden import lines found in `path`.

    Ignores imports inside `if TYPE_CHECKING:` blocks — those are
    type-annotation only and never executed at runtime.
    """
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []

    violations: list[str] = []

    for node in ast.walk(tree):
        # Skip the body of `if TYPE_CHECKING:` guards.
        if isinstance(node, ast.If):
            test = node.test
            is_type_checking = (
                (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
                or (
                    isinstance(test, ast.Attribute)
                    and test.attr == "TYPE_CHECKING"
                )
            )
            if is_type_checking:
                continue  # don't descend into TYPE_CHECKING blocks

        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                full = alias.name
                if root in _FORBIDDEN_MODULES or full in _FORBIDDEN_MODULES:
                    violations.append(
                        f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: "
                        f"import {alias.name}"
                    )

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            if root in _FORBIDDEN_MODULES or module in _FORBIDDEN_MODULES:
                violations.append(
                    f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: "
                    f"from {module} import ..."
                )

    return violations


def test_default_scan_path_does_not_import_xai_static() -> None:
    """
    AST walk of hot-path source files — catches lazy/conditional imports
    that the subprocess test might miss if the code path isn't exercised.
    """
    all_violations: list[str] = []

    for directory in _HOT_PATH_DIRS:
        for py_file in sorted(directory.rglob("*.py")):
            if py_file == _EXPLAIN_PATH:
                continue  # explain.py is the one allowed place
            violations = _forbidden_imports_in_file(py_file)
            all_violations.extend(violations)

    assert not all_violations, (
        "Forbidden XAI imports found in hot-path source files:\n"
        + "\n".join(f"  {v}" for v in all_violations)
        + "\nXAI imports are only permitted inside Harpocrates/ml/explain.py."
    )
