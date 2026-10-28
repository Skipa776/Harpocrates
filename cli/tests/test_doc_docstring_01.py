import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_docstrings.py"
DOC = '"""Description: Do something.\nParameters: None\nOutput: Nothing.\n"""'


def run_checker(*paths, script=SCRIPT):
    assert script.is_file(), "The docstring checker must exist"
    return subprocess.run(
        [sys.executable, str(script), *map(str, paths)], capture_output=True, text=True
    )


@pytest.mark.parametrize("heading", ["Description:", "Parameters:", "Output:"])
def test_doc_docstring_01_missing_heading_fails(tmp_path, heading):
    module = tmp_path / "module.py"
    module.write_text(f"def example():\n    {DOC.replace(heading, 'Other:')}\n")
    result = run_checker(module)
    assert result.returncode == 1
    assert result.stdout == f"{module}:1: example: missing {heading}\n"


def test_doc_docstring_01_wrong_order_fails(tmp_path):
    module = tmp_path / "module.py"
    module.write_text(
        'async def example():\n    """Output: Nothing.\n'
        'Parameters: None\nDescription: Do something.\n"""\n'
    )
    result = run_checker(module)
    assert result.returncode == 1
    assert result.stdout == f"{module}:1: example: headings out of order\n"


def test_doc_docstring_01_compliant_passes(tmp_path):
    module = tmp_path / "module.py"
    module.write_text(
        f"class Example:\n    {DOC}\n"
        f"    def method(self):\n        {DOC}\n"
        f"async def example():\n    {DOC}\n"
    )
    result = run_checker(tmp_path)
    assert result.returncode == 0
    assert result.stdout == result.stderr == ""


def test_doc_docstring_01_tests_exempt(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "helper.py").write_text("def helper(): pass\n")
    module = tmp_path / "test_example.py"
    module.write_text("def example(): pass\n")
    for paths in [(tmp_path,), (tests / "helper.py", module)]:
        result = run_checker(*paths)
        assert result.returncode == 0
        assert result.stdout == result.stderr == ""


def test_doc_docstring_01_unparseable_file_fails(tmp_path):
    module = tmp_path / "module.py"
    module.write_text("def broken(\n")
    result = run_checker(module)
    assert result.returncode == 1
    assert result.stdout.startswith(f"{module}: parse error: ")
    assert result.stderr == ""


def test_doc_docstring_01_enforce_reads_pyproject(tmp_path):
    assert SCRIPT.is_file(), "The docstring checker must exist"
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    script = scripts / SCRIPT.name
    shutil.copyfile(SCRIPT, script)
    (tmp_path / "module.py").write_text("def example(): pass\n")
    config = tmp_path / "pyproject.toml"
    for contents in ["", "[tool.harpo.docstrings]\nenforce = []\n"]:
        config.write_text(contents)
        result = run_checker("--enforce", script=script)
        assert result.returncode == 0
        assert result.stdout == result.stderr == ""
    config.write_text('[tool.harpo.docstrings]\nenforce = [\n    "module.py",\n]\n')
    result = run_checker("--enforce", script=script)
    assert result.returncode == 1
    assert "example: missing Description:" in result.stdout
    assert len(result.stdout.splitlines()) == 3


def test_doc_docstring_01_nested_private_exempt(tmp_path):
    module = tmp_path / "module.py"
    module.write_text(
        f"def outer():\n    {DOC}\n"
        "    def _private(): pass\n"
        "    async def _async_private(): pass\n"
        "    def public(): pass\n"
        "def _top_level(): pass\n"
        f"class Example:\n    {DOC}\n"
        "    def _method(self): pass\n"
    )
    result = run_checker(module)
    assert result.returncode == 1
    assert "_private:" not in result.stdout
    assert "_async_private:" not in result.stdout
    for name in ["public", "_top_level", "_method"]:
        assert f"{name}: missing Description:" in result.stdout


def test_doc_docstring_01_every_package_module_enforced():
    """Every tracked package module that defines a function or class is on the enforce list."""
    import ast

    import tomllib

    root = Path(__file__).resolve().parents[2]
    tracked = subprocess.run(
        ["git", "ls-files", "cli/Harpocrates"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.split()
    defines = {
        p for p in tracked
        if p.endswith(".py")
        and any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                for n in ast.walk(ast.parse((root / p).read_text(encoding="utf-8"))))
    }
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    enforced = set(config["tool"]["harpo"]["docstrings"]["enforce"])
    assert defines <= enforced, f"not enforced: {sorted(defines - enforced)}"
