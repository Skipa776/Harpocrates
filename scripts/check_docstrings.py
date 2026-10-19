"""Check the staged DOC-DOCSTRING-01 docstring convention."""

import argparse
import ast
from pathlib import Path

import tomllib  # ponytail: dev/CI script, needs Python 3.11+ (CI runs 3.11); the package itself still supports 3.10

HEADINGS = ("Description:", "Parameters:", "Output:")


def enforced_paths(root):
    """
    Description: Read the converted modules list so enforcement can grow incrementally.
    Parameters:
        root: Repository directory containing pyproject.toml.
    Output: List of paths; raises OSError or ValueError for unreadable or invalid configuration.
    """
    names = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    names = names.get("tool", {}).get("harpo", {}).get("docstrings", {}).get("enforce", [])
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("docstrings.enforce must be a list of paths")
    return [root / name for name in names]


def check_node(node, path, inside_function=False):
    """
    Description: Inspect definitions recursively and report missing or misplaced headings.
    Parameters:
        node: AST node to inspect, including its descendants.
        path: Source path displayed in diagnostics.
        inside_function: Whether an ancestor is a function, for private helper exemptions.
    Output: Diagnostic strings yielded without side effects or expected exceptions.
    """
    is_function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    if isinstance(node, ast.ClassDef) or is_function:
        exempt = is_function and inside_function and node.name.startswith("_")
        if not exempt:
            lines = (ast.get_docstring(node) or "").splitlines()
            positions = [
                next((i for i, line in enumerate(lines) if line.lstrip().startswith(heading)), -1)
                for heading in HEADINGS
            ]
            prefix = f"{path}:{node.lineno}: {node.name}: "
            for heading, position in zip(HEADINGS, positions):
                if position == -1:
                    yield prefix + f"missing {heading}"
            if -1 not in positions and positions != sorted(positions):
                yield prefix + "headings out of order"
    for child in ast.iter_child_nodes(node):
        yield from check_node(child, path, inside_function or is_function)


def main(argv=None):
    """
    Description: Check requested Python modules or the repository enforcement list.
    Parameters:
        argv: CLI arguments, or None to read the process arguments.
    Output: Exit status 0 or 1; prints violations and errors. Invalid CLI usage exits with status 2.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args(argv)
    if not args.paths and not args.enforce:
        parser.error("provide one or more paths or --enforce")
    paths = args.paths
    if args.enforce:
        try:
            paths += enforced_paths(Path(__file__).resolve().parent.parent)
        except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
            print(f"pyproject.toml: parse error: {error}")
            return 1
    files = set()
    for path in paths:
        files.update(path.rglob("*.py") if path.is_dir() else [path])
    failed = False
    for path in sorted(files):
        if "tests" in path.parts or path.match("test_*.py"):
            continue
        try:
            tree = ast.parse(path.read_bytes(), filename=str(path))
        except (OSError, SyntaxError, ValueError) as error:
            print(f"{path}: parse error: {getattr(error, 'msg', str(error))}")
            failed = True
            continue
        for violation in check_node(tree, path):
            print(violation)
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
