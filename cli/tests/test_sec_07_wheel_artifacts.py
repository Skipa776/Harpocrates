"""SEC-07: the wheel ships every model artifact named in the hash manifest.

Inspects the built wheel's file list with zipfile, so it does not depend on
installing into a venv or on the console-script path.
"""
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_PREFIX = "Harpocrates/ml/models/"


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the wheel into a throwaway directory (leaves dist/ alone)."""
    outdir = tmp_path_factory.mktemp("wheel")
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(outdir)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Wheel build failed:\n{result.stdout}\n{result.stderr}"
    wheels: List[Path] = sorted(outdir.glob("harpocrates-*.whl"))
    assert wheels, "No wheel produced"
    return wheels[-1]


@pytest.mark.integration
def test_sec_07_wheel_ships_every_manifest_artifact(built_wheel: Path) -> None:
    with zipfile.ZipFile(built_wheel) as wheel:
        names = set(wheel.namelist())
        manifest_name = MODELS_PREFIX + "onnx_model_hashes.json"
        assert manifest_name in names, "hash manifest missing from wheel"
        manifest = json.loads(wheel.read(manifest_name))

    assert manifest, "manifest in wheel is empty"
    missing = sorted(key for key in manifest if MODELS_PREFIX + key not in names)
    assert not missing, f"manifest artifacts missing from wheel: {missing}"
