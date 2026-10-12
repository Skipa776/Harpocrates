"""Local-first promise (design.md, Goals): no telemetry, no network calls.

ONNX Runtime >= 1.29 writes a device id under
~/Library/Application Support/Microsoft/DeveloperTools/.onnxruntime at import and
uploads usage events to Microsoft. Harpocrates must switch that off before the
first onnxruntime import. ORT state is per-process, so each case runs in a subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_CODE = textwrap.dedent(
    """
    import os
    import Harpocrates
    from Harpocrates.ml.onnx_verifier import OnnxVerifier

    v = OnnxVerifier(lazy_load=False)  # creates an InferenceSession
    assert v._session is not None
    assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"
    """
)


@pytest.mark.parametrize("preset", [None, "0"])
def test_sec_local_first_onnxruntime_telemetry_disabled(tmp_path: Path, preset):
    pytest.importorskip("onnxruntime")
    env = {k: v for k, v in os.environ.items() if k != "ORT_DISABLE_TELEMETRY"}
    env["HOME"] = str(tmp_path)
    if preset is not None:
        env["ORT_DISABLE_TELEMETRY"] = preset

    result = subprocess.run(
        [sys.executable, "-c", _CODE], env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    leaked = [p for p in tmp_path.rglob(".onnxruntime")]
    assert leaked == [], f"onnxruntime telemetry state created: {leaked}"
