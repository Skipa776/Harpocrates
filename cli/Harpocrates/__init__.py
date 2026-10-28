from __future__ import annotations

import os

# Local-first promise: no network calls. onnxruntime >= 1.29 creates a device id under
# ~/Library/Application Support/Microsoft/DeveloperTools/.onnxruntime at import time and
# uploads usage events to Microsoft (default on from 1.31; see onnxruntime docs/Privacy.md).
# ORT_DISABLE_TELEMETRY must be set BEFORE onnxruntime is first imported, and is forced
# (not setdefault) so a user-exported 0 cannot re-enable it. Limit: this does nothing if a
# host process imported onnxruntime before Harpo.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

__all__ = ["__version__"]

__version__ = "1.4.0"
