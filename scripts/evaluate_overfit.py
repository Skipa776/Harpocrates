#!/usr/bin/env python3
"""
Overfitting and latency evaluation for the v0.4 Harpocrates model.

Three evaluation tracks:
  1. Shortcut-feature adversarials  — probe file_is_test and is_hex_with_no_alpha_mix
  2. Distribution-shift stress test — inputs that look nothing like synthetic corpus
  3. End-to-end latency benchmark   — p50/p95/p99 per-file timing on detect_file_with_ml

Usage:
    python scripts/evaluate_overfit.py
    python scripts/evaluate_overfit.py --verbose
    python scripts/evaluate_overfit.py --latency-only
"""
from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Harpocrates.core.detector import detect_file_with_ml
from Harpocrates.ml.onnx_verifier import OnnxVerifier

# ---------------------------------------------------------------------------
# Adversarial sample definitions
# ---------------------------------------------------------------------------

@dataclass
class AdversarialCase:
    description: str
    code: str
    file_path: str          # controls in_test_file, file_extension_risk, etc.
    expect_secret: bool     # True → model must flag at least one finding
    shortcut_targeted: str  # which feature bias this tests


ADVERSARIAL_CASES: List[AdversarialCase] = [
    # --- file_is_test shortcut (model bias: test files → safe) ---
    AdversarialCase(
        description="Real AWS key hardcoded in test file (file_is_test=True should NOT suppress)",
        code='aws_access_key_id = "AKIAIOSFODNN7EXAMPLE"  # DO NOT COMMIT\n',
        file_path="tests/test_integration.py",
        expect_secret=True,
        shortcut_targeted="file_is_test",
    ),
    AdversarialCase(
        description="High-entropy API token in test fixture file",
        code='GITHUB_TOKEN = "xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3uW7yB2dF5hK8m"\n',
        file_path="tests/fixtures/config.py",
        expect_secret=True,
        shortcut_targeted="file_is_test",
    ),
    AdversarialCase(
        description="High-entropy payment key in test helper",
        code='stripe.api_key = "xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3uW7yB2"\n',
        file_path="tests/helpers/payment.py",
        expect_secret=True,
        shortcut_targeted="file_is_test",
    ),
    AdversarialCase(
        description="Non-secret in test file (should stay safe — correct behavior)",
        code='expected_status = "authenticated"\ndefault_host = "localhost"\n',
        file_path="tests/test_auth.py",
        expect_secret=False,
        shortcut_targeted="file_is_test",
    ),

    # --- is_hex_with_no_alpha_mix shortcut (model bias: pure hex → hash/safe) ---
    AdversarialCase(
        description="32-byte AES key as lowercase hex (pure hex, IS a real secret)",
        code='aes_key = "deadbeefcafebabe0123456789abcdef"\n',
        file_path="config/crypto.py",
        expect_secret=True,
        shortcut_targeted="is_hex_with_no_alpha_mix",
    ),
    AdversarialCase(
        description="Vault token hex-format (pure hex, IS a real secret)",
        code='VAULT_TOKEN = "00000000-0000-0000-0000-000000000000abcdef1234567890"\n',
        file_path="deploy/vault_config.py",
        expect_secret=True,
        shortcut_targeted="is_hex_with_no_alpha_mix",
    ),
    AdversarialCase(
        description="Git SHA (pure hex, is NOT a secret — model should leave it alone)",
        code='LAST_DEPLOY_SHA = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"\n',
        file_path="deploy/version.py",
        expect_secret=False,
        shortcut_targeted="is_hex_with_no_alpha_mix",
    ),
    AdversarialCase(
        description="Docker image digest (pure hex, NOT a secret)",
        code='IMAGE_DIGEST = "sha256:abc123def456abc123def456abc123def456abc123def456abc123def456abc1"\n',
        file_path="Dockerfile",
        expect_secret=False,
        shortcut_targeted="is_hex_with_no_alpha_mix",
    ),

    # --- Distribution shift: real-world patterns not in synthetic corpus ---
    AdversarialCase(
        description="Secret buried in multiline dict (not a test file)",
        code=(
            'config = {\n'
            '    "host": "db.prod.internal",\n'
            '    "port": 5432,\n'
            '    "password": "xK9#mP2$qL8vR3nT",\n'
            '    "database": "production",\n'
            '}\n'
        ),
        file_path="app/db_config.py",
        expect_secret=True,
        shortcut_targeted="distribution_shift",
    ),
    AdversarialCase(
        description="Secret in YAML-like assignment (no quotes, no equals sign pattern)",
        code='DATABASE_URL: postgres://admin:s3cur3P@ss!@db.internal:5432/prod\n',
        file_path="config/settings.yaml",
        expect_secret=True,
        shortcut_targeted="distribution_shift",
    ),
    AdversarialCase(
        description="Anthropic key in .env file",
        code='ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n',
        file_path=".env",
        expect_secret=True,
        shortcut_targeted="distribution_shift",
    ),
    AdversarialCase(
        description="Enum-style constant with secret-looking name but safe value",
        code='STATUS_AUTHENTICATED = "authenticated"\nROLE_ADMIN = "admin"\nPERM_READ = "read"\n',
        file_path="app/constants.py",
        expect_secret=False,
        shortcut_targeted="distribution_shift",
    ),
    AdversarialCase(
        description="File path assigned to a key-named variable (should NOT trigger)",
        code='ssl_key_path = "/etc/ssl/private/server.key"\ncert_path = "/etc/ssl/certs/server.crt"\n',
        file_path="app/tls_config.py",
        expect_secret=False,
        shortcut_targeted="distribution_shift",
    ),
    AdversarialCase(
        description="Host config with secret-ish variable names (should NOT trigger)",
        code='api_host = "0.0.0.0"\napi_port = 8443\nbind_address = "127.0.0.1"\n',
        file_path="app/server.py",
        expect_secret=False,
        shortcut_targeted="distribution_shift",
    ),

    # --- Commented-out secrets (detector.py Phase 3.6.1 behavior) ---
    AdversarialCase(
        description="AWS key commented out in production file (must still detect)",
        code='# aws_access_key_id = "AKIAIOSFODNN7EXAMPLE"  # old key, rotated\n',
        file_path="scripts/migrate.py",
        expect_secret=True,
        shortcut_targeted="commented_secret",
    ),
    AdversarialCase(
        description="GitHub PAT commented in config (must detect)",
        code='# GITHUB_TOKEN = "xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3uW7yB2dF5hK8m"\n',
        file_path="config/ci.py",
        expect_secret=True,
        shortcut_targeted="commented_secret",
    ),
]

# ---------------------------------------------------------------------------
# Latency benchmark files
# ---------------------------------------------------------------------------

_SMALL_FILE = """\
# Small config file — 20 lines
import os

DATABASE_URL = os.environ.get("DATABASE_URL")
SECRET_KEY = os.environ.get("SECRET_KEY")
DEBUG = False
ALLOWED_HOSTS = ["localhost", "127.0.0.1"]
STATIC_URL = "/static/"
MEDIA_URL = "/media/"
LOG_LEVEL = "INFO"
CACHE_TTL = 300
MAX_CONNECTIONS = 10
TIMEOUT = 30
RETRY_COUNT = 3
API_VERSION = "v2"
BASE_URL = "https://api.example.com"
RATE_LIMIT = 100
CORS_ORIGINS = ["https://example.com"]
FEATURE_FLAGS = {"new_ui": True, "beta": False}
"""

_MEDIUM_FILE = (_SMALL_FILE * 10) + """\
# Buried secret — makes the ML path fire
stripe_api_key = "xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3uW7yB2dF5hK8mN1pR4sU7v"
"""

_LARGE_FILE_CLEAN = _SMALL_FILE * 50  # ~1000 lines, no secrets

_LARGE_FILE_WITH_SECRET = _LARGE_FILE_CLEAN + """\
# Secret near end
OPENAI_API_KEY = "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
"""

# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    case: AdversarialCase
    found_secret: bool
    n_findings: int
    pass_: bool  # found_secret == expect_secret
    severities: List[str] = field(default_factory=list)
    error: Optional[str] = None


def _run_case(case: AdversarialCase, verifier: OnnxVerifier) -> CaseResult:
    """Write content to a temp file at the expected path, run detect_file_with_ml."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Recreate the directory structure implied by file_path
            full_path = Path(tmpdir) / case.file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(case.code, encoding="utf-8")

            findings = detect_file_with_ml(full_path, verifier)
            found = len(findings) > 0
            severities = [f.severity.value for f in findings]
            return CaseResult(
                case=case,
                found_secret=found,
                n_findings=len(findings),
                pass_=found == case.expect_secret,
                severities=severities,
            )
    except Exception as exc:
        return CaseResult(
            case=case,
            found_secret=False,
            n_findings=0,
            pass_=False,
            error=str(exc),
        )


def run_adversarial_suite(verifier: OnnxVerifier, verbose: bool = False) -> Tuple[int, int]:
    """Return (passed, total)."""
    results_by_shortcut: dict[str, List[CaseResult]] = {}

    for case in ADVERSARIAL_CASES:
        r = _run_case(case, verifier)
        results_by_shortcut.setdefault(case.shortcut_targeted, []).append(r)

    total = sum(len(v) for v in results_by_shortcut.values())
    passed = sum(r.pass_ for v in results_by_shortcut.values() for r in v)

    print("\n=== ADVERSARIAL EVALUATION ===")
    for shortcut, results in results_by_shortcut.items():
        sub_pass = sum(r.pass_ for r in results)
        print(f"\n  [{shortcut}]  {sub_pass}/{len(results)} passed")
        for r in results:
            icon = "✓" if r.pass_ else "✗"
            expected = "SECRET" if r.case.expect_secret else "SAFE"
            actual = f"found {r.n_findings}" if r.found_secret else "SAFE"
            sev_str = f" [{','.join(r.severities)}]" if r.severities else ""
            err_str = f"  ERROR: {r.error}" if r.error else ""
            print(f"    {icon} {r.case.description}")
            if verbose or not r.pass_:
                print(f"        expected={expected}  actual={actual}{sev_str}{err_str}")

    recall_cases = [r for v in results_by_shortcut.values()
                    for r in v if r.case.expect_secret]
    fp_cases = [r for v in results_by_shortcut.values()
                for r in v if not r.case.expect_secret]

    recall = sum(r.found_secret for r in recall_cases) / max(len(recall_cases), 1)
    fp_rate = sum(r.found_secret for r in fp_cases) / max(len(fp_cases), 1)

    print(f"\n  Adversarial recall (must-detect cases): {recall:.1%}  ({sum(r.found_secret for r in recall_cases)}/{len(recall_cases)})")
    print(f"  Adversarial FP rate (must-not-detect):  {fp_rate:.1%}  ({sum(r.found_secret for r in fp_cases)}/{len(fp_cases)})")
    print(f"\n  Overall: {passed}/{total} passed")

    return passed, total


# ---------------------------------------------------------------------------
# Train/val gap check (overfitting indicator)
# ---------------------------------------------------------------------------

def check_train_val_gap() -> None:
    """Compare synthetic test metrics (in-distribution) vs golden OOD."""
    import json
    config_path = Path("Harpocrates/ml/models/model_config.json")
    if not config_path.exists():
        print("\n[train/val gap] model_config.json not found — skipping")
        return

    with open(config_path) as f:
        content = f.read().replace("NaN", "null")  # JSON doesn't support NaN
    cfg = json.loads(content)

    syn = cfg.get("metrics", {})
    gold = cfg.get("golden_metrics", {})

    syn_recall = syn.get("recall", 0)
    gold_recall = gold.get("recall", 0)
    syn_prec = syn.get("precision", 0)
    gold_prec = gold.get("precision", 0)
    syn_f1 = syn.get("f1", 0)
    gold_f1 = gold.get("f1", 0)

    recall_gap = syn_recall - gold_recall
    prec_gap = syn_prec - gold_prec
    f1_gap = syn_f1 - gold_f1

    print("\n=== TRAIN/VAL GAP (overfitting indicator) ===")
    print(f"  {'Metric':<12} {'Synthetic':>10}  {'Golden OOD':>10}  {'Gap':>8}  {'Status':>8}")
    print(f"  {'-'*56}")

    def _flag(gap: float, warn_at: float = 0.03, fail_at: float = 0.07) -> str:
        if abs(gap) >= fail_at:
            return "⚠ LARGE"
        if abs(gap) >= warn_at:
            return "△ WARN"
        return "✓ OK"

    print(f"  {'Recall':<12} {syn_recall:>10.3f}  {gold_recall:>10.3f}  {recall_gap:>+8.3f}  {_flag(recall_gap):>8}")
    print(f"  {'Precision':<12} {syn_prec:>10.3f}  {gold_prec:>10.3f}  {prec_gap:>+8.3f}  {_flag(prec_gap):>8}")
    print(f"  {'F1':<12} {syn_f1:>10.3f}  {gold_f1:>10.3f}  {f1_gap:>+8.3f}  {_flag(f1_gap):>8}")

    print()
    if abs(recall_gap) >= 0.07 or abs(f1_gap) >= 0.07:
        print("  ⚠  Large gap detected. Model may be overfitting to synthetic distribution.")
        print("     Consider: more diverse negatives, dropout, lower max_depth, or real-world data.")
    elif abs(recall_gap) >= 0.03 or abs(f1_gap) >= 0.03:
        print("  △  Moderate gap. Acceptable for v0.4 but worth watching in v0.5.")
    else:
        print("  ✓  Gap within acceptable range. No strong overfitting signal from metrics alone.")

    print(f"\n  Feature importance concern: file_is_test={cfg['feature_importance_top10'].get('file_is_test', 0):.1%} "
          f"(rank #1 — shortcut bias risk)")


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

@dataclass
class LatencyResult:
    label: str
    n_lines: int
    has_secret: bool
    samples: List[float]  # wall-clock seconds per call

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.samples) * 1000

    @property
    def p95_ms(self) -> float:
        s = sorted(self.samples)
        return s[int(len(s) * 0.95)] * 1000

    @property
    def p99_ms(self) -> float:
        s = sorted(self.samples)
        return s[int(len(s) * 0.99)] * 1000

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.samples) * 1000


def _bench_file(label: str, content: str, verifier: OnnxVerifier, n_iter: int = 100) -> LatencyResult:
    has_secret = False
    samples: List[float] = []

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        tmp_path = Path(f.name)

    try:
        # Warm up — exclude from samples
        for _ in range(3):
            detect_file_with_ml(tmp_path, verifier)

        for _ in range(n_iter):
            t0 = time.perf_counter()
            findings = detect_file_with_ml(tmp_path, verifier)
            elapsed = time.perf_counter() - t0
            samples.append(elapsed)
            if findings:
                has_secret = True
    finally:
        tmp_path.unlink(missing_ok=True)

    return LatencyResult(
        label=label,
        n_lines=content.count("\n"),
        has_secret=has_secret,
        samples=samples,
    )


def run_latency_benchmark(verifier: OnnxVerifier, n_iter: int = 100) -> None:
    TARGET_MS = 2.0

    cases = [
        ("small (20 lines, clean)", _SMALL_FILE),
        ("medium (200 lines + 1 secret)", _MEDIUM_FILE),
        ("large (1000 lines, clean)", _LARGE_FILE_CLEAN),
        ("large (1000 lines + 1 secret at end)", _LARGE_FILE_WITH_SECRET),
    ]

    print(f"\n=== LATENCY BENCHMARK (n={n_iter} iterations per case, target ≤{TARGET_MS}ms) ===")
    print(f"  {'Case':<40} {'Lines':>6}  {'p50':>7}  {'p95':>7}  {'p99':>7}  {'mean':>7}  {'Status'}")
    print(f"  {'-'*85}")

    all_mean_ms: List[float] = []
    worst_p95_ms: float = 0.0

    for label, content in cases:
        result = _bench_file(label, content, verifier, n_iter=n_iter)
        all_mean_ms.append(result.mean_ms)
        worst_p95_ms = max(worst_p95_ms, result.p95_ms)

        status = "✓" if result.p95_ms <= TARGET_MS else "⚠ OVER"
        print(
            f"  {result.label:<40} {result.n_lines:>6}  "
            f"{result.p50_ms:>6.1f}ms  {result.p95_ms:>6.1f}ms  "
            f"{result.p99_ms:>6.1f}ms  {result.mean_ms:>6.1f}ms  {status}"
        )

    overall_mean = statistics.mean(all_mean_ms)
    print(f"\n  Overall mean across cases: {overall_mean:.1f}ms")
    print(f"  Worst p95 across cases:    {worst_p95_ms:.1f}ms")

    if worst_p95_ms > TARGET_MS:
        actual_rounded = round(worst_p95_ms)
        print(f"\n  ⚠  BUDGET EXCEEDED: worst p95 = {worst_p95_ms:.1f}ms > {TARGET_MS}ms target.")
        print(f"     Documentation update required: change '2ms' → '~{actual_rounded}ms' in:")
        print("       .claude/CLAUDE.md  (§Architectural Baselines)")
        print("       README.md          (performance section if present)")
    else:
        print(f"\n  ✓  All cases within {TARGET_MS}ms p95 budget.")

    return worst_p95_ms


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Harpocrates v0.4 overfitting + latency eval")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all case details")
    parser.add_argument("--latency-only", action="store_true", help="Skip adversarial suite")
    parser.add_argument("--adversarial-only", action="store_true", help="Skip latency benchmark")
    parser.add_argument("--iterations", type=int, default=100, help="Latency benchmark iterations")
    args = parser.parse_args()

    print("Loading ONNX verifier …")
    verifier = OnnxVerifier(lazy_load=False)

    check_train_val_gap()

    adv_passed = adv_total = 0
    if not args.latency_only:
        adv_passed, adv_total = run_adversarial_suite(verifier, verbose=args.verbose)

    if not args.adversarial_only:
        worst_p95 = run_latency_benchmark(verifier, n_iter=args.iterations)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if not args.latency_only:
        adv_pct = adv_passed / max(adv_total, 1) * 100
        print(f"  Adversarial suite:  {adv_passed}/{adv_total} ({adv_pct:.0f}%)")
    if not args.adversarial_only:
        budget_status = "✓ within budget" if worst_p95 <= 2.0 else f"⚠ OVER budget ({worst_p95:.1f}ms p95)"
        print(f"  Latency budget:     {budget_status}")
    print()


if __name__ == "__main__":
    main()
