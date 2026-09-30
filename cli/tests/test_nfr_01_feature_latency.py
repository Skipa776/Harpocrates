"""NFR-01: featurizing candidates on huge minified lines stays within budget; short lines unchanged."""

import time

from Harpocrates.core.detector import _collect_text_findings, _prepare_ml_context_from_lines
from Harpocrates.ml.features import extract_features

PW = "Wint3r" + "#Sales9"


def _featurize(line):
    lines = [line]
    out = []
    for f in _collect_text_findings(line):
        if f.token:
            ctx = _prepare_ml_context_from_lines(f, lines)
            out.append((f.token, ctx, extract_features(f, ctx).to_array()))
    return out


def test_nfr_01_long_minified_line_featurizes_fast():
    filler = "var a=1;//x\n".replace("\n", "") * 12_000  # ~140k chars of minified JS with many //
    line = filler + f'var p="{PW}";' + filler
    start = time.perf_counter()
    rows = _featurize(line)
    assert any(tok == PW for tok, _c, _x in rows)
    assert time.perf_counter() - start < 1.0


def test_context_window_keeps_token_and_offsets():
    line = "x" * 5000 + f' p = "{PW}"; ' + "y" * 5000
    tok, ctx, _x = next(r for r in _featurize(line) if r[0] == PW)
    assert PW in ctx.line_content and len(ctx.line_content) < 1200
    tm = ctx.token_match
    assert ctx.line_content[tm.start:tm.end] == PW


def test_short_lines_unchanged_by_window():
    line = f'password = "{PW}"'
    (_tok, ctx, _x), = [r for r in _featurize(line) if r[0] == PW]
    assert ctx.line_content == line
