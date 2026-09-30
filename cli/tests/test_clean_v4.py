"""Cleaning the 40k synthetic set: dedupe, drop conflicting tokens, relabel non-secret positives."""

from scripts.clean_v4 import clean


def _rec(token, label, line=None, source="llm"):
    return {"token": token, "label": label, "line_content": line or f'x = "{token}"', "source": source}


def test_clean_rules():
    records = [
        _rec("Zq8vK2mN4pL7xR1t", 1),                 # kept as-is
        _rec("Zq8vK2mN4pL7xR1t", 1),                 # exact duplicate -> dropped
        _rec("from_secs", 1),                        # identifier labeled secret -> relabeled 0
        _rec("./deploy.sh", 1, source="script"),     # path labeled secret -> relabeled 0
        _rec("Ab3dE5gH7jK9mN1p", 1),                 # conflicting labels -> both dropped
        _rec("Ab3dE5gH7jK9mN1p", 0, line='y = "Ab3dE5gH7jK9mN1p"'),
    ]
    out, stats = clean(records)
    assert [(r["token"], r["label"]) for r in out] == [
        ("Zq8vK2mN4pL7xR1t", 1), ("from_secs", 0), ("./deploy.sh", 0)]
    relabeled = [r for r in out if r.get("relabeled")]
    assert {r["relabel_reason"] for r in relabeled} == {"identifier/word", "url/path"}
    assert all(r["original_label"] == 1 for r in relabeled)
    assert out[0]["source"] == "llm_synthetic" and out[2]["source"] == "script"
    assert stats == {"input": 6, "duplicates": 1, "conflicting_dropped": 2, "ambiguous_dropped": 0, "relabeled": 2, "output": 3}


def test_dotted_and_kebab_identifiers_are_relabeled():
    records = [_rec("signer.verifySignature", 1), _rec("com.mysql.cj.jdbc.Driver", 1),
               _rec("your-github-client-id", 1), _rec("oauth-client-secret", 1),
               _rec("z395jy7f4kRm9qwezXb", 1)]  # random value: stays a secret
    out, _ = clean(records)
    assert [(r["token"], r["label"]) for r in out] == [
        ("signer.verifySignature", 0), ("com.mysql.cj.jdbc.Driver", 0),
        ("your-github-client-id", 0), ("oauth-client-secret", 0), ("z395jy7f4kRm9qwezXb", 1)]


def test_jwt_is_not_mistaken_for_a_dotted_identifier():
    jwt = "eyJhbGciOiJIUzI1NiJ9" + "." + "eyJzdWIiOiJ1c2VyMTIzIiwiaWF0IjoxNzAwMDAwMDAwfQ" + "." + "Tq9xK2mN8pL4vR7wZ3bY"
    out, _ = clean([_rec(jwt, 1), _rec("app.config.loader", 1)])
    assert [(r["token"], r["label"]) for r in out] == [(jwt, 1), ("app.config.loader", 0)]


def test_round2_word_paths_and_calls_are_relabeled():
    records = [_rec("America/New_York", 1, line='"defaultTimeZone": "America/New_York",'),
               _rec("actions/checkout", 1, line="uses: actions/checkout@v2"),
               _rec("3306/orders_db", 1, line="url: jdbc:mysql://localhost:3306/orders_db"),
               _rec("sha256_file", 1, line="h = sha256_file(path)"),
               _rec("Zq8vK2mN/4pL7xR1tYw", 1)]  # random value with a slash: stays a secret
    out, _ = clean(records)
    assert [(r["token"], r["label"], r.get("relabel_reason")) for r in out] == [
        ("America/New_York", 0, "word path"), ("actions/checkout", 0, "word path"),
        ("3306/orders_db", 0, "word path"), ("sha256_file", 0, "function call"),
        ("Zq8vK2mN/4pL7xR1tYw", 1, None)]


def test_round2_hash_named_positives_are_dropped_not_relabeled():
    """A hex value in a variable named hash may be a content hash or an HMAC key: we can't tell."""
    hexval = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    out, stats = clean([_rec(hexval, 1, line=f'let content_hash = "{hexval}"'),
                        _rec(hexval[:40], 1, line=f'HMAC_KEY = "{hexval[:40]}"')])
    assert [r["token"] for r in out] == [hexval[:40]]
    assert stats["ambiguous_dropped"] == 1


def test_round2_hash_rule_looks_only_at_the_assigned_name():
    key = "Zq8vK2mN4pL7xR1tYw3e"
    out, stats = clean([_rec(key, 1, line=f'# verify digest first\nSIGNING_KEY = "{key}"')])
    assert stats["ambiguous_dropped"] == 0 and out[0]["label"] == 1
