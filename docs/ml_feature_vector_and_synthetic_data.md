# Harpocrates ML Feature Vector and Synthetic Data Generator — State Report

**As of commit `aeafc0c` (Phase 7.0 complete). Model not yet retrained; the shipped ONNX was trained on the 65-feature vector. This report describes the 67-feature vector that the next retrain will consume.**

---

## Part 1: The 67-Feature Vector

The feature vector is defined in `Harpocrates/ml/features.py` as `FEATURE_NAMES: Tuple[str, ...]` (the canonical ordering) and materialized by `FeatureVector.to_array()`. The two must stay in sync; `get_feature_names()` is implemented as `return list(FEATURE_NAMES)` to guarantee they cannot drift.

Features are organized into seven groups.

---

### Group 1 — Token Features (20 features)

Properties derived from the token string itself, independent of context.

| Feature | Type | What it measures |
|---|---|---|
| `token_length` | int | Raw character count |
| `token_entropy` | float | Shannon entropy of the token |
| `char_class_count` | int | Count of distinct character classes present: lowercase, uppercase, digit, special (0–4) |
| `digit_ratio` | float | Fraction of characters that are digits |
| `uppercase_ratio` | float | Fraction of characters that are uppercase |
| `special_char_ratio` | float | Fraction of characters that are non-alphanumeric |
| `is_base64_like` | bool | ≥95% of characters are in the base64 or base64url alphabet |
| `has_padding` | bool | Token ends with `=` or `==` |
| `regex_match_type` | int | 0 = entropy path, 1+ = which regex tier matched |
| `token_structure_score` | float | 0 = random-looking, 1 = structured (dictionary words + separators) |
| `has_version_pattern` | bool | Contains `v1.2.3` or `1.0.0`-style version strings |
| `vendor_prefix_boost` | float | Weighted boost for known vendor prefixes (AKIA, ghp_, sk-, xox, AIza, eyJ, npm_, pypi-) |
| `token_span_offset` | float | Position of token within its line (0 = start, 1 = end) |
| `token_in_multiline_block` | bool | Token appears to be part of a PEM/SSH multi-line block |
| `embedded_token_flag` | bool | Token was extracted from inside a URL or DSN |
| `token_quote_type` | int | 0 = no quotes, 1 = single, 2 = double, 3 = backtick |
| `is_uuid_v4` | bool | Matches UUID v4 format exactly (8-4-4-4-12, version nibble = 4) — strong non-secret signal |
| `jwt_structure_valid` | bool | Token has three `.`-separated segments where the first decodes to valid JSON (a real JWT header) |
| `entropy_charset_mismatch` | float | High entropy token with low character-class diversity — signals base64 or hex payloads that look random but lack the variety of true secrets |
| `has_hash_prefix` | bool | Token starts with `sha256:`, `md5:`, `sha1:`, etc. |

**Dropped from earlier versions:** `normalized_entropy` (collinear with `token_entropy`), `cryptographic_score` (collinear with entropy + length + char class), `is_known_hash_length` (inverted signal — the synthetic corpus put hashes only in the negative class, so the model learned `hash_length=True → safe`, while real 40-char OpenAI legacy keys share those lengths).

---

### Group 2 — Variable Name Features (9 features)

Properties of the variable or key name on the left-hand side of the assignment.

| Feature | Type | What it measures |
|---|---|---|
| `var_name_extracted` | bool | Whether a variable name was successfully extracted from the line |
| `var_contains_safe` | bool | Variable name contains a safe-indicator substring (hash, sha, commit, uuid, version, etc.) |
| `var_name_length` | int | Character length of the extracted variable name |
| `var_is_uppercase` | bool | Variable is in CONSTANT_CASE style |
| `var_is_camelcase` | bool | Variable uses camelCase |
| `assignment_type` | int | 0 = unknown, 1 = `=`, 2 = `:`, 3 = `:=`, 4 = function argument |
| `in_string_literal` | bool | Token is wrapped in quotes |
| `var_ngram_secret_score` | float | Weighted sum of N-gram matches against the secret lexicon (api_key → 0.40, secret → 0.36, password → 0.36, token → 0.28, key → 0.20, etc.) — max weight 0.40, scaled down from 1.0 to prevent this feature from dominating |
| `var_ngram_safe_score` | float | Weighted sum of N-gram matches against the safe lexicon (commit_sha → 0.40, uuid → 0.38, sha256 → 0.38, hash → 0.34, example → 0.36, dummy → 0.36, etc.) |

**Dropped:** `var_contains_secret` — this was 25.4% feature importance but was literally the same regex as the heuristic classification layer (`secret|password|token|api_key`). The model spent a quarter of its capacity mirroring the rule instead of learning complementary signals, which left no capacity to distinguish `key_path`, `AUTHENTICATED`, `host`, and `port` from real credentials. The N-gram scores (`var_ngram_secret_score`, `var_ngram_safe_score`) remain as healthier continuous replacements.

---

### Group 3 — Context Features (17 features)

Properties of the surrounding code window (3 lines before and after).

| Feature | Type | What it measures |
|---|---|---|
| `line_is_comment` | bool | The line containing the token is a comment |
| `context_mentions_test` | bool | Nearby lines mention test, mock, fake, fixture, spec |
| `context_mentions_git` | bool | Nearby lines mention git, commit, sha, merge, branch |
| `context_mentions_hash` | bool | Nearby lines mention hash, checksum, digest, sha256 |
| `context_has_import` | bool | Nearby lines contain an import or require statement |
| `context_has_function_def` | bool | Nearby lines define a function or method |
| `file_is_test` | bool | File path matches test/spec/fixture patterns |
| `file_is_config` | bool | File path matches config/settings/env patterns |
| `file_extension_risk` | int | 0 = low risk (.md, .txt), 1 = medium (.py, .js), 2 = high (.env, .pem) |
| `surrounding_entropy_avg` | float | Average Shannon entropy of high-entropy tokens on nearby lines |
| `semantic_context_score` | float | −1 = safe context (hash/test keywords), 0 = neutral, +1 = risky (credential/auth keywords) |
| `surrounding_secret_density` | float | Ratio of nearby tokens that appear credential-like |
| `surrounding_token_count` | int | Count of high-entropy tokens on the same line |
| `key_value_distance` | int | Character distance between the variable name and the token (−1 if no variable found) |
| `json_path_hint` | int | Estimated nesting depth in a JSON/YAML structure |
| `adjacency_ngram_score` | float | Sum of N-gram secret scores from variable names on adjacent lines |
| `cross_line_entropy` | float | Average Shannon entropy across the ±3 line window |

**Dropped:** `line_position_ratio` — in synthetic training data this was always approximately 0.5 (derived from the context window size), but in production it varies 0.0–1.0 based on actual file length. This created train/serve skew.

---

### Group 4 — Stage B Precision Features (7 features)

Targeted FP suppression features added to address observed false-positive patterns at that phase.

| Feature | Type | What it catches |
|---|---|---|
| `is_hex_len_40` | bool | Exactly 40 hex chars — git SHA1 signature |
| `is_hex_len_64` | bool | Exactly 64 hex chars — SHA256 hash signature |
| `is_test_token` | bool | Token contains `_test_`, `_staging_`, or `test_` prefix — Stripe/Twilio test mode keys |
| `contains_example_keyword` | bool | Token contains EXAMPLE, xxxx, demo, or placeholder |
| `file_is_git_related` | bool | File path is inside `.git/`, or references hash or commit |
| `file_is_build` | bool | File path is inside `build/`, `dist/`, `node_modules/`, or similar artifact dirs |
| `file_is_example` | bool | File path is inside `example/`, `docs/`, or `demo/` |

---

### Group 5 — Stage B Generalization Features (5 features)

Hex disambiguation features to separate hex-encoded secrets from hex-encoded checksums and git objects.

| Feature | Type | What it measures |
|---|---|---|
| `hex_context_git_keywords` | bool | Context contains git, commit, merge, branch, rebase |
| `hex_context_crypto_keywords` | bool | Context contains encrypt, sign, key, secret — favors treating hex as a credential |
| `hex_adjacent_assignment_pattern` | int | 0 = unknown, 1 = env-var style, 2 = config style, 3 = function argument |
| `hex_in_url_or_dsn` | bool | Token is embedded inside a URL or DSN string |
| `hex_file_suggests_secret` | bool | File path contains `secrets/`, `.env`, or `credentials/` |

---

### Group 6 — Env-Loading Awareness Features (2 features)

| Feature | Type | What it measures |
|---|---|---|
| `env_loader_in_context` | bool | Context contains an env-loading call (dotenv, load_dotenv, require('dotenv'), os.getenv) |
| `is_env_fallback_value` | bool | Token is the fallback/default argument position in an env-loader call (the dangerous case: `os.getenv("KEY", "literal_secret")`) |

---

### Group 7 — Phase 7.0 Value-Shape Features (7 features)

New in the current vector. These target the four FP classes observed in the v0.3.0 production scan (`failedrun.json`): file paths, enum constants, server binding configs, and template placeholders.

| Feature | Type | Fires when |
|---|---|---|
| `value_starts_with_slash` | bool | Token starts with `/`, `~`, `./`, or `../` |
| `value_contains_path_separator` | bool | Token contains `/` or `\` AND has no URL scheme (`://`); excludes s3://, https://, postgres://, etc. |
| `value_ends_with_known_ext` | bool | Token ends with `.pem`, `.key`, `.crt`, `.jks`, `.p12`, `.pfx`, `.der`, `.json`, `.yaml`, `.yml`, `.txt`, `.env`, `.conf`, `.config` |
| `value_is_lowercase_word` | bool | `re.fullmatch(r"[a-z]+", token)` — catches `"authenticated"`, `"admin"`, `"anonymous"` |
| `value_is_dotted_quad_or_host` | bool | Token matches RFC1918 ranges (10.x.x.x, 172.16-31.x.x, 192.168.x.x), loopback (127.0.0.1, ::1), 0.0.0.0, or `localhost` |
| `value_is_template_syntax` | bool | Token contains `${...}`, `{{...}}`, `__X__`, `<<...>>`, or matches AI placeholder patterns (`YOUR_`, `_HERE`, `CHANGEME`, `PLACEHOLDER`, `FAKE`, `TEST_`) |
| `is_hex_with_no_alpha_mix` | bool | `re.fullmatch(r"[a-f0-9]+", token)` AND length ≥ 20 — all-lowercase hex, no uppercase; cryptographic hashes are all-lowercase while real API keys mix cases |

These replace `is_known_hash_length` with semantically correct signals. A git SHA is all-lowercase hex; an API key in hex format mixes upper and lowercase. The distinction is captured by `is_hex_with_no_alpha_mix` without the inverted-signal problem of the old feature.

---

## Part 2: The Synthetic Data Generator

`scripts/generate_synthetic_data.py` generates labeled training records through two parallel pipelines: LLM-generated code (via LM Studio, OpenAI-compatible API) and script-generated deterministic samples. The generator records `token_start`/`token_end` offsets for every sample, which feeds the `TokenMatch` infrastructure introduced in Phase 3.

---

### Positive Class (label = 1): What the model should flag

#### LLM Pipeline — Dynamic Prompt Matrix

The positive LLM prompt is assembled by independently sampling one element from each of six dimensions, creating combinatorial variety that prevents semantic collapse:

- **Language (15):** Python, Node.js, Go, Java, Ruby, PHP, C#/.NET, Swift, Perl, Rust, TypeScript, Kotlin, Scala, Bash, R
- **File type (18):** module with classes, settings file, test suite, CI/CD workflow, deployment script, DB migration, initializer, CLI tool, REST API client, background worker, Makefile, IaC, Docker Compose, AI-scaffolded integration, Dockerfile with ENV secrets, Helm values.yaml with defaults, Terraform .tf with inline credentials, Jupyter notebook cell
- **Industry (12):** fintech, healthcare, e-commerce, SaaS analytics, gaming, media streaming, enterprise HR, startup MVP, government, education, IoT, logistics
- **Secret type (10):** API key with realistic prefix, DB connection string, RSA/EC private key PEM, JWT signing secret, SMTP credentials, OAuth client secret, cloud provider key pair, encryption key, service account token, Redis AUTH URL
- **Assignment pattern (11):** module-level constant, class property default, constructor default, config dict, `os.getenv()` fallback, inline function argument, YAML/JSON field, Docker `${VAR:-default}`, Ruby `ENV.fetch { fallback }`, commented-out (git history leak), dev key active + prod key in trailing comment
- **Noise elements (10):** SHA-256 constants nearby, UUID constants nearby, 30+ lines of error handling, SRI hashes, bcrypt hashes, inline comments, type annotations, 20+ unrelated config settings, feature flags, base64 non-secret data nearby

#### Script-Generated Positive Generators (15 functions)

| Generator | What it produces |
|---|---|
| `_pos_hardcoded_api_key` | Vendor-prefixed API key (sk_live_, AKIA, ghp_, xoxb-, etc.) assigned to a secret-named variable |
| `_pos_connection_string` | Full DSN with embedded password (postgresql, mysql, mongodb+srv, redis) |
| `_pos_private_key_pem` | PEM-wrapped private key body assigned to a signing/jwt key variable |
| `_pos_env_fallback_secret` | `os.getenv("KEY", "literal_secret")` and 11 framework variants: django-environ config(), Pydantic BaseSettings, Spring @Value, Rails credentials.dig, Node ?? operator, Zod envSchema, Helm default filter, Go fallback pattern |
| `_pos_value_contains_var_name` | Token starts with the variable name as a substring (`password = "password{random}"`) — the duplicate-token distribution gap |
| `_pos_value_equals_var_name_uppercase` | Value embeds the variable name uppercased (`API_KEY = "API_KEY_LIVE_{rand}"`) — legacy tag-inside-secret pattern |
| `_pos_yaml_duplicate_key_value` | YAML key name appears as prefix in the value (`database_password: "database_password_2024_{rand}"`) |
| `_pos_jsonish_duplicate` | JS/JSON config where the key appears as prefix in the value (`"apiKey": "apiKey-prod-{rand}"`) |
| `_pos_env_file_duplicate` | `.env` format where var name is embedded in value (`DATABASE_KEY=DATABASE_KEY_{rand}`) |
| `_pos_cross_lang_duplicate` | JS/TS/Go/Java/Ruby variants of the var-name-as-value-prefix pattern |
| `_pos_commented_secret` | Commented-out secret in `#`, `//`, `--`, `<!--`, `/*` styles — git history leaks |
| `_pos_devprod_swap` | Active test key on the line + prod key leaking in a trailing comment (`# PROD: sk_live_xxx`) |
| `_pos_python_getenv_fallback_explicit` | Python-specific `os.getenv`, `os.environ.get`, `getattr(settings, ...)`, django-environ `config()`, Pydantic `Field(default=...)` |
| `_pos_apim_style_var_names` *(Phase 7.2)* | APIM/MGMT/SVC/INTERNAL/VENDOR/PARTNER prefix-suffix variable names — the real-world pattern from failedrun.json |
| `_pos_value_shape_adversarial` *(Phase 7.2)* | Real secrets whose values share shape with negatives: s3:// URIs, gs:// URIs, `Bearer eyJ...`, postgres DSNs, lowercase short passwords (`hunter{rand}2x`), IP-prefix values (`10.x.x.{rand_suffix}`) — prevents value-shape features from becoming 1-bit classifiers |

---

### Negative Class (label = 0): What the model should not flag

#### LLM Pipeline — Dynamic Prompt Matrix

The negative LLM prompt independently samples from:

- **Language (15):** same as positive
- **File type (11):** test suite, verification module, data pipeline, integrity checking tool, content-addressable storage, crypto verification module, CI/CD config, build file, ConfigMap (no Secrets), benchmark harness, schema migration with checksums
- **High-entropy type (14):** SHA-256 content digests, SHA-384 SRI hashes, bcrypt hashes (derived, not passwords), HMAC-SHA256 webhook signatures, CSP nonces, Ed25519/X25519 PUBLIC keys, mock bearer tokens labeled as fixtures, base64 image data URIs, Docker layer digests, UUID v4 correlation IDs, IPFS CIDs, git tree/blob object hashes, runtime-generated tokens (secrets.token_hex, crypto.randomBytes, uuid.uuid4 — value produced at runtime, never a literal), correct env-var loading with no fallback string
- **Context hint (11):** label as mock/fake/test, explicit "NOT real credentials" comments, mock_*/fake_*/test_* variable names, verification function context, "PUBLIC key / DERIVED value" labels, assertions verifying hash correctness, deduplication/cache-key context, test fixture setup, content-addressable context, show the generation call prominently (not a literal), load secrets via env with sys.exit(1) if missing

#### Script-Generated Negative Generators (28 functions)

**Original corpus negatives (16):**

| Generator | Pattern |
|---|---|
| `_neg_lock_file_hash` | npm package-lock sha512 integrity hash |
| `_neg_css_hex_color` | CSS 6-char hex color code |
| `_neg_git_sha` | 40-char git commit SHA in a comment or deploy script |
| `_neg_uuid` | UUID v4 in a JSON fixture or test file |
| `_neg_gpg_pubkey_line` | Base64 line from a GPG public key block |
| `_neg_docker_digest` | SHA256 digest in a Dockerfile `FROM image@sha256:` directive |
| `_neg_yarn_lock_sha` | SHA256 in a yarn.lock entry |
| `_neg_asset_manifest` | 40-char hash in a Webpack/Parcel asset manifest |
| `_neg_bcrypt_hash` | $2b$ bcrypt hash in a test fixture |
| `_neg_base64_image_data` | base64 inline image data URI |
| `_neg_hmac_webhook_signature` | HMAC-SHA256 computed webhook signature in test headers |
| `_neg_sri_hash` | SRI hash in an HTML `<script integrity="...">` tag |
| `_neg_test_fixture_mock_token` | Base64 token explicitly labeled as a mock/fake test fixture |
| `_neg_csprng_nonce` | CSPRNG nonce for CSP header or AES-GCM IV |
| `_neg_content_addressable_hash` | IPFS CID, git tree/blob SHA, or Docker manifest digest |
| `_neg_env_loaded_secret` | `VAR = os.getenv("VAR")` — correct env loading with no fallback |

**Phase 3.5 augmentation negatives (7):**

| Generator | Pattern |
|---|---|
| `_neg_env_loaded_v2` | Corrected version of env-loaded-secret with `token_start` pointing to the lookup site, not position 0 |
| `_neg_value_matches_var_typed_default` | Config default whose value contains the var name (`PORT = "PORT_DEFAULT_8080_abc"`) |
| `_neg_logger_template` | Log line where a sensitive-named variable appears but the high-entropy string is a trace ID |
| `_neg_test_assertion_with_var_name` | `assert response.token == "test_token_for_unit_test_only_{rand}"` |
| `_neg_llm_placeholder` | `API_KEY = "YOUR_API_KEY_HERE"` — documentation placeholders |
| `_neg_runtime_token_generation` | `secrets.token_hex(32)`, `uuid.uuid4().hex`, `crypto.randomBytes(32)`, `UUID.randomUUID()` across Python, JS, Go, Java, Ruby, Rust — token is the variable name (LHS), not a literal value |
| `_neg_documented_placeholder_in_docstring` | Placeholder inside a Python docstring, RST code-block, JSDoc `@example`, or markdown fence |

**Phase 7.1 negatives targeting v0.3.0 FP classes (5):**

| Generator | FP class it targets | Examples |
|---|---|---|
| `_neg_file_path_value` | Credential-named variable holding a filesystem path | `key_path = "/etc/ssl/keys/server.key"`, `cert_file = "C:\certs\server.crt"`, `ca_cert = "~/.config/gcloud/application_default_credentials.json"` |
| `_neg_enum_constant_lowercase` | Uppercase constant holding a lowercase enum identifier | `AUTHENTICATED = "authenticated"`, `STATUS_ANONYMOUS = "anonymous"`, `ROLE_ADMIN = "admin"` across Python, JS, YAML, Java, Go |
| `_neg_host_port_config` | Host/port/bind server configuration | `host = "0.0.0.0"`, `bind = ":8443"`, `database_host = "db.internal"`, `PORT = "3000"`, covering DB hosts, Redis hosts, SMTP hosts |
| `_neg_template_placeholder` | Helm/Kustomize/Mustache/double-bracket template syntax | `apiKey: "${API_KEY}"`, `secret: "{{ .Values.secret }}"`, `hostName: "__EKSHOSTNAME__"`, `password: "<<REPLACE_ME>>"` in k8s/helm/docker-compose templates |
| `_neg_file_extension_value` | Variable holding a bare filename with security extension | `truststore = "truststore.jks"`, `cert_chain = "chain.pem"`, `ssl_certificate = "server.crt"`, used in SSL context loading code |

---

## Part 3: Coverage Gaps and Structural Notes

**What the generator covers well:**
- High-entropy false positives from cryptographic tooling (hashes, SRI, bcrypt, IPFS)
- Environment-loading patterns in 8+ frameworks
- Duplicate-token lines where the variable name appears inside its own value
- AI-generated code structure (Dockerfiles, Helm charts, Terraform, Jupyter) via LLM prompts
- Runtime token generation across 6 languages
- Commented-out secrets in 5 comment styles
- Cross-language coverage for most patterns (15 languages in the LLM matrix)

**What is not yet covered and causes false positives until retrained:**
- File paths, enum constants, host bindings, and template placeholders at the ML layer — Phase 7.1 generators are written but not yet run
- APIM/MGMT/SVC prefix-suffix variable names — `_pos_apim_style_var_names` is written but not yet run
- Adversarial positives whose values share shape with the new negative classes — `_pos_value_shape_adversarial` is written but not yet run

**The current shipped ONNX** was trained on the 65-feature vector without any Phase 7.1 or 7.2 data. The v0.3.2 severity calibration and lexicon expansion (Phases 6.1–6.4) partially compensate at the heuristic layer, but the underlying model still produces false positives on file paths, enum constants, and host configs at the detection stage. The full fix requires running the generators, re-extracting features with the 67-feature extractor (`scripts/regenerate_features.py --expected-feature-count 67`), and retraining.
