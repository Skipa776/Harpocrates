# Sample: OpenAI Legacy Key Detection
# Run: harpocrates scan user_test/samples/04_openai_keys.py
#
# The existing regex only matched the strict 48-char format and sk-proj-... form.
# A new OPENAI_API_KEY_LEGACY pattern (HIGH tier) catches shorter/legacy keys.

# Current format — 48 alphanumeric chars after sk-  → severity: CRITICAL
OPENAI_API_KEY = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJK"

# Legacy / short format with hyphens in body          → severity: HIGH (new in v0.4)
OPENAI_KEY_LEGACY = "sk-RQMJj8ELDjv7TRcaB9xTestValue"

# Project-scoped key                                  → severity: CRITICAL
OPENAI_PROJ_KEY = "sk-proj-xK9mN3pQ7rS2tV8wY1zA5T3cE0gH4jL6nP9qT3uW7"
