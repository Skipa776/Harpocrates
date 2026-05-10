#!/usr/bin/env bash
# Shell deployment script — OpenAI key format coverage
# Run: harpocrates scan user_test/samples/04_openai_keys.sh
#
# Three formats are now detected:
#
#   Current format  sk-[alphanumeric]{48}        → CRITICAL (OPENAI_API_KEY)
#   Legacy format   sk-[alphanumeric/_-]{16,}    → HIGH     (OPENAI_API_KEY_LEGACY, new in v0.4)
#   Project-scoped  sk-proj-...-T3...            → CRITICAL (OPENAI_API_KEY)
#
# The legacy/short format was previously missed. HIGH (not CRITICAL) because
# the broader pattern sk- + 16+ chars has higher false-positive potential.

# Current-format key — 48 alphanumeric chars after sk-
export OPENAI_API_KEY="sk-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKL"

# Legacy / short format with hyphens in body — NEW detection in v0.4
export OPENAI_KEY_LEGACY="sk-RQMJj8ELDjv7TRcaB9xTestValue"

# Project-scoped key
export OPENAI_PROJ_KEY="sk-proj-xK9mN3pQ7rS2tV8wY1zAT3cE0gH4jL6nP9qT3uW7yB"
