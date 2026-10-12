#!/usr/bin/env bash
# Shell deployment script — OpenAI key format coverage
# Run: harpocrates scan docs/samples/samples/04_openai_keys.sh
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
# Built at runtime ("sk-" + five "abc-" groups + "XYZ") so the repo holds no key
# literal; see TESTING_GUIDE.txt, Feature 4, for how to scan the expanded value.
export OPENAI_KEY_LEGACY="sk-$(printf 'abc-%.0s' 1 2 3 4 5)XYZ"

# Project-scoped key
export OPENAI_PROJ_KEY="sk-proj-xK9mN3pQ7rS2tV8wY1zAT3cE0gH4jL6nP9qT3uW7yB"
