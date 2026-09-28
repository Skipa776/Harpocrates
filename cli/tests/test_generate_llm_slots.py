"""LLM slot generation: deterministic jobs and strict output validation."""

from scripts.generate_llm_slots import clean_output, jobs


def test_jobs_are_deterministic_and_typed():
    a, b = jobs(50, seed=3), jobs(50, seed=3)
    assert a == b and len(a) == 50
    assert all(j["secret_kinds"] and j["nonsecret_kinds"] for j in a)
    assert len({j["language"] for j in a}) > 10


def test_clean_output_strips_fences_and_requires_markers():
    raw = '```python\nkey = "{{SECRET:stripe_key}}"\nsha = "{{NONSECRET:git_sha}}"\n```\n'
    assert clean_output(raw) == 'key = "{{SECRET:stripe_key}}"\nsha = "{{NONSECRET:git_sha}}"\n'
    assert clean_output("print('no slots here')") is None
    assert clean_output('x = "{{SECRET}}"') is None            # malformed
    assert clean_output('x = "{{SECRET:made_up}}"') is None    # unknown kind
