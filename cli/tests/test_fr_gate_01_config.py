"""FR-GATE-01: the gate's settings. It binds 127.0.0.1 only, and a plain-HTTP upstream is allowed for loopback only.

Pure unit tests: no socket is opened here.
"""
from __future__ import annotations

import pytest

from Harpocrates.gate.config import GateConfig, Upstream, parse_upstream


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "::1", "localhost", "127.0.0.2", "", "192.0.2.1", "example.com"])
def test_fr_gate_01_bind_host_must_be_127_0_0_1(host: str) -> None:
    with pytest.raises(ValueError, match="127.0.0.1"):
        GateConfig(host=host)


def test_fr_gate_01_defaults_bind_loopback_and_use_the_provider_https_hosts() -> None:
    config = GateConfig()
    assert config.host == "127.0.0.1"
    assert config.upstream_for("anthropic") == Upstream("https", "api.anthropic.com", 443)
    assert config.upstream_for("openai") == Upstream("https", "api.openai.com", 443)


def test_fr_gate_01_loopback_overrides_may_use_plain_http() -> None:
    config = GateConfig(anthropic_upstream="http://127.0.0.1:8081", openai_upstream="http://[::1]:8082")
    assert config.upstream_for("anthropic") == Upstream("http", "127.0.0.1", 8081)
    assert config.upstream_for("openai") == Upstream("http", "::1", 8082)


@pytest.mark.parametrize("url", [
    "http://example.com", "http://example.com:80", "http://10.0.0.1:8080", "http://localhost:8080",
    "http://127.0.0.1.evil.example", "http://api.anthropic.com",
])
def test_fr_gate_01_plain_http_to_a_non_loopback_upstream_is_refused(url: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        parse_upstream(url)


def test_fr_gate_01_https_overrides_keep_their_port_and_default_to_443() -> None:
    assert parse_upstream("https://gateway.example.com") == Upstream("https", "gateway.example.com", 443)
    assert parse_upstream("https://gateway.example.com:8443") == Upstream("https", "gateway.example.com", 8443)
    assert parse_upstream("https://127.0.0.1:8443") == Upstream("https", "127.0.0.1", 8443)


@pytest.mark.parametrize("url", [
    "", "https://", "ftp://example.com", "example.com", "https://user:pw@example.com", "https://example.com/v1",
    "https://example.com?x=1", "https://example.com#frag", "https://example.com:notaport", "https://example.com:0", "https://exa_mple.com", "https://exa%0d%0ample.com",
])
def test_fr_gate_01_upstream_must_be_a_bare_scheme_host_and_port(url: str) -> None:
    with pytest.raises(ValueError):
        parse_upstream(url)


def test_fr_gate_01_upstream_error_never_echoes_userinfo() -> None:
    with pytest.raises(ValueError) as caught:
        parse_upstream("https://user:hunter2@example.com")
    assert "hunter2" not in str(caught.value)


@pytest.mark.parametrize("field,value", [
    ("max_body", 0), ("scan_timeout", 0), ("scan_timeout", -1.0), ("connect_timeout", 0), ("read_timeout", 0),
    ("client_timeout", 0), ("port", -1), ("port", 65536),
])
def test_fr_gate_01_limits_must_be_positive(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        GateConfig(**{field: value})


def test_fr_gate_01_spec_defaults() -> None:
    config = GateConfig()
    assert config.max_body == 64 * 1024 * 1024  # decision 9
    assert config.scan_timeout == 5.0  # FR-GATE-03 budget


def test_fr_gate_01_unknown_provider_is_refused() -> None:
    with pytest.raises(ValueError, match="provider"):
        GateConfig().upstream_for("gemini")
