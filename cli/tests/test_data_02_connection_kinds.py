"""DATA-02: connection strings and credentialed URLs, with look-alike negatives."""

import random
import re

from Harpocrates.training.generators import secret_templates as t

URI_WITH_PASSWORD = re.compile(r"^[a-z0-9+]+://[^:/@\s]+:[^@\s]+@[\w.-]+(:\d+)?(/[\w.-]*)?")
PLACEHOLDER = re.compile(r"\$\{|<[a-z_]+>|\{\{|%s|changeme|\*\*\*")


def test_data_02_connection_uri_embeds_a_password():
    random.seed(0)
    for _ in range(50):
        uri = t.generate_connection_uri()
        assert URI_WITH_PASSWORD.match(uri), uri
        assert "\n" not in uri and not PLACEHOLDER.search(uri)


def test_data_02_ado_and_jdbc_carry_password_field():
    random.seed(1)
    for _ in range(20):
        ado, jdbc = t.generate_ado_connection_string(), t.generate_jdbc_url()
        assert re.search(r"(?i)password=[^;{}<>$]{6,}", ado) and not PLACEHOLDER.search(ado)
        assert re.search(r"password=[^&{}<>$]{6,}", jdbc) and not PLACEHOLDER.search(jdbc)


def test_data_02_uri_password_never_contains_mask_marker():
    random.seed(1234)  # this seed produced "***" inside a real password before the fix
    assert not any("***" in t._uri_safe_password() for _ in range(200_000))


def test_data_02_token_url_carries_token_param():
    random.seed(2)
    for _ in range(20):
        assert re.search(r"[?&](token|access_token|api_key|key|sig)=[\w-]{16,}", t.generate_token_url())


def test_data_02_negative_lookalikes_have_no_real_credential():
    random.seed(3)
    for _ in range(50):
        neg = t.generate_connection_placeholder()
        # Either no password at all, or an obvious placeholder.
        assert not URI_WITH_PASSWORD.match(neg) or PLACEHOLDER.search(neg), neg
