# Sample: False Positive Suppression (Value-Shape Features)
# Run: harpocrates scan --ml user_test/samples/05_false_positive_suppression.py
#
# Before v0.4.0 the model had no features distinguishing credentials from
# file paths, enum values, host literals, or template placeholders.
# Six new value-shape binary features now suppress these false positives.
#
# EXPECTED: all lines below should appear at INFO or produce no finding.
# If any appear at HIGH or CRITICAL that is a regression.

key_path          = "/etc/ssl/private/server.key"     # file path  → INFO (value_starts_with_slash)
AUTHENTICATED     = "authenticated"                    # enum const → INFO (value_is_lowercase_word)
host              = "0.0.0.0"                          # host bind  → INFO (value_is_dotted_quad_or_host)
api_key_template  = "${API_KEY}"                       # Helm/env   → INFO (value_is_template_syntax)
kube_template     = "{{ .Values.apiKey }}"             # Helm chart → INFO (value_is_template_syntax)
keystore_path     = "truststore.jks"                   # filename   → INFO (value_ends_with_known_ext)
cert_file         = "server.pem"                       # filename   → INFO (value_ends_with_known_ext)
bind_address      = "127.0.0.1"                        # loopback   → INFO (value_is_dotted_quad_or_host)
placeholder       = "__API_KEY_PLACEHOLDER__"           # sentinel   → INFO (value_is_template_syntax)
