# Sample: Violation Categories
# Run: harpocrates scan --json user_test/samples/02_violation_categories.py | python -m json.tool
#
# Every finding now includes "category" and "category_reason" fields.
# Category is inferred via: regex signature → var-name lexicon → value structure.

DB_PASSWORD      = "Tr0ub4dor&3_longEnoughForEntropy_xyz"                 # category: password
client_secret    = "xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3uW7yB2dF5hK8m"  # category: oauth_secret
hmac_signing_key = "xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3uW7yB2dF5hK8m"  # category: crypto_key
session_id_token = "xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3uW7yB2dF5hK8m"  # category: session_token

# Value-structure inference: eyJ prefix + two dots → JWT regardless of var name
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyMTIzIn0.SflKxwRJSMeK"  # category: jwt

# Database connection string with embedded credentials
db_url = "postgres://admin:Tr0ub4dor3_SecretPass@prod-db.internal:5432/appdb"  # category: connection_string

# Falls through all layers → generic_secret
SOME_SECRET = "xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3uW7yB2dF5hK8m"  # category: generic_secret
