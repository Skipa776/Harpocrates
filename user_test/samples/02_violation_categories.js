// JavaScript/Node.js config — violation category inference across secret types
// Run: harpocrates scan --json user_test/samples/02_violation_categories.js | python -m json.tool
//
// Every finding includes "category" and "category_reason" in the JSON output.
// Three inference layers: regex signature → var-name lexicon → value structure.
//
// Expected categories:
//   DB_PASSWORD         → password           (var-name lexicon, confidence 0.90)
//   clientSecret        → oauth_secret       (var-name lexicon, confidence 0.85)
//   hmacSigningKey      → crypto_key         (var-name lexicon, confidence 0.85)
//   sessionIdToken      → session_token      (var-name lexicon, confidence 0.75)
//   token (eyJ...)      → jwt               (value-structure: eyJ prefix segment)
//   dbUrl               → connection_string  (value-structure: postgres:// with credentials)
//   NONDESCRIPT_VALUE   → generic_secret     (no var-name or value-structure signal)

const DB_PASSWORD    = "Tr0ub4dor3_longEnoughForEntropy_xyz123";
const clientSecret   = "xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3uW7yB2dF5hK8m";
const hmacSigningKey = "xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3uW7yB2dF5hK8m";
const sessionIdToken = "xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3uW7yB2dF5hK8m";

// Value-structure inference: eyJ prefix + two dots → JWT regardless of var name
const token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyMTIzIn0.SflKxwRJSMeK";

// Postgres URI with embedded credentials
const dbUrl = "postgres://admin:Tr0ub4dor3_SecretPass@prod-db.internal:5432/appdb";

// Falls through all layers → generic_secret (no recognisable var-name signal)
const NONDESCRIPT_VALUE = "xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3uW7yB2dF5hK8m";
