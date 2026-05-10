# Sample: Commented-out Secret Detection
# Run: harpocrates scan user_test/samples/03_commented_secrets.py
#
# Before v0.4.0 the detector skipped every line starting with # or //.
# Now all comment styles are scanned. The "in_comment" field is set in JSON output.
# Use: harpocrates scan --json user_test/samples/03_commented_secrets.py | python -m json.tool

# EXPECT FINDINGS on these lines:

# DB_PASSWORD = "Tr0ub4dor&3_longEnoughForEntropy_xyz"       # Python comment — must detect
# api_key = "xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3uW7yB2dF"  # commented out — must detect

# Equivalent in other languages (scan as plain text to verify):
# // const apiKey = "xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3uW7yB2dF";  // JS — must detect

# This line has high-entropy on the right but is clearly a file path (safe):
# key_path = "/etc/ssl/private/server.key"

# -- sql_secret = "xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3uW7yB2dF"   -- SQL comment — must detect

# EXPECT NO FINDING on this line (pure prose — no = or " chars):
# This is just a description of what the next section does, nothing sensitive here

print("scan me")
