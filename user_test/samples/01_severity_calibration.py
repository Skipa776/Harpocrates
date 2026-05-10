# Sample: Severity Calibration
# Run: harpocrates scan --ml user_test/samples/01_severity_calibration.py
#
# Before v0.4.0 every entropy/ML finding showed severity: INFO.
# Now var-name + classification confidence drives the severity band.

APIM_CLIENT_KEY = "eDRlUVQ7dt5nKmP9rS2tV8wY1zA5cE0g"   # expect MEDIUM
APIM_SECRET_KEY = "40z_9Yw7dt5nKmP9rS2tV8wY1zA5cE0gH4"  # expect MEDIUM

# Strong lexicon match → HIGH
DB_PASSWORD = "Tr0ub4dor&3_longEnoughForEntropy_xyz"      # expect HIGH

# JWT var name with a high-entropy bearer-style value → HIGH
jwt_bearer_token = "xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3uW7yB2dF5hK8m"  # expect HIGH

# Weak var-name signal → INFO (expected not to elevate)
generic_val = "xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3uW7yB2dF5hK8m"       # expect INFO
