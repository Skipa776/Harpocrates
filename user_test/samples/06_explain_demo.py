# Sample: Explainability (--explain flag)
# Run: harpocrates scan --explain user_test/samples/06_explain_demo.py
# Requires: pip install harpocrates[ml]
#
# --explain outputs JSON with per-feature TreeSHAP contributions showing
# WHICH features drove each ML detection decision.
#
# Useful commands:
#   Pretty-print:
#     harpocrates scan --explain user_test/samples/06_explain_demo.py | python -m json.tool
#
#   Show top contributing features for every ML finding:
#     harpocrates scan --explain user_test/samples/06_explain_demo.py | python -c "
#     import json, sys
#     data = json.load(sys.stdin)
#     for item in data['findings']:
#         f = item['finding']
#         e = item['explanation']
#         print(f\"--- {f['type']} at line {f['line']} (evidence: {f['evidence']}) ---\")
#         if e:
#             print('  Top positive features (pushing toward SECRET):')
#             for c in e['top_positive']:
#                 print(f\"    {c['name']:40s} contribution={c['contribution']:+.3f}\")
#             print('  Top negative features (pushing toward SAFE):')
#             for c in e['top_negative']:
#                 print(f\"    {c['name']:40s} contribution={c['contribution']:+.3f}\")
#         else:
#             print('  (regex-tier match — no ML explanation needed)')
#     "

# ML-stage finding — explanation will be non-null with full feature contributions
APIM_SECRET_KEY = "40z_9Yw7dt5nKmP9rS2tV8wY1zA5cE0gH4jL6n"

# Another ML-stage finding for comparison
client_secret = "xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3uW7yB2dF5hK8m"

# Regex-tier (CRITICAL) — explanation will be null
OPENAI_API_KEY = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJK"
