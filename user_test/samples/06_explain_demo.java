// Java Spring Boot config class — explainability demo
// Run: harpocrates scan --explain user_test/samples/06_explain_demo.java
// Requires: pip install harpocrates[ml]
//
// --explain outputs JSON with per-feature TreeSHAP contributions showing
// which model features drove each ML detection decision.
//
// Useful one-liners:
//
//   Pretty-print the full output:
//     harpocrates scan --explain user_test/samples/06_explain_demo.java | python -m json.tool
//
//   Show only the top features per finding:
//     harpocrates scan --explain user_test/samples/06_explain_demo.java | python -c "
//     import json, sys
//     data = json.load(sys.stdin)
//     for item in data['findings']:
//         f = item['finding']
//         e = item['explanation']
//         print(f\"--- {f['type']} line {f['line']} ({f['evidence']}) ---\")
//         if e:
//             for c in e['top_positive']:
//                 print(f\"  + {c['name']:40s} {c['contribution']:+.3f}\")
//         else:
//             print('  (regex match — no ML explanation)')
//     "
//
// EXPECTED OUTPUT SHAPE:
//   APIM_SECRET_KEY  → ML finding, non-null explanation, top_positive shows
//                      var_ngram_secret_score and token_entropy as top drivers
//   CLIENT_SECRET    → ML finding, non-null explanation
//   OPENAI_API_KEY   → CRITICAL regex match, explanation: null

@Configuration
public class AppConfig {

    // ML-stage finding — explanation will show feature contributions
    private static final String APIM_SECRET_KEY = "40z_9Yw7dt5nKmP9rS2tV8wY1zA5cE0gH4jL6n";

    // Another ML-stage finding for comparison
    private static final String CLIENT_SECRET = "xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3uW7yB2dF5hK8m";

    // Regex-tier CRITICAL — explanation will be null (deterministic match needs no explanation)
    private static final String OPENAI_API_KEY = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJK";

}
