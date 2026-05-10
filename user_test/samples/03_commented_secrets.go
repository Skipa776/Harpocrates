// Go source file — commented-out secret detection across comment styles
// Run: harpocrates scan user_test/samples/03_commented_secrets.go
// Use --json to see the "in_comment": true field on each finding.
//
// Before v0.4.0 every commented line was silently skipped.
// Now // and /* */ blocks are scanned. Severity is NOT downgraded —
// a commented-out credential is still a leaked credential.
//
// EXPECT FINDINGS on these lines:

// DB_PASSWORD = "Tr0ub4dor3_longEnoughForEntropy_xyz123"  // old password — must detect
// apiKey := "xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3uW7yB2dF"  // rotated key — must detect

/*
   clientSecret = "xK9mN3pQ7rS2tV8wY1zA5cE0gH4jL6nP9qT3uW7yB2dF"
   block comment above — must detect (in_comment: true)
*/

// EXPECT NO FINDING — pure prose comment with no assignment or quote chars:
// This function handles authentication by delegating to the identity provider.

package main

func main() {}
