use harpocrates_rust_scanner::{Evidence, ScanOptions, scan_text};

fn options(file: Option<&str>) -> ScanOptions {
    ScanOptions {
        file: file.map(str::to_owned),
    }
}

#[test]
fn finds_structured_regex_secrets_without_ml() {
    let findings = scan_text(
        "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n",
        &options(Some("secrets.env")),
    )
    .expect("scan succeeds");

    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0].kind, "AWS_ACCESS_KEY_ID");
    assert_eq!(findings[0].evidence, Evidence::Regex);
    assert_eq!(findings[0].line, 1);
    assert_eq!(findings[0].token, "AKIAIOSFODNN7EXAMPLE");
}

#[test]
fn emits_entropy_and_assignment_candidates_for_python_ml() {
    let findings = scan_text(
        "opaque = aB3dEfGhIjKlMnOpQrStUvWxYz012\npassword = \"hunter2\"\n",
        &options(Some("config.py")),
    )
    .expect("scan succeeds");

    assert!(findings.iter().any(|finding| {
        finding.kind == "ENTROPY_CANDIDATE" && finding.evidence == Evidence::Entropy
    }));
    assert!(findings.iter().any(|finding| {
        finding.kind == "ML_CANDIDATE"
            && finding.evidence == Evidence::Ml
            && finding.token == "hunter2"
            && finding.var_name.as_deref() == Some("password")
    }));
}

#[test]
fn regex_hits_take_priority_over_ml_candidates_on_the_same_line() {
    let token = format!("ghp_{}", "a".repeat(36));
    let findings = scan_text(
        &format!("token = \"{token}\"\n"),
        &options(Some("config.py")),
    )
    .expect("scan succeeds");

    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0].kind, "GITHUB_PAT");
    assert_eq!(findings[0].evidence, Evidence::Regex);
}

#[test]
fn preserves_comment_and_dotenv_semantics() {
    let comment = scan_text(
        &format!("# token = ghp_{}\n", "b".repeat(36)),
        &options(Some("config.py")),
    )
    .expect("scan succeeds");
    assert_eq!(comment[0].in_comment, Some(true));

    let dotenv = scan_text(
        "OPENAI_BASE_URL=https://api.openai.com/v1\n",
        &options(Some(".env.local")),
    )
    .expect("scan succeeds");
    assert!(dotenv.iter().any(|finding| {
        finding.kind == "ENV_ASSIGNMENT" && finding.token == "https://api.openai.com/v1"
    }));
}

#[test]
fn suppresses_entropy_for_low_noise_files_but_keeps_regex() {
    let noisy = scan_text(
        "asset=aB3dEfGhIjKlMnOpQrStUvWxYz012\n",
        &options(Some("index.html")),
    )
    .expect("scan succeeds");
    assert!(noisy.is_empty());

    let token = format!("ghp_{}", "c".repeat(36));
    let regex = scan_text(
        &format!("<script>token='{token}'</script>\n"),
        &options(Some("index.html")),
    )
    .expect("scan succeeds");
    assert_eq!(regex[0].kind, "GITHUB_PAT");
}

#[test]
fn supports_every_python_signature_family() {
    let samples = [
        ("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE".to_owned()),
        ("GITHUB_PAT", format!("ghp_{}", "a".repeat(36))),
        (
            "SLACK_TOKEN",
            format!("xoxb-1234567890123-{}", "A".repeat(24)),
        ),
        ("STRIPE_KEY", format!("sk_live_{}", "x".repeat(24))),
        ("OPENAI_API_KEY", format!("sk-{}", "x".repeat(48))),
        (
            "ANTHROPIC_API_KEY",
            format!("sk-ant-api03-{}", "a".repeat(93)),
        ),
        ("GCP_API_KEY", format!("AIza{}", "A".repeat(35))),
        ("NPM_TOKEN", format!("npm_{}", "a".repeat(36))),
        ("PYPI_TOKEN", format!("pypi-{}", "a".repeat(50))),
        ("PRIVATE_KEY", "-----BEGIN PRIVATE KEY-----".to_owned()),
        (
            "SLACK_WEBHOOK",
            format!(
                "https://hooks.slack.com/services/T12345678/B12345678/{}",
                "a".repeat(24)
            ),
        ),
        (
            "DISCORD_WEBHOOK",
            format!(
                "https://discord.com/api/webhooks/123456789012345678/{}",
                "a".repeat(68)
            ),
        ),
        (
            "SENDGRID_API_KEY",
            format!("SG.{}.{}", "a".repeat(22), "b".repeat(43)),
        ),
        ("TWILIO_API_KEY", format!("SK{}", "a1b2c3d4".repeat(4))),
        ("DATABRICKS_TOKEN", format!("dapi{}", "a1b2".repeat(8))),
        ("HASHICORP_VAULT_TOKEN", format!("hvs.{}", "a".repeat(90))),
        (
            "OPENAI_API_KEY_LEGACY",
            "sk-RQMJj8ELDjv7TRc-dS9sSw".to_owned(),
        ),
    ];

    for (expected, token) in samples {
        let findings = scan_text(&token, &options(Some("secrets.txt"))).expect("scan succeeds");
        assert!(
            findings.iter().any(|finding| finding.kind == expected),
            "{expected} did not match {token}"
        );
    }
}

#[test]
fn legacy_openai_pattern_does_not_duplicate_strict_formats() {
    for (token, expected) in [
        (format!("sk-{}", "x".repeat(48)), "OPENAI_API_KEY"),
        (
            format!("sk-proj-{}T3{}", "a".repeat(20), "b".repeat(20)),
            "OPENAI_API_KEY",
        ),
        (
            format!("sk-ant-api03-{}", "a".repeat(93)),
            "ANTHROPIC_API_KEY",
        ),
    ] {
        let findings = scan_text(&token, &options(Some("secrets.txt"))).expect("scan succeeds");
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].kind, expected);
    }
}
