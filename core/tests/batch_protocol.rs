use harpocrates_rust_scanner::{BatchScanRequest, FileScanRequest, PROTOCOL_VERSION, scan_batch};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock is after epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "harpocrates-rust-{label}-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir_all(&path).expect("temporary directory is created");
    path
}

fn file(path: &Path, max_bytes: Option<usize>) -> FileScanRequest {
    FileScanRequest {
        path: path.to_path_buf(),
        max_bytes,
    }
}

#[test]
fn empty_batch_returns_empty_versioned_response() {
    let response = scan_batch(BatchScanRequest {
        protocol_version: PROTOCOL_VERSION,
        files: Vec::new(),
    })
    .expect("empty batch succeeds");

    assert_eq!(response.protocol_version, PROTOCOL_VERSION);
    assert!(response.files.is_empty());
}

#[test]
fn batch_scans_multiple_files_in_request_order() {
    let root = temp_dir("ordered");
    let second = root.join("b.env");
    let first = root.join("a.env");
    fs::write(&second, "APP_NAME=Test\n").expect("file written");
    fs::write(&first, "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n").expect("file written");

    let response = scan_batch(BatchScanRequest {
        protocol_version: PROTOCOL_VERSION,
        files: vec![file(&second, None), file(&first, None)],
    })
    .expect("batch succeeds");

    assert_eq!(response.files[0].path, second);
    assert_eq!(response.files[1].path, first);
    assert!(response.files[0].findings.is_empty());
    assert_eq!(response.files[1].findings[0].kind, "AWS_ACCESS_KEY_ID");

    fs::remove_dir_all(root).expect("temporary directory removed");
}

#[test]
fn one_unreadable_file_does_not_discard_successful_files() {
    let root = temp_dir("partial-error");
    let missing = root.join("missing.env");
    let valid = root.join("valid.env");
    fs::write(&valid, "AKIAIOSFODNN7EXAMPLE\n").expect("file written");

    let response = scan_batch(BatchScanRequest {
        protocol_version: PROTOCOL_VERSION,
        files: vec![file(&missing, None), file(&valid, None)],
    })
    .expect("per-file errors do not fail the batch");

    assert!(!response.files[0].scanned);
    assert!(response.files[0].error.is_some());
    assert!(response.files[0].findings.is_empty());
    assert!(response.files[1].scanned);
    assert_eq!(response.files[1].findings.len(), 1);

    fs::remove_dir_all(root).expect("temporary directory removed");
}

#[test]
fn batch_honors_max_bytes_and_reports_counts() {
    let root = temp_dir("bounds");
    let path = root.join("bounded.env");
    let first_line = "APP_NAME=Test\n";
    let content = format!("{first_line}AKIAIOSFODNN7EXAMPLE\n");
    fs::write(&path, &content).expect("file written");

    let response = scan_batch(BatchScanRequest {
        protocol_version: PROTOCOL_VERSION,
        files: vec![file(&path, Some(first_line.len()))],
    })
    .expect("batch succeeds");
    let result = &response.files[0];

    assert!(result.findings.is_empty());
    assert_eq!(result.bytes_scanned, first_line.len());
    assert_eq!(result.line_count, 1);

    fs::remove_dir_all(root).expect("temporary directory removed");
}

#[test]
fn batch_scans_valid_unicode_text_instead_of_treating_it_as_binary() {
    let root = temp_dir("unicode-text");
    let path = root.join("設定.env");
    let content = format!(
        "{}\nAWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n",
        "これは設定ファイルです".repeat(80)
    );
    fs::write(&path, &content).expect("unicode file written");

    let response = scan_batch(BatchScanRequest {
        protocol_version: PROTOCOL_VERSION,
        files: vec![file(&path, None)],
    })
    .expect("batch succeeds");

    assert!(response.files[0].scanned);
    assert_eq!(response.files[0].line_count, 2);
    assert_eq!(response.files[0].findings[0].kind, "AWS_ACCESS_KEY_ID");

    fs::remove_dir_all(root).expect("temporary directory removed");
}

#[test]
fn batch_rejects_unknown_protocol_version() {
    let error = scan_batch(BatchScanRequest {
        protocol_version: PROTOCOL_VERSION + 1,
        files: Vec::new(),
    })
    .expect_err("unknown protocol must fail");

    assert!(error.to_string().contains("protocol version"));
}

#[test]
fn scan_batch_binary_accepts_json_and_emits_json() {
    let root = temp_dir("binary");
    let path = root.join("secret.env");
    fs::write(&path, "AKIAIOSFODNN7EXAMPLE\n").expect("file written");
    let request = serde_json::json!({
        "protocol_version": PROTOCOL_VERSION,
        "files": [{"path": path, "max_bytes": null}],
    });

    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_harpocrates-rust-scanner"))
        .arg("scan-batch")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("binary starts");
    use std::io::Write;
    child
        .stdin
        .as_mut()
        .expect("stdin is piped")
        .write_all(request.to_string().as_bytes())
        .expect("request written");
    let output = child.wait_with_output().expect("binary exits");

    assert!(output.status.success());
    let response: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("response is JSON");
    assert_eq!(response["protocol_version"], PROTOCOL_VERSION);
    assert_eq!(
        response["files"][0]["findings"][0]["type"],
        "AWS_ACCESS_KEY_ID"
    );

    fs::remove_dir_all(root).expect("temporary directory removed");
}
