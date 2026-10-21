//! FR-CORE-05: a file is scanned whole, in pieces, and gives the findings a scan of the whole text gives.

use harpocrates_rust_scanner::{PIECE_OVERLAP, RawFinding, ScanOptions, scan_file, scan_text};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_file(label: &str, content: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock is after epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "harpocrates-stream-{label}-{}-{nonce}.txt",
        std::process::id()
    ));
    fs::write(&path, content).expect("file written");
    path
}

fn regex_keys(findings: &[RawFinding]) -> Vec<(String, usize, Option<usize>, String)> {
    let mut keys: Vec<_> = findings
        .iter()
        .filter(|finding| finding.kind != "ENTROPY_CANDIDATE" && finding.kind != "ML_CANDIDATE")
        .map(|finding| {
            (
                finding.kind.clone(),
                finding.line,
                finding.token_start,
                finding.token.clone(),
            )
        })
        .collect();
    keys.sort();
    keys
}

/// A line of filler several pieces long with the AWS documentation example key placed on and around piece
/// boundaries, plus a multi-byte character just before each boundary.
fn long_line(pieces: usize) -> (String, usize) {
    let piece = 2 * PIECE_OVERLAP; // what a tiny memory cap gives
    let mut line = String::new();
    while line.len() < pieces * piece {
        line.push_str("lorem ipsum dolor sit amet ");
    }
    let mut planted = 0;
    for boundary in (1..pieces).map(|k| k * (piece - PIECE_OVERLAP)) {
        for at in [boundary - 3, boundary - PIECE_OVERLAP + 7, boundary + 11] {
            let mut cut = at.min(line.len() - 1);
            while !line.is_char_boundary(cut) {
                cut -= 1;
            }
            line.insert_str(cut, " é AKIAIOSFODNN7EXAMPLE ");
            planted += 1;
        }
    }
    (line, planted)
}

#[test]
fn a_long_line_in_pieces_gives_the_whole_line_findings() {
    let (line, planted) = long_line(5);
    let content = format!("first\n  # {line}\nAKIAIOSFODNN7EXAMPLE\nlast");
    let path = temp_file("long", &content);
    let file = path.to_string_lossy().into_owned();
    let whole = scan_text(&content, &ScanOptions { file: Some(file) }).expect("text scan");
    let streamed = scan_file(&path, Some(1)).expect("file scan");

    assert_eq!(regex_keys(&streamed), regex_keys(&whole));
    assert_eq!(regex_keys(&streamed).len(), planted + 1);
    fs::remove_file(path).expect("file removed");
}

#[test]
fn ordinary_files_give_identical_findings_at_any_cap() {
    let content = "API=AKIAIOSFODNN7EXAMPLE\n\n// token = \"ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"\r\nplain text\nno newline at the end";
    let path = temp_file("ordinary", content);
    let file = path.to_string_lossy().into_owned();
    let whole = scan_text(content, &ScanOptions { file: Some(file) }).expect("text scan");
    for cap in [None, Some(1), Some(1 << 30)] {
        assert_eq!(scan_file(&path, cap).expect("file scan"), whole);
    }
    fs::remove_file(path).expect("file removed");
}
