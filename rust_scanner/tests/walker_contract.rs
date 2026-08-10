use harpocrates_rust_scanner::{DirectoryScanRequest, PROTOCOL_VERSION, scan_directory};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock is after epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "harpocrates-walker-{label}-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir_all(&path).expect("temporary directory is created");
    path
}

fn request(root: PathBuf, recursive: bool, ignores: &[&str]) -> DirectoryScanRequest {
    DirectoryScanRequest {
        protocol_version: PROTOCOL_VERSION,
        root,
        recursive,
        max_file_size: 10 * 1024 * 1024,
        ignore_patterns: ignores.iter().map(|value| (*value).to_owned()).collect(),
    }
}

#[test]
fn walker_prunes_exact_and_globbed_directories() {
    let root = temp_dir("prune");
    for directory in ["node_modules", "build-generated", "src"] {
        fs::create_dir(root.join(directory)).expect("directory created");
        fs::write(
            root.join(directory).join("secret.env"),
            "AKIAIOSFODNN7EXAMPLE\n",
        )
        .expect("file written");
    }

    let response = scan_directory(request(root.clone(), true, &["node_modules", "build-*"]))
        .expect("directory scan succeeds");
    let paths: Vec<_> = response.files.iter().map(|file| &file.path).collect();

    assert_eq!(paths, vec![&root.join("src/secret.env")]);

    fs::remove_dir_all(root).expect("temporary directory removed");
}

#[test]
fn walker_supports_glob_character_classes() {
    let root = temp_dir("character-class");
    for file in ["a.env", "b.env", "c.env"] {
        fs::write(root.join(file), "APP_NAME=Test\n").expect("file written");
    }

    let response = scan_directory(request(root.clone(), true, &["[ab].env"]))
        .expect("directory scan succeeds");
    let paths: Vec<_> = response.files.iter().map(|file| &file.path).collect();

    assert_eq!(paths, vec![&root.join("c.env")]);

    fs::remove_dir_all(root).expect("temporary directory removed");
}

#[test]
fn walker_ignores_cache_directories() {
    let root = temp_dir("cache");
    for directory in [
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".taskmaster",
        ".claude",
    ] {
        fs::create_dir(root.join(directory)).expect("directory created");
        fs::write(
            root.join(directory).join("cache.json"),
            "AKIAIOSFODNN7EXAMPLE\n",
        )
        .expect("file written");
    }

    let response = scan_directory(request(
        root.clone(),
        true,
        &[
            ".mypy_cache",
            ".ruff_cache",
            ".pytest_cache",
            ".taskmaster",
            ".claude",
        ],
    ))
    .expect("directory scan succeeds");

    assert!(response.files.is_empty());

    fs::remove_dir_all(root).expect("temporary directory removed");
}

#[test]
fn walker_does_not_implicitly_honor_gitignore() {
    let root = temp_dir("gitignore");
    fs::write(root.join(".gitignore"), ".env\n").expect("gitignore written");
    fs::write(root.join(".env"), "AKIAIOSFODNN7EXAMPLE\n").expect("env written");

    let response =
        scan_directory(request(root.clone(), true, &[])).expect("directory scan succeeds");

    assert!(
        response
            .files
            .iter()
            .any(|file| file.path == root.join(".env"))
    );

    fs::remove_dir_all(root).expect("temporary directory removed");
}

#[test]
fn non_recursive_walk_is_sorted_and_excludes_nested_files() {
    let root = temp_dir("flat");
    fs::create_dir(root.join("nested")).expect("directory created");
    fs::write(root.join("z.txt"), "z\n").expect("file written");
    fs::write(root.join("a.txt"), "a\n").expect("file written");
    fs::write(root.join("nested/secret.env"), "AKIAIOSFODNN7EXAMPLE\n").expect("file written");

    let response =
        scan_directory(request(root.clone(), false, &[])).expect("directory scan succeeds");
    let paths: Vec<_> = response
        .files
        .iter()
        .map(|file| file.path.clone())
        .collect();

    assert_eq!(paths, vec![root.join("a.txt"), root.join("z.txt")]);

    fs::remove_dir_all(root).expect("temporary directory removed");
}

#[test]
fn parallel_walk_preserves_order_findings_and_accounting() {
    let root = temp_dir("parallel");
    for index in 0..40 {
        let content = if index % 2 == 0 {
            "AKIAIOSFODNN7EXAMPLE\n"
        } else {
            "APP_NAME=Test\n"
        };
        fs::write(root.join(format!("file-{index:02}.env")), content).expect("file written");
    }

    // Forty files crosses the parallel threshold; the public response must
    // remain deterministic regardless of worker completion order.
    let response =
        scan_directory(request(root.clone(), true, &[])).expect("parallel directory scan succeeds");
    let paths: Vec<_> = response
        .files
        .iter()
        .map(|file| file.path.clone())
        .collect();
    let mut sorted_paths = paths.clone();
    sorted_paths.sort();

    assert_eq!(paths, sorted_paths);
    assert_eq!(response.files.len(), 40);
    assert_eq!(
        response
            .files
            .iter()
            .map(|file| file.line_count)
            .sum::<usize>(),
        40
    );
    assert_eq!(
        response
            .files
            .iter()
            .flat_map(|file| &file.findings)
            .count(),
        20
    );

    fs::remove_dir_all(root).expect("temporary directory removed");
}

#[cfg(unix)]
#[test]
fn walker_skips_symlinks() {
    use std::os::unix::fs::symlink;

    let root = temp_dir("symlink");
    let real = root.join("real.env");
    fs::write(&real, "AKIAIOSFODNN7EXAMPLE\n").expect("file written");
    symlink(&real, root.join("linked.env")).expect("symlink created");

    let response =
        scan_directory(request(root.clone(), true, &[])).expect("directory scan succeeds");

    assert_eq!(response.files.len(), 1);
    assert_eq!(response.files[0].path, real);

    fs::remove_dir_all(root).expect("temporary directory removed");
}
