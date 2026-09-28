use harpocrates_rust_scanner::{ScanOptions, scan_text};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let mut fixture = String::new();
    for index in 0..1_000 {
        fixture.push_str(&format!(
            "const value_{index} = \"ordinary-configuration-value-{index}\";\n"
        ));
    }
    fixture.push_str("api_secret = \"aB3dEfGhIjKlMnOpQrStUvWxYz012345\";\n");

    let options = ScanOptions {
        file: Some("benchmark.rs".to_owned()),
    };
    for _ in 0..20 {
        black_box(scan_text(black_box(&fixture), black_box(&options)).expect("scan succeeds"));
    }

    let iterations = 500_u32;
    let started = Instant::now();
    for _ in 0..iterations {
        black_box(scan_text(black_box(&fixture), black_box(&options)).expect("scan succeeds"));
    }
    let elapsed = started.elapsed();
    let mean_ms = elapsed.as_secs_f64() * 1_000.0 / f64::from(iterations);
    let throughput_mib_s =
        fixture.len() as f64 * f64::from(iterations) / elapsed.as_secs_f64() / 1_048_576.0;

    println!(
        "{{\"iterations\":{iterations},\"fixture_bytes\":{},\"mean_ms\":{mean_ms:.6},\"throughput_mib_s\":{throughput_mib_s:.3}}}",
        fixture.len()
    );
}
