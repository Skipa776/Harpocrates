use harpocrates_rust_scanner::{
    BatchScanRequest, DirectoryScanRequest, ScanOptions, scan_batch, scan_directory, scan_file,
    scan_text,
};
use serde::Serialize;
use std::env;
use std::io::{self, Read};
use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let command = args.next().ok_or_else(usage)?;
    match command.as_str() {
        "scan-text" => {
            let mut file = None;
            while let Some(argument) = args.next() {
                match argument.as_str() {
                    "--file" => file = Some(args.next().ok_or_else(usage)?),
                    _ => return Err(usage()),
                }
            }
            let mut text = String::new();
            io::stdin()
                .read_to_string(&mut text)
                .map_err(|error| format!("failed to read stdin: {error}"))?;
            let findings =
                scan_text(&text, &ScanOptions { file }).map_err(|error| error.to_string())?;
            write_output(&findings)?;
        }
        "scan-file" => {
            let path = args.next().ok_or_else(usage)?;
            let mut max_bytes = None;
            while let Some(argument) = args.next() {
                match argument.as_str() {
                    "--max-bytes" => {
                        max_bytes = Some(
                            args.next()
                                .ok_or_else(usage)?
                                .parse::<usize>()
                                .map_err(|error| format!("invalid --max-bytes: {error}"))?,
                        );
                    }
                    _ => return Err(usage()),
                }
            }
            let findings = scan_file(path, max_bytes).map_err(|error| error.to_string())?;
            write_output(&findings)?;
        }
        "scan-batch" => {
            if args.next().is_some() {
                return Err(usage());
            }
            let request: BatchScanRequest = serde_json::from_reader(io::stdin())
                .map_err(|error| format!("failed to decode batch request: {error}"))?;
            let response = scan_batch(request).map_err(|error| error.to_string())?;
            write_output(&response)?;
        }
        "scan-directory" => {
            if args.next().is_some() {
                return Err(usage());
            }
            let request: DirectoryScanRequest = serde_json::from_reader(io::stdin())
                .map_err(|error| format!("failed to decode directory request: {error}"))?;
            let response = scan_directory(request).map_err(|error| error.to_string())?;
            write_output(&response)?;
        }
        _ => return Err(usage()),
    }
    Ok(())
}

fn write_output(value: &impl Serialize) -> Result<(), String> {
    serde_json::to_writer(io::stdout(), value)
        .map_err(|error| format!("failed to encode scan result: {error}"))
}

fn usage() -> String {
    "usage: harpocrates-rust-scanner scan-text [--file PATH] | scan-file PATH [--max-bytes N] | scan-batch | scan-directory"
        .to_owned()
}
