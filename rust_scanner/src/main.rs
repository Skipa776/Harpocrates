use harpocrates_rust_scanner::{ScanOptions, scan_file, scan_text};
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
    let findings = match command.as_str() {
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
            scan_text(&text, &ScanOptions { file }).map_err(|error| error.to_string())?
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
            scan_file(path, max_bytes).map_err(|error| error.to_string())?
        }
        _ => return Err(usage()),
    };

    serde_json::to_writer(io::stdout(), &findings)
        .map_err(|error| format!("failed to encode scan result: {error}"))?;
    Ok(())
}

fn usage() -> String {
    "usage: harpocrates-rust-scanner scan-text [--file PATH] | scan-file PATH [--max-bytes N]"
        .to_owned()
}
