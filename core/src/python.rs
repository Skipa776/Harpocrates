//! The Python extension module `harpo_native` (feature "python"): the detector's line scan, called in process by the
//! read tool, the hook, the gate and the Python scan engine, so they all run these rules without a subprocess.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// The findings of one line as JSON in the scanner protocol's finding format, or None when there are none, which is
/// most lines. The GIL is released while the line is scanned, so the gate's threads scan in parallel.
#[pyfunction]
#[pyo3(signature = (line, lineno, extension, continued=None))]
fn scan_line(
    py: Python<'_>,
    line: &str,
    lineno: usize,
    extension: &str,
    continued: Option<bool>,
) -> PyResult<Option<String>> {
    let findings = py
        .allow_threads(|| crate::scan_line_raw(line, lineno, extension, continued))
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    if findings.is_empty() {
        return Ok(None);
    }
    serde_json::to_string(&findings)
        .map(Some)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

/// Every pattern and constant the rules use, by its Python name, so Python can check it runs the same rules.
#[pyfunction]
fn patterns() -> Vec<(&'static str, String)> {
    crate::patterns()
}

#[pymodule]
fn harpo_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(scan_line, module)?)?;
    module.add_function(wrap_pyfunction!(patterns, module)?)?;
    module.add("PROTOCOL_VERSION", crate::PROTOCOL_VERSION)?;
    Ok(())
}
