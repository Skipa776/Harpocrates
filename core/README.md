# harpo-native

The regex and entropy rules of [Harpo](https://github.com/Skipa776/Harpocrates) in Rust, as a Python extension module. Install it through Harpo:

```bash
pip install "harpo-scan[native]"
```

Harpo's read tool, hook, egress gate and scanner then run these rules in process. Without it, they run the same rules in Python. Harpo uses the module only when its rules are exactly those of the installed `harpo-scan`, and otherwise falls back to Python, so a mismatched version changes speed, not findings.

Prebuilt wheels cover Linux and macOS on x86_64 and arm64, for CPython 3.10 and later (abi3). The sdist builds with Rust and maturin.

MIT license.
