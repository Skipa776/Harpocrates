"""
Command-line interface for Harpocrates secrets detection.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from Harpocrates.core.result import Severity
from Harpocrates.core.scanner import scan_directory, scan_file

app = typer.Typer(
    name="harpocrates",
    help="Harpocrates - Secrets detection for code repositories",
    add_completion=False,
)
console = Console()
error_console = Console(stderr=True)


@app.command()
def scan(
    path: Path = typer.Argument(..., help="File or directory to scan"),
    recursive: bool = typer.Option(
        True, "--recursive/--no-recursive", "-r",
        help="Scan directories recursively"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
    max_file_size: int = typer.Option(10, "--max-size", help="Max file size in MB"),
    ignore: Optional[str] = typer.Option(
        None, "--ignore",
        help="Comma-separated patterns to ignore"
    ),
    use_ml: bool = typer.Option(
        False, "--ml",
        help="Enable ML-based verification to reduce false positives"
    ),
    ml_threshold: float = typer.Option(
        0.5, "--ml-threshold",
        help="ML confidence threshold (0.0-1.0, default: 0.5)"
    ),
) -> None:
    """
    Scan a file or directory for secrets.

    Examples:

        # Scan a single file
        harpocrates scan config.env

        # Scan a directory recursively
        harpocrates scan ./my_project

        # Output as JSON
        harpocrates scan ./my_project --json

        # Ignore specific patterns
        harpocrates scan ./my_project --ignore "*.test.js,test_*"

        # Enable ML verification (reduces false positives)
        harpocrates scan ./my_project --ml

        # ML with custom threshold
        harpocrates scan ./my_project --ml --ml-threshold 0.7
    """
    if not path.exists():
        error_console.print(f"[red]✗[/red] Path not found: {path}")
        raise typer.Exit(code=2)  # Error exit code

    ignore_patterns = set(ignore.split(",")) if ignore else set()

    max_bytes = max_file_size * 1024 * 1024

    # Initialize ML verifier if requested
    verifier = None
    if use_ml:
        try:
            from Harpocrates.ml.ensemble import EnsembleVerifier
            from Harpocrates.ml.lightgbm_verifier import LightGBMVerifier
            from Harpocrates.ml.verifier import XGBoostVerifier

            # Attempt to load models in order of preference: Ensemble > XGBoost > LightGBM
            try:
                verifier = EnsembleVerifier.get_instance(threshold=ml_threshold)
                if not json_output:
                    console.print("[cyan]ℹ[/cyan] Ensemble ML verification enabled")
            except Exception:
                try:
                    verifier = XGBoostVerifier.get_instance(threshold=ml_threshold)
                    if not json_output:
                        console.print("[cyan]ℹ[/cyan] XGBoost ML verification enabled")
                except Exception:
                    try:
                        verifier = LightGBMVerifier.get_instance(threshold=ml_threshold)
                        if not json_output:
                            console.print("[cyan]ℹ[/cyan] LightGBM ML verification enabled")
                    except Exception:
                        error_console.print(
                            "[yellow]⚠[/yellow] No ML model found. "
                            "Train a model with 'harpocrates train'."
                        )
                        error_console.print("[yellow]⚠[/yellow] Falling back to standard detection")
        except ImportError:
            error_console.print(
                "[yellow]⚠[/yellow] ML dependencies not installed. "
                "Install with: pip install harpocrates[ml]"
            )
            error_console.print("[yellow]⚠[/yellow] Falling back to standard detection")

    # Perform scan
    if path.is_file():
        result = scan_file(
            path,
            max_file_size=max_bytes,
            verifier=verifier,
            ml_threshold=ml_threshold,
        )
    else:
        result = scan_directory(
            path,
            recursive=recursive,
            max_file_size=max_bytes,
            ignore_patterns=ignore_patterns,
            verifier=verifier,
            ml_threshold=ml_threshold,
        )

    # Handle errors
    if result.errors:
        for error in result.errors:
            error_console.print(f"[yellow]⚠[/yellow]  {error}")

    if json_output:
        print(json.dumps(result.to_dict(), indent=2))
        raise typer.Exit(code=1 if result.found_secrets else 0)  # 1=findings, 0=clean

    if not result.found_secrets:
        console.print("[green]✓[/green] No secrets detected")
        console.print(
            f"Scanned {result.scanned_files} files ({result.total_lines} lines) "
            f"in {result.duration_ms:.0f}ms"
        )
        raise typer.Exit(code=0)

    table = Table(
        title=f"Found {len(result.findings)} potential secrets",
        show_header=True
    )
    table.add_column("Severity", style="bold")
    table.add_column("Type", style="cyan")
    table.add_column("Location")
    table.add_column("Evidence")

    for finding in sorted(
        result.findings,
        key=lambda f: (f.severity.value, f.file or "", f.line or 0)
    ):
        # Color severity
        severity_colors = {
            Severity.CRITICAL: "red bold",
            Severity.HIGH: "red",
            Severity.MEDIUM: "yellow",
            Severity.LOW: "blue",
            Severity.INFO: "white",
        }
        severity_style = severity_colors.get(finding.severity, "white")
        severity_text = f"[{severity_style}]{finding.severity.value.upper()}[/{severity_style}]"

        # Format location
        location = f"{finding.file}:{finding.line}" if finding.file else "text input"

        table.add_row(
            severity_text,
            finding.type,
            location,
            finding.evidence.value,
        )

    console.print(table)
    console.print(
        f"\nScanned {result.scanned_files} files ({result.total_lines} lines) "
        f"in {result.duration_ms:.0f}ms"
    )
    console.print(
        f"[yellow]⚠[/yellow]  Found {result.critical_count} critical, "
        f"{result.high_count} high severity secrets"
    )

    raise typer.Exit(code=1)  # 1=findings detected


@app.command()
def version() -> None:
    """Display version information."""
    from Harpocrates import __version__
    console.print(f"Harpocrates version {__version__}")


@app.command()
def train(
    data: Path = typer.Argument(..., help="Path to training data JSONL file"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output path for model (auto-determined if not specified)"
    ),
    val_data: Optional[Path] = typer.Option(
        None, "--val-data",
        help="Path to validation data (if not provided, will split training data)"
    ),
    model_type: str = typer.Option(
        "xgboost", "--model-type", "-m",
        help="Model type: xgboost, lightgbm, or ensemble"
    ),
    cross_validate: bool = typer.Option(
        False, "--cross-validate", "-cv",
        help="Use k-fold cross validation instead of single split"
    ),
    folds: int = typer.Option(
        5, "--folds", "-k",
        help="Number of folds for cross validation"
    ),
    target_precision: float = typer.Option(
        0.95, "--target-precision",
        help="Target precision for threshold tuning"
    ),
    max_depth: int = typer.Option(
        6, "--max-depth",
        help="Maximum tree depth"
    ),
    learning_rate: float = typer.Option(
        0.1, "--learning-rate",
        help="Learning rate"
    ),
    n_estimators: int = typer.Option(
        100, "--n-estimators",
        help="Number of trees"
    ),
    seed: int = typer.Option(
        42, "--seed",
        help="Random seed"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q",
        help="Suppress verbose output"
    ),
) -> None:
    """
    Train an ML model for secrets detection.

    Examples:

        # Train XGBoost model (default)
        harpocrates train training_data.jsonl

        # Train with custom output path
        harpocrates train training_data.jsonl -o models/my_model.json

        # Train LightGBM model
        harpocrates train training_data.jsonl --model-type lightgbm

        # Train ensemble (XGBoost + LightGBM)
        harpocrates train training_data.jsonl --model-type ensemble

        # Use k-fold cross validation
        harpocrates train training_data.jsonl --cross-validate --folds 5

        # Custom training parameters
        harpocrates train training_data.jsonl --max-depth 8 --n-estimators 200
    """
    # Validate model type
    valid_model_types = ["xgboost", "lightgbm", "ensemble"]
    if model_type not in valid_model_types:
        error_console.print(
            f"[red]✗[/red] Invalid model type: {model_type}. "
            f"Must be one of: {', '.join(valid_model_types)}"
        )
        raise typer.Exit(code=2)

    if not data.exists():
        error_console.print(f"[red]✗[/red] Training data not found: {data}")
        raise typer.Exit(code=2)

    # Set default output path based on model type
    if output is None:
        if model_type == "lightgbm":
            output = Path("Harpocrates/ml/models/lightgbm_v1.txt")
        elif model_type == "ensemble":
            output = Path("Harpocrates/ml/models/ensemble_config.json")
        else:
            output = Path("Harpocrates/ml/models/xgboost_v1.json")

    try:
        from Harpocrates.training.dataset import Dataset
        from Harpocrates.training.train import (
            print_feature_importance,
            save_model,
            train_ensemble,
            train_lightgbm_model,
            train_model,
        )
    except ImportError as e:
        error_console.print(
            "[red]✗[/red] ML dependencies not installed. "
            "Install with: pip install harpocrates[ml]"
        )
        raise typer.Exit(code=2)

    if not quiet:
        console.print(f"[cyan]ℹ[/cyan] Loading training data from {data}...")

    dataset = Dataset.from_jsonl(data)
    if not quiet:
        console.print(dataset.summary())

    # Use cross-validation if requested
    if cross_validate:
        from Harpocrates.training.cross_validation import cross_validate_with_best_model

        if not quiet:
            console.print(f"\n[cyan]ℹ[/cyan] Running {folds}-fold cross validation with {model_type}...")

        model, cv_result = cross_validate_with_best_model(
            dataset=dataset,
            k=folds,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            target_precision=target_precision,
            seed=seed,
            verbose=not quiet,
            model_type=model_type,
        )

        threshold = cv_result.mean_threshold
        metrics = {
            "precision": cv_result.mean_precision,
            "recall": cv_result.mean_recall,
            "f1": cv_result.mean_f1,
            "auc_roc": cv_result.mean_auc_roc,
            "threshold": threshold,
            "cv_folds": folds,
            "cv_std_precision": cv_result.std_precision,
            "cv_std_recall": cv_result.std_recall,
        }
    else:
        # Load or create validation data
        if val_data:
            if not quiet:
                console.print(f"\n[cyan]ℹ[/cyan] Loading validation data from {val_data}...")
            val_dataset = Dataset.from_jsonl(val_data)
            train_dataset = dataset
        else:
            if not quiet:
                console.print("\n[cyan]ℹ[/cyan] Splitting into train/val sets (80/20)...")
            train_dataset, val_dataset, _ = dataset.split(
                train_ratio=0.8,
                val_ratio=0.2,
                seed=seed,
            )

        if not quiet:
            console.print(f"Train: {len(train_dataset)} samples")
            console.print(f"Val: {len(val_dataset)} samples")

        # Train model based on type
        if model_type == "ensemble":
            if not quiet:
                console.print("\n[cyan]ℹ[/cyan] Training ensemble (XGBoost + LightGBM)...")
            model, threshold, metrics = train_ensemble(
                train_data=train_dataset,
                val_data=val_dataset,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                target_precision=target_precision,
                verbose=not quiet,
            )
        elif model_type == "lightgbm":
            if not quiet:
                console.print("\n[cyan]ℹ[/cyan] Training LightGBM model...")
            model, threshold, metrics = train_lightgbm_model(
                train_data=train_dataset,
                val_data=val_dataset,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                target_precision=target_precision,
                verbose=not quiet,
            )
        else:
            if not quiet:
                console.print("\n[cyan]ℹ[/cyan] Training XGBoost model...")
            model, threshold, metrics = train_model(
                train_data=train_dataset,
                val_data=val_dataset,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                target_precision=target_precision,
                verbose=not quiet,
            )

    # Print feature importance
    if not quiet:
        if model_type == "ensemble":
            # For ensemble, print XGBoost feature importance
            xgb_model, _ = model
            console.print("\nTop 10 Feature Importance (XGBoost component):")
            print_feature_importance(xgb_model)
        else:
            print_feature_importance(model)

    # Save model
    save_model(
        model=model,
        output_path=output,
        threshold=threshold,
        metrics=metrics,
        model_type=model_type,
    )

    console.print(f"\n[green]✓[/green] Training complete!")
    if "precision" in metrics:
        console.print(f"  Precision: {metrics['precision']:.3f}")
        console.print(f"  Recall: {metrics['recall']:.3f}")
        console.print(f"  F1 Score: {metrics['f1']:.3f}")


@app.command("generate-data")
def generate_data(
    output: Path = typer.Option(
        Path("training_data.jsonl"), "--output", "-o",
        help="Output JSONL file path"
    ),
    count: int = typer.Option(
        1000, "--count", "-n",
        help="Number of examples to generate"
    ),
    balance: float = typer.Option(
        0.5, "--balance", "-b",
        help="Ratio of positive (secret) examples (0.0-1.0)"
    ),
    adversarial: bool = typer.Option(
        False, "--adversarial",
        help="Include hard negative examples for adversarial training"
    ),
    split: bool = typer.Option(
        False, "--split",
        help="Create train/val/test split (80/10/10)"
    ),
    seed: int = typer.Option(
        42, "--seed",
        help="Random seed for reproducibility"
    ),
) -> None:
    """
    Generate synthetic training data for ML model training.

    Examples:

        # Generate 1000 examples
        harpocrates generate-data -n 1000 -o training_data.jsonl

        # Generate with custom balance (70% secrets, 30% non-secrets)
        harpocrates generate-data -n 5000 --balance 0.7

        # Include hard negatives for better training
        harpocrates generate-data -n 5000 --adversarial

        # Create train/val/test split
        harpocrates generate-data -n 10000 --split -o data/train.jsonl
    """
    if balance < 0.0 or balance > 1.0:
        error_console.print(
            f"[red]✗[/red] Balance must be between 0.0 and 1.0, got: {balance}"
        )
        raise typer.Exit(code=2)

    try:
        from Harpocrates.training.generators.generate_data import (
            generate_adversarial_test_data,
            generate_training_data,
        )
    except ImportError as e:
        error_console.print(
            "[red]✗[/red] Training module not available. "
            f"Error: {e}"
        )
        raise typer.Exit(code=2)

    console.print(f"[cyan]ℹ[/cyan] Generating {count} training examples...")
    console.print(f"  Balance: {balance:.0%} positive / {1-balance:.0%} negative")
    console.print(f"  Adversarial: {'Yes' if adversarial else 'No'}")

    # Generate data
    if adversarial:
        # Mix standard and adversarial data
        adversarial_count = int(count * 0.3)  # 30% adversarial
        standard_count = count - adversarial_count

        standard_data = generate_training_data(
            count=standard_count,
            balance=balance,
            seed=seed,
        )
        adversarial_data = generate_adversarial_test_data(
            count=adversarial_count,
            seed=seed + 1,
        )
        data = standard_data + adversarial_data

        # Shuffle combined data
        import random
        random.seed(seed)
        random.shuffle(data)
    else:
        data = generate_training_data(
            count=count,
            balance=balance,
            seed=seed,
        )

    if split:
        # Create train/val/test split (80/10/10)
        n = len(data)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        # Determine output paths
        base = output.stem
        suffix = output.suffix or ".jsonl"
        parent = output.parent

        train_path = parent / f"{base}_train{suffix}"
        val_path = parent / f"{base}_val{suffix}"
        test_path = parent / f"{base}_test{suffix}"

        # Ensure output directory exists
        parent.mkdir(parents=True, exist_ok=True)

        # Write files
        _write_jsonl(train_path, train_data)
        _write_jsonl(val_path, val_data)
        _write_jsonl(test_path, test_data)

        console.print(f"\n[green]✓[/green] Generated split datasets:")
        console.print(f"  Train: {train_path} ({len(train_data)} examples)")
        console.print(f"  Val: {val_path} ({len(val_data)} examples)")
        console.print(f"  Test: {test_path} ({len(test_data)} examples)")
    else:
        # Ensure output directory exists
        output.parent.mkdir(parents=True, exist_ok=True)

        # Write single file
        _write_jsonl(output, data)

        # Count labels
        n_positive = sum(1 for d in data if d.get("label") == 1)
        n_negative = len(data) - n_positive

        console.print(f"\n[green]✓[/green] Generated {len(data)} examples to {output}")
        console.print(f"  Positive (secrets): {n_positive}")
        console.print(f"  Negative (non-secrets): {n_negative}")


def _write_jsonl(path: Path, data: list) -> None:
    """Write data to JSONL file."""
    with open(path, "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")


def main() -> None:
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
