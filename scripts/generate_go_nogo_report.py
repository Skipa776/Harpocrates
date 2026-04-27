#!/usr/bin/env python3
"""
Generate Go/No-Go production readiness report for Harpocrates ML model.

This script evaluates the model against production acceptance criteria
and generates a comprehensive decision report.

Usage:
    python scripts/generate_go_nogo_report.py [--model-path PATH] [--output FILE]

Criteria evaluated:
1. Aggregate Metrics (30%): Precision, Recall, F1, AUPRC
2. Per-Secret-Type Metrics (25%): 8/9 types must pass
3. Stress Tests (25%): 7/8 tests must pass
4. Calibration (10%): ECE, Brier, high-confidence accuracy
5. Stage Routing (10%): Routing rates and error rates
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class CriterionResult:
    """Result for a single go/no-go criterion."""

    name: str
    category: str
    requirement: str
    actual_value: float
    threshold: float
    passed: bool
    notes: str = ""


@dataclass
class CategoryResult:
    """Result for a category of criteria."""

    category: str
    weight: float
    passed_criteria: int
    total_criteria: int
    pass_requirement: str
    passed: bool
    criteria: List[CriterionResult] = field(default_factory=list)


@dataclass
class GoNoGoReport:
    """Complete Go/No-Go decision report."""

    timestamp: str
    model_version: str
    weighted_score: float
    decision: str  # "GO" or "NO-GO"
    category_results: List[CategoryResult] = field(default_factory=list)
    critical_failures: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# --- Threshold Definitions ---

AGGREGATE_THRESHOLDS = {
    "precision": 0.88,
    "recall": 0.92,
    "f1_score": 0.90,
    "auprc": 0.93,
}

PER_TYPE_THRESHOLDS = {
    "aws_access_key": {"recall": 0.98, "precision": 0.90},
    "aws_secret_key": {"recall": 0.95, "precision": 0.88},
    "github_token": {"recall": 0.98, "precision": 0.90},
    "stripe_live_key": {"recall": 0.99, "precision": 0.92},
    "openai_key": {"recall": 0.95, "precision": 0.88},
    "jwt": {"recall": 0.90, "precision": 0.85},
    "pem_private_key": {"recall": 0.95, "precision": 0.90},
    "database_url": {"recall": 0.90, "precision": 0.85},
    "generic_entropy": {"recall": 0.85, "precision": 0.80},
}

STRESS_TEST_THRESHOLDS = {
    "prefix_collision": {"metric": "precision", "threshold": 0.85},
    "git_sha_masquerading": {"metric": "recall", "threshold": 0.90},
    "context_inversion": {"metric": "accuracy", "threshold": 0.80},
    "stage_routing_balance": {"metric": "stage_a_reject_rate", "min": 0.25, "max": 0.40},
    "vendor_neutralization": {"metric": "variance", "threshold": 0.10},
    "token_length_attack": {"metric": "variance", "threshold": 0.10},
    "confidence_calibration": {"metric": "ece", "threshold": 0.10},
    "adversarial_perturbation": {"metric": "recall", "threshold": 0.90},
}

CALIBRATION_THRESHOLDS = {
    "ece": 0.10,
    "brier_score": 0.15,
    "high_confidence_accuracy": 0.95,
    "max_feature_importance": 0.15,
}

ROUTING_THRESHOLDS = {
    "stage_a_reject_rate": {"min": 0.20, "max": 0.45},
    "stage_a_accept_rate": {"min": 0.05, "max": 0.20},
    "stage_b_rate": {"min": 0.40, "max": 0.70},
    "stage_a_reject_fn_rate": {"max": 0.01},
    "stage_a_accept_fp_rate": {"max": 0.10},
}

CATEGORY_WEIGHTS = {
    "aggregate_metrics": 0.30,
    "per_type_metrics": 0.25,
    "stress_tests": 0.25,
    "calibration": 0.10,
    "stage_routing": 0.10,
}


def evaluate_aggregate_metrics(
    metrics: Dict[str, float],
) -> CategoryResult:
    """Evaluate aggregate metrics category."""
    criteria = []

    for name, threshold in AGGREGATE_THRESHOLDS.items():
        value = metrics.get(name, 0.0)
        passed = value >= threshold

        criteria.append(CriterionResult(
            name=name,
            category="aggregate_metrics",
            requirement=f">= {threshold:.0%}",
            actual_value=value,
            threshold=threshold,
            passed=passed,
        ))

    passed_count = sum(1 for c in criteria if c.passed)
    all_passed = passed_count == len(criteria)

    return CategoryResult(
        category="aggregate_metrics",
        weight=CATEGORY_WEIGHTS["aggregate_metrics"],
        passed_criteria=passed_count,
        total_criteria=len(criteria),
        pass_requirement="All 4 must pass",
        passed=all_passed,
        criteria=criteria,
    )


def evaluate_per_type_metrics(
    type_metrics: Dict[str, Dict[str, float]],
) -> CategoryResult:
    """Evaluate per-secret-type metrics category."""
    criteria = []

    for secret_type, thresholds in PER_TYPE_THRESHOLDS.items():
        metrics = type_metrics.get(secret_type, {})

        recall = metrics.get("recall", 0.0)
        precision = metrics.get("precision", 0.0)

        recall_passed = recall >= thresholds["recall"]
        precision_passed = precision >= thresholds["precision"]
        both_passed = recall_passed and precision_passed

        criteria.append(CriterionResult(
            name=secret_type,
            category="per_type_metrics",
            requirement=f"recall >= {thresholds['recall']:.0%}, precision >= {thresholds['precision']:.0%}",
            actual_value=recall,  # Primary metric
            threshold=thresholds["recall"],
            passed=both_passed,
            notes=f"recall={recall:.1%}, precision={precision:.1%}",
        ))

    passed_count = sum(1 for c in criteria if c.passed)
    # 8/9 types must pass
    category_passed = passed_count >= 8

    return CategoryResult(
        category="per_type_metrics",
        weight=CATEGORY_WEIGHTS["per_type_metrics"],
        passed_criteria=passed_count,
        total_criteria=len(criteria),
        pass_requirement="8/9 types must pass",
        passed=category_passed,
        criteria=criteria,
    )


def evaluate_stress_tests(
    stress_results: Dict[str, Dict[str, float]],
) -> CategoryResult:
    """Evaluate stress test category."""
    criteria = []

    for test_name, config in STRESS_TEST_THRESHOLDS.items():
        result = stress_results.get(test_name, {})

        if "min" in config and "max" in config:
            # Range check
            value = result.get(config["metric"], 0.0)
            passed = config["min"] <= value <= config["max"]
            requirement = f"{config['min']:.0%} <= {config['metric']} <= {config['max']:.0%}"
            threshold = (config["min"] + config["max"]) / 2
        else:
            # Threshold check
            value = result.get(config["metric"], 0.0)
            passed = value <= config["threshold"] if config["metric"] in ("variance", "ece") else value >= config["threshold"]
            if config["metric"] in ["variance", "ece"]:
                requirement = f"{config['metric']} <= {config['threshold']:.0%}"
            else:
                requirement = f"{config['metric']} >= {config['threshold']:.0%}"
            threshold = config["threshold"]

        criteria.append(CriterionResult(
            name=test_name,
            category="stress_tests",
            requirement=requirement,
            actual_value=value,
            threshold=threshold,
            passed=passed,
        ))

    passed_count = sum(1 for c in criteria if c.passed)
    # 7/8 tests must pass
    category_passed = passed_count >= 7

    return CategoryResult(
        category="stress_tests",
        weight=CATEGORY_WEIGHTS["stress_tests"],
        passed_criteria=passed_count,
        total_criteria=len(criteria),
        pass_requirement="7/8 tests must pass",
        passed=category_passed,
        criteria=criteria,
    )


def evaluate_calibration(
    calibration_metrics: Dict[str, float],
) -> CategoryResult:
    """Evaluate calibration category."""
    criteria = []

    for name, threshold in CALIBRATION_THRESHOLDS.items():
        value = calibration_metrics.get(name, 0.0)

        # ECE, Brier, and max_feature_importance should be BELOW threshold
        # high_confidence_accuracy should be ABOVE threshold
        if name in ["ece", "brier_score", "max_feature_importance"]:
            passed = value <= threshold
            requirement = f"<= {threshold:.0%}"
        else:
            passed = value >= threshold
            requirement = f">= {threshold:.0%}"

        criteria.append(CriterionResult(
            name=name,
            category="calibration",
            requirement=requirement,
            actual_value=value,
            threshold=threshold,
            passed=passed,
        ))

    passed_count = sum(1 for c in criteria if c.passed)
    all_passed = passed_count == len(criteria)

    return CategoryResult(
        category="calibration",
        weight=CATEGORY_WEIGHTS["calibration"],
        passed_criteria=passed_count,
        total_criteria=len(criteria),
        pass_requirement="All 4 must pass",
        passed=all_passed,
        criteria=criteria,
    )


def evaluate_routing(
    routing_metrics: Dict[str, float],
) -> CategoryResult:
    """Evaluate stage routing category."""
    criteria = []

    for name, config in ROUTING_THRESHOLDS.items():
        value = routing_metrics.get(name, 0.0)

        if "min" in config and "max" in config:
            passed = config["min"] <= value <= config["max"]
            requirement = f"{config['min']:.0%} <= value <= {config['max']:.0%}"
            threshold = (config["min"] + config["max"]) / 2
        else:
            passed = value <= config["max"]
            requirement = f"<= {config['max']:.0%}"
            threshold = config["max"]

        criteria.append(CriterionResult(
            name=name,
            category="stage_routing",
            requirement=requirement,
            actual_value=value,
            threshold=threshold,
            passed=passed,
        ))

    passed_count = sum(1 for c in criteria if c.passed)
    all_passed = passed_count == len(criteria)

    return CategoryResult(
        category="stage_routing",
        weight=CATEGORY_WEIGHTS["stage_routing"],
        passed_criteria=passed_count,
        total_criteria=len(criteria),
        pass_requirement="All 5 must pass",
        passed=all_passed,
        criteria=criteria,
    )


def generate_go_nogo_report(
    aggregate_metrics: Dict[str, float],
    type_metrics: Dict[str, Dict[str, float]],
    stress_results: Dict[str, Dict[str, float]],
    calibration_metrics: Dict[str, float],
    routing_metrics: Dict[str, float],
    model_version: str = "unknown",
) -> GoNoGoReport:
    """Generate complete Go/No-Go report."""

    # Evaluate each category
    category_results = [
        evaluate_aggregate_metrics(aggregate_metrics),
        evaluate_per_type_metrics(type_metrics),
        evaluate_stress_tests(stress_results),
        evaluate_calibration(calibration_metrics),
        evaluate_routing(routing_metrics),
    ]

    # Calculate weighted score
    weighted_score = sum(
        cat.weight * (cat.passed_criteria / cat.total_criteria if cat.total_criteria > 0 else 0)
        for cat in category_results
    )

    # Check for critical failures (aggregate metrics must all pass)
    critical_failures = []
    aggregate_result = category_results[0]
    for criterion in aggregate_result.criteria:
        if not criterion.passed:
            critical_failures.append(
                f"CRITICAL: {criterion.name} = {criterion.actual_value:.1%} "
                f"(required {criterion.requirement})"
            )

    # Make decision
    # Go if: weighted score >= 85% AND no critical failures
    decision = "GO" if weighted_score >= 0.85 and not critical_failures else "NO-GO"

    # Generate recommendations
    recommendations = []
    if decision == "NO-GO":
        if critical_failures:
            recommendations.append("Address critical aggregate metric failures first")

        for cat in category_results:
            if not cat.passed:
                failed = [c.name for c in cat.criteria if not c.passed]
                recommendations.append(
                    f"Improve {cat.category}: {', '.join(failed)}"
                )

    return GoNoGoReport(
        timestamp=datetime.now().isoformat(),
        model_version=model_version,
        weighted_score=weighted_score,
        decision=decision,
        category_results=category_results,
        critical_failures=critical_failures,
        recommendations=recommendations,
    )


def simulate_metrics() -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """
    Simulate metrics for demonstration purposes.

    In production, these would come from actual model evaluation.
    """
    # Simulated aggregate metrics (current reported values)
    aggregate_metrics = {
        "precision": 0.9054,
        "recall": 0.9481,
        "f1_score": 0.926,
        "auprc": 0.94,  # Estimated
    }

    # Simulated per-type metrics
    type_metrics = {
        "aws_access_key": {"recall": 0.99, "precision": 0.92},
        "aws_secret_key": {"recall": 0.96, "precision": 0.90},
        "github_token": {"recall": 0.98, "precision": 0.91},
        "stripe_live_key": {"recall": 0.99, "precision": 0.93},
        "openai_key": {"recall": 0.95, "precision": 0.89},
        "jwt": {"recall": 0.91, "precision": 0.86},
        "pem_private_key": {"recall": 0.96, "precision": 0.91},
        "database_url": {"recall": 0.92, "precision": 0.87},
        "generic_entropy": {"recall": 0.86, "precision": 0.81},
    }

    # Simulated stress test results
    stress_results = {
        "prefix_collision": {"precision": 0.87},
        "git_sha_masquerading": {"recall": 0.91},
        "context_inversion": {"accuracy": 0.82},
        "stage_routing_balance": {"stage_a_reject_rate": 0.32},
        "vendor_neutralization": {"variance": 0.08},
        "token_length_attack": {"variance": 0.07},
        "confidence_calibration": {"ece": 0.08},
        "adversarial_perturbation": {"recall": 0.92},
    }

    # Simulated calibration metrics
    calibration_metrics = {
        "ece": 0.08,
        "brier_score": 0.12,
        "high_confidence_accuracy": 0.96,
        "max_feature_importance": 0.12,
    }

    # Simulated routing metrics
    routing_metrics = {
        "stage_a_reject_rate": 0.32,
        "stage_a_accept_rate": 0.12,
        "stage_b_rate": 0.56,
        "stage_a_reject_fn_rate": 0.005,
        "stage_a_accept_fp_rate": 0.06,
    }

    return aggregate_metrics, type_metrics, stress_results, calibration_metrics, routing_metrics


def print_report(report: GoNoGoReport) -> None:
    """Print report to console."""
    print("\n" + "=" * 70)
    print("HARPOCRATES ML GO/NO-GO PRODUCTION READINESS REPORT")
    print("=" * 70)

    print(f"\nTimestamp: {report.timestamp}")
    print(f"Model Version: {report.model_version}")

    # Decision banner
    print("\n" + "-" * 70)
    if report.decision == "GO":
        print("  ✅  DECISION: GO - Ready for production deployment")
    else:
        print("  ❌  DECISION: NO-GO - Not ready for production")
    print("-" * 70)

    print(f"\nWeighted Score: {report.weighted_score:.1%} (threshold: 85%)")

    # Critical failures
    if report.critical_failures:
        print("\n⚠️  CRITICAL FAILURES:")
        for failure in report.critical_failures:
            print(f"    {failure}")

    # Category results
    print("\n" + "-" * 70)
    print("CATEGORY BREAKDOWN")
    print("-" * 70)

    for cat in report.category_results:
        status = "✅" if cat.passed else "❌"
        print(f"\n{status} {cat.category.upper()} (weight: {cat.weight:.0%})")
        print(f"   Passed: {cat.passed_criteria}/{cat.total_criteria} ({cat.pass_requirement})")

        for criterion in cat.criteria:
            c_status = "✓" if criterion.passed else "✗"
            print(f"   [{c_status}] {criterion.name}: {criterion.actual_value:.1%} {criterion.requirement}")
            if criterion.notes:
                print(f"       {criterion.notes}")

    # Recommendations
    if report.recommendations:
        print("\n" + "-" * 70)
        print("RECOMMENDATIONS")
        print("-" * 70)
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Go/No-Go production readiness report"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to trained model for evaluation",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        help="JSON file with pre-computed metrics",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for JSON report",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use simulated metrics for demonstration",
    )

    args = parser.parse_args()

    # Load or simulate metrics
    if args.metrics_file:
        with open(args.metrics_file) as f:
            data = json.load(f)
        aggregate_metrics = data.get("aggregate_metrics", {})
        type_metrics = data.get("type_metrics", {})
        stress_results = data.get("stress_results", {})
        calibration_metrics = data.get("calibration_metrics", {})
        routing_metrics = data.get("routing_metrics", {})
        model_version = data.get("model_version", "loaded-from-file")
    elif args.simulate or not args.model_path:
        print("Using simulated metrics for demonstration...")
        aggregate_metrics, type_metrics, stress_results, calibration_metrics, routing_metrics = simulate_metrics()
        model_version = "simulated-v1.0"
    else:
        print("Error: --model-path provided but model evaluation is not yet implemented.", file=sys.stderr)
        print("Use --simulate for demonstration or --metrics-file with pre-computed metrics.", file=sys.stderr)
        sys.exit(1)

    # Generate report
    report = generate_go_nogo_report(
        aggregate_metrics=aggregate_metrics,
        type_metrics=type_metrics,
        stress_results=stress_results,
        calibration_metrics=calibration_metrics,
        routing_metrics=routing_metrics,
        model_version=model_version,
    )

    # Output report
    if args.output:
        output_data = {
            "timestamp": report.timestamp,
            "model_version": report.model_version,
            "weighted_score": report.weighted_score,
            "decision": report.decision,
            "category_results": [
                {
                    "category": cat.category,
                    "weight": cat.weight,
                    "passed_criteria": cat.passed_criteria,
                    "total_criteria": cat.total_criteria,
                    "pass_requirement": cat.pass_requirement,
                    "passed": cat.passed,
                    "criteria": [asdict(c) for c in cat.criteria],
                }
                for cat in report.category_results
            ],
            "critical_failures": report.critical_failures,
            "recommendations": report.recommendations,
        }
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"Report written to {args.output}")
    else:
        print_report(report)

    # Exit with appropriate code
    sys.exit(0 if report.decision == "GO" else 1)


if __name__ == "__main__":
    main()
