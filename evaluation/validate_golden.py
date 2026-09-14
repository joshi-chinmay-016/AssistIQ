"""
AssistIQ: Golden Evaluation Set Validation Module
Inspects dataset/golden_set.csv for data integrity, schema compliance,
duplicates, nulls, taxonomy validity, and class balance.
Produces a machine-readable validation report and preserves dataset provenance.
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional
import pandas as pd

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.src.intent.data import INTENT_TAXONOMY, find_dataset_path


def validate_golden_set_detailed(filepath: Optional[str] = None) -> Dict[str, Any]:
    """
    Performs comprehensive validation on the golden evaluation dataset:
    1. File existence and loadability
    2. Required columns presence
    3. Exact row count (200 expected)
    4. Null or empty text entries
    5. Null or empty intent labels
    6. Taxonomy compliance against the 11-intent schema
    7. Duplicate tweet IDs
    8. Duplicate customer text bodies
    9. Class distribution analysis
    10. Provenance documentation (AI-assisted candidate labels with partial human review)

    Returns:
        Dict[str, Any]: Structured validation report containing all check results and statistics.
    """
    if filepath is None:
        filepath = find_dataset_path("golden_set.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Golden evaluation set not found at: {filepath}")

    df = pd.read_csv(filepath)
    issues: List[str] = []
    checks: Dict[str, Any] = {}

    # Check 1: Required columns
    required_cols = ["tweet_id", "text", "intent"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    checks["required_columns"] = {
        "passed": len(missing_cols) == 0,
        "required": required_cols,
        "found": list(df.columns),
        "missing": missing_cols
    }
    if missing_cols:
        issues.append(f"Missing required column(s): {missing_cols}")

    # Check 2: Row count
    expected_rows = 200
    actual_rows = len(df)
    checks["row_count"] = {
        "passed": actual_rows == expected_rows,
        "expected": expected_rows,
        "actual": actual_rows
    }
    if actual_rows != expected_rows:
        issues.append(f"Expected {expected_rows} rows, found {actual_rows}")

    # Check 3: Missing / empty text
    empty_text_mask = df["text"].isna() | (df["text"].astype(str).str.strip() == "")
    empty_text_count = int(empty_text_mask.sum())
    checks["empty_text"] = {
        "passed": empty_text_count == 0,
        "empty_count": empty_text_count,
        "empty_tweet_ids": df.loc[empty_text_mask, "tweet_id"].tolist() if empty_text_count > 0 else []
    }
    if empty_text_count > 0:
        issues.append(f"Found {empty_text_count} empty or null text entries")

    # Check 4: Missing / empty intent
    empty_intent_mask = df["intent"].isna() | (df["intent"].astype(str).str.strip() == "")
    empty_intent_count = int(empty_intent_mask.sum())
    checks["empty_intent"] = {
        "passed": empty_intent_count == 0,
        "empty_count": empty_intent_count,
        "empty_tweet_ids": df.loc[empty_intent_mask, "tweet_id"].tolist() if empty_intent_count > 0 else []
    }
    if empty_intent_count > 0:
        issues.append(f"Found {empty_intent_count} empty or null intent entries")

    # Check 5: Taxonomy membership
    present_intents = set(df["intent"].dropna().unique())
    valid_intents = set(INTENT_TAXONOMY)
    invalid_intents = list(present_intents - valid_intents)
    checks["taxonomy_membership"] = {
        "passed": len(invalid_intents) == 0,
        "taxonomy_classes": len(INTENT_TAXONOMY),
        "classes_represented": len(present_intents),
        "invalid_classes": invalid_intents
    }
    if invalid_intents:
        issues.append(f"Found invalid intent labels not in taxonomy: {invalid_intents}")

    # Check 6: Duplicate tweet IDs
    dup_id_mask = df.duplicated(subset=["tweet_id"], keep=False)
    dup_id_count = int(dup_id_mask.sum())
    checks["duplicate_tweet_ids"] = {
        "passed": dup_id_count == 0,
        "duplicate_count": dup_id_count,
        "duplicate_ids": df.loc[dup_id_mask, "tweet_id"].tolist() if dup_id_count > 0 else []
    }
    if dup_id_count > 0:
        issues.append(f"Found {dup_id_count} duplicate tweet_id rows")

    # Check 7: Duplicate customer text bodies
    normalized_texts = df["text"].astype(str).str.strip().str.lower()
    dup_text_mask = normalized_texts.duplicated(keep=False)
    dup_text_count = int(dup_text_mask.sum())
    checks["duplicate_texts"] = {
        "passed": dup_text_count == 0,
        "duplicate_count": dup_text_count,
        "duplicate_sample_ids": df.loc[dup_text_mask, "tweet_id"].head(5).tolist() if dup_text_count > 0 else []
    }
    if dup_text_count > 0:
        issues.append(f"Found {dup_text_count} duplicate or identical text messages")

    # Class distribution analysis
    dist_series = df["intent"].value_counts()
    class_distribution = {}
    for intent_name in INTENT_TAXONOMY:
        count = int(dist_series.get(intent_name, 0))
        pct = round((count / actual_rows) * 100, 2) if actual_rows > 0 else 0.0
        class_distribution[intent_name] = {
            "count": count,
            "percentage": pct
        }

    # Provenance metadata (preserving absolute honesty)
    provenance = {
        "total_examples": actual_rows,
        "selected_brand": "SpotifyCares",
        "dataset_source": "Customer Support on Twitter (TWCS)",
        "labeling_method": "AI-assisted candidate generation with partial human spot-check review",
        "human_verification_status": "Partially human-reviewed (14 candidate disagreements verified and corrected; not 100% hand-labelled from scratch)",
        "intended_use": "Evaluation benchmark for Intent, Retrieval, Generation, and Escalation"
    }

    report = {
        "status": "PASSED" if len(issues) == 0 else "FAILED",
        "dataset_path": filepath,
        "total_issues": len(issues),
        "issues": issues,
        "checks": checks,
        "class_distribution": class_distribution,
        "provenance": provenance
    }

    return report


def run_validation(
    filepath: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Runs golden set validation, writes machine-readable report JSON, and prints human-readable summary.
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "evaluation", "results")
    os.makedirs(output_dir, exist_ok=True)

    report = validate_golden_set_detailed(filepath)

    out_path = os.path.join(output_dir, "golden_set_validation_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n==================================================================")
    print(f"GOLDEN EVALUATION SET VALIDATION: {report['status']}")
    print("==================================================================")
    print(f"Dataset path:       {report['dataset_path']}")
    print(f"Total rows:         {report['checks']['row_count']['actual']}")
    print(f"Unique Tweet IDs:   {report['checks']['row_count']['actual'] - report['checks']['duplicate_tweet_ids']['duplicate_count']}")
    print(f"Duplicate texts:    {report['checks']['duplicate_texts']['duplicate_count']}")
    print(f"Taxonomy classes:   {report['checks']['taxonomy_membership']['classes_represented']}/11 represented")
    print(f"Labeling Method:    {report['provenance']['labeling_method']}")
    print(f"Review Status:      {report['provenance']['human_verification_status']}")
    print("------------------------------------------------------------------")
    print("Class Distribution:")
    for intent, stats in report["class_distribution"].items():
        print(f"  - {intent:26s} : {stats['count']:3d} ({stats['percentage']:5.1f}%)")
    print("------------------------------------------------------------------")
    if report["issues"]:
        print("Identified Issues:")
        for iss in report["issues"]:
            print(f"  [!] {iss}")
    else:
        print("[OK] All validation integrity checks passed successfully.")
    print(f"Report saved to:    {out_path}")
    print("==================================================================\n")

    return report


if __name__ == "__main__":
    run_validation()
