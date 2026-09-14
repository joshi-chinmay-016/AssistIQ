"""
AssistIQ: Intent Classification Evaluation Harness
Evaluates Baseline 1 (Majority), Baseline 2 (TF-IDF + Logistic Regression),
and Proposed Model (TF-IDF + LinearSVC) on the isolated 25% test set (N=50).
Generates summary metrics, per-class metrics, confusion matrices (CSV + PNG),
and detailed misclassification error analysis.
"""

import os
import sys
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Use headless Agg backend for matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix
)

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.src.intent.data import INTENT_TAXONOMY, load_and_split_data, find_dataset_path
from backend.src.intent.models import (
    build_majority_baseline,
    build_tfidf_logistic_regression,
    build_proposed_model
)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str,
    output_path: str
) -> None:
    """
    Plots and saves a high-resolution confusion matrix with count annotations.
    """
    fig, ax = plt.subplots(figsize=(11, 9))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_xlabel("Predicted Intent", fontsize=11, labelpad=10)
    ax.set_ylabel("True Intent", fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=13, pad=20, fontweight="bold")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color,
                    fontsize=9, fontweight="bold" if val > 0 else "normal")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def run_intent_evaluation(
    dataset_path: Optional[str] = None,
    results_dir: Optional[str] = None,
    random_state: int = 2026,
    test_size: float = 0.25
) -> Dict[str, Any]:
    """
    Executes the intent evaluation harness across all three models:
    1. Majority Baseline (Minimum class-imbalance baseline)
    2. TF-IDF + Logistic Regression (Standard linear probabilistic baseline)
    3. Proposed Model: TF-IDF + LinearSVC (Geometric max-margin classifier)

    Guarantees strict train/test separation (N=150 train, N=50 test).
    """
    if dataset_path is None:
        dataset_path = find_dataset_path("golden_set.csv")
    if results_dir is None:
        results_dir = os.path.join(PROJECT_ROOT, "evaluation", "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n==================================================================")
    print("ASSISTIQ INTENT CLASSIFICATION EVALUATION HARNESS")
    print("==================================================================")
    print(f"Dataset path:       {dataset_path}")
    print(f"Results dir:        {results_dir}")
    print(f"Split:              75% Train (N=150) / 25% Test (N=50) [Stratified]")
    print(f"Random seed:        {random_state}")

    # 1. Stratified split (strictly unseen test set)
    X_train, X_test, y_train, y_test, train_df, test_df = load_and_split_data(
        filepath=dataset_path,
        test_size=test_size,
        random_state=random_state
    )

    models = {
        "majority_baseline": build_majority_baseline(),
        "tfidf_logistic_regression": build_tfidf_logistic_regression(random_state=random_state),
        "proposed_model": build_proposed_model(random_state=random_state)
    }

    summary_records = []
    per_class_records = []
    predictions_map = {}
    cm_dict = {}

    for model_name, model in models.items():
        print(f"\n--- Training [{model_name}] ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions_map[model_name] = y_pred

        # Macro and weighted metrics
        acc = accuracy_score(y_test, y_pred)
        macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        weighted_rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        summary_records.append({
            "model": model_name,
            "accuracy": round(acc, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_precision": round(weighted_prec, 4),
            "weighted_recall": round(weighted_rec, 4),
            "weighted_f1": round(weighted_f1, 4)
        })

        print(f"  Accuracy:           {acc:.4f}")
        print(f"  Macro Precision:    {macro_prec:.4f}")
        print(f"  Macro Recall:       {macro_rec:.4f}")
        print(f"  Macro F1:           {macro_f1:.4f}")
        print(f"  Weighted F1:        {weighted_f1:.4f}")

        # Per-class metrics across all 11 taxonomy classes
        prec, rec, f1, support = precision_recall_fscore_support(
            y_test,
            y_pred,
            labels=INTENT_TAXONOMY,
            zero_division=0
        )
        for i, intent in enumerate(INTENT_TAXONOMY):
            per_class_records.append({
                "model": model_name,
                "intent": intent,
                "precision": round(float(prec[i]), 4),
                "recall": round(float(rec[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i])
            })

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=INTENT_TAXONOMY)
        cm_dict[model_name] = cm

        # Save plot
        cm_png_path = os.path.join(results_dir, f"confusion_matrix_{model_name}.png")
        plot_confusion_matrix(
            cm=cm,
            labels=INTENT_TAXONOMY,
            title=f"Confusion Matrix: {model_name} (Test Set, N=50)",
            output_path=cm_png_path
        )

    # 2. Save summary metrics table: intent_metrics.csv
    summary_df = pd.DataFrame(summary_records)
    metrics_csv_path = os.path.join(results_dir, "intent_metrics.csv")
    summary_df.to_csv(metrics_csv_path, index=False)
    # Also maintain intent_results.csv for backwards compatibility
    legacy_results_path = os.path.join(results_dir, "intent_results.csv")
    summary_df[["model", "accuracy", "macro_f1", "weighted_f1"]].to_csv(legacy_results_path, index=False)
    print(f"\n[OK] Saved metrics table to: {metrics_csv_path}")

    # 3. Save per-class results: intent_per_class_results.csv
    per_class_df = pd.DataFrame(per_class_records)
    per_class_csv_path = os.path.join(results_dir, "intent_per_class_results.csv")
    per_class_df.to_csv(per_class_csv_path, index=False)
    print(f"[OK] Saved per-class metrics to: {per_class_csv_path}")

    # 4. Save confusion matrix CSV for Proposed Model
    proposed_cm = cm_dict["proposed_model"]
    cm_df = pd.DataFrame(proposed_cm, index=INTENT_TAXONOMY, columns=INTENT_TAXONOMY)
    cm_csv_path = os.path.join(results_dir, "intent_confusion_matrix.csv")
    cm_df.to_csv(cm_csv_path)
    print(f"[OK] Saved confusion matrix CSV to: {cm_csv_path}")

    # 5. Proposed Model Error Analysis: intent_errors.csv
    # LinearSVC outputs margin distances from decision_function.
    # We apply softmax over margins to produce a margin-derived confidence proxy.
    svc_model = models["proposed_model"]
    svc_preds = predictions_map["proposed_model"]
    decision_scores = svc_model.decision_function(X_test)
    classes = list(svc_model.classes_)

    # Softmax over decision function margins
    exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
    margin_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    max_conf = np.max(margin_probs, axis=1)

    eval_df = pd.DataFrame({
        "tweet_id": test_df["tweet_id"].values,
        "customer_text": test_df["text"].values,
        "true_intent": y_test.values,
        "predicted_intent": svc_preds,
        "margin_confidence": np.round(max_conf, 4)
    })
    eval_df["test_pos"] = np.arange(len(eval_df))

    # Misclassified examples
    error_mask = (eval_df["true_intent"] != eval_df["predicted_intent"])
    errors_df = eval_df[error_mask].copy()

    top2_intents, top2_confs = [], []
    top3_intents, top3_confs = [], []

    for test_pos in errors_df["test_pos"].values:
        dist = margin_probs[int(test_pos)]
        sorted_indices = np.argsort(dist)[::-1]
        t2 = classes[sorted_indices[1]] if len(sorted_indices) > 1 else "None"
        p2 = float(dist[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
        t3 = classes[sorted_indices[2]] if len(sorted_indices) > 2 else "None"
        p3 = float(dist[sorted_indices[2]]) if len(sorted_indices) > 2 else 0.0

        top2_intents.append(t2)
        top2_confs.append(round(p2, 4))
        top3_intents.append(t3)
        top3_confs.append(round(p3, 4))

    errors_df["top_2_intent"] = top2_intents
    errors_df["top_2_confidence"] = top2_confs
    errors_df["top_3_intent"] = top3_intents
    errors_df["top_3_confidence"] = top3_confs
    errors_df = errors_df.drop(columns=["test_pos"])

    errors_csv_path = os.path.join(results_dir, "intent_errors.csv")
    errors_df.to_csv(errors_csv_path, index=False)
    print(f"[OK] Saved error analysis to: {errors_csv_path} ({len(errors_df)} misclassified / {len(X_test)} test examples)")

    print("\n==================================================================")
    print("INTENT BENCHMARK RESULTS TABLE")
    print("==================================================================")
    print(summary_df.to_string(index=False))
    print("==================================================================\n")

    return {
        "summary": summary_df,
        "per_class": per_class_df,
        "confusion_matrix": cm_df,
        "errors": errors_df
    }


if __name__ == "__main__":
    run_intent_evaluation()
