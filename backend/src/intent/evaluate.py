"""
AssistIQ: Phase 1 Intent Classification Evaluation Pipeline
Evaluates Baseline 1 (Majority), Baseline 2 (TF-IDF + Logistic Regression),
and Proposed Model (TF-IDF + LinearSVC) on the isolated 25% test set.
Generates metrics, confusion matrices, predictions, and error analysis files.
"""

import os
import sys
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Use headless Agg backend for matplotlib before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix
)

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def find_project_root() -> str:
    """
    Finds the AssistIQ project root directory regardless of whether this script
    is called from repo root or inside backend/.
    """
    if os.path.exists("dataset") and os.path.isdir("dataset"):
        return os.path.abspath(".")
    if os.path.exists(os.path.join("..", "dataset")):
        return os.path.abspath("..")
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


# Resolve imports
project_root = find_project_root()
backend_dir = os.path.join(project_root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if project_root not in sys.path:
    sys.path.insert(1, project_root)

from src.intent.data import INTENT_TAXONOMY, load_and_split_data
from src.intent.models import (
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
    Plots a readable confusion matrix with all 11 taxonomy labels on both axes and counts in cells.
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

    # Add count annotations inside cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=9, fontweight="bold" if val > 0 else "normal")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def run_evaluation(
    dataset_path: Optional[str] = None,
    results_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the complete Phase 1 intent classification evaluation pipeline.
    """
    root = find_project_root()
    if dataset_path is None:
        dataset_path = os.path.join(root, "dataset", "golden_set.csv")
    if results_dir is None:
        results_dir = os.path.join(root, "evaluation", "results")

    os.makedirs(results_dir, exist_ok=True)

    # 1. Load and split dataset (unseen 25% test set)
    X_train, X_test, y_train, y_test, train_df, test_df = load_and_split_data(
        filepath=dataset_path,
        test_size=0.25,
        random_state=2026
    )

    models = {
        "majority_baseline": build_majority_baseline(),
        "tfidf_logistic_regression": build_tfidf_logistic_regression(random_state=2026),
        "proposed_model": build_proposed_model(random_state=2026)
    }

    results_summary = []
    per_class_records = []
    predictions_map = {}

    print("\n==================================================================")
    print("TRAINING AND EVALUATION")
    print("==================================================================")

    for model_name, model in models.items():
        print(f"\nTraining [{model_name}]...")
        # Train strictly on training data
        model.fit(X_train, y_train)

        # Predict strictly on unseen test data
        y_pred = model.predict(X_test)
        predictions_map[model_name] = y_pred

        # Core evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results_summary.append({
            "model": model_name,
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4)
        })

        print(f"-> Accuracy:    {acc:.4f}")
        print(f"-> Macro F1:    {macro_f1:.4f}")
        print(f"-> Weighted F1: {weighted_f1:.4f}")

        # Per-class metrics across all 11 taxonomy labels
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test,
            y_pred,
            labels=INTENT_TAXONOMY,
            zero_division=0
        )

        for i, intent in enumerate(INTENT_TAXONOMY):
            per_class_records.append({
                "model": model_name,
                "intent": intent,
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i])
            })

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=INTENT_TAXONOMY)
        cm_path = os.path.join(results_dir, f"confusion_matrix_{model_name}.png")
        plot_confusion_matrix(
            cm=cm,
            labels=INTENT_TAXONOMY,
            title=f"Confusion Matrix: {model_name} (Test Set, N=50)",
            output_path=cm_path
        )
        print(f"-> Confusion matrix saved to: {cm_path}")

    # 2. Save results table: intent_results.csv
    results_df = pd.DataFrame(results_summary)
    results_csv_path = os.path.join(results_dir, "intent_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\n Saved summary metrics to: {results_csv_path}")

    # 3. Save per-class results table: intent_per_class_results.csv
    per_class_df = pd.DataFrame(per_class_records)
    per_class_csv_path = os.path.join(results_dir, "intent_per_class_results.csv")
    per_class_df.to_csv(per_class_csv_path, index=False)
    print(f" Saved per-class metrics to: {per_class_csv_path}")

    # 4. Save test predictions for TF-IDF + Logistic Regression: tfidf_lr_predictions.csv
    lr_model = models["tfidf_logistic_regression"]
    lr_preds = predictions_map["tfidf_logistic_regression"]
    lr_probs = lr_model.predict_proba(X_test)
    classes = list(lr_model.classes_)

    max_probs = np.max(lr_probs, axis=1)

    predictions_df = pd.DataFrame({
        "tweet_id": test_df["tweet_id"].values,
        "text": test_df["text"].values,
        "true_intent": y_test.values,
        "predicted_intent": lr_preds,
        "max_class_probability": np.round(max_probs, 4)
    })
    preds_csv_path = os.path.join(results_dir, "tfidf_lr_predictions.csv")
    predictions_df.to_csv(preds_csv_path, index=False)
    print(f" Saved test predictions to: {preds_csv_path}")

    # 5. Error Analysis: intent_errors.csv (misclassified examples with top-3 predictions)
    # Explicitly associate each row with its original test-set positional index before filtering
    eval_df = predictions_df.copy()
    eval_df["test_pos"] = np.arange(len(eval_df))

    error_mask = (eval_df["true_intent"] != eval_df["predicted_intent"])
    errors_df = eval_df[error_mask].copy()

    top2_intents = []
    top2_probs = []
    top3_intents = []
    top3_probs = []

    for test_pos in errors_df["test_pos"].values:
        test_pos = int(test_pos)
        prob_dist = lr_probs[test_pos]
        sorted_indices = np.argsort(prob_dist)[::-1]

        t2_class = classes[sorted_indices[1]] if len(sorted_indices) > 1 else "None"
        t2_p = float(prob_dist[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
        t3_class = classes[sorted_indices[2]] if len(sorted_indices) > 2 else "None"
        t3_p = float(prob_dist[sorted_indices[2]]) if len(sorted_indices) > 2 else 0.0

        top2_intents.append(t2_class)
        top2_probs.append(round(t2_p, 4))
        top3_intents.append(t3_class)
        top3_probs.append(round(t3_p, 4))

    errors_df["top_2_intent"] = top2_intents
    errors_df["top_2_prob"] = top2_probs
    errors_df["top_3_intent"] = top3_intents
    errors_df["top_3_prob"] = top3_probs

    # Drop temporary positional index to preserve exact output schema
    errors_df = errors_df.drop(columns=["test_pos"])

    errors_csv_path = os.path.join(results_dir, "intent_errors.csv")
    errors_df.to_csv(errors_csv_path, index=False)
    print(f" Saved misclassified error analysis to: {errors_csv_path} ({len(errors_df)} errors / {len(X_test)} test examples)")

    print("\n==================================================================")
    print("FINAL PHASE 1 RESULTS TABLE")
    print("==================================================================")
    print(results_df.to_string(index=False))
    print("==================================================================")

    return {
        "results_summary": results_df,
        "per_class_results": per_class_df,
        "predictions": predictions_df,
        "errors": errors_df
    }


def main():
    print("Running AssistIQ Phase 1 Intent Classification Pipeline (backend)...")
    run_evaluation()


if __name__ == "__main__":
    main()
