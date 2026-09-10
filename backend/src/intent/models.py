"""
AssistIQ: Intent Classification Models
Implements Baseline 1 (Majority), Baseline 2 (TF-IDF + Logistic Regression),
and Proposed Model (TF-IDF + LinearSVC).
"""

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


def build_majority_baseline() -> DummyClassifier:
    """
    BASELINE 1: Majority-Class Classifier.
    Finds the most frequent class in the training set and always predicts it.
    Establishes the minimum baseline performance for class imbalance.
    """
    return DummyClassifier(strategy="most_frequent")


def build_tfidf_logistic_regression(random_state: int = 2026) -> Pipeline:
    """
    BASELINE 2: TF-IDF + Logistic Regression Pipeline.
    Uses n-gram character/word statistics and balanced class weighting.
    Preprocessing is fully encapsulated within the sklearn Pipeline to prevent data leakage.
    """
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=10000
            )
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state
            )
        )
    ])


def build_proposed_model(random_state: int = 2026) -> Pipeline:
    """
    PROPOSED MODEL: TF-IDF + Linear Support Vector Classifier (LinearSVC).
    Maximizes the geometric margin between intent classes in high-dimensional n-gram space.
    Balanced class weights penalize minority misclassifications more heavily.
    Fast, deterministic, and avoids the heavy latency and compute overhead of fine-tuning transformers.

    Note: LinearSVC outputs decision function distances (margin distances) rather than
    calibrated probabilities.
    """
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=10000
            )
        ),
        (
            "clf",
            LinearSVC(
                class_weight="balanced",
                random_state=random_state,
                max_iter=2000
            )
        )
    ])
