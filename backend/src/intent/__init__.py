"""
AssistIQ Intent Classification Package
"""

from .data import INTENT_TAXONOMY, INTENT_DEFINITIONS, validate_golden_set, load_and_split_data
from .models import (
    build_majority_baseline,
    build_tfidf_logistic_regression,
    build_proposed_model
)

__all__ = [
    "INTENT_TAXONOMY",
    "INTENT_DEFINITIONS",
    "validate_golden_set",
    "load_and_split_data",
    "build_majority_baseline",
    "build_tfidf_logistic_regression",
    "build_proposed_model"
]
