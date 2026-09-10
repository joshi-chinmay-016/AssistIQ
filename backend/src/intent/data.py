"""
AssistIQ: Golden Set Validation and Data Ingestion Pipeline
Validates and splits dataset/golden_set.csv for Phase 1 Intent Classification.
"""

import os
from typing import Tuple, List, Dict, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

INTENT_TAXONOMY: List[str] = [
    "playback_and_app_issues",
    "search_and_discovery",
    "account_and_login",
    "billing_and_payment",
    "premium_and_subscription",
    "music_availability",
    "playlist_and_library",
    "feature_requests",
    "content_metadata",
    "ads_and_privacy",
    "other_non_actionable"
]

INTENT_DEFINITIONS: Dict[str, str] = {
    "playback_and_app_issues": "Problems playing music or using the Spotify app, including crashes, pausing, skipping, loading, offline playback, or device/app compatibility.",
    "search_and_discovery": "Searching for music, lyrics, recommendations, Discover Weekly, mixes, and discovery-related functionality.",
    "account_and_login": "Login, password, account access, hacked accounts, email/username changes, account recovery.",
    "billing_and_payment": "Charges, refunds, payment failures, duplicate charges, payment methods, and payment-related problems.",
    "premium_and_subscription": "Premium activation, Premium not appearing, upgrades/downgrades, Family, Student, trials, and subscription eligibility.",
    "music_availability": "Songs, albums, artists, podcasts, or other content being missing or unavailable.",
    "playlist_and_library": "Creating/editing/importing/saving/downloading playlists or managing the user's library.",
    "feature_requests": "Requests or suggestions for new Spotify features or changes to existing functionality.",
    "content_metadata": "Incorrect song titles, artist names, album structure, artwork, credits, or other metadata/content presentation errors.",
    "ads_and_privacy": "Advertisements, tracking, privacy, or advertising-related concerns.",
    "other_non_actionable": "Thanks, acknowledgements, greetings, unclear/random messages, or messages that do not represent an actionable support issue."
}


def find_dataset_path(filename: str = "golden_set.csv") -> str:
    """
    Robustly resolves the path to the dataset file regardless of working directory
    (e.g., repository root or backend/ directory).
    """
    candidates = [
        os.path.join("dataset", filename),
        os.path.join("..", "dataset", filename),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "dataset", filename))
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return os.path.join("dataset", filename)


def validate_golden_set(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Validates dataset/golden_set.csv according to strict Phase 1 requirements:
    - File exists
    - Required columns exist: ['tweet_id', 'text', 'intent']
    - Exactly 200 rows
    - Text is not empty or null
    - Intent is not empty or null
    - All intents belong to the 11-intent taxonomy
    - No duplicate tweet_id values
    - Class distribution is displayed

    Returns the validated DataFrame if all checks pass; raises ValueError/FileNotFoundError otherwise.
    """
    if filepath is None:
        filepath = find_dataset_path("golden_set.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Golden set file not found at: {filepath}")

    df = pd.read_csv(filepath)

    # 1. Check required columns
    required_cols = ["tweet_id", "text", "intent"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Golden set validation failed: Missing required columns: {missing_cols}")

    # 2. Check row count
    if len(df) != 200:
        raise ValueError(f"Golden set validation failed: Expected exactly 200 rows, found {len(df)}")

    # 3. Check for empty/null text
    null_texts = df["text"].isna().sum() + (df["text"].astype(str).str.strip() == "").sum()
    if null_texts > 0:
        raise ValueError(f"Golden set validation failed: Found {null_texts} empty/null text entries")

    # 4. Check for empty/null intent
    null_intents = df["intent"].isna().sum() + (df["intent"].astype(str).str.strip() == "").sum()
    if null_intents > 0:
        raise ValueError(f"Golden set validation failed: Found {null_intents} empty/null intent entries")

    # 5. Check taxonomy membership
    invalid_intents = set(df["intent"].unique()) - set(INTENT_TAXONOMY)
    if invalid_intents:
        raise ValueError(f"Golden set validation failed: Unrecognized intent(s) not in taxonomy: {invalid_intents}")

    # 6. Check for duplicate tweet_id values
    duplicates = df[df.duplicated(subset=["tweet_id"], keep=False)]
    if not duplicates.empty:
        duplicate_ids = duplicates["tweet_id"].unique().tolist()
        raise ValueError(f"Golden set validation failed: Found duplicate tweet_id values: {duplicate_ids}")

    print("==================================================================")
    print("GOLDEN SET VALIDATION: PASSED")
    print("==================================================================")
    print(f"Dataset path:       {filepath}")
    print(f"Total valid rows:   {len(df)}")
    print(f"Unique tweet IDs:   {df['tweet_id'].nunique()}")
    print("------------------------------------------------------------------")
    print("Class Distribution:")
    dist = df["intent"].value_counts()
    for intent, count in dist.items():
        pct = (count / len(df)) * 100
        print(f"  - {intent:26s} : {count:3d} ({pct:5.1f}%)")
    print("==================================================================")

    return df


def load_and_split_data(
    filepath: Optional[str] = None,
    test_size: float = 0.25,
    random_state: int = 2026
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Loads and validates dataset/golden_set.csv, then performs a stratified train/test split.
    Guarantees no test-set leakage: the test set remains completely unseen.

    Returns:
        X_train (pd.Series): Training tweet texts (150 examples)
        X_test  (pd.Series): Test tweet texts (50 examples)
        y_train (pd.Series): Training intent labels
        y_test  (pd.Series): Test intent labels
        train_df (pd.DataFrame): Complete training DataFrame
        test_df  (pd.DataFrame): Complete test DataFrame
    """
    df = validate_golden_set(filepath)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["intent"]
    )

    X_train = train_df["text"]
    y_train = train_df["intent"]
    X_test = test_df["text"]
    y_test = test_df["intent"]

    print("\nTRAIN / TEST SPLIT SUMMARY (Stratified 75% / 25% | random_state=2026)")
    print(f"• Training examples:   {len(train_df)} (75.0%)")
    print(f"• Test examples:       {len(test_df)} (25.0%)")
    print("• Test set is strictly isolated and unseen during feature extraction / training.")

    return X_train, X_test, y_train, y_test, train_df, test_df
