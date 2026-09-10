# AssistIQ: AI Customer-Support Agent (SpotifyCares)

AssistIQ is an intelligent customer-support agent built for the Hiver SDE Intern take-home assignment, trained on the **Customer Support on Twitter** dataset specifically for **@SpotifyCares**.

---

## Phase 1: Intent Classification

### 1. Why Intent Classification is Essential
In an automated customer-support system, accurately identifying customer intent is the primary gatekeeper for downstream agent actions:
- **Routing & Triaging**: Technical bugs (e.g., audio playback glitches) must be handled differently from account security alerts (e.g., unauthorized logins) or billing inquiries.
- **Contextual Retrieval**: Intent determines which subset of historical solutions and documentation should be retrieved for contextual generation.
- **Escalation Detection**: Identifies whether a ticket can be resolved automatically or requires human support intervention.

---

### 2. Intent Taxonomy
AssistIQ uses an 11-intent taxonomy tailored to the Spotify customer-support domain:

| Intent | Definition |
| :--- | :--- |
| `playback_and_app_issues` | Problems playing music or using the Spotify app, including crashes, pausing, skipping, loading, offline playback, or device/app compatibility. |
| `search_and_discovery` | Searching for music, lyrics, recommendations, Discover Weekly, mixes, and discovery-related functionality. |
| `account_and_login` | Login, password, account access, hacked accounts, email/username changes, account recovery. |
| `billing_and_payment` | Charges, refunds, payment failures, duplicate charges, payment methods, and payment-related problems. |
| `premium_and_subscription` | Premium activation, Premium not appearing, upgrades/downgrades, Family, Student, trials, and subscription eligibility. |
| `music_availability` | Songs, albums, artists, podcasts, or other content being missing or unavailable. |
| `playlist_and_library` | Creating/editing/importing/saving/downloading playlists or managing the user's library. |
| `feature_requests` | Requests or suggestions for new Spotify features or changes to existing functionality. |
| `content_metadata` | Incorrect song titles, artist names, album structure, artwork, credits, or other metadata/content presentation errors. |
| `ads_and_privacy` | Advertisements, tracking, privacy, or advertising-related concerns. |
| `other_non_actionable` | Thanks, acknowledgements, greetings, unclear/random messages, or messages that do not represent an actionable support issue. |

---

### 3. Classifiers & Baselines

To ensure rigorous and honest benchmarking, we implemented three distinct classifiers:

#### Baseline 1: Majority-Class Classifier (`DummyClassifier`)
- **Mechanism**: Learns the most frequent class in the training set (`other_non_actionable`, $N=40$) and unconditionally predicts it for all test inputs.
- **Role**: Establishes the performance floor under severe class imbalance. Any viable model must demonstrate statistically meaningful improvements over this baseline.

#### Baseline 2: TF-IDF + Logistic Regression (`LogisticRegression`)
- **Mechanism**: Encapsulated in an `sklearn.pipeline.Pipeline`:
  - `TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=10000)`
  - `LogisticRegression(max_iter=1000, class_weight="balanced", random_state=2026)`
- **Role**: A standard, interpretable linear text baseline with probabilistic output (`max_class_probability`) and balanced class weighting.

#### Proposed Model: TF-IDF + Linear Support Vector Machine (`LinearSVC`)
- **Mechanism**:
  - `TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=10000)`
  - `LinearSVC(class_weight="balanced", random_state=2026, max_iter=2000)`
- **Role**: On short text tweets with sparse n-gram vocabularies, linear max-margin hyperplanes often outperform cross-entropy loss by enforcing geometric separation. It provides superior sensitivity on minority intent classes.

---

### 4. Experimental Setup & Evaluation Methodology

- **Golden Evaluation Dataset**: `dataset/golden_set.csv` containing **200 verified examples**.
- **Train/Test Split**:
  - 75% Training ($N=150$) / 25% Test ($N=50$)
  - `random_state=2026`
  - **Stratified Split**: Preserves class distribution proportions across splits.
  - *Small-sample note*: `ads_and_privacy` has $N=2$ total examples in the dataset, allocating 2 train and 0 test under standard floor rounding; `search_and_discovery` ($N=3$) allocates 2 train and 1 test. All other 9 classes are well-represented across both splits.
- **Strict Anti-Leakage Guarantee**:
  - All text vectorization (vocabulary learning, IDF weighting) and model fitting are performed **exclusively** on `X_train`.
  - The test set ($N=50$) remains completely unseen until inference.
- **Evaluation Metrics**:
  - **Accuracy**: Overall fraction of correct predictions.
  - **Macro F1**: Unweighted mean of F1 scores across all 11 classes (crucial metric for class-imbalanced evaluation).
  - **Weighted F1**: F1 score weighted by class support.
  - **Per-Class Precision, Recall, F1, Support**: Tracks granular performance per intent.
  - **Confusion Matrices**: 11x11 matrices saved as visual plots.

---

### 5. Empirical Results

Evaluated on the isolated 50-example test set:

| Model | Accuracy | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: |
| **Majority Baseline** | 0.2600 | 0.0413 | 0.1073 |
| **TF-IDF + Logistic Regression** | 0.4400 | 0.3101 | 0.4187 |
| **Proposed Model (TF-IDF + LinearSVC)** | **0.4600** | **0.4004** | **0.4534** |

#### Key Insights:
1. **Macro F1 Gain**: The proposed LinearSVC model achieves a **+9.0 percentage point increase in Macro F1** (0.4004 vs 0.3101) over Logistic Regression.
2. **Minority Class Sensitivity**: LinearSVC successfully recovers minority classes where Logistic Regression failed (e.g. `music_availability` F1: 0.40 vs 0.00; `content_metadata` F1: 0.50 vs 0.00).
3. **Majority Baseline Comparison**: Both machine learning models dramatically outperform the naive majority baseline (Accuracy 0.2600, Macro F1 0.0413).

---

### 6. Project Structure

```
AssistIQ/
│
├── dataset/
│   ├── golden_set.csv                       # 200-example human-verified golden evaluation set
│   └── golden_set_candidate_review.csv      # Audit candidate review dataset
│
├── backend/                                 # Backend service layer (FastAPI / Core logic)
│   ├── requirements.txt                     # Backend dependencies
│   └── src/
│       ├── __init__.py
│       └── intent/
│           ├── __init__.py                  # Public API exports
│           ├── data.py                      # Validation & stratified train/test split
│           ├── models.py                    # Model architectures & pipeline factories
│           └── evaluate.py                  # Main evaluation runner & reporting
│
├── evaluation/
│   └── results/
│       ├── intent_results.csv               # Model comparison table
│       ├── intent_per_class_results.csv     # Granular precision/recall/F1 per intent
│       ├── tfidf_lr_predictions.csv         # Test predictions with max_class_probability
│       ├── intent_errors.csv                # Misclassified test examples with top-3 predictions
│       ├── confusion_matrix_majority_baseline.png
│       ├── confusion_matrix_tfidf_logistic_regression.png
│       └── confusion_matrix_proposed_model.png
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   └── 02_intent_baseline.ipynb             # Interactive demonstration & failure analysis
│
├── requirements.txt                         # Root convenience dependencies
└── README.md
```

---

### 7. How to Reproduce Phase 1

1. **Install Dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

2. **Run Evaluation Pipeline**:
   ```bash
   python backend/src/intent/evaluate.py
   ```
   *Alternatively, run as a module from root:*
   ```bash
   python -m backend.src.intent.evaluate
   ```
   *Execution finishes in < 5 seconds and updates all results under `evaluation/results/`.*

3. **Interactive Exploration Notebook**:
   Open and run `notebooks/02_intent_baseline.ipynb` in your Jupyter environment.
