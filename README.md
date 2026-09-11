# AssistIQ: AI Customer-Support Agent (SpotifyCares)

AssistIQ is an intelligent customer-support agent, trained on the **Customer Support on Twitter** dataset specifically for **@SpotifyCares**.

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
│       ├── confusion_matrix_proposed_model.png
│       ├── retrieval_results.csv            # Quantitative retrieval evaluation metrics
│       ├── retrieval_examples.csv           # Qualitative Top-5 retrieval across 4 archetypes
│       └── retrieval_errors.csv             # Retrieval failure and boundary case analysis
│
├── tests/
│   ├── __init__.py
│   ├── test_retrieval_data.py               # Corpus schema, data integrity & anti-leakage tests
│   └── test_retrieval.py                    # FAISS index, persistence, search API & top-k tests
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_intent_baseline.ipynb             # Interactive Phase 1 demonstration
│   └── 03_historical_retrieval.ipynb        # Interactive Phase 2 retrieval demonstration
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

---

## Phase 2 — Historical Support Retrieval

### 1. Why Historical Retrieval is Necessary
While intent classification (Phase 1) categorizes customer issues into broad operational buckets (e.g., `billing_and_payment`), it cannot provide specific troubleshooting instructions, account verification links, or support policies.
Historical support retrieval acts as AssistIQ's knowledge base. Given an incoming customer message, it searches through tens of thousands of verified historical **@SpotifyCares** Twitter support interactions to surface historically similar customer cases and their associated official Spotify resolutions.
This retrieved evidence will later be used by the downstream LLM (Phase 3) to generate grounded, fact-based responses rather than hallucinating Spotify policies.

---

### 2. Architecture & Retrieval Flow

```
Incoming Customer Message
        ↓
Lightweight Text Normalization (strip @handles, URLs, HTML unescape)
        ↓
Dense Semantic Embedding (sentence-transformers/all-MiniLM-L6-v2)
        ↓ [384-dimensional unit vector, L2 norm = 1.0]
FAISS Vector Search (IndexFlatIP: Exact Cosine Similarity)
        ↓
Top-K Similar Historical Customer Messages (sorted by similarity descending)
        ↓
Associated Historical Spotify Responses / Evidence + Traceability Metadata
```

---

### 3. Construction of Customer → Spotify Support Pairs

From `dataset/spotify_tweets.csv` (88,445 tweets):
1. **Inbound / Author Partitioning**:
   - Customer tweets: `inbound=True` (45,180 tweets)
   - Support tweets: `inbound=False` and `author_id=SpotifyCares` (43,265 tweets)
2. **Pairing via Tweet Relationships**:
   - `in_response_to_tweet_id` on support tweets references the customer `tweet_id`.
   - 43,092 support tweets match 41,585 unique customer tweets.
3. **Multi-Part Tweet Resolution**:
   - On Twitter, 1,471 customer tweets received multiple support tweets from SpotifyCares (e.g., split into `1:` and `2:` due to the 140/280 character limit).
   - Rather than dropping or fragmenting solutions, we chronologically concatenate multi-part support tweets (`"\n".join(texts)`), ensuring complete, coherent resolution instructions.
4. **Deterministic Conversation Root Tracing**:
   - By traversing `in_response_to_tweet_id` backwards to the thread root, we map all 40k+ cases into **28,425 unique conversation threads** (`conversation_id`).
5. **No Fabricated Labels**:
   - Raw historical tweets do not have human intent annotations. Ground-truth `intent` is left `null` (None) rather than generating pseudo-labels.

---

### 4. Dataset Transformation Audit Trail

| Pipeline Stage | Tweet / Case Count | Description |
| :--- | :---: | :--- |
| **Raw Spotify Tweets** | 88,445 | Complete Spotify subset extracted from TWCS |
| **Customer Tweets** | 45,180 | `inbound == True` |
| **SpotifyCares Replies** | 43,265 | `inbound == False` & `author_id == 'SpotifyCares'` |
| **Reconstructed Pairs** | 41,585 | Customer tweets with matched support responses |
| **Golden-Set Leakage Excluded** | -182 | Strictly removed all matching golden evaluation cases |
| **Duplicate Queries Deduplicated** | -609 | Retained case with most comprehensive support text |
| **Final Retrieval Corpus** | **40,794** | Clean cases written to `dataset/spotify_support_cases.csv` |

---

### 5. Semantic Embeddings vs. Keyword Search

- **Why Not BM25 / Keyword Search**: Customer support queries exhibit extreme lexical divergence. A customer writing *"I was double charged"* shares zero keywords with another writing *"Money was deducted two times"*. Keyword search fails on vocabulary mismatch.
- **Pretrained Dense Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`:
  - **384-dimensional** dense vector representation.
  - Pretrained on >1 billion sentence pairs; highly effective semantic capture.
  - **CPU-Optimized**: Encodes at ~150 sentences/second on local CPU, indexing the entire 40,794-case corpus in **4.55 minutes** (well under the 15-minute budget).
  - No fine-tuning required, ensuring reproducible, deterministic results.
- **Embedding Strategy**: We embed **historical customer messages** and search against them using the **new customer query**. Comparing customer queries to customer queries matches identical problems, then retrieves the attached Spotify resolution.

---

### 6. FAISS Vector Indexing & Similarity Metric

- **Index Type**: `faiss.IndexFlatIP` (Inner Product).
- **Exact Cosine Similarity Guarantee**:
  Because all embeddings are unit-normalized ($L_2\text{ norm} = 1.0$), the inner product mathematically equals exact cosine similarity:
  $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \mathbf{u} \cdot \mathbf{v}$$
  This avoids approximate nearest neighbor (ANN) recall errors and executes exhaustive vector search in **< 1 ms per query**.
- **Metadata Traceability**:
  Returned cases include `case_id`, `customer_tweet_id`, `support_tweet_id`, `conversation_id`, `customer_text`, `support_text`, and `similarity` (rounded to 4 decimals).

---

### 7. Strict Anti-Leakage Guarantee

A retrieval system must not evaluate on queries present in its own knowledge base.
- All 200 examples in `dataset/golden_set.csv` were originally drawn from the Twitter dataset, and 182 of them had support responses in `spotify_tweets.csv`.
- **Enforcement**: Before indexing, we completely exclude any customer tweet matching `golden_set.csv` by **tweet ID** AND by **normalized customer text**.
- **Automated Verification**: `verify_anti_leakage()` runs in both unit tests and the evaluation runner, confirming **exactly 0.00% overlap**.

---

### 8. Empirical Evaluation & Quality Metrics

Evaluated across all 200 golden set queries:

| Evaluation Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **Indexed Support Cases** | **40,794** | Total clean historical support cases in FAISS index |
| **Golden Anti-Leakage Overlap** | **0 (0.0%)** | Verified zero leakage between test queries and index |
| **Intent Consistency @ 1 (vs True Intent)** | **0.5800** | Fraction of Top-1 retrieved cases matching query intent |
| **Intent Consistency @ 3 (vs True Intent)** | **0.5683** | Mean fraction of Top-3 retrieved cases matching query intent |
| **Intent Consistency @ 5 (vs True Intent)** | **0.5730** | Mean fraction of Top-5 retrieved cases matching query intent |
| **Intent Hit Rate @ 3** | **0.8050** | 80.5% of queries have $\ge 1$ matching intent in Top-3 |
| **Intent Hit Rate @ 5** | **0.8850** | 88.5% of queries have $\ge 1$ matching intent in Top-5 |
| **Manual Review Precision @ 1** | **1.0000** | Top-1 cases relevant or partially relevant (35-query sample) |
| **Manual Review Strict Precision @ 1** | **0.7143** | Top-1 cases strictly relevant to specific customer issue |
| **Manual Review Strict Precision @ 3** | **0.6381** | Top-3 cases strictly relevant to specific customer issue |
| **Manual Review Hit Rate @ 3** | **1.0000** | 100% of reviewed queries have $\ge 1$ relevant case in Top-3 |
| **95th Percentile Latency** | **33.76 ms** | P95 retrieval latency per query |

*Note on Intent Consistency: Intent consistency is an automated proxy metric measuring whether semantic search preserves high-level domain boundaries; it is not human ground truth relevance.*

---

### 9. Qualitative Examples (Top-1 Highlights)

From `evaluation/results/retrieval_examples.csv`:

1. **Strong Semantic Match**:
   - *Query*: `"I was charged twice for Spotify Premium subscription this month. Can I get a refund?"`
   - *Top-1 Match (Sim: 0.8893)*: `"I was charged 3 times for Spotify premium and I still don’t have premium and it’s been a day..."`
   - *Spotify*: `"Hey there! Can you DM us your account's email address? We'll take a look /RB https://t.co/ldFdZRiNAt"`
2. **Lexical Diversity (Colloquial Paraphrase)**:
   - *Query*: `"songs keep stopping on my phone when screen turns off without me touching anything"`
   - *Top-1 Match (Sim: 0.7394)*: `"Still having issues with songs stopping after a few seconds of playing. This has been an issue for over a month on my android phone."`
   - *Spotify*: `"Hey Owen, that doesn't sound right! Can you let us know the device, Android, and Spotify version you're using?..."`
3. **Ambiguous Query**:
   - *Query*: `"why does this app always do this every single time i use it"`
   - *Top-1 Match (Sim: 0.6869)*: `"why do I have to constantly restart my phone to get your app to work as intended?"`
4. **Boundary Case (Ultra-Short / Non-Actionable)**:
   - *Query*: `"hello??? @SpotifyCares"`
   - *Top-1 Match (Sim: 1.0000)*: `"Hello??? https://t.co/..."`

---

### 10. Known Failure Modes & Limitations

Documented in `evaluation/results/retrieval_errors.csv`:
1. **Ultra-Short Customer Queries (< 6 words)**: Tweets like `"2012 https://..."` lack sufficient syntactic context, leading to vector drift toward generic conversational replies.
2. **Heavy Slang & Emotional Venting**: Colloquial expressions (*"what's tea? Why y'all always glitch when I play CTRL?"*) diminish cosine similarity against professional support replies.
3. **Minority Intent Classes**: Issues in sparse categories (e.g., `ads_and_privacy`, $N=2$) have fewer historical examples in the corpus than dominant issues like playback or billing.
4. **Missing Hardware / Platform Context**: Customers omitting whether they are on iOS, Android, Desktop, or Web receive general troubleshooting advice rather than OS-specific steps.

---

### 11. How to Reproduce Phase 2

All commands are runnable from the project root (`C:\AssistIQ`):

1. **Build Historical Retrieval Corpus**:
   ```bash
   python -m backend.src.retrieval.data
   ```
   *Pairs customer and support tweets, resolves multi-part replies, excludes golden set leakage, and outputs `dataset/spotify_support_cases.csv`.*

2. **Generate Embeddings & Build FAISS Index**:
   ```bash
   python -m backend.src.retrieval.index
   ```
   *Generates 384-d normalized embeddings and persists `spotify_cases.faiss` and `case_metadata.csv` under `backend/src/retrieval/artifacts/` in ~4.5 minutes.*

3. **Run Retrieval Evaluation Pipeline**:
   ```bash
   python -m backend.src.retrieval.evaluate
   ```
   *Validates anti-leakage, computes intent consistency across 200 golden queries, benchmarks latency, and outputs all results to `evaluation/results/`.*

4. **Run Unit Tests**:
   ```bash
   python -m unittest discover tests
   ```
   *Executes all 10 unit tests for data integrity, FAISS indexing, and retrieval search.*

5. **Interactive Demonstration Notebook**:
   Open and run `notebooks/03_historical_retrieval.ipynb` in your Jupyter environment.
