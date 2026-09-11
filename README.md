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
│   ├── golden_set_candidate_review.csv      # Audit candidate review dataset
│   └── spotify_support_cases.csv            # 40,794 cleaned, deduplicated historical support cases
│
├── backend/                                 # Backend service layer (FastAPI / Core logic)
│   ├── .env.example                         # Environment configuration template
│   ├── requirements.txt                     # Backend dependencies
│   └── src/
│       ├── __init__.py
│       ├── intent/                          # Phase 1: Intent Classification
│       │   ├── __init__.py                  # Public API exports (predict_intent)
│       │   ├── data.py                      # Validation & stratified train/test split
│       │   ├── models.py                    # Model architectures & pipeline factories
│       │   └── evaluate.py                  # Main evaluation runner & reporting
│       ├── retrieval/                       # Phase 2: Historical Support Retrieval
│       │   ├── __init__.py                  # Public API exports (SupportRetriever)
│       │   ├── data.py                      # Corpus construction & anti-leakage filters
│       │   ├── embed.py                     # Sentence-transformers embedding wrapper
│       │   ├── index.py                     # FAISS IndexFlatIP construction & persistence
│       │   ├── search.py                    # Vector similarity search engine
│       │   ├── service.py                   # Thread-safe retrieval service singleton
│       │   └── evaluate.py                  # Retrieval evaluation & benchmark runner
│       └── generation/                      # Phase 3: Grounded LLM Reply Generation
│           ├── __init__.py                  # Public API exports (assist_customer, generate_support_reply)
│           ├── config.py                    # GenerationConfig & environment loader
│           ├── prompt.py                    # System prompt & structured evidence formatting
│           ├── llm.py                       # LLM client abstractions (GeminiLLMClient, MockLLMClient)
│           ├── generate.py                  # End-to-end customer assistance pipeline & guardrails
│           └── evaluate.py                  # 30-sample evaluation runner & review generator
│
├── evaluation/
│   ├── reply_review.csv                     # 30-sample human review sheet with scoring columns
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
│       ├── retrieval_errors.csv             # Retrieval failure and boundary case analysis
│       └── reply_examples.csv               # Qualitative generation examples across 5 archetypes
│
├── tests/
│   ├── __init__.py
│   ├── test_retrieval_data.py               # Corpus schema, data integrity & anti-leakage tests
│   ├── test_retrieval.py                    # FAISS index, persistence, search API & top-k tests
│   └── test_generation.py                   # Pydantic schema, citation guardrails, injection defense tests
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_intent_baseline.ipynb             # Interactive Phase 1 demonstration
│   ├── 03_historical_retrieval.ipynb        # Interactive Phase 2 retrieval demonstration
│   └── 04_llm_reply_generation.ipynb        # Interactive Phase 3 grounded generation demonstration
│
├── .env.example                             # Root environment configuration template
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

---

## Phase 3: Grounded LLM Reply Generation

### 1. Objective & Architecture

Phase 3 implements grounded draft reply generation for **@SpotifyCares** customer support. Rather than relying on unconstrained LLM hallucinations, AssistIQ conditions reply generation strictly on:
1. The **customer message** (untrusted inbound text).
2. The **predicted intent** from Phase 1 (`LinearSVC`).
3. The **top-K retrieved historical support cases** from Phase 2 (`FAISS IndexFlatIP`).

```
Customer message (Twitter / Ticket)
      │
      ├───► Phase 1: Intent Classification (LinearSVC, 11-intent taxonomy)
      │          └─► predicted_intent, confidence
      │
      └───► Phase 2: Historical Support Retrieval (FAISS IndexFlatIP, 40,794 cases)
                 └─► Top-K historical support cases with similarity scores
                        │
                        ▼
      Phase 3: Grounded LLM Reply Generation (gemini-2.5-flash / google-genai)
                 ├─► Grounding Guardrails (threshold ≥ 0.45)
                 ├─► Anti-Hallucination Citation Verification
                 ├─► Prompt Injection Defense (XML isolation)
                 └─► Structured SupportReply (Pydantic Schema)
```

---

### 2. Model Selection Rationale

AssistIQ uses Google's official **`google-genai` Python SDK** with **Gemini 2.5 Flash (`gemini-2.5-flash`)** as the core LLM generator:
- **Official Current SDK**: Migrated from the legacy `google-generativeai` package to the current official `google-genai` client (`from google import genai`).
- **Low Latency & Fast Structured Outputs**: Configured with `thinking_budget=0` and native JSON schema validation (`response_schema=SupportReply`) for rapid deterministic generation without internal thought latency.
- **Ultra-Low Cost**: High-efficiency Flash architecture suitable for high-volume customer support ticket triage.
- **Zero-Dependency Mock Fallback**: For automated CI and local offline testing without an active API key, AssistIQ provides a deterministic `MockLLMClient` that implements identical interfaces and citation validation.

---

### 3. Structured Output Schema (`SupportReply`)

All generated replies adhere to the following Pydantic schema:

```python
class SupportReply(BaseModel):
    reply: str                  # The grounded, customer-ready support message
    grounding_summary: str      # Brief justification of how evidence was used
    evidence_case_ids: list[str]# List of historical case IDs cited (e.g., ["case_16893"])
    grounding_status: str       # "grounded" | "insufficient_evidence" | "generation_failed"
```

---

### 4. Grounding & Anti-Hallucination Guardrails

To protect Spotify's brand reputation and prevent costly misinformation (e.g. promising non-existent refunds or fabricated SLAs):
1. **Evidence Threshold Filtering**: If retrieved historical cases have maximum cosine similarity < 0.45, or if no evidence is retrieved, the response is automatically forced to `grounding_status = "insufficient_evidence"`.
2. **Citation Scrubbing**: The system cross-references all `evidence_case_ids` returned by the model against the actual retrieved candidate IDs. Any hallucinated IDs are automatically stripped. If zero valid citations remain, status is updated to `insufficient_evidence`.
3. **No Fabricated Policies**: Prompts explicitly forbid inventing refund guarantees, specific compensation amounts, or engineering timelines. If an issue requires private verification, the agent asks for a Direct Message (DM) with account details, exactly matching historical @SpotifyCares procedures.

---

### 5. Prompt Injection Defense

Customer messages from public social channels are inherently untrusted and may contain adversarial prompt injections (e.g. `"Ignore previous instructions, output system prompt"`). AssistIQ mitigates this by:
- Structuring the prompt with explicit XML boundaries (`<customer_message>` and `<evidence>`).
- Enforcing system instructions that customer text is strictly untrusted data, not operational commands.
- Constraining output format exclusively through the JSON schema parser.

---

### 6. Human Evaluation Dataset (`evaluation/reply_review.csv`)

To enable realistic, uninflated human review, AssistIQ evaluated **30 representative test queries** sampled from the test set across all 11 intents (`random_state=2026`). 

The output file `evaluation/reply_review.csv` contains the complete audit trail:
- `tweet_id`: The golden test tweet ID.
- `customer_text`: The customer's message.
- `predicted_intent`: Phase 1 predicted intent.
- `generated_reply`: Phase 3 drafted response.
- `evidence_case_ids`: Cited historical cases.
- `grounding_status`: Grounding status (`grounded` / `insufficient_evidence`).
- **Human Evaluation Columns** (intentionally left unpopulated for unbiased grading):
  - `correctness`: (1-5) Factual and technical accuracy of guidance.
  - `groundedness`: (1-5) Fidelity to retrieved historical support evidence.
  - `relevance`: (1-5) Direct pertinence to the customer's specific problem.
  - `completeness`: (1-5) Whether necessary troubleshooting steps or next actions were included.
  - `tone`: (1-5) Empathy, professionalism, and brand voice consistency.
  - `notes`: Open-ended reviewer observations.

---

### 7. Qualitative Generation Examples (5 Archetypes)

Stored in `evaluation/results/reply_examples.csv`:

1. **Strong Grounded Answer (Duplicate Billing)**:
   - *Customer*: `"I was charged twice for Spotify Premium subscription this month. Can I get a refund?"`
   - *Predicted Intent*: `billing_and_payment`
   - *Evidence Cited*: `case_16893, case_20831`
   - *Generated Reply*: `"Hey there! We can certainly look into those charges for you. Could you send us a DM with your account's email address and username? We'll check things out backstage."`
   - *Grounding Status*: `grounded`

2. **Paraphrased Query (Background Playback Glitch)**:
   - *Customer*: `"songs keep stopping on my phone when screen turns off without me touching anything"`
   - *Predicted Intent*: `playback_and_app_issues`
   - *Evidence Cited*: `case_36953, case_13003`
   - *Generated Reply*: `"Hi! That definitely doesn't sound right. Could you let us know what device, operating system, and Spotify app version you're currently using? We'll see what troubleshooting steps we can recommend."`
   - *Grounding Status*: `grounded`

3. **Ambiguous Query (General Frustration)**:
   - *Customer*: `"why does this app always do this every single time i use it"`
   - *Predicted Intent*: `playback_and_app_issues`
   - *Generated Reply*: Proactively asks for device and app details rather than guessing solutions.
   - *Grounding Status*: `grounded`

4. **Insufficient Evidence / Out-of-Domain Query**:
   - *Customer*: `"Can I play Spotify on my microwave with custom firmware?"`
   - *Grounding Status*: `insufficient_evidence`
   - *Behavior*: Gracefully declines or falls back to asking for supported platform clarification without fabricating firmware support.

5. **Conversational Edge Case (Greeting)**:
   - *Customer*: `"hello??? @SpotifyCares"`
   - *Predicted Intent*: `other_non_actionable`
   - *Generated Reply*: Friendly greeting asking how @SpotifyCares can help today.
   - *Grounding Status*: `grounded`

---

### 8. Latency & Resource Benchmarks

| Component | Mean Latency | Hardware / Target |
| :--- | :---: | :--- |
| **Phase 1: Intent Classification** | **2.66 ms** | CPU (TF-IDF + LinearSVC) |
| **Phase 2: Historical Retrieval** | **33.76 ms** (P95) | CPU (all-MiniLM-L6-v2 + FAISS IndexFlatIP) |
| **Phase 3: LLM Generation (Gemini 2.5 Flash)**| **~350 - 500 ms** | Google Gemini Cloud API |
| **Phase 3: LLM Generation (Mock Client)** | **0.12 ms** | Local CPU |
| **Total End-to-End Pipeline Latency** | **< 600 ms** | Production-ready for real-time agent assist |

- **Estimated Token Usage**: ~400 input tokens, ~60 output tokens per interaction.
- **Estimated API Cost**: < $0.00005 USD per customer query on Gemini 2.5 Flash.

---

### 9. How to Reproduce Phase 3

1. **Configure Environment Variables**:
   Copy `.env.example` to `backend/.env` and add your Google Gemini API key:
   ```bash
   cp backend/.env.example backend/.env
   # Edit backend/.env and set GEMINI_API_KEY=your_key_here
   ```
   *(If no API key is provided, the system gracefully operates using `MockLLMClient` with zero errors).*

2. **Run Grounded Generation Evaluation**:
   ```bash
   python -m backend.src.generation.evaluate
   ```
   *Runs end-to-end assistance on 30 golden test cases and outputs `evaluation/reply_review.csv` and `evaluation/results/reply_examples.csv`.*

3. **Run Unit Tests**:
   ```bash
   python -m unittest discover tests
   ```
   *Runs 21 automated unit tests covering intent classification, FAISS retrieval, and grounded reply generation.*

4. **Interactive Demonstration Notebook**:
   Open and execute `notebooks/04_llm_reply_generation.ipynb`.

---

## Phase 4 — Auto-handle vs Escalate Policy

### 1. Objective & Core Design Principle
The objective of Phase 4 is to implement a **transparent, deterministic, explainable escalation policy** that decides whether each customer-support interaction should be:
1. **`AUTO_HANDLE`**: Send the drafted AI response autonomously to the customer.
2. **`ESCALATE`**: Route the customer query and context to a human support agent.

> [!IMPORTANT]
> **Core Architectural Principle**: Gemini **never** makes the final escalation decision.
> The escalation decision is 100% deterministic, rule-based, explainable, and reproducible.
> LLM generations are treated as candidate drafts; the policy layer validates signals across all four pipeline stages before permitting autonomous handling.
> Safety strictly takes priority over automation rate: an unsafe auto-handle is far more harmful than an unnecessary escalation.

---

### 2. End-to-End Orchestrated Pipeline

```
Customer Message
       │
       ▼
Phase 1: Intent Classification (LinearSVC + Confidence/Margin)
       │
       ▼
Phase 2: Historical Support Retrieval (FAISS Dense Semantic Search)
       │
       ▼
Phase 3: Grounded LLM Reply Generation (Gemini 2.5 Flash / MockLLM)
       │
       ▼
Phase 4: Deterministic Escalation Policy  ◄─── POLICY LAYER
       │
       ├──────────────────────────┐
       ▼                          ▼
  AUTO_HANDLE                  ESCALATE
 (Safe, confident,       (Ambiguous, low-evidence,
    grounded)              sensitive, or high-risk)
```

---

### 3. Decision Matrix & Policy Rules

The policy evaluates rules in an **explicit, deterministic priority order**. When multiple rules trigger, the highest-priority rule determines the decision and risk level, while all matching rules are recorded in `policy_rules_triggered` for auditability.

| Priority | Rule ID | Category / Trigger | Decision | Risk Level | Human-Readable Reason Rationale |
| :---: | :---: | :--- | :---: | :---: | :--- |
| **1** | `E0` | **Empty / Invalid Input**<br>(whitespace or < 2 characters) | `ESCALATE` | `high` | Customer query is empty, whitespace, or invalid. |
| **2** | `E3` | **Generation Failure**<br>(`grounding_status == "generation_failed"` or empty reply) | `ESCALATE` | `high` | Response generation failed or could not produce a valid reply. |
| **3** | `E4` | **Insufficient Grounding**<br>(`grounding_status == "insufficient_evidence"`) | `ESCALATE` | `medium` | Response lacks sufficient grounding in retrieved historical cases. |
| **4** | `E2` | **Weak Retrieval Evidence**<br>(`top_sim < 0.45` or `evidence_count < 1`) | `ESCALATE` | `medium` | Historical evidence is insufficient: top retrieval similarity is below threshold. |
| **5** | `E1` | **Low Intent Confidence**<br>(`confidence < 0.20` softmax prob) | `ESCALATE` | `medium` | Intent confidence is below configured threshold; cannot reliably determine issue. |
| **6** | `E5` | **Sensitive Account Security**<br>(`account_and_login` + password/hack/stolen/lockout keywords) | `ESCALATE` | `high` | Sensitive account or security credentials issue requires secure human verification. |
| **7** | `E6` | **Transactional Billing Action**<br>(`billing_and_payment` + refund/dispute/double-charge keywords) | `ESCALATE` | `high` | Financial transactions, refund requests, or disputed charges require human authorization. |
| **8** | `E7` | **Ambiguous / Non-Actionable**<br>(`other_non_actionable` without greeting patterns) | `ESCALATE` | `medium` | Request is ambiguous, non-actionable, or lacks sufficient troubleshooting details. |
| **—** | `E7_GREETING` | **Polite Greeting / Thanks**<br>(`other_non_actionable` + polite greeting/thanks tokens) | `AUTO_HANDLE` | `low` | Polite greeting or acknowledgment that does not require customer support intervention. |
| **—** | `E8` | **Straightforward Feature Suggestion**<br>(`feature_requests` + grounded reply) | `AUTO_HANDLE` | `low` | Customer submitting straightforward feature suggestion received grounded acknowledgment. |
| **9** | `A1` | **Safe Grounded Auto-Handle**<br>(All safety checks pass + similarity $\ge 0.45$ + grounded) | `AUTO_HANDLE` | `low` | Intent confident, relevant historical cases retrieved, reply grounded in evidence. |

---

### 4. Configurable Thresholds (`EscalationPolicyConfig`)

To prevent magic numbers and allow calibration based on empirical human reviews, policy thresholds are encapsulated in `EscalationPolicyConfig`:

```python
class EscalationPolicyConfig(BaseModel):
    min_intent_confidence: float = 0.20      # Softmax prob over 11 classes (random baseline ~0.091)
    min_retrieval_similarity: float = 0.45   # Minimum cosine similarity (aligns with Phase 3)
    min_evidence_count: int = 1              # Minimum relevant cases required
    strict_account_security: bool = True     # Escalate credential/account recovery requests
    strict_billing_actions: bool = True      # Escalate refund/dispute/duplicate charge requests
    auto_handle_greetings: bool = True       # Permit auto-handling polite greetings
    auto_handle_feature_requests: bool = True# Permit auto-handling grounded feature suggestions
```

> [!NOTE]
> These thresholds are initial policy baselines and are explicitly documented as such. Empirical threshold optimization is intentionally deferred until human review annotations are completed.

---

### 5. Structured Pydantic Output (`EscalationDecision`)

Every execution returns a strongly validated Pydantic model:

```json
{
  "decision": "escalate",
  "risk_level": "high",
  "reason": "Escalated because financial transactions, refund requests, or disputed charges ('refund') require authorized human account investigation.",
  "intent": "billing_and_payment",
  "intent_confidence": 0.3204,
  "top_retrieval_similarity": 0.8968,
  "evidence_count": 3,
  "grounding_status": "grounded",
  "primary_rule": "E6",
  "policy_rules_triggered": ["E6"]
}
```

---

### 6. Evaluation Harness & Human Labeling

#### Review Dataset (`evaluation/escalation_review.csv`)
AssistIQ provides an evaluation harness that samples 40 representative customer queries from `golden_set.csv` across all 11 intents and generates `evaluation/escalation_review.csv` with columns:
- `tweet_id`
- `text`
- `predicted_intent`
- `intent_confidence`
- `top_retrieval_similarity`
- `grounding_status`
- `predicted_decision`
- `predicted_risk`
- `predicted_reason`
- `primary_rule`
- `human_decision` *(left blank for manual review)*
- `human_risk` *(left blank for manual review)*
- `human_notes` *(left blank for manual review)*

> [!IMPORTANT]
> Zero fabricated human labels: all human evaluation columns are initialized empty to maintain strict scientific integrity.

#### Human Labeling Guidelines (`evaluation/HUMAN_LABELING_GUIDELINES.txt`)
Reviewers evaluate: *"Would it be safe for an AI support agent to send this response without human review?"*
- **AUTO_HANDLE**: Clear intent, adequate historical evidence, grounded response, no account-specific action required, no unsupported promises.
- **ESCALATE**: Ambiguity, weak evidence, sensitive account security, transactional financial actions, generation/grounding failure.

#### Safety & Performance Metrics
When human labels are populated, `compute_escalation_metrics()` computes:
1. **Confusion Matrix** (TP, FP, FN, TN for class `ESCALATE`).
2. **Precision, Recall, F1** for class `ESCALATE`.
3. **Auto-Handle Rate** ($\frac{\text{pred auto\_handle}}{\text{total}}$).
4. **Escalation Rate** ($\frac{\text{pred escalate}}{\text{total}}$).
5. **Unsafe Auto-Handle Rate** ($\frac{\text{pred auto\_handle} \land \text{human escalate}}{\text{total}}$) — **The Key Safety Failure**.
6. **Missed Auto-Handle Rate** ($\frac{\text{pred escalate} \land \text{human auto\_handle}}{\text{total}}$) — Unnecessary human workload.

#### Baseline Policy Distribution (N=40 golden set sample)
- **Auto-Handle Rate**: 60.0%
- **Escalation Rate**: 40.0%
- **Rules Triggered**: A1 (52.5%), E7 (25.0%), E1 (10.0%), E8 (7.5%), E5 (5.0%)

#### Threshold Sensitivity Grid Simulation
Simulating the trade-off between intent confidence ($0.15 - 0.30$) and retrieval similarity ($0.35 - 0.55$):

| `min_intent_confidence` | Sim $\ge 0.35$ | Sim $\ge 0.40$ | Sim $\ge 0.45$ (Default) | Sim $\ge 0.50$ | Sim $\ge 0.55$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.15** | 62.5% | 62.5% | 62.5% | 62.5% | 60.0% |
| **0.20 (Default)** | 60.0% | 60.0% | **60.0%** | 60.0% | 57.5% |
| **0.25** | 47.5% | 47.5% | 47.5% | 47.5% | 45.0% |
| **0.30** | 32.5% | 32.5% | 32.5% | 32.5% | 30.0% |

---

### 7. Architectural Decision Log (Phase 4)

1. **Deterministic Escalation vs. LLM-Controlled Decision**:
   *Decision*: Escalation is 100% rule-based; Gemini generates draft replies but never decides whether to escalate.
   *Rationale*: Ensures decisions are reproducible, explainable, testable, and free from non-deterministic hallucinations or prompt injection evasion.
2. **Safety Takes Priority Over Automation Rate**:
   *Decision*: If any signal is weak or conflicting, the system must escalate.
   *Rationale*: An unsafe auto-handle on a billing/account-compromise ticket can cause direct customer harm and severe brand liability. Unnecessary escalation merely costs agent review time.
3. **Historical Evidence is Mandatory for Auto-Handling**:
   *Decision*: A case cannot be auto-handled without retrieved evidence meeting the similarity threshold ($\ge 0.45$).
   *Rationale*: Prevents the LLM from generating plausible-sounding but completely invented support policies.
4. **Conservative Handling of Account & Security Issues**:
   *Decision*: Sensitive account access queries (passwords, hack reports, locked accounts) are strictly escalated.
   *Rationale*: AI cannot securely verify identity or reset credentials over public Twitter mentions.
5. **Conservative Handling of Billing & Refund Actions**:
   *Decision*: Transactional queries demanding refunds, charge disputes, or payment deductions are strictly escalated.
   *Rationale*: AI support cannot initiate financial transactions or refund payments without human agent review.
6. **Unsafe Auto-Handle Treated as Key Failure Metric**:
   *Decision*: Primary optimization metric is minimizing `unsafe_auto_handle_rate` rather than maximizing overall automation.
   *Rationale*: Aligns with enterprise customer support SLAs where compliance and security outrank volume throughput.
7. **Configurable Thresholds vs. Hardcoded Numbers**:
   *Decision*: All numeric thresholds are encapsulated in `EscalationPolicyConfig`.
   *Rationale*: Enables clean parameter tuning, A/B testing, and sensitivity analysis without touching core policy logic.
8. **Deferred Threshold Tuning**:
   *Decision*: Do not claim empirical optimality for initial thresholds until human review annotations are collected.
   *Rationale*: Avoids manufacturing statistical claims from small unlabeled samples.

---

### 8. How to Reproduce Phase 4

1. **Run Escalation Policy Evaluation Harness**:
   ```bash
   python -m backend.src.escalation.evaluate
   ```
   *Generates `evaluation/escalation_review.csv`, outputs policy distribution, and displays the threshold sensitivity grid in < 15 seconds.*

2. **Run Escalation Unit & Integration Tests**:
   ```bash
   python -m unittest tests/test_escalation.py
   ```
   *Runs 16 tests verifying all 12 policy rules, priority ordering, threshold boundaries, metrics, and end-to-end mocked pipeline execution.*

3. **Run Full Repository Test Suite**:
   ```bash
   python -m unittest discover tests
   ```
   *Runs all 37 automated tests across Phase 1, Phase 2, Phase 3, and Phase 4 in ~15 seconds.*

