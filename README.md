# AssistIQ: AI Customer Support Agent (@SpotifyCares)

AssistIQ is an evaluation-driven prototype AI customer-support agent built on the **Customer Support on Twitter (TWCS)** dataset for **@SpotifyCares**. It pairs incoming customer messages with historical support resolutions, drafts grounded responses via Google Gemini, and uses a deterministic rule-based escalation policy to decide whether to auto-handle or route to human agents.

---

## Problem
Customer support on public social platforms presents unique operational hurdles:
- **High Inbound Velocity & Noise**: Tweets are constrained ($\le 280$ characters), noisy, lack system context (OS, device, app version), and frequently convey emotional frustration without diagnostic details.
- **Extreme Class Imbalance**: Common playback bugs heavily outnumber niche licensing or metadata issues.
- **Enterprise Liability**: Tier-1 automated agents lack database write permissions; an AI hallucinating refund promises or publishing account credentials in public tweets creates serious brand and financial liability.
- **Evaluation Defensibility**: High retrieval recall often gives false confidence if the retrieved evidence does not offer actionable troubleshooting steps for the customer's specific problem.

---

## Solution
AssistIQ decouples generation from decision-making:
1. **Linear Margin Intent Classifier**: Categorizes inquiries across an 11-intent taxonomy with balanced class weighting.
2. **Dense Semantic Retrieval**: Searches 40,794 historical Spotify support dialogues using FAISS and `all-MiniLM-L6-v2` to surface proven resolutions.
3. **Grounded Drafting**: Generates polite, structured candidate replies constrained by retrieved citations using the official `google-genai` SDK.
4. **Deterministic Escalation**: An auditable rule engine (E0–E8, A1) retains sole authority over auto-handling, ensuring generative models never decide their own escalation.

---

## Architecture

### System Flow Diagram
```
Customer Tweet
      ↓
Intent Classification (TF-IDF + LinearSVC, < 2ms)
      ↓
Historical Support Retrieval (all-MiniLM-L6-v2 + FAISS IndexFlatIP, ~25ms)
      ↓
Grounded Reply Drafting (Gemini 2.5 Flash / MockLLM, Pydantic SupportReply)
      ↓
Citation Verification & Grounding Check (Deterministic scrubber)
      ↓
Deterministic Escalation Policy (Rules E0–E8, A1)
      ↓
FastAPI Backend (REST API /api/v1/assist)
      ↓
Next.js Web Frontend (Interactive Agent Console)
```

### Internal Data Flow
```
Customer Message
      ↓
TF-IDF N-Gram Vectorizer
      ↓
LinearSVC Classifier ──> [Predicted Intent + Margin-Derived Confidence Score]
      ↓
SentenceTransformer (all-MiniLM-L6-v2, 384-d L2-normalized)
      ↓
FAISS IndexFlatIP ──> [Top-k Historical Spotify Support Cases + Cosine Similarities]
      ↓
Prompt Assembly (Query + Evidence + Citation Constraints)
      ↓
Google Gemini API ──> [Drafted Reply + Grounding Status + Cited Case IDs]
      ↓
Citation Scrubber (Validates cited case IDs against retrieved candidate list)
      ↓
Escalation Policy Engine ──> [AUTO_HANDLE vs ESCALATE + Risk Level + Rule Audit Trail]
```

**Key Architectural Rule**: Generative LLMs generate candidate text, but **never make the final escalation decision**. The final action is determined by auditable Python rules.

---

## Dataset & Data Pipeline

From the Kaggle Customer Support on Twitter (TWCS) dataset (~3 million tweets), @SpotifyCares conversations were extracted and processed:

```
RAW TWCS DATA (~3M tweets)
      ↓ Filter: author_id == "SpotifyCares" & inbound == True
Conversation Reconstruction (in_response_to_tweet_id backward traversal)
      ↓ Chronological merging of multi-part support tweets (e.g. 1/2, 2/2)
DERIVED SUPPORT CASES (40,794 paired cases in dataset/spotify_support_cases.csv)
      ↓ Anti-leakage exclusion: strictly remove all golden set examples
FAISS Vector Index (backend/src/retrieval/artifacts/spotify_cases.index)
```

### Extraction Caveat Handling
In TWCS, `response_tweet_id` can contain multiple comma-separated IDs or multi-part threads. AssistIQ handles this by tracing backward via `in_response_to_tweet_id` to establish root conversations and forward pairing, merging split agent responses chronologically into a single complete resolution.

### Golden Set Construction & Provenance
- **Size**: Exactly 200 representative customer messages (`dataset/golden_set.csv`).
- **Taxonomy**: 11-intent taxonomy formulated from recurring operational Spotify support workflows.
- **Labeling Process**: Initial candidate labels were generated with AI assistance, after which **all 200 examples were personally reviewed, verified, and adjudicated by the project author**.
- **Adjudication**: 14 initial candidate disagreements were identified and corrected in `dataset/golden_set_candidate_review.csv` (93.0% initial candidate agreement) before finalizing ground truth.
- **Anti-Leakage**: All 200 golden set tweets were strictly excluded by tweet ID and normalized text from the 40,794 historical retrieval index (`anti_leakage_overlap = 0`).

---

## Intent Taxonomy

The 11-intent taxonomy balances operational granularity with sample tractability:

| Intent | Description | Distribution ($N=200$) |
| :--- | :--- | :---: |
| `playback_and_app_issues` | Audio playback glitches, freezing, crashing, Bluetooth, offline sync | 31 (15.5%) |
| `search_and_discovery` | Search bar failures, recommendations, Discover Weekly | 3 (1.5%) |
| `account_and_login` | Password resets, compromised accounts, email changes, login locks | 15 (7.5%) |
| `billing_and_payment` | Charges, double renewals, payment methods, bank disputes | 19 (9.5%) |
| `premium_and_subscription` | Upgrades, Student verification (SheerID), Family plan address issues | 22 (11.0%) |
| `music_availability` | Missing songs, regional licensing restrictions, greyed-out tracks | 12 (6.0%) |
| `playlist_and_library` | Disappeared playlists, wiped offline downloads, library management | 11 (5.5%) |
| `feature_requests` | Feature suggestions (sleep timer, clean lyrics toggle, light mode) | 24 (12.0%) |
| `content_metadata` | Wrong song version, incorrect artist credit, lyrics/artwork typos | 8 (4.0%) |
| `ads_and_privacy` | Unwanted ads on Premium, tracking, data privacy concerns | 2 (1.0%) |
| `other_non_actionable` | Thanks, acknowledgements, greetings, vague venting without request | 53 (26.5%) |

---

## Retrieval

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors, L2 normalized).
- **Index Type**: `faiss.IndexFlatIP` (exhaustive inner product search, mathematically equivalent to exact cosine similarity).
- **Corpus**: 40,794 historical resolved Spotify support dialogues.
- **Why Historical Cases**: Retrieving resolved customer-support pairs provides demonstrated solutions rather than raw unanswered tweets.
- **Key Metrics ($N=200$)**:
  - Intent Hit Rate@1: **56.7% – 58.0%**
  - Intent Hit Rate@3: **80.0% – 80.5%**
  - Intent Hit Rate@5: **88.5% – 90.0%** (Headline Metric)
  - Intent Consistency@3: **54.4% – 56.8%**
  - Search Latency: **~25–45ms** (warm index, commodity CPU)
- **Manual Relevance Review ($N=35$ queries, 105 top-3 cases)**:
  - **Strict Precision@1**: **71.4%** (Directly actionable exact solution)
  - **Strict Precision@3**: **63.8%** (Directly actionable exact solution)
  - **Lenient Precision@3**: **100.0%** (Topically relevant context)

---

## Grounded Reply Generation

- **SDK**: Official Google GenAI SDK (`google-genai`), model `gemini-2.5-flash`.
- **Schema Enforcement**: Structured JSON output validated via Pydantic `SupportReply`:
  - `reply`: Customer-facing text.
  - `grounding_summary`: Explanation of supporting historical evidence.
  - `evidence_case_ids`: Case IDs explicitly cited.
  - `grounding_status`: `grounded`, `insufficient_evidence`, or `generation_failed`.
- **Guardrails**:
  - Grounding instructions prevent hallucinating refund commitments or credential changes.
  - Citation scrubber scrubs any hallucinated `case_id` not present in retrieved candidates.
  - Deterministic `MockLLMClient` provides zero-cost offline execution for CI and development.

---

## Escalation Policy

The deterministic `EscalationPolicy` evaluates rules in strict priority:
1. `E0_EMPTY_MESSAGE`: Message missing or empty.
2. `E3_GENERATION_FAILED`: LLM error or unparseable output.
3. `E4_UNGROUNDED_REPLY`: Model reports insufficient evidence grounding.
4. `E2_WEAK_RETRIEVAL`: Top retrieval cosine similarity $< 0.45$.
5. `E1_LOW_INTENT_CONFIDENCE`: Margin-derived intent confidence $< 0.20$.
6. `E5_ACCOUNT_SECURITY`: Account security keywords (hack, stolen, breach, password).
7. `E6_BILLING_ACTION`: Transactional billing keywords (refund, double charge, dispute).
8. `E7_AMBIGUOUS`: `other_non_actionable` query that is not a polite greeting.
9. Auto-handle rules: `E7_GREETING` (polite thanks/greetings), `E8_FEATURE_REQUEST` (grounded feature feedback), and `A1_DEFAULT_AUTOHANDLE` (confident intent + grounded retrieval).

- **Output Contract**: `decision` (`auto_handle` | `escalate`), `risk_level` (`low` | `medium` | `high`), `rule_id`, `rule_name`, and `explanation`.

---

## Evaluation Results

### 1. Intent Classification Benchmarks
Evaluated on the isolated 25% test set ($N=50$, random seed 2026, strictly unseen):

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Majority Class** | 0.2600 | 0.0260 | 0.1000 | 0.0413 | 0.1073 | < 0.01 ms |
| **Baseline 2: TF-IDF + Logistic Regression** | 0.4400 | 0.3024 | 0.3276 | 0.3101 | 0.4187 | 0.08 ms |
| **Proposed Model: TF-IDF + LinearSVC** | **0.4600** | **0.4241** | **0.3942** | **0.4004** | **0.4534** | 0.07 ms |

- **Why LinearSVC Wins**: Achieves a **+9.0 percentage point Macro F1 lift** over Logistic Regression (0.4004 vs 0.3101) by enforcing maximum geometric separation in sparse n-gram space, recovering minority classes like `music_availability` (0.40 F1 vs 0.00).
- **LinearSVC Confidence Disclosure**: LinearSVC outputs signed geometric margin distances. AssistIQ applies a temperature-scaled softmax to produce a **margin-derived confidence proxy**, not a calibrated Bayesian posterior probability.

### 2. Escalation Policy & Threshold Sensitivity
Evaluated across 40 representative golden interactions:
- **Auto-Handle Rate**: **60.0%** (24 / 40)
- **Escalation Rate**: **40.0%** (16 / 40)
- **Unsafe Auto-Handle Rate**: **0.0%** (Zero security or billing dispute tickets auto-handled)

#### 2D Sensitivity Grid (Auto-Handle Rate):
```
Confidence \ Similarity | Sim=0.35 | Sim=0.40 | Sim=0.45 (Default) | Sim=0.50 | Sim=0.55
------------------------+----------+----------+--------------------+----------+---------
Conf = 0.15             |  62.5%   |  62.5%   |       62.5%        |  62.5%   |  60.0%
Conf = 0.20 (Default)   |  60.0%   |  60.0%   |     **60.0%**      |  60.0%   |  57.5%
Conf = 0.25             |  47.5%   |  47.5%   |       47.5%        |  47.5%   |  45.0%
Conf = 0.30             |  32.5%   |  32.5%   |       32.5%        |  32.5%   |  30.0%
```
*Note*: The operating point (Confidence $\ge 0.20$, Similarity $\ge 0.45$) represents an **engineering tradeoff between automation and safety**, not a mathematically proven global optimum. Unsafe auto-handling creates high enterprise liability, while unnecessary escalation costs only minor human review time.

### 3. LLM Judge Quality
Evaluated across 5 dimensions on a 1.0 to 5.0 scale (`evaluation/results/reply_llm_judge.csv`, Mock Judge mode):
- **Relevance**: 4.47 / 5.0 (97.5% pass)
- **Groundedness**: 4.31 / 5.0 (93.5% pass)
- **Helpfulness**: 4.34 / 5.0 (95.0% pass)
- **Completeness**: 4.09 / 5.0 (90.0% pass)
- **Safety**: 4.96 / 5.0 (100.0% pass)
- **Overall Score**: **4.27 / 5.0** (94.0% pass rate)

### 4. Human vs. LLM Agreement
- **Production Status**: `HUMAN_REVIEW_REQUIRED`. Human evaluation columns in `evaluation/reply_review.csv` remain unpopulated awaiting double-blind human labels; we do not fabricate human data.
- **Synthetic Agreement Validation (`--demo-mock-human`)**: Verified the mathematical implementation of the agreement harness:
  - Pearson Correlation ($r$): `0.9412`
  - Spearman Rank Correlation ($\rho$): `0.9412`
  - **Quadratic Weighted Cohen's Kappa ($\kappa$)**: **`0.9412`**
  - *Why Quadratic Kappa*: Penalizes distant rating disagreements quadratically: $|1-5|$ disagreement has weight $1.0$, which is 16× more severe than a $|4-5|$ nuance (weight $0.0625$).

---

## Misleading Headline Number Deep Dive

> **Observed Metric: "88.5% Retrieval Intent Hit Rate@5"**
> **Audit Reality: "Strict Actionable Precision@3 is only 63.8%"**

### Why It Sounds Impressive:
Reporting an **88.5% Hit Rate@5** suggests historical retrieval surfaces relevant solutions in nearly 9 out of 10 customer interactions.

### Why It Is Misleading:
1. **Surrogate Heuristic**: Hit Rate@5 merely verifies that *at least one* historical ticket in the top 5 shares the predicted intent label. In dense clusters (`playback_and_app_issues` with 2,400+ vectors), finding at least one playback ticket is nearly guaranteed.
2. **Topical Overlap $\ne$ Actionable Solution**: A tweet complaining of *"Spotify desktop crashes on Windows 10"* and a retrieved ticket about *"music pauses on iPhone CarPlay"* both share the intent `playback_and_app_issues`. Hit Rate marks this as a success, but the iPhone steps are useless for fixing the Windows desktop crash.
3. **The 24.7-Point Drop**: Manual human auditing over 35 queries (105 cases) reveals **Strict Precision@3 is only 63.8%**. Over **36% of top-3 retrieved cases do not contain actionable solutions** for the customer's specific problem.
4. **Engineering Defense**: High retrieval recall creates false confidence. The escalation policy must enforce strict similarity thresholds ($\ge 0.45$) and verify device context before allowing autonomous replies.

---

## Failure Analysis

Pipeline errors are automatically diagnosed across 10 failure categories (`evaluation/results/failure_examples.csv`):

| Category | Description | Primary Root Cause | Component | Proposed Next Fix |
| :--- | :--- | :--- | :--- | :--- |
| `wrong_intent` | Misclassified intent | Short/ambiguous follow-up tweet | Phase 1 Classifier | Include parent tweet in feature vector |
| `low_retrieval_similarity` | Top similarity $< 0.45$ | Unsupported hardware/niche query | Phase 2 Retrieval | Escalated cleanly via rule `E2` |
| `irrelevant_retrieved_evidence`| Lexical false match | Polysemous keyword (e.g. "release") | Phase 2 Embedding | Add cross-encoder reranker |
| `unsupported_generation` | Uncited factual advice | Missing verified citations in prompt | Phase 3 Prompt | Reject generation if similarity $< 0.45$ |
| `hallucinated_claim` | Promising refund/action | LLM adopting overly helpful persona | Phase 3 Generation | Strict prompt ban; escalated via `E6` |
| `missing_required_clarification`| No device check | Customer omitted OS/device context | Phase 3 Policy | Add mandatory clarification gate |
| `unsafe_auto_handle` | Login failure auto-handled | Private credentials in public tweet | Phase 4 Policy | Escalate account access queries via `E5` |
| `unnecessary_escalation` | Grounded FAQ escalated | Conservative confidence cutoff | Phase 4 Thresholds | Calibrate probabilities via Platt scaling |
| `low_quality_response` | Generic boilerplate reply | Multi-part question partially answered| Phase 3 Generation | Penalize incomplete replies in LLM judge |
| `ambiguous_customer_message` | Emotional complaint | Customer vented without stating bug | Phase 4 Escalation | Escalate cleanly via rule `E7_AMBIGUOUS` |

---

## Key Decisions

14 non-obvious engineering decisions are documented in detail in [DECISION_LOG.md](./DECISION_LOG.md):
1. **Target Brand Selection**: @SpotifyCares from TWCS (~40K conversations) for bounded technical troubleshooting.
2. **11-Intent Taxonomy**: Formulated domain-specific schema over generic 3-class sentiment.
3. **Pipeline Encapsulation**: Strict anti-leakage guarantee using scikit-learn Pipelines.
4. **TF-IDF + LinearSVC**: Maximum geometric margin classifier chosen over deep transformers for CPU speed (< 2ms) and sample efficiency on $N=150$.
5. **Multi-Turn Case Pairing**: Grouped customer tweets and agent replies chronologically into grounded historical cases.
6. **SentenceTransformer all-MiniLM-L6-v2**: 384-d normalized embeddings balancing recall and CPU inference speed (~25ms).
7. **FAISS IndexFlatIP**: Exhaustive exact cosine similarity avoiding approximate nearest-neighbor recall loss.
8. **0.45 Retrieval Similarity Cutoff**: Initial operational engineering heuristic, not a mathematically optimal global cutoff.
9. **Deterministic Escalation Policy**: Rule engine retains sole authority; LLM never decides escalation.
10. **Mandatory Sensitive Escalation**: Strict rules for billing refunds (`E6`) and account security (`E5`).
11. **LinearSVC Margin Confidence**: Softmax over decision function margins disclosed as uncalibrated pseudo-confidence.
12. **Structured Pydantic Contract**: Schema enforcement with post-generation citation verification.
13. **Mock LLM & Mock Judge**: Zero-cost offline execution for continuous integration and deterministic testing.
14. **Decoupled Architecture**: Independent FastAPI backend and Next.js frontend communicating via REST.

---

## Limitations

- **Single-Reviewer Golden Set**: While all 200 golden examples were personally verified by the project author, enterprise production requires multi-annotator blind labeling with Fleiss' Kappa.
- **Static Retrieval Corpus**: The FAISS index is built on historical 2017 TWCS data and lacks knowledge of contemporary Spotify features (AI DJ, Daylist, Jam).
- **Uncalibrated Confidence**: LinearSVC confidence scores provide monotonic rankings but are not calibrated Bayesian posterior probabilities.

---

## Next-Week Plan

### P0 — Safety & Correctness
- **Platt Scaling / Isotonic Calibration**: Fit a sigmoid calibrator (`CalibratedClassifierCV`) to convert LinearSVC decision margins into true posterior probabilities.
- **Mandatory Clarification Gate**: Require the model to prompt for device/OS before suggesting platform-specific remedies for `playback_and_app_issues`.

### P1 — Quality
- **Cross-Encoder Reranker**: Integrate `cross-encoder/ms-marco-MiniLM-L-6-v2` to rerank Top-10 FAISS candidates down to Top-3, targeting Strict Precision@3 lift from 63.8% to $> 80\%$.
- **Dense + Sparse Hybrid Retrieval**: Combine FAISS dense embeddings with BM25 keyword matching via Reciprocal Rank Fusion (RRF) to resolve alphanumeric error codes (e.g. "Error 104").
- **Intent-Filtered Retrieval**: Partition the FAISS search space by predicted intent to eliminate cross-domain lexical matches.

### P2 — Scale & Observability
- **Expanded Golden Set**: Expand evaluation set from 200 to 500 examples using active learning uncertainty sampling with multi-annotator agreement.
- **In-Browser Review Dashboard**: Add a review UI in Next.js to allow support leads to score replies and record human reviews directly.

---

## Project Structure

```
AssistIQ/
├── backend/
│   ├── api/
│   │   ├── main.py              # FastAPI application & CORS setup
│   │   ├── routes.py            # /api/v1/assist endpoint
│   │   └── schemas.py           # Request / response Pydantic models
│   └── src/
│       ├── intent/              # Phase 1: Classifier, baselines, taxonomy
│       ├── retrieval/           # Phase 2: FAISS index, embeddings, case pairing
│       ├── generation/          # Phase 3: Gemini SDK integration, Pydantic schemas
│       └── escalation/          # Phase 4: Deterministic policy engine (E0–E8, A1)
├── dataset/
│   ├── golden_set.csv           # 200 verified evaluation examples
│   ├── golden_set_candidate_review.csv # Candidate review & adjudication log
│   └── spotify_support_cases.csv# 40,794 historical support cases
├── evaluation/
│   ├── run.py                   # Master evaluation CLI runner
│   ├── validate_golden.py       # Golden set schema & integrity validator
│   ├── evaluate_intent.py       # 3-model intent benchmark harness
│   ├── evaluate_retrieval.py    # FAISS retrieval benchmark & latency profiler
│   ├── llm_judge.py             # 5-dimension LLM judge (Live & Mock)
│   ├── human_agreement.py       # Pearson, Spearman, Weighted Cohen's Kappa
│   ├── evaluate_escalation.py   # Escalation policy & 2D sensitivity grid
│   ├── failure_analysis.py      # 10-bucket failure taxonomy diagnostics
│   ├── reply_review.csv         # Human review template (blank by default)
│   └── results/                 # Output CSVs, JSON reports, confusion matrices
├── frontend/                    # Phase 6: Next.js 16 (React 19 / TypeScript)
├── tests/                       # 65 automated unit & regression tests
├── DECISION_LOG.md              # 14 non-obvious engineering decisions
├── EVALUATION_REPORT.md         # Formal evaluation report (< 6 pages equivalent)
└── README.md                    # Project documentation & execution guide
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Node.js 18+ and npm
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/joshi-chinmay-016/AssistIQ.git
cd AssistIQ

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

---

## Running the Backend

Start the FastAPI backend with Uvicorn:
```bash
uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger API documentation is available at `http://127.0.0.1:8000/docs`.

---

## Running the Frontend

Start the Next.js development server:
```bash
cd frontend
npm run dev
```
Open `http://localhost:3000` to interact with the support agent console.

---

## Running Evaluation

The evaluation framework supports granular stage execution and zero-cost offline mock execution:

```bash
# Run entire evaluation suite offline (0.00 cost, ~25 seconds runtime)
python -m evaluation.run --stage all --mode mock

# Run individual evaluation stages
python -m evaluation.run --stage golden       # Stage 1: Golden set validation
python -m evaluation.run --stage intent       # Stage 2: Intent classification benchmark
python -m evaluation.run --stage retrieval    # Stage 3: Retrieval metrics & latency
python -m evaluation.run --stage reply        # Stage 4: Reply generation
python -m evaluation.run --stage judge        # Stage 5: 5-dimension LLM judge
python -m evaluation.run --stage agreement    # Stage 6: Human-LLM agreement analysis
python -m evaluation.run --stage escalation   # Stage 7: Escalation & sensitivity grid
python -m evaluation.run --stage failures     # Stage 8: Failure taxonomy diagnostics

# Run live evaluation with Google Gemini (requires GEMINI_API_KEY)
python -m evaluation.run --stage all --mode live --limit 30

# Verify agreement calculation formulas with synthetic mock ratings
python -m evaluation.human_agreement --demo-mock-human
```

---

## Running Automated Tests

Run the complete test suite across all modules:
```bash
python -m pytest tests
```
*Result: 65 passed in ~30s (48 core pipeline tests + 17 evaluation framework tests).*

---

## Environment Variables

Create a `backend/.env` file (or set variables in your shell):
```env
# Optional: Only required for live Gemini drafting or live LLM judge
GEMINI_API_KEY=your_gemini_api_key_here

# Optional model configurations
GEMINI_MODEL=gemini-2.5-flash
APP_ENV=development
LOG_LEVEL=INFO
```
*Note: If `GEMINI_API_KEY` is not provided, AssistIQ automatically falls back to deterministic mock execution for both generation and evaluation.*

---

## Reproducibility & Measured Runtimes

All evaluation results were measured on commodity hardware (Windows 11, AMD/Intel CPU, no GPU required):

| Stage | Command | Measured Runtime | Output Artifacts |
| :--- | :--- | :---: | :--- |
| **Golden Validation** | `python -m evaluation.validate_golden` | 0.01s | `golden_set_validation_report.json` |
| **Intent Benchmark** | `python -m evaluation.evaluate_intent` | 0.13s | `intent_metrics.csv`, `intent_confusion_matrix.csv` |
| **Retrieval Benchmark** | `python -m evaluation.evaluate_retrieval` | 5.80s | `retrieval_metrics.csv`, `retrieval_errors.csv` |
| **Reply & LLM Judge** | `python -m evaluation.llm_judge` | 18.66s | `reply_llm_judge.csv` |
| **Human Agreement** | `python -m evaluation.human_agreement` | 0.01s | `human_llm_agreement.csv` |
| **Escalation & Grid** | `python -m evaluation.evaluate_escalation`| 0.02s | `escalation_metrics.csv`, `threshold_sensitivity.csv` |
| **Failure Analysis** | `python -m evaluation.failure_analysis` | 0.02s | `failure_examples.csv` |
| **Full Suite Runner** | `python -m evaluation.run --stage all --mode mock` | **24.64s** | Populates all 20 artifacts in `evaluation/results/` |

---

## Security & Secret Management

- **Zero Secret Leaks**: No API keys, credentials, or private tokens are stored in the repository.
- **Git Hygiene**: `.gitignore` strictly excludes `.env`, `backend/.env`, raw Twitter TWCS dumps, cache directories, and virtual environments.
- **Mock Fallback**: Tests and CI workflows run completely offline with zero risk of quota exhaustion or credential exposure.

---

## Interview Notes & Key Design Defenses

### 1. Why LinearSVC instead of fine-tuning BERT / RoBERTa?
On short customer tweets with a 150-example training set, deep transformers easily overfit, require GPU resources, and add ~50–100ms latency. LinearSVC finds the maximum geometric margin in high-dimensional sparse n-gram space, running in < 2ms on CPU and outperforming Logistic Regression by **+9.0 percentage points in Macro F1** (0.4004 vs 0.3101).

### 2. Why deterministic escalation rules instead of letting Gemini decide?
In enterprise customer service, escalation decisions must be compliant, predictable, and fully auditable. Delegating escalation to an LLM exposes the workflow to prompt injections, non-deterministic drift, and hallucinated policy commitments. An auditable Python rule engine guarantees zero bypasses of sensitive account security or refund escalation.

### 3. Why FAISS IndexFlatIP instead of HNSW or IVF?
With 40,794 384-dimensional vectors, exact inner product search (`IndexFlatIP`) takes ~25ms on CPU with a 62MB memory footprint. Approximate nearest neighbors (HNSW/IVF) introduce recall degradation and indexing hyperparameters with negligible latency benefit at this corpus size.

### 4. What does the "88.5% Hit Rate@5" headline number actually mean?
It is a broad intent-matching recall metric, not a measure of solution correctness. Manual audit shows that **Strict Precision@3 is only 63.8%**—a 24.7 percentage point drop—highlighting the necessity of strict retrieval similarity thresholds ($\ge 0.45$) to prevent ungrounded auto-handling.
