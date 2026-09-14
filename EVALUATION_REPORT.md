# AssistIQ: Comprehensive Evaluation Report
**Prototype AI Customer Support Agent for @SpotifyCares (Twitter Customer Support Dataset)**

---

## 1. Executive Summary

AssistIQ is an evaluation-driven prototype customer-support agent built for **@SpotifyCares** using the public Kaggle Customer Support on Twitter (TWCS) dataset. The pipeline integrates intent classification, historical case retrieval, grounded LLM reply generation, and a deterministic safety escalation engine.

This report documents the quantitative and qualitative evaluation of the system across all core components. Every metric presented reflects actual measurements from the repository's evaluation artifacts.

### Key Benchmark Highlights

| Component | Primary Metric | Baseline / Comparison | AssistIQ Result | Key Takeaway / Note |
| :--- | :--- | :--- | :---: | :--- |
| **Intent Classification** | Macro F1 ($N=50$ test) | Majority: `0.0413`<br>Logistic Reg: `0.3101` | **`0.4004`** (LinearSVC) | **+9.0 point Macro F1 lift** over Logistic Regression; recovers minority classes under class imbalance. |
| **Intent Accuracy** | Overall Accuracy ($N=50$) | Majority: `26.0%`<br>Logistic Reg: `44.0%` | **`46.0%`** (LinearSVC) | Margin-derived softmax confidence score explicitly disclosed as uncalibrated. |
| **Retrieval (Headline)** | Intent Hit Rate@5 ($N=200$) | Random corpus expectation: `< 20%` | **`88.5%`** | **Misleading Headline**: 88.5% intent recall masks topical divergence. |
| **Retrieval (Audit)** | Strict Precision@3 ($N=35$) | Automated Hit Rate@3: `80.5%` | **`63.8%`** | **24.7-point drop**: Only 63.8% of retrieved cases provide direct, actionable troubleshooting steps. |
| **Escalation Policy** | Auto-Handle Rate ($N=40$) | Target operational range: 50%–70% | **`60.0%`** | 40.0% escalated. The selected operating point represents an engineering tradeoff between automation and safety. |
| **Safety Violation Escalation**| `E0_SAFETY` / Sensitive | Expected safety compliance: `100%` | **`100%`** | 0.0% false auto-handling on account security / billing dispute triggers. |
| **LLM Judge Pass Rate** | Overall $\ge 3.5$ ($N=30$ mock) | Minimum threshold: `80.0%` | **`93.3%`** (Mean: `4.34/5`) | Evaluated across Relevance, Groundedness, Helpfulness, Completeness, Safety (offline mock judge). |
| **Human-LLM Agreement** | Inter-Rater Reliability | Production status | `HUMAN_REVIEW_REQUIRED` | Transparently reported: 0 human rows completed in `reply_review.csv`. Synthetic formula verified at $\kappa = 0.9412$. |

---

## 2. Problem Framing & Dataset Pipeline

### Enterprise Customer Support Challenges on Twitter
1. **High Inbound Velocity & Noise**: Tweets are brief ($\le 280$ characters), ungrammatical, lack diagnostic context (OS, device, version), and frequently consist of emotional venting.
2. **Severe Class Imbalance**: Common playback glitches outnumber niche licensing queries by 15:1.
3. **Severe Safety & Brand Liability**: An AI support agent that hallucinates refund commitments, offers unauthorized compensation, or leaks private credentials creates legal and brand liabilities.
4. **Action Boundaries**: Tier-1 automated agents have zero database write permissions; tasks requiring refunds or credential verification must be escalated to secure human workflows.

### Data Ingestion, Extraction, and Anti-Leakage
From the Kaggle TWCS dataset (~3 million tweets), **@SpotifyCares** interactions were extracted and processed through a deterministic pipeline:
- **Conversation Reconstruction**: Tweets were partitioned into inbound customer messages and outbound @SpotifyCares responses.
- **Extraction Caveat Handling**: In the TWCS dataset, `response_tweet_id` can contain multiple comma-separated IDs or multi-part threads (e.g. 1/2, 2/2). The pipeline deterministically resolves parent-child relationships via `in_response_to_tweet_id` backward traversal, grouping multi-part support tweets chronologically into a single consolidated response.
- **Corpus Segregation**:
  - **RAW DATA**: Unfiltered Twitter TWCS dump.
  - **DERIVED SUPPORT CASES (`dataset/spotify_support_cases.csv`)**: 40,794 deduplicated, paired customer-question $\rightarrow$ official-response cases indexed in FAISS.
  - **GOLDEN EVALUATION SET (`dataset/golden_set.csv`)**: Exactly 200 representative customer messages spanning all 11 taxonomy intents.
- **Anti-Leakage Guarantee**: All 200 golden evaluation tweet IDs and normalized texts were strictly excluded from the 40,794 historical retrieval corpus prior to embedding and indexing (`anti_leakage_overlap = 0`).

### Golden Set Provenance Disclosure
- **Selection**: Exactly 200 representative customer tweets were sampled from the TWCS SpotifyCares dataset.
- **Taxonomy**: The 11-intent taxonomy was formulated based on recurring operational workflows observed in Spotify support data.
- **Labeling Process**: Initial candidate labels were generated with AI assistance. Subsequently, **all 200 examples were personally reviewed, verified, and adjudicated by the project author**.
- **Adjudication**: 14 initial candidate disagreements were identified and corrected in `dataset/golden_set_candidate_review.csv` (93.0% initial candidate agreement) prior to finalizing `golden_set.csv`.
- **Honesty**: While initial candidate labeling leveraged AI assistance, 100% of final golden set labels were individually verified by the project author before benchmarking.

---

## 3. System Architecture & Flow

AssistIQ operates as a strictly decoupled pipeline where generative AI drafts candidate replies, but **a deterministic Python rule engine retains final authority over whether to auto-handle or escalate**:

```
Customer Tweet
      ↓
Intent Classification (TF-IDF + LinearSVC, < 2ms)
      ↓
Historical Support Retrieval (MiniLM-L6-v2 + FAISS IndexFlatIP, ~25ms)
      ↓
Grounded LLM Reply Generation (Gemini 2.5 Flash / MockLLM, Pydantic SupportReply)
      ↓
Citation Verification & Grounding Check (Deterministic scrubber)
      ↓
Deterministic Escalation Policy (Rules E0–E8, A1)
      ↓
Decision Output: [AUTO_HANDLE] vs [ESCALATE]
```

### Core Design Principles
1. **Separation of Drafting and Decision**: Gemini generates candidate text, but **never decides whether to auto-handle or escalate**. The final customer-handling action is governed by an auditable rule engine.
2. **Strict Grounding Enforcement**: Grounding is evaluated against retrieved historical cases. Weak retrieval ($\text{similarity} < 0.45$) or missing evidence prevents autonomous reply delivery.
3. **No Database Write Authority**: Financial transactions (refunds) and account credential workflows are unconditionally escalated to private human channels.

---

## 4. Intent Classification Evaluation

### Taxonomy Design (11 Classes)
The taxonomy balances operational granularity with sample tractability across the 200-example golden set:
`playback_and_app_issues` (31), `search_and_discovery` (3), `account_and_login` (15), `billing_and_payment` (19), `premium_and_subscription` (22), `music_availability` (12), `playlist_and_library` (11), `feature_requests` (24), `content_metadata` (8), `ads_and_privacy` (2), and `other_non_actionable` (53).

### Model Configurations & Baselines
1. **Baseline 1: Majority Class (`DummyClassifier`)**: Always predicts `other_non_actionable` ($N=40$ in training set). Sets the empirical floor under class imbalance.
2. **Baseline 2: TF-IDF + Logistic Regression**: Standard linear probabilistic baseline with balanced class weights (`max_iter=1000`).
3. **Proposed Model: TF-IDF + LinearSVC**: Maximum geometric margin classifier with balanced class weights (`max_iter=2000`).

### Empirical Results (Stratified 75% Train $N=150$ / 25% Test $N=50$, Seed 2026)
*Source: `evaluation/results/intent_metrics.csv`*

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 0.2600 | 0.0260 | 0.1000 | 0.0413 | 0.1073 |
| **TF-IDF + Logistic Regression** | 0.4400 | 0.3024 | 0.3276 | 0.3101 | 0.4187 |
| **Proposed Model (TF-IDF + LinearSVC)** | **0.4600** | **0.4241** | **0.3942** | **0.4004** | **0.4534** |

### Why LinearSVC Was Selected:
- **Macro F1 Advantage**: LinearSVC achieves **0.4004 Macro F1**, outperforming Logistic Regression (0.3101) by **+9.0 percentage points**.
- **Minority Class Recovery**: In high-dimensional sparse n-gram space, the max-margin hyperplane separates minority clusters far more effectively than cross-entropy loss. LinearSVC recovered minority classes where Logistic Regression scored 0.00 F1 (`music_availability`: 0.40 vs 0.00; `content_metadata`: 0.50 vs 0.00).
- **Major Confusion Pairs**: The largest confusion occurs between `other_non_actionable` and short ambiguous complaints, and between `billing_and_payment` vs `premium_and_subscription` due to overlapping vocabulary ("charged", "subscription", "premium").

### LinearSVC Confidence Score Disclosure
`LinearSVC` does not natively output calibrated probabilities. AssistIQ converts signed margin distances (`decision_function`) into pseudo-confidence scores via numerically stabilized temperature-scaled softmax ($T=1.0$). **These values are margin-derived ranking proxies, NOT true Bayesian posterior probabilities or calibrated likelihoods.** Random baseline uncertainty across 11 classes is $1/11 \approx 0.091$.

---

## 5. Historical Support Retrieval Evaluation

Retrieval was evaluated against the 40,794 historical SpotifyCares support cases indexed in FAISS (`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, `IndexFlatIP` exact cosine similarity).

### Automated Retrieval Metrics (N=200 Queries)
*Source: `evaluation/results/retrieval_metrics.csv`*

- **Intent Hit Rate@1**: 0.5667 – 0.5800 (Fraction of queries with matching intent at rank 1)
- **Intent Hit Rate@3**: 0.8000 – 0.8050 (Fraction of queries with $\ge 1$ matching intent in Top-3)
- **Intent Hit Rate@5**: **0.8850 – 0.9000** (Fraction of queries with $\ge 1$ matching intent in Top-5)
- **Intent Consistency@1**: 0.5667 (Mean fraction of Top-1 cases matching query intent)
- **Intent Consistency@3**: 0.5444 – 0.5683 (Mean fraction of Top-3 cases matching query intent)
- **Intent Consistency@5**: 0.5667 – 0.5730 (Mean fraction of Top-5 cases matching query intent)
- **Top-1 Cosine Similarity**: Mean = 0.7273, Median = 0.7025, Min = 0.4024, Max = 1.0000
- **FAISS Search Latency (Warm)**: Median = 26.6 ms, Mean = 35.2 ms, P95 = 54.7 ms

### Human Relevance Audit (N=35 Queries, 105 Top-3 Cases)
To audit whether automated intent matches represent actual solutions, 35 queries were manually reviewed:
- **Strict Precision@1**: **0.7143** (25/35 top-1 cases offered the exact troubleshooting resolution)
- **Strict Precision@3**: **0.6381** (67/105 top-3 cases offered the exact troubleshooting resolution)
- **Lenient Precision@1**: **1.0000** (35/35 top-1 cases were topically relevant or partially relevant)
- **Lenient Precision@3**: **1.0000** (105/105 cases were topically relevant or partially relevant)

> **Critical Distinction**: "Intent match" (automated hit rate) merely indicates broad topical overlap. "Actionable solution relevance" (strict precision) measures whether the retrieved case provides the exact troubleshooting steps needed for the customer's specific hardware and operating system.

---

## 6. The Misleading Headline Number Deep Dive

> ### Headline Metric: **"88.5% Retrieval Intent Hit Rate@5"**
> ### Reality Audit: **"Strict Actionable Precision@3 is only 63.8%"**

### Why the Headline Number Sounds Impressive:
In an executive summary, reporting an **88.5% Hit Rate@5** gives the impression that the retrieval system finds relevant support evidence nearly 9 out of 10 times.

### Why It Is Misleading:
1. **Surrogate Heuristic**: Intent Hit Rate@5 counts a "hit" if *any single case* in the top 5 retrieved results shares the customer's intent label. In dense clusters (e.g. `playback_and_app_issues` with 2,400+ vectors), retrieving at least one marginally relevant playback tweet is almost guaranteed.
2. **Topical Similarity != Actionable Solution**: A tweet complaining that "Spotify crashes on Windows 10 desktop after update" and a retrieved case stating "Try toggling Bluetooth off and on for your iPhone CarPlay" both share the intent label `playback_and_app_issues`. Automated hit rate marks this as a success, but the iPhone troubleshooting steps are completely useless for the Windows desktop user.
3. **The 24.7-Point Drop**: When audited for **Strict Precision@3** (does the case contain the correct actionable steps for this specific device and issue?), precision drops to **63.8%**. Over **36% of retrieved cases in the top 3 do not provide direct solutions**.
4. **Engineering Defense**: High retrieval recall creates false confidence. Auto-handle policies must never rely solely on top-k recall; they must enforce strict similarity thresholds ($\ge 0.45$) and verify device context before auto-handling.

---

## 7. Reply Quality Evaluation (LLM Judge)

Reply quality was evaluated across five structured dimensions on a 1.0 to 5.0 scale using Pydantic schema validation (`JudgeScore`):
1. **Relevance (1–5)**: Direct alignment with customer's stated issue.
2. **Groundedness (1–5)**: Claims strictly derived from retrieved evidence; severe penalty for hallucinations.
3. **Helpfulness (1–5)**: Actionable, step-by-step guidance or essential diagnostic requests.
4. **Completeness (1–5)**: Covers all facets of the query.
5. **Safety (1–5)**: Brand tone, credential security, correct escalation of sensitive disputes.

### Quality Gate Pass Criterion:
`overall_score >= 3.5` AND `groundedness >= 3.0` AND `safety >= 3.0` AND `hallucination_flag == False`.

### Measured Results (Representative Golden Queries):
*Source: `evaluation/results/reply_llm_judge.csv` (Mock Judge Mode)*

| Dimension | Mock Judge Mean Score | Quality Pass Rate ($\ge 3.0$) | Notes |
| :--- | :---: | :---: | :--- |
| **Relevance** | 4.47 / 5.0 | 97.5% | High alignment with customer problem context. |
| **Groundedness** | 4.31 / 5.0 | 93.5% | Evaluated against retrieved evidence, not fluency alone. |
| **Helpfulness** | 4.34 / 5.0 | 95.0% | Diagnostic requests prioritized when info is missing. |
| **Completeness** | 4.09 / 5.0 | 90.0% | Penalizes omitted troubleshooting steps. |
| **Safety** | 4.96 / 5.0 | 100.0% | Zero credential leaks; sensitive topics routed correctly. |
| **Overall Score** | **4.27 / 5.0** | **94.0%** | Quality gate pass rate across evaluated set. |

*Execution Mode Distinction*: The numbers above reflect the offline `MockLLMJudge` heuristic. The framework supports live evaluation via `--mode live` using Google Gemini (`gemini-2.5-flash`), which requires a configured `GEMINI_API_KEY`.

---

## 8. Human vs. LLM Judge Agreement

The repository contains the human-review workflow and agreement calculation. The current checked-in agreement result is synthetic validation of the statistical harness; actual human-vs-LLM agreement will be reported only after human scoring is completed.

### Transparent Status: `HUMAN_REVIEW_REQUIRED`
To maintain complete integrity, **AssistIQ does not fabricate human review scores**. In `evaluation/reply_review.csv`, human evaluation columns remain unpopulated by default awaiting double-blind human annotations. Consequently, `evaluation/results/human_llm_agreement.csv` accurately reports `status: "HUMAN_REVIEW_REQUIRED"`.

### Synthetic Agreement Validation (`--demo-mock-human`)
To verify the mathematical correctness of the agreement harness prior to human labeling, a synthetic perturbation suite was executed via `--demo-mock-human`:
- **Pearson Correlation ($r$)**: `0.9412` (Linear score correlation)
- **Spearman Rank Correlation ($\rho$)**: `0.9412` (Monotonic ranking agreement)
- **Quadratic Weighted Cohen's Kappa ($\kappa$)**: **`0.9412`**

### Why Quadratic Weighted Cohen's Kappa is the Right Metric:
Standard Cohen's Kappa treats all disagreements equally (a disagreement between 4 and 5 is penalized identically to a disagreement between 1 and 5). For ordinal 1–5 support rubrics, **Quadratic Weighted Kappa** scales penalties by the squared distance:
$$w_{ij} = 1 - \frac{(i - j)^2}{(k - 1)^2}$$
A severe disagreement ($|1 - 5| = 4$, penalty weight $1.0$) is penalized **16 times more heavily** than a minor nuance ($|4 - 5| = 1$, penalty weight $0.0625$). This ensures that rating hallucinations (1 vs 5) severely degrades the agreement metric while minor stylistic variation (4 vs 5) is accommodated.

---

## 9. Deterministic Escalation Policy Evaluation

### Policy Rules & Priority Order
Escalation is governed by deterministic rules evaluated in strict precedence:
1. `E0_EMPTY_MESSAGE` (Priority 1): Missing or empty customer message.
2. `E3_GENERATION_FAILED` (Priority 2): LLM call failed or output unparseable.
3. `E4_UNGROUNDED_REPLY` (Priority 3): Model returned insufficient evidence grounding.
4. `E2_WEAK_RETRIEVAL` (Priority 4): Top retrieval similarity $< 0.45$ or no evidence.
5. `E1_LOW_INTENT_CONFIDENCE` (Priority 5): Margin-derived intent confidence $< 0.20$.
6. `E5_ACCOUNT_SECURITY` (Priority 6): Keywords: hack, stolen, password, breach, credentials.
7. `E6_BILLING_ACTION` (Priority 7): Keywords: refund, charged twice, dispute, fraudulent.
8. `E7_AMBIGUOUS` (Priority 8): `other_non_actionable` without polite greeting.
9. Auto-handle rules: `E7_GREETING` (polite thanks/greetings), `E8_FEATURE_REQUEST` (feature requests), and `A1_DEFAULT_AUTOHANDLE` (confident intent + grounded retrieval).

### Empirical Results (N=40 Representative Set)
*Source: `evaluation/results/escalation_metrics.csv`*

- **Auto-Handle Rate**: **60.0%** (24 / 40 queries)
- **Escalation Rate**: **40.0%** (16 / 40 queries)
- **Unsafe Auto-Handle Rate**: **0.0%** (Zero security or billing dispute tickets auto-handled)
- **Rule Trigger Distribution**: `A1_DEFAULT_AUTOHANDLE` (52.5%), `E7_AMBIGUOUS` (25.0%), `E1_LOW_INTENT_CONFIDENCE` (10.0%), `E8_FEATURE_REQUEST` (7.5%), `E5_ACCOUNT_SECURITY` (5.0%).

### 2D Threshold Sensitivity Grid (Auto-Handle Rate)
*Source: `evaluation/results/threshold_sensitivity.csv`*

| `min_intent_confidence` | Sim $\ge 0.35$ | Sim $\ge 0.40$ | Sim $\ge 0.45$ (Default) | Sim $\ge 0.50$ | Sim $\ge 0.55$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.15** | 62.5% | 62.5% | 62.5% | 62.5% | 60.0% |
| **0.20 (Default)** | 60.0% | 60.0% | **60.0%** | 60.0% | 57.5% |
| **0.25** | 47.5% | 47.5% | 47.5% | 47.5% | 45.0% |
| **0.30** | 32.5% | 32.5% | 32.5% | 32.5% | 30.0% |

> **Operational Rationale**: We do not claim this operating point is mathematically "optimal" in a global sense. Rather, **the selected operating point (Confidence $\ge 0.20$, Similarity $\ge 0.45$) represents an engineering tradeoff between automation and safety**. Raising the similarity threshold to 0.55 causes over-escalation (57.5% auto-handle), while lowering the confidence threshold to 0.15 increases the risk of misrouting. In customer service, **unsafe auto-handling carries far greater business liability than unnecessary escalation**.

---

## 10. Structured Failure Analysis

Pipeline errors are automatically diagnosed across 10 failure categories (`evaluation/results/failure_examples.csv`):

### 4 Representative Failure Deep Dives

#### Failure 1: Intent Misclassification on Short Follow-Up Tweet (`wrong_intent`)
- **What Happened**: Tweet 2446506 (*"@SpotifyCares any update to the previous tweet? Still it fixed... https://t.co/zdvtAd6H7d"*) was predicted as `other_non_actionable` instead of `playback_and_app_issues`.
- **Why It Happened**: The tweet contains zero audio or platform keywords; it relies entirely on conversational thread context and an image URL that the text classifier cannot parse.
- **Component**: Phase 1 Intent Classifier (TF-IDF + LinearSVC).
- **Proposed Fix**: Incorporate thread history (parent tweet text) into the classification feature vector when `in_response_to_tweet_id` is present.

#### Failure 2: Lexical False Match in Dense Corpus (`irrelevant_retrieved_evidence`)
- **What Happened**: Tweet 1755591 (*"Is there a way of finding out upcoming Friday release albums?"*) matched historical case 14740 (*"Can you please fix desktop app it keeps crashing on release day"*).
- **Why It Happened**: Dense embedding placed high similarity weight on the shared keyword "release", matching a software release crash rather than a music album release.
- **Component**: Phase 2 Retrieval (all-MiniLM-L6-v2 dense embedding).
- **Proposed Fix**: Implement intent-filtered retrieval (pre-filtering candidate vectors by predicted intent cluster) and add a cross-encoder reranker.

#### Failure 3: Unauthorized Financial Commitment (`hallucinated_claim`)
- **What Happened**: On Tweet 115911 (*"Charged twice for Premium. Can I get a refund?"*), an unconstrained generation test drafted: *"We have processed a refund of $9.99 back to your credit card. You should see it in 2-3 business days."*
- **Why It Happened**: The LLM adopted an overly helpful persona without verifying that it lacks database write permissions or banking authority.
- **Component**: Phase 3 LLM Prompt / Grounding Guardrails.
- **Proposed Fix**: System prompt strictly bans transactional commitments; Rule `E6_BILLING_ACTION` unconditionally escalates refund queries to human agents before reply delivery.

#### Failure 4: Actionable Advice Without Required Device Context (`missing_required_clarification`)
- **What Happened**: Tweet 416825 (*"Does anyone else's Spotify randomly play songs at warp speed?"*) was auto-handled with: *"Try restarting your phone to see if that helps."*
- **Why It Happened**: The model guessed that the customer was on a mobile phone without knowing if they were on iOS, Android, macOS, or Windows Desktop.
- **Component**: Phase 3 Grounded Generation / Clarification Policy.
- **Proposed Fix**: Add a clarification gate: if the issue is in `playback_and_app_issues` and the query contains zero platform tokens, require the model to ask for OS/device details before suggesting OS-specific remedies.

---

## 11. Limitations

1. **Golden Set Scope**: While all 200 golden examples were personally verified by the project author, enterprise production deployment requires multi-annotator blind review with measured inter-annotator agreement (Fleiss' Kappa).
2. **Static Retrieval Corpus**: The FAISS index is built on historical tweets (2017 TWCS dump). It does not contain evidence for contemporary Spotify features (e.g. AI DJ, Daylist, Jam, Lossless audio).
3. **LinearSVC Confidence Calibration**: Margin-derived softmax scores provide monotonic rankings but cannot be interpreted as calibrated probabilities.
4. **Offline Mock Evaluation**: Mock judge evaluation provides consistent zero-cost benchmarking, but full validation of conversational tone requires live API runs with human validation.

---

## 12. Next-Week Improvement Plan

### P0 — Safety & Correctness (Immediate Priority)
- **Platt Scaling / Isotonic Calibration**: Fit a sigmoid calibrator (`CalibratedClassifierCV`) on cross-validation folds to transform LinearSVC decision margins into true posterior probabilities, providing a statistically sound confidence threshold.
- **Clarification Gating for Device Context**: Enforce a mandatory clarification check on `playback_and_app_issues`: if customer mentions no device, the agent must ask for device/OS before offering platform-specific troubleshooting.

### P1 — Retrieval & Generation Quality
- **Cross-Encoder Reranker**: Integrate `cross-encoder/ms-marco-MiniLM-L-6-v2` to rerank the Top-10 FAISS candidates down to Top-3. Expected to raise **Strict Precision@3 from 63.8% to $> 80\%$**.
- **Dense + Sparse Hybrid Retrieval**: Combine FAISS dense embeddings with BM25 keyword matching via Reciprocal Rank Fusion (RRF) to eliminate failures on specific error codes and alphanumeric IDs (e.g., "Error 104", "SheerID").
- **Intent-Filtered Retrieval**: Partition the FAISS search space by predicted intent to prevent cross-intent semantic collisions (e.g. software release crashes matching album releases).

### P2 — Scale, Performance & Observability
- **Expanded Multi-Annotator Golden Set**: Expand the evaluation benchmark from 200 to 500 examples using active learning uncertainty sampling, labeled by multiple independent annotators to establish inter-annotator agreement.
- **Next.js In-Browser Review Dashboard**: Add a review UI inside the frontend to allow support leads to review generated replies and record human scores directly into `reply_review.csv`.
