# AssistIQ: Comprehensive Evaluation Report
**AI Customer Support Agent for @SpotifyCares (Twitter Customer Support Dataset)**

---

## 1. Problem Framing & System Objective
Customer support on public social channels such as Twitter (@SpotifyCares) presents severe challenges for automated AI agents:
1. **High Inbound Velocity & Noise**: Customer messages are ultra-short (<= 280 characters), ungrammatical, lack system context (device, OS, app version), or consist of emotional venting.
2. **Class Imbalance**: Common queries (playback bugs, billing questions, greetings) heavily outnumber niche issues (regional licensing, lyrics metadata typos).
3. **Severe Safety & Brand Liability**: An AI support agent that hallucinates refund commitments, offers unauthorized compensation, or leaks private credentials in public tweets creates legal and brand liabilities.
4. **Account Action Limitations**: Tier-1 automated agents have zero database write permissions; tasks requiring billing refunds or account credential changes must be escalated to secure human workflows.

**AssistIQ's Objective**: Build a high-throughput, low-latency AI support pipeline that provides grounded, accurate replies for common issues, while deterministically escalating high-risk, ambiguous, or ungrounded tickets to human agents.

---

## 2. Dataset & SpotifyCares Selection
From the Kaggle Customer Support on Twitter (TWCS) dataset (~3 million tweets), **@SpotifyCares** was selected as the target enterprise brand:
- **Corpus Size**: 40,794 historical customer-support dialogue pairs indexed.
- **Why SpotifyCares**: Digital audio streaming features well-bounded, repeatable technical troubleshooting categories (offline sync, Bluetooth/CarPlay, cache clearing, app crashes), structured subscription tiers (Family, Student, Premium trials), and distinct security workflows (hacked accounts, password resets).
- **Golden Evaluation Set (`dataset/golden_set.csv`)**: 200 representative Spotify customer messages.
- **Transparency on Labeling Provenance**:
  - The 200 golden examples were labeled using an **AI-assisted candidate generation pipeline with partial human spot-checking/review**.
  - Specifically, 14 candidate label disagreements were manually inspected and corrected in `golden_set_candidate_review.csv`.
  - **Honesty Disclosure**: AssistIQ does NOT claim all 200 examples were 100% hand-annotated from scratch by human domain experts. Evaluation metrics must be interpreted with this semi-automated ground truth in mind.

---

## 3. Intent Taxonomy (11 Classes)
The system uses an 11-intent taxonomy tailored to the Spotify customer-support domain:

| Intent Class | Definition | Distribution (N=200) |
| :--- | :--- | :---: |
| `playback_and_app_issues` | Audio glitches, freezing, crashing, Bluetooth/CarPlay, offline playback | 31 (15.5%) |
| `search_and_discovery` | Search bar failures, recommendations, Discover Weekly algorithm | 3 (1.5%) |
| `account_and_login` | Password resets, hacked accounts, email changes, login locks | 15 (7.5%) |
| `billing_and_payment` | Charges, double renewals, payment methods (cards, PayPal, gift cards) | 19 (9.5%) |
| `premium_and_subscription` | Upgrades, Student verification (SheerID), Family plan address mismatch | 22 (11.0%) |
| `music_availability` | Missing tracks/albums, regional licensing restrictions, greyed-out songs | 12 (6.0%) |
| `playlist_and_library` | Disappeared playlists, wiped offline downloads, library editing | 11 (5.5%) |
| `feature_requests` | Feature suggestions (sleep timer, clean lyrics toggle, light mode) | 24 (12.0%) |
| `content_metadata` | Wrong audio version, incorrect artist credit, artwork/lyrics typos | 8 (4.0%) |
| `ads_and_privacy` | Unwanted popup ads on Premium, privacy tracking, data policies | 2 (1.0%) |
| `other_non_actionable` | Thanks, acknowledgements, greetings, vague venting without request | 53 (26.5%) |

---

## 4. Pipeline Architecture
AssistIQ operates as a strictly decoupled 4-phase inference pipeline:

```
Customer Tweet
      ↓
[Phase 1] Intent Classification (TF-IDF + LinearSVC, < 2ms)
      ↓
[Phase 2] Historical Support Retrieval (MiniLM-L6-v2 + FAISS IndexFlatIP, ~20ms)
      ↓
[Phase 3] Grounded Generation (Gemini 2.5 Flash / MockLLM, Pydantic SupportReply)
      ↓
[Phase 4] Deterministic Escalation Policy (Rules E0–E8, A1)
      ↓
Decision Output: [AUTO_HANDLE] vs [ESCALATE]
```

**Key Architectural Principle**: Gemini generates candidate text, but **NEVER makes the final escalation decision**. The final customer-handling decision is governed by an auditable, deterministic rule engine.

---

## 5. Intent Baselines & Empirical Results

### Baselines & Models
1. **Baseline 1: Majority Classifier (`DummyClassifier`)**: Always predicts `other_non_actionable` ($N=40$ in training set). Establishes the floor for severe class imbalance.
2. **Baseline 2: TF-IDF + Logistic Regression**: Standard linear probabilistic baseline with balanced class weighting.
3. **Proposed Model: TF-IDF + LinearSVC**: Support vector machine maximizing geometric separation margins in sparse n-gram space.

### Evaluation Setup
- Isolated 75% train ($N=150$) / 25% test ($N=50$) stratified split (`random_state=2026`).
- Preprocessing strictly encapsulated inside scikit-learn Pipeline to guarantee zero data leakage.

### Benchmark Results (Isolated Test Set, N=50)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 0.2600 | 0.0260 | 0.1000 | 0.0413 | 0.1073 |
| **TF-IDF + Logistic Regression** | 0.4400 | 0.3024 | 0.3276 | 0.3101 | 0.4187 |
| **Proposed Model (TF-IDF + LinearSVC)** | **0.4600** | **0.4241** | **0.3942** | **0.4004** | **0.4534** |

### Why Proposed Model Wins:
- **Macro F1 Gain**: LinearSVC achieves **0.4004 Macro F1**, outperforming Logistic Regression (0.3101) by **+9.0 percentage points**.
- **Minority Sensitivity**: In high-dimensional sparse n-gram space, the max-margin hyperplane separates minority clusters far more effectively than cross-entropy loss. It recovered minority classes where Logistic Regression scored 0.00 F1 (`music_availability`: 0.40 vs 0.00; `content_metadata`: 0.50 vs 0.00).

### Clarification on Intent Confidence:
`LinearSVC` outputs signed margin distances (`decision_function`), not calibrated Bayesian probabilities. AssistIQ applies a numerically stabilized softmax over margins to produce a **margin-derived confidence proxy** (where 1/11 = 0.091 represents baseline uncertainty). We explicitly treat this score as an uncalibrated margin metric rather than a true probability.

---

## 6. Historical Retrieval Evaluation
Retrieval was evaluated independently from generation over all 200 golden queries against the 40,794-case FAISS index (`all-MiniLM-L6-v2`, 384 dimensions, normalized cosine similarity).

### Anti-Leakage Verification
- Confirmed: **0 golden set tweets exist in the 40,794 historical index corpus** (zero data contamination).

### Automated Retrieval Metrics (N=200 Queries)
- **Intent Consistency@1**: 0.5800
- **Intent Consistency@3**: 0.5683
- **Intent Consistency@5**: 0.5730
- **Intent Hit Rate@1**: 0.5800
- **Intent Hit Rate@3**: 0.8050
- **Intent Hit Rate@5**: 0.8850 (Headline Metric)
- **Top-1 Cosine Similarity**: Mean = 0.7298, Median = 0.7304, Range = [0.4024, 1.0000]
- **Retrieval Latency**: Mean = 24.12 ms, Median = 21.05 ms, P95 = 38.40 ms (commodity CPU)

### Human Relevance Review (N=35 Queries, 105 Top-3 Cases)
To cross-validate automated metrics against true semantic utility, 35 diverse queries were manually graded:
- **Strict Precision@1**: **0.7143** (25/35 top-1 cases offered the direct, correct resolution).
- **Strict Precision@3**: **0.6381** (67/105 cases offered the direct, correct resolution).
- **Lenient Precision@1**: **1.0000** (35/35 top-1 cases were topically relevant or partially relevant).
- **Lenient Precision@3**: **1.0000** (105/105 cases were topically relevant or partially relevant).
- **Manual Hit Rate@3**: **1.0000** (100% of queries had at least one relevant case in top-3).

---

## 7. Reply Quality Evaluation (LLM Judge)
Rather than asking a vague question ("Is this reply good?"), an independent LLM Judge (`evaluation/llm_judge.py`) evaluated generated replies across five explicit dimensions on a 1.0 to 5.0 scale:

1. **Relevance (1–5)**: Direct alignment with customer's stated issue.
2. **Groundedness (1–5)**: Claims strictly derived from retrieved evidence; severe penalty for hallucinations.
3. **Helpfulness (1–5)**: Actionable, step-by-step guidance or essential diagnostic requests.
4. **Completeness (1–5)**: Covers all facets of the query.
5. **Safety (1–5)**: Brand tone, credential security, correct escalation of sensitive disputes.

### Quality Gate Pass Criterion:
`overall_score >= 3.5` AND `groundedness >= 3.0` AND `safety >= 3.0` AND `hallucination_flag == False`.

### Evaluation Results (30 Representative Queries):
- **Mean Relevance Score**: 4.10 / 5.0
- **Mean Groundedness Score**: 4.45 / 5.0
- **Mean Helpfulness Score**: 3.95 / 5.0
- **Mean Completeness Score**: 4.25 / 5.0
- **Mean Safety Score**: 4.95 / 5.0
- **Mean Overall Score**: **4.34 / 5.0**
- **Quality Gate Pass Rate**: **93.3%**
- **Hallucination Flags Detected**: 0 / 30

---

## 8. Escalation & Safety Policy Evaluation
The deterministic escalation policy (`EscalationPolicy`) enforces safety rules in strict priority:
- Priority 1: `E0` (Empty/invalid message)
- Priority 2: `E3` (Generation failure)
- Priority 3: `E4` (Insufficient grounding)
- Priority 4: `E2` (Weak retrieval evidence, sim < 0.45)
- Priority 5: `E1` (Low intent confidence, margin < 0.20)
- Priority 6: `E5` (Sensitive account security/hack credentials)
- Priority 7: `E6` (Billing refunds / double charge financial actions)
- Priority 8: `E7_AMBIGUOUS` (Unclear venting)
- Priority 9: Auto-handle rules (`E7_GREETING`, `E8_FEATURE_REQUEST`, `A1`)

### Results on Representative Evaluation Set (N=40):
- **Auto-Handle Rate**: **60.0%** (24 / 40 queries)
- **Escalation Rate**: **40.0%** (16 / 40 queries)
- **Rule Distribution**: Rule A1 (52.5%), Rule E7_AMBIGUOUS (25.0%), Rule E1 (10.0%), Rule E8 (7.5%), Rule E5 (5.0%).

### 2D Threshold Sensitivity Grid (Auto-Handle Rate):

| `min_intent_confidence` | Sim >= 0.35 | Sim >= 0.40 | Sim >= 0.45 (Default) | Sim >= 0.50 | Sim >= 0.55 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.15** | 62.5% | 62.5% | 62.5% | 62.5% | 60.0% |
| **0.20 (Default)** | 60.0% | 60.0% | **60.0%** | 60.0% | 57.5% |
| **0.25** | 47.5% | 47.5% | 47.5% | 47.5% | 45.0% |
| **0.30** | 32.5% | 32.5% | 32.5% | 32.5% | 30.0% |

### Core Safety Tradeoff:
**Unsafe auto-handling carries vastly higher enterprise cost than unnecessary escalation.**
If the agent auto-handles a hacked account ticket with generic password-reset advice, an attacker can hijack the account. If the agent unnecessarily escalates a simple FAQ to a human agent, the only cost is 2 minutes of agent time. AssistIQ is deliberately tuned with safety-first thresholds.

---

## 9. Automated Failure Analysis
Pipeline errors are automatically diagnosed and sorted into 10 explicit failure categories (`evaluation/results/failure_examples.csv`):

1. **`wrong_intent`**: Tweet 2446506 ("any update to previous tweet?") was classified as `other_non_actionable` due to missing audio keywords, yielding a generic greeting instead of technical troubleshooting.
2. **`low_retrieval_similarity`**: Microwave custom firmware query yielded top similarity 0.3214 (< 0.45 threshold), triggering rule E2 escalation.
3. **`irrelevant_retrieved_evidence`**: Album release inquiry matched a desktop crash ticket on the word "release", causing the model to ask irrelevant OS questions.
4. **`unsupported_generation`**: Offline library query answered without verified citations in prompt.
5. **`hallucinated_claim`**: Agent claiming to have issued a $9.99 refund when it has no database write access.
6. **`missing_required_clarification`**: Customer complaining of "warp speed audio" without stating OS; agent suggested iPhone restart without checking device.
7. **`unsafe_auto_handle`**: Specific username login failure auto-handled instead of escalating to private 1:1 verification.
8. **`unnecessary_escalation`**: Grounded FAQ about marking podcast episodes played escalated due to over-conservative confidence cutoff.
9. **`low_quality_response`**: Boilerplate response ignoring customer's specific question about explicit lyrics toggles.
10. **`ambiguous_customer_message`**: Vague emotional complaint ("why does app do this every single time") correctly caught and escalated under rule E7_AMBIGUOUS.

---

## 10. Misleading Headline Number Deep Dive

> ### Observed Metric: **"88.5% Retrieval Intent Hit Rate@5"**

### Why It Sounds Impressive:
In executive summaries or marketing presentations, an **88.5% Hit Rate@5** suggests that historical retrieval succeeds almost 9 out of 10 times.

### Why It Is Misleading Without Context:
1. **Surrogate Heuristic, Not True Relevance**: Intent Hit Rate@5 merely measures whether at least ONE historical customer query in the top 5 was assigned the same predicted intent label by the classifier.
2. **Topical Match != Useful Solution**: A ticket complaining of "desktop app crashes on Windows 10" and a ticket complaining of "music pauses on iPhone screen lock" both share the intent label `playback_and_app_issues`. A match is recorded as a "Hit", but the iPhone troubleshooting steps in that retrieved case are useless for solving the Windows desktop crash.
3. **Ground Truth Comparison**: In human relevance review over 35 queries (105 cases), **Strict Precision@3 was only 63.81%**. That means over **36% of retrieved cases in the top 3 were NOT direct solutions** for the customer's specific problem.
4. **Conclusion**: Relying on Intent Hit Rate@5 as proof of retrieval quality is dangerous; true utility must be measured via strict answer groundedness and manual human review.

---

## 11. Key Engineering Decisions Summary
1. **SpotifyCares TWCS Selection**: Isolated domain noise to build deep, high-precision retrieval over 40,794 cases.
2. **TF-IDF + LinearSVC**: Fast (< 2ms), CPU-friendly, sample-efficient max-margin classifier beating Logistic Regression by +9.0 Macro F1 points.
3. **FAISS IndexFlatIP (Exact Cosine)**: Exhaustive inner product search over 384-d normalized embeddings avoiding ANN recall degradation.
4. **Deterministic Policy Engine**: Guarantees zero prompt injections can bypass escalation; full auditability for compliance.
5. **Zero-Cost Offline Evaluation**: Mock clients allow full test and benchmark reproduction in < 15 minutes without cloud credentials.

---

## 12. Limitations & Enterprise Risks
- **Golden Set Size & Provenance**: The 200 golden set queries were generated with AI assistance and partially human-reviewed (14 corrections). Real enterprise deployment requires a multi-annotator blind golden set with inter-annotator agreement (Fleiss' Kappa).
- **Static Retrieval Corpus**: The FAISS index is static; new Spotify product features (e.g. AI DJ, Daylist, Jam sessions) cannot be retrieved without re-indexing updated support dialogues.
- **Uncalibrated Margin Scores**: LinearSVC confidence is a margin proxy, not a posterior probability. Platt scaling or temperature scaling should be evaluated if true risk probabilities are required.

---

## 13. Next-Week Improvement Plan
If granted an additional week to iterate on AssistIQ, the engineering focus would be:

1. **Active Learning Golden Set Expansion**:
   - Expand the golden evaluation set from 200 to 500 examples using uncertainty sampling on LinearSVC margin boundaries (< 0.20 confidence).
   - Implement multi-annotator blind labeling to measure human inter-annotator agreement.
2. **Dense + Sparse Hybrid Retrieval**:
   - Combine FAISS dense embeddings with BM25 keyword matching via Reciprocal Rank Fusion (RRF).
   - Solves lexical mismatch on specific error codes (e.g. "Error 104", "CSRF failure", "SheerID") that dense embeddings sometimes dilute.
3. **Cross-Encoder Reranker**:
   - Add a lightweight cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-rank the Top-10 retrieved cases down to Top-3.
   - Expected to raise Strict Precision@3 from 63.8% to > 80%.
4. **Dynamic Context-Aware Escalation**:
   - Allow customer sentiment and repeat tweet frequency (e.g. customer tweeting 3 times in 1 hour) to dynamically escalate tickets regardless of intent confidence.
5. **Automated Human-Agreement Dashboard**:
   - Build an in-browser annotation review UI in the Next.js frontend to allow tier-2 support leads to grade generated replies directly, updating `human_llm_agreement.csv` in real time.
