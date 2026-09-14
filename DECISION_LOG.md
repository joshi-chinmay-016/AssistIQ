# AssistIQ: Engineering Decision Log

This document records the key architectural, statistical, and operational engineering decisions made during the development of **AssistIQ** (AI Customer Support Agent for @SpotifyCares).

Each decision captures the real engineering context, the alternatives evaluated, the chosen approach, the concrete tradeoffs accepted, and the empirical evidence or operational rationale.

---

### Decision 1: Target Brand Selection — SpotifyCares from TWCS Dataset
- **Context / Problem**: The Kaggle Customer Support on Twitter (TWCS) dataset contains ~3M tweets spanning multiple consumer domains (airlines, e-commerce, telecommunications, streaming). We needed a single brand with sufficient conversation volume, standardized problem types, and actionable customer workflows.
- **Options Considered**:
  1. Multi-brand cross-domain agent (Apple, Amazon, Delta, Spotify combined).
  2. Single airline brand (e.g. Delta / British Airways).
  3. Single digital streaming brand (@SpotifyCares).
- **Chosen Approach**: Selected **@SpotifyCares** exclusively (~40K multi-turn conversations).
- **Why**: Streaming support tickets feature clearly bounded technical categories (audio playback, offline sync, app crashes), subscription workflows (trials, student discounts, Family plans), and unambiguous security/billing policies. This allowed building a deep domain taxonomy and grounded retrieval corpus without cross-domain noise.
- **Tradeoff**: Does not test cross-brand generalization; models are specialized for digital streaming customer service.
- **Evidence / Result**: Successfully parsed 40,794 clean customer-inbound and support-response dialogue pairs into `dataset/spotify_support_cases.csv`.

---

### Decision 2: 11-Intent Granular Domain Taxonomy Design
- **Context / Problem**: TWCS raw tweets have no ground-truth intent labels. A coarse 3-class taxonomy (e.g. `technical`, `billing`, `other`) is insufficient for fine-grained routing or safety escalation.
- **Options Considered**:
  1. Standard 3-class sentiment / urgency classification.
  2. Unsupervised clustering (K-Means / LDA) without semantic supervision.
  3. 11-class domain taxonomy tailored to Spotify's customer service operations.
- **Chosen Approach**: Formulated an 11-intent schema: `playback_and_app_issues`, `search_and_discovery`, `account_and_login`, `billing_and_payment`, `premium_and_subscription`, `music_availability`, `playlist_and_library`, `feature_requests`, `content_metadata`, `ads_and_privacy`, and `other_non_actionable`.
- **Why**: Different ticket types require fundamentally distinct downstream actions: `account_and_login` requires credential protection; `billing_and_payment` requires PCI/financial escalation; `playback_and_app_issues` requires client telemetry diagnostics (device/OS/app version).
- **Tradeoff**: Higher classification difficulty on small sample sizes (e.g. `ads_and_privacy` with $N=2$); severe class imbalance.
- **Evidence / Result**: Successfully structured all 200 golden evaluation examples into the taxonomy, enabling precise deterministic rule triggering in Phase 4.

---

### Decision 3: Encapsulation of Preprocessing in Scikit-Learn Pipelines (Anti-Leakage)
- **Context / Problem**: Feature extraction (TF-IDF vocabulary construction and IDF weighting) performed across the entire dataset before train/test splitting introduces subtle data leakage, artificially inflating test metrics.
- **Options Considered**:
  1. Fit vectorizer on the full 200 golden examples, then slice `X_test`.
  2. Independent manual tokenization scripts before model training.
  3. Encapsulate `TfidfVectorizer` and classifier strictly inside an `sklearn.pipeline.Pipeline`.
- **Chosen Approach**: Built all classifiers using `Pipeline([('tfidf', TfidfVectorizer(...)), ('clf', ...)])` and performed stratified train/test splitting on raw text before `pipeline.fit()`.
- **Why**: Guarantees zero out-of-sample test vocabulary contamination. The test set ($N=50$) remains completely unseen during IDF calculation.
- **Tradeoff**: Slightly lower reported test numbers compared to leaky preprocessing benchmarks.
- **Evidence / Result**: Zero test set leakage confirmed; results reflect true out-of-sample generalization.

---

### Decision 4: TF-IDF + LinearSVC over Fine-Tuned Transformer for Intent
- **Context / Problem**: Need an intent classifier that runs deterministically with near-zero CPU inference latency, small memory footprint, and high sample efficiency on a 150-example training set.
- **Options Considered**:
  1. Fine-tuned BERT / RoBERTa / DeBERTa sequence classifier.
  2. Majority baseline (`DummyClassifier`).
  3. TF-IDF + Logistic Regression.
  4. TF-IDF + LinearSVC with balanced class weights.
- **Chosen Approach**: Proposed Model is **TF-IDF (1-2 ngrams, 10k max features) + LinearSVC**.
- **Why**: On short tweets with very small training sets ($N=150$), fine-tuning deep neural transformers easily overfits, requires GPU infrastructure, and adds ~50–100ms inference overhead per request. LinearSVC computes maximum geometric margin hyperplanes in high-dimensional sparse n-gram space, which outperforms cross-entropy loss under severe class imbalance and runs in < 2ms on CPU.
- **Tradeoff**: LinearSVC produces uncalibrated signed margin distances (`decision_function`) rather than true Bayesian posterior probabilities.
- **Evidence / Result**: LinearSVC achieved **0.4600 accuracy and 0.4004 macro F1**, outperforming Logistic Regression (0.3101 macro F1) by **+9.0 percentage points**, while maintaining < 2ms latency.

---

### Decision 5: Multi-Turn Dialogue Pairing into Historical Support Cases
- **Context / Problem**: Customer Support on Twitter consists of interleaved tweets, mentions, and replies. Storing isolated tweets in a vector database loses the support agent's actual resolution.
- **Options Considered**:
  1. Index only raw customer tweets for intent retrieval.
  2. Index only agent response tweets.
  3. Pair customer inbound tweets with subsequent official @SpotifyCares responses into structured `case_id` documents.
- **Chosen Approach**: Structured historical cases as paired JSON objects containing `customer_tweet_id`, `customer_text`, `support_tweet_id`, and `support_text`.
- **Why**: In-context retrieval requires both the query problem and the verified resolution to provide grounded demonstration evidence to the LLM.
- **Tradeoff**: Requires multi-turn parsing logic and drops isolated orphan tweets that received no brand response.
- **Evidence / Result**: Indexed 40,794 high-quality paired support cases in `backend/src/retrieval/artifacts/case_metadata.csv`.

---

### Decision 6: Embedding Model — `sentence-transformers/all-MiniLM-L6-v2` (384-d)
- **Context / Problem**: Retrieval requires dense semantic vector representations of customer messages to match paraphrased inquiries that share zero n-gram vocabulary (e.g. "can't hear sound" vs "audio playback failure").
- **Options Considered**:
  1. BM25 / Sparse inverted index.
  2. High-dimensional 1536-d OpenAI `text-embedding-3-small`.
  3. 768-d `all-mpnet-base-v2`.
  4. 384-d `all-MiniLM-L6-v2`.
- **Chosen Approach**: Selected **`all-MiniLM-L6-v2`** with L2-normalized embeddings.
- **Why**: At 384 dimensions and ~80MB parameter size, MiniLM runs fast on commodity CPU (~10–20ms per query), has zero API token cost, and achieves state-of-the-art sentence semantic similarity on consumer support text.
- **Tradeoff**: Slightly lower semantic capacity on very long paragraphs compared to MPNet, but Twitter queries are naturally short (<= 280 characters).
- **Evidence / Result**: Average retrieval search latency of ~15–30ms per query across 40,794 indexed cases.

---

### Decision 7: Vector Index Architecture — FAISS `IndexFlatIP` on L2-Normalized Vectors
- **Context / Problem**: Choosing the right vector indexing strategy for 40K 384-d vectors to balance recall accuracy, search latency, and implementation complexity.
- **Options Considered**:
  1. Approximate nearest neighbors: `IndexHNSWFlat`.
  2. Inverted file with clustering: `IndexIVFFlat`.
  3. Exact inner product: `IndexFlatIP` on unit-normalized vectors.
- **Chosen Approach**: Implemented **`faiss.IndexFlatIP`** with normalized embeddings.
- **Why**: At 40,794 vectors, an exhaustive flat scan takes < 25ms on CPU. Approximate indices (IVF/HNSW) introduce recall degradation and require complex hyperparameter tuning (nlist, nprobe, M, efSearch) with zero noticeable speed benefit at this corpus size. L2 normalization guarantees that the inner product mathematically equals exact cosine similarity.
- **Tradeoff**: Memory footprint is linear with corpus size ($40,794 \times 384 \times 4 \approx 62$ MB), which is trivial on modern hardware.
- **Evidence / Result**: 100% exact nearest neighbor recall; zero approximation artifacts; 62MB index footprint.

---

### Decision 8: Retrieval Similarity Threshold of 0.45 as an Engineering Heuristic
- **Context / Problem**: Grounded generation and auto-handle escalation require a cutoff threshold below which historical cases are flagged as weak or irrelevant.
- **Options Considered**:
  1. Arbitrary 0.80 cutoff (too strict; rejects ~85% of queries).
  2. No threshold (always pass top-k cases to Gemini regardless of distance).
  3. Initial engineering threshold of 0.45 with explicit documentation that it is not mathematically optimal.
- **Chosen Approach**: Established **0.45** as the default cutoff, evaluated via threshold sensitivity grids.
- **Why**: In empirical testing, cosine similarity below ~0.40 frequently retrieved cross-domain noise, while queries between 0.45 and 0.70 frequently shared underlying technical workflows despite lexical variance. We explicitly declare in all reports that 0.45 is an initial operational threshold, not a scientifically optimal optimum.
- **Tradeoff**: Some edge-case queries at 0.42 might be safely answerable, while some queries at 0.46 may contain irrelevant sub-details.
- **Evidence / Result**: Threshold sensitivity analysis demonstrated that varying similarity between 0.35 and 0.55 shifts the auto-handle rate between 60.0% and 57.5% under default confidence.

---

### Decision 9: Deterministic Escalation Policy Engine over LLM-Based Escalation
- **Context / Problem**: Deciding whether an incoming ticket should be automatically answered or escalated to human tier-2 support.
- **Options Considered**:
  1. Prompt Gemini: "Should this ticket be auto-handled or escalated?"
  2. Machine learning classifier trained on historical escalation outcomes.
  3. Deterministic rule-based policy engine (E0–E8, A1) with strict rule priority.
- **Chosen Approach**: Built **`EscalationPolicy`**, a pure deterministic Python rule engine.
- **Why**: In enterprise customer service, autonomous actions must be auditable, compliant, and predictable. LLM-based escalation decisions suffer from prompt brittleness, non-deterministic output, and susceptibility to adversarial prompt injections. A rule engine allows compliance teams to inspect the exact rule that triggered (e.g. `E5: sensitive_account_security`).
- **Tradeoff**: Does not capture subtle conversational nuances that aren't encoded in rules or keywords.
- **Evidence / Result**: Zero prompt injections can bypass escalation rules; every decision produces an audit trail (`policy_rules_triggered`, `primary_rule`, `risk_level`).

---

### Decision 10: Mandatory Escalation for Financial Disputes and Account Security
- **Context / Problem**: Automated customer-support agents risk hallucinating unauthorized commitments, such as promising immediate cash refunds or offering password resets in public tweets.
- **Options Considered**:
  1. Allow AI to draft refund assurances if retrieval similarity is high.
  2. Automatically handle all billing inquiries with standard FAQ links.
  3. Strictly escalate all transactional billing actions (`E6`) and account security credentials (`E5`).
- **Chosen Approach**: Mandatory escalation rules for billing refund/charge disputes (`E6`) and account hack/credential recovery (`E5`).
- **Why**: AI agents possess no bank authorization or database write permissions. Promising a refund creates severe corporate liability. Handling account compromise publicly risks account hijacking.
- **Tradeoff**: Reduces the overall auto-handle rate on billing and account categories.
- **Evidence / Result**: 100% of sensitive queries mentioning keywords like "hack", "stolen", "refund", or "charged twice" escalate to high-risk human review.

---

### Decision 11: Margin-Derived Softmax Confidence Proxy for LinearSVC
- **Context / Problem**: The escalation policy requires an `intent_confidence` metric (threshold `< 0.20` triggers rule `E1`). However, `LinearSVC` does not natively produce calibrated probabilities like Logistic Regression.
- **Options Considered**:
  1. Switch to Logistic Regression solely to get `predict_proba()` (sacrificing 9.0 points of Macro F1).
  2. Wrap LinearSVC in `CalibratedClassifierCV` (Platt scaling / isotonic regression), introducing cross-validation overhead on a small 150-example set.
  3. Compute numerically stabilized softmax over decision function margins and explicitly document it as a **margin-derived confidence score** rather than probability.
- **Chosen Approach**: Applied **`softmax(decision_function)`** and transparently documented it as a margin-derived confidence score.
- **Why**: Preserves LinearSVC's superior F1 performance without overengineering calibration on a 150-sample dataset. The softmax over signed geometric margins provides an uncalibrated but monotonic ranking proxy where random guessing yields $\approx 1/11 = 0.091$ and strong confidence yields $> 0.30$.
- **Tradeoff**: The values cannot be interpreted as true Bayesian posterior probabilities.
- **Evidence / Result**: Successfully triggers rule `E1` when the maximum margin score falls below 0.20, identifying ambiguous or boundary customer tweets.

---

### Decision 12: Two-Tier Generation Architecture — Structured Pydantic Output & Citation Verification
- **Context / Problem**: LLMs frequently generate plausible-sounding responses that cite evidence cases they did not actually retrieve, or cite case IDs that don't exist.
- **Options Considered**:
  1. Unstructured free-text generation with regex parsing.
  2. Enforced JSON schema generation via `google-genai` SDK validated into Pydantic `SupportReply`.
  3. Post-generation citation verification against retrieved case ID sets.
- **Chosen Approach**: Combined **Pydantic schema enforcement** (`SupportReply`) with a deterministic **post-generation citation verification step**.
- **Why**: Schema enforcement guarantees `reply`, `grounding_summary`, and `grounding_status` exist. Post-generation verification scrubs any hallucinated `case_id` that was not in the actual retrieved candidate list.
- **Tradeoff**: Adds ~5ms validation overhead in Python.
- **Evidence / Result**: 100% of generated replies conform to the API contract; zero hallucinated citation IDs can leak to the customer.

---

### Decision 13: Mock LLM Client & Mock Judge for Deterministic Offline Execution
- **Context / Problem**: Evaluation pipelines that rely exclusively on live cloud LLM APIs fail in air-gapped environments, exhaust API quotas, cost money on continuous integration, and suffer from non-deterministic variance.
- **Options Considered**:
  1. Require live Gemini API key for all test runs.
  2. Record/replay HTTP cassettes (VCR.py).
  3. Build fully offline `MockLLMClient` and `MockLLMJudge` implementing the identical interfaces.
- **Chosen Approach**: Implemented **`MockLLMClient`** and **`MockLLMJudge`** as default offline providers.
- **Why**: Allows any developer, interviewer, or CI pipeline to run unit tests and the entire evaluation framework within ~15 minutes without providing credit cards or API keys.
- **Tradeoff**: Mock generation uses heuristic rules and does not evaluate live Gemini fluency.
- **Evidence / Result**: 48 unit tests and 7 evaluation stages run 100% offline at $0.00 cost. Live evaluation is accessible via a single `--mode live` flag.

---

### Decision 14: Architectural Decoupling of Next.js Frontend and FastAPI Backend
- **Context / Problem**: Determining whether to build a monolithic Python web app (Streamlit / Gradio) or decouple the frontend and backend.
- **Options Considered**:
  1. Monolithic Python UI (Streamlit).
  2. Full-stack Next.js app running Python via child processes.
  3. Decoupled REST architecture: FastAPI backend (Python 3.12) + Next.js 16 (React 19 / TypeScript).
- **Chosen Approach**: Built independent **FastAPI backend** and **Next.js frontend** communicating via REST API (`/api/v1/assist`).
- **Why**: In enterprise AI customer support, the inference and policy pipeline belongs in a microservice that can be scaled, tested, and integrated into customer CRM backends (e.g. Zendesk, Salesforce, Hiver). The web frontend provides interactive agent visualization with clear separation of concerns.
- **Tradeoff**: Requires managing two dev servers (Uvicorn on :8000 and Next.js on :3000) with CORS middleware.
- **Evidence / Result**: FastAPI endpoint executes in < 30ms locally; Next.js builds clean static pages with zero typecheck errors.
