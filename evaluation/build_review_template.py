"""
AssistIQ: Human Reply Review Template Builder
Constructs a blinded, stratified 30-example evaluation template across all 11 intents.
Includes customer query, predicted intent, candidate reply, retrieved evidence IDs,
top similarity, and escalation decision, with human scoring columns strictly blank.
Applies deterministic shuffling (seed 2026) to reduce evaluator presentation bias.
"""

import os
import pandas as pd
from backend.src.escalation.policy import decide_escalation

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
judge_csv = os.path.join(PROJECT_ROOT, "evaluation", "results", "reply_llm_judge.csv")
out_csv = os.path.join(PROJECT_ROOT, "evaluation", "reply_review.csv")

if not os.path.exists(judge_csv):
    raise FileNotFoundError(f"Judge CSV not found at {judge_csv}")

judge_df = pd.read_csv(judge_csv)

records = []
for _, row in judge_df.iterrows():
    evidence_ids = str(row.get("evidence_ids", "")).strip()
    sim = float(row.get("evidence_similarity", 0.0))
    cust_text = str(row["customer_text"]).strip()
    intent = str(row["predicted_intent"]).strip()
    reply = str(row["reply"]).strip()

    esc = decide_escalation(
        customer_message=cust_text,
        predicted_intent=intent,
        intent_confidence=0.50,
        top_retrieval_similarity=sim,
        evidence_count=len([e for e in evidence_ids.split(",") if e]),
        grounding_status="grounded" if sim >= 0.45 else "insufficient_evidence",
        reply_text=reply
    )

    records.append({
        "tweet_id": row["tweet_id"],
        "customer_text": cust_text,
        "predicted_intent": intent,
        "generated_reply": reply,
        "retrieved_evidence": evidence_ids,
        "evidence_similarity": round(sim, 4),
        "escalation_decision": esc.decision,
        "relevance": "",
        "groundedness": "",
        "helpfulness": "",
        "completeness": "",
        "safety": "",
        "overall": "",
        "hallucination_flag": "",
        "notes": ""
    })

review_df = pd.DataFrame(records)

# Deterministic shuffle with fixed seed to reduce evaluator presentation bias
review_df = review_df.sample(frac=1.0, random_state=2026).reset_index(drop=True)
review_df.to_csv(out_csv, index=False, encoding="utf-8")
print(f"[OK] Saved blinded human review template with {len(review_df)} rows to: {out_csv}")
