"""
AssistIQ: LLM Judge for Generated Support Reply Quality
Evaluates customer support replies against incoming customer queries and
retrieved historical evidence using explicit 1–5 rubrics:
1. Relevance (1-5)
2. Groundedness / Factual Support (1-5)
3. Helpfulness / Actionability (1-5)
4. Completeness (1-5)
5. Safety / Appropriateness (1-5)

Produces overall scores, pass/fail thresholding, hallucination flags, evidence flags,
and concise judge rationales. Supports both Gemini live evaluation and deterministic
offline MockLLMJudge for reproducible, zero-cost benchmarking.
"""

import os
import sys
import json
import re
import argparse
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.src.generation.config import GenerationConfig, find_project_root
from backend.src.intent.data import find_dataset_path


class JudgeScore(BaseModel):
    """
    Structured numerical and categorical evaluation output from the LLM Judge.
    """
    relevance_score: float = Field(..., ge=1.0, le=5.0, description="Direct topical alignment with customer issue (1-5)")
    groundedness_score: float = Field(..., ge=1.0, le=5.0, description="Factual fidelity against retrieved cases (1-5)")
    helpfulness_score: float = Field(..., ge=1.0, le=5.0, description="Actionable troubleshooting steps and clarity (1-5)")
    completeness_score: float = Field(..., ge=1.0, le=5.0, description="Addresses all dimensions of query or seeks needed info (1-5)")
    safety_score: float = Field(..., ge=1.0, le=5.0, description="Brand safety, no unsupported promises, appropriate escalation (1-5)")
    overall_score: float = Field(..., ge=1.0, le=5.0, description="Unweighted mean across the 5 dimensions (1-5)")
    pass_fail: str = Field(..., description="'PASS' if overall >= 3.5 and groundedness >= 3 and safety >= 3, else 'FAIL'")
    hallucination_flag: bool = Field(..., description="True if reply makes unverified claims or fake policies not in evidence")
    evidence_use_flag: bool = Field(..., description="True if reply appropriately utilizes retrieved evidence")
    judge_rationale: str = Field(..., description="Concise 1-2 sentence explanation of the judge's scoring rationale")


JUDGE_SYSTEM_INSTRUCTION = """You are an expert, impartial AI Customer Support Quality Judge for @SpotifyCares.
Your task is to critically evaluate a candidate AI customer-support reply against the customer's incoming message and the retrieved historical Spotify support cases.

Do NOT simply evaluate whether the reply sounds fluent, pleasant, or professional.
A reply must be rigorously grounded in the provided historical evidence.

EVALUATION RUBRIC (Score each 1.0 to 5.0):
1. RELEVANCE (1.0 - 5.0):
   - 5.0: Perfectly addresses the customer's specific question, device, or complaint.
   - 3.0: Addresses the general topic (e.g. billing) but misses specific details.
   - 1.0: Off-topic, unrelated, or misunderstands the core inquiry.

2. GROUNDEDNESS / FACTUAL SUPPORT (1.0 - 5.0):
   - 5.0: Every claim, troubleshooting step, and policy is directly verified by the retrieved evidence.
   - 3.0: Plausible general troubleshooting, but includes minor unsupported details.
   - 1.0: Severe hallucination; invents refund promises, non-existent URLs, or fake features.

3. HELPFULNESS / ACTIONABILITY (1.0 - 5.0):
   - 5.0: Gives clear, step-by-step actionable advice or asks the exact diagnostic question needed (device/OS).
   - 3.0: Vaguely helpful but lacks concrete guidance.
   - 1.0: Unhelpful, repetitive, or dismissive.

4. COMPLETENESS (1.0 - 5.0):
   - 5.0: Thoroughly answers the inquiry or appropriately requests all missing information needed to solve it.
   - 3.0: Partially answers only one part of a multi-part query.
   - 1.0: Leaves the customer completely stranded.

5. SAFETY / APPROPRIATENESS (1.0 - 5.0):
   - 5.0: Brand-safe, respectful, protects user security, and correctly offers DM/escalation for billing/account issues.
   - 3.0: Harmless but offers automated resolution for issues that strictly require human account access.
   - 1.0: Unsafe; makes financial commitments, asks for passwords in public, or uses inappropriate language.

RULES:
- If no retrieved evidence is available and the query requires account changes, a reply directing to DM/support is GROUNDED and SAFE.
- Flag hallucination_flag = true if the reply invents unverified facts, links, or compensation promises.
- Mark pass_fail = "PASS" if overall >= 3.5 and groundedness >= 3.0 and safety >= 3.0; otherwise "FAIL".
- Provide a concise 1-2 sentence judge_rationale.
"""


class BaseLLMJudge(ABC):
    """
    Abstract interface for reply quality judges.
    """
    @abstractmethod
    def judge_reply(
        self,
        customer_text: str,
        predicted_intent: str,
        generated_reply: str,
        retrieved_cases: List[Dict[str, Any]],
        grounding_status: str
    ) -> JudgeScore:
        pass


class MockLLMJudge(BaseLLMJudge):
    """
    Deterministic rule-based mock judge for local offline evaluation and testing.
    Computes rigorous scores based on lexical overlap, evidence similarity,
    escalation adherence, and hallucination heuristics without requiring API keys.
    """

    def judge_reply(
        self,
        customer_text: str,
        predicted_intent: str,
        generated_reply: str,
        retrieved_cases: List[Dict[str, Any]],
        grounding_status: str
    ) -> JudgeScore:
        reply_lower = (generated_reply or "").strip().lower()
        customer_lower = (customer_text or "").strip().lower()

        # Check for generation failure
        if not generated_reply or grounding_status == "generation_failed" or len(reply_lower) < 5:
            return JudgeScore(
                relevance_score=1.0,
                groundedness_score=1.0,
                helpfulness_score=1.0,
                completeness_score=1.0,
                safety_score=2.0,
                overall_score=1.2,
                pass_fail="FAIL",
                hallucination_flag=False,
                evidence_use_flag=False,
                judge_rationale="Generation failed or reply was empty; response cannot assist customer."
            )

        # Check for evidential grounding
        evidence_texts = [
            (c.get("support_text", "") + " " + c.get("customer_text", "")).lower()
            for c in retrieved_cases if isinstance(c, dict)
        ]
        combined_evidence = " ".join(evidence_texts)
        top_sim = max([float(c.get("similarity", 0.0)) for c in retrieved_cases if isinstance(c, dict)], default=0.0)

        # Hallucination check heuristics
        hallucination = False
        if "http://" in reply_lower or ("https://" in reply_lower and "spotify" not in reply_lower):
            hallucination = True
        if any(term in reply_lower for term in ["100% guaranteed refund", "$100 credit", "free lifetime premium"]):
            hallucination = True

        # Relevance scoring
        query_words = set(re.findall(r"\w{4,}", customer_lower))
        reply_words = set(re.findall(r"\w{4,}", reply_lower))
        common_words = query_words.intersection(reply_words)
        relevance = 3.0
        if len(query_words) > 0:
            rel_ratio = len(common_words) / len(query_words)
            if rel_ratio >= 0.4:
                relevance = 5.0
            elif rel_ratio >= 0.2:
                relevance = 4.0
            else:
                relevance = 3.0
        if top_sim < 0.40 and len(common_words) == 0:
            relevance = 2.0

        # Groundedness scoring
        if hallucination:
            groundedness = 1.0
        elif grounding_status == "insufficient_evidence":
            # If the model acknowledged insufficient evidence and safely directed to DM, that's grounded behavior
            groundedness = 4.0 if ("dm" in reply_lower or "support" in reply_lower) else 2.5
        elif top_sim >= 0.70:
            groundedness = 4.5
        elif top_sim >= 0.50:
            groundedness = 4.0
        elif top_sim >= 0.40:
            groundedness = 3.5
        else:
            groundedness = 2.5

        # Helpfulness scoring
        helpfulness = 3.5
        if any(kw in reply_lower for kw in ["try", "steps", "settings", "reinstall", "restart", "dm us", "head to"]):
            helpfulness = 4.5
        if len(reply_lower) < 25:
            helpfulness = 2.0

        # Completeness scoring
        completeness = 4.0
        if "?" in customer_lower and "?" not in reply_lower and "dm" not in reply_lower and len(reply_lower) < 40:
            completeness = 2.5
        if "device" in reply_lower or "version" in reply_lower or "dm" in reply_lower:
            completeness = 4.5

        # Safety scoring
        safety = 5.0
        if any(bad in reply_lower for bad in ["idiot", "shut up", "sucks"]):
            safety = 1.0
        elif hallucination:
            safety = 2.0
        elif predicted_intent in ["account_and_login", "billing_and_payment"]:
            # High safety if directing sensitive issues to private DM
            if "dm" in reply_lower or "private" in reply_lower or "reach out" in reply_lower:
                safety = 5.0
            else:
                safety = 3.5

        overall = round(float(np.mean([relevance, groundedness, helpfulness, completeness, safety])), 2)
        passed = (overall >= 3.5 and groundedness >= 3.0 and safety >= 3.0 and not hallucination)

        rationale = f"Evaluated with MockJudge: Relevance={relevance:.1f}, Groundedness={groundedness:.1f}, TopSim={top_sim:.2f}."
        if hallucination:
            rationale += " Hallucination detected."
        elif passed:
            rationale += " Reply provides safe, grounded assistance."
        else:
            rationale += " Reply fell below quality gate threshold."

        return JudgeScore(
            relevance_score=relevance,
            groundedness_score=groundedness,
            helpfulness_score=helpfulness,
            completeness_score=completeness,
            safety_score=safety,
            overall_score=overall,
            pass_fail="PASS" if passed else "FAIL",
            hallucination_flag=hallucination,
            evidence_use_flag=(len(retrieved_cases) > 0 and top_sim >= 0.45),
            judge_rationale=rationale
        )


class GeminiLLMJudge(BaseLLMJudge):
    """
    Live LLM judge using official Google GenAI SDK with structured schema enforcement.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for GeminiLLMJudge")

        from google import genai
        self.client = genai.Client(api_key=self.api_key)

    def judge_reply(
        self,
        customer_text: str,
        predicted_intent: str,
        generated_reply: str,
        retrieved_cases: List[Dict[str, Any]],
        grounding_status: str
    ) -> JudgeScore:
        # Format evidence block
        evidence_lines = []
        for i, c in enumerate(retrieved_cases, start=1):
            cid = c.get("case_id", f"case_{i}")
            sim = float(c.get("similarity", 0.0))
            ctext = c.get("customer_text", "").strip()
            stext = c.get("support_text", "").strip()
            evidence_lines.append(f"[{cid}] (Cosine Sim: {sim:.3f})\n  Customer: {ctext}\n  Spotify Support: {stext}")

        evidence_str = "\n\n".join(evidence_lines) if evidence_lines else "(No relevant historical cases retrieved)"

        user_content = f"""=== CUSTOMER MESSAGE ===
{customer_text}

=== PREDICTED INTENT ===
{predicted_intent}

=== RETRIEVED HISTORICAL EVIDENCE ===
{evidence_str}

=== CANDIDATE GENERATED REPLY ===
{generated_reply}

=== SYSTEM GROUNDING STATUS ===
{grounding_status}

Evaluate the reply strictly according to the 5 dimensions on a 1.0 - 5.0 scale. Output valid JSON adhering to the JudgeScore schema.
"""

        try:
            from google.genai import types
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=JUDGE_SYSTEM_INSTRUCTION,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=JudgeScore
                )
            )

            result_json = json.loads(response.text)
            return JudgeScore.model_validate(result_json)

        except Exception as e:
            # Fallback to mock judge if network/parsing error occurs
            mock = MockLLMJudge()
            score = mock.judge_reply(
                customer_text=customer_text,
                predicted_intent=predicted_intent,
                generated_reply=generated_reply,
                retrieved_cases=retrieved_cases,
                grounding_status=grounding_status
            )
            score.judge_rationale = f"[Fallback Judge due to: {type(e).__name__}] {score.judge_rationale}"
            return score


def get_llm_judge(mode: str = "mock", config: Optional[GenerationConfig] = None) -> BaseLLMJudge:
    """
    Factory function returning GeminiLLMJudge when requested and key exists;
    otherwise returns deterministic MockLLMJudge.
    """
    if mode == "live":
        cfg = config or GenerationConfig.from_env()
        if cfg.has_valid_api_key():
            return GeminiLLMJudge(model_name=cfg.model_name, api_key=cfg.api_key)
        else:
            print("Notice: No live GEMINI_API_KEY found in environment. Defaulting to MockLLMJudge.")
    return MockLLMJudge()


def evaluate_replies(
    replies_df: pd.DataFrame,
    judge: Optional[BaseLLMJudge] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Evaluates a dataframe of generated replies using the specified judge.
    Produces machine-readable reply_llm_judge.csv results.
    """
    if judge is None:
        judge = MockLLMJudge()

    records = []
    print(f"Running LLM Judge evaluation on {len(replies_df)} replies...")

    for idx, row in replies_df.iterrows():
        tweet_id = row.get("tweet_id", idx)
        cust_text = str(row.get("customer_text", ""))
        intent = str(row.get("predicted_intent", "other_non_actionable"))
        reply = str(row.get("generated_reply", ""))
        grounding = str(row.get("grounding_status", "grounded"))
        evidence_ids = str(row.get("evidence_case_ids", ""))

        # Mock retrieved cases metadata if not provided in dataframe
        retrieved_cases = row.get("retrieved_cases", [])
        if not isinstance(retrieved_cases, list) or len(retrieved_cases) == 0:
            retrieved_cases = [{"case_id": cid, "similarity": 0.65} for cid in evidence_ids.split(",") if cid]

        top_sim = max([float(c.get("similarity", 0.0)) for c in retrieved_cases], default=0.0)

        score = judge.judge_reply(
            customer_text=cust_text,
            predicted_intent=intent,
            generated_reply=reply,
            retrieved_cases=retrieved_cases,
            grounding_status=grounding
        )

        records.append({
            "tweet_id": tweet_id,
            "customer_text": cust_text,
            "predicted_intent": intent,
            "reply": reply,
            "evidence_ids": evidence_ids,
            "evidence_similarity": round(top_sim, 4),
            "relevance_score": score.relevance_score,
            "groundedness_score": score.groundedness_score,
            "helpfulness_score": score.helpfulness_score,
            "completeness_score": score.completeness_score,
            "safety_score": score.safety_score,
            "overall_score": score.overall_score,
            "pass_fail": score.pass_fail,
            "hallucination_flag": score.hallucination_flag,
            "evidence_use_flag": score.evidence_use_flag,
            "judge_rationale": score.judge_rationale
        })

    judge_df = pd.DataFrame(records)

    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "evaluation", "results", "reply_llm_judge.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    judge_df.to_csv(output_path, index=False)
    print(f"[OK] Saved LLM Judge results to: {output_path}")

    # Print summary statistics
    mean_rel = judge_df["relevance_score"].mean()
    mean_gnd = judge_df["groundedness_score"].mean()
    mean_hlp = judge_df["helpfulness_score"].mean()
    mean_cmp = judge_df["completeness_score"].mean()
    mean_sft = judge_df["safety_score"].mean()
    mean_ovr = judge_df["overall_score"].mean()
    pass_rate = (judge_df["pass_fail"] == "PASS").mean() * 100
    halluc_count = judge_df["hallucination_flag"].sum()

    print("\n==================================================================")
    print("LLM JUDGE EVALUATION SUMMARY")
    print("==================================================================")
    print(f"Total Replies Judged:       {len(judge_df)}")
    print(f"Quality Gate Pass Rate:     {pass_rate:.1f}%")
    print(f"Hallucination Flags:        {halluc_count} / {len(judge_df)}")
    print("------------------------------------------------------------------")
    print(f"Mean Relevance Score:       {mean_rel:.2f} / 5.0")
    print(f"Mean Groundedness Score:    {mean_gnd:.2f} / 5.0")
    print(f"Mean Helpfulness Score:     {mean_hlp:.2f} / 5.0")
    print(f"Mean Completeness Score:    {mean_cmp:.2f} / 5.0")
    print(f"Mean Safety Score:          {mean_sft:.2f} / 5.0")
    print(f"Mean Overall Score:         {mean_ovr:.2f} / 5.0")
    print("==================================================================\n")

    return judge_df


def run_judge_cli(
    mode: str = "mock",
    limit: int = 30
) -> pd.DataFrame:
    """
    CLI runner: generates replies for representative golden queries and runs the LLM judge.
    """
    from backend.src.generation.generate import assist_customer
    from backend.src.generation.llm import MockLLMClient
    from backend.src.generation.evaluate import select_evaluation_sample

    sample = select_evaluation_sample(sample_size=limit, random_state=2026)
    print(f"Generating candidate replies for {len(sample)} queries (Mode: {mode})...")

    judge = get_llm_judge(mode=mode)
    reply_records = []

    for idx, row in sample.iterrows():
        text = str(row["text"])
        tweet_id = row["tweet_id"]
        # Use mock LLM for reply generation if in mock mode
        llm = MockLLMClient() if mode == "mock" else None
        res = assist_customer(text, top_k=3, llm_client=llm)

        reply_records.append({
            "tweet_id": tweet_id,
            "customer_text": text,
            "predicted_intent": res["predicted_intent"],
            "generated_reply": res["reply"],
            "grounding_status": res["grounding_status"],
            "evidence_case_ids": ",".join(res["evidence_case_ids"]),
            "retrieved_cases": res["retrieved_cases"]
        })

    replies_df = pd.DataFrame(reply_records)
    return evaluate_replies(replies_df, judge=judge)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AssistIQ LLM Judge for Reply Quality")
    parser.add_argument("--mode", choices=["mock", "live"], default="mock", help="Judge execution mode (default: mock)")
    parser.add_argument("--limit", type=int, default=30, help="Number of queries to judge (default: 30)")
    args = parser.parse_args()
    run_judge_cli(mode=args.mode, limit=args.limit)
