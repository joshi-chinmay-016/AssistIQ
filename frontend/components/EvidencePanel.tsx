"use client";

import React from "react";
import { AssistResponse, formatIntentName } from "@/lib/types";
import ConfidenceMeter from "./ConfidenceMeter";
import DecisionStamp from "./DecisionStamp";
import EvidenceCard from "./EvidenceCard";

interface EvidencePanelProps {
  currentResponse: AssistResponse | null;
  isAnalyzing: boolean;
}

export default function EvidencePanel({
  currentResponse,
  isAnalyzing,
}: EvidencePanelProps) {
  return (
    <aside className="w-full lg:w-[340px] shrink-0 bg-[#111217] border-t lg:border-t-0 lg:border-l border-[#1f222e] flex flex-col h-full overflow-y-auto select-none">
      {/* Panel Top Title */}
      <div className="p-4 border-b border-[#1f222e] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-sm bg-[#f59e0b]" />
          <span className="text-xs font-mono font-semibold uppercase tracking-wider text-[#eae8e3]">
            AI Inspection & Policy
          </span>
        </div>
        <span className="text-[10px] font-mono text-[#555967]">
          SpotifyCares Guardrails
        </span>
      </div>

      <div className="p-4 space-y-4 flex-1">
        {/* Placeholder if no response yet */}
        {!currentResponse && !isAnalyzing && (
          <div className="h-64 flex flex-col items-center justify-center text-center p-4 border border-dashed border-[#1f222e] rounded-lg">
            <svg
              className="w-8 h-8 text-[#555967] mb-2"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="1.5"
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            </svg>
            <p className="text-xs font-mono text-[#858a98]">
              No query evaluated yet.
            </p>
            <p className="text-[11px] text-[#555967] mt-1 font-sans">
              AI intent classification, confidence scores, and FAISS evidence will display here.
            </p>
          </div>
        )}

        {/* Loading Spinner */}
        {isAnalyzing && (
          <div className="p-8 flex flex-col items-center justify-center text-center border border-[#1f222e] rounded-lg bg-[#0b0c10]">
            <div className="w-7 h-7 border-2 border-[#00d4c8] border-t-transparent rounded-full animate-spin mb-3" />
            <span className="text-xs font-mono text-[#00d4c8] uppercase tracking-wider">
              Evaluating Guardrails...
            </span>
            <span className="text-[10.5px] font-mono text-[#555967] mt-1">
              Deterministic Policy Engine
            </span>
          </div>
        )}

        {/* Active AI Response Inspection */}
        {currentResponse && !isAnalyzing && (
          <div className="space-y-4">
            {/* 1. Intent Classification Card */}
            <div className="p-3.5 rounded-lg bg-[#0b0c10] border border-[#1f222e] space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-mono uppercase tracking-wider text-[#858a98]">
                  Intent Classification
                </span>
                <span className="text-[9px] font-mono px-1 py-0.2 rounded bg-[#181a24] text-[#00d4c8]">
                  Phase 1 Model
                </span>
              </div>

              <div>
                <div className="text-sm font-semibold text-[#eae8e3] font-sans">
                  {formatIntentName(currentResponse.intent.name)}
                </div>
                <div className="text-[11px] font-mono text-[#555967]">
                  {currentResponse.intent.name}
                </div>
              </div>

              {/* Confidence Meter */}
              <div className="pt-2 border-t border-[#1f222e]">
                <ConfidenceMeter confidence={currentResponse.intent.confidence} />
              </div>
            </div>

            {/* 2. Decision Stamp Card */}
            <DecisionStamp
              decision={currentResponse.decision}
              groundingStatus={currentResponse.reply.grounding_status}
            />

            {/* 3. Retrieved FAISS Evidence Cards */}
            <div className="space-y-2.5 pt-1">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-mono uppercase tracking-wider text-[#858a98]">
                  Retrieved Evidence — FAISS Top {currentResponse.evidence.length}
                </span>
                <span className="text-[9px] font-mono text-[#555967]">
                  IndexFlatIP
                </span>
              </div>

              {currentResponse.evidence.length === 0 ? (
                <div className="p-3 rounded bg-[#0b0c10] border border-[#1f222e] text-xs font-mono text-[#858a98]">
                  No historical evidence cases retrieved.
                </div>
              ) : (
                <div className="space-y-2.5">
                  {currentResponse.evidence.map((caseItem, idx) => {
                    const isCited = currentResponse.reply.evidence_case_ids?.includes(
                      caseItem.case_id
                    );
                    return (
                      <EvidenceCard
                        key={caseItem.case_id || idx}
                        evidence={caseItem}
                        index={idx}
                        isCited={isCited}
                      />
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
