"use client";

import React, { useState } from "react";
import { ConversationTurn } from "@/lib/types";

interface ConversationPanelProps {
  turns: ConversationTurn[];
  isAnalyzing: boolean;
  onSelectPrompt: (prompt: string) => void;
}

export default function ConversationPanel({
  turns,
  isAnalyzing,
  onSelectPrompt,
}: ConversationPanelProps) {
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const samplePrompts = [
    "I was charged twice for Spotify Premium this month. Can I get a refund?",
    "Every song stops after 30 seconds on my phone.",
    "How can I download music to listen offline while flying?",
  ];

  return (
    <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6">
      {/* Empty Welcome State */}
      {turns.length === 0 && !isAnalyzing && (
        <div className="h-full flex flex-col items-center justify-center text-center max-w-lg mx-auto py-12 px-4 select-none">
          <div className="w-10 h-10 rounded-lg bg-[#181a24] border border-[#2a2e3d] flex items-center justify-center mb-4 text-[#00d4c8] shadow-[0_0_15px_rgba(0,212,200,0.1)]">
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="1.8"
                d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
              />
            </svg>
          </div>

          <h2 className="text-base font-medium text-[#eae8e3] tracking-tight">
            AssistIQ Support Console
          </h2>
          <p className="text-xs text-[#858a98] mt-1.5 leading-relaxed font-sans">
            Send a customer message to analyze with the 4-phase AssistIQ pipeline:
            intent classification, FAISS vector retrieval, grounded Gemini reply generation, and deterministic escalation policy.
          </p>

          <div className="mt-6 w-full text-left">
            <span className="text-[10px] font-mono uppercase tracking-wider text-[#555967] block mb-2 text-center">
              Sample queries to test
            </span>
            <div className="space-y-2">
              {samplePrompts.map((prompt, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => onSelectPrompt(prompt)}
                  className="w-full text-left p-2.5 rounded bg-[#111217] hover:bg-[#181a24] border border-[#1f222e] hover:border-[#00d4c8]/40 transition-all text-xs font-mono text-[#858a98] hover:text-[#eae8e3] flex items-center justify-between group"
                >
                  <span className="truncate pr-2">&ldquo;{prompt}&rdquo;</span>
                  <span className="text-[10px] text-[#00d4c8] opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
                    Use ↗
                  </span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Conversation Stream */}
      {turns.map((turn, index) => (
        <div key={turn.id} className="space-y-4 font-sans">
          {/* Query Header Divider for subsequent turns */}
          {index > 0 && (
            <div className="relative flex py-2 items-center">
              <div className="flex-grow border-t border-[#1f222e]"></div>
              <span className="flex-shrink mx-3 text-[10px] font-mono text-[#555967] uppercase">
                Query {index + 1} · {turn.timestamp}
              </span>
              <div className="flex-grow border-t border-[#1f222e]"></div>
            </div>
          )}

          {/* Customer Inquiry Bubble */}
          <div className="flex justify-end">
            <div className="max-w-[85%] md:max-w-[75%] rounded-lg p-3.5 bg-[#171922] border border-[#2a2e3d] shadow-sm">
              <div className="flex items-center justify-between gap-3 mb-1.5">
                <span className="text-[10px] font-mono uppercase tracking-wider text-[#858a98]">
                  Customer Message
                </span>
                <span className="text-[10px] font-mono text-[#555967]">
                  {turn.timestamp}
                </span>
              </div>
              <p className="text-sm text-[#eae8e3] leading-relaxed whitespace-pre-wrap">
                {turn.customerMessage}
              </p>
            </div>
          </div>

          {/* AssistIQ Grounded Reply Bubble */}
          {turn.response && (
            <div className="flex justify-start">
              <div className="max-w-[90%] md:max-w-[85%] rounded-lg p-4 bg-[#111217] border border-[#1f222e] shadow-md relative">
                {/* Header Badge */}
                <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5 pb-2 border-b border-[#1f222e]">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-sm bg-[#00d4c8]" />
                    <span className="text-xs font-mono font-medium tracking-tight text-[#00d4c8]">
                      assistiq — {turn.response.reply.grounding_status === "grounded"
                        ? "grounded reply"
                        : turn.response.reply.grounding_status === "insufficient_evidence"
                        ? "general guidance"
                        : "fallback"}
                    </span>
                  </div>

                  {/* Copy Action */}
                  <button
                    type="button"
                    onClick={() => handleCopy(turn.response!.reply.text, turn.id)}
                    className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#181a24] hover:bg-[#202330] text-[#858a98] hover:text-[#eae8e3] border border-[#2a2e3d] transition-colors"
                  >
                    {copiedId === turn.id ? "Copied ✓" : "Copy Reply"}
                  </button>
                </div>

                {/* Reply Body */}
                <p className="text-sm text-[#eae8e3] leading-relaxed whitespace-pre-wrap font-sans">
                  {turn.response.reply.text}
                </p>

                {/* Grounding Citations */}
                {turn.response.reply.evidence_case_ids &&
                  turn.response.reply.evidence_case_ids.length > 0 && (
                    <div className="mt-3 pt-2.5 border-t border-[#1f222e] flex flex-wrap items-center gap-1.5">
                      <span className="text-[10px] font-mono text-[#555967] uppercase">
                        Evidence Cited:
                      </span>
                      {turn.response.reply.evidence_case_ids.map((caseId, cid) => (
                        <span
                          key={cid}
                          className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#0b0c10] text-[#00d4c8] border border-[#1f222e]"
                        >
                          #{caseId}
                        </span>
                      ))}
                    </div>
                  )}

                {/* Internal Grounding Note */}
                {turn.response.reply.grounding_summary && (
                  <div className="mt-2 text-[10.5px] font-mono text-[#858a98] bg-[#0b0c10] p-2 rounded border border-[#1a1c24]">
                    <span className="text-[#555967] uppercase">Verification: </span>
                    {turn.response.reply.grounding_summary}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Error Message if Turn Failed */}
          {turn.error && (
            <div className="flex justify-start">
              <div className="max-w-[85%] rounded-lg p-3 bg-[#ef4444]/10 border border-[#ef4444]/30 text-xs font-mono text-[#ef4444]">
                <div className="font-semibold mb-1">Pipeline Error</div>
                <p>{turn.error}</p>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
