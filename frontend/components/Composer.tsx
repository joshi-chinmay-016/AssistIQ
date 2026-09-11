"use client";

import React, { useState, useEffect, KeyboardEvent } from "react";

interface ComposerProps {
  input: string;
  onInputChange: (val: string) => void;
  onAnalyze: (message: string) => void;
  isAnalyzing: boolean;
  error: string | null;
  onClearError: () => void;
}

const STAGES = [
  "ANALYZING REQUEST...",
  "CLASSIFYING INTENT...",
  "RETRIEVING EVIDENCE...",
  "GENERATING GROUNDED REPLY...",
  "CHECKING ESCALATION POLICY...",
];

export default function Composer({
  input,
  onInputChange,
  onAnalyze,
  isAnalyzing,
  error,
  onClearError,
}: ComposerProps) {
  const [stageIndex, setStageIndex] = useState(0);

  // Cycle through visual loading stages while analyzing
  useEffect(() => {
    if (!isAnalyzing) return;

    const interval = setInterval(() => {
      setStageIndex((prev) => (prev + 1) % STAGES.length);
    }, 1800);

    return () => clearInterval(interval);
  }, [isAnalyzing]);

  const handleSubmit = () => {
    if (!input.trim() || isAnalyzing) return;
    onAnalyze(input.trim());
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="p-3 sm:p-4 bg-[#111217] border-t border-[#1f222e] select-none">
      {/* Active Pipeline Progress Banner */}
      {isAnalyzing && (
        <div className="mb-3 px-3 py-2 rounded bg-[#0b0c10] border border-[#f59e0b]/40 flex items-center justify-between font-mono text-xs text-[#f59e0b] shadow-[0_0_12px_rgba(245,158,11,0.15)]">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-[#f59e0b] animate-ping" />
            <span className="font-semibold tracking-wider">
              {STAGES[stageIndex]}
            </span>
          </div>
          <span className="text-[10px] text-[#858a98]">
            Phase 1 → 4 Execution
          </span>
        </div>
      )}

      {/* Error State Banner */}
      {error && (
        <div className="mb-3 px-3 py-2.5 rounded bg-[#ef4444]/15 border border-[#ef4444]/40 flex items-center justify-between text-xs font-mono text-[#ef4444]">
          <div className="flex items-center gap-2 overflow-hidden">
            <svg
              className="w-4 h-4 shrink-0 text-[#ef4444]"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <span className="truncate">{error}</span>
          </div>
          <button
            type="button"
            onClick={onClearError}
            className="text-[11px] underline hover:text-white shrink-0 ml-2"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Input Composer Box */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-end gap-2 bg-[#0b0c10] border border-[#1f222e] focus-within:border-[#00d4c8]/60 focus-within:ring-1 focus-within:ring-[#00d4c8]/30 rounded-lg p-2 transition-all">
        <textarea
          rows={2}
          value={input}
          onChange={(e) => {
            onInputChange(e.target.value);
            if (error) onClearError();
          }}
          onKeyDown={handleKeyDown}
          disabled={isAnalyzing}
          placeholder="Type a customer message... (Press Enter to analyze)"
          className="w-full bg-transparent text-sm text-[#eae8e3] placeholder-[#555967] resize-none focus:outline-none px-2 py-1 font-sans leading-relaxed disabled:opacity-50"
        />

        <div className="flex items-center justify-between sm:justify-end gap-2 shrink-0 pt-1 sm:pt-0">
          <span className="text-[10px] font-mono text-[#555967] hidden md:inline">
            Enter ↵
          </span>

          <button
            type="button"
            onClick={handleSubmit}
            disabled={!input.trim() || isAnalyzing}
            className={`px-4 py-2 rounded font-mono text-xs font-semibold tracking-wider uppercase transition-all flex items-center gap-1.5 ${
              !input.trim() || isAnalyzing
                ? "bg-[#181a24] text-[#555967] border border-[#1f222e] cursor-not-allowed"
                : "bg-[#00d4c8] hover:bg-[#22d3ee] text-[#090a0d] shadow-[0_0_12px_rgba(0,212,200,0.25)] hover:shadow-[0_0_18px_rgba(0,212,200,0.4)] cursor-pointer active:scale-[0.98]"
            }`}
          >
            {isAnalyzing ? (
              <>
                <span className="w-2.5 h-2.5 rounded-full border-2 border-[#555967] border-t-transparent animate-spin" />
                <span>Running...</span>
              </>
            ) : (
              <>
                <svg
                  className="w-3.5 h-3.5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M13 10V3L4 14h7v7l9-11h-7z"
                  />
                </svg>
                <span>Analyze</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
