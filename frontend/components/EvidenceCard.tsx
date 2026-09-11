"use client";

import React from "react";
import { EvidenceCase } from "@/lib/types";

interface EvidenceCardProps {
  evidence: EvidenceCase;
  index: number;
  isCited?: boolean;
}

export default function EvidenceCard({
  evidence,
  index,
  isCited = false,
}: EvidenceCardProps) {
  return (
    <div
      className="evidence-paper-card rounded-md p-3 select-none text-left border border-[#ded8cb]"
      style={{
        animationDelay: `${index * 80}ms`,
      }}
    >
      {/* Card Header: Case ID and Cosine Similarity */}
      <div className="flex items-center justify-between gap-2 mb-2 pb-1.5 border-b border-[#ded8cb]/80">
        <div className="flex items-center gap-1.5 font-mono">
          <span className="font-bold text-xs text-[#16171a]">
            #{evidence.case_id.startsWith("SPT-") ? evidence.case_id : `SPT-${evidence.case_id}`}
          </span>
          {isCited && (
            <span className="text-[9px] px-1 py-0.2 rounded bg-[#0f766e] text-white font-medium">
              CITED
            </span>
          )}
        </div>

        <div className="flex items-center gap-1 font-mono text-[11px] font-semibold text-[#0f766e]">
          <span>sim</span>
          <span>{evidence.similarity.toFixed(4)}</span>
        </div>
      </div>

      {/* Inquiry Snippet */}
      <div className="mb-2">
        <span className="text-[9px] font-mono uppercase tracking-wider text-[#5a5d6a] block mb-0.5 font-semibold">
          Customer Inquiry
        </span>
        <p className="text-[11px] text-[#16171a] font-sans leading-snug line-clamp-3 italic">
          &ldquo;{evidence.customer_text}&rdquo;
        </p>
      </div>

      {/* Historical Resolution Snippet */}
      <div>
        <span className="text-[9px] font-mono uppercase tracking-wider text-[#5a5d6a] block mb-0.5 font-semibold">
          Historical Resolution
        </span>
        <p className="text-[11px] text-[#2c2f38] font-sans leading-snug line-clamp-3">
          {evidence.support_text || "Case resolved via standard Spotify support procedure."}
        </p>
      </div>
    </div>
  );
}
