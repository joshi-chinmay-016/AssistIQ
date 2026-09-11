"use client";

import React from "react";
import { DecisionInfo, GroundingStatus } from "@/lib/types";

interface DecisionStampProps {
  decision: DecisionInfo;
  groundingStatus: GroundingStatus;
}

export default function DecisionStamp({
  decision,
  groundingStatus,
}: DecisionStampProps) {
  const isAutoHandle = decision.decision === "auto_handle";
  const isHighRisk = decision.risk_level === "high";

  // Stamp appearance
  const stampClasses = isAutoHandle
    ? "stamp-auto-handle"
    : isHighRisk
    ? "stamp-high-risk"
    : "stamp-escalate";

  const title = isAutoHandle ? "AUTO-HANDLED" : "ESCALATE";

  return (
    <div className="rounded-lg p-3.5 bg-[#111217] border border-[#1f222e] space-y-3 select-none">
      {/* Top Header */}
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-mono uppercase tracking-wider text-[#858a98]">
          Operational Decision
        </span>
        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#181a24] text-[#858a98] border border-[#2a2e3d]">
          Phase 4 Policy
        </span>
      </div>

      {/* Prominent Physical Stamp */}
      <div className="py-2 flex justify-center">
        <div
          className={`px-5 py-2 rounded border-2 font-mono font-black tracking-widest text-sm transition-transform duration-300 ${stampClasses}`}
        >
          {title}
        </div>
      </div>

      {/* Badges: Risk Level, Grounding Status, Primary Rule */}
      <div className="grid grid-cols-3 gap-1.5 text-[11px] font-mono">
        {/* Risk Level */}
        <div className="p-1.5 rounded bg-[#0b0c10] border border-[#1f222e]">
          <span className="text-[8.5px] uppercase text-[#555967] block">Risk</span>
          <span
            className={`font-semibold uppercase text-[11px] ${
              decision.risk_level === "low"
                ? "text-[#00d4c8]"
                : decision.risk_level === "medium"
                ? "text-[#f59e0b]"
                : "text-[#ef4444]"
            }`}
          >
            {decision.risk_level}
          </span>
        </div>

        {/* Grounding Status */}
        <div className="p-1.5 rounded bg-[#0b0c10] border border-[#1f222e]">
          <span className="text-[8.5px] uppercase text-[#555967] block">Grounding</span>
          <span
            className={`font-semibold text-[11px] ${
              groundingStatus === "grounded"
                ? "text-[#00d4c8]"
                : groundingStatus === "insufficient_evidence"
                ? "text-[#f59e0b]"
                : "text-[#ef4444]"
            }`}
          >
            {groundingStatus === "grounded" ? "Verified" : "Ungrounded"}
          </span>
        </div>

        {/* Primary Rule */}
        <div className="p-1.5 rounded bg-[#0b0c10] border border-[#1f222e]">
          <span className="text-[8.5px] uppercase text-[#555967] block">Rule</span>
          <span className="font-semibold text-[#eae8e3] text-[11px]">
            {decision.primary_rule || "—"}
          </span>
        </div>
      </div>

      {/* Triggered Policy Rules */}
      {decision.policy_rules_triggered && decision.policy_rules_triggered.length > 0 && (
        <div className="p-2 rounded bg-[#0b0c10] border border-[#1f222e] text-[11px] font-mono">
          <span className="text-[9px] uppercase text-[#555967] block mb-1">
            Rules Triggered
          </span>
          <div className="flex flex-wrap gap-1">
            {decision.policy_rules_triggered.map((rule, idx) => (
              <span
                key={idx}
                className="px-1.5 py-0.5 rounded bg-[#181a24] text-[#f59e0b] border border-[#2a2e3d]"
              >
                {rule}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Decision Rationale */}
      <div className="p-2.5 rounded bg-[#0b0c10] border border-[#1f222e] text-xs font-mono text-[#858a98]">
        <span className="text-[9px] uppercase tracking-wider text-[#555967] block mb-1">
          Policy Rationale
        </span>
        <p className="text-[#eae8e3] text-[11.5px] leading-relaxed">
          {decision.reason}
        </p>
      </div>
    </div>
  );
}
