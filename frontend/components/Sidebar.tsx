"use client";

import React from "react";
import { SessionStats } from "@/lib/types";

interface SidebarProps {
  stats: SessionStats;
  isBackendConnected: boolean | null;
  onRefreshHealth: () => void;
  activeNav?: string;
  onSelectNav?: (nav: string) => void;
}

export default function Sidebar({
  stats,
  isBackendConnected,
  onRefreshHealth,
  activeNav = "queue",
  onSelectNav,
}: SidebarProps) {
  const navItems = [
    { id: "queue", label: "Queue", badge: stats.analyzed > 0 ? `${stats.analyzed}` : null, active: true },
    { id: "escalated", label: "Escalated", badge: stats.escalated > 0 ? `${stats.escalated}` : null, active: false },
    { id: "analytics", label: "Analytics", tag: "v0.6", active: false },
    { id: "evidence", label: "Evidence base", tag: "40.8k", active: false },
    { id: "settings", label: "Settings", tag: null, active: false },
  ];

  return (
    <aside className="w-full md:w-[220px] shrink-0 bg-[#111217] border-r border-[#1f222e] flex flex-col justify-between select-none">
      {/* Top Brand Header */}
      <div>
        <div className="p-4 border-b border-[#1f222e]">
          <div className="flex items-center justify-between">
            <div className="flex items-baseline gap-1">
              <span className="font-bold tracking-tight text-lg text-[#eae8e3]">
                ASSIST
              </span>
              <span className="font-bold tracking-tight text-lg text-[#f59e0b]">
                IQ
              </span>
            </div>
            <span className="font-mono text-[10px] tracking-wider uppercase px-1.5 py-0.5 rounded bg-[#1c1f2b] text-[#858a98] border border-[#2a2e3d]">
              v0.5.0
            </span>
          </div>
          <div className="text-[11px] font-mono text-[#858a98] mt-1 flex items-center gap-1.5">
            <span>SpotifyCares Operations</span>
          </div>

          {/* Backend Connection Indicator */}
          <div
            onClick={onRefreshHealth}
            title="Click to re-check FastAPI backend health (/health)"
            className="mt-3 flex items-center justify-between px-2 py-1.5 rounded bg-[#0b0c10] border border-[#1f222e] text-[11px] font-mono cursor-pointer hover:border-[#2c3040] transition-colors"
          >
            <div className="flex items-center gap-1.5">
              <span
                className={`inline-block w-2 h-2 rounded-full ${
                  isBackendConnected === true
                    ? "bg-[#00d4c8] shadow-[0_0_8px_#00d4c8]"
                    : isBackendConnected === false
                    ? "bg-[#ef4444] shadow-[0_0_8px_#ef4444]"
                    : "bg-[#858a98]"
                }`}
              />
              <span
                className={
                  isBackendConnected === true
                    ? "text-[#00d4c8]"
                    : isBackendConnected === false
                    ? "text-[#ef4444]"
                    : "text-[#858a98]"
                }
              >
                {isBackendConnected === true
                  ? "API Connected"
                  : isBackendConnected === false
                  ? "API Offline"
                  : "Checking API..."}
              </span>
            </div>
            <span className="text-[10px] text-[#555967]">:8000</span>
          </div>
        </div>

        {/* Navigation Section */}
        <nav className="p-2 space-y-1" aria-label="Support console navigation">
          {navItems.map((item) => {
            const isActive = activeNav === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onSelectNav && onSelectNav(item.id)}
                className={`w-full flex items-center justify-between px-3 py-2 rounded text-xs font-mono text-left transition-all ${
                  isActive
                    ? "bg-[#181a24] text-[#eae8e3] border-l-2 border-[#00d4c8] font-medium"
                    : "text-[#858a98] hover:text-[#eae8e3] hover:bg-[#151720]"
                }`}
              >
                <div className="flex items-center gap-2">
                  <span
                    className={`w-1.5 h-1.5 rounded-sm ${
                      isActive ? "bg-[#00d4c8]" : "bg-transparent"
                    }`}
                  />
                  <span>{item.label}</span>
                </div>

                {item.badge && (
                  <span className="px-1.5 py-0.5 text-[10px] font-mono rounded bg-[#1f222e] text-[#00d4c8]">
                    {item.badge}
                  </span>
                )}

                {item.tag && (
                  <span className="text-[10px] font-mono text-[#555967]">
                    {item.tag}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Bottom Queue / Session Statistics */}
      <div className="p-3 m-2 rounded bg-[#0b0c10] border border-[#1f222e]">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[10px] font-mono uppercase tracking-wider text-[#858a98]">
            Session Metrics
          </span>
          <span className="text-[9px] font-mono px-1 py-0.2 rounded bg-[#191c26] text-[#555967]">
            LIVE
          </span>
        </div>

        <div className="space-y-2 font-mono">
          <div className="flex items-center justify-between text-xs">
            <span className="text-[#858a98]">Analyzed</span>
            <span className="text-[#eae8e3] font-semibold">{stats.analyzed}</span>
          </div>

          <div className="flex items-center justify-between text-xs">
            <span className="text-[#00d4c8]">Auto-handled</span>
            <span className="text-[#00d4c8] font-semibold">
              {stats.autoHandled}
            </span>
          </div>

          <div className="flex items-center justify-between text-xs">
            <span className="text-[#f59e0b]">Escalated</span>
            <span className="text-[#f59e0b] font-semibold">
              {stats.escalated}
            </span>
          </div>

          <div className="pt-1 border-t border-[#1f222e] flex items-center justify-between text-xs">
            <span className="text-[#858a98]">Avg Conf.</span>
            <span className="text-[#eae8e3]">
              {stats.analyzed > 0 ? stats.avgConfidence.toFixed(2) : "—"}
            </span>
          </div>
        </div>

        <p className="text-[9px] font-mono text-[#555967] mt-2 leading-tight">
          Calculated dynamically from real browser session requests.
        </p>
      </div>
    </aside>
  );
}
