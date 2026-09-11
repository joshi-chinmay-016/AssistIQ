"use client";

import React, { useState, useEffect, useCallback } from "react";
import Sidebar from "@/components/Sidebar";
import HeroEmbedding from "@/components/HeroEmbedding";
import ConversationPanel from "@/components/ConversationPanel";
import Composer from "@/components/Composer";
import EvidencePanel from "@/components/EvidencePanel";
import {
  AssistResponse,
  ConversationTurn,
  SessionStats,
} from "@/lib/types";
import { analyzeCustomerMessage, checkBackendHealth } from "@/lib/api";

export default function Home() {
  // Session State
  const [turns, setTurns] = useState<ConversationTurn[]>([]);
  const [currentResponse, setCurrentResponse] = useState<AssistResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);
  const [isBackendConnected, setIsBackendConnected] = useState<boolean | null>(null);
  const [customerInput, setCustomerInput] = useState<string>("");

  // Mobile View Tab: "workspace" | "inspection"
  const [mobileTab, setMobileTab] = useState<"workspace" | "inspection">("workspace");

  // Real session statistics calculated solely from live API interactions
  const [stats, setStats] = useState<SessionStats>({
    analyzed: 0,
    autoHandled: 0,
    escalated: 0,
    avgConfidence: 0,
  });

  // Check backend health on mount and periodically
  const verifyHealth = useCallback(() => {
    checkBackendHealth().then((health) => {
      setIsBackendConnected(health !== null);
    });
  }, []);

  useEffect(() => {
    let active = true;
    checkBackendHealth().then((health) => {
      if (active) {
        setIsBackendConnected(health !== null);
      }
    });

    const interval = setInterval(() => {
      checkBackendHealth().then((health) => {
        if (active) {
          setIsBackendConnected(health !== null);
        }
      });
    }, 20000);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  // Main Query Analysis Handler
  const handleAnalyze = async (message: string) => {
    if (!message.trim() || isAnalyzing) return;

    setIsAnalyzing(true);
    setApiError(null);
    setCustomerInput("");

    const turnId = `turn-${Date.now()}`;
    const timestamp = new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });

    // Optimistically add customer message to thread
    const newTurn: ConversationTurn = {
      id: turnId,
      timestamp,
      customerMessage: message,
      status: "loading",
    };

    setTurns((prev) => [...prev, newTurn]);

    try {
      const response = await analyzeCustomerMessage(message);

      // Backend connection confirmed
      setIsBackendConnected(true);

      // Update current active response for inspection panel
      setCurrentResponse(response);

      // Update conversation thread with successful grounded reply
      setTurns((prev) =>
        prev.map((t) =>
          t.id === turnId
            ? { ...t, response, status: "success" }
            : t
        )
      );

      // Update session statistics with genuine data
      setStats((prev) => {
        const newAnalyzed = prev.analyzed + 1;
        const isAuto = response.decision.decision === "auto_handle";
        const isEsc = response.decision.decision === "escalate";
        const newAuto = prev.autoHandled + (isAuto ? 1 : 0);
        const newEsc = prev.escalated + (isEsc ? 1 : 0);
        const newAvgConf =
          (prev.avgConfidence * prev.analyzed + response.intent.confidence) /
          newAnalyzed;

        return {
          analyzed: newAnalyzed,
          autoHandled: newAuto,
          escalated: newEsc,
          avgConfidence: newAvgConf,
        };
      });
    } catch (err: unknown) {
      const errorMsg =
        err instanceof Error
          ? err.message
          : "An unexpected error occurred while analyzing query.";

      setApiError(errorMsg);

      // Mark turn as failed
      setTurns((prev) =>
        prev.map((t) =>
          t.id === turnId
            ? { ...t, status: "error", error: errorMsg }
            : t
        )
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleSelectSamplePrompt = (prompt: string) => {
    setCustomerInput(prompt);
  };

  return (
    <div className="flex flex-col md:flex-row h-screen w-screen overflow-hidden bg-[#090a0d] text-[#eae8e3]">
      {/* 1. Left Sidebar: Fixed ~220px on desktop */}
      <Sidebar
        stats={stats}
        isBackendConnected={isBackendConnected}
        onRefreshHealth={verifyHealth}
      />

      {/* Mobile Tab Switcher (Visible on mobile/small screens only) */}
      <div className="md:hidden flex border-b border-[#1f222e] bg-[#111217] shrink-0 font-mono text-xs">
        <button
          type="button"
          onClick={() => setMobileTab("workspace")}
          className={`flex-1 py-2.5 text-center transition-colors ${
            mobileTab === "workspace"
              ? "border-b-2 border-[#00d4c8] text-[#00d4c8] font-semibold"
              : "text-[#858a98]"
          }`}
        >
          Conversation Workspace
        </button>
        <button
          type="button"
          onClick={() => setMobileTab("inspection")}
          className={`flex-1 py-2.5 text-center transition-colors ${
            mobileTab === "inspection"
              ? "border-b-2 border-[#f59e0b] text-[#f59e0b] font-semibold"
              : "text-[#858a98]"
          }`}
        >
          AI Inspection {currentResponse ? "•" : ""}
        </button>
      </div>

      {/* 2. Central Workspace: Flexible center column */}
      <main
        className={`flex-1 flex flex-col h-full overflow-hidden bg-[#0e0f14] ${
          mobileTab === "inspection" ? "hidden md:flex" : "flex"
        }`}
      >
        {/* Top Hero: 3D Vector Embedding Space & Telemetry */}
        <HeroEmbedding
          isAnalyzing={isAnalyzing}
          evidence={currentResponse ? currentResponse.evidence : null}
          latency={currentResponse ? currentResponse.latency : null}
          predictedIntent={currentResponse ? currentResponse.intent.name : null}
        />

        {/* Central Conversation Stream */}
        <ConversationPanel
          turns={turns}
          isAnalyzing={isAnalyzing}
          onSelectPrompt={handleSelectSamplePrompt}
        />

        {/* Bottom Input Composer */}
        <Composer
          input={customerInput}
          onInputChange={setCustomerInput}
          onAnalyze={handleAnalyze}
          isAnalyzing={isAnalyzing}
          error={apiError}
          onClearError={() => setApiError(null)}
        />
      </main>

      {/* 3. Right Panel: Fixed ~340px on desktop */}
      <div
        className={`h-full ${
          mobileTab === "workspace" ? "hidden md:flex" : "flex"
        }`}
      >
        <EvidencePanel
          currentResponse={currentResponse}
          isAnalyzing={isAnalyzing}
        />
      </div>
    </div>
  );
}
