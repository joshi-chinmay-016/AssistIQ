"use client";

import React, { useEffect, useState } from "react";

interface ConfidenceMeterProps {
  confidence: number;
}

export default function ConfidenceMeter({ confidence }: ConfidenceMeterProps) {
  const [animatedWidth, setAnimatedWidth] = useState(0);

  useEffect(() => {
    // Clamp safely between 0 and 1
    const clamped = Math.max(0, Math.min(1, confidence));
    const target = clamped * 100;

    // Small delay to trigger smooth transition on new response
    const timeout = setTimeout(() => {
      setAnimatedWidth(target);
    }, 50);

    return () => clearTimeout(timeout);
  }, [confidence]);

  const clamped = Math.max(0, Math.min(1, confidence));

  return (
    <div className="space-y-1.5 font-mono select-none">
      <div className="flex items-center justify-between text-xs">
        <span className="text-[10px] tracking-wider uppercase text-[#858a98]">
          Intent Confidence
        </span>
        <span className="text-[#00d4c8] font-bold text-xs">
          {clamped.toFixed(2)} ({Math.round(clamped * 100)}%)
        </span>
      </div>

      {/* Track */}
      <div className="h-1.5 w-full bg-[#181a24] rounded-full overflow-hidden border border-[#2a2e3d]">
        <div
          className="h-full bg-gradient-to-r from-[#00d4c8] to-[#22d3ee] rounded-full transition-all duration-700 ease-out shadow-[0_0_8px_#00d4c8]"
          style={{ width: `${animatedWidth}%` }}
        />
      </div>
    </div>
  );
}
