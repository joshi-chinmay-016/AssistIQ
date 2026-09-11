/**
 * AssistIQ: TypeScript API Contract Types
 * Directly mirrors backend/api/schemas.py Pydantic schemas.
 * Strictly typed for client-safe operations with zero data duplication.
 */

export interface AssistRequest {
  message: string;
  top_k?: number;
}

export interface IntentInfo {
  name: string;
  confidence: number;
}

export type GroundingStatus = "grounded" | "insufficient_evidence" | "generation_failed";

export interface ReplyInfo {
  text: string;
  grounding_status: GroundingStatus;
  grounding_summary: string;
  evidence_case_ids: string[];
}

export type DecisionType = "auto_handle" | "escalate";
export type RiskLevel = "low" | "medium" | "high";

export interface DecisionInfo {
  decision: DecisionType;
  risk_level: RiskLevel;
  reason: string;
  primary_rule: string;
  policy_rules_triggered: string[];
}

export interface EvidenceCase {
  case_id: string;
  similarity: number;
  customer_text: string;
  support_text: string;
  rank?: number | null;
  conversation_id?: number | null;
}

export interface LatencyInfo {
  intent_ms: number;
  retrieval_ms: number;
  generation_ms: number;
  escalation_ms: number;
  total_ms: number;
}

export interface AssistResponse {
  message: string;
  intent: IntentInfo;
  reply: ReplyInfo;
  decision: DecisionInfo;
  evidence: EvidenceCase[];
  latency: LatencyInfo;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  version: string;
}

export interface RootInfoResponse {
  name: string;
  version: string;
  status: string;
  brand: string;
  docs_url: string;
}

export interface ApiErrorResponse {
  error: string;
  detail: string;
}

/**
 * Frontend Session State Types
 */

export interface SessionStats {
  analyzed: number;
  autoHandled: number;
  escalated: number;
  avgConfidence: number;
}

export interface ConversationTurn {
  id: string;
  timestamp: string;
  customerMessage: string;
  response?: AssistResponse;
  status: "loading" | "success" | "error";
  error?: string;
}

/**
 * Presentation Mappings for Intent Taxonomy
 */
export const INTENT_DISPLAY_NAMES: Record<string, string> = {
  playback_and_app_issues: "Playback & App Issues",
  billing_and_payment: "Billing & Payment",
  premium_and_subscription: "Premium & Subscription",
  account_and_login: "Account & Login",
  music_availability: "Music Availability",
  playlist_and_library: "Playlist & Library",
  content_metadata: "Content & Metadata",
  feature_requests: "Feature Requests",
  search_and_discovery: "Search & Discovery",
  ads_and_privacy: "Ads & Privacy",
  other_non_actionable: "Other / General Support",
};

export function formatIntentName(name: string): string {
  if (INTENT_DISPLAY_NAMES[name]) {
    return INTENT_DISPLAY_NAMES[name];
  }
  return name
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}
