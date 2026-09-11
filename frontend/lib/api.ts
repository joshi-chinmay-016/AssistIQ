/**
 * AssistIQ: Centralized HTTP API Client
 * Dispatches requests to the FastAPI backend service (Phases 1-5).
 * Never exposes server paths, stack traces, or credentials.
 */

import { AssistRequest, AssistResponse, HealthResponse, ApiErrorResponse } from "./types";

const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"
).replace(/\/$/, "");

/**
 * Checks server health via GET /health with a lightweight timeout.
 */
export async function checkBackendHealth(): Promise<HealthResponse | null> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 3500);

  try {
    const res = await fetch(`${API_BASE_URL}/health`, {
      method: "GET",
      headers: { Accept: "application/json" },
      signal: controller.signal,
      cache: "no-store",
    });

    if (!res.ok) {
      return null;
    }

    const data: HealthResponse = await res.json();
    return data;
  } catch {
    return null;
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * Executes full AssistIQ 4-phase pipeline via POST /api/v1/assist.
 */
export async function analyzeCustomerMessage(
  message: string,
  topK: number = 5
): Promise<AssistResponse> {
  const trimmed = message.trim();

  if (!trimmed) {
    throw new Error("Please enter a customer message before analyzing.");
  }

  if (trimmed.length > 2000) {
    throw new Error("Message exceeds maximum supported length of 2000 characters.");
  }

  const payload: AssistRequest = {
    message: trimmed,
    top_k: topK,
  };

  let response: Response;
  const controller = new AbortController();
  // Allow up to 35 seconds for cold start / Gemini inference
  const timeoutId = setTimeout(() => controller.abort(), 35000);

  try {
    response = await fetch(`${API_BASE_URL}/api/v1/assist`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
  } catch (err: unknown) {
    clearTimeout(timeoutId);
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error(
        "Request timed out. The backend or LLM took longer than 35 seconds to respond."
      );
    }
    throw new Error(
      `AssistIQ backend unavailable at ${API_BASE_URL}. Ensure the FastAPI server is running.`
    );
  } finally {
    clearTimeout(timeoutId);
  }

  // Handle HTTP error responses
  if (!response.ok) {
    let errorDetail = `Server returned status ${response.status}`;
    try {
      const errorJson: ApiErrorResponse = await response.json();
      if (errorJson.detail) {
        errorDetail = errorJson.detail;
      } else if (errorJson.error) {
        errorDetail = errorJson.error;
      }
    } catch {
      // Non-JSON response body
    }

    if (response.status === 422) {
      throw new Error(`Validation Error: ${errorDetail}`);
    }

    if (response.status >= 500) {
      throw new Error(`AssistIQ could not complete this request: ${errorDetail}`);
    }

    throw new Error(`API Error (${response.status}): ${errorDetail}`);
  }

  const data: AssistResponse = await response.json();
  return data;
}

export function getApiBaseUrl(): string {
  return API_BASE_URL;
}
