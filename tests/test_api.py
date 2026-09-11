"""
AssistIQ: Unit & Integration Tests for Phase 5 FastAPI Backend
Tests endpoints, request validation, error handling, CORS, and response contract schemas.
"""

import unittest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.schemas import AssistResponse
from backend.api.dependencies import get_pipeline_runner
from backend.src.generation.generate import assist_customer
from backend.src.generation.llm import MockLLMClient


class TestFastAPIBackend(unittest.TestCase):
    """
    Test suite for AssistIQ FastAPI REST API.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def tearDown(self):
        # Reset dependency overrides after each test
        app.dependency_overrides.clear()

    # 1. GET /health -> 200, {"status": "ok"}
    def test_01_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("timestamp", data)
        self.assertEqual(data["version"], "0.5.0")

    # 2. GET / -> 200, root service info
    def test_02_root_info(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["name"], "AssistIQ API")
        self.assertEqual(data["brand"], "SpotifyCares")
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["docs_url"], "/docs")

    # 3. Empty message -> 422 Validation Error
    def test_03_empty_message_validation_error(self):
        response = self.client.post("/api/v1/assist", json={"message": ""})
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("error", data)
        self.assertIn("cannot be empty", data["detail"])

    # 4. Whitespace-only message -> 422 Validation Error
    def test_04_whitespace_message_validation_error(self):
        response = self.client.post("/api/v1/assist", json={"message": "    \n\t  "})
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("error", data)
        self.assertIn("cannot be empty", data["detail"])

    # 5. Oversized message (>2000 chars) -> 422 Validation Error
    def test_05_oversized_message_validation_error(self):
        long_message = "A" * 2001
        response = self.client.post("/api/v1/assist", json={"message": long_message})
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("exceeds maximum length", data["detail"])

    # 6. Missing message field -> 422 Validation Error
    def test_06_missing_message_field(self):
        response = self.client.post("/api/v1/assist", json={})
        self.assertEqual(response.status_code, 422)

    # 7. Mock pipeline execution works end-to-end
    def test_07_valid_assist_request_mocked_runner(self):
        # Inject deterministic mock pipeline runner
        def mock_runner(customer_message, top_k=5, config=None, escalation_config=None):
            return {
                "customer_message": customer_message,
                "predicted_intent": "playback_and_app_issues",
                "intent_confidence": 0.35,
                "retrieved_cases": [
                    {
                        "case_id": "case_101",
                        "similarity": 0.82,
                        "customer_text": "Song skips on iOS",
                        "support_text": "Restart your phone and reinstall the app.",
                        "rank": 1,
                        "conversation_id": 999
                    }
                ],
                "reply": "Restart your phone and reinstall the Spotify app.",
                "grounding_summary": "Grounded in historical case case_101.",
                "evidence_case_ids": ["case_101"],
                "grounding_status": "grounded",
                "decision": "auto_handle",
                "risk_level": "low",
                "escalation_reason": "Auto-handled because intent is confident and reply is grounded.",
                "primary_rule": "A1",
                "policy_rules_triggered": ["A1"],
                "latency": {
                    "intent_ms": 1.2,
                    "retrieval_ms": 15.4,
                    "generation_ms": 0.1,
                    "escalation_ms": 0.1,
                    "total_ms": 16.8
                }
            }

        app.dependency_overrides[get_pipeline_runner] = lambda: mock_runner

        payload = {"message": "My music skips on iPhone"}
        response = self.client.post("/api/v1/assist", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["message"], "My music skips on iPhone")
        self.assertEqual(data["intent"]["name"], "playback_and_app_issues")
        self.assertEqual(data["intent"]["confidence"], 0.35)
        self.assertEqual(data["reply"]["grounding_status"], "grounded")
        self.assertEqual(data["decision"]["decision"], "auto_handle")
        self.assertEqual(data["decision"]["risk_level"], "low")
        self.assertEqual(len(data["evidence"]), 1)
        self.assertEqual(data["evidence"][0]["case_id"], "case_101")
        self.assertEqual(data["latency"]["total_ms"], 16.8)

    # 8. Complete response validation into Pydantic schema
    def test_08_response_validates_into_pydantic_schema(self):
        def mock_runner(customer_message, top_k=5, config=None, escalation_config=None):
            return {
                "customer_message": customer_message,
                "predicted_intent": "billing_and_payment",
                "intent_confidence": 0.38,
                "retrieved_cases": [
                    {
                        "case_id": "case_16893",
                        "similarity": 0.8968,
                        "customer_text": "I got charged twice",
                        "support_text": "Please DM us your account email so we can check.",
                        "rank": 1,
                        "conversation_id": 12345
                    }
                ],
                "reply": "Please DM us your account email so we can investigate the double charge.",
                "grounding_summary": "Grounded in case_16893.",
                "evidence_case_ids": ["case_16893"],
                "grounding_status": "grounded",
                "decision": "escalate",
                "risk_level": "high",
                "escalation_reason": "Escalated because financial transactions require human investigation.",
                "primary_rule": "E6",
                "policy_rules_triggered": ["E6"],
                "latency": {
                    "intent_ms": 2.1,
                    "retrieval_ms": 18.2,
                    "generation_ms": 0.2,
                    "escalation_ms": 0.1,
                    "total_ms": 20.6
                }
            }

        app.dependency_overrides[get_pipeline_runner] = lambda: mock_runner

        response = self.client.post(
            "/api/v1/assist",
            json={"message": "I was charged twice for Spotify Premium this month."}
        )
        self.assertEqual(response.status_code, 200)

        # Parse directly into Pydantic model - raises exception if schema violates contract
        parsed = AssistResponse.model_validate(response.json())
        self.assertEqual(parsed.intent.name, "billing_and_payment")
        self.assertEqual(parsed.decision.decision, "escalate")
        self.assertEqual(parsed.decision.risk_level, "high")
        self.assertEqual(parsed.decision.primary_rule, "E6")

    # 9. Pipeline exception is converted into a safe 500 error without leaking secrets or traces
    def test_09_pipeline_exception_handled_safely(self):
        def failing_runner(customer_message, top_k=5, config=None, escalation_config=None):
            raise RuntimeError("Database connection secret_key=xyz failed at /var/internal/path")

        app.dependency_overrides[get_pipeline_runner] = lambda: failing_runner

        response = self.client.post(
            "/api/v1/assist",
            json={"message": "Help me with my account"}
        )
        self.assertEqual(response.status_code, 500)
        data = response.json()
        # Must not expose the internal exception message, secret, or path
        self.assertNotIn("secret_key", str(data))
        self.assertNotIn("/var/internal/path", str(data))
        self.assertIn("error", data)

    # 10. CORS headers verification
    def test_10_cors_headers(self):
        # Simulate preflight OPTIONS request from Next.js dev server
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        }
        response = self.client.options("/api/v1/assist", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            "http://localhost:3000"
        )

    # 11. End-to-end integration test with real assist_customer and MockLLMClient
    def test_11_live_integration_with_mock_llm(self):
        # Wraps the real assist_customer with MockLLMClient so it runs offline
        def local_mock_runner(customer_message, top_k=3, config=None, escalation_config=None):
            return assist_customer(
                customer_message=customer_message,
                top_k=top_k,
                config=config,
                llm_client=MockLLMClient(),
                escalation_config=escalation_config
            )

        app.dependency_overrides[get_pipeline_runner] = lambda: local_mock_runner

        response = self.client.post(
            "/api/v1/assist",
            json={"message": "I was charged twice for Spotify Premium this month. Can I get a refund?"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["decision"]["decision"], "escalate")
        self.assertIn(data["decision"]["risk_level"], ["medium", "high"])
        self.assertIn("E6", data["decision"]["policy_rules_triggered"])
        self.assertTrue(len(data["evidence"]) > 0)


if __name__ == "__main__":
    unittest.main()
