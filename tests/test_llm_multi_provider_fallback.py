"""
Unit tests for multi-provider LLM configuration, fallback chain ordering,
transient vs non-transient error handling, zero-key fallback, and /health sanitization.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

from rag.llm_client import (
    call_llm,
    call_llm_json,
    llm_configured,
    llm_status,
    provider_chain,
    provider_key,
    provider_model,
    _is_transient,
)
from fastapi.testclient import TestClient
from rag.api import app


class TestMultiProviderLLMFallback(unittest.TestCase):
    def setUp(self):
        self.orig_env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.orig_env)

    def test_zero_api_key_configuration(self):
        """When no API keys are present in env, llm_configured is False and chain is empty."""
        for k in ["LLM_PROVIDER", "OPENAI_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "LLM_API_KEY"]:
            os.environ.pop(k, None)

        self.assertFalse(llm_configured())
        self.assertEqual(provider_chain(), [])
        # call_llm returns None immediately without error
        res = call_llm("system", "user")
        self.assertIsNone(res)

    def test_provider_chain_ordering(self):
        """Primary provider goes first, followed by custom LLM_FALLBACK_CHAIN."""
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_API_KEY"] = "fake-gemini-key"
        os.environ["OPENAI_API_KEY"] = "fake-openai-key"
        os.environ["GROQ_API_KEY"] = "fake-groq-key"
        os.environ["LLM_FALLBACK_CHAIN"] = "groq,openai"

        chain = provider_chain()
        self.assertEqual(chain, ["gemini", "groq", "openai"])

    def test_is_transient_error_classification(self):
        """Permanent auth errors fail immediately, while timeouts and rate limits are transient."""
        self.assertTrue(_is_transient(Exception("Connection timeout while reaching endpoint")))
        self.assertTrue(_is_transient(Exception("HTTP 429 Too Many Requests")))
        self.assertTrue(_is_transient(Exception("503 Service Unavailable")))

        # Permanent errors
        self.assertFalse(_is_transient(Exception("401 Unauthorized: Invalid API key provided")))
        self.assertFalse(_is_transient(Exception("403 Forbidden: Invalid_api_key")))
        self.assertFalse(_is_transient(Exception("Model_not_found: gpt-nonexistent")))

    def test_health_endpoint_never_exposes_secrets(self):
        """GET /health provides diagnostic metadata without exposing API keys or tokens."""
        os.environ["LLM_PROVIDER"] = "openai"
        os.environ["OPENAI_API_KEY"] = "sk-proj-super-secret-key-12345"
        os.environ["GEMINI_API_KEY"] = "AIzaSySecretGeminiKey67890"

        client = TestClient(app)
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertIn("status", data)
        self.assertIn("llm", data)
        llm = data["llm"]
        self.assertTrue(llm["configured"])

        # Convert full response to string and verify no secret keys exist
        raw_str = str(data)
        self.assertNotIn("sk-proj-super-secret-key-12345", raw_str)
        self.assertNotIn("AIzaSySecretGeminiKey67890", raw_str)
        self.assertNotIn("API_KEY", raw_str)
        self.assertNotIn("Authorization", raw_str)


if __name__ == "__main__":
    unittest.main()
