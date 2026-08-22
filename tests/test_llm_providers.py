"""
LLM provider abstraction: resolution, fallback, retries, secret hygiene.

No real network calls are made — provider invocation is patched so that
timeout/rate-limit/malformed-JSON behaviour can be asserted deterministically.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag import llm_client

PROVIDER_ENV = [
    "LLM_PROVIDER",
    "LLM_FALLBACK_PROVIDERS",
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_TIMEOUT_SECONDS",
    "LLM_MAX_RETRIES",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
]


class ProviderEnvCase(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in PROVIDER_ENV}
        for k in PROVIDER_ENV:
            os.environ.pop(k, None)

    def tearDown(self):
        for k in PROVIDER_ENV:
            os.environ.pop(k, None)
            if self._saved.get(k) is not None:
                os.environ[k] = self._saved[k]


class TestProviderResolution(ProviderEnvCase):
    def test_no_keys_means_not_configured(self):
        self.assertFalse(llm_client.llm_configured())
        self.assertEqual(llm_client.provider_chain(), [])

    def test_disabled_provider_short_circuits(self):
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["LLM_PROVIDER"] = "none"
        self.assertFalse(llm_client.llm_configured())

    def test_each_provider_supported(self):
        for provider, env in (
            ("openai", "OPENAI_API_KEY"),
            ("gemini", "GEMINI_API_KEY"),
            ("groq", "GROQ_API_KEY"),
            ("openrouter", "OPENROUTER_API_KEY"),
        ):
            with self.subTest(provider=provider):
                for k in PROVIDER_ENV:
                    os.environ.pop(k, None)
                os.environ[env] = "test-key"
                os.environ["LLM_PROVIDER"] = provider
                self.assertTrue(llm_client.llm_configured())
                self.assertEqual(llm_client.provider_chain()[0], provider)

    def test_default_chain_order(self):
        for env in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
            os.environ[env] = "test-key"
        self.assertEqual(
            llm_client.provider_chain(), ["openai", "gemini", "groq", "openrouter"]
        )

    def test_configurable_chain_order(self):
        for env in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY"):
            os.environ[env] = "test-key"
        os.environ["LLM_PROVIDER"] = "groq"
        os.environ["LLM_FALLBACK_PROVIDERS"] = "gemini,openai"
        self.assertEqual(llm_client.provider_chain(), ["groq", "gemini", "openai"])

    def test_providers_without_keys_are_skipped(self):
        os.environ["GEMINI_API_KEY"] = "test-key"
        os.environ["LLM_PROVIDER"] = "openai"  # no OpenAI key present
        self.assertEqual(llm_client.provider_chain(), ["gemini"])

    def test_model_defaults_and_overrides(self):
        os.environ["GROQ_API_KEY"] = "test-key"
        self.assertTrue(llm_client.provider_model("groq"))
        os.environ["GROQ_MODEL"] = "custom-model-name"
        self.assertEqual(llm_client.provider_model("groq"), "custom-model-name")

    def test_legacy_single_key_config(self):
        os.environ["LLM_PROVIDER"] = "openai"
        os.environ["LLM_API_KEY"] = "legacy-key"
        os.environ["LLM_MODEL"] = "legacy-model"
        self.assertEqual(llm_client.provider_chain(), ["openai"])
        self.assertEqual(llm_client.provider_model("openai"), "legacy-model")


class TestSecretHygiene(ProviderEnvCase):
    def test_status_never_exposes_keys(self):
        os.environ["OPENAI_API_KEY"] = "sk-super-secret-value"
        os.environ["GEMINI_API_KEY"] = "gemini-secret-value"
        status = llm_client.llm_status()
        blob = repr(status)
        self.assertNotIn("sk-super-secret-value", blob)
        self.assertNotIn("gemini-secret-value", blob)
        self.assertTrue(status["configured"])
        self.assertIn("openai", status["provider_chain"])

    def test_failure_trace_never_exposes_keys(self):
        os.environ["OPENAI_API_KEY"] = "sk-super-secret-value"

        def boom(provider, *a, **kw):
            raise RuntimeError("auth failed for key sk-super-secret-value")

        with patch.object(llm_client, "_invoke", side_effect=boom):
            text, attempts = llm_client.call_llm_with_trace("sys", "user")
        self.assertIsNone(text)
        self.assertNotIn("sk-super-secret-value", repr(attempts))


class TestFallbackBehaviour(ProviderEnvCase):
    def setUp(self):
        super().setUp()
        os.environ["OPENAI_API_KEY"] = "k1"
        os.environ["GEMINI_API_KEY"] = "k2"
        os.environ["GROQ_API_KEY"] = "k3"
        os.environ["LLM_MAX_RETRIES"] = "1"

    def test_timeout_falls_through_to_next_provider(self):
        calls = []

        def fake(provider, *a, **kw):
            calls.append(provider)
            if provider == "openai":
                raise TimeoutError("request timed out")
            return "recovered text"

        with patch.object(llm_client, "_invoke", side_effect=fake):
            text, attempts = llm_client.call_llm_with_trace("sys", "user")
        self.assertEqual(text, "recovered text")
        self.assertEqual(calls[0], "openai")
        self.assertIn("gemini", calls)
        self.assertTrue(any(a["outcome"] == "transient_error" for a in attempts))

    def test_rate_limit_retries_then_falls_through(self):
        calls = []

        def fake(provider, *a, **kw):
            calls.append(provider)
            if provider in ("openai", "gemini"):
                raise RuntimeError("429 rate limit exceeded")
            return "groq text"

        with patch.object(llm_client, "_invoke", side_effect=fake):
            text, _attempts = llm_client.call_llm_with_trace("sys", "user")
        self.assertEqual(text, "groq text")
        # retries bounded: 2 attempts each for the two failing providers
        self.assertEqual(calls.count("openai"), 2)
        self.assertEqual(calls.count("gemini"), 2)

    def test_non_transient_error_does_not_retry_same_provider(self):
        calls = []

        def fake(provider, *a, **kw):
            calls.append(provider)
            if provider == "openai":
                raise ValueError("model not found")
            return "ok"

        with patch.object(llm_client, "_invoke", side_effect=fake):
            text, _ = llm_client.call_llm_with_trace("sys", "user")
        self.assertEqual(text, "ok")
        self.assertEqual(calls.count("openai"), 1)

    def test_all_providers_failing_returns_none(self):
        with patch.object(llm_client, "_invoke", side_effect=RuntimeError("connection reset")):
            text, attempts = llm_client.call_llm_with_trace("sys", "user")
        self.assertIsNone(text)
        self.assertTrue(attempts)

    def test_never_loops_forever(self):
        counter = {"n": 0}

        def fake(provider, *a, **kw):
            counter["n"] += 1
            raise RuntimeError("503 service unavailable")

        with patch.object(llm_client, "_invoke", side_effect=fake):
            llm_client.call_llm_with_trace("sys", "user")
        # 3 providers x (1 + 1 retry) = 6 bounded attempts
        self.assertLessEqual(counter["n"], 6)


class TestJsonHandling(ProviderEnvCase):
    def setUp(self):
        super().setUp()
        os.environ["OPENAI_API_KEY"] = "k1"
        os.environ["GEMINI_API_KEY"] = "k2"
        os.environ["LLM_MAX_RETRIES"] = "0"

    def test_malformed_json_falls_through(self):
        def fake(provider, *a, **kw):
            if provider == "openai":
                return "this is not json at all"
            return '{"questions": [{"question_id": "Q1(a)"}]}'

        with patch.object(llm_client, "_invoke", side_effect=fake):
            data = llm_client.call_llm_json("sys", "user")
        self.assertIsInstance(data, dict)
        self.assertEqual(len(data["questions"]), 1)

    def test_fenced_json_parsed(self):
        with patch.object(
            llm_client, "_invoke", return_value='```json\n{"label": "DIFFERENT"}\n```'
        ):
            data = llm_client.call_llm_json("sys", "user")
        self.assertEqual(data["label"], "DIFFERENT")

    def test_bare_array_wrapped(self):
        with patch.object(llm_client, "_invoke", return_value='[{"question_id": "Q2(b)"}]'):
            data = llm_client.call_llm_json("sys", "user")
        self.assertEqual(data["questions"][0]["question_id"], "Q2(b)")

    def test_all_malformed_returns_none(self):
        with patch.object(llm_client, "_invoke", return_value="no json here"):
            self.assertIsNone(llm_client.call_llm_json("sys", "user"))


class TestExtractionSurvivesLlmFailure(ProviderEnvCase):
    """LLM outage must never block or corrupt deterministic extraction."""

    def test_deterministic_extraction_unaffected(self):
        os.environ["OPENAI_API_KEY"] = "k1"
        from rag.hybrid_question_extraction import hybrid_extract_document

        text = (
            "Q1(a) Explain the OSI reference model in detail.\n"
            "Q1(b) Describe TCP congestion control mechanisms.\n"
            "Q2(a) Explain subnetting with a worked example.\n"
        )
        pages = [
            {
                "page": 1,
                "raw_native_text": text,
                "raw_ocr_text": "",
                "reconstructed_text": text,
                "ocr_used": False,
            }
        ]
        with patch("rag.llm_client._invoke", side_effect=TimeoutError("timed out")):
            result = hybrid_extract_document(
                pages, filename="x.pdf", workspace_id="ws", year=0
            )
        ids = [q["question_id"] for q in result["accepted_questions"]]
        self.assertEqual(ids, ["Q1(a)", "Q1(b)", "Q2(a)"])
        self.assertFalse(result["llm_used"])
        self.assertEqual(result["quality"]["extraction_quality"], "COMPLETE")

    def test_llm_cannot_inject_ungrounded_questions(self):
        os.environ["OPENAI_API_KEY"] = "k1"
        from rag.hybrid_question_extraction import hybrid_extract_document

        text = "Q1(a) Explain the OSI reference model in detail.\nQ1(b) Describe TCP congestion control.\n"
        pages = [
            {
                "page": 1,
                "raw_native_text": text,
                "raw_ocr_text": "",
                "reconstructed_text": text,
                "ocr_used": False,
            }
        ]
        invented = (
            '{"questions": [{"question_id": "Q9(a)", "parent_question": "Q9", '
            '"subquestion": "a", "exact_text": "Derive the Shannon capacity theorem '
            'for a noisy quantum channel.", "marks": 10, "source_pages": [1]}]}'
        )
        with patch("rag.llm_client._invoke", return_value=invented):
            result = hybrid_extract_document(
                pages, filename="x.pdf", workspace_id="ws", year=0
            )
        ids = [q["question_id"] for q in result["accepted_questions"]]
        self.assertNotIn("Q9(a)", ids)
        self.assertEqual(ids, ["Q1(a)", "Q1(b)"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
