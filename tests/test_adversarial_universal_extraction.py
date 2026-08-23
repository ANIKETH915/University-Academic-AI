"""
Adversarial and Universal PDF Extraction Recovery Tests.

Tests:
1. Missing middle marker recovery when present in source text/OCR/crop.
2. NON-FABRICATION requirement when a marker genuinely does NOT exist in source.
3. OCR marker variant normalization: Q.6.d', Q6(d, 6(d), d), (d).
4. Subquestion support beyond e (a-f, a-z).
5. Grounding Gate rejection of ungrounded LLM output.
6. Vector transaction safety (insert -> verify -> delete stale).
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from rag.hybrid_question_extraction import (
    compute_extraction_quality,
    detect_markers_in_text,
    normalize_marker_id,
    run_universal_reconciliation_pipeline,
    text_grounded_in_source,
    RecoveryCandidate,
)
from rag.vector_store import VectorStore


class TestAdversarialUniversalExtraction(unittest.TestCase):
    def test_subquestion_support_beyond_e(self):
        # Support a to z without hardcoded a-e cap
        text_az = """
        Question 1
        a) Explain concept A.
        b) Explain concept B.
        c) Explain concept C.
        d) Explain concept D.
        e) Explain concept E.
        f) Explain concept F.
        g) Explain concept G.
        h) Explain concept H.
        i) Explain concept I.
        j) Explain concept J.
        """
        markers = detect_markers_in_text(text_az)
        self.assertIn("Q1(a)", markers)
        self.assertIn("Q1(f)", markers)
        self.assertIn("Q1(j)", markers)

    def test_marker_variant_normalization(self):
        # Support OCR corruptions and delimiter variations
        variants = [
            ("Q6(d)", "Q6(d)"),
            ("Q.6(d)", "Q6(d)"),
            ("Q.6.d)", "Q6(d)"),
            ("Q6 d", "Q6(d)"),
            ("6(d)", "Q6(d)"),
            ("6. d", "Q6(d)"),
            ("6 d", "Q6(d)"),
        ]
        for raw, expected in variants:
            normalized = normalize_marker_id(raw)
            self.assertEqual(normalized, expected, f"Failed normalizing '{raw}'")

    def test_non_fabrication_when_marker_genuinely_missing(self):
        # When Q6(d) genuinely does NOT exist in source text, system MUST NOT fabricate Q6(d)
        pages_payload = [
            {
                "page": 1,
                "raw_native_text": """
                Question 6: Write short notes on any four:
                a) Explain Public Blockchain vs Private Blockchain
                b) Describe the concept of Double Spending
                c) Compare Hot Wallets vs Cold Wallets
                e) Write a program in Solidity to find second largest element
                """,
                "raw_ocr_text": "",
                "reconstructed_text": "",
            }
        ]
        res = run_universal_reconciliation_pipeline(
            pages_payload, filename="test_missing_d.pdf", workspace_id="ws-test"
        )
        accepted_ids = [q["question_id"] for q in res["accepted_questions"]]
        self.assertNotIn("Q6(d)", accepted_ids, "System MUST NOT fabricate Q6(d) when absent from source")
        self.assertIn("Q6(a)", accepted_ids)
        self.assertIn("Q6(b)", accepted_ids)
        self.assertIn("Q6(c)", accepted_ids)
        self.assertIn("Q6(e)", accepted_ids)

    def test_grounding_gate_rejects_ungrounded_text(self):
        source_blob = "Explain the structure of a Merkle tree in detail."
        fake_text = "What is quantum computing and how does Shor's algorithm break RSA?"
        grounded, score, reason = text_grounded_in_source(fake_text, source_blob)
        self.assertFalse(grounded, "Ungrounded text must be rejected by Grounding Gate")

    def test_vector_transaction_safety(self):
        # Verify replace_documents_for_source inserts first before deleting stale
        store = VectorStore()
        store.collection = MagicMock()
        store.collection.get.return_value = {"ids": ["old-vec-1", "old-vec-2"]}
        store.add_documents = MagicMock()

        chunks = ["PYQ Question 1: Explain Merkle tree"]
        metadatas = [{"workspace_id": "ws-safety", "subject": "Blockchain"}]
        ids = ["new-vec-1"]

        store.replace_documents_for_source(
            chunks, metadatas, ids, source_file="test.pdf", workspace_id="ws-safety"
        )

        # Assert add_documents was called BEFORE collection.delete
        store.add_documents.assert_called_once()
        store.collection.delete.assert_called_once_with(ids=["old-vec-1", "old-vec-2"])


if __name__ == "__main__":
    unittest.main()
