"""
GT-driven real PDF matrix. Paper-specific IDs live only in
tests/fixtures/real_pdf_gt/*.json — never in rag/.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("WORKSPACE_DB_TEST_MODE", "1")
os.environ.setdefault("PYQRAG_TEST_COLLECTION", "pyqrag_pytest_collection")

from rag.dynamic_ingest import DynamicIngestPipeline
from rag.vector_store import VectorStore

GT_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "real_pdf_gt")
SEARCH_ROOTS = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pyq"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "uploads"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "scratch"),
]


def _gt_files():
    return sorted(glob.glob(os.path.join(GT_DIR, "*.json")))


def _find_pdf(filename: str):
    for root in SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        for path in glob.glob(os.path.join(root, "**", filename), recursive=True):
            if "Copy" in os.path.basename(path):
                continue
            return path
    return None


class TestRealPdfMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = VectorStore()
        cls.pipe = DynamicIngestPipeline(vector_store=cls.store)

    def test_gt_matrix_when_present(self):
        gts = _gt_files()
        if not gts:
            self.skipTest("no real_pdf_gt JSON files yet")
        failures = []
        ran = 0
        for gt_path in gts:
            with open(gt_path, encoding="utf-8") as f:
                gt = json.load(f)
            filename = gt["filename"]
            pdf = _find_pdf(filename)
            if not pdf:
                continue
            ran += 1
            ws_id = f"ws-gt-{os.path.splitext(os.path.basename(gt_path))[0][:28]}"
            try:
                self.store.delete_by_workspace(ws_id)
            except Exception:
                pass
            metas = self.pipe.parse_pyq_pdf(
                pdf,
                {
                    "id": ws_id,
                    "subject": gt.get("subject") or "Audit Subject",
                    "university": gt.get("university") or "Audit U",
                    "semester": str(gt.get("semester") or "Unknown"),
                },
            )
            audit = self.pipe.last_pyq_questions_audit or {}
            accepted = audit.get("accepted_questions") or []
            extracted = [m.get("question_id") for m in accepted if m.get("question_id")]
            genuine = gt.get("genuine_question_ids") or []
            missing = sorted(set(genuine) - set(extracted))
            fabricated = sorted(set(extracted) - set(genuine))
            dupes = sorted({i for i in extracted if extracted.count(i) > 1})
            quality = audit.get("extraction_quality")
            if missing or fabricated or dupes or quality not in ("COMPLETE", "RECOVERED") or len(metas) != len(extracted):
                failures.append(
                    f"{filename}: quality={quality} missing={missing} "
                    f"fabricated={fabricated} dupes={dupes} vec={len(metas)} ext={len(extracted)}"
                )
        self.assertTrue(ran, "GT files exist but no matching PDFs were found")
        self.assertEqual(failures, [], "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
