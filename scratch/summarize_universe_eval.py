"""Post-process full_universe_evaluation.json into a compact taxonomy report."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scratch" / "restore_points" / "20260823" / "full_universe_evaluation.json"
OUT = ROOT / "scratch" / "restore_points" / "20260823" / "failure_taxonomy_report.json"

NOISY_OCR_KINDS = {"ocr_noise"}
ACADEMIC_WORD = re.compile(r"[A-Za-z]{3,}")


def _reclassify_ocr(paper: dict) -> list[dict]:
    kept = []
    for f in paper.get("failures") or []:
        if f.get("kind") != "ocr_noise":
            kept.append(f)
            continue
        qid = (f.get("detail") or "").split()[0]
        rec = next((r for r in paper.get("extracted_records") or [] if r.get("question_id") == qid), None)
        text = (rec or {}).get("exact_text") or ""
        words = ACADEMIC_WORD.findall(text)
        # Keep only if the body is not a readable academic sentence.
        if len(words) < 6 or (len(text) > 80 and len(words) / max(1, len(text.split())) < 0.25):
            kept.append(f)
    return kept


def main() -> dict:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    papers = data.get("papers") or []
    intel = data.get("intelligence") or []

    paper_rows = []
    tax = defaultdict(Counter)
    for p in papers:
        failures = _reclassify_ocr(p)
        for f in failures:
            tax[f.get("layer") or "unknown"][f.get("kind") or "unknown"] += 1
        paper_rows.append({
            "pdf": p.get("pdf"),
            "family": p.get("family"),
            "year": p.get("year"),
            "pages": p.get("pages"),
            "has_ground_truth": p.get("has_ground_truth"),
            "genuine_source_authority": p.get("genuine_source_authority"),
            "genuine_source_questions": p.get("genuine_source_questions") or [],
            "canonical_extracted_questions": p.get("canonical_extracted_questions") or [],
            "missing_genuine_markers": p.get("missing_genuine_markers") or [],
            "fabricated_markers": p.get("fabricated_markers") or [],
            "duplicate_ids": p.get("duplicate_ids") or [],
            "wrong_question_boundaries": p.get("wrong_question_boundaries") or [],
            "wrong_parent_child_relationships": p.get("wrong_parent_child_relationships") or [],
            "cross_page_errors": p.get("cross_page_errors") or [],
            "table_diagram_attachment_errors": p.get("table_diagram_attachment_errors") or [],
            "ocr_noise_flagged": [
                x for x in (p.get("ocr_noise") or [])
                if any(f.get("kind") == "ocr_noise" and (f.get("detail") or "").startswith(x.get("question_id") or "") for f in failures)
            ],
            "ocr_noise_raw_count": len(p.get("ocr_noise") or []),
            "extraction_confidence": p.get("extraction_confidence"),
            "extraction_status": p.get("extraction_status"),
            "vectors": p.get("vectors"),
            "failure_count": len(failures),
            "failures": failures,
            "marks_zero_count": sum(1 for r in (p.get("extracted_records") or []) if not r.get("marks")),
            "marks_present_count": sum(1 for r in (p.get("extracted_records") or []) if r.get("marks")),
        })

    intel_rows = []
    for block in intel:
        for f in block.get("failures") or []:
            tax[f.get("layer") or "unknown"][f.get("kind") or "unknown"] += 1
        planner = []
        for item in block.get("study_planner") or []:
            planner.append({
                "title": item.get("title"),
                "priority_score": item.get("priority_score"),
                "score_components": item.get("score_components") or {},
                "source_questions": [
                    f"{sq.get('year')} {sq.get('question_id')}"
                    for sq in (item.get("source_questions") or [])[:8]
                ],
                "years": item.get("years") or [],
                "recurrence": item.get("recurrence") or {},
                "marks": item.get("marks_range") or item.get("marks"),
                "syllabus_mapping": item.get("syllabus_mapping"),
                "why": item.get("why"),
                "study_band": item.get("study_band"),
            })
        intel_rows.append({
            "family": block.get("family"),
            "papers": block.get("papers"),
            "papers_analyzed": block.get("papers_analyzed"),
            "questions_analyzed": block.get("questions_analyzed"),
            "years_covered": block.get("years_covered"),
            "exact_repeat_count": block.get("exact_repeat_count"),
            "semantic_repeat_count": block.get("semantic_repeat_count"),
            "related_topic_count": block.get("related_topic_count"),
            "topic_count": block.get("topic_count"),
            "exact_repeat_false_positives": block.get("exact_repeat_false_positives") or [],
            "exact_repeat_false_negatives": block.get("exact_repeat_false_negatives") or [],
            "semantic_repeat_false_positives": block.get("semantic_repeat_false_positives") or [],
            "semantic_repeat_false_negatives": block.get("semantic_repeat_false_negatives") or [],
            "related_topic_errors": block.get("related_topic_errors") or [],
            "garbage_topics": block.get("garbage_topics") or [],
            "duplicate_groups": block.get("duplicate_groups") or [],
            "self_groups": block.get("self_groups") or [],
            "study_planner": planner,
            "failures": block.get("failures") or [],
        })

    with_gt = [p for p in paper_rows if p["has_ground_truth"]]
    gt_id_perfect = [
        p for p in with_gt
        if not p["missing_genuine_markers"] and not p["fabricated_markers"]
    ]
    gt_complete = [
        p for p in gt_id_perfect
        if p["extraction_status"] in ("COMPLETE", "RECOVERED")
    ]

    report = {
        "frozen_architecture": True,
        "production_code_modified": False,
        "source": str(SRC),
        "unique_pdfs": len(paper_rows),
        "gt_pdfs_missing_on_disk": data.get("gt_pdfs_missing_on_disk") or [],
        "contamination": data.get("contamination") or {},
        "summary": {
            "unique_pdfs": len(paper_rows),
            "complete": sum(1 for p in paper_rows if p["extraction_status"] == "COMPLETE"),
            "partial": sum(1 for p in paper_rows if p["extraction_status"] == "PARTIAL"),
            "failed": sum(1 for p in paper_rows if str(p["extraction_status"]).startswith(("FAIL", "ERROR"))),
            "gt_papers": len(with_gt),
            "gt_id_match": len(gt_id_perfect),
            "gt_complete_and_id_match": len(gt_complete),
            "papers_with_missing_markers": sum(1 for p in paper_rows if p["missing_genuine_markers"]),
            "papers_with_fabricated_markers": sum(1 for p in paper_rows if p["fabricated_markers"]),
            "papers_with_duplicate_ids": sum(1 for p in paper_rows if p["duplicate_ids"]),
            "papers_with_cross_page_errors": sum(1 for p in paper_rows if p["cross_page_errors"]),
            "papers_with_visual_errors": sum(1 for p in paper_rows if p["table_diagram_attachment_errors"]),
            "papers_with_zero_marks": sum(1 for p in paper_rows if p["marks_present_count"] == 0 and p["canonical_extracted_questions"]),
        },
        "taxonomy": {layer: dict(kinds) for layer, kinds in tax.items()},
        "papers": paper_rows,
        "intelligence": intel_rows,
        "limitations": [
            "Four GT fixtures have no matching PDF on disk (BDA 2022 Dec, NLP 2024 May, NLP 2024 Dec, NLP 2025 May).",
            "AIDSAIML-SEM-7_2024_compressed.pdf is a multi-paper scanned booklet, not a single examination.",
            "OCR-noise ratio from structural_ocr_noise_ratio() over-flags formula/table-heavy but readable questions; taxonomy only keeps sparse-word bodies.",
            "Semantic FN/FP are classifier-agreement checks, not human exam-intent labels.",
            "Table/diagram attachment is only judged when GT expect_tables/expect_diagrams is set.",
            "Marks are frequently 0 even when the paper prints (5)/(10); scoring then under-weights marks.",
            "Production extraction was not modified during this evaluation.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print("TAXONOMY")
    print(json.dumps(report["taxonomy"], indent=2))
    print(f"Wrote {OUT}")
    return report


if __name__ == "__main__":
    main()
