"""
Evaluation-only full-pipeline audit.

Does not change rag/ production logic. Writes incremental JSON so a long
run can be inspected before it finishes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("WORKSPACE_DB_TEST_MODE", "1")
os.environ.setdefault("PYQRAG_TEST_COLLECTION", "pyqrag_eval_universe")

from rag.dynamic_ingest import DynamicIngestPipeline
from rag.pyq_intelligence import PYQIntelligenceEngine, generic_normalize_topic_title
from rag.question_extractor import (
    classify_repeat_relationship_full,
    compute_text_similarity,
    is_instruction_frame_text,
    is_valid_question_id,
    looks_like_ocr_garbage_topic,
    normalize_question_text,
    structural_ocr_noise_ratio,
)
from rag.vector_store import VectorStore

OUT_DIR = ROOT / "scratch" / "restore_points" / "20260823"
OUT_PATH = OUT_DIR / "full_universe_evaluation.json"
GT_DIR = ROOT / "tests" / "fixtures" / "real_pdf_gt"
SEARCH_ROOTS = [ROOT / "data" / "pyq", ROOT / "data" / "uploads", ROOT / "scratch"]
SKIP_NAME_RE = re.compile(
    r"(copy|stress_|test_agg_|proof_pyq|diag_)",
    re.I,
)

# Evaluation labels only — never imported by rag/.
FAMILY_TOKENS = (
    ("internet-programming", "Internet Programming"),
    ("software-engineering", "Software Engineering"),
    ("blockchain", "Blockchain"),
    ("operating-system", "Operating Systems"),
    ("big-data", "Big Data Analytics"),
    ("natural-language", "Natural Language Processing"),
    ("deep-learning", "Deep Learning"),
    ("computer-network", "Computer Networks"),
    ("database", "Database Systems"),
)


def _family(name: str, gt: dict | None = None) -> str:
    if gt and gt.get("subject"):
        return str(gt["subject"])
    low = name.lower()
    for tok, label in FAMILY_TOKENS:
        if tok in low:
            return label
    return "Unclassified"


def _year(name: str) -> int | None:
    m = re.search(r"(20\d{2})", name)
    return int(m.group(1)) if m else None


def _load_gt() -> dict[str, dict]:
    out = {}
    if not GT_DIR.is_dir():
        return out
    for path in GT_DIR.glob("*.json"):
        if path.name == "README.md":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        fn = data.get("filename")
        if fn:
            out[fn.lower()] = data
    return out


def _iter_unique_pdfs():
    seen = {}
    for root in SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.pdf"):
            if SKIP_NAME_RE.search(path.name):
                continue
            if "stress_pdfs" in str(path) or "corrupted" in str(path).lower():
                continue
            key = path.name.lower()
            if key not in seen:
                seen[key] = path
    return [seen[k] for k in sorted(seen)]


def _layer_for_extraction_issue(kind: str, *, in_genuine_markers: bool) -> str:
    if kind in ("missing_genuine_gt", "missing_system_marker"):
        return "canonical_reconciliation" if in_genuine_markers else "extraction"
    if kind in ("fabricated_gt", "invalid_id", "instruction_as_question"):
        return "extraction"
    if kind in ("duplicate_id", "wrong_parent_child", "partial_without_gt_gap"):
        return "canonical_reconciliation"
    if kind in ("cross_page_miss", "table_attachment_miss", "ocr_noise"):
        return "extraction"
    if kind == "wrong_boundary":
        return "extraction"
    return "extraction"


def _audit_one_pdf(
    pipe: DynamicIngestPipeline,
    store: VectorStore,
    pdf: Path,
    gt: dict | None,
    family: str,
    workspace_id: str | None = None,
    reset_workspace: bool = False,
) -> dict:
    ws_id = workspace_id or ("ws-eval-" + hashlib.sha1(pdf.name.encode("utf-8")).hexdigest()[:12])
    if reset_workspace:
        try:
            store.delete_by_workspace(ws_id)
        except Exception:
            pass

    pages = None
    try:
        import fitz
        doc = fitz.open(str(pdf))
        pages = len(doc)
        doc.close()
    except Exception:
        pass

    row = {
        "pdf": pdf.name,
        "path": str(pdf),
        "family": family,
        "year": _year(pdf.name),
        "pages": pages,
        "workspace_id": ws_id,
        "has_ground_truth": bool(gt),
        "genuine_source_questions": (gt or {}).get("genuine_question_ids") or [],
        "genuine_source_authority": "ground_truth" if gt else "system_markers_only",
        "canonical_extracted_questions": [],
        "extracted_records": [],
        "missing_genuine_markers": [],
        "fabricated_markers": [],
        "duplicate_ids": [],
        "wrong_question_boundaries": [],
        "wrong_parent_child_relationships": [],
        "cross_page_errors": [],
        "table_diagram_attachment_errors": [],
        "ocr_noise": [],
        "extraction_confidence": None,
        "extraction_status": "ERROR",
        "vectors": 0,
        "representation": {},
        "failures": [],
        "gt_expect_cross_page": bool((gt or {}).get("expect_cross_page")),
        "gt_expect_tables": bool((gt or {}).get("expect_tables")),
        "gt_expect_diagrams": bool((gt or {}).get("expect_diagrams")),
    }

    try:
        metas = pipe.parse_pyq_pdf(
            str(pdf),
            {
                "id": ws_id,
                "subject": (gt or {}).get("subject") or family,
                "university": (gt or {}).get("university") or "Audit Institution",
                "semester": str((gt or {}).get("semester") or "Unknown"),
            },
        )
    except Exception as exc:
        row["extraction_status"] = f"ERROR: {exc}"
        row["failures"].append({
            "layer": "extraction",
            "kind": "ingest_exception",
            "detail": traceback.format_exc()[-800:],
        })
        return row

    audit = pipe.last_pyq_questions_audit or {}
    accepted = audit.get("accepted_questions") or []
    qsum = audit.get("quality_summary") or {}
    ext_audit = audit.get("extraction_audit") or {}
    ids = [q.get("question_id") for q in accepted if q.get("question_id")]
    genuine_sys = list(qsum.get("genuine_markers") or ext_audit.get("genuine_markers") or [])
    missing_sys = list(qsum.get("missing_questions") or qsum.get("missing_question_candidates") or [])
    if not genuine_sys:
        genuine_sys = list(ext_audit.get("reconciled_questions") or [])

    if not row["genuine_source_questions"]:
        row["genuine_source_questions"] = list(dict.fromkeys(genuine_sys or ids))
        row["genuine_source_authority"] = "system_markers_only"

    gt_ids = set((gt or {}).get("genuine_question_ids") or [])
    ext_set = set(ids)
    if gt_ids:
        missing_gt = sorted(gt_ids - ext_set)
        fabricated_gt = sorted(ext_set - gt_ids)
        row["missing_genuine_markers"] = missing_gt
        row["fabricated_markers"] = fabricated_gt
        sys_markers = set(genuine_sys)
        for mid in missing_gt:
            row["failures"].append({
                "layer": _layer_for_extraction_issue(
                    "missing_genuine_gt", in_genuine_markers=mid in sys_markers or mid in set(missing_sys)
                ),
                "kind": "missing_genuine_marker",
                "detail": mid,
                "in_system_missing_set": mid in set(missing_sys),
            })
        for fid in fabricated_gt:
            row["failures"].append({
                "layer": "extraction",
                "kind": "fabricated_marker",
                "detail": fid,
            })
    else:
        row["missing_genuine_markers"] = missing_sys
        row["fabricated_markers"] = []
        for mid in missing_sys:
            row["failures"].append({
                "layer": "canonical_reconciliation",
                "kind": "system_unrecovered_marker",
                "detail": mid,
                "note": "No human GT; system claims source-proven marker was not extracted.",
            })

    dupes = sorted({i for i in ids if ids.count(i) > 1})
    row["duplicate_ids"] = dupes
    for d in dupes:
        row["failures"].append({"layer": "canonical_reconciliation", "kind": "duplicate_id", "detail": d})

    parents_with_children = {q.get("parent_question") or q.get("parent_id") for q in accepted if q.get("subquestion")}
    for q in accepted:
        qid = q.get("question_id")
        text = (q.get("exact_text") or "").strip()
        rec = {
            "question_id": qid,
            "parent_id": q.get("parent_id") or q.get("parent_question"),
            "subquestion": q.get("subquestion"),
            "exact_text": text[:400],
            "page": q.get("source_page"),
            "pages": q.get("source_pages"),
            "marks": q.get("marks"),
            "cross_page_merged": bool(q.get("cross_page_merged")),
            "visual_appendix": q.get("visual_appendix") or [],
            "grounding_status": q.get("grounding_status"),
            "grounding_score": q.get("grounding_score"),
            "extraction_method": q.get("extraction_method"),
            "under_instruction_parent": bool(q.get("under_instruction_parent")),
        }
        row["extracted_records"].append(rec)

        if qid and not is_valid_question_id(str(qid)):
            row["failures"].append({"layer": "extraction", "kind": "invalid_id", "detail": qid})
        if text and is_instruction_frame_text(text):
            row["wrong_question_boundaries"].append(qid)
            row["failures"].append({
                "layer": "extraction",
                "kind": "instruction_as_question",
                "detail": f"{qid}: {text[:80]}",
            })
        if qid and qid in parents_with_children and not q.get("subquestion"):
            row["wrong_parent_child_relationships"].append(qid)
            row["failures"].append({
                "layer": "canonical_reconciliation",
                "kind": "wrong_parent_child",
                "detail": f"parent container {qid} kept alongside children",
            })
        noise = structural_ocr_noise_ratio(text)
        if noise >= 0.35:
            row["ocr_noise"].append({"question_id": qid, "noise_ratio": noise, "text": text[:120]})
            row["failures"].append({
                "layer": "extraction",
                "kind": "ocr_noise",
                "detail": f"{qid} noise={noise}",
            })
        if text and len(text) < 8 and not q.get("under_instruction_parent"):
            row["wrong_question_boundaries"].append(qid)
            row["failures"].append({
                "layer": "extraction",
                "kind": "wrong_boundary",
                "detail": f"{qid} suspiciously short: {text!r}",
            })

    if row["gt_expect_cross_page"] and pages and pages > 1:
        merged = [q.get("question_id") for q in accepted if q.get("cross_page_merged")]
        multi_page = [q.get("question_id") for q in accepted if len(q.get("source_pages") or []) > 1]
        if not merged and not multi_page:
            row["cross_page_errors"].append("expected_cross_page_but_no_merge")
            row["failures"].append({
                "layer": "extraction",
                "kind": "cross_page_miss",
                "detail": "GT expects cross-page continuation; no merged/multi-page question recorded",
            })

    if row["gt_expect_tables"] or row["gt_expect_diagrams"]:
        attached = [q.get("question_id") for q in accepted if q.get("visual_appendix")]
        if not attached:
            row["table_diagram_attachment_errors"].append("expected_visual_region_but_none_attached")
            row["failures"].append({
                "layer": "extraction",
                "kind": "table_attachment_miss",
                "detail": "GT expects table/diagram; no visual_appendix on any question",
            })

    status = qsum.get("extraction_quality") or audit.get("extraction_quality") or "UNKNOWN"
    row["canonical_extracted_questions"] = ids
    row["extraction_status"] = status
    row["extraction_confidence"] = qsum.get("confidence") or audit.get("question_extraction_confidence")
    row["vectors"] = len(metas or [])
    row["representation"] = ext_audit.get("representation_sources") or {}
    row["system_missing_markers"] = missing_sys
    row["system_genuine_markers"] = genuine_sys
    if status == "PARTIAL" and gt_ids and not (gt_ids - ext_set):
        row["failures"].append({
            "layer": "canonical_reconciliation",
            "kind": "partial_without_gt_gap",
            "detail": f"PARTIAL while all GT IDs extracted; system_missing={missing_sys}",
        })
    return row


def _audit_intelligence(engine: PYQIntelligenceEngine, workspace_id: str, family: str, papers: list[str]) -> dict:
    analysis = engine.get_pyq_analysis(workspace_id=workspace_id, subject=family, include_source_questions=True)
    qs = analysis.get("extracted_questions") or []
    if not qs:
        qs = engine.get_source_questions(workspace_id)
    failures = []

    exact = analysis.get("exact_repeats") or []
    semantic = analysis.get("semantic_repeats") or []
    related = analysis.get("related_topics") or []
    topics = analysis.get("topics") or []

    exact_fp = []
    exact_fn = []
    semantic_fp = []
    semantic_fn = []
    related_errors = []
    garbage_topics = []
    duplicate_groups = []
    self_groups = []

    # Exact FP: members whose safe-normalized text is not essentially identical
    for g in exact:
        originals = g.get("original_questions") or []
        norms = [normalize_question_text(o.get("text") or "") for o in originals]
        norms = [n for n in norms if n]
        if len(set(norms)) > 1:
            # allow near-exact via classifier
            bad = False
            for i in range(len(originals)):
                for j in range(i + 1, len(originals)):
                    n1, n2 = normalize_question_text(originals[i].get("text") or ""), normalize_question_text(originals[j].get("text") or "")
                    sim = 1.0 if n1 == n2 else compute_text_similarity(n1, n2)
                    rel, _, _, _ = classify_repeat_relationship_full(
                        sim, n1, n2, originals[i].get("text") or "", originals[j].get("text") or ""
                    )
                    if rel != "EXACT_REPEAT":
                        bad = True
            if bad:
                exact_fp.append({"group": g.get("display_title") or g.get("exact_text"), "refs": g.get("source_refs")})
                failures.append({"layer": "intelligence_grouping", "kind": "exact_repeat_false_positive", "detail": g.get("source_refs")})

        keys = [(o.get("source_file"), o.get("question_id")) for o in originals]
        if len(keys) != len(set(keys)):
            self_groups.append(g.get("source_refs"))
            failures.append({"layer": "intelligence_grouping", "kind": "self_group", "detail": g.get("source_refs")})
        if len(originals) < 2 and len(g.get("question_ids") or []) < 2:
            self_groups.append(g.get("question_ids"))
            failures.append({"layer": "intelligence_grouping", "kind": "self_group", "detail": g.get("question_ids")})

    # Exact FN: identical normalized text across different papers, not grouped
    by_norm = defaultdict(list)
    for q in qs:
        n = q.get("normalized_text") or normalize_question_text(q.get("exact_text") or "")
        if n:
            by_norm[n].append(q)
    grouped_exact_keys = set()
    for g in exact:
        for o in g.get("original_questions") or []:
            grouped_exact_keys.add((o.get("source_file"), o.get("question_id")))
        for qid, ref in zip(g.get("question_ids") or [], g.get("source_refs") or []):
            grouped_exact_keys.add((None, qid))
    for n, members in by_norm.items():
        files = {m.get("source_file") for m in members}
        if len(members) > 1 and len(files) > 1:
            if not any((m.get("source_file"), m.get("question_id")) in grouped_exact_keys for m in members):
                exact_fn.append({
                    "normalized": n[:120],
                    "refs": [f"{m.get('year')} {m.get('question_id')}" for m in members],
                })
                failures.append({
                    "layer": "intelligence_grouping",
                    "kind": "exact_repeat_false_negative",
                    "detail": [f"{m.get('source_file')}:{m.get('question_id')}" for m in members],
                })

    for g in semantic:
        originals = g.get("original_questions") or []
        for i in range(len(originals)):
            for j in range(i + 1, len(originals)):
                t1, t2 = originals[i].get("text") or "", originals[j].get("text") or ""
                n1, n2 = normalize_question_text(t1), normalize_question_text(t2)
                sim = compute_text_similarity(n1, n2)
                rel, _, _, reason = classify_repeat_relationship_full(sim, n1, n2, t1, t2)
                if rel in ("RELATED_TOPIC", "DIFFERENT"):
                    semantic_fp.append({"q1": t1[:100], "q2": t2[:100], "recheck": rel, "reason": reason})
                    failures.append({
                        "layer": "intelligence_grouping",
                        "kind": "semantic_repeat_false_positive",
                        "detail": reason,
                    })
        keys = [(o.get("source_file"), o.get("question_id")) for o in originals]
        if len(keys) != len(set(keys)):
            self_groups.append(g.get("source_refs"))
            failures.append({"layer": "intelligence_grouping", "kind": "self_group", "detail": "semantic self-group"})

    # Semantic FN: high-confidence classifier SEMANTIC not grouped and not exact
    semantic_keys = set()
    for g in semantic:
        for o in g.get("original_questions") or []:
            semantic_keys.add((o.get("source_file"), o.get("question_id")))
    n = len(qs)
    for i in range(n):
        for j in range(i + 1, n):
            q1, q2 = qs[i], qs[j]
            if q1.get("source_file") == q2.get("source_file"):
                continue
            k1 = (q1.get("source_file"), q1.get("question_id"))
            k2 = (q2.get("source_file"), q2.get("question_id"))
            if k1 in grouped_exact_keys or k2 in grouped_exact_keys:
                continue
            if k1 in semantic_keys and k2 in semantic_keys:
                continue
            n1 = q1.get("normalized_text") or normalize_question_text(q1.get("exact_text") or "")
            n2 = q2.get("normalized_text") or normalize_question_text(q2.get("exact_text") or "")
            if not n1 or n1 == n2:
                continue
            sim = compute_text_similarity(n1, n2)
            if sim < 0.55:
                continue
            rel, _, conf, _ = classify_repeat_relationship_full(
                sim, n1, n2, q1.get("exact_text") or "", q2.get("exact_text") or ""
            )
            if rel == "SEMANTIC_REPEAT" and conf >= 0.75:
                if len(semantic_fn) < 25:
                    semantic_fn.append({
                        "q1": q1.get("question_id"),
                        "q2": q2.get("question_id"),
                        "sim": round(sim, 3),
                        "conf": conf,
                    })
                    failures.append({
                        "layer": "intelligence_grouping",
                        "kind": "semantic_repeat_false_negative",
                        "detail": f"{k1} ~ {k2} sim={sim:.2f}",
                    })

    for pair in related:
        t1 = ((pair.get("q1") or {}).get("text") or "")
        t2 = ((pair.get("q2") or {}).get("text") or "")
        if normalize_question_text(t1) and normalize_question_text(t1) == normalize_question_text(t2):
            related_errors.append({"issue": "related_but_exact", "q1": t1[:80], "q2": t2[:80]})
            failures.append({"layer": "intelligence_grouping", "kind": "related_topic_error", "detail": "exact pair labeled related"})

    seen_topic = set()
    for t in topics:
        name = t.get("topic_name") or ""
        if looks_like_ocr_garbage_topic(name) or not generic_normalize_topic_title(name):
            garbage_topics.append(name)
            failures.append({"layer": "topic_generation", "kind": "garbage_topic", "detail": name})
        key = name.strip().lower()
        if key in seen_topic:
            duplicate_groups.append(name)
            failures.append({"layer": "intelligence_grouping", "kind": "duplicate_group", "detail": name})
        seen_topic.add(key)

    planner_items = []
    for item in (analysis.get("recommended_study_plan") or []):
        planner_items.append({
            "title": item.get("title"),
            "priority_score": item.get("priority_score"),
            "score_components": item.get("signals") or {},
            "source_questions": item.get("source_questions") or [],
            "years": item.get("years") or [],
            "recurrence": {
                "exact": item.get("exact_repeat_count"),
                "semantic": item.get("semantic_repeat_count"),
            },
            "marks": item.get("typical_marks") or item.get("max_marks") or item.get("sample_question"),
            "marks_range": {
                "typical": item.get("typical_marks"),
                "from_sources": [sq.get("marks") for sq in (item.get("source_questions") or []) if sq.get("marks") is not None],
            },
            "syllabus_mapping": item.get("unit"),
            "why": item.get("why"),
            "explanation": item.get("explanation") or [],
            "study_band": item.get("study_band"),
        })
        if not (item.get("source_questions") or []):
            failures.append({
                "layer": "study_priority_scoring",
                "kind": "priority_without_source",
                "detail": item.get("title"),
            })
        if not (item.get("why") or item.get("explanation")):
            failures.append({
                "layer": "study_priority_scoring",
                "kind": "missing_why",
                "detail": item.get("title"),
            })
        unit = item.get("unit") or ""
        if unit and re.fullmatch(r"Module\s+[123]\b", str(unit), re.I) and "uncertain" not in str(unit).lower():
            # Only flag if no syllabus was uploaded — dynamic modules are allowed.
            pass

    return {
        "family": family,
        "workspace_id": workspace_id,
        "papers": papers,
        "available": analysis.get("available"),
        "extraction_incomplete": analysis.get("extraction_incomplete"),
        "questions_analyzed": analysis.get("total_valid_questions"),
        "papers_analyzed": analysis.get("total_papers"),
        "years_covered": analysis.get("years_covered"),
        "exact_repeat_count": analysis.get("exact_repeat_count"),
        "semantic_repeat_count": analysis.get("semantic_repeat_count"),
        "related_topic_count": len(related),
        "topic_count": len(topics),
        "exact_repeat_false_positives": exact_fp,
        "exact_repeat_false_negatives": exact_fn,
        "semantic_repeat_false_positives": semantic_fp,
        "semantic_repeat_false_negatives": semantic_fn,
        "related_topic_errors": related_errors,
        "garbage_topics": garbage_topics,
        "duplicate_groups": duplicate_groups,
        "self_groups": self_groups,
        "study_planner": planner_items,
        "module_wise_priority": analysis.get("module_wise_priority") or [],
        "prediction_notice": analysis.get("prediction_notice"),
        "failures": failures,
    }


def _contamination_check(engine: PYQIntelligenceEngine, ws_a: str, ws_b: str) -> dict:
    a = {f"{q.get('source_file')}:{q.get('question_id')}" for q in engine.get_source_questions(ws_a)}
    b = {f"{q.get('source_file')}:{q.get('question_id')}" for q in engine.get_source_questions(ws_b)}
    leak = sorted(a & b)
    return {
        "workspace_a": ws_a,
        "workspace_b": ws_b,
        "overlap": leak,
        "contaminated": bool(leak),
        "failures": (
            [{"layer": "intelligence_grouping", "kind": "cross_workspace_contamination", "detail": leak}]
            if leak else []
        ),
    }


def _save(report: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def main() -> dict:
    gts = _load_gt()
    pdfs = _iter_unique_pdfs()
    missing_gt_pdfs = sorted(
        fn for fn in (d.get("filename") for d in gts.values() if d.get("filename"))
        if not any(p.name.lower() == str(fn).lower() for p in pdfs)
    )
    store = VectorStore()
    pipe = DynamicIngestPipeline(vector_store=store)
    engine = PYQIntelligenceEngine(vector_store=store)

    report = {
        "frozen_architecture": True,
        "production_code_modified": False,
        "pdf_count": len(pdfs),
        "gt_count": len(gts),
        "gt_pdfs_missing_on_disk": missing_gt_pdfs,
        "papers": [],
        "intelligence": [],
        "contamination": {},
        "taxonomy": {},
        "limitations": [],
    }
    if OUT_PATH.exists():
        try:
            report = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    done = {p.get("pdf") for p in report.get("papers") or []}
    family_ws: dict[str, str] = {}
    by_family: dict[str, list[Path]] = defaultdict(list)
    for pdf in pdfs:
        by_family[_family(pdf.name, gts.get(pdf.name.lower()))].append(pdf)

    intelligence = list(report.get("intelligence") or [])
    intel_done = {b.get("family") for b in intelligence}

    for family, fam_pdfs in sorted(by_family.items()):
        isolated = family == "Unclassified"
        fam_ws = (
            "ws-eval-unclassified"
            if isolated
            else "ws-eval-fam-" + hashlib.sha1(family.encode()).hexdigest()[:10]
        )
        family_ws[family] = fam_ws
        need_intel = family not in intel_done and not isolated
        if need_intel:
            try:
                store.delete_by_workspace(fam_ws)
            except Exception:
                pass

        for pdf in fam_pdfs:
            gt = gts.get(pdf.name.lower())
            if pdf.name in done and not need_intel and not isolated:
                print(f"[SKIP] {pdf.name}", flush=True)
                continue
            if pdf.name in done and need_intel:
                print(f"[REINGEST] {pdf.name} family={family}", flush=True)
                try:
                    pipe.parse_pyq_pdf(
                        str(pdf),
                        {
                            "id": fam_ws,
                            "subject": (gt or {}).get("subject") or family,
                            "university": (gt or {}).get("university") or "Audit Institution",
                            "semester": str((gt or {}).get("semester") or "Unknown"),
                        },
                    )
                except Exception as exc:
                    print(f"  reingest failed: {exc}", flush=True)
                continue
            print(f"[EVAL] {pdf.name} family={family}", flush=True)
            if isolated:
                ws_id = "ws-eval-u-" + hashlib.sha1(pdf.name.encode()).hexdigest()[:12]
                row = _audit_one_pdf(pipe, store, pdf, gt, family, workspace_id=ws_id, reset_workspace=True)
            else:
                row = _audit_one_pdf(
                    pipe, store, pdf, gt, family, workspace_id=fam_ws, reset_workspace=False
                )
            row["family_workspace_id"] = row["workspace_id"]
            row["ingested_for_intelligence"] = not str(row["extraction_status"]).startswith("ERROR")
            report.setdefault("papers", []).append(row)
            done.add(pdf.name)
            _save(report)
            print(
                f"  status={row['extraction_status']} extracted={len(row['canonical_extracted_questions'])} "
                f"missing={row['missing_genuine_markers']} fabricated={row['fabricated_markers']} "
                f"failures={len(row['failures'])}",
                flush=True,
            )

        if family in intel_done:
            continue
        papers_in_family = [p.name for p in fam_pdfs]
        if isolated:
            # Single-paper intelligence only; mixed unclassified PDFs are not merged.
            for pdf in fam_pdfs:
                uws = "ws-eval-u-" + hashlib.sha1(pdf.name.encode()).hexdigest()[:12]
                intel = _audit_intelligence(engine, uws, f"Unclassified:{pdf.name}", [pdf.name])
                intelligence.append(intel)
        else:
            intel = _audit_intelligence(engine, fam_ws, family, papers_in_family)
            intelligence.append(intel)
            print(
                f"[INTEL] {family} papers={intel.get('papers_analyzed')} q={intel.get('questions_analyzed')} "
                f"exact_fp={len(intel.get('exact_repeat_false_positives') or [])} "
                f"fail={len(intel.get('failures') or [])}",
                flush=True,
            )
        report["intelligence"] = intelligence
        _save(report)

    # Cross-workspace contamination: first two distinct families
    fams = [f for f in family_ws if f != "Unclassified"]
    if len(fams) >= 2:
        report["contamination"] = _contamination_check(engine, family_ws[fams[0]], family_ws[fams[1]])
    else:
        report["contamination"] = {"contaminated": False, "note": "fewer than two families"}

    taxonomy = defaultdict(lambda: defaultdict(int))
    for row in report.get("papers") or []:
        for f in row.get("failures") or []:
            taxonomy[f.get("layer") or "unknown"][f.get("kind") or "unknown"] += 1
    for block in report.get("intelligence") or []:
        for f in block.get("failures") or []:
            taxonomy[f.get("layer") or "unknown"][f.get("kind") or "unknown"] += 1
    for f in (report.get("contamination") or {}).get("failures") or []:
        taxonomy[f.get("layer") or "unknown"][f.get("kind") or "unknown"] += 1
    report["taxonomy"] = {k: dict(v) for k, v in taxonomy.items()}

    papers = report.get("papers") or []
    with_gt = [p for p in papers if p.get("has_ground_truth")]
    perfect_gt = [
        p for p in with_gt
        if not p.get("missing_genuine_markers")
        and not p.get("fabricated_markers")
        and p.get("extraction_status") in ("COMPLETE", "RECOVERED")
    ]
    report["summary"] = {
        "unique_pdfs": len(papers),
        "gt_papers": len(with_gt),
        "gt_perfect": len(perfect_gt),
        "complete": sum(1 for p in papers if p.get("extraction_status") == "COMPLETE"),
        "partial": sum(1 for p in papers if p.get("extraction_status") == "PARTIAL"),
        "failed": sum(1 for p in papers if str(p.get("extraction_status") or "").startswith(("FAIL", "ERROR"))),
        "total_extraction_failures": sum(len(p.get("failures") or []) for p in papers),
        "total_intelligence_failures": sum(len(b.get("failures") or []) for b in report.get("intelligence") or []),
        "contaminated": bool((report.get("contamination") or {}).get("contaminated")),
    }
    report["limitations"] = [
        "PDFs without human GT use system markers as the genuine set; missing/fabricated vs human source is unknown.",
        "Semantic false negatives are heuristic (high similarity + classifier SEMANTIC not grouped), not labeled exam intent.",
        "Table/diagram errors are only judged when GT expect_tables/expect_diagrams is set.",
        "Synthetic stress/test_agg/proof PDFs were excluded as non-real examinations.",
        "Production extraction logic was not modified during this evaluation run.",
    ]
    report["production_code_modified"] = False
    _save(report)
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote {OUT_PATH}")
    return report


if __name__ == "__main__":
    main()
