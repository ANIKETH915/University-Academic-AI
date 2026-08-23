import json
from pathlib import Path

p = Path(r"D:\pyqrag\scratch\restore_points\20260823\failure_taxonomy_report.json")
d = json.loads(p.read_text(encoding="utf-8"))
print("=== PAPERS ===")
for x in d["papers"]:
    miss = x["missing_genuine_markers"]
    kinds = ",".join(sorted({f["kind"] for f in x["failures"]}))
    print(
        f"{x['family'][:24]:24} {x['year']} n={len(x['canonical_extracted_questions']):2} "
        f"conf={x['extraction_confidence']} marks0={x['marks_zero_count']} "
        f"gt={int(bool(x['has_ground_truth']))} {x['extraction_status']:8} "
        f"fail={x['failure_count']} {kinds} miss={miss} "
        f"bound={x['wrong_question_boundaries'][:4]} xpage={x['cross_page_errors']} "
        f"{x['pdf'][:64]}"
    )
print()
print("=== INTEL ===")
for b in d["intelligence"]:
    print(
        b["family"],
        "papers",
        b["papers_analyzed"],
        "q",
        b["questions_analyzed"],
        "exact",
        b["exact_repeat_count"],
        "sem",
        b["semantic_repeat_count"],
        "rel",
        b["related_topic_count"],
        "topics",
        b["topic_count"],
    )
    print(
        "  fp/fn",
        len(b["exact_repeat_false_positives"]),
        len(b["exact_repeat_false_negatives"]),
        len(b["semantic_repeat_false_positives"]),
        len(b["semantic_repeat_false_negatives"]),
        "garbage",
        b["garbage_topics"],
        "dup",
        b["duplicate_groups"],
        "self",
        b["self_groups"],
    )
    for g in (b["exact_repeat_false_positives"] or [])[:4]:
        print("   EFP", g)
    for g in (b["semantic_repeat_false_positives"] or [])[:4]:
        print("   SFP", {k: str(v)[:100] for k, v in g.items()})
    for item in (b["study_planner"] or [])[:4]:
        print(
            "   PLAN",
            item.get("priority_score"),
            item.get("title"),
            "yrs",
            item.get("years"),
            item.get("recurrence"),
            item.get("syllabus_mapping"),
            (item.get("why") or "")[:140],
        )
print()
print("contam", d.get("contamination"))
print("missing gt", d.get("gt_pdfs_missing_on_disk"))
print("taxonomy", json.dumps(d["taxonomy"], indent=2))
