import json
from pathlib import Path

full = json.loads(Path(r"D:\pyqrag\scratch\restore_points\20260823\full_universe_evaluation.json").read_text(encoding="utf-8"))

keys = [
    "be_computer-engineering_semester-4_2023_may_operating-systemrev-2019-c-scheme.pdf",
    "be_computer-engineering_semester-7_2023_may_dloc-iii-natural-language-processing-rev-2019-c-scheme.pdf",
    "be_computer-engineering_semester-7_2024_december_big-data-analysis-rev-2019-c-scheme.pdf",
    "be_computer-engineering_semester-7_2025_may_big-data-analysis-rev-2019-c-scheme.pdf",
    "be_computer-engineering_semester-5_2025_may_internet-programmingrev-2019-c-scheme.pdf",
    "be_computer-engineering_semester-5_2022_december_software-engineeringrev-2019-c-scheme.pdf",
    "be_computer-engineering_semester-5_2022_may_software-engineeringrev-2019-c-scheme.pdf",
    "be_computer-engineering_semester-4_2023_december_operating-systemrev-2019-c-scheme.pdf",
    "be_computer-engineering_semester-5_2023_december_software-engineeringrev-2019-c-scheme.pdf",
    "AIDSAIML-SEM-7_2024_compressed.pdf",
    "be_computer-engineering_semester-4_2024_december_operating-systemrev-2019-c-scheme.pdf",
    "be_computer-engineering_semester-5_2023_may_internet-programmingrev-2019-c-scheme.pdf",
]
want = {k.lower() for k in keys}
for p in full["papers"]:
    if p["pdf"].lower() not in want:
        continue
    print("=" * 80)
    print(p["pdf"])
    print("status", p["extraction_status"], "conf", p["extraction_confidence"], "n", len(p["canonical_extracted_questions"]))
    print("ids", p["canonical_extracted_questions"])
    print("genuine_auth", p.get("genuine_source_authority"))
    print("genuine", p.get("genuine_source_questions"))
    print("sys_genuine", p.get("system_genuine_markers"))
    print("sys_missing", p.get("system_missing_markers"))
    print("missing", p.get("missing_genuine_markers"))
    print("fabricated", p.get("fabricated_markers"))
    print("failures", p.get("failures"))
    print("ocr", p.get("ocr_noise"))
    print("pages", p.get("pages"), "rep", p.get("representation"))
    for rec in p.get("extracted_records") or []:
        print("  ", rec.get("question_id"), "marks", rec.get("marks"), "pages", rec.get("pages"), "xpage", rec.get("cross_page_merged"), "vis", bool(rec.get("visual_appendix")), rec.get("exact_text", "")[:140])

print("\n\n=== EXACT GROUPS / INTEL FAILURES ===")
for b in full["intelligence"]:
    print("\n##", b["family"])
    for f in b.get("failures") or []:
        print(" ", f)
    for g in b.get("exact_repeat_false_positives") or []:
        print(" EFP", g)
    # print original texts from planner? not stored. print exact_repeats if present? we didn't store full groups.
    print(" planner titles:")
    for item in b.get("study_planner") or []:
        src = item.get("source_questions") or []
        print("  ", item.get("priority_score"), item.get("title"), "src", len(src), "why", (item.get("why") or "")[:100])
        if src:
            print("    first", {k: src[0].get(k) for k in ("question_id", "year", "marks", "relationship")})
            print("    text", (src[0].get("exact_text") or "")[:160])
