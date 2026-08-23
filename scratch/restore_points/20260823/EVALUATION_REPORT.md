# Full-universe evaluation — three-layer failure taxonomy

**Architecture frozen.** Production extraction was not modified during this run.
**System is not complete.** Every failure below is classified; remaining limitations are explicit.

Source JSON: `scratch/restore_points/20260823/full_universe_evaluation.json`

## Scope

- 34 unique real PDFs (deduped by filename; skipped Copy / stress / test_agg / proof_pyq).
- 17 GT fixtures; 13 matched a file on disk; 4 GT PDFs are missing from disk.
- Isolated eval collection `pyqrag_eval_universe`. Family workspaces for intelligence; two-family contamination check.
- `AIDSAIML-SEM-7_2024_compressed.pdf` is an 80-page multi-subject booklet, not a single exam.

GT PDFs missing on disk:

- BDA 2022 December
- NLP 2024 May
- NLP 2024 December
- NLP 2025 May

## Headline

| Metric | Value |
|---|---|
| Unique PDFs | 34 |
| COMPLETE | 27 |
| PARTIAL | 6 |
| FAILED | 1 (OS 2023 May, 0 questions) |
| GT papers on disk | 13 |
| GT ID match (no missing/fabricated vs human GT) | 12 / 13 |
| GT COMPLETE and ID match | 10 / 13 |
| Fabricated IDs vs GT | 0 |
| Duplicate extracted IDs | 0 |
| Cross-workspace contamination | none |

`COMPLETE` without human GT does **not** mean every source question was extracted. Several OS/IP/SE papers reported COMPLETE while marker detection clearly under-counted.

## Reviewed taxonomy (not raw heuristic counts)

Automated OCR-noise flags over-fired on formula/table text. Semantic “false positives” from a recheck that ignored intent bundles and `[IMAGE]` suffixes were discarded after inspection. The counts below are the reviewed classification.

### Extraction

| Kind | Count / papers | Example |
|---|---|---|
| Total extraction collapse | 1 paper | OS 2023 May — 0 questions, confidence 0 |
| Missing genuine IDs (GT) | 2 IDs | NLP 2023 May Q4(a), Q4(b) — cross-page continuation never detected as markers |
| Cross-page annotation miss | 2 papers | NLP 2023 May (missing bodies); BDA 2024 May (GT expects cross-page; IDs present, no merge flag) |
| Garbage ID accepted | 1 ID | OS 2023 Dec `Q6(t)` body is alphanumeric shreds |
| False genuine markers | 4+ IDs | BDA 2024 Dec `Q6(c)`/`Q6(d)` (PARTIAL with full GT); BDA 2025 May `Q2(c)`/`Q2(d)` + `Q2(o)` in representations |
| Under-extraction (markers never found) | 4 papers | OS 2024 Dec (5 IDs), OS 2024 May (5), IP 2023 May (9; Q3/Q4/Q6 absent), SE 2022 May (6) |
| Wrong question boundary | 2+ | OS 2023 Dec Q3(a) swallows Q4; SE 2022 Dec Q2(b) swallows Q3 stem |
| OCR watermark in body | 2 questions | BDA 2024 Dec Q4(b), Q6(b) (`VOITV ANS…`) |
| Marks not recovered | 18 papers | Typical `(5)`/`[10]` left as marks=0 |
| Multi-paper booklet | 1 | AIDSAIML 80 pages → 5 lab-outcome “questions” |

### Canonical reconciliation

| Kind | Detail |
|---|---|
| Proven marker, no body | SE 2022 Dec `Q6(e)`, `Q6(f)` (also in representation as `Q3(b)` unrecovered) |
| Proven marker, no body | IP 2025 May `Q1(a)` |
| Status over-conservative | BDA 2024 Dec and DL 2024 Dec PARTIAL after all GT IDs extracted, because leftover OCR markers (`Q6(c)`/`Q6(d)`) entered the missing set |
| Status over-optimistic | NLP 2023 May COMPLETE because Q4 markers were never seen |

### Intelligence grouping

| Kind | Reviewed result |
|---|---|
| Exact FP | 0 confirmed after reading group members (same DGIM / Girvan-Newman / Hyperledger / gradient-descent questions) |
| Exact FN | 0 |
| Semantic FP | 1 confirmed: Deep Learning “CNN architecture” grouped with “RNN working”. 1 likely: Flajolet–Martin with different streams/moduli |
| Semantic FN | 0 |
| Related-topic errors (exact labeled related) | 0 |
| Self-groups | 0 |
| Duplicate groups | 0 |
| Cross-workspace contamination | none (BDA workspace ∩ Blockchain workspace = empty) |

Image-appendix twins (“Describe Ethereum” vs same + `[IMAGE]`) are **true** repeats, not FPs.

### Topic generation

Garbage or stem-as-topic titles that passed QC:

- Communities Given Social Calculation
- Down Six Constraints / Suppose Input Volume
- React Display Hello
- Abor-Rt Disk Scheduling / File Access Ethods / One Call Calls
- Cohesion Coupling Ypetailed / Coupling And-Cohesi / Princi Tracking
- Out Software Testing vs Software Testing (vs-merge)
- Techniques Test Semantic (lab outcome, not an exam question)

### Study-priority scoring

Planner items always had a WHY string, score components, and source-question traces. Systematic scoring defects:

- Marks component under-weighted: 18 papers with every/most marks=0.
- Year 0 on `os_unseen_paper.pdf` (no year in filename) → “years (0, 2023)”.
- No syllabus uploaded in this eval → every unit is “Syllabus mapping uncertain”.
- Over-clustered topics inflate appearance (OS “Multiprocessor Context Switching vs Context Switching” = 4 appearances in 2025 only).

## Per-PDF extraction record (summary)

Full question lists and texts are in the JSON. Short names below.

| Family | Year | Status | n | GT | Missing | Fab | Conf | Notes |
|---|---|---|---|---|---|---|---|---|
| BDA | 2023 Dec | COMPLETE | 14 | yes | — | — | 0.99 | ID match |
| BDA | 2023 May | COMPLETE | 14 | yes | — | — | 0.99 | ID match |
| BDA | 2024 Dec | PARTIAL | 14 | yes | — | — | 0.875 | All GT IDs; system missing Q6(c)(d); watermark OCR |
| BDA | 2024 May | COMPLETE | 14 | yes | — | — | 0.99 | ID match; no cross-page merge flag |
| BDA | 2025 May | PARTIAL | 15 | no | Q2(c) Q2(d) | — | 0.882 | Extracted Q2(e); Q2(o) in reps |
| Blockchain | 2022–2025 (6) | COMPLETE | 17 each | 1 GT | — | — | 0.99 | Cleanest family |
| DL | 2023 Dec | COMPLETE | 15 | yes | — | — | 0.99 | ID match |
| DL | 2024 Dec | PARTIAL | 15 | yes | — | — | 0.882 | All GT IDs; PARTIAL leftover |
| DL | 2024 May | COMPLETE | 15 | yes | — | — | 0.99 | ID match |
| DL | 2025 May | COMPLETE | 15 | yes | — | — | 0.99 | ID match |
| IP | 2023 Dec | COMPLETE | 13 | no | — | — | 0.98 | |
| IP | 2023 May | COMPLETE | 9 | no | unmarked Q3/Q4/Q6 | — | 0.94 | Under-extraction |
| IP | 2024 Dec | COMPLETE | 13 | no | — | — | 0.98 | |
| IP | 2024 May | COMPLETE | 15 | no | — | — | 0.99 | |
| IP | 2025 May | PARTIAL | 8 | no | Q1(a) | — | 0.889 | Marker seen, body not recovered |
| NLP | 2022 Dec | COMPLETE | 18 | yes | — | — | 0.99 | ID match |
| NLP | 2023 Dec | COMPLETE | 15 | yes | — | — | 0.99 | ID match |
| NLP | 2023 May | COMPLETE | 13 | yes | Q4(a) Q4(b) | — | 0.98 | Cross-page; markers never seen |
| OS | 2023 Dec | COMPLETE | 12 | no | — | Q6(t)* | 0.97 | Garbage ID; Q3 swallows Q4 |
| OS | 2023 May | FAILED | 0 | no | all | — | 0.00 | Total collapse |
| OS | 2024 Dec | COMPLETE | 5 | no | unmarked mid paper | — | 0.90 | Only Q1 a–c, Q4 a–b |
| OS | 2024 May | COMPLETE | 5 | no | unmarked | — | 0.90 | Same shape |
| OS | 2025 May | COMPLETE | 12 | no | — | — | 0.97 | |
| OS | unseen | COMPLETE | 16 | yes | — | — | 0.99 | ID match; year missing |
| SE | 2022 Dec | PARTIAL | 17 | no | Q6(e) Q6(f) | — | 0.895 | Known short-note tail |
| SE | 2022 May | PARTIAL | 6 | no | Q2(c) + unmarked | — | 0.857 | 3-page OCR |
| SE | 2023 Dec | COMPLETE | 15 | no | — | — | 0.99 | |
| SE | 2023 May | COMPLETE | 18 | no | — | — | 0.99 | |
| Unclassified | 2024 booklet | COMPLETE | 5 | no | — | — | 0.90 | Out of contract |

\* `Q6(t)` is a fabricated marker the GT-less audit could not name as fabricated.

## Intelligence + study planner by family

| Family | Papers / Q | Exact / Sem / Related / Topics | Planner top | Issues |
|---|---|---|---|---|
| BDA | 5 / 70 | 8 / 3 / 40 / 48 | Communities… 89.9; MapReduce 83.9 | Topic titles garbled; likely FM semantic FP |
| Blockchain | 6 / 78 | 9 / 4 / 40 / 51 | Double spending 82.4; public/private 79.4 | Clean grouping |
| Deep Learning | 4 / 60 | 8 / 2 / 40 / 47 | Activation functions 71.6 | CNN vs RNN semantic FP |
| Internet Programming | 5 / 58 | 0 / 1 / 40 / 45 | React Hello 70.9; JSX 69.9 | Stem-as-topic; under-extracted papers starve repeats |
| NLP | 3 / 46 | 1 / 1 / 40 / 39 | Porter 54.4 | Missing 2024–25 PDFs; Q4 hole |
| OS | 5 / 48 | 0 / 0 / 37 / 44 | Context switching 52.2 | Failed + 5-question papers; year 0; OCR topics |
| SE | 4 / 54 | 1 / 3 / 40 / 39 | Testing 75.1 | Duplicate vs-merge topics; Q6(e)(f) |
| Booklet | 1 / 5 | 0 / 0 / 0 / 5 | Lab outcomes ~30 | Not an exam |

Study planner fields present on every item: priority_score, signals, source_questions, years, exact/semantic counts, unit, why, explanation, study_band.

## Remaining limitations (do not treat as done)

1. Cross-page recovery still drops questions whose markers never appear as labelled IDs on the continuation page (NLP 2023 May Q4).
2. Dense scanned OS/IP papers can lose whole parent blocks; COMPLETE then lies.
3. Isolated-letter OCR (`t`, `o`, `v`, `c`, `d`) still becomes genuine or accepted IDs.
4. Visual attachment is unjudged without region-level GT; tables can be inlined as `[TABLE]` text.
5. Marks recovery from `(5)` / `[10]` is unreliable after OCR, so Layer 3 under-weights marks.
6. Topic titles can still keep OCR tokens and question stems; `X vs X` merges are now collapsed.
7. Sibling architectures with distinct source acronyms (CNN vs RNN) are no longer SEMANTIC; lowercase-only names still can be.
8. Repeat detection is O(n²); related-topic list is capped at 40.
9. English academic-verb heuristics; no non-English papers in this universe.
10. Multi-paper booklets are out of contract.
11. Four GT papers are not on disk, so the GT matrix is incomplete.
12. Semantic FN/FP without human pair labels remain heuristic.

## Generic fixes applied after this report (no per-PDF patches)

These changes landed only after the 34-PDF taxonomy was written. They generalize to unseen papers:

1. Reject question bodies with almost no academic words and high structural OCR noise (`Q6(t)`-class shreds).
2. Distinct 2–6 letter source acronyms (CNN vs RNN) cannot be a SEMANTIC repeat.
3. Topic titles of the form `X vs X` / `…X vs X` collapse to one side.

Core suite after the change: 78 passed. GT real-PDF matrix: same 3 known failures as before the change (BDA 2024 Dec PARTIAL leftover, DL 2024 Dec PARTIAL leftover, NLP 2023 May missing Q4). Previously passing GT papers did not regress.

## Generic fixes still open (no per-PDF patches)

1. Do not promote isolated-letter leaps (`Qn(t)`, `Qn(o)`, `Qn(v)`) to genuine markers without local labelled evidence.
2. Recover printed marks from `(N)` / `[N]` after OCR normalization.
3. Recover cross-page questions whose markers never appear on the continuation page.
4. Drop question-stem topic titles (`Suppose Input Volume`, `React Display Hello`).
5. Do not emit COMPLETE when whole parent blocks were never marked on a multi-question paper.

Do not patch NLP 2023 May, SE 2022 Dec, or any filename/subject.
