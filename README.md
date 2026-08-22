# Mumbai University Previous Year Question Paper (PYQ) Dataset (2020–2026)

A structured, validated, and traceable dataset of Previous Year Question Papers (PYQs) for the University of Mumbai (MU), covering examinations from **2020 through 2026**.

> [!NOTE]
> This repository contains the extracted dataset, metadata, validation reports, raw source PDFs, and reusable processing scripts. It serves as the primary data foundation for Mumbai University Academic RAG systems.

---

## Repository Structure

```text
.
├── README.md                           # Main repository documentation
├── dataset/
│   ├── mu_pyq_questions.jsonl          # Primary JSONL structured dataset
│   ├── mu_pyq_questions.csv            # Tabular CSV export of questions
│   ├── papers_metadata.json            # Catalog of raw PDFs and source URLs
│   └── coverage_matrix.csv             # Coverage Matrix (FOUND vs NOT_FOUND)
├── raw_papers/
│   └── MU_<YEAR>_<SESSION>_<SEM>_<BRANCH>_<SUBJECT>.pdf # Verified raw PDFs
├── validation/
│   ├── validation_report.md            # Markdown executive audit summary
│   ├── validation_report.json          # Machine-readable validation metrics
│   ├── missing_papers.csv              # Matrix records for unavailable papers
│   ├── missing_questions.csv           # Question sequence gaps detected
│   ├── duplicate_papers.csv            # Duplicate PDF hashes catalog
│   ├── questions_needing_review.csv    # Flagged records needing review
│   ├── ocr_issues.csv                  # Character encoding/OCR artifacts
│   └── metadata_issues.csv             # Missing marks or metadata entries
└── scripts/
    ├── collect.py                      # Data collection & downloading script
    ├── extract.py                      # PDF parsing & question extraction script
    ├── label.py                        # NEP, unit, and type labeling engine
    ├── validate.py                     # Dataset validation & audit engine
    └── README.md                       # Documentation for pipeline scripts
```

---

## Primary Dataset Schema (`dataset/mu_pyq_questions.jsonl`)

Each line contains a JSON object formatted as follows:

```json
{
  "question_id": "PAPER_2024_MAY_SEM5_CMPN_COMPUTER_NETWORK_Q1_c",
  "university": "University of Mumbai",
  "year": 2024,
  "semester": "Semester 5",
  "branch": "Computer Engineering",
  "subject": "Computer Network",
  "subject_code": "31923",
  "exam_session": "May",
  "paper_type": "Regular",
  "question_number": "Q1(c)",
  "question_text": "Explain the count to infinity problem in detail.",
  "marks": 5,
  "unit": "Unit 4",
  "question_type": ["EXPLANATION", "CONCEPTUAL", "DESCRIPTIVE"],
  "nep_status": "NEP_2020",
  "source_pdf": "MU_2024_MAY_SEM5_CMPN_COMPUTER_NETWORK.pdf",
  "source_url": "https://www.munotes.in/uploads/question-papers/Engineering/CMPN/Sem%205/May%202024/10038392%20%20Computer%20Network.pdf",
  "source_page": 1,
  "extraction_status": "verified"
}
```

---

## Classification Guidelines

1. **NEP 2020 Classification**:
   - `NEP_2020`: Assigned to papers/questions from 2024–2026 under the new curriculum structure.
   - `NON_NEP`: Assigned to papers/questions from 2020–2023 under Choice Based Credit and Grading Scheme (CBCGS).
   - `UNKNOWN`: Assigned if curriculum evidence is ambiguous.

2. **Question Type Multi-label Tags**:
   - `DEFINITION`, `CONCEPTUAL`, `EXPLANATION`, `NUMERICAL`, `DERIVATION`, `PROGRAMMING`, `COMPARISON`, `DIAGRAM`, `PROBLEM_SOLVING`, `DESCRIPTIVE`, `CASE_STUDY`, `OTHER`.

---

## Dataset Quality Metrics

- **Dataset Quality Rating**: **HIGH**
- **Total Questions Extracted**: 292
- **Verified Questions**: 292
- **NEP 2020 Questions**: 122
- **NON-NEP (CBCGS) Questions**: 170
- **Ready for RAG**: **YES**
