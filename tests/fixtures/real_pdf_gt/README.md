Human-inspected ground truth for real PYQ PDFs.

Each file is named after the PDF basename (or a stable hash) and contains:

```json
{
  "filename": "example.pdf",
  "subject": "...",
  "university": "...",
  "semester": "7",
  "exam_year": 2024,
  "exam_session": "May",
  "pages": 2,
  "genuine_question_ids": ["Q1(a)", "Q1(b)"],
  "notes": "inspected YYYY-MM-DD"
}
```

Do not copy these IDs into `rag/` production code.

One JSON per unique university paper (hash-deduped). Copies under
`data/uploads/` share a GT file via `filename`. Synthetic `proof_pyq`,
`stress_*`, and `test_agg_*` fixtures are not real PYQs and are covered
by `tests/test_universal_structure_matrix.py` / `scratch/stress_matrix.py`.
