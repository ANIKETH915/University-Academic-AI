"""Diagnostic: dump OCR layout classification for real PYQ PDFs."""
import glob
import re
import sys

import fitz

sys.path.insert(0, r"D:\pyqrag")
from rag.ocr_layout import _MARKS_TAG, _classify, _modal_x, ocr_page_lines
from rag.question_extractor import is_header_or_instruction

patterns = sys.argv[1:] or ["*2024*may*", "*2023*december*"]
for pat in patterns:
    matches = glob.glob(r"D:\pyqrag\data\pyq\deep-learning\\" + pat)
    if not matches:
        print("no match for", pat)
        continue
    doc = fitz.open(matches[0])
    for pno, page in enumerate(doc):
        lines = ocr_page_lines(page)
        body = [
            l
            for l in lines
            if l["text"].strip()
            and not is_header_or_instruction(l["text"])
            and not re.fullmatch(r"[\W_]+", l["text"])
        ]
        mx = _modal_x(
            [
                l["x0"]
                for l in body
                if _classify(l["text"])[0] in ("marker_only", "marker_lead", "parent_sub")
            ]
        )
        print("=" * 25, pat, "page", pno + 1, "marker_x =", mx)
        for l in body:
            kind, parent, sub, _rest = _classify(l["text"])
            marks = bool(_MARKS_TAG.search(l["text"]))
            print(
                "x0=%5d %-12s p=%-5s s=%-4s marks=%-5s | %s"
                % (l["x0"], kind, parent, sub, marks, l["text"][:82])
            )
    doc.close()
