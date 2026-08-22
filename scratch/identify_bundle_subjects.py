"""Identify the subject of each page in a scanned multi-subject PYQ bundle."""
import io
import json
import re
import sys

import fitz
import pytesseract
from PIL import Image

sys.path.insert(0, r"D:\pyqrag")

PDF = r"D:\pyqrag\data\pyq\deep-learning\AIDSAIML-SEM-7_2024_compressed.pdf"

doc = fitz.open(PDF)
rows = []
for i, page in enumerate(doc):
    rect = page.rect
    header = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + rect.height * 0.16)
    pix = page.get_pixmap(dpi=150, clip=header)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    text = pytesseract.image_to_string(img).replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    m = re.search(r"Paper\s*/\s*Subject Code\s*[:\-]?\s*(\d+)\s*/\s*([^/]{3,60})", text, re.I)
    subject = m.group(2).strip() if m else ""
    code = m.group(1) if m else ""
    rows.append({"page": i + 1, "code": code, "subject": subject, "header": text[:120]})
    print(f"{i+1:3} {code:8} {subject[:45]:45} | {text[:70]}")
doc.close()

out = r"D:\pyqrag\scratch\extraction_audits\bundle_subjects.json"
with open(out, "w", encoding="utf-8") as fh:
    json.dump(rows, fh, indent=2)
print("wrote", out)

subjects = {}
for r in rows:
    key = (r["code"], r["subject"])
    subjects.setdefault(key, []).append(r["page"])
print("\nDISTINCT SUBJECTS:")
for (code, subject), pages in sorted(subjects.items()):
    print(f"  {code:8} {subject[:50]:50} pages={pages}")
