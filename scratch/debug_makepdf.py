import os, tempfile, fitz
from rag.question_extractor import extract_questions_from_page_text

def make_pdf(path, pages):
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        y = 40
        for raw_line in text.splitlines():
            while raw_line:
                chunk = raw_line[:95]
                raw_line = raw_line[95:]
                if y > 780:
                    page = doc.new_page()
                    y = 40
                page.insert_text((40, y), chunk, fontsize=9)
                y += 12
    doc.save(path)
    doc.close()

tmpdir = tempfile.mkdtemp()
path = os.path.join(tmpdir, "t.pdf")
make_pdf(path, [
    "UNIVERSITY EXAM 2024\n"
    "Q1(a) What is the significance of Activation Functions in Neural Networks, explain different types Activation functions used in NN.\n"
    "Q1(b) Explain the dropout method and its advantages.\n"
    "Q2(a) Explain CNN architecture in detail.\n"
    "Q2(b) Explain LSTM architecture.\n"
    "Q3(a) Explain Gradient Descent in Deep Learning.\n"
])
doc = fitz.open(path)
text = "\n".join(page.get_text() for page in doc)
doc.close()
print(repr(text))
acc, rej = extract_questions_from_page_text(text, 1, "t.pdf", "ws", year=2024)
print("ids", [q["question_id"] for q in acc])
print("rej", [(r.get("question_id"), r.get("reason"), r.get("raw_text","")[:70]) for r in rej])
