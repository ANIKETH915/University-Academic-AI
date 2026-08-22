import os
import fitz
from rag.dynamic_ingest import DynamicIngestPipeline, filter_noise_lines
from rag.question_extractor import extract_questions_from_page_text

p = r"d:/pyqrag/data/pyq/deep-learning"
files = [f for f in os.listdir(p) if f.endswith(".pdf") and "Copy" not in f and "compressed" not in f]
path = os.path.join(p, files[0])
doc = fitz.open(path)
raw = doc[0].get_text()
doc.close()
print("raw len", len(raw))
cleaned = filter_noise_lines(raw)
print("cleaned len", len(cleaned), "preview", repr(cleaned[:300]))
pipe = DynamicIngestPipeline()
ok, reason, metrics = pipe.validate_text_quality(cleaned)
print("page valid", ok, reason, metrics)
acc, rej = extract_questions_from_page_text(cleaned or raw, 1, files[0], "ws", year=2023)
print("accepted", len(acc), "rejected", len(rej))
if acc:
    for q in acc[:3]:
        print(q["question_id"], repr(q["exact_text"][:80]))
