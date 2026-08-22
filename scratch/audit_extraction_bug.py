"""Reproduce extraction corruption before fixes."""
import sys
sys.path.insert(0, r"D:\pyqrag")
from rag.question_extractor import extract_questions_from_page_text

sample = """
University of Mumbai
B.E. (Computer Engineering) Sem VII
QP CODE: 12345
Dec-2023 10:30 am Engineering

Q3(a) Explain CNN architecture in detail. Suppose, we have input volume
of 32*32*3 for a layer in CNN with 10 filters of size 5*5*3 and stride of 1.
Calculate the number of parameters.
Q3(b) Explain early stopping, batch normalization, and data augmentation.
"""

accepted, rejected = extract_questions_from_page_text(
    sample, page_num=1, source_file="DL_2023.pdf", workspace_id="audit", year=2023
)
print("=== ACCEPTED ===")
for q in accepted:
    print(f"ID={q['question_id']!r}")
    print(f"  TEXT={q['exact_text'][:120]!r}")
    print(f"  TOPICS={q['detected_topics']}")
print("=== REJECTED ===", len(rejected))
for r in rejected[:5]:
    print(r)
