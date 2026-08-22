import re
from rag.question_extractor import (
    ACADEMIC_QUESTION_VERBS,
    _SUB_TOKEN,
    extract_questions_from_page_text,
    _should_skip_line,
)

verbs = "|".join(sorted(ACADEMIC_QUESTION_VERBS))
sub_loose = re.compile(rf"^({_SUB_TOKEN})\s+((?:{verbs})\b.*)$", re.I)
line = "b Explain early stopping, batch normalization, and data augmentation."
m = sub_loose.match(line)
print("match", bool(m), m.groups() if m else None)
print("skip", _should_skip_line(line))

sample = """Q3(a) Explain CNN architecture in detail. Suppose, we have input volume
*32*3 for a layer in CNN with 10 filters of size 5*5*3 and stride of 1. Calculate the number of parameters.
b Explain early stopping, batch normalization, and data augmentation.
"""
# print each line decision
for i, ln in enumerate(sample.splitlines()):
    print(i, repr(ln), "skip=", _should_skip_line(ln.strip()) if ln.strip() else None)

a, r = extract_questions_from_page_text(sample, 1, "x.pdf", "ws", year=2023)
print("ids", [q["question_id"] for q in a])
for q in a:
    print(q["question_id"], q["exact_text"][:80])
