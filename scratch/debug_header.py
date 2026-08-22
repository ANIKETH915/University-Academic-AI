from rag.question_extractor import is_header_or_instruction, INSTRUCTION_PATTERNS, FOOTER_HEADER_TOKENS
import re

text = "Explain early stopping, batch normalization, and data augmentation."
print("header?", is_header_or_instruction(text))
low = text.lower()
for token in FOOTER_HEADER_TOKENS:
    if token in low:
        print("footer token hit:", repr(token))
for pat in INSTRUCTION_PATTERNS:
    if re.search(pat, low):
        print("instruction hit:", pat)
