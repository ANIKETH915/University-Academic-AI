from rag.question_extractor import extract_questions_from_page_text, classify_repeat_relationship, normalize_question_text, compute_text_similarity

sample = """Q3(a) Explain CNN architecture in detail. Suppose, we have input volume
*32*3 for a layer in CNN with 10 filters of size 5*5*3 and stride of 1. Calculate the number of parameters.
b Explain early stopping, batch normalization, and data augmentation.
Q2(6)
Dec-2023 10:30 am Engineering Artificial Intelligence
"""
a, r = extract_questions_from_page_text(sample, 1, "x.pdf", "ws", year=2023)
print("ids", [q["question_id"] for q in a])
print("rej", [(x.get("question_id"), x.get("reason"), x.get("raw_text", "")[:50]) for x in r])

a = "Explain the dropout method and its advantages."
b = "Explain dropout. How does it solve the problem of overfitting?"
n1, n2 = normalize_question_text(a), normalize_question_text(b)
sim = compute_text_similarity(n1, n2)
print("dropout", classify_repeat_relationship(sim, n1, n2, a, b))
