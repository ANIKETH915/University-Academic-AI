from rag.question_extractor import (
    classify_repeat_relationship,
    normalize_question_text,
    compute_text_similarity,
    extract_entities,
    extract_constraints,
    detect_question_type,
    _constraint_overlap,
    _entity_overlap,
)

a = "Explain the dropout method and its advantages."
b = "Explain dropout. How does it solve the problem of overfitting?"
print("type", detect_question_type(a), detect_question_type(b))
print("ent", extract_entities(a), extract_entities(b))
print("cons", extract_constraints(a), extract_constraints(b))
print("ent_ov", _entity_overlap(extract_entities(a), extract_entities(b)))
print("cons_ov", _constraint_overlap(extract_constraints(a), extract_constraints(b)))
n1, n2 = normalize_question_text(a), normalize_question_text(b)
sim = compute_text_similarity(n1, n2)
print("sim", sim)
print(classify_repeat_relationship(sim, n1, n2, a, b))
