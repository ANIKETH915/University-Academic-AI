from pathlib import Path

p = Path("rag/dynamic_ingest.py")
text = p.read_text(encoding="utf-8")
start = text.index("DEFAULT_SUBJECT_SYLLABUS_INDEXES")
end = text.index("class DynamicIngestPipeline")
replacement = '''def get_subject_syllabus_index(subject: str, workspace_id: str = "", vector_store=None) -> Dict[str, Any]:
    """Build syllabus index from uploaded workspace syllabus only (subject-agnostic)."""
    if vector_store and workspace_id:
        return build_syllabus_index_from_workspace(
            vector_store, workspace_id, subject=subject or "Academic Subject"
        )
    return empty_syllabus_index(subject or "Academic Subject")


def map_subquestion_to_syllabus_index(
    question_text: str,
    detected_topics: List[str],
    subject: str,
    syllabus_index: Optional[Dict[str, Any]] = None,
    workspace_id: str = "",
    vector_store=None,
) -> Tuple[Dict[str, str], float]:
    """Map PYQ against uploaded syllabus index; Unmapped when evidence is insufficient."""
    index_data = syllabus_index
    if index_data is None:
        index_data = get_subject_syllabus_index(
            subject, workspace_id=workspace_id, vector_store=vector_store
        )
    return map_question_to_syllabus_index(question_text, detected_topics, index_data)


'''
p.write_text(text[:start] + replacement + text[end:], encoding="utf-8")
print("OK replaced", end - start, "chars")
