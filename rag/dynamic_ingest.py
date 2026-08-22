import os
import re
import json
import fitz  # PyMuPDF
from typing import Dict, Any, List, Tuple, Optional
from rag.vector_store import VectorStore
from rag.question_extractor import (
    extract_questions_from_page_text,
    detect_suspicious_alphanumeric_noise,
    calculate_entropy,
    is_header_or_instruction,
    question_structure_score,
)
from rag.syllabus_index import (
    build_syllabus_index_from_workspace,
    empty_syllabus_index,
    map_question_to_syllabus_index,
)


def filter_noise_lines(page_text: str) -> str:
    """Strips administrative headers, instructions, and suspicious OCR garbage lines while preserving valid question text."""
    if not page_text:
        return ""

    cleaned = []
    for line in page_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if is_header_or_instruction(stripped):
            continue
        if detect_suspicious_alphanumeric_noise(stripped):
            continue
        if len(stripped.split()) <= 2 and re.fullmatch(r'[A-Za-z0-9]{5,}', stripped):
            continue
        cleaned.append(stripped)
    return "\n".join(cleaned)

def perform_ocr_page(page: fitz.Page, dpi: int = 150) -> str:
    """Renders PyMuPDF page as an image and executes OCR fallback if pytesseract is available."""
    try:
        import pytesseract
        from PIL import Image
        import io

        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        ocr_text = pytesseract.image_to_string(image)
        return ocr_text.strip()
    except Exception as e:
        print(f"[OCR_FALLBACK_INFO] Pytesseract/Tesseract unavailable: {e}")
        return ""


def get_subject_syllabus_index(subject: str, workspace_id: str = "", vector_store=None) -> Dict[str, Any]:
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


class DynamicIngestPipeline:
    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.store = vector_store or VectorStore()
        self.last_audit_log = []
        self.last_pyq_questions_audit = {
            "accepted_questions": [],
            "rejected_candidates": [],
            "quality_summary": {}
        }

    def validate_text_quality(self, text: str) -> Tuple[bool, str, Dict[str, float]]:
        """
        Strong OCR & Text Quality Gate:
        Evaluates text readability, printable character ratio, alphabetic ratio,
        replacement character count, entropy, unique token ratio, and detects alphanumeric noise.
        """
        if not text or len(text.strip()) < 15:
            return False, "text_too_short", {}

        total_chars = len(text)
        printable_chars = sum(1 for c in text if c.isprintable())
        alpha_chars = sum(1 for c in text if c.isalpha())
        replacement_chars = text.count('\ufffd') + text.count('\ufffe') + text.count('\u0001') + text.count('\u0002') + text.count('Ã©')

        printable_ratio = printable_chars / total_chars
        alpha_ratio = alpha_chars / total_chars
        replacement_ratio = replacement_chars / total_chars

        words = re.findall(r'\b[A-Za-z0-9]{2,}\b', text)
        if not words:
            return False, "no_valid_words_found", {}

        entropy = calculate_entropy(text)
        if detect_suspicious_alphanumeric_noise(text):
            return False, "garbled_ocr_alphanumeric_noise", {"entropy": round(entropy, 2)}

        unique_words = set(w.lower() for w in words)
        unique_token_ratio = len(unique_words) / len(words)
        # Repetitive phrasing across many real subquestions is common; only treat as
        # OCR garbage when uniqueness is extreme AND question boundaries are absent.
        question_markers = len(
            re.findall(
                r"(?:^|\n)\s*(?:Q\.?|Question)?\s*\d+\s*[\.\):\-]?\s*\(?[A-Za-zivx]+\)?",
                text,
                flags=re.IGNORECASE,
            )
        )
        if len(words) >= 8 and unique_token_ratio < 0.22 and question_markers < 2:
            return False, "excessive_repeated_token_noise", {"unique_token_ratio": round(unique_token_ratio, 2)}
        if len(words) >= 8 and unique_token_ratio < 0.12:
            return False, "excessive_repeated_token_noise", {"unique_token_ratio": round(unique_token_ratio, 2)}

        avg_word_length = sum(len(w) for w in words) / len(words)

        valid_words = 0
        garbled_count = 0
        for w in words:
            if w.isdigit():
                valid_words += 1
                continue
            w_lower = w.lower()
            vowels = sum(1 for char in w_lower if char in "aeiouy")
            if len(w) >= 5 and vowels == 0:
                garbled_count += 1
            elif re.search(r'([bcdfghjklmnpqrstvwxyz]){4,}', w_lower):
                garbled_count += 1
            else:
                valid_words += 1

        word_quality_ratio = valid_words / len(words)
        garbled_ratio = garbled_count / len(words)

        metrics = {
            "printable_ratio": round(printable_ratio, 3),
            "alpha_ratio": round(alpha_ratio, 3),
            "replacement_ratio": round(replacement_ratio, 3),
            "entropy": round(entropy, 2),
            "unique_token_ratio": round(unique_token_ratio, 2),
            "avg_word_length": round(avg_word_length, 2),
            "word_quality_ratio": round(word_quality_ratio, 3),
            "garbled_ratio": round(garbled_ratio, 3)
        }

        if replacement_chars >= 2 or replacement_ratio > 0.05:
            return False, "corrupted_replacement_characters", metrics

        if printable_ratio < 0.75:
            return False, "low_printable_character_ratio", metrics

        if alpha_ratio < 0.35:
            return False, "low_alphabetic_character_ratio", metrics

        if garbled_ratio > 0.18 or word_quality_ratio < 0.70:
            return False, "garbled_ocr_font_encoding", metrics

        if avg_word_length > 25:
            return False, "unusual_word_length_artifact", metrics

        return True, "valid", metrics

    def match_syllabus_subject_section(self, page_text: str, target_subject: str, target_code: str) -> bool:
        """
        Generic subject-scoped syllabus filter.
        Accepts pages that mention the target subject/code.
        Rejects pages that clearly declare a different SUBJECT/COURSE header.
        """
        text_lower = page_text.lower()
        sub_lower = (target_subject or "").lower().strip()
        code_lower = (target_code or "").lower().strip() if target_code else ""

        if code_lower and len(code_lower) >= 3 and code_lower in text_lower:
            return True

        if sub_lower and sub_lower in text_lower:
            return True

        words = [
            w for w in re.findall(r"\b[A-Za-z]+\b", target_subject or "")
            if w.lower() not in {"and", "of", "the", "systems", "system"}
        ]
        acronym = "".join(w[0] for w in words).lower()
        if len(acronym) >= 2 and re.search(r"\b" + re.escape(acronym) + r"\b", text_lower):
            return True

        # Explicit foreign subject/course headers → reject when target not evidenced
        declared = re.findall(
            r"(?:subject|course)\s*[:\-]\s*([^\n(]+?)(?:\s*\(|$)",
            text_lower,
            flags=re.IGNORECASE,
        )
        if declared and sub_lower:
            for decl in declared:
                decl_clean = decl.strip()
                if not decl_clean:
                    continue
                if sub_lower in decl_clean or decl_clean in sub_lower:
                    return True
                # Distinct declared subject name
                overlap = set(decl_clean.split()) & set(sub_lower.split())
                meaningful = {w for w in overlap if len(w) > 3 and w not in {"systems", "system", "engineering"}}
                if not meaningful:
                    return False

        has_target_keywords = any(w.lower() in text_lower for w in words if len(w) > 3)
        return bool(has_target_keywords) if declared else True

    def parse_syllabus_pdf(self, file_path: str, workspace_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parses syllabus PDF with text quality validation and subject-scoped syllabus filtering."""
        doc = fitz.open(file_path)
        chunks = []
        metadatas = []
        ids = []

        filename = os.path.basename(file_path)
        ws_id = workspace_info.get("id", "custom-ws")
        subject = workspace_info.get("subject", "Academic Subject")
        university = workspace_info.get("university", "Academic Institution")
        branch = workspace_info.get("branch") or workspace_info.get("program") or "General Program"
        semester = workspace_info.get("semester", "Semester 1")
        course_code = workspace_info.get("subjectCode") or workspace_info.get("subject_code") or "COURSE"

        current_block = "General Block"
        current_unit = "General Unit"
        audit_records = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()

            is_valid, reason, metrics = self.validate_text_quality(page_text)
            ocr_used = False

            if not is_valid:
                ocr_text = perform_ocr_page(page, dpi=100)
                if ocr_text and self.match_syllabus_subject_section(ocr_text, subject, course_code):
                    ocr_valid, ocr_reason, ocr_metrics = self.validate_text_quality(ocr_text)
                    if ocr_valid:
                        page_text = ocr_text
                        is_valid = True
                        reason = "valid_via_ocr"
                        metrics = ocr_metrics
                        ocr_used = True

            audit_records.append({
                "source_file": filename,
                "page": page_num + 1,
                "native_valid": is_valid and not ocr_used,
                "ocr_used": ocr_used,
                "status": "ACCEPTED" if is_valid else "REJECTED",
                "rejection_reason": None if is_valid else reason,
                "metrics": metrics
            })

            if not is_valid:
                continue

            if not self.match_syllabus_subject_section(page_text, subject, course_code):
                continue

            lines = page_text.split("\n")
            buffer = []

            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue

                block_match = re.search(r'^(Block)\s*([0-9IVX]+)\s*:?\s*(.*)', line_str, re.IGNORECASE)
                if block_match:
                    current_block = f"Block {block_match.group(2)}: {block_match.group(3)}".strip()

                unit_match = re.search(r'^(Unit|Module|Chapter|Section)\s*([0-9IVX]+|[A-Z])\s*:?\s*(.*)', line_str, re.IGNORECASE)
                if unit_match:
                    # Flush prior module buffer so topics never bleed across units
                    if buffer and len(" ".join(buffer)) >= 40:
                        chunk_text = (
                            f"Syllabus Document [{filename} | {university} | {subject} ({course_code}) | "
                            f"{current_block} | {current_unit}]\n" + " ".join(buffer)
                        )
                        c_id = f"syl_{ws_id}_{page_num+1}_{len(chunks)}"
                        chunks.append(chunk_text)
                        metadatas.append({
                            "workspace_id": ws_id,
                            "university": university,
                            "branch": branch,
                            "semester": semester,
                            "subject": subject,
                            "course_code": course_code,
                            "doc_type": "syllabus",
                            "block": current_block,
                            "unit": current_unit,
                            "source_file": filename,
                            "source_page": page_num + 1
                        })
                        ids.append(c_id)
                    buffer = []
                    current_unit = f"{unit_match.group(1).title()} {unit_match.group(2)}: {unit_match.group(3)}".strip()

                buffer.append(line_str)
                if len(" ".join(buffer)) >= 380:
                    chunk_text = f"Syllabus Document [{filename} | {university} | {subject} ({course_code}) | {current_block} | {current_unit}]\n" + " ".join(buffer)
                    c_id = f"syl_{ws_id}_{page_num+1}_{len(chunks)}"

                    chunks.append(chunk_text)
                    metadatas.append({
                        "workspace_id": ws_id,
                        "university": university,
                        "branch": branch,
                        "semester": semester,
                        "subject": subject,
                        "course_code": course_code,
                        "doc_type": "syllabus",
                        "block": current_block,
                        "unit": current_unit,
                        "source_file": filename,
                        "source_page": page_num + 1
                    })
                    ids.append(c_id)
                    buffer = buffer[-2:]

            if buffer:
                chunk_text = f"Syllabus Document [{filename} | {university} | {subject} ({course_code}) | {current_block} | {current_unit}]\n" + " ".join(buffer)
                c_id = f"syl_{ws_id}_{page_num+1}_{len(chunks)}"
                chunks.append(chunk_text)
                metadatas.append({
                    "workspace_id": ws_id,
                    "university": university,
                    "branch": branch,
                    "semester": semester,
                    "subject": subject,
                    "course_code": course_code,
                    "doc_type": "syllabus",
                    "block": current_block,
                    "unit": current_unit,
                    "source_file": filename,
                    "source_page": page_num + 1
                })
                ids.append(c_id)

        doc.close()
        self.last_audit_log = audit_records

        if chunks:
            self.store.add_documents(chunks, metadatas, ids)
            print(f"Universal Ingestion: Indexed {len(chunks)} subject-scoped syllabus chunks for {subject} ({ws_id})")

        return metadatas

    def parse_pyq_pdf(self, file_path: str, workspace_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parses PYQ PDF with text quality validation, OCR fallback,
        boundary detection, exact question preservation, and Hard Question Validation Gate.
        """
        doc = fitz.open(file_path)
        chunks = []
        metadatas = []
        ids = []

        filename = os.path.basename(file_path)
        ws_id = workspace_info.get("id", "custom-ws")
        subject = workspace_info.get("subject", "Academic Subject")
        university = workspace_info.get("university", "Academic Institution")
        branch = workspace_info.get("branch") or workspace_info.get("program") or "General Program"
        semester = workspace_info.get("semester", "Semester 1")
        course_code = workspace_info.get("subjectCode") or workspace_info.get("subject_code") or "COURSE"

        total_pages = len(doc)

        # Collect header text for year/session detection (subject-agnostic)
        header_blob = filename + "\n"
        for pi in range(min(2, total_pages)):
            try:
                header_blob += doc[pi].get_text()[:2000] + "\n"
            except Exception:
                pass

        from rag.question_extractor import detect_exam_year_and_session

        year, session = detect_exam_year_and_session(
            filename, header_blob[len(filename) + 1 :]
        )

        # Syllabus index from THIS workspace's uploaded syllabus only
        syllabus_index = build_syllabus_index_from_workspace(self.store, ws_id, subject=subject)
        syl_topics = []
        for m in syllabus_index.get("modules", []):
            syl_topics.extend(m.get("topics", []))

        rejected_count = 0
        all_accepted_questions = []
        all_rejected_candidates = []
        audit_records = []
        page_extraction_audit = []
        pages_payload: List[Dict[str, Any]] = []

        all_layout_lines: List[Dict[str, Any]] = []
        from rag.ocr_layout import ocr_page_lines, reconstruct_questions_from_layout

        for page_num in range(total_pages):
            page = doc[page_num]
            raw_native = page.get_text() or ""
            filtered_native = filter_noise_lines(raw_native)

            is_valid, reason, metrics = self.validate_text_quality(filtered_native)
            ocr_used = False
            ocr_raw = ""
            # Mixed garbled+readable pages: filtered may look valid while native is font-encoded.
            # Force OCR so we do not trust a half-cleaned remnant that loses question markers.
            raw_garbled = detect_suspicious_alphanumeric_noise(raw_native[:1200]) if raw_native else False
            if (
                is_valid
                and len(raw_native) >= 800
                and len(filtered_native) < max(120, int(0.25 * len(raw_native)))
            ):
                is_valid = False
                reason = "filtered_text_collapsed_from_garbled_page"
                metrics = {**(metrics or {}), "raw_chars": len(raw_native), "filtered_chars": len(filtered_native)}
            elif is_valid and raw_garbled and len(raw_native) >= 800:
                is_valid = False
                reason = "native_contains_garbled_prefix_force_ocr"
                metrics = {**(metrics or {}), "raw_chars": len(raw_native), "filtered_chars": len(filtered_native)}

            page_text = filtered_native
            if not is_valid:
                ocr_raw = perform_ocr_page(page, dpi=150) or ""
                if ocr_raw:
                    ocr_text = filter_noise_lines(ocr_raw)
                    ocr_valid, ocr_reason, ocr_metrics = self.validate_text_quality(ocr_text)
                    if ocr_valid:
                        page_text = ocr_text
                        is_valid = True
                        reason = "valid_via_ocr"
                        metrics = ocr_metrics
                        ocr_used = True
                    else:
                        # Prefer filtered native if OCR also bad but native filtered had academic verbs
                        if filtered_native and any(
                            re.search(rf"\b{v}\b", filtered_native.lower())
                            for v in ("explain", "what", "discuss", "derive", "describe", "differentiate")
                        ):
                            page_text = filtered_native
                            is_valid = True
                            reason = "fallback_filtered_native_after_ocr_fail"
                        else:
                            reason = ocr_reason or reason
                            metrics = ocr_metrics or metrics

            # Choose the representation with the strongest valid-question evidence.
            # Candidates: filtered native text, plain OCR text, layout-aware OCR.
            from rag.question_extractor import prepare_page_text_for_extraction
            from rag.question_extractor import extract_questions_from_page_text as _ex_qs

            def _evaluate(name: str, text_repr: str, *, is_ocr: bool, evidence: int):
                prepared = prepare_page_text_for_extraction(text_repr) if text_repr else ""
                if not prepared.strip():
                    return None
                acc, rej = _ex_qs(
                    prepared, page_num + 1, filename, ws_id, subject=subject, year=year or 0
                )
                _v, _r, q_metrics = self.validate_text_quality(text_repr)
                return {
                    "name": name,
                    "accepted": len(acc),
                    "rejected": len(rej),
                    "structure_score": question_structure_score(
                        [q["question_id"] for q in acc]
                    ),
                    "text_quality": q_metrics.get("word_quality_ratio", 0.0),
                    "prepared": prepared,
                    "source_text": text_repr,
                    "chars": len(text_repr),
                    "ocr": is_ocr,
                    "evidence": evidence,
                }

            candidates: List[Dict[str, Any]] = []
            native_eval = None
            if filtered_native.strip():
                native_eval = _evaluate("native", filtered_native, is_ocr=False, evidence=1)
                if native_eval:
                    candidates.append(native_eval)
            if is_valid and page_text.strip() and page_text is not filtered_native:
                candidates.append(
                    _evaluate("ocr_text", page_text, is_ocr=True, evidence=0)
                )
            # Always try geometry-preserving OCR when native structure is incomplete
            # or the page already required OCR. Do not skip it merely because
            # native text passed a character-quality gate.
            native_incomplete = bool(
                native_eval
                and (
                    native_eval["structure_score"] < 0.99
                    or native_eval["rejected"] > 0
                    or native_eval["accepted"] < 3
                )
            )
            if ocr_used or not is_valid or not filtered_native.strip() or native_incomplete:
                layout_lines = ocr_page_lines(page, dpi=150)
                for ln in layout_lines:
                    all_layout_lines.append({**ln, "top": ln["top"] + page_num * 12000})
                layout_text = reconstruct_questions_from_layout(layout_lines)
                if layout_text.strip():
                    candidates.append(
                        _evaluate("ocr_layout", layout_text, is_ocr=True, evidence=2)
                    )
            candidates = [c for c in candidates if c]

            representation_audit = [
                {
                    "representation": c["name"],
                    "chars": c["chars"],
                    "accepted_questions": c["accepted"],
                    "rejected_candidates": c["rejected"],
                    "structure_score": round(c["structure_score"], 3),
                    "text_quality": c["text_quality"],
                }
                for c in candidates
            ]

            if candidates:
                # Rank by recovered-question evidence, weighted by internal
                # structure. A representation that invents extra regex markers
                # but cannot accept them must not beat a coherent layout.
                def _rank(c):
                    structural = c["accepted"] * (0.5 + 0.5 * (c["structure_score"] or 0.0))
                    return (structural, c["evidence"], c["accepted"], c["text_quality"], c["chars"])

                best_structural = max(c["accepted"] * (0.5 + 0.5 * (c["structure_score"] or 0)) for c in candidates)
                contenders = [
                    c for c in candidates
                    if (c["accepted"] * (0.5 + 0.5 * (c["structure_score"] or 0)))
                    >= max(0.5, 0.55 * best_structural)
                ] or candidates
                best = max(contenders, key=_rank)
                reconstructed = best["prepared"]
                page_text = best["source_text"]
                ocr_used = ocr_used or best["ocr"]
                is_valid = True
                reason = f"selected_{best['name']}_yield_{best['accepted']}"
                selected_representation = best["name"]
            else:
                reconstructed = prepare_page_text_for_extraction(page_text) if is_valid else ""
                selected_representation = "native" if is_valid else "none"

            page_audit = {
                "page": page_num + 1,
                "pymupdf_chars": len(raw_native),
                "filtered_native_chars": len(filtered_native),
                "ocr_used": ocr_used,
                "ocr_chars": len(ocr_raw) if ocr_raw else 0,
                "final_chars": len(reconstructed or page_text or ""),
                "quality_reason": reason,
                "quality_valid": is_valid,
                "raw_native_chars": len(raw_native),
                "raw_ocr_chars": len(ocr_raw or ""),
                "reconstructed_chars": len(reconstructed),
                "selected_representation": selected_representation,
                "representations": representation_audit,
            }

            if not is_valid:
                rejected_count += 1
                all_rejected_candidates.append({
                    "raw_text": (page_text or raw_native)[:200] if (page_text or raw_native) else "Empty Page Text",
                    "reason": f"PAGE_LEVEL_REJECTED_{str(reason).upper()}",
                    "page": page_num + 1,
                    "metrics": metrics,
                    "question_id": None,
                })
                page_audit["status"] = "REJECTED"
                page_audit["questions_accepted"] = 0
                page_audit["questions_rejected"] = 0
                page_extraction_audit.append(page_audit)
                audit_records.append({
                    "source_file": filename,
                    "page": page_num + 1,
                    "native_valid": False,
                    "ocr_used": ocr_used,
                    "status": "REJECTED",
                    "rejection_reason": reason,
                    "metrics": metrics,
                    "questions_extracted": [],
                })
                print(f"[PYQ_PAGE_REJECTED] filename: {filename} page: {page_num+1} reason: {reason}")
                print(
                    f"[PYQ_PAGE_EXTRACT] page={page_num+1} pymupdf={page_audit['pymupdf_chars']} "
                    f"ocr={'USED' if ocr_used else 'NOT USED'} ocr_chars={page_audit['ocr_chars']} "
                    f"final={page_audit['final_chars']}"
                )
                # Still keep empty page record for audit continuity
                pages_payload.append({
                    "page": page_num + 1,
                    "raw_native_text": raw_native,
                    "raw_ocr_text": ocr_raw,
                    "reconstructed_text": "",
                    "ocr_used": ocr_used,
                })
                continue

            # OCR rescue if reconstructed looks empty of questions later handled in hybrid
            if not ocr_used and len(raw_native) >= 400 and len(reconstructed) < 80:
                ocr_raw = perform_ocr_page(page, dpi=150) or ""
                if ocr_raw:
                    ocr_text = filter_noise_lines(ocr_raw)
                    ocr_valid, _, ocr_metrics = self.validate_text_quality(ocr_text)
                    if ocr_valid:
                        page_text = ocr_text
                        reconstructed = prepare_page_text_for_extraction(ocr_text)
                        ocr_used = True
                        reason = "ocr_rescue_short_reconstructed"
                        metrics = ocr_metrics
                        page_audit["ocr_used"] = True
                        page_audit["ocr_chars"] = len(ocr_raw)
                        page_audit["final_chars"] = len(reconstructed)
                        page_audit["quality_reason"] = reason

            pages_payload.append({
                "page": page_num + 1,
                "raw_native_text": raw_native,
                "raw_ocr_text": ocr_raw,
                "reconstructed_text": reconstructed,
                "ocr_used": ocr_used,
            })
            page_audit["status"] = "TEXT_READY"
            page_extraction_audit.append(page_audit)
            print(
                f"[PYQ_PAGE_EXTRACT] page={page_num+1} pymupdf={page_audit['pymupdf_chars']} "
                f"ocr={'USED' if page_audit['ocr_used'] else 'NOT USED'} "
                f"ocr_chars={page_audit['ocr_chars']} final={page_audit['final_chars']} "
                f"reconstructed={len(reconstructed)}"
            )

        doc.close()

        # Stitch layout across pages before completeness. A question that starts
        # at the bottom of page N is not truncated until page N+1 is included.
        if total_pages > 1 and all_layout_lines:
            doc_layout = reconstruct_questions_from_layout(all_layout_lines)
            n_doc = sum(1 for ln in (doc_layout or "").splitlines() if ln.startswith("Q"))
            n_pages = sum(
                1
                for p in pages_payload
                for ln in (p.get("reconstructed_text") or "").splitlines()
                if ln.startswith("Q")
            )
            if n_doc >= max(3, n_pages):
                from rag.question_extractor import prepare_page_text_for_extraction as _prep

                pages_payload[0]["reconstructed_text"] = _prep(doc_layout)
                for p in pages_payload[1:]:
                    p["reconstructed_text"] = ""

        # Hybrid document-level question understanding
        from rag.hybrid_question_extraction import hybrid_extract_document

        hybrid = hybrid_extract_document(
            pages_payload,
            filename=filename,
            workspace_id=ws_id,
            subject=subject,
            year=year if year else 0,
            syllabus_topics=syl_topics,
        )
        all_accepted_questions = hybrid.get("accepted_questions") or []
        all_rejected_candidates.extend(hybrid.get("rejected_candidates") or [])
        quality = hybrid.get("quality") or {}
        extraction_quality = quality.get("extraction_quality", "FAILED")

        for q_obj in all_accepted_questions:
            syl_map, confidence = map_subquestion_to_syllabus_index(
                question_text=q_obj["exact_text"],
                detected_topics=q_obj.get("detected_topics") or [],
                subject=subject,
                syllabus_index=syllabus_index,
                workspace_id=ws_id,
                vector_store=self.store,
            )
            q_obj["syllabus_mapping"] = syl_map
            q_obj["confidence"] = confidence
            q_obj["exam_session"] = session
            if year:
                q_obj["year"] = year

            year_label = str(year) if year else "Unknown"
            pages = q_obj.get("source_pages") or [q_obj.get("source_page") or 1]
            page_label = pages[0]
            c_id = f"pyq_{ws_id}_{filename}_{page_label}_{q_obj['question_number']}"
            chunk_text = (
                f"PYQ Question Item {q_obj['question_number']} "
                f"[{year_label} {session} | {q_obj['marks']} Marks | {subject} ({course_code}) | {filename}]\n"
                f"{q_obj['exact_text']}"
            )
            chunks.append(chunk_text)
            metadatas.append({
                "workspace_id": ws_id,
                "university": university,
                "branch": branch,
                "semester": semester,
                "subject": subject,
                "course_code": course_code,
                "doc_type": "pyq",
                "year": str(year) if year else "Unknown",
                "exam_session": session,
                "marks": str(q_obj["marks"]),
                "question_id": q_obj["question_id"],
                "question_number": q_obj["question_number"],
                "parent_question": q_obj.get("parent_question", "Q1"),
                "subquestion": q_obj.get("subquestion") or "",
                "exact_text": q_obj["exact_text"],
                "normalized_text": q_obj.get("normalized_text", ""),
                "question_intent": q_obj.get("question_intent", ""),
                "question_type": q_obj.get("question_type", ""),
                "entities": json.dumps(q_obj.get("entities", [])),
                "constraints": json.dumps(q_obj.get("constraints", [])),
                "detected_topics": json.dumps(q_obj.get("detected_topics") or []) if isinstance(q_obj.get("detected_topics"), list) else str(q_obj.get("detected_topics") or ""),
                "primary_topic": q_obj.get("primary_topic") or "",
                "syllabus_module": syl_map.get("module", "Unmapped"),
                "syllabus_chapter": syl_map.get("chapter", "Unmapped"),
                "syllabus_topic": syl_map.get("topic", "Unmapped"),
                "confidence": str(confidence),
                "content_type": "question",
                "document_id": f"doc-{filename}",
                "source_file": filename,
                "source_page": page_label,
                "source_pages": json.dumps(pages),
                "source_page_start": pages[0],
                "source_page_end": q_obj.get("source_page_end", pages[-1]),
                "cross_page_merged": bool(q_obj.get("cross_page_merged", False)),
                "grounding_score": str(q_obj.get("grounding_score", "")),
                "extraction_method": q_obj.get("extraction_method", "hybrid"),
                "extraction_quality": extraction_quality,
                "rejected_count": rejected_count,
            })
            ids.append(c_id)

        from rag.question_extractor import validate_analyzed_pyq_paper
        is_valid_paper, validation_errors = validate_analyzed_pyq_paper(all_accepted_questions, syllabus_index)
        if not is_valid_paper:
            print(f"[PYQ_VALIDATION_WARNING] Validation issues detected in {filename}: {validation_errors}")

        extraction_incomplete = extraction_quality in ("PARTIAL", "FAILED")
        incomplete_reason = None
        if extraction_quality == "FAILED":
            ingestion_status = "INGESTION_FAILED"
            incomplete_reason = (
                "Question extraction failed. Please review the extraction audit."
            )
        elif extraction_quality == "PARTIAL":
            ingestion_status = "INGESTION_PARTIAL"
            incomplete_reason = (
                f"Question extraction is incomplete. "
                f"Extracted {quality.get('questions_extracted')} of "
                f"{quality.get('source_markers_detected')} detected markers. "
                f"Missing: {quality.get('missing_questions')}. "
                "Please review the extraction audit."
            )
        elif len(all_accepted_questions) == 0:
            ingestion_status = "ingestion_failed_no_valid_questions"
            extraction_incomplete = True
            incomplete_reason = "No valid questions could be extracted."
        else:
            ingestion_status = "ready"

        self.last_audit_log = audit_records
        self.last_pyq_questions_audit = {
            "accepted_questions": all_accepted_questions,
            "rejected_candidates": all_rejected_candidates,
            "validation_errors": validation_errors,
            "ingestion_status": ingestion_status,
            "extraction_incomplete": extraction_incomplete,
            "incomplete_reason": incomplete_reason,
            "page_extraction_audit": page_extraction_audit,
            "source_markers": hybrid.get("source_markers") or [],
            "extraction_quality": extraction_quality,
            "question_extraction_confidence": quality.get("confidence"),
            "quality_summary": {
                **quality,
                "accepted_count": len(all_accepted_questions),
                "rejected_count": len(all_rejected_candidates),
                "total_pages": total_pages,
                "exam_year": year if year else None,
                "exam_session": session,
                "ocr_pages": sum(1 for p in page_extraction_audit if p.get("ocr_used")),
                "llm_used": hybrid.get("llm_used", False),
                "llm_candidates": hybrid.get("llm_candidates", 0),
                "llm_rejected_for_truncation": hybrid.get("llm_rejected_for_truncation", 0),
                "cross_page_merges": hybrid.get("cross_page_merges", 0),
                "grounding_coverage": hybrid.get("grounding_coverage", 0.0),
                "selected_representations": [
                    {"page": p.get("page"), "selected": p.get("selected_representation")}
                    for p in page_extraction_audit
                ],
                "rejection_reasons": sorted(
                    {
                        str(r.get("reason") or r.get("rejection_reason") or "unknown")
                        for r in all_rejected_candidates
                    }
                ),
            },
        }

        print("==================================================")
        print(f"UPLOAD INGESTION PIPELINE PROOF LOG")
        print(f"SOURCE FILE:      {filename}")
        print(f"DOCUMENT ID:      doc-{filename}")
        print(f"WORKSPACE ID:     {ws_id}")
        print(f"PAGES EXTRACTED:  {total_pages}")
        print(f"QUESTIONS FOUND:  {len(all_accepted_questions)}")
        print(f"MARKERS DETECTED: {quality.get('source_markers_detected')}")
        print(f"EXTRACTION QUALITY:{extraction_quality}")
        print(f"CHUNKS CREATED:   {len(chunks)}")
        print(f"EMBEDDINGS READY: {len(chunks)}")
        print(f"INGEST STATUS:    {ingestion_status}")
        print("==================================================")

        # Do not silently create incomplete vectors as "ready"
        if extraction_quality in ("FAILED",) or not chunks:
            print(f"[INGESTION_FAILURE] No complete extraction for {filename}; not inserting vectors.")
            return []

        if extraction_quality == "PARTIAL":
            print(
                f"[INGESTION_PARTIAL] Incomplete extraction for {filename}; "
                "vectors NOT inserted. Review extraction audit."
            )
            return []

        self.store.replace_documents_for_source(
            chunks,
            metadatas,
            ids,
            source_file=filename,
            workspace_id=ws_id,
        )
        print(f"VECTOR INSERT COMPLETE: {len(chunks)} vectors inserted into ChromaDB for workspace {ws_id}")

        return metadatas

    def parse_textbook_pdf(self, file_path: str, workspace_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parses textbook PDF with text quality validation."""
        doc = fitz.open(file_path)
        chunks = []
        metadatas = []
        ids = []

        filename = os.path.basename(file_path)
        ws_id = workspace_info.get("id", "custom-ws")
        subject = workspace_info.get("subject", "Academic Subject")
        university = workspace_info.get("university", "Academic Institution")
        branch = workspace_info.get("branch") or workspace_info.get("program") or "General Program"
        semester = workspace_info.get("semester", "Semester 1")
        course_code = workspace_info.get("subjectCode") or workspace_info.get("subject_code") or "COURSE"

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()

            is_valid, reason, metrics = self.validate_text_quality(page_text)
            if not is_valid:
                print(f"[TEXTBOOK_REJECTED] filename: {filename} page: {page_num+1} reason: {reason}")
                continue

            lines = page_text.split("\n")
            buffer = []

            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue

                ch_match = re.search(r'^(Chapter|Module|Part)\s*([0-9IVX]+|[A-Z])\s*:?\s*(.*)', line_str, re.IGNORECASE)
                current_chapter = f"{ch_match.group(1).title()} {ch_match.group(2)}: {ch_match.group(3)}".strip() if ch_match else "General Chapter"

                buffer.append(line_str)
                if len(" ".join(buffer)) >= 450:
                    chunk_text = f"Textbook Document [{filename} | {subject} ({course_code}) | {current_chapter}]\n" + " ".join(buffer)
                    c_id = f"tb_{ws_id}_{page_num+1}_{len(chunks)}"

                    chunks.append(chunk_text)
                    metadatas.append({
                        "workspace_id": ws_id,
                        "university": university,
                        "branch": branch,
                        "semester": semester,
                        "subject": subject,
                        "course_code": course_code,
                        "doc_type": "textbook",
                        "chapter": current_chapter,
                        "source_file": filename,
                        "source_page": page_num + 1
                    })
                    ids.append(c_id)
                    buffer = buffer[-2:]

            if buffer:
                chunk_text = f"Textbook Document [{filename} | {subject} ({course_code}) | General Chapter]\n" + " ".join(buffer)
                c_id = f"tb_{ws_id}_{page_num+1}_{len(chunks)}"
                chunks.append(chunk_text)
                metadatas.append({
                    "workspace_id": ws_id,
                    "university": university,
                    "branch": branch,
                    "semester": semester,
                    "subject": subject,
                    "course_code": course_code,
                    "doc_type": "textbook",
                    "chapter": "General Chapter",
                    "source_file": filename,
                    "source_page": page_num + 1
                })
                ids.append(c_id)

        doc.close()

        if chunks:
            self.store.add_documents(chunks, metadatas, ids)
            print(f"Universal Ingestion: Indexed {len(chunks)} textbook chunks for {subject} ({ws_id})")

        return metadatas
