"""
Generic source-paper identity for PYQ Intelligence counting.

Recurrence is the number of distinct canonical exam papers a question
appears in — never the raw matching-question record count.

Question numbers (Q1(a), Q6(a), …) are source-location metadata only.
They never determine paper identity or recurrence.

When identity cannot be determined confidently, papers stay separate
and are marked uncertain. No university / subject / year / filename
catalog is used.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from rag.config import BASE_DIR


UNKNOWN_META = {
    "",
    "unknown",
    "unknown session",
    "exam",
    "exam session",
    "academic institution",
    "academic subject",
    "course",
    "unmapped",
    "subject",
    "semester",
}

# Generic OS / browser copy suffixes. Not tied to any exam paper name.
_COPY_SUFFIX_RE = re.compile(
    r"(?:"
    r"\s*\(\d+\)"
    r"|\s*[-_]\s*copy(?:\s*\(\d+\))?"
    r"|\s+copy(?:\s*\(\d+\))?"
    r"|_copy(?:_\d+)?"
    r")+$",
    re.IGNORECASE,
)

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _norm_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_exam_session(session: Any) -> str:
    """Case-fold and unify separators. Empty if session is unknown."""
    raw = _norm_space(session).lower()
    if raw in UNKNOWN_META:
        return ""
    raw = raw.replace("–", "/").replace("—", "/").replace("-", "/").replace("_", "/")
    raw = re.sub(r"/+", "/", raw).strip("/")
    return raw


def normalize_filename_stem(filename: Any) -> str:
    """
    Filename identity after stripping generic copy/duplicate suffixes.
    Used only as a corroborating signal, never as the sole merge key
    unless content also agrees.
    """
    name = os.path.basename(str(filename or "")).strip()
    stem, _ext = os.path.splitext(name)
    prev = None
    while prev != stem:
        prev = stem
        stem = _COPY_SUFFIX_RE.sub("", stem).strip(" ._-\t")
    stem = _norm_space(stem).lower()
    return stem


def known_meta(value: Any) -> str:
    raw = _norm_space(value)
    if not raw or raw.lower() in UNKNOWN_META:
        return ""
    return raw.lower()


def _meta_compatible(a: str, b: str) -> bool:
    """Unknown is compatible with anything; two known distinct values are not."""
    if not a or not b:
        return True
    if a == b:
        return True
    return a in b or b in a


def _stable_hash(parts: Sequence[str]) -> str:
    blob = "|".join(parts).encode("utf-8", "ignore")
    return hashlib.sha1(blob).hexdigest()[:16]


def question_text_key(record: Dict[str, Any]) -> str:
    """Question body identity — never uses question number/letter."""
    text = (
        record.get("normalized_text")
        or _norm_space(record.get("exact_text") or "").lower()
    )
    return _norm_space(text).lower()


def content_fingerprint(texts: Iterable[str]) -> str:
    keys = sorted({_norm_space(t).lower() for t in texts if _norm_space(t)})
    if not keys:
        return ""
    return _stable_hash(keys)


def file_bytes_hash(path: str) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def resolve_upload_path(
    workspace_id: Optional[str],
    source_file: Optional[str],
    source_path: Optional[str] = None,
) -> Optional[str]:
    if source_path and os.path.isfile(source_path):
        return source_path
    if not workspace_id or not source_file:
        return None
    candidate = os.path.join(BASE_DIR, "data", "uploads", workspace_id, os.path.basename(source_file))
    if os.path.isfile(candidate):
        return candidate
    return None


def paper_id_of(record: Dict[str, Any]) -> str:
    pid = record.get("canonical_paper_id")
    if pid:
        return str(pid)
    sf = record.get("source_file") or "unknown.pdf"
    return f"file:{sf}"


def unique_paper_ids(records: Iterable[Dict[str, Any]]) -> List[str]:
    seen: List[str] = []
    seen_set: Set[str] = set()
    for rec in records:
        pid = paper_id_of(rec)
        if pid not in seen_set:
            seen_set.add(pid)
            seen.append(pid)
    return seen


def unique_occurrence_count(records: Iterable[Dict[str, Any]]) -> int:
    return len(unique_paper_ids(records))


def unique_years(records: Iterable[Dict[str, Any]]) -> List[int]:
    years: Set[int] = set()
    for rec in records:
        year = rec.get("year")
        try:
            yi = int(year)
        except (TypeError, ValueError):
            continue
        if yi:
            years.add(yi)
    return sorted(years)


def unique_session_identities(records: Iterable[Dict[str, Any]]) -> List[Tuple[int, str, str]]:
    """Unique canonical (year, session, paper) identities."""
    slots: List[Tuple[int, str, str]] = []
    seen: Set[Tuple[int, str, str]] = set()
    for rec in records:
        try:
            year = int(rec.get("year") or 0)
        except (TypeError, ValueError):
            year = 0
        session = normalize_exam_session(rec.get("exam_session"))
        pid = paper_id_of(rec)
        key = (year, session, pid)
        if key not in seen:
            seen.add(key)
            slots.append(key)
    return slots


def dedupe_records_by_paper(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One representative record per canonical paper (first seen)."""
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for rec in records:
        pid = paper_id_of(rec)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(rec)
    return out


def dedupe_records_by_paper_and_question(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Same question in the same canonical paper counts once.
    Different questions in the same paper remain distinct records.
    Question number is ignored.
    """
    out: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for rec in records:
        key = (paper_id_of(rec), question_text_key(rec))
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def source_identity_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    pid = paper_id_of(record)
    return {
        "canonical_paper_id": pid,
        "source_identity_confidence": record.get("source_identity_confidence") or "uncertain",
        "duplicate_source_ids": list(record.get("duplicate_source_ids") or []),
        "unique_occurrence_count": 1,
    }


class _UnionFind:
    def __init__(self, items: Sequence[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> Dict[str, List[str]]:
        clustered: Dict[str, List[str]] = defaultdict(list)
        for item in self.parent:
            clustered[self.find(item)].append(item)
        return clustered


def _text_jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _should_merge(pa: Dict[str, Any], pb: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Conservative merge. Returns (merge?, confidence).

    High-confidence: identical bytes, or identical question-set fingerprint
    in the same year+session slot (or unknown slot + copy-filename).

    Medium-confidence: very high question-set overlap in the same slot
    with compatible university/subject.

    Never merge across different years or different known sessions.
    Never merge on question number. Never merge solely because wording matches
    unless the whole paper identity agrees.
    """
    year_a, year_b = pa["year"], pb["year"]
    if year_a and year_b and year_a != year_b:
        return False, ""

    sess_a, sess_b = pa["session"], pb["session"]
    if sess_a and sess_b and sess_a != sess_b:
        return False, ""

    if not _meta_compatible(pa["university"], pb["university"]):
        return False, ""
    if not _meta_compatible(pa["subject"], pb["subject"]):
        return False, ""
    if pa["course_code"] and pb["course_code"] and not _meta_compatible(pa["course_code"], pb["course_code"]):
        return False, ""

    same_bytes = bool(pa["bytes_hash"] and pa["bytes_hash"] == pb["bytes_hash"])
    same_fp = bool(pa["fingerprint"] and pa["fingerprint"] == pb["fingerprint"])
    copy_names = bool(
        pa["filename_stem"]
        and pa["filename_stem"] == pb["filename_stem"]
        and pa["source_file"] != pb["source_file"]
    )
    jaccard = _text_jaccard(pa["text_set"], pb["text_set"])
    year_ok = (not year_a or not year_b or year_a == year_b)
    session_ok = (not sess_a or not sess_b or sess_a == sess_b)
    slot_known = bool((year_a or year_b) and (sess_a or sess_b))

    if same_bytes:
        return True, "high"

    if same_fp and year_ok and session_ok:
        # A multi-question inventory is a paper fingerprint, not a single wording match.
        if len(pa["text_set"]) >= 2 and len(pb["text_set"]) >= 2:
            return True, "high"
        # Single-question papers: only merge with an independent source signal.
        # Same wording + same year/session is not enough — that can be two real papers.
        if copy_names:
            return True, "high"
        return False, ""

    if jaccard >= 0.85 and len(pa["text_set"]) >= 2 and len(pb["text_set"]) >= 2 and year_ok and session_ok:
        if slot_known or copy_names:
            return True, "high" if jaccard >= 0.95 else "medium"
        if pa["university"] and pb["university"] and pa["subject"] and pb["subject"]:
            return True, "medium"

    if copy_names and year_ok and session_ok and (same_fp or jaccard >= 0.50):
        return True, "high"

    return False, ""


def _build_file_profile(
    source_file: str,
    records: Sequence[Dict[str, Any]],
    workspace_id: Optional[str],
) -> Dict[str, Any]:
    first = records[0] if records else {}
    year = 0
    for rec in records:
        try:
            yi = int(rec.get("year") or 0)
        except (TypeError, ValueError):
            yi = 0
        if yi:
            year = yi
            break
    session = ""
    for rec in records:
        session = normalize_exam_session(rec.get("exam_session"))
        if session:
            break
    university = ""
    subject = ""
    course_code = ""
    for rec in records:
        university = university or known_meta(rec.get("university"))
        subject = subject or known_meta(rec.get("subject"))
        course_code = course_code or known_meta(rec.get("course_code") or rec.get("paper_code"))
    texts = [question_text_key(r) for r in records if question_text_key(r)]
    text_set = set(texts)
    provided_hash = ""
    for rec in records:
        provided_hash = str(rec.get("source_bytes_hash") or rec.get("file_sha256") or "")
        if provided_hash:
            break
    path = resolve_upload_path(
        workspace_id,
        source_file,
        first.get("source_path") or first.get("persisted_path"),
    )
    bytes_hash = provided_hash or (file_bytes_hash(path) if path else None) or ""
    return {
        "source_file": source_file,
        "year": year,
        "session": session,
        "university": university,
        "subject": subject,
        "course_code": course_code,
        "filename_stem": normalize_filename_stem(source_file),
        "text_set": text_set,
        "fingerprint": content_fingerprint(texts),
        "bytes_hash": bytes_hash,
        "records": list(records),
    }


def _canonical_id_for_cluster(members: Sequence[Dict[str, Any]], confidence: str) -> str:
    reps = sorted(members, key=lambda p: p["source_file"])
    head = reps[0]
    if confidence == "uncertain" or len(members) == 1:
        # Unmerged files stay distinct even when they share one question's wording.
        prefix = "uncertain" if confidence == "uncertain" else "paper"
        return f"{prefix}:{_stable_hash([head['source_file'], str(head['year'] or ''), head['session'], head['fingerprint'] or ''])}"
    parts = [
        head["university"],
        head["subject"],
        str(head["year"] or ""),
        head["session"],
        head["course_code"],
        head["fingerprint"] or head["bytes_hash"] or head["filename_stem"],
    ]
    return f"paper:{_stable_hash(parts)}"


def attach_source_identity(
    questions: Sequence[Dict[str, Any]],
    workspace_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Annotate each question with canonical paper identity.

    Mutates records in place and returns the same list.
    """
    if not questions:
        return list(questions)

    by_file: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for q in questions:
        sf = q.get("source_file") or "unknown.pdf"
        by_file[str(sf)].append(q)

    profiles = {
        sf: _build_file_profile(sf, recs, workspace_id)
        for sf, recs in by_file.items()
    }
    files = list(profiles.keys())
    uf = _UnionFind(files)
    pair_confidence: Dict[Tuple[str, str], str] = {}

    for i, fa in enumerate(files):
        for fb in files[i + 1 :]:
            merge, conf = _should_merge(profiles[fa], profiles[fb])
            if merge:
                uf.union(fa, fb)
                pair_confidence[tuple(sorted((fa, fb)))] = conf

    clusters = uf.groups()
    file_to_identity: Dict[str, Dict[str, Any]] = {}
    for _root, members in clusters.items():
        member_profiles = [profiles[sf] for sf in sorted(members)]
        if len(members) == 1:
            prof = member_profiles[0]
            has_slot = bool(prof["year"] or prof["session"])
            has_body = bool(prof["fingerprint"] or prof["bytes_hash"])
            confidence = "high" if (has_slot and has_body) else ("medium" if has_slot or has_body else "uncertain")
        else:
            confs = [
                pair_confidence.get(tuple(sorted((a, b))), "medium")
                for idx, a in enumerate(members)
                for b in members[idx + 1 :]
            ]
            confidence = "high" if "high" in confs else ("medium" if confs else "medium")
        pid = _canonical_id_for_cluster(member_profiles, confidence if len(members) > 1 or confidence != "uncertain" else confidence)
        if len(members) == 1 and confidence == "uncertain":
            pid = _canonical_id_for_cluster(member_profiles, "uncertain")
        dups = [sf for sf in sorted(members)]
        for sf in members:
            file_to_identity[sf] = {
                "canonical_paper_id": pid,
                "source_identity_confidence": confidence,
                "duplicate_source_ids": [x for x in dups if x != sf],
            }

    for q in questions:
        sf = str(q.get("source_file") or "unknown.pdf")
        ident = file_to_identity.get(sf) or {
            "canonical_paper_id": f"uncertain:{_stable_hash([sf])}",
            "source_identity_confidence": "uncertain",
            "duplicate_source_ids": [],
        }
        q["canonical_paper_id"] = ident["canonical_paper_id"]
        q["source_identity_confidence"] = ident["source_identity_confidence"]
        q["duplicate_source_ids"] = list(ident["duplicate_source_ids"])
    return list(questions)


def source_dedup_summary(questions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    files = {str(q.get("source_file") or "") for q in questions if q.get("source_file")}
    papers = unique_paper_ids(questions)
    uncertain = {
        paper_id_of(q)
        for q in questions
        if str(q.get("source_identity_confidence") or "") == "uncertain"
    }
    return {
        "unique_papers": len(papers),
        "raw_source_files": len(files),
        "collapsed_duplicate_files": max(0, len(files) - len(papers)),
        "uncertain_source_identities": len(uncertain),
        "canonical_paper_ids": papers,
    }


def merge_paper_stats(
    paper_stats: Dict[str, Dict[str, Any]],
    questions: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Collapse per-filename paper_stats onto canonical papers."""
    if not paper_stats:
        return {}
    sample = next(iter(paper_stats.values()), None)
    if sample and sample.get("canonical_paper_id") and sample.get("canonical_paper_id") in paper_stats:
        return paper_stats
    file_to_paper: Dict[str, str] = {}
    file_meta: Dict[str, Dict[str, Any]] = {}
    for q in questions:
        sf = str(q.get("source_file") or "")
        if not sf:
            continue
        file_to_paper[sf] = paper_id_of(q)
        file_meta[sf] = {
            "confidence": q.get("source_identity_confidence") or "uncertain",
            "duplicates": list(q.get("duplicate_source_ids") or []),
        }
    merged: Dict[str, Dict[str, Any]] = {}
    for sf, stats in paper_stats.items():
        pid = file_to_paper.get(sf) or f"file:{sf}"
        if pid not in merged:
            ident = file_meta.get(sf, {})
            merged[pid] = {
                **stats,
                "source_file": stats.get("source_file") or sf,
                "canonical_paper_id": pid,
                "source_identity_confidence": ident.get("confidence") or "uncertain",
                "duplicate_source_ids": list(ident.get("duplicates") or []),
            }
        else:
            dest = merged[pid]
            dest["valid_questions"] = max(int(dest.get("valid_questions") or 0), int(stats.get("valid_questions") or 0))
            dest["rejected_questions"] = max(int(dest.get("rejected_questions") or 0), int(stats.get("rejected_questions") or 0))
            dest["incomplete"] = bool(dest.get("incomplete") or stats.get("incomplete"))
            extras = dest.get("duplicate_source_ids") or []
            if sf != dest.get("source_file") and sf not in extras:
                extras.append(sf)
            dest["duplicate_source_ids"] = extras
    return merged
