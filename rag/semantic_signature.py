"""
Deterministic semantic signatures for PYQ question-level intelligence.

A signature answers: what is being asked, of which entity, with what
output, constraints, and comparison target?

No subject / university / filename catalog. Embeddings are never the
authority for a semantic repeat.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from rag.question_extractor import (
    ACADEMIC_QUESTION_VERBS,
    GENERIC_DOMAIN_TERMS,
    STOPWORDS,
    extract_constraints,
    extract_entities,
    requested_output_focus,
)


# Academic filler — never enough, on its own, to claim two questions are the same.
WEAK_GENERIC = set(GENERIC_DOMAIN_TERMS) | {
    "analytics", "analytic", "analysis", "social", "media", "data", "information",
    "based", "digital", "computer", "using", "study", "studies", "approach",
    "approaches", "role", "need", "given", "case", "cases", "tool", "tools",
    "area", "areas", "field", "basic", "important", "main", "key", "various",
    "different", "general", "specific", "common", "following", "above", "below",
    "help", "suitable", "diagram", "neat", "short", "note", "notes",
    "protocol", "scheme", "framework", "deep",
}

# Shared only these → no RELATED group either.
ULTRA_GENERIC = {
    "system", "method", "process", "approach", "based", "using", "study", "role",
    "data", "information", "digital", "computer", "area", "field", "tool", "case",
    "general", "specific", "common", "following", "question", "paper", "exam",
    "detail", "suitable", "help", "given", "need", "basic", "important",
    "image", "diagram", "table", "figure", "page", "example",
    "answer", "illustrate", "briefly", "real", "life",
}

_LAYOUT_NOISE_RE = re.compile(
    r"\[(?:IMAGE|TABLE|diagram[^\]]*)\]"
    r"|diagram on page \d+"
    r"|\bpage\s+\d+\s+of\s+\d+\b"
    r"|\(\s*\d+\s*(?:marks?|m)\s*\)"
    r"|\[\s*\d+\s*(?:marks?|m)\s*\]",
    re.I,
)
_GLUE_WORDS = ("from", "and", "with", "between", "into", "versus")
_TRAIL_HEADS = (
    "systems", "system", "analytics", "analytic", "algorithms", "algorithm",
    "networks", "network", "methods", "method", "procedures", "procedure",
)
_LEAD_ROOTS = ("conventional", "traditional", "classic", "classical")

# Output / ask words — not the technical entity.
SOFT_FOCUS = {
    "advantage", "disadvantage", "overfit", "underfit", "significance",
    "problem", "solve", "method", "technique", "process", "operation",
    "working", "work", "detail", "type", "kind", "feature", "property",
    "function", "funct", "role", "need", "importance", "benefit", "merit",
    "limitation", "drawback", "demerit", "application", "applic", "use",
    "architecture", "structure", "component", "layer", "step", "procedure",
    "algorithm", "mechanism", "compare", "difference", "prevent", "detect",
    "learn", "learning", "include", "includ", "used", "how", "does", "what",
}

# Morphological / academic-English folding. Not a topic catalog.
_SYNONYM_ROOT = {
    "traditional": "conventional",
    "classic": "conventional",
    "classical": "conventional",
    "standard": "conventional",
    "usual": "conventional",
    "conventional": "conventional",
    "advantage": "advantage",
    "benefit": "advantage",
    "merit": "advantage",
    "disadvantage": "disadvantage",
    "drawback": "disadvantage",
    "limitation": "disadvantage",
    "demerit": "disadvantage",
    "working": "working",
    "operation": "working",
    "mechanism": "working",
    "operates": "working",
    "application": "application",
    "usecase": "application",
    "architecture": "architecture",
    "structure": "architecture",
    "component": "architecture",
    "type": "classification",
    "kind": "classification",
    "category": "classification",
    "algorithm": "procedure",
    "procedure": "procedure",
    "process": "procedure",
    "step": "procedure",
    "prevent": "prevent",
    "prevention": "prevent",
    "detect": "detect",
    "detection": "detect",
    "avoid": "prevent",
    "avoidance": "prevent",
    "differ": "compare",
    "difference": "compare",
    "differentiate": "compare",
    "distinguish": "compare",
    "contrast": "compare",
    "compare": "compare",
    "versus": "compare",
    "recommend": "recommend",
    "recommendation": "recommend",
    "recommender": "recommend",
}

_STEM_SUFFIXES = ("ations", "ation", "ions", "ion", "ings", "ing", "ies", "ed", "es", "s")

INTENT_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("definition", r"\b(?:define|definition|meaning of|what is|what are|what do you mean by)\b"),
    ("explanation", r"\b(?:explain|describe|discuss|elaborate|outline)\b"),
    ("comparison", r"\b(?:compare|comparison|differentiate|distinguish|contrast|differences?|differs?|versus|\bvs\.?)\b"),
    ("advantages", r"\b(?:advantages?|benefits?|merits?)\b"),
    ("disadvantages", r"\b(?:disadvantages?|drawbacks?|limitations?|demerits?)\b"),
    ("application", r"\b(?:applications?|use cases?|uses of)\b"),
    ("working", r"\b(?:working|how (?:does )?(?:it|they|this) work|mechanism|operation)\b"),
    ("algorithm", r"\b(?:algorithm|steps?|procedure)\b"),
    ("derivation", r"\b(?:derive|prove|proof|show mathematically)\b"),
    ("calculation", r"\b(?:calculate|compute|number of|find the)\b"),
    ("architecture", r"\b(?:architecture|components?|layers?|structure)\b"),
    ("classification", r"\b(?:types?|categories|classification|kinds? of)\b"),
    ("significance", r"\b(?:significance|importance of)\b"),
    ("features", r"\b(?:functions? of|features of|characteristics of)\b"),
)

OUTPUT_FROM_INTENT = {
    "definition": "definition",
    "explanation": "explanation",
    "comparison": "comparison",
    "advantages": "advantages",
    "disadvantages": "disadvantages",
    "application": "applications",
    "working": "mechanism",
    "algorithm": "procedure",
    "derivation": "derivation",
    "calculation": "calculation",
    "architecture": "architecture",
    "classification": "classification",
    "significance": "significance",
    "features": "features",
}

# Soft task families: describing / defining / showing the working of the same entity.
# Academic verbs (explain / what is / describe / define / discuss) live here.
SOFT_INTENTS = {"explanation", "working", "architecture", "algorithm", "definition"}
EXCLUSIVE_INTENTS = {
    "application", "advantages", "disadvantages", "classification",
    "calculation", "derivation", "comparison", "significance", "features",
}
HARD_FACETS = {
    "applications", "advantages_disadvantages", "enumerate_types",
    "calculation", "derivation", "comparison", "features", "properties", "challenges",
    "significance",
}
HARD_OUTPUTS = {
    "applications", "advantages", "disadvantages", "classification",
    "calculation", "derivation", "comparison", "significance", "features",
    "properties", "challenges", "advantages_disadvantages", "enumerate_types",
}
SOFT_OUTPUTS = {"definition", "explanation", "mechanism", "architecture", "procedure"}

_FILLER_VERBS = set(ACADEMIC_QUESTION_VERBS) | {
    "explain", "discuss", "describe", "what", "list", "write", "following",
    "question", "define", "state", "give", "mention", "outline", "elaborate",
    "how", "does", "work", "works", "working",
}

_EXAM_ACRONYM_STOP = {
    "QB", "QP", "ID", "OR", "AND", "THE", "FOR", "NOT", "ANY", "ALL",
    "MAY", "NOV", "DEC", "JAN", "FEB", "MAR", "APR", "JUN", "JUL",
    "AUG", "SEP", "OCT", "PDF", "OCR", "BE", "ME", "II", "III", "IV",
    "VI", "HTTP", "HTTPS", "SEM", "PAGE", "MARKS",
}


def _light_stem(token: str) -> str:
    t = (token or "").lower()
    t = _SYNONYM_ROOT.get(t, t)
    for suf in _STEM_SUFFIXES:
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            t = t[: -len(suf)]
            break
    return _SYNONYM_ROOT.get(t, t)


_WEAK_STEMS = {_light_stem(w) for w in WEAK_GENERIC} | set(WEAK_GENERIC)
_ULTRA_STEMS = {_light_stem(w) for w in ULTRA_GENERIC} | set(ULTRA_GENERIC)
_SOFT_STEMS = {_light_stem(w) for w in SOFT_FOCUS} | set(SOFT_FOCUS)
_STOP_STEMS = {_light_stem(w) for w in STOPWORDS} | set(STOPWORDS)


def _edit_distance_le1(a: str, b: str) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if min(len(a), len(b)) < 6:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    i = j = diffs = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        diffs += 1
        if diffs > 1:
            return False
        j += 1
    return True


def _unpack_glued_token(tok: str) -> List[str]:
    raw = tok.strip("'\",.;:`")
    if len(raw) < 16:
        return [tok] if tok else []
    low = raw.lower()
    for head in _TRAIL_HEADS:
        if low.endswith(head) and len(low) - len(head) >= 6:
            stem = raw[: len(raw) - len(head)]
            return [p for p in _unpack_glued_token(stem) if p] + [head]
    for root in _LEAD_ROOTS:
        if low.startswith(root) and len(low) - len(root) >= 6:
            rest = raw[len(root) :]
            return [root] + [p for p in _unpack_glued_token(rest) if p]
    return [tok]


def preprocess_question_text(text: str) -> str:
    """Strip layout tags and split obvious OCR concatenations. No subject catalog."""
    t = _LAYOUT_NOISE_RE.sub(" ", text or "")
    t = re.sub(r"\b\d+\s*(?:marks?|m)\b", " ", t, flags=re.I)
    for glue in _GLUE_WORDS:
        t = re.sub(rf"([A-Za-z]{{4,}})({glue})([A-Za-z]{{4,}})", rf"\1 \2 \3", t, flags=re.I)
    parts: List[str] = []
    for tok in re.findall(r"[A-Za-z0-9+\-']+|[^\sA-Za-z0-9+\-']+", t):
        if re.fullmatch(r"[A-Za-z0-9+\-']+", tok) and len(tok) >= 16:
            parts.extend(_unpack_glued_token(tok))
        else:
            parts.append(tok)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]{2,}", (text or "").lower())


def _content_tokens(text: str) -> List[str]:
    out = []
    for tok in _tokens(text):
        if tok in _FILLER_VERBS or tok in STOPWORDS:
            continue
        st = _light_stem(tok)
        if len(st) < 3 or st in _STOP_STEMS:
            continue
        out.append(st)
    return out


def _is_weak(token: str) -> bool:
    t = _light_stem(token)
    return t in _WEAK_STEMS or token in _WEAK_STEMS


def _is_ultra(token: str) -> bool:
    t = _light_stem(token)
    if t in _ULTRA_STEMS or token in _ULTRA_STEMS:
        return True
    for s in _ULTRA_STEMS:
        if len(s) >= 6 and len(t) >= 6 and (t.startswith(s) or s.startswith(t)):
            return True
    return False


def _is_soft(token: str) -> bool:
    t = _light_stem(token)
    if t in _SOFT_STEMS or token in _SOFT_STEMS:
        return True
    for s in _SOFT_STEMS:
        if len(s) >= 5 and len(t) >= 5 and (t.startswith(s) or s.startswith(t)):
            return True
    return False


def _is_specific(token: str) -> bool:
    t = _light_stem(token)
    if t in _STOP_STEMS or t in _FILLER_VERBS or t in STOPWORDS:
        return False
    if _is_weak(t) or _is_soft(t) or _is_ultra(t):
        return False
    if len(t) >= 5:
        return True
    # Short names / acronyms (cnn, rnn, gru, prim, tcp) after stopword filtering.
    return len(t) >= 3


def detect_intent_families(text: str) -> Set[str]:
    low = (text or "").lower()
    found: Set[str] = set()
    for name, pattern in INTENT_PATTERNS:
        if re.search(pattern, low, re.I):
            found.add(name)
    if "?" in (text or "") and not found:
        found.add("definition" if re.search(r"\bwhat\b", low) else "explanation")
    if not found:
        found.add("explanation")
    return found


def detect_requested_outputs(text: str) -> Set[str]:
    outs: Set[str] = set()
    facet = requested_output_focus(text)
    if facet in {
        "applications", "advantages_disadvantages", "enumerate_types",
        "features", "properties", "challenges",
    }:
        mapping = {
            "applications": "applications",
            "advantages_disadvantages": "advantages",
            "enumerate_types": "classification",
            "features": "features",
            "properties": "properties",
            "challenges": "disadvantages",
            "significance": "significance",
        }
        outs.add(mapping[facet])
    for intent in detect_intent_families(text):
        mapped = OUTPUT_FROM_INTENT.get(intent)
        if mapped:
            outs.add(mapped)
    return outs or {"explanation"}


def primary_facet_of(text: str) -> Optional[str]:
    return requested_output_focus(text)


def extract_comparison_spans(text: str) -> List[str]:
    low = (text or "").strip()
    spans: List[str] = []
    patterns = [
        r"(?:differentiate|distinguish|compare|difference)\s+(?:between\s+)?(.+?)\s+and\s+(.+?)(?:[.?;]|$)",
        r"(?:differs?|different)\s+from\s+(?:a |an |the )?(.+?)(?:[.?;]|$)",
        r"(?:compared|as compared)\s+(?:to|with)\s+(?:a |an |the )?(.+?)(?:[.?;]|$)",
        r"\b([A-Za-z][A-Za-z0-9+\-]{1,20})\s+(?:vs\.?|versus)\s+([A-Za-z][A-Za-z0-9+\-]{1,20})\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, low, re.I):
            spans.extend(g.strip(" .,;:") for g in m.groups() if g and g.strip())
    return spans[:6]


def extract_named_tokens(text: str) -> Set[str]:
    names = {
        m.lower()
        for m in re.findall(r"\b[A-Z]{2,8}\b", text or "")
        if m not in _EXAM_ACRONYM_STOP
    }
    names.update(m.lower() for m in re.findall(r"\b\d+[A-Za-z]{1,4}\b", text or ""))
    return names


def _core_entity_phrase(text: str) -> str:
    """Longest technical noun phrase after stripping the academic ask."""
    t = (text or "").strip()
    t = re.sub(r"^(?:Q\.?\s*\d+[a-z]?[\).:]?\s*)", "", t, flags=re.I)
    t = re.sub(
        r"^(?:what\s+is|what\s+are|what\s+do\s+you\s+mean\s+by|define|explain|describe|"
        r"discuss|elaborate|outline|state|list|write)\s+",
        "",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"\b(?:and\s+)?(?:explain|describe|discuss)\s+how\s+it\s+differs?.*$",
        "",
        t,
        flags=re.I,
    )
    t = re.sub(r"\b(?:how\s+does\s+it\s+differ.*)$", "", t, flags=re.I)
    t = re.sub(r"[?.!,;:]+$", "", t).strip(" -")
    t = re.split(
        r"\b(?:and how it|and explain how|and its|differs? from|compared (?:to|with)|versus| vs\.? )\b",
        t,
        maxsplit=1,
        flags=re.I,
    )[0].strip(" -")
    t = re.sub(r"\band\s*$", "", t, flags=re.I).strip(" -")
    return t.strip()[:160]


def _overlay_constraints(raw: str, constraints: List[str]) -> List[str]:
    low = (raw or "").lower()
    extra = list(constraints)
    if re.search(r"\bdiffers?\b|\bdifferent from\b|\bas compared\b", low):
        if "comparison" not in extra:
            extra.append("comparison")
    if re.search(
        r"\bapply\b|\bgiven (?:the )?(?:following|transaction|data|graph|string|keys?|database)\b",
        low,
    ):
        if "worked_example" not in extra:
            extra.append("worked_example")
    if re.search(r"\bany\s+(?:\d+|two|three|four|five|six|seven)\b", low):
        if "enumerate_count" not in extra:
            extra.append("enumerate_count")
    if re.search(r"\bsignificance\b|\bimportance of\b", low):
        if "significance" not in extra:
            extra.append("significance")
    return extra


@dataclass
class SemanticSignature:
    text: str
    core_entity: str
    core_tokens: Set[str]
    specific_tokens: Set[str]
    intents: Set[str]
    outputs: Set[str]
    primary_facet: Optional[str]
    constraints: List[str]
    comparison_targets: List[str]
    comparison_tokens: Set[str]
    named_tokens: Set[str]
    qualifiers: Set[str]
    domain_tokens: Set[str]
    entities: List[str]
    scope: str = "standard"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "core_entity": self.core_entity,
            "core_tokens": sorted(self.core_tokens),
            "specific_tokens": sorted(self.specific_tokens),
            "intents": sorted(self.intents),
            "outputs": sorted(self.outputs),
            "primary_facet": self.primary_facet,
            "constraints": list(self.constraints),
            "comparison_targets": list(self.comparison_targets),
            "named_tokens": sorted(self.named_tokens),
            "qualifiers": sorted(self.qualifiers),
            "domain_tokens": sorted(self.domain_tokens),
            "entities": list(self.entities),
            "scope": self.scope,
        }


def build_semantic_signature(text: str) -> SemanticSignature:
    raw = preprocess_question_text(text or "")
    intents = detect_intent_families(raw)
    outputs = detect_requested_outputs(raw)
    facet = primary_facet_of(raw)
    constraints = _overlay_constraints(raw, extract_constraints(raw))
    entities = extract_entities(raw)
    core = _core_entity_phrase(raw)
    core_tokens = set(_content_tokens(core) or _content_tokens(raw)[:8])
    named = extract_named_tokens(raw)
    specific = {t for t in core_tokens if _is_specific(t)} | named
    if not specific:
        specific = {t for t in _content_tokens(raw) if _is_specific(t)} | named
    comps = extract_comparison_spans(raw)
    comp_tokens: Set[str] = set()
    for span in comps:
        comp_tokens.update(_content_tokens(span))
        comp_tokens.update({t for t in _tokens(span) if _is_specific(t) or t in named})
    domain = set(_content_tokens(raw)) - core_tokens
    qualifiers = {t for t in core_tokens if t in {"prevent", "detect", "linear", "logistic"}}
    for tok in _content_tokens(raw):
        if tok in {"prevent", "detect", "linear", "logistic"}:
            qualifiers.add(tok)
    scope = "detailed" if len(raw.split()) > 28 or re.search(r"\bin detail\b", raw, re.I) else "standard"
    return SemanticSignature(
        text=raw,
        core_entity=core or (entities[0] if entities else ""),
        core_tokens=core_tokens,
        specific_tokens=specific,
        intents=intents,
        outputs=outputs,
        primary_facet=facet,
        constraints=constraints,
        comparison_targets=comps,
        comparison_tokens=comp_tokens,
        named_tokens=named,
        qualifiers=qualifiers,
        domain_tokens={t for t in domain if not _is_ultra(t)},
        entities=entities,
        scope=scope,
    )


def intents_compatible(a: Set[str], b: Set[str]) -> bool:
    """Academic verbs in the same family are compatible; exclusive facets are not."""
    if not a or not b:
        return False
    a_ex, b_ex = a & EXCLUSIVE_INTENTS, b & EXCLUSIVE_INTENTS
    if a_ex != b_ex:
        return False
    if a & b:
        return True
    if a <= SOFT_INTENTS and b <= SOFT_INTENTS:
        return True
    return False


def _hard_outputs(outs: Set[str], facet: Optional[str]) -> Set[str]:
    hard = set(outs or ()) & HARD_OUTPUTS
    if facet in HARD_FACETS:
        hard.add(facet)
    return hard


def outputs_compatible(a: Set[str], b: Set[str], facet_a: Optional[str], facet_b: Optional[str]) -> bool:
    ha, hb = _hard_outputs(a, facet_a), _hard_outputs(b, facet_b)
    if ha != hb:
        return False
    if a & b:
        return True
    if a <= SOFT_OUTPUTS and b <= SOFT_OUTPUTS:
        return True
    return False


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _containment(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    smaller, larger = (a, b) if len(a) <= len(b) else (b, a)
    return len(smaller & larger) / len(smaller)


@dataclass
class SignatureMatch:
    relationship: str
    score: float
    confidence: float
    reason: str
    contradictions: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


def _core_initials(sig: SemanticSignature) -> str:
    skip = _ULTRA_STEMS | _SOFT_STEMS | _STOP_STEMS | {"architecture", "algorithm", "procedure"}
    toks = [t for t in _content_tokens(sig.core_entity) if t not in skip]
    return "".join(t[0] for t in toks if t)


def _expansion_tokens(sig: SemanticSignature, acronym: str) -> Set[str]:
    skip = _ULTRA_STEMS | _SOFT_STEMS | _STOP_STEMS | {"architecture", "algorithm", "procedure"}
    toks = [t for t in _content_tokens(sig.core_entity) if t not in skip]
    if "".join(t[0] for t in toks) == acronym:
        return set(toks)
    return set()


def _named_compatible(a: SemanticSignature, b: SemanticSignature) -> bool:
    """True when disjoint surface acronyms are just expansions of the other phrase."""
    if not a.named_tokens or not b.named_tokens:
        if a.named_tokens and not b.named_tokens:
            return any(_core_initials(b) == n for n in a.named_tokens)
        if b.named_tokens and not a.named_tokens:
            return any(_core_initials(a) == n for n in b.named_tokens)
        return True
    if not a.named_tokens.isdisjoint(b.named_tokens):
        return True
    init_a, init_b = _core_initials(a), _core_initials(b)
    if any(init_b == n for n in a.named_tokens) or any(init_a == n for n in b.named_tokens):
        return True
    return False


def _absorb_split_compounds(only_left: Set[str], all_right: Set[str], shared: Set[str]) -> Set[str]:
    """Treat OCR-split compounds (auto + encoder) as the concatenated token."""
    remaining = set(only_left)
    left = list(only_left)
    consumed: Set[str] = set()
    for i, x in enumerate(left):
        if x in consumed:
            continue
        for y in left[i + 1 :]:
            if y in consumed:
                continue
            matched = None
            for cat in (x + y, y + x):
                if cat in all_right:
                    matched = cat
                    break
                for z in all_right:
                    if abs(len(cat) - len(z)) <= 1 and _edit_distance_le1(cat, z):
                        matched = z
                        break
                if matched:
                    break
            if not matched:
                continue
            consumed.add(x)
            consumed.add(y)
            shared.add(matched)
    remaining -= consumed
    return remaining


def related_canonical_key(shared_specific: Set[str]) -> Optional[str]:
    """Canonical entity for a RELATED group. Generic filler is never a key."""
    keys = [t for t in shared_specific if _is_specific(t)]
    keys.sort(key=lambda t: (-len(t), t))
    return keys[0] if keys else None


def compare_semantic_signatures(
    a: SemanticSignature,
    b: SemanticSignature,
    lexical_sim: float = 0.0,
    embedding_sim: Optional[float] = None,
) -> SignatureMatch:
    """
    Contradiction-first structured match.

    Embedding similarity is supporting evidence only and cannot create a
    semantic repeat or bypass an entity/intent veto.
    """
    contradictions: List[str] = []

    if a.named_tokens and b.named_tokens and a.named_tokens.isdisjoint(b.named_tokens):
        if not _named_compatible(a, b):
            contradictions.append(
                f"contradictory named entities {sorted(a.named_tokens)} vs {sorted(b.named_tokens)}"
            )

    spec_a = {t for t in a.specific_tokens if _is_specific(t)} | set(a.named_tokens)
    spec_b = {t for t in b.specific_tokens if _is_specific(t)} | set(b.named_tokens)
    drop = a.comparison_tokens | b.comparison_tokens | _SOFT_STEMS
    shared_specific = spec_a & spec_b
    only_a = spec_a - spec_b - drop
    only_b = spec_b - spec_a - drop
    if _named_compatible(a, b):
        for n in a.named_tokens:
            consumed = _expansion_tokens(b, n)
            if consumed or _core_initials(b) == n:
                only_a.discard(n)
                only_b -= consumed
                shared_specific.add(n)
        for n in b.named_tokens:
            consumed = _expansion_tokens(a, n)
            if consumed or _core_initials(a) == n:
                only_b.discard(n)
                only_a -= consumed
                shared_specific.add(n)

    # OCR near-matches against the other question's full token set.
    all_a = spec_a | a.core_tokens | a.named_tokens
    all_b = spec_b | b.core_tokens | b.named_tokens
    for xa in list(only_a):
        if any(_edit_distance_le1(xa, xb) for xb in all_b):
            only_a.discard(xa)
            shared_specific.add(xa)
    for xb in list(only_b):
        if any(_edit_distance_le1(xb, xa) for xa in all_a | shared_specific):
            only_b.discard(xb)
            shared_specific.add(xb)

    only_a = _absorb_split_compounds(only_a, all_b | shared_specific, shared_specific)
    only_b = _absorb_split_compounds(only_b, all_a | shared_specific, shared_specific)
    only_a -= shared_specific
    only_b -= shared_specific

    if only_a and only_b:
        contradictions.append(
            f"contradictory technical entities {sorted(only_a)} vs {sorted(only_b)}"
        )

    exclusive_qual_pairs = [
        ({"prevent"}, {"detect"}),
        ({"linear"}, {"logistic"}),
    ]
    for qa, qb in exclusive_qual_pairs:
        if (a.qualifiers & qa and b.qualifiers & qb) or (a.qualifiers & qb and b.qualifiers & qa):
            contradictions.append(
                f"contradictory qualifiers {sorted(a.qualifiers & (qa | qb))} vs {sorted(b.qualifiers & (qa | qb))}"
            )

    cons_a, cons_b = set(a.constraints), set(b.constraints)
    if ("numerical_calculation" in cons_a) != ("numerical_calculation" in cons_b) and (
        "numerical_calculation" in cons_a or "numerical_calculation" in cons_b
    ):
        contradictions.append("calculation vs non-calculation")
    if ("worked_example" in cons_a) != ("worked_example" in cons_b) and (
        "worked_example" in cons_a or "worked_example" in cons_b
    ):
        contradictions.append("worked example vs explanation")
    if ("derivation" in cons_a) != ("derivation" in cons_b) and (
        "derivation" in cons_a or "derivation" in cons_b
    ):
        contradictions.append("derivation vs non-derivation")
    if ("enumerate_count" in cons_a) != ("enumerate_count" in cons_b):
        contradictions.append("count-limited enumeration vs open-ended ask")
    if ("significance" in cons_a) != ("significance" in cons_b):
        contradictions.append("significance vs non-significance")

    intent_ok = intents_compatible(a.intents, b.intents)
    output_ok = outputs_compatible(a.outputs, b.outputs, a.primary_facet, b.primary_facet)
    if not intent_ok:
        contradictions.append(f"incompatible intent {sorted(a.intents)} vs {sorted(b.intents)}")
    if not output_ok:
        contradictions.append(
            f"incompatible requested output {a.primary_facet or sorted(a.outputs)} vs {b.primary_facet or sorted(b.outputs)}"
        )

    core_a = {t for t in a.core_tokens if not _is_weak(t) and not _is_ultra(t) and t not in _STOP_STEMS}
    core_b = {t for t in b.core_tokens if not _is_weak(t) and not _is_ultra(t) and t not in _STOP_STEMS}
    entity_j = _jaccard(core_a, core_b)
    entity_c = _containment(core_a, core_b)
    spec_j = _jaccard(spec_a, spec_b)
    spec_c = _containment(spec_a, spec_b)
    entity_s = max(entity_j, entity_c, spec_j, spec_c)
    if shared_specific and not only_a and not only_b:
        entity_s = max(entity_s, 0.72)
    if a.core_entity and b.core_entity:
        ca = " ".join(_content_tokens(a.core_entity))
        cb = " ".join(_content_tokens(b.core_entity))
        if ca and cb and (ca in cb or cb in ca) and min(len(ca.split()), len(cb.split())) >= 2:
            entity_s = max(entity_s, 0.72)

    phrase_a = {t for t in a.core_tokens if not _is_ultra(t) and t not in _STOP_STEMS}
    phrase_b = {t for t in b.core_tokens if not _is_ultra(t) and t not in _STOP_STEMS}
    phrase_overlap = _jaccard(phrase_a, phrase_b)

    intent_s = 1.0 if intent_ok else 0.0
    output_s = 1.0 if output_ok else 0.0
    if a.constraints or b.constraints:
        inter = cons_a & cons_b
        union = cons_a | cons_b
        cons_s = (len(inter) / len(union)) if union else 1.0
        if not inter and not (cons_a and cons_b):
            cons_s = 0.55
    else:
        cons_s = 1.0

    if a.comparison_tokens or b.comparison_tokens:
        comp_s = max(
            _jaccard(a.comparison_tokens, b.comparison_tokens),
            _containment(a.comparison_tokens, b.comparison_tokens),
        )
        if not a.comparison_tokens or not b.comparison_tokens:
            comp_s = 0.4
    else:
        comp_s = 0.75

    scope_s = 1.0 if a.scope == b.scope else 0.7
    domain_s = max(_jaccard(a.domain_tokens, b.domain_tokens), _containment(a.domain_tokens, b.domain_tokens))
    if not a.domain_tokens or not b.domain_tokens:
        domain_s = 0.5

    emb = 0.0
    if embedding_sim is not None:
        try:
            emb = max(0.0, min(1.0, float(embedding_sim)))
        except (TypeError, ValueError):
            emb = 0.0

    score = (
        0.34 * entity_s
        + 0.20 * intent_s
        + 0.16 * output_s
        + 0.12 * cons_s
        + 0.10 * comp_s
        + 0.04 * scope_s
        + 0.04 * domain_s
    )
    # Embedding may nudge ranking but cannot manufacture a match.
    score = min(1.0, score + 0.03 * emb)
    score = round(score, 3)

    evidence = {
        "entity_match": round(entity_s, 3),
        "intent_match": round(intent_s, 3),
        "output_match": round(output_s, 3),
        "constraint_match": round(cons_s, 3),
        "comparison_target_match": round(comp_s, 3),
        "scope_match": round(scope_s, 3),
        "domain_match": round(domain_s, 3),
        "embedding_support": round(emb, 3),
        "lexical_similarity": round(float(lexical_sim or 0.0), 3),
        "shared_specific_tokens": sorted(shared_specific),
        "intents_a": sorted(a.intents),
        "intents_b": sorted(b.intents),
        "core_entity_a": a.core_entity,
        "core_entity_b": b.core_entity,
        "contradictions": contradictions,
    }

    hard_veto = any(
        x.startswith("contradictory")
        or x.startswith("incompatible")
        or "vs non-" in x
        or x.startswith("worked example")
        or x.startswith("calculation vs")
        or x.startswith("derivation vs")
        or x.startswith("count-limited")
        for x in contradictions
    )

    quality_specific = (
        len(shared_specific) >= 2
        or any(len(t) >= 7 for t in shared_specific)
        or bool(shared_specific & (a.named_tokens | b.named_tokens))
    )
    phrase_ok = (
        phrase_overlap >= 0.68
        and not only_a
        and not only_b
        and entity_s >= 0.40
    )
    specific_enough = quality_specific or phrase_ok
    # Extra specific entity on one side (pooling in CNN vs CNN architecture).
    asymmetric = bool((only_a and not only_b) or (only_b and not only_a))

    if (
        not hard_veto
        and not asymmetric
        and intent_ok
        and output_ok
        and entity_s >= 0.48
        and specific_enough
        and score >= 0.58
    ):
        shared_intents = a.intents & b.intents or a.intents
        shared_outs = a.outputs & b.outputs or a.outputs
        entity_label = a.core_entity or b.core_entity or (a.entities[:1] or [""])[0]
        if (a.intents <= SOFT_INTENTS and b.intents <= SOFT_INTENTS) and (
            {"definition"} & (a.intents | b.intents) and {"explanation"} & (a.intents | b.intents)
        ):
            why = (
                f"Same technical entity and same underlying explanatory/definition ask; "
                f"wording differs only in academic framing."
            )
        else:
            why = (
                f"same core entity ({entity_label}); "
                f"compatible intent {sorted(shared_intents)}; "
                f"compatible requested output {sorted(shared_outs)}"
            )
        if a.comparison_targets and b.comparison_targets:
            why += "; same comparison target"
        return SignatureMatch(
            "SEMANTIC_REPEAT",
            score,
            max(score, 0.62),
            why,
            contradictions,
            evidence,
        )

    shared_for_related = {t for t in shared_specific if _is_specific(t)}
    shared_core_specific = {t for t in (core_a & core_b) if _is_specific(t)}
    meaningful = bool(shared_for_related) or bool(shared_core_specific)
    if meaningful:
        if hard_veto:
            reason = contradictions[0] + " — related topic only"
        elif not intent_ok or not output_ok:
            reason = f"shared concept but different ask ({sorted(a.intents)} vs {sorted(b.intents)})"
        elif asymmetric:
            extra = sorted(only_a or only_b)
            reason = f"shared entity with extra specific focus {extra}"
        else:
            reason = "shared technical concept with different entity or constraint"
        rel_conf = round(max(entity_s, 0.35 if asymmetric else 0.0), 3)
        return SignatureMatch("RELATED_TOPIC", rel_conf, rel_conf, reason, contradictions, evidence)

    return SignatureMatch(
        "DIFFERENT",
        0.0,
        0.0,
        "Insufficient evidence of a conceptual relationship",
        contradictions,
        evidence,
    )


def signature_from_bundle(bundle: Optional[Dict[str, Any]], text: str) -> SemanticSignature:
    if bundle and isinstance(bundle.get("semantic_signature"), SemanticSignature):
        return bundle["semantic_signature"]
    stored = (bundle or {}).get("semantic_signature") if bundle else None
    if isinstance(stored, dict) and stored.get("core_entity") is not None:
        d = stored
        return SemanticSignature(
            text=text,
            core_entity=str(d.get("core_entity") or ""),
            core_tokens=set(d.get("core_tokens") or []),
            specific_tokens=set(d.get("specific_tokens") or []),
            intents=set(d.get("intents") or []),
            outputs=set(d.get("outputs") or []),
            primary_facet=d.get("primary_facet"),
            constraints=list(d.get("constraints") or []),
            comparison_targets=list(d.get("comparison_targets") or []),
            comparison_tokens=set(_content_tokens(" ".join(d.get("comparison_targets") or []))),
            named_tokens=set(d.get("named_tokens") or []),
            qualifiers=set(d.get("qualifiers") or []),
            domain_tokens=set(d.get("domain_tokens") or []),
            entities=list(d.get("entities") or []),
            scope=str(d.get("scope") or "standard"),
        )
    return build_semantic_signature(text)


def candidate_pair_indices(
    signatures: Sequence[SemanticSignature],
    embeddings: Any = None,
    embed_floor: float = 0.60,
) -> List[Tuple[int, int]]:
    """
    Cheap candidate generation: inverted index on specific tokens, plus
    embedding neighbours as advisory candidates. Classifier still decides.
    """
    n = len(signatures)
    pairs: Set[Tuple[int, int]] = set()
    index: Dict[str, List[int]] = defaultdict(list)
    for i, sig in enumerate(signatures):
        keys = {t for t in sig.specific_tokens if _is_specific(t)} | {
            t for t in sig.core_tokens if not _is_ultra(t) and not _is_weak(t)
        }
        if not keys:
            keys = {t for t in sig.core_tokens if not _is_ultra(t)}
        for tok in keys:
            index[tok].append(i)
    for ids in index.values():
        if len(ids) < 2:
            continue
        uniq = sorted(set(ids))
        for a in range(len(uniq)):
            for b in range(a + 1, len(uniq)):
                pairs.add((uniq[a], uniq[b]))

    if embeddings is not None:
        try:
            import numpy as np

            mat = np.asarray(embeddings)
            if mat.ndim == 2 and mat.shape[0] == n:
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                sim = (mat @ mat.T) / (norms @ norms.T)
                for i in range(n):
                    for j in range(i + 1, n):
                        if float(sim[i, j]) >= embed_floor:
                            pairs.add((i, j))
        except Exception:
            pass
    return sorted(pairs)
