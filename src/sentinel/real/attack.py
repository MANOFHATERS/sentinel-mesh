"""The real MITRE ATT&CK catalogue as a knowledge base.

The bundled knowledge base is a small curated corpus whose prose was written for this
project. This module builds the same kind of knowledge base — same chunking, same hybrid
retrieval, same citations — from MITRE's own published STIX 2.1 bundle
(``enterprise-attack.json`` from github.com/mitre-attack/attack-stix-data), so a query can
be answered from the real catalogue and compared with the curated one.

ATT&CK content is MITRE's, used here under MITRE's published terms of use, and every hit
links back to attack.mitre.org. Only ATT&CK techniques and sub-techniques are loaded; CVE
and advisory data are not part of this corpus.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from sentinel.core.errors import SentinelError
from sentinel.core.schemas import TECHNIQUE_ID_PATTERN
from sentinel.kb.corpus import Corpus, DocumentKind, KBDocument
from sentinel.kb.retrieve import KnowledgeBase

__all__ = ["AttackError", "build_attack_kb", "documents_from_stix", "load_bundle", "search_view"]

ATTACK_URL: Final[str] = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
)
DEFAULT_PATH: Final[Path] = Path("data/real/enterprise-attack.json")

_CITATION = re.compile(r"\(Citation:[^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\((?:https?://)[^)]*\)")
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_SPACES = re.compile(r"[ \t]+")


class AttackError(SentinelError):
    """The ATT&CK bundle is missing or is not a STIX bundle."""


def _clean(text: str) -> str:
    text = _CITATION.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _TAG.sub("", text)
    text = text.replace("`", "")
    text = _SPACES.sub(" ", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()


def load_bundle(path: Path | str = DEFAULT_PATH) -> dict[str, Any]:
    file = Path(path)
    if not file.is_file():
        raise AttackError(f"{file} not found; run scripts/fetch_real_data.py")
    try:
        bundle = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AttackError(f"{file} is not readable JSON: {exc}") from exc
    if not isinstance(bundle, dict) or bundle.get("type") != "bundle":
        raise AttackError(f"{file} is not a STIX bundle")
    return bundle


def _external_id(obj: Mapping[str, Any]) -> str | None:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack" and ref.get("external_id"):
            return str(ref["external_id"])
    return None


def documents_from_stix(bundle: Mapping[str, Any]) -> list[KBDocument]:
    """ATT&CK techniques and sub-techniques as knowledge-base documents.

    Revoked and deprecated techniques are dropped, and so is a sub-technique whose parent
    is not present (a corpus with a dangling parent is refused by the knowledge base).
    """
    documents: dict[str, KBDocument] = {}
    for obj in bundle.get("objects", []):
        if (
            obj.get("type") != "attack-pattern"
            or obj.get("revoked")
            or obj.get("x_mitre_deprecated")
        ):
            continue
        technique_id = _external_id(obj)
        if not technique_id or not TECHNIQUE_ID_PATTERN.match(technique_id):
            continue
        description = _clean(str(obj.get("description", "")))
        if not description:
            continue
        sections: list[tuple[str, str]] = [("Description", description)]
        detection = _clean(str(obj.get("x_mitre_detection", "")))
        if detection:
            sections.append(("Detection", detection))
        tactics = tuple(
            str(p["phase_name"])
            for p in obj.get("kill_chain_phases", [])
            if p.get("kill_chain_name") == "mitre-attack"
        )
        documents[technique_id] = KBDocument(
            doc_id=technique_id,
            kind=DocumentKind.TECHNIQUE,
            title=str(obj.get("name", technique_id)),
            sections=tuple(sections),
            tactics=tactics,
            platforms=tuple(str(p).lower() for p in obj.get("x_mitre_platforms", [])),
        )
    return [d for d in documents.values() if d.parent_id is None or d.parent_id in documents]


def build_attack_kb(path: Path | str = DEFAULT_PATH) -> KnowledgeBase:
    documents = documents_from_stix(load_bundle(path))
    if not documents:
        raise AttackError("no ATT&CK techniques found in the bundle")
    return KnowledgeBase.build(Corpus(documents=tuple(documents)))


def _hit(hit: Any) -> dict[str, Any]:
    chunk = hit.chunk
    is_attack = chunk.kind == "technique"
    return {
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "section": chunk.section,
        "excerpt": chunk.excerpt[:420],
        "relevance": round(float(hit.relevance), 3),
        "url": f"https://attack.mitre.org/techniques/{chunk.doc_id.replace('.', '/')}/"
        if is_attack
        else None,
        "kind": chunk.kind,
    }


def search_view(kb: KnowledgeBase, query: str, *, k: int = 5) -> list[dict[str, Any]]:
    """Retrieval hits for ``query`` as JSON-able rows. An unanswerable query is an empty list."""
    hits = kb.search(query, k=k, kinds={DocumentKind.TECHNIQUE})
    return [_hit(h) for h in hits]
