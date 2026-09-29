"""The knowledge-base corpus: ATT&CK techniques, vulnerabilities, playbooks.

PRD Section 5.5.6 specifies *"a FAISS vector index over MITRE ATT&CK technique
descriptions and the NVD/CVE corpus, chunked and embedded once at build time"*, and
Section 7.1 lists both as sprint data sources.

Why the corpus ships in the repository
--------------------------------------
Downloading the ATT&CK STIX bundle and an NVD snapshot at test time would make the
suite non-reproducible (the upstream content changes), network-dependent (so it
fails in CI and offline) and slow. The same reasoning that keeps ``torch`` out of
:mod:`sentinel.ml.nn` applies here: the corpus is a versioned in-repo artifact, and
:class:`Corpus` loads from a path, so pointing it at a real STIX-derived export is a
call-site change rather than a rewrite.

Every description and remediation note is written for this project. Technique and
vulnerability identifiers, names and product names are factual identifiers and are
used as such; the prose is not copied from the upstream catalogues.

What a document carries, and why
--------------------------------
*   ``aliases`` — the single highest-leverage field for retrieval quality. Analysts
    query in operator vocabulary ("pass the hash", "mfa fatigue", "log4shell"), not
    in catalogue titles, and an alias list is the cheapest possible bridge. The
    measured effect is reported in ``docs/BUILD_PLAN.md``.
*   ``techniques`` — cross-links from a vulnerability or playbook to the ATT&CK ids
    it relates to. :meth:`Corpus.validate` rejects a cross-link that points at a
    document the corpus does not contain, for the same reason
    :class:`~sentinel.core.schemas.InvestigationReport` rejects a citation that
    points at nothing.
*   ``detection`` / ``remediation`` / ``steps`` — these are chunked *separately* from
    the description, because an analyst asking "what do I do about this" and one
    asking "what is this" want different chunks of the same document, and a single
    blended chunk serves neither well.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final

from sentinel.core.errors import SentinelError
from sentinel.core.schemas import CVE_ID_PATTERN, TECHNIQUE_ID_PATTERN, EvidenceKind

__all__ = [
    "DEFAULT_CORPUS_DIR",
    "Corpus",
    "CorpusError",
    "DocumentKind",
    "KBDocument",
    "load_default_corpus",
]

DEFAULT_CORPUS_DIR: Final[Path] = Path(__file__).parent / "data"

#: Recognised advisory-identifier shapes for non-CVE supply-chain entries and for
#: this project's own playbooks. Kept loose on purpose: real advisory namespaces
#: (GHSA, OSV, RUSTSEC, vendor ids) are not going to agree on a format.
_ADVISORY_PREFIXES: Final[tuple[str, ...]] = ("GHSA-", "OSV-", "RUSTSEC-", "PYSEC-")
_PLAYBOOK_PREFIX: Final[str] = "PB-"


class CorpusError(SentinelError):
    """A corpus file is malformed, or a document violates a corpus invariant."""


class DocumentKind(StrEnum):
    """What kind of source a document came from.

    Maps onto :class:`~sentinel.core.schemas.EvidenceKind` through
    :attr:`evidence_kind`, so a retrieval hit can be turned into typed
    :class:`~sentinel.core.schemas.Evidence` without the agent layer guessing.
    """

    TECHNIQUE = "technique"
    CVE = "cve"
    ADVISORY = "advisory"
    PLAYBOOK = "playbook"

    @property
    def evidence_kind(self) -> EvidenceKind:
        return _EVIDENCE_KIND[self]


_EVIDENCE_KIND: Final[dict[DocumentKind, EvidenceKind]] = {
    DocumentKind.TECHNIQUE: EvidenceKind.ATTACK_TECHNIQUE,
    DocumentKind.CVE: EvidenceKind.CVE_RECORD,
    # An advisory is not a CVE record and pretending otherwise would misreport
    # provenance in the audit log: several of these entries deliberately have no CVE
    # because the incident (a sabotaging maintainer, a trust-building campaign) is
    # not a software defect. They are knowledge-base chunks.
    DocumentKind.ADVISORY: EvidenceKind.KB_CHUNK,
    DocumentKind.PLAYBOOK: EvidenceKind.KB_CHUNK,
}


@dataclass(frozen=True, slots=True)
class KBDocument:
    """One knowledge-base document, before chunking.

    ``sections`` is an ordered mapping of section name to prose. Chunking walks it in
    order, so section boundaries are always chunk boundaries and a retrieved chunk
    can always say which section it came from.
    """

    doc_id: str
    kind: DocumentKind
    title: str
    sections: tuple[tuple[str, str], ...]
    aliases: tuple[str, ...] = ()
    tactics: tuple[str, ...] = ()
    platforms: tuple[str, ...] = ()
    products: tuple[str, ...] = ()
    techniques: tuple[str, ...] = ()
    cvss: float | None = None

    def __post_init__(self) -> None:
        if not self.doc_id or self.doc_id != self.doc_id.strip():
            raise CorpusError(f"doc_id {self.doc_id!r} must be non-empty and unpadded")
        if not self.title.strip():
            raise CorpusError(f"{self.doc_id}: title is empty")
        if not self.sections:
            raise CorpusError(f"{self.doc_id}: no sections, nothing to retrieve")
        for name, body in self.sections:
            if not name.strip():
                raise CorpusError(f"{self.doc_id}: section with an empty name")
            if not body.strip():
                raise CorpusError(f"{self.doc_id}: section {name!r} is empty")
        if self.cvss is not None and not 0.0 <= self.cvss <= 10.0:
            raise CorpusError(f"{self.doc_id}: cvss {self.cvss} outside 0-10")
        self._check_id_shape()
        for technique in self.techniques:
            if not TECHNIQUE_ID_PATTERN.match(technique):
                raise CorpusError(f"{self.doc_id}: cross-link {technique!r} is not an ATT&CK id")

    def _check_id_shape(self) -> None:
        """Identifier shape must match the declared kind.

        A CVE document whose id is not a CVE id would produce an
        ``EvidenceKind.CVE_RECORD`` citation pointing at something that is not a CVE
        record, which is exactly the class of quiet provenance error the audit log
        exists to make impossible.
        """
        match self.kind:
            case DocumentKind.TECHNIQUE:
                valid = bool(TECHNIQUE_ID_PATTERN.match(self.doc_id))
            case DocumentKind.CVE:
                valid = bool(CVE_ID_PATTERN.match(self.doc_id))
            case DocumentKind.ADVISORY:
                valid = self.doc_id.startswith(_ADVISORY_PREFIXES)
            case DocumentKind.PLAYBOOK:
                valid = self.doc_id.startswith(_PLAYBOOK_PREFIX)
        if not valid:
            raise CorpusError(f"{self.doc_id!r} is not a valid id for kind {self.kind.value!r}")

    @property
    def is_subtechnique(self) -> bool:
        return self.kind is DocumentKind.TECHNIQUE and "." in self.doc_id

    @property
    def parent_id(self) -> str | None:
        """The parent technique id for a sub-technique, else ``None``."""
        return self.doc_id.split(".")[0] if self.is_subtechnique else None

    @property
    def full_text(self) -> str:
        """Title, aliases and every section, for whole-document scoring and display."""
        parts = [self.title]
        if self.aliases:
            parts.append("Also known as: " + ", ".join(self.aliases))
        parts.extend(body for _, body in self.sections)
        return "\n".join(parts)

    def header(self) -> str:
        """The context prefix prepended to every chunk of this document.

        Without it, the second chunk of ``T1021.002`` contains detection guidance and
        no mention of SMB, admin shares or the technique id, and is unretrievable by
        any query naming the technique. Prepending a compact header to each chunk's
        *embedded* text is the standard fix and it is worth a large amount of recall
        on the second and later chunks of multi-chunk documents.
        """
        parts = [self.doc_id, self.title]
        if self.aliases:
            parts.append(", ".join(self.aliases))
        if self.tactics:
            parts.append(", ".join(self.tactics))
        if self.products:
            parts.append(", ".join(self.products))
        return " | ".join(parts)


@dataclass(frozen=True, slots=True)
class Corpus:
    """An immutable, indexable collection of :class:`KBDocument`."""

    documents: tuple[KBDocument, ...]
    _by_id: dict[str, KBDocument] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        by_id = self._by_id
        for document in self.documents:
            if document.doc_id in by_id:
                raise CorpusError(f"duplicate doc_id {document.doc_id!r}")
            by_id[document.doc_id] = document

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self) -> Iterator[KBDocument]:
        return iter(self.documents)

    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self._by_id

    def get(self, doc_id: str) -> KBDocument | None:
        return self._by_id.get(doc_id)

    def of_kind(self, kind: DocumentKind) -> tuple[KBDocument, ...]:
        return tuple(d for d in self.documents if d.kind is kind)

    def validate(self) -> None:
        """Assert the corpus-level invariants a single document cannot check.

        Specifically: every ATT&CK cross-link resolves, and every sub-technique's
        parent is present. A cross-link to a missing technique would let a retrieval
        hit cite a related technique the knowledge base cannot produce, which is the
        dangling-citation failure mode one layer down.
        """
        dangling = sorted(
            f"{d.doc_id} -> {t}"
            for d in self.documents
            for t in d.techniques
            if t not in self._by_id
        )
        if dangling:
            raise CorpusError(f"unresolved technique cross-links: {dangling}")
        orphans = sorted(
            d.doc_id
            for d in self.documents
            if d.parent_id is not None and d.parent_id not in self._by_id
        )
        if orphans:
            raise CorpusError(f"sub-techniques with no parent in corpus: {orphans}")

    # --- loading ------------------------------------------------------------- #

    @classmethod
    def from_dir(cls, directory: Path | str = DEFAULT_CORPUS_DIR) -> Corpus:
        """Load techniques, vulnerabilities and playbooks from a corpus directory."""
        path = Path(directory)
        documents: list[KBDocument] = []
        documents.extend(_load_techniques(path / "attack_techniques.jsonl"))
        documents.extend(_load_vulnerabilities(path / "vulnerabilities.jsonl"))
        documents.extend(_load_playbooks(path / "playbooks.jsonl"))
        corpus = cls(documents=tuple(documents))
        corpus.validate()
        return corpus


def _read_jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    if not path.exists():
        raise CorpusError(f"corpus file missing: {path}")
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"{path.name}:{number}: {exc}") from exc
            if not isinstance(record, dict):
                raise CorpusError(f"{path.name}:{number}: expected an object")
            yield number, record


def _require(record: dict, key: str, path: Path, number: int) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CorpusError(f"{path.name}:{number}: {key!r} must be a non-empty string")
    return value


def _strings(record: dict, key: str) -> tuple[str, ...]:
    value = record.get(key, ())
    if isinstance(value, str):
        raise CorpusError(f"{key!r} must be a list of strings, not a string")
    return tuple(str(item) for item in value)


def _load_techniques(path: Path) -> Iterator[KBDocument]:
    for number, record in _read_jsonl(path):
        sections: list[tuple[str, str]] = [
            ("description", _require(record, "description", path, number))
        ]
        detection = record.get("detection")
        if isinstance(detection, str) and detection.strip():
            sections.append(("detection", detection))
        yield KBDocument(
            doc_id=_require(record, "id", path, number),
            kind=DocumentKind.TECHNIQUE,
            title=_require(record, "name", path, number),
            sections=tuple(sections),
            aliases=_strings(record, "aliases"),
            tactics=_strings(record, "tactics"),
            platforms=_strings(record, "platforms"),
        )


def _load_vulnerabilities(path: Path) -> Iterator[KBDocument]:
    for number, record in _read_jsonl(path):
        kind_value = record.get("kind", "cve")
        try:
            kind = DocumentKind(kind_value)
        except ValueError as exc:
            raise CorpusError(f"{path.name}:{number}: unknown kind {kind_value!r}") from exc
        if kind not in (DocumentKind.CVE, DocumentKind.ADVISORY):
            raise CorpusError(f"{path.name}:{number}: kind {kind_value!r} not valid here")
        sections: list[tuple[str, str]] = [
            ("description", _require(record, "description", path, number))
        ]
        remediation = record.get("remediation")
        if isinstance(remediation, str) and remediation.strip():
            sections.append(("remediation", remediation))
        cvss = record.get("cvss")
        yield KBDocument(
            doc_id=_require(record, "id", path, number),
            kind=kind,
            title=_require(record, "title", path, number),
            sections=tuple(sections),
            aliases=_strings(record, "aliases"),
            products=_strings(record, "products"),
            techniques=_strings(record, "techniques"),
            cvss=float(cvss) if cvss is not None else None,
        )


def _load_playbooks(path: Path) -> Iterator[KBDocument]:
    for number, record in _read_jsonl(path):
        sections: list[tuple[str, str]] = [
            ("description", _require(record, "description", path, number)),
            ("steps", _require(record, "steps", path, number)),
        ]
        escalation = record.get("escalation")
        if isinstance(escalation, str) and escalation.strip():
            sections.append(("escalation", escalation))
        yield KBDocument(
            doc_id=_require(record, "id", path, number),
            kind=DocumentKind.PLAYBOOK,
            title=_require(record, "title", path, number),
            sections=tuple(sections),
            aliases=_strings(record, "aliases"),
            platforms=_strings(record, "applies_to"),
            techniques=_strings(record, "techniques"),
        )


_CACHED: dict[str, Corpus] = {}


def load_default_corpus(directory: Path | str = DEFAULT_CORPUS_DIR) -> Corpus:
    """Load (and memoize) the in-repo corpus.

    Memoized because parsing is pure and the result is immutable, and because the
    test suite builds a knowledge base in several modules; re-parsing 186 documents
    per test is wasted time with no isolation benefit.
    """
    key = str(Path(directory).resolve())
    if key not in _CACHED:
        _CACHED[key] = Corpus.from_dir(directory)
    return _CACHED[key]


def document_ids(documents: Sequence[KBDocument]) -> tuple[str, ...]:
    """Ids in corpus order. Small helper, but it keeps test assertions readable."""
    return tuple(d.doc_id for d in documents)
