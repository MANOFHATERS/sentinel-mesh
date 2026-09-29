"""Chunking: turn documents into retrievable, citable units.

A chunk is the unit of citation, so its identifier is the thing that ends up in
:attr:`sentinel.core.schemas.Evidence.ref` and therefore in the audit log. Three
properties follow from that and drive every decision here.

**Stable.** ``kb://technique/T1021.002#1`` must mean the same text next week, or an
audit trail citing it is worthless. Chunk ids are derived from ``(kind, doc_id,
section, ordinal)`` — never from a hash of the text, which would change on a typo
fix, and never from a global counter, which would change when an unrelated document
is inserted.

**Resolvable.** :meth:`sentinel.kb.retrieve.KnowledgeBase.resolve` must be able to
turn a ref back into the text it names.
:meth:`sentinel.core.schemas.InvestigationReport._every_claim_is_grounded` already
rejects a claim citing a ref that is not in the report's evidence list; resolvability
closes the other half, that the ref in the evidence list points at real content.

**Sentence-aligned.** Chunks never split a sentence. A citation whose excerpt begins
mid-clause reads as a fabrication to the analyst reviewing it even when the
retrieval was correct, and the whole point of F-05 is that the analyst can check.

Two texts per chunk, deliberately
---------------------------------
:attr:`Chunk.embed_text` carries the document header (id, title, aliases, tactics)
prepended to the body slice; :attr:`Chunk.excerpt` carries only the body slice.
The first is what gets indexed, the second is what a human reads. Mixing them would
either put boilerplate in every citation excerpt or make later chunks of a document
unretrievable by the document's own name. Both are real failures; keeping two fields
costs nothing.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Final

from sentinel.core.schemas import Evidence, EvidenceKind
from sentinel.kb.corpus import Corpus, KBDocument

__all__ = [
    "DEFAULT_OVERLAP_SENTENCES",
    "DEFAULT_TARGET_WORDS",
    "Chunk",
    "chunk_corpus",
    "chunk_document",
    "split_sentences",
]

#: Target body length per chunk, in whitespace-delimited words. Sized so a typical
#: technique description is one or two chunks: small enough that a retrieved chunk is
#: about one idea, large enough that an analyst reading the excerpt gets the claim and
#: its qualification rather than a fragment.
DEFAULT_TARGET_WORDS: Final[int] = 70

#: Sentences repeated from the end of the previous chunk. One sentence of overlap
#: keeps a claim and the sentence that qualifies it in the same chunk at least once,
#: which is what stops a boundary from silently truncating meaning.
DEFAULT_OVERLAP_SENTENCES: Final[int] = 1

#: Sentence boundary: terminator, then whitespace, then something that starts a
#: sentence. The negative lookbehind list is the set of abbreviations that actually
#: appear in this corpus and would otherwise split a sentence in half. Extending it
#: is a corpus-maintenance task, and ``test_kb_chunk.py`` asserts no chunk in the
#: shipped corpus begins with a lowercase letter, which is what catches a miss.
_SENTENCE_BOUNDARY: Final[re.Pattern[str]] = re.compile(
    r"""
    (?<!\be\.g)(?<!\bi\.e)(?<!\bvs)(?<!\bno)(?<!\bfig)(?<!\bapprox)
    (?<=[.!?])
    [ \t]+
    (?=[A-Z0-9"'(\[])
    """,
    re.VERBOSE,
)

#: Numbered-step boundary, for playbook ``steps`` sections written as "1. ... 2. ...".
#: Without it a whole procedure is one 200-word chunk and a query about step three
#: retrieves the entire playbook, which is correct but unhelpfully coarse.
_STEP_BOUNDARY: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?])\s+(?=\d{1,2}\.\s)")

#: A piece that is nothing but a step number. :func:`split_sentences` produces these
#: because ``1. Isolate the host.`` looks exactly like two sentences to a terminator
#: pattern: a period followed by whitespace followed by a capital. Left alone it
#: yields a chunk whose entire text is ``1.``, and it strips the step number off the
#: step it belongs to — so an excerpt cited to an analyst would say "Isolate the host"
#: with no indication of where in the procedure it sits. Merged back below.
_BARE_ENUMERATOR: Final[re.Pattern[str]] = re.compile(r"^\d{1,2}\.$")


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable unit of the knowledge base.

    ``chunk_id`` doubles as the citation ref and is stable across rebuilds.
    """

    chunk_id: str
    doc_id: str
    kind: str
    title: str
    section: str
    ordinal: int
    excerpt: str
    embed_text: str

    @property
    def word_count(self) -> int:
        return len(self.excerpt.split())

    def as_evidence(
        self,
        *,
        evidence_kind: EvidenceKind,
        relevance: float = 1.0,
        retrieved_at: object = None,
    ) -> Evidence:
        """Render as typed :class:`~sentinel.core.schemas.Evidence`.

        ``excerpt`` is passed through the schema's untrusted-text wrapper, which is
        not ceremony: a poisoned CVE description or a tampered playbook reaches the
        Investigation Agent's prompt through exactly this field. Retrieved evidence
        is attacker-influenceable content, and the one place it must not be treated
        as trusted is the place it is about to be rendered into a prompt.
        """
        return Evidence(
            kind=evidence_kind,
            ref=self.chunk_id,
            excerpt=self.excerpt,
            relevance=max(0.0, min(1.0, relevance)),
            retrieved_at=retrieved_at,  # type: ignore[arg-type]
        )


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, keeping numbered steps as their own units."""
    collapsed = " ".join(text.split())
    if not collapsed:
        return []
    pieces: list[str] = []
    for block in _STEP_BOUNDARY.split(collapsed):
        pieces.extend(part for part in _SENTENCE_BOUNDARY.split(block) if part)
    return _merge_enumerators(pieces)


def _merge_enumerators(pieces: list[str]) -> list[str]:
    """Re-attach a bare step number to the sentence it introduces.

    Fixing this in the boundary regex instead would need a lookbehind that refuses to
    split after a digit and a period, which also suppresses the legitimate break in
    "...rose to 48%. The next edition..." whenever a sentence happens to end in a
    numeral. Merging afterwards is narrower and says what it means.
    """
    merged: list[str] = []
    pending = ""
    for piece in pieces:
        if _BARE_ENUMERATOR.match(piece):
            pending = f"{pending} {piece}".strip() if pending else piece
            continue
        merged.append(f"{pending} {piece}" if pending else piece)
        pending = ""
    if pending:
        merged.append(pending)
    return merged


def _pack(
    sentences: Sequence[str], *, target_words: int, overlap: int
) -> Iterator[str]:
    """Greedily pack sentences into chunks of about ``target_words`` words.

    Greedy rather than optimal on purpose. An optimal packing would move existing
    chunk boundaries whenever a sentence is edited, changing the meaning of already
    issued citation refs across the whole document. Greedy packing confines the
    damage of an edit to the chunks after it.
    """
    if not sentences:
        return
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        words = len(sentence.split())
        # Emit before adding when the current chunk is already at target, so a single
        # over-long sentence becomes its own chunk rather than being split.
        if current and current_words + words > target_words:
            yield " ".join(current)
            tail = current[-overlap:] if overlap > 0 else []
            current = [*tail]
            current_words = sum(len(s.split()) for s in current)
        current.append(sentence)
        current_words += words
    if current:
        yield " ".join(current)


def chunk_document(
    document: KBDocument,
    *,
    target_words: int = DEFAULT_TARGET_WORDS,
    overlap: int = DEFAULT_OVERLAP_SENTENCES,
) -> tuple[Chunk, ...]:
    """Chunk one document, section by section.

    Section boundaries are always chunk boundaries: "what is this" and "what do I do"
    are different questions, and a chunk spanning a description and its remediation
    answers neither cleanly while diluting both in the index.
    """
    if target_words < 10:
        raise ValueError("target_words below 10 produces single-sentence chunks")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    header = document.header()
    chunks: list[Chunk] = []
    ordinal = 0
    for section, body in document.sections:
        sentences = split_sentences(body)
        for piece in _pack(sentences, target_words=target_words, overlap=overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"kb://{document.kind.value}/{document.doc_id}#{section}.{ordinal}",
                    doc_id=document.doc_id,
                    kind=document.kind.value,
                    title=document.title,
                    section=section,
                    ordinal=ordinal,
                    excerpt=piece,
                    embed_text=f"{header} | {section} | {piece}",
                )
            )
            ordinal += 1
    if not chunks:
        raise ValueError(f"{document.doc_id}: produced no chunks")
    return tuple(chunks)


def chunk_corpus(
    corpus: Corpus,
    *,
    target_words: int = DEFAULT_TARGET_WORDS,
    overlap: int = DEFAULT_OVERLAP_SENTENCES,
) -> tuple[Chunk, ...]:
    """Chunk an entire corpus in document order."""
    chunks: list[Chunk] = []
    for document in corpus:
        chunks.extend(
            chunk_document(document, target_words=target_words, overlap=overlap)
        )
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            raise ValueError(f"duplicate chunk id {chunk.chunk_id!r}")
        seen.add(chunk.chunk_id)
    return tuple(chunks)
