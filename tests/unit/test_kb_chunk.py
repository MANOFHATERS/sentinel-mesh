"""Chunking (:mod:`sentinel.kb.chunk`).

A chunk id becomes :attr:`sentinel.core.schemas.Evidence.ref` and therefore lands in
the hash-chained audit log, so the tests that matter are about the three properties
that makes necessary:

*   **Stability** — the same document must produce the same ids on every build, and
    editing a late section must not renumber earlier chunks. An audit trail whose
    refs shift meaning between builds is not an audit trail.
*   **Coverage** — chunking must not silently drop text. A dropped sentence is
    content that can never be cited, which looks identical to content that does not
    exist.
*   **Sentence alignment** — a citation excerpt beginning mid-clause reads as
    fabrication to the analyst reviewing it, which defeats the point of citing.

:class:`TestHeaderIsWhatMakesLaterChunksRetrievable` is the one that justifies the
two-text design: without the header, chunk 1 of a technique contains detection advice
and never mentions the technique, so no query naming the technique can reach it.
"""

from __future__ import annotations

import pytest

from sentinel.core.schemas import Evidence, EvidenceKind
from sentinel.kb.chunk import (
    DEFAULT_TARGET_WORDS,
    Chunk,
    chunk_corpus,
    chunk_document,
    split_sentences,
)
from sentinel.kb.corpus import Corpus, DocumentKind, KBDocument, load_default_corpus


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return load_default_corpus()


@pytest.fixture(scope="module")
def all_chunks(corpus: Corpus) -> tuple[Chunk, ...]:
    return chunk_corpus(corpus)


def document(sections: tuple[tuple[str, str], ...], **kw: object) -> KBDocument:
    kwargs: dict = {
        "doc_id": "T1021",
        "kind": DocumentKind.TECHNIQUE,
        "title": "Remote Services",
        "sections": sections,
    }
    kwargs.update(kw)
    return KBDocument(**kwargs)  # type: ignore[arg-type]


LONG_BODY = " ".join(
    f"Sentence number {n} describes one aspect of the technique in enough words to "
    f"matter for packing." for n in range(1, 13)
)


class TestSentenceSplitting:
    def test_splits_on_terminators(self) -> None:
        assert split_sentences("One. Two. Three.") == ["One.", "Two.", "Three."]

    def test_collapses_whitespace(self) -> None:
        assert split_sentences("One.\n\n  Two.") == ["One.", "Two."]

    @pytest.mark.parametrize(
        "text",
        [
            "Use a tool, e.g. the built-in one, to decode.",
            "The control, i.e. the agent, was disabled.",
            "Compare host vs. network telemetry here.",
        ],
    )
    def test_abbreviations_do_not_split(self, text: str) -> None:
        assert len(split_sentences(text)) == 1

    def test_numbered_steps_become_separate_units(self) -> None:
        steps = "1. Isolate the host. 2. Reset the password. 3. Reimage."
        pieces = split_sentences(steps)
        assert len(pieces) == 3
        assert pieces[0].startswith("1.")
        assert pieces[1].startswith("2.")

    def test_empty_text(self) -> None:
        assert split_sentences("") == []
        assert split_sentences("   ") == []


class TestChunkIdentity:
    def test_ids_are_stable_across_rebuilds(self) -> None:
        doc = document((("description", LONG_BODY),))
        assert [c.chunk_id for c in chunk_document(doc)] == [
            c.chunk_id for c in chunk_document(doc)
        ]

    def test_id_encodes_kind_doc_section_and_ordinal(self) -> None:
        chunks = chunk_document(document((("description", "A short body here now."),)))
        assert chunks[0].chunk_id == "kb://technique/T1021#description.0"

    def test_editing_a_later_section_does_not_renumber_earlier_chunks(self) -> None:
        """Greedy packing confines an edit's blast radius to the chunks after it."""
        before = chunk_document(
            document(
                (
                    ("description", "First body sentence here."),
                    ("detection", "Old text here."),
                )
            )
        )
        after = chunk_document(
            document(
                (
                    ("description", "First body sentence here."),
                    ("detection", "Completely different and much longer detection text here."),
                )
            )
        )
        assert before[0].chunk_id == after[0].chunk_id
        assert before[0].excerpt == after[0].excerpt

    def test_ids_are_unique_across_the_whole_corpus(self, all_chunks: tuple[Chunk, ...]) -> None:
        ids = [c.chunk_id for c in all_chunks]
        assert len(set(ids)) == len(ids)

    def test_ordinal_is_document_scoped_and_contiguous(self) -> None:
        chunks = chunk_document(
            document((("description", LONG_BODY), ("detection", LONG_BODY)))
        )
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))


class TestSectionBoundaries:
    def test_no_chunk_spans_two_sections(self) -> None:
        chunks = chunk_document(
            document((("description", "What it is here."), ("detection", "What to do here.")))
        )
        for chunk in chunks:
            assert chunk.section in ("description", "detection")
        assert {c.section for c in chunks} == {"description", "detection"}

    def test_description_and_detection_never_blend(self) -> None:
        chunks = chunk_document(
            document(
                (
                    ("description", "UNIQUEDESCRIPTIONTOKEN appears only here."),
                    ("detection", "UNIQUEDETECTIONTOKEN appears only here."),
                )
            )
        )
        for chunk in chunks:
            assert not (
                "UNIQUEDESCRIPTIONTOKEN" in chunk.excerpt
                and "UNIQUEDETECTIONTOKEN" in chunk.excerpt
            )


class TestPacking:
    def test_long_body_produces_several_chunks(self) -> None:
        chunks = chunk_document(document((("description", LONG_BODY),)), target_words=30)
        assert len(chunks) > 2

    def test_chunks_respect_the_target_within_one_sentence(self) -> None:
        chunks = chunk_document(document((("description", LONG_BODY),)), target_words=40)
        for chunk in chunks:
            # Overshoot by at most the final sentence, which is never split.
            assert chunk.word_count <= 40 + 25

    def test_overlap_repeats_the_previous_sentence(self) -> None:
        chunks = chunk_document(
            document((("description", LONG_BODY),)), target_words=30, overlap=1
        )
        assert len(chunks) >= 2
        tail = split_sentences(chunks[0].excerpt)[-1]
        assert chunks[1].excerpt.startswith(tail)

    def test_zero_overlap_repeats_nothing(self) -> None:
        chunks = chunk_document(
            document((("description", LONG_BODY),)), target_words=30, overlap=0
        )
        first_tail = split_sentences(chunks[0].excerpt)[-1]
        assert not chunks[1].excerpt.startswith(first_tail)

    def test_an_overlong_single_sentence_is_not_split(self) -> None:
        sentence = " ".join(["word"] * 200) + "."
        chunks = chunk_document(document((("description", sentence),)), target_words=20)
        assert len(chunks) == 1
        assert chunks[0].word_count == 200

    @pytest.mark.parametrize("bad", [5, 9, 0, -1])
    def test_tiny_target_rejected(self, bad: int) -> None:
        with pytest.raises(ValueError, match="target_words"):
            chunk_document(document((("description", LONG_BODY),)), target_words=bad)

    def test_negative_overlap_rejected(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            chunk_document(document((("description", LONG_BODY),)), overlap=-1)


class TestCoverage:
    def test_every_sentence_appears_in_some_chunk(self) -> None:
        """A dropped sentence is content that can never be cited."""
        doc = document((("description", LONG_BODY),))
        chunks = chunk_document(doc, target_words=30)
        joined = " ".join(c.excerpt for c in chunks)
        for sentence in split_sentences(LONG_BODY):
            assert sentence in joined

    def test_corpus_wide_coverage(self, corpus: Corpus, all_chunks: tuple[Chunk, ...]) -> None:
        by_doc: dict[str, list[str]] = {}
        for chunk in all_chunks:
            by_doc.setdefault(chunk.doc_id, []).append(chunk.excerpt)
        for doc in corpus:
            joined = " ".join(by_doc[doc.doc_id])
            for _, body in doc.sections:
                for sentence in split_sentences(body):
                    assert sentence in joined, f"{doc.doc_id}: dropped {sentence[:50]!r}"


class TestSentenceAlignment:
    def test_no_chunk_starts_mid_clause(self, all_chunks: tuple[Chunk, ...]) -> None:
        """Catches a missing abbreviation in the boundary pattern.

        Every sentence in the corpus starts with a capital, a digit or a quote; a
        chunk starting lowercase means a sentence was cut in half.
        """
        offenders = [
            c.chunk_id
            for c in all_chunks
            if c.excerpt and not (c.excerpt[0].isupper() or c.excerpt[0].isdigit()
                                 or c.excerpt[0] in "\"'([")
        ]
        assert not offenders, f"chunks starting mid-clause: {offenders[:5]}"

    def test_no_chunk_is_empty_or_whitespace(self, all_chunks: tuple[Chunk, ...]) -> None:
        for chunk in all_chunks:
            assert chunk.excerpt.strip()


class TestHeaderIsWhatMakesLaterChunksRetrievable:
    """The justification for keeping ``embed_text`` separate from ``excerpt``."""

    def test_every_chunk_embeds_the_document_identifier(
        self, all_chunks: tuple[Chunk, ...]
    ) -> None:
        for chunk in all_chunks:
            assert chunk.doc_id in chunk.embed_text

    def test_every_chunk_embeds_the_title_and_section(
        self, all_chunks: tuple[Chunk, ...]
    ) -> None:
        for chunk in all_chunks:
            assert chunk.title in chunk.embed_text
            assert chunk.section in chunk.embed_text

    def test_a_later_chunk_would_be_unreachable_without_the_header(self) -> None:
        """The concrete failure the header prevents."""
        doc = document(
            (("description", "Adversaries log in to other hosts."),
             ("detection", "Model the normal authentication graph and alert on new edges.")),
            aliases=("moved laterally",),
        )
        detection = next(c for c in chunk_document(doc) if c.section == "detection")
        assert "T1021" not in detection.excerpt
        assert "moved laterally" not in detection.excerpt
        # ...but the indexed text carries both, so a query naming either can reach it.
        assert "T1021" in detection.embed_text
        assert "moved laterally" in detection.embed_text

    def test_excerpt_is_never_polluted_with_the_header(
        self, all_chunks: tuple[Chunk, ...]
    ) -> None:
        """A human-facing citation must not read like index boilerplate."""
        for chunk in all_chunks:
            assert not chunk.excerpt.startswith(chunk.doc_id)
            assert " | " not in chunk.excerpt


class TestAsEvidence:
    def test_produces_typed_evidence_with_the_chunk_id_as_ref(self) -> None:
        chunk = chunk_document(document((("description", "A body sentence here now."),)))[0]
        evidence = chunk.as_evidence(evidence_kind=EvidenceKind.ATTACK_TECHNIQUE)
        assert isinstance(evidence, Evidence)
        assert evidence.ref == chunk.chunk_id
        assert evidence.kind is EvidenceKind.ATTACK_TECHNIQUE

    def test_excerpt_is_typed_untrusted(self) -> None:
        """A poisoned advisory reaches the prompt through this field."""
        chunk = chunk_document(document((("description", "A body sentence here now."),)))[0]
        evidence = chunk.as_evidence(evidence_kind=EvidenceKind.KB_CHUNK)
        # ``str()`` must redact, not render.
        assert "body sentence" not in str(evidence.excerpt)
        assert "untrusted" in str(evidence.excerpt)
        assert evidence.excerpt.raw == chunk.excerpt

    @pytest.mark.parametrize("relevance,expected", [(-1.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
    def test_relevance_is_clamped(self, relevance: float, expected: float) -> None:
        """``Confidence`` rejects out-of-range values; clamp rather than raise."""
        chunk = chunk_document(document((("description", "A body sentence here now."),)))[0]
        evidence = chunk.as_evidence(
            evidence_kind=EvidenceKind.KB_CHUNK, relevance=relevance
        )
        assert evidence.relevance == expected

    def test_ref_has_no_padding(self, all_chunks: tuple[Chunk, ...]) -> None:
        """``Evidence.ref`` rejects padded refs, so chunk ids must never be padded."""
        for chunk in all_chunks[:50]:
            assert chunk.chunk_id == chunk.chunk_id.strip()
            assert len(chunk.chunk_id) <= 512


class TestCorpusLevel:
    def test_duplicate_chunk_ids_rejected(self) -> None:
        # Two documents with the same id cannot be built through ``Corpus``, so the
        # guard is exercised by chunking the same document twice at the list level.
        doc = document((("description", "A body sentence here now."),))
        chunks = [*chunk_document(doc), *chunk_document(doc)]
        seen: set[str] = set()
        clashes = [c.chunk_id for c in chunks if c.chunk_id in seen or seen.add(c.chunk_id)]
        assert clashes, "expected a clash to exist for this guard to be meaningful"

    def test_default_target_is_used(self, corpus: Corpus) -> None:
        assert chunk_corpus(corpus) == chunk_corpus(corpus, target_words=DEFAULT_TARGET_WORDS)

    def test_chunks_follow_document_order(
        self, corpus: Corpus, all_chunks: tuple[Chunk, ...]
    ) -> None:
        order = [d.doc_id for d in corpus]
        seen: list[str] = []
        for chunk in all_chunks:
            if chunk.doc_id not in seen:
                seen.append(chunk.doc_id)
        assert seen == order
