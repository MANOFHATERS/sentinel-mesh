"""The knowledge-base corpus (:mod:`sentinel.kb.corpus`).

Two kinds of test here, and the split matters:

*   **Invariant tests** — that a malformed document is *rejected*. These guard the
    validators, and they are what make the corpus loader a contract rather than a
    parser.
*   **Shipped-corpus tests** — that the 198 documents actually in the repository
    satisfy those invariants, that identifiers are well-formed, and that every
    cross-link resolves. This is the data equivalent of a type check: the corpus is
    hand-written, so it will drift, and a dangling cross-link becomes a citation
    the Investigation Agent cannot produce.

``DocumentKind.evidence_kind`` gets its own class because an advisory deliberately
maps to ``KB_CHUNK`` rather than ``CVE_RECORD`` — several advisories have no CVE at
all — and mislabelling provenance in the audit log is exactly the quiet error the
audit log exists to prevent.
"""

from __future__ import annotations

import pytest

from sentinel.core.schemas import CVE_ID_PATTERN, TECHNIQUE_ID_PATTERN, EvidenceKind
from sentinel.kb.corpus import (
    Corpus,
    CorpusError,
    DocumentKind,
    KBDocument,
    load_default_corpus,
)


def make_document(**overrides: object) -> KBDocument:
    kwargs: dict = {
        "doc_id": "T1021",
        "kind": DocumentKind.TECHNIQUE,
        "title": "Remote Services",
        "sections": (("description", "Adversaries log in to other hosts."),),
    }
    kwargs.update(overrides)
    return KBDocument(**kwargs)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return load_default_corpus()


class TestDocumentInvariants:
    def test_valid_document_constructs(self) -> None:
        assert make_document().doc_id == "T1021"

    @pytest.mark.parametrize(
        "overrides,fragment",
        [
            ({"doc_id": ""}, "non-empty"),
            ({"doc_id": " T1021 "}, "unpadded"),
            ({"title": "   "}, "title is empty"),
            ({"sections": ()}, "no sections"),
            ({"sections": (("", "body"),)}, "empty name"),
            ({"sections": (("description", "  "),)}, "is empty"),
            ({"cvss": 11.0}, "outside 0-10"),
            ({"cvss": -1.0}, "outside 0-10"),
            ({"techniques": ("T99",)}, "not an ATT&CK id"),
            ({"techniques": ("not-an-id",)}, "not an ATT&CK id"),
        ],
    )
    def test_malformed_document_rejected(self, overrides: dict, fragment: str) -> None:
        with pytest.raises(CorpusError, match=fragment):
            make_document(**overrides)

    @pytest.mark.parametrize(
        "kind,bad_id",
        [
            (DocumentKind.TECHNIQUE, "CVE-2021-44228"),
            (DocumentKind.CVE, "T1021"),
            (DocumentKind.ADVISORY, "T1021"),
            (DocumentKind.PLAYBOOK, "CVE-2021-44228"),
            (DocumentKind.CVE, "CVE-21-1"),
        ],
    )
    def test_id_shape_must_match_kind(self, kind: DocumentKind, bad_id: str) -> None:
        """A CVE-kinded document with a non-CVE id would emit a mislabelled citation."""
        with pytest.raises(CorpusError, match="not a valid id for kind"):
            make_document(doc_id=bad_id, kind=kind)

    @pytest.mark.parametrize(
        "kind,good_id",
        [
            (DocumentKind.TECHNIQUE, "T1021.002"),
            (DocumentKind.CVE, "CVE-2024-3094"),
            (DocumentKind.ADVISORY, "GHSA-EVENT-STREAM-2018"),
            (DocumentKind.PLAYBOOK, "PB-LATERAL-SMB"),
        ],
    )
    def test_id_shape_accepted(self, kind: DocumentKind, good_id: str) -> None:
        assert make_document(doc_id=good_id, kind=kind).doc_id == good_id

    def test_document_is_immutable(self) -> None:
        document = make_document()
        with pytest.raises(AttributeError):
            document.title = "changed"  # type: ignore[misc]


class TestHierarchy:
    def test_subtechnique_detected(self) -> None:
        assert make_document(doc_id="T1021.002").is_subtechnique
        assert make_document(doc_id="T1021.002").parent_id == "T1021"

    def test_parent_technique_has_no_parent(self) -> None:
        assert not make_document(doc_id="T1021").is_subtechnique
        assert make_document(doc_id="T1021").parent_id is None

    def test_non_technique_never_reports_a_parent(self) -> None:
        cve = make_document(doc_id="CVE-2021-44228", kind=DocumentKind.CVE)
        assert cve.parent_id is None


class TestHeaderAndFullText:
    def test_header_carries_the_identifiers_a_query_would_name(self) -> None:
        document = make_document(
            aliases=("lateral movement", "moved laterally"),
            tactics=("lateral-movement",),
        )
        header = document.header()
        assert "T1021" in header
        assert "Remote Services" in header
        assert "moved laterally" in header
        assert "lateral-movement" in header

    def test_full_text_includes_aliases_and_every_section(self) -> None:
        document = make_document(
            aliases=("alias-one",),
            sections=(("description", "first body"), ("detection", "second body")),
        )
        text = document.full_text
        assert "alias-one" in text
        assert "first body" in text
        assert "second body" in text


class TestEvidenceKindMapping:
    def test_every_kind_maps(self) -> None:
        for kind in DocumentKind:
            assert isinstance(kind.evidence_kind, EvidenceKind)

    def test_techniques_and_cves_map_to_their_own_kinds(self) -> None:
        assert DocumentKind.TECHNIQUE.evidence_kind is EvidenceKind.ATTACK_TECHNIQUE
        assert DocumentKind.CVE.evidence_kind is EvidenceKind.CVE_RECORD

    def test_advisory_is_not_labelled_a_cve_record(self) -> None:
        """Several advisories have no CVE; calling them CVE records misreports it."""
        assert DocumentKind.ADVISORY.evidence_kind is EvidenceKind.KB_CHUNK
        assert DocumentKind.PLAYBOOK.evidence_kind is EvidenceKind.KB_CHUNK


class TestCorpusContainer:
    def test_duplicate_ids_rejected(self) -> None:
        with pytest.raises(CorpusError, match="duplicate doc_id"):
            Corpus(documents=(make_document(), make_document()))

    def test_lookup_and_membership(self) -> None:
        built = Corpus(documents=(make_document(),))
        assert built.get("T1021") is not None
        assert built.get("nope") is None
        assert "T1021" in built
        assert len(built) == 1

    def test_validate_rejects_dangling_cross_link(self) -> None:
        built = Corpus(
            documents=(
                make_document(
                    doc_id="CVE-2021-44228",
                    kind=DocumentKind.CVE,
                    title="Log4Shell",
                    techniques=("T9999",),
                ),
            )
        )
        with pytest.raises(CorpusError, match="unresolved technique cross-links"):
            built.validate()

    def test_validate_rejects_orphan_subtechnique(self) -> None:
        built = Corpus(documents=(make_document(doc_id="T1021.002"),))
        with pytest.raises(CorpusError, match="no parent in corpus"):
            built.validate()

    def test_of_kind_filters(self) -> None:
        built = Corpus(
            documents=(
                make_document(),
                make_document(
                    doc_id="CVE-2021-44228", kind=DocumentKind.CVE, title="Log4Shell"
                ),
            )
        )
        assert len(built.of_kind(DocumentKind.TECHNIQUE)) == 1
        assert len(built.of_kind(DocumentKind.CVE)) == 1


class TestShippedCorpus:
    """The corpus in the repository. It is hand-written, so it will drift."""

    def test_loads_and_validates(self, corpus: Corpus) -> None:
        corpus.validate()
        assert len(corpus) > 150

    def test_every_kind_is_represented(self, corpus: Corpus) -> None:
        for kind in DocumentKind:
            assert corpus.of_kind(kind), f"no {kind.value} documents"

    def test_technique_ids_are_well_formed(self, corpus: Corpus) -> None:
        for document in corpus.of_kind(DocumentKind.TECHNIQUE):
            assert TECHNIQUE_ID_PATTERN.match(document.doc_id)

    def test_cve_ids_are_well_formed(self, corpus: Corpus) -> None:
        for document in corpus.of_kind(DocumentKind.CVE):
            assert CVE_ID_PATTERN.match(document.doc_id)

    def test_memoized_load_returns_the_same_object(self) -> None:
        assert load_default_corpus() is load_default_corpus()

    def test_every_technique_has_a_tactic(self, corpus: Corpus) -> None:
        for document in corpus.of_kind(DocumentKind.TECHNIQUE):
            assert document.tactics, f"{document.doc_id} has no tactic"

    def test_every_document_has_aliases(self, corpus: Corpus) -> None:
        """Aliases are the bridge from operator vocabulary to catalogue titles.

        A document with none is only reachable by its formal name, which is not how
        analysts type. Enforced because it is the cheapest recall win in the corpus
        and the easiest thing to forget when adding an entry.
        """
        missing = [d.doc_id for d in corpus if not d.aliases]
        assert not missing, f"documents with no aliases: {missing}"

    def test_aliases_are_lowercase_and_unpadded(self, corpus: Corpus) -> None:
        for document in corpus:
            for alias in document.aliases:
                assert alias == alias.strip().lower(), f"{document.doc_id}: {alias!r}"

    def test_aliases_are_unique_within_a_document(self, corpus: Corpus) -> None:
        for document in corpus:
            assert len(set(document.aliases)) == len(document.aliases), document.doc_id

    def test_cves_carry_a_severity(self, corpus: Corpus) -> None:
        for document in corpus.of_kind(DocumentKind.CVE):
            assert document.cvss is not None, f"{document.doc_id} has no cvss"
            assert 0.0 <= document.cvss <= 10.0

    def test_playbooks_have_steps(self, corpus: Corpus) -> None:
        for document in corpus.of_kind(DocumentKind.PLAYBOOK):
            sections = dict(document.sections)
            assert "steps" in sections, f"{document.doc_id} has no steps section"

    def test_playbooks_and_cves_cross_link_to_techniques(self, corpus: Corpus) -> None:
        """These links are what ``KnowledgeBase.related`` traverses; without them it
        returns nothing and the technique mapping F-05 wants is unreachable."""
        for kind in (DocumentKind.PLAYBOOK, DocumentKind.CVE, DocumentKind.ADVISORY):
            for document in corpus.of_kind(kind):
                assert document.techniques, f"{document.doc_id} links to no technique"

    def test_techniques_and_detection_sections_are_substantial(self, corpus: Corpus) -> None:
        """A one-line description retrieves poorly and cites uselessly."""
        for document in corpus:
            for name, body in document.sections:
                words = len(body.split())
                assert words >= 12, f"{document.doc_id}/{name} is {words} words"

    def test_no_document_text_contains_an_unresolvable_technique_reference(
        self, corpus: Corpus
    ) -> None:
        """Prose that names ``T####`` must name something the corpus has.

        Catches the case where an entry's text references a technique that was never
        added, which would make a cited excerpt point the analyst at a dead end.
        """
        import re

        known = {d.doc_id for d in corpus}
        pattern = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
        for document in corpus:
            for name, body in document.sections:
                for found in pattern.findall(body):
                    assert found in known, f"{document.doc_id}/{name} names {found}"
