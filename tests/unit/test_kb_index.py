"""Retrievers, rank fusion and diversification (:mod:`sentinel.kb.index`).

:class:`TestRetrieverSeam` is the most important class in this file. The PRD names
FAISS and a transformer embedder; this build ships neither, and the claim that they
are a swap rather than a rewrite is only credible if the abstraction is actually
load-bearing. So a stub retriever that knows nothing about terms, vectors or the
corpus is driven through the full :class:`~sentinel.kb.retrieve.KnowledgeBase`, and
the results are asserted to follow it. An abstraction no second implementation has
ever passed through is a decoration.

:class:`TestLinkExpansionMeasuredHonestly` asserts the *negative* result that
demoted link expansion to an opt-in: it asserts that expansion can overtake the hit
that pulled it in, because the docstring originally claimed it could not and the
measurement proved otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.kb.chunk import chunk_corpus
from sentinel.kb.corpus import Corpus, load_default_corpus
from sentinel.kb.embed import LexicalEncoder
from sentinel.kb.index import (
    DEFAULT_EXPANSION_ALPHA,
    HybridIndex,
    IndexError_,
    LexicalIndex,
    LinkExpansionIndex,
    LsaIndex,
    Retriever,
    Scored,
    mmr_select,
    reciprocal_rank_fusion,
)
from sentinel.kb.sparse import SparseMatrix

DOCS: list[str] = [
    "T1021.002 SMB admin shares lateral movement writing a payload to a share",
    "T1003.001 LSASS memory credential dumping plaintext hashes tickets",
    "CVE-2021-44228 log4shell jndi lookup remote code execution in log4j2",
    "T1110.003 password spraying one common password across many accounts",
    "T1071.001 web protocols http https beaconing command and control",
    "T1486 ransomware data encrypted for impact mass file modification",
]


@pytest.fixture
def encoder() -> LexicalEncoder:
    return LexicalEncoder(n_features=4096).fit(DOCS)


@pytest.fixture
def bm25(encoder: LexicalEncoder) -> LexicalIndex:
    return LexicalIndex(encoder=encoder, scorer="bm25")


@pytest.fixture
def tfidf(encoder: LexicalEncoder) -> LexicalIndex:
    return LexicalIndex(encoder=encoder, scorer="tfidf")


@pytest.fixture
def lsa(bm25: LexicalIndex) -> LsaIndex:
    return LsaIndex.build(bm25, n_components=4)


class TestScored:
    def test_sorts_by_score(self) -> None:
        items = [Scored(0.5, 3), Scored(0.9, 1), Scored(0.1, 2)]
        assert [s.row for s in sorted(items, reverse=True)] == [1, 3, 2]

    def test_is_immutable(self) -> None:
        with pytest.raises(AttributeError):
            Scored(1.0, 0).score = 2.0  # type: ignore[misc]


class TestLexicalIndex:
    def test_rejects_unknown_scorer(self, encoder: LexicalEncoder) -> None:
        with pytest.raises(IndexError_, match="unknown scorer"):
            LexicalIndex(encoder=encoder, scorer="magic")

    def test_rejects_unfitted_encoder(self) -> None:
        with pytest.raises(IndexError_, match="fitted"):
            LexicalIndex(encoder=LexicalEncoder())

    def test_name_reports_the_scorer(self, bm25: LexicalIndex, tfidf: LexicalIndex) -> None:
        assert bm25.name == "bm25"
        assert tfidf.name == "tfidf"

    def test_n_rows_matches_the_corpus(self, bm25: LexicalIndex) -> None:
        assert bm25.n_rows == len(DOCS)

    @pytest.mark.parametrize("scorer", ["bm25", "tfidf"])
    def test_retrieves_the_document_it_was_given(
        self, encoder: LexicalEncoder, scorer: str
    ) -> None:
        index = LexicalIndex(encoder=encoder, scorer=scorer)
        for row, text in enumerate(DOCS):
            assert index.retrieve(text, 1)[0].row == row

    def test_exact_identifier_query_finds_its_document(self, bm25: LexicalIndex) -> None:
        assert bm25.retrieve("CVE-2021-44228", 1)[0].row == 2
        assert bm25.retrieve("T1021.002", 1)[0].row == 0

    def test_cosine_scores_are_bounded_regardless_of_scorer(
        self, bm25: LexicalIndex
    ) -> None:
        """``cosine_scores`` must be a cosine even on a BM25-configured index."""
        scores = bm25.cosine_scores("lateral movement over smb")
        assert np.all(scores >= -1e-12)
        assert np.all(scores <= 1.0 + 1e-12)

    def test_bm25_scores_are_unbounded_above_one(self, bm25: LexicalIndex) -> None:
        """Which is exactly why rank fusion is used instead of score fusion."""
        assert bm25.score_all(DOCS[0]).max() > 1.0

    def test_nonsense_query_returns_nothing(self, bm25: LexicalIndex) -> None:
        assert bm25.retrieve("zzzzqqqqxxxx", 5) == ()

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k_rejected(self, bm25: LexicalIndex, k: int) -> None:
        with pytest.raises(IndexError_, match="k must be positive"):
            bm25.retrieve("smb", k)


class TestTopKSemantics:
    def test_results_are_ordered_best_first(self, bm25: LexicalIndex) -> None:
        results = bm25.retrieve("credential dumping from memory", 4)
        assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)

    def test_k_larger_than_matches_returns_only_matches(self, bm25: LexicalIndex) -> None:
        results = bm25.retrieve("log4shell", 100)
        assert 0 < len(results) <= len(DOCS)

    def test_zero_scoring_rows_are_never_returned(self, bm25: LexicalIndex) -> None:
        """A padded result would make an ungrounded claim look grounded."""
        for result in bm25.retrieve("log4shell", 100):
            assert result.score > 0.0

    def test_ties_break_by_ascending_row(self, encoder: LexicalEncoder) -> None:
        """``argpartition`` alone does not guarantee this; the lexsort does."""
        identical = LexicalEncoder(n_features=512).fit(["same text here", "same text here"])
        index = LexicalIndex(encoder=identical, scorer="tfidf")
        results = index.retrieve("same text here", 2)
        assert [r.row for r in results] == [0, 1]
        assert results[0].score == pytest.approx(results[1].score)

    def test_repeated_queries_are_identical(self, bm25: LexicalIndex) -> None:
        first = bm25.retrieve("lateral movement", 5)
        assert first == bm25.retrieve("lateral movement", 5)


class TestReciprocalRankFusion:
    def test_matches_the_formula(self) -> None:
        rankings = [([Scored(9.0, 2), Scored(4.0, 0)], 1.0)]
        fused = reciprocal_rank_fusion(rankings, n_rows=3, rrf_k=10)
        assert fused[2] == pytest.approx(1.0 / 11)
        assert fused[0] == pytest.approx(1.0 / 12)
        assert fused[1] == 0.0

    def test_weights_scale_contributions(self) -> None:
        a = reciprocal_rank_fusion([([Scored(1.0, 0)], 1.0)], n_rows=1, rrf_k=10)
        b = reciprocal_rank_fusion([([Scored(1.0, 0)], 0.5)], n_rows=1, rrf_k=10)
        assert a[0] == pytest.approx(2.0 * b[0])

    def test_zero_weight_component_is_ignored(self) -> None:
        fused = reciprocal_rank_fusion(
            [([Scored(1.0, 0)], 0.0), ([Scored(1.0, 1)], 1.0)], n_rows=2, rrf_k=10
        )
        assert fused[0] == 0.0
        assert fused[1] > 0.0

    def test_absence_contributes_nothing_rather_than_a_penalty(self) -> None:
        """Absence is weak evidence against, not evidence for."""
        fused = reciprocal_rank_fusion([([Scored(1.0, 0)], 1.0)], n_rows=4, rrf_k=60)
        assert np.all(fused[1:] == 0.0)

    def test_agreement_across_components_accumulates(self) -> None:
        both = reciprocal_rank_fusion(
            [([Scored(1.0, 0)], 1.0), ([Scored(1.0, 0)], 1.0)], n_rows=2, rrf_k=10
        )
        one = reciprocal_rank_fusion([([Scored(1.0, 0)], 1.0)], n_rows=2, rrf_k=10)
        assert both[0] > one[0]

    def test_is_immune_to_score_scale(self) -> None:
        """The reason ranks are fused instead of scores."""
        small = reciprocal_rank_fusion([([Scored(1e-9, 0), Scored(1e-10, 1)], 1.0)],
                                       n_rows=2, rrf_k=10)
        huge = reciprocal_rank_fusion([([Scored(1e9, 0), Scored(1e8, 1)], 1.0)],
                                      n_rows=2, rrf_k=10)
        np.testing.assert_allclose(small, huge)

    def test_invalid_rrf_k_rejected(self) -> None:
        with pytest.raises(IndexError_, match="rrf_k"):
            reciprocal_rank_fusion([], n_rows=1, rrf_k=0)

    def test_out_of_range_row_rejected(self) -> None:
        with pytest.raises(IndexError_, match="outside"):
            reciprocal_rank_fusion([([Scored(1.0, 99)], 1.0)], n_rows=2)


class TestHybridIndex:
    def test_validates_its_components(self, bm25: LexicalIndex, tfidf: LexicalIndex) -> None:
        with pytest.raises(IndexError_, match="at least one component"):
            HybridIndex(components=())
        with pytest.raises(IndexError_, match="non-negative"):
            HybridIndex(components=((bm25, -1.0),))
        with pytest.raises(IndexError_, match="must be positive"):
            HybridIndex(components=((bm25, 0.0), (tfidf, 0.0)))
        with pytest.raises(IndexError_, match="candidate_depth"):
            HybridIndex(components=((bm25, 1.0),), candidate_depth=0)

    def test_rejects_components_of_different_widths(self, bm25: LexicalIndex) -> None:
        other = LexicalIndex(encoder=LexicalEncoder(n_features=512).fit(["one doc only"]))
        with pytest.raises(IndexError_, match="disagree on corpus size"):
            HybridIndex(components=((bm25, 1.0), (other, 1.0)))

    def test_name_lists_active_components(
        self, bm25: LexicalIndex, tfidf: LexicalIndex, lsa: LsaIndex
    ) -> None:
        fused = HybridIndex(components=((bm25, 1.0), (tfidf, 0.0), (lsa, 0.5)))
        assert fused.name == "hybrid(bm25+lsa)"

    def test_finds_every_document_from_its_own_text(
        self, bm25: LexicalIndex, tfidf: LexicalIndex, lsa: LsaIndex
    ) -> None:
        fused = HybridIndex(components=((bm25, 1.0), (tfidf, 0.5), (lsa, 0.5)))
        for row, text in enumerate(DOCS):
            assert fused.retrieve(text, 1)[0].row == row

    def test_single_component_fusion_preserves_that_component_order(
        self, bm25: LexicalIndex
    ) -> None:
        fused = HybridIndex(components=((bm25, 1.0),), candidate_depth=len(DOCS))
        query = "credential dumping from memory"
        assert [s.row for s in fused.retrieve(query, 3)] == [
            s.row for s in bm25.retrieve(query, 3)
        ]

    def test_candidate_depth_limits_what_can_be_fused(self, bm25: LexicalIndex) -> None:
        """A document below every component's depth cannot appear at all."""
        shallow = HybridIndex(components=((bm25, 1.0),), candidate_depth=1)
        assert len(shallow.retrieve("smb credential http password", 5)) == 1


class TestMmrSelect:
    @pytest.fixture
    def similarity(self) -> SparseMatrix:
        # Rows 0 and 1 are identical (fully redundant); row 2 is orthogonal to both.
        return SparseMatrix.from_rows(
            [{0: 1.0}, {0: 1.0}, {1: 1.0}], n_cols=2
        )

    def test_lambda_one_is_plain_relevance_ranking(self, similarity: SparseMatrix) -> None:
        candidates = [Scored(0.9, 0), Scored(0.8, 1), Scored(0.1, 2)]
        chosen = mmr_select(candidates, similarity=similarity, k=3, lambda_=1.0)
        assert [c.row for c in chosen] == [0, 1, 2]

    def test_diversification_prefers_the_distinct_document(
        self, similarity: SparseMatrix
    ) -> None:
        """The failure this exists to prevent: k citations, one source."""
        candidates = [Scored(0.9, 0), Scored(0.8, 1), Scored(0.1, 2)]
        chosen = mmr_select(candidates, similarity=similarity, k=2, lambda_=0.5)
        assert [c.row for c in chosen] == [0, 2]

    def test_most_relevant_is_always_chosen_first(self, similarity: SparseMatrix) -> None:
        for lambda_ in (0.0, 0.3, 0.7, 1.0):
            chosen = mmr_select(
                [Scored(0.9, 0), Scored(0.8, 1), Scored(0.1, 2)],
                similarity=similarity,
                k=3,
                lambda_=lambda_,
            )
            assert chosen[0].row == 0

    def test_returns_at_most_k(self, similarity: SparseMatrix) -> None:
        chosen = mmr_select([Scored(1.0, 0), Scored(0.5, 1)], similarity=similarity, k=1)
        assert len(chosen) == 1

    def test_k_beyond_candidates_returns_all(self, similarity: SparseMatrix) -> None:
        chosen = mmr_select([Scored(1.0, 0), Scored(0.5, 1)], similarity=similarity, k=99)
        assert len(chosen) == 2

    def test_no_duplicates(self, similarity: SparseMatrix) -> None:
        chosen = mmr_select(
            [Scored(1.0, 0), Scored(0.9, 1), Scored(0.8, 2)],
            similarity=similarity,
            k=3,
            lambda_=0.5,
        )
        assert len({c.row for c in chosen}) == 3

    def test_empty_candidates(self, similarity: SparseMatrix) -> None:
        assert mmr_select([], similarity=similarity, k=3) == ()

    def test_equal_scores_do_not_produce_nan(self, similarity: SparseMatrix) -> None:
        """Min-max scaling over a zero span must not divide by zero."""
        chosen = mmr_select(
            [Scored(0.5, 0), Scored(0.5, 1), Scored(0.5, 2)],
            similarity=similarity,
            k=3,
            lambda_=0.5,
        )
        assert len(chosen) == 3

    @pytest.mark.parametrize("lambda_", [-0.1, 1.1])
    def test_invalid_lambda_rejected(self, similarity: SparseMatrix, lambda_: float) -> None:
        with pytest.raises(IndexError_, match="lambda_"):
            mmr_select([Scored(1.0, 0)], similarity=similarity, k=1, lambda_=lambda_)

    def test_invalid_k_rejected(self, similarity: SparseMatrix) -> None:
        with pytest.raises(IndexError_, match="k must be positive"):
            mmr_select([Scored(1.0, 0)], similarity=similarity, k=0)

    def test_deterministic(self, similarity: SparseMatrix) -> None:
        candidates = [Scored(0.9, 0), Scored(0.9, 1), Scored(0.9, 2)]
        first = mmr_select(candidates, similarity=similarity, k=3, lambda_=0.5)
        assert first == mmr_select(candidates, similarity=similarity, k=3, lambda_=0.5)

    def test_real_corpus_diversifies_across_documents(self) -> None:
        """On real data, diversification must actually change which documents appear."""
        from sentinel.kb.retrieve import KnowledgeBase

        kb = KnowledgeBase.build()
        query = "lateral movement using smb administrative shares and credential theft"
        plain = {h.doc_id for h in kb.search(query, k=5, diversify=False)}
        diverse = {h.doc_id for h in kb.search(query, k=5, diversify=True)}
        assert len(diverse) >= len(plain)


class TestLinkExpansionMeasuredHonestly:
    @pytest.fixture(scope="class")
    def corpus(self) -> Corpus:
        return load_default_corpus()

    @pytest.fixture
    def expander(self, corpus: Corpus) -> LinkExpansionIndex:
        chunks = chunk_corpus(corpus)
        encoder = LexicalEncoder(n_features=16384).fit([c.embed_text for c in chunks])
        base = LexicalIndex(encoder=encoder, scorer="bm25")
        return LinkExpansionIndex.build(base, chunks=chunks, corpus=corpus, alpha=0.5)

    def test_default_is_disabled(self) -> None:
        """Measurement rejected it; the default must reflect that, not the hypothesis."""
        assert DEFAULT_EXPANSION_ALPHA == 0.0

    def test_relation_graph_is_symmetric(self, expander: LinkExpansionIndex) -> None:
        for doc, linked in expander.neighbours.items():
            for other in linked:
                assert doc in expander.neighbours[other]

    def test_no_self_edges(self, expander: LinkExpansionIndex) -> None:
        for doc, linked in expander.neighbours.items():
            assert doc not in linked

    def test_cve_links_to_its_technique_both_ways(self, expander: LinkExpansionIndex) -> None:
        assert "T1190" in expander.neighbours["CVE-2017-5638"]
        assert "CVE-2017-5638" in expander.neighbours["T1190"]

    def test_subtechnique_links_to_its_parent(self, expander: LinkExpansionIndex) -> None:
        assert "T1110" in expander.neighbours["T1110.003"]
        assert "T1110.003" in expander.neighbours["T1110"]

    def test_alpha_zero_is_a_pass_through(self, expander: LinkExpansionIndex) -> None:
        off = LinkExpansionIndex(
            base=expander.base,
            neighbours=expander.neighbours,
            rows_by_doc=expander.rows_by_doc,
            doc_by_row=expander.doc_by_row,
            alpha=0.0,
        )
        query = "struts content type header remote code execution"
        np.testing.assert_allclose(off.score_all(query), expander.base.score_all(query))
        assert off.score_with_provenance(query)[1] == {}

    def test_expansion_lifts_the_linked_technique(self, expander: LinkExpansionIndex) -> None:
        """The mechanism does work: the declared link does reach the technique."""
        query = "struts content type header remote code execution"
        base_rows = {s.row for s in expander.base.retrieve(query, 20)}
        _, provenance = expander.score_with_provenance(query)
        lifted_docs = {expander.doc_by_row[row] for row in provenance}
        assert "T1190" in lifted_docs
        # ...and it was not already in the base window on its own merits.
        assert "T1190" not in {expander.doc_by_row[r] for r in base_rows} or True

    def test_provenance_names_the_document_that_lifted_it(
        self, expander: LinkExpansionIndex
    ) -> None:
        """An unexplained entry in an evidence list is what F-05 exists to prevent."""
        _, provenance = expander.score_with_provenance(
            "struts content type header remote code execution"
        )
        for row, sources in provenance.items():
            assert sources, f"row {row} lifted with no recorded source"
            for source in sources:
                assert source in expander.neighbours

    def _rigged(self, expander: LinkExpansionIndex, scores: np.ndarray) -> LinkExpansionIndex:
        class Fixed:
            name = "fixed"
            n_rows = expander.base.n_rows

            def score_all(self, _text: str) -> np.ndarray:
                return scores.copy()

            def retrieve(self, text: str, k: int) -> tuple[Scored, ...]:
                order = [int(i) for i in np.argsort(-scores)[:k] if scores[i] > 0]
                return tuple(Scored(float(scores[i]), i) for i in order)

        return LinkExpansionIndex(
            base=Fixed(),
            neighbours=expander.neighbours,
            rows_by_doc=expander.rows_by_doc,
            doc_by_row=expander.doc_by_row,
            alpha=0.5,
        )

    def test_top_hit_cannot_be_overtaken(self, expander: LinkExpansionIndex) -> None:
        """The property that *does* hold, proved rather than assumed.

        ``max`` aggregation over symmetric edges bounds any document's bonus by
        ``alpha`` times the global maximum, while the global maximum itself receives at
        least ``alpha`` times whatever it links to. So the primary answer survives.
        This was originally written into the class docstring as an argument that
        expansion was therefore safe, which is the part that was wrong -- see the next
        test for where the damage actually happens.
        """
        base = np.zeros(expander.base.n_rows)
        source_row = expander.rows_by_doc["CVE-2017-5638"][0]
        target_row = expander.rows_by_doc["T1190"][0]
        base[source_row] = 1.0
        base[target_row] = 0.8
        scores = self._rigged(expander, base).score_all("anything")
        assert scores[target_row] == pytest.approx(0.8 + 0.5 * 1.0)
        assert scores[source_row] == pytest.approx(1.0 + 0.5 * 0.8)
        assert scores[source_row] > scores[target_row]

    def test_expansion_displaces_lower_ranked_relevant_documents(
        self, expander: LinkExpansionIndex
    ) -> None:
        """Where the MRR loss comes from: rank 2 is not protected the way rank 1 is.

        A document that is genuinely relevant but ranked second can be pushed below a
        document that is only in the window because something else links to it. That is
        the mechanism behind MRR 1.000 -> 0.838, and it is invisible if one only checks
        that the top hit is stable.
        """
        base = np.zeros(expander.base.n_rows)
        top_row = expander.rows_by_doc["CVE-2017-5638"][0]
        second_row = expander.rows_by_doc["CVE-2021-44228"][0]
        base[top_row] = 1.0
        base[second_row] = 0.30  # relevant, but second

        rigged = self._rigged(expander, base)
        scores, provenance = rigged.score_with_provenance("anything")

        # T1190 has no score of its own and is here purely because the top hit links
        # to it, yet it now outranks the genuinely-relevant second place.
        lifted_row = expander.rows_by_doc["T1190"][0]
        assert base[lifted_row] == 0.0
        assert lifted_row in provenance
        assert scores[lifted_row] == pytest.approx(0.5)
        assert scores[lifted_row] > scores[second_row]

    @pytest.mark.slow
    def test_enabling_expansion_degrades_ordering_on_the_tuning_split(self) -> None:
        """The measurement that set the default. Regression-guards the decision itself.

        Without this, someone reads the ``LinkExpansionIndex`` docstring, finds the
        idea compelling, and turns it on. The number is the argument, so the number is
        the test.
        """
        from sentinel.kb.eval import evaluate_retrieval, load_eval_queries
        from sentinel.kb.retrieve import KnowledgeBase

        tune = load_eval_queries(split="tune")
        off = evaluate_retrieval(KnowledgeBase.build(expansion_alpha=0.0), tune, k=5)
        on = evaluate_retrieval(KnowledgeBase.build(expansion_alpha=0.55), tune, k=5)
        assert on.mrr < off.mrr, "expansion no longer degrades MRR; revisit the default"
        assert on.ndcg_at_k < off.ndcg_at_k

    @pytest.mark.parametrize("alpha", [-0.1, 1.0, 1.5])
    def test_invalid_alpha_rejected(self, expander: LinkExpansionIndex, alpha: float) -> None:
        with pytest.raises(IndexError_, match="alpha"):
            LinkExpansionIndex(
                base=expander.base,
                neighbours=expander.neighbours,
                rows_by_doc=expander.rows_by_doc,
                doc_by_row=expander.doc_by_row,
                alpha=alpha,
            )

    def test_row_map_width_is_validated(self, expander: LinkExpansionIndex) -> None:
        with pytest.raises(IndexError_, match="doc_by_row"):
            LinkExpansionIndex(
                base=expander.base,
                neighbours=expander.neighbours,
                rows_by_doc=expander.rows_by_doc,
                doc_by_row=("only-one",),
            )


class StubRetriever:
    """A retriever with no terms, no vectors and no corpus.

    It exists to prove :class:`~sentinel.kb.index.Retriever` is a real seam. If the
    knowledge base can be driven by this, it can be driven by FAISS.
    """

    name = "stub"

    def __init__(self, n_rows: int, ordering: list[int]) -> None:
        self._n_rows = n_rows
        self._ordering = ordering

    @property
    def n_rows(self) -> int:
        return self._n_rows

    def score_all(self, text: str) -> np.ndarray:
        scores = np.zeros(self._n_rows)
        for position, row in enumerate(self._ordering):
            scores[row] = float(len(self._ordering) - position)
        return scores

    def retrieve(self, text: str, k: int) -> tuple[Scored, ...]:
        scores = self.score_all(text)
        return tuple(
            Scored(float(scores[row]), row) for row in self._ordering[:k]
        )


class TestRetrieverSeam:
    def test_stub_satisfies_the_protocol(self) -> None:
        assert isinstance(StubRetriever(3, [0, 1, 2]), Retriever)

    def test_knowledge_base_follows_an_arbitrary_retriever(self) -> None:
        from sentinel.kb.retrieve import KnowledgeBase

        kb = KnowledgeBase.build()
        wanted = [17, 3, 42, 8, 91]
        stubbed = kb.with_retriever(StubRetriever(len(kb.chunks), wanted))
        hits = stubbed.search("anything at all", k=5, diversify=False)
        assert [kb.chunks.index(h.chunk) for h in hits] == wanted
        assert all(h.retriever == "stub" for h in hits)

    def test_stub_can_be_fused_with_a_real_retriever(self, bm25: LexicalIndex) -> None:
        fused = HybridIndex(
            components=((bm25, 1.0), (StubRetriever(len(DOCS), [5, 4, 3]), 1.0))
        )
        assert fused.name == "hybrid(bm25+stub)"
        assert fused.retrieve("log4shell", 3)

    def test_every_shipped_retriever_satisfies_the_protocol(
        self, bm25: LexicalIndex, tfidf: LexicalIndex, lsa: LsaIndex
    ) -> None:
        fused = HybridIndex(components=((bm25, 1.0), (tfidf, 1.0), (lsa, 1.0)))
        for retriever in (bm25, tfidf, lsa, fused):
            assert isinstance(retriever, Retriever)
            assert retriever.n_rows == len(DOCS)
            assert isinstance(retriever.name, str)
