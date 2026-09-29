"""Retrieval metrics and acceptance gates (:mod:`sentinel.kb.eval`).

The metrics are asserted against hand-computed values on tiny fixtures, because a
metric that is subtly wrong makes every number downstream of it meaningless while
looking entirely plausible. An nDCG with the wrong discount, or a recall that counts
chunks instead of documents, produces numbers in the right range and the wrong order.

:class:`TestGates` asserts the gate refuses to be evaluated on the tuning split. That
is the one guard that stops the tune/test discipline being quietly abandoned by
somebody passing the convenient half of the query set.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.kb.eval import (
    EVAL_QUERIES_PATH,
    GATE_MRR,
    GATE_RECALL_AT_5,
    GATE_RECALL_WITH_LINKS,
    EvalError,
    EvalQuery,
    RetrievalReport,
    assert_gates,
    evaluate_retrieval,
    load_eval_queries,
)
from sentinel.kb.index import Scored
from sentinel.kb.retrieve import KnowledgeBase


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    return KnowledgeBase.build()


class FixedRetriever:
    """Returns a chosen ordering of rows, so metrics can be computed by hand."""

    name = "fixed"

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
        return tuple(Scored(float(scores[row]), row) for row in self._ordering[:k])


class TestEvalQuery:
    def test_valid_query(self) -> None:
        query = EvalQuery(query="smb lateral movement", gold=("T1021.002",), split="test")
        assert query.gold == ("T1021.002",)

    @pytest.mark.parametrize(
        "kwargs,fragment",
        [
            ({"query": "  ", "gold": ("T1021",), "split": "test"}, "empty query"),
            ({"query": "q", "gold": (), "split": "test"}, "no gold"),
            ({"query": "q", "gold": ("T1021",), "split": "holdout"}, "tune"),
            ({"query": "q", "gold": ("T1021", "T1021"), "split": "test"}, "duplicate"),
        ],
    )
    def test_malformed_rejected(self, kwargs: dict, fragment: str) -> None:
        with pytest.raises(EvalError, match=fragment):
            EvalQuery(**kwargs)


class TestLoadEvalQueries:
    def test_loads_the_shipped_set(self) -> None:
        queries = load_eval_queries()
        assert len(queries) > 80

    def test_split_filter(self) -> None:
        tune = load_eval_queries(split="tune")
        test = load_eval_queries(split="test")
        assert all(q.split == "tune" for q in tune)
        assert all(q.split == "test" for q in test)
        assert len(tune) + len(test) == len(load_eval_queries())

    def test_both_splits_are_substantial(self) -> None:
        """A tuning split too small to tune on, or a test split too small to trust."""
        assert len(load_eval_queries(split="tune")) >= 25
        assert len(load_eval_queries(split="test")) >= 50

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(EvalError, match="missing"):
            load_eval_queries(tmp_path / "absent.jsonl")

    def test_malformed_line_raises(self, tmp_path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text("{not json}\n", encoding="utf-8")
        with pytest.raises(EvalError):
            load_eval_queries(path)

    def test_duplicate_query_text_raises(self, tmp_path) -> None:
        path = tmp_path / "dupe.jsonl"
        path.write_text(
            '{"query":"a","gold":["T1021"],"split":"test"}\n'
            '{"query":"a","gold":["T1110"],"split":"test"}\n',
            encoding="utf-8",
        )
        with pytest.raises(EvalError, match="duplicate query"):
            load_eval_queries(path)

    def test_every_gold_id_exists_in_the_corpus(self, kb: KnowledgeBase) -> None:
        indexed = {chunk.doc_id for chunk in kb.chunks}
        for query in load_eval_queries():
            for gold in query.gold:
                assert gold in indexed, f"{query.query!r} -> {gold}"

    def test_queries_are_not_corpus_text(self) -> None:
        """A labelled query that quotes the corpus measures string matching."""
        from sentinel.kb.corpus import load_default_corpus

        bodies = " ".join(
            body for d in load_default_corpus() for _, body in d.sections
        ).lower()
        for query in load_eval_queries():
            assert query.query.lower() not in bodies, query.query

    def test_path_constant_points_at_the_file(self) -> None:
        assert EVAL_QUERIES_PATH.exists()


class TestMetricsAgainstHandComputedValues:
    """Tiny fixtures, exact expected numbers."""

    def _report(
        self, kb: KnowledgeBase, ordering_docs: list[str], gold: tuple[str, ...], k: int = 5
    ) -> RetrievalReport:
        rows = [kb.chunks_for(doc)[0] for doc in ordering_docs]
        ordering = [kb.chunks.index(chunk) for chunk in rows]
        stubbed = kb.with_retriever(FixedRetriever(len(kb.chunks), ordering))
        query = EvalQuery(query="synthetic probe query", gold=gold, split="test")
        return evaluate_retrieval(stubbed, [query], k=k, diversify=False)

    def test_perfect_single_gold(self, kb: KnowledgeBase) -> None:
        report = self._report(kb, ["T1021.002", "T1110", "T1190"], ("T1021.002",))
        assert report.recall_at_k == 1.0
        assert report.mrr == 1.0
        assert report.ndcg_at_k == 1.0
        assert report.misses == ()

    def test_gold_at_rank_two_halves_mrr(self, kb: KnowledgeBase) -> None:
        report = self._report(kb, ["T1110", "T1021.002", "T1190"], ("T1021.002",))
        assert report.mrr == pytest.approx(0.5)
        assert report.recall_at_k == 1.0

    def test_gold_at_rank_three(self, kb: KnowledgeBase) -> None:
        report = self._report(kb, ["T1110", "T1190", "T1021.002"], ("T1021.002",))
        assert report.mrr == pytest.approx(1.0 / 3.0)

    def test_complete_miss(self, kb: KnowledgeBase) -> None:
        report = self._report(kb, ["T1110", "T1190", "T1486"], ("T1021.002",), k=3)
        assert report.recall_at_k == 0.0
        assert report.mrr == 0.0
        assert report.ndcg_at_k == 0.0
        assert report.misses == ("synthetic probe query",)

    def test_partial_recall_with_two_gold(self, kb: KnowledgeBase) -> None:
        report = self._report(
            kb, ["T1021.002", "T1110", "T1190"], ("T1021.002", "T1486"), k=3
        )
        assert report.recall_at_k == pytest.approx(0.5)
        assert report.mrr == 1.0

    def test_ndcg_uses_the_log2_rank_plus_one_discount(self, kb: KnowledgeBase) -> None:
        """Gold at rank 2 of 1 possible: DCG = 1/log2(3), ideal = 1/log2(2) = 1."""
        report = self._report(kb, ["T1110", "T1021.002", "T1190"], ("T1021.002",))
        assert report.ndcg_at_k == pytest.approx(1.0 / np.log2(3.0))

    def test_ndcg_with_two_gold_both_found_in_order(self, kb: KnowledgeBase) -> None:
        """DCG = 1/log2(2) + 1/log2(3); ideal is the same, so nDCG = 1."""
        report = self._report(
            kb, ["T1021.002", "T1486", "T1110"], ("T1021.002", "T1486"), k=3
        )
        assert report.recall_at_k == 1.0
        assert report.ndcg_at_k == pytest.approx(1.0)

    def test_ndcg_penalises_a_worse_ordering_of_the_same_set(
        self, kb: KnowledgeBase
    ) -> None:
        good = self._report(kb, ["T1021.002", "T1486", "T1110"], ("T1021.002", "T1486"), k=3)
        bad = self._report(kb, ["T1110", "T1021.002", "T1486"], ("T1021.002", "T1486"), k=3)
        assert bad.recall_at_k == good.recall_at_k
        assert bad.ndcg_at_k < good.ndcg_at_k

    def test_k_truncates(self, kb: KnowledgeBase) -> None:
        ordering = ["T1110", "T1190", "T1486", "T1021.002"]
        assert self._report(kb, ordering, ("T1021.002",), k=3).recall_at_k == 0.0
        assert self._report(kb, ordering, ("T1021.002",), k=4).recall_at_k == 1.0

    def test_metrics_are_document_level_not_chunk_level(self, kb: KnowledgeBase) -> None:
        """Four chunks of one document must count once, not four times."""
        chunks = kb.chunks_for("T1021.002")
        assert len(chunks) >= 2
        ordering = [kb.chunks.index(chunk) for chunk in chunks]
        stubbed = kb.with_retriever(FixedRetriever(len(kb.chunks), ordering))
        query = EvalQuery(
            query="probe", gold=("T1021.002", "T1110"), split="test"
        )
        report = evaluate_retrieval(stubbed, [query], k=len(ordering), diversify=False)
        assert report.recall_at_k == pytest.approx(0.5)


class TestEvaluateRetrieval:
    def test_rejects_an_empty_query_list(self, kb: KnowledgeBase) -> None:
        with pytest.raises(EvalError, match="no queries"):
            evaluate_retrieval(kb, [], k=5)

    def test_rejects_gold_that_is_not_indexed(self, kb: KnowledgeBase) -> None:
        """A gold label naming nothing would look like a retrieval failure."""
        query = EvalQuery(query="probe", gold=("T9999",), split="test")
        with pytest.raises(EvalError, match="not present in the knowledge base"):
            evaluate_retrieval(kb, [query], k=5)

    def test_split_label_reflects_the_queries(self, kb: KnowledgeBase) -> None:
        assert evaluate_retrieval(kb, load_eval_queries(split="test")[:3], k=5).split == "test"
        mixed = [*load_eval_queries(split="tune")[:1], *load_eval_queries(split="test")[:1]]
        assert evaluate_retrieval(kb, mixed, k=5).split == "test+tune"

    def test_per_kind_recall_is_reported(self, kb: KnowledgeBase) -> None:
        report = evaluate_retrieval(kb, load_eval_queries(split="test"), k=5)
        assert set(report.per_kind_recall) <= {"technique", "cve", "advisory", "playbook"}
        assert all(0.0 <= v <= 1.0 for v in report.per_kind_recall.values())

    def test_link_following_never_lowers_recall(self, kb: KnowledgeBase) -> None:
        """Appending cannot remove a ranked result, so this must hold by construction."""
        report = evaluate_retrieval(kb, load_eval_queries(split="test"), k=5)
        assert report.recall_with_links >= report.recall_at_k

    def test_recall_rises_with_k(self, kb: KnowledgeBase) -> None:
        queries = load_eval_queries(split="test")
        at3 = evaluate_retrieval(kb, queries, k=3).recall_at_k
        at5 = evaluate_retrieval(kb, queries, k=5).recall_at_k
        at10 = evaluate_retrieval(kb, queries, k=10).recall_at_k
        assert at3 <= at5 <= at10

    def test_summary_and_dict_round_trip_the_numbers(self, kb: KnowledgeBase) -> None:
        report = evaluate_retrieval(kb, load_eval_queries(split="test")[:5], k=5)
        assert "recall@5" in report.summary()
        payload = report.as_dict()
        assert payload["recall_at_k"] == report.recall_at_k
        assert payload["mrr"] == report.mrr
        assert payload["recall_with_links"] == report.recall_with_links

    def test_deterministic(self, kb: KnowledgeBase) -> None:
        queries = load_eval_queries(split="test")[:10]
        first = evaluate_retrieval(kb, queries, k=5)
        assert first.as_dict() == evaluate_retrieval(kb, queries, k=5).as_dict()


class TestGates:
    def test_the_shipped_index_clears_the_gates(self, kb: KnowledgeBase) -> None:
        """The F-05 acceptance criterion, on the held-out split."""
        assert_gates(evaluate_retrieval(kb, load_eval_queries(split="test"), k=5))

    def test_gates_refuse_the_tuning_split(self, kb: KnowledgeBase) -> None:
        """The guard against quietly abandoning the tune/test discipline."""
        report = evaluate_retrieval(kb, load_eval_queries(split="tune"), k=5)
        with pytest.raises(EvalError, match="held-out split only"):
            assert_gates(report)

    def test_gates_refuse_a_mixed_split(self, kb: KnowledgeBase) -> None:
        mixed = [*load_eval_queries(split="tune")[:2], *load_eval_queries(split="test")[:2]]
        with pytest.raises(EvalError, match="held-out split only"):
            assert_gates(evaluate_retrieval(kb, mixed, k=5))

    def test_a_failing_recall_is_reported(self) -> None:
        report = RetrievalReport(
            retriever="stub", split="test", k=5, n_queries=61,
            recall_at_k=0.10, mrr=0.99, ndcg_at_k=0.5, recall_with_links=0.99,
        )
        with pytest.raises(EvalError, match="recall@5"):
            assert_gates(report)

    def test_a_failing_mrr_is_reported(self) -> None:
        report = RetrievalReport(
            retriever="stub", split="test", k=5, n_queries=61,
            recall_at_k=0.99, mrr=0.10, ndcg_at_k=0.5, recall_with_links=0.99,
        )
        with pytest.raises(EvalError, match="mrr"):
            assert_gates(report)

    def test_a_failing_linked_recall_is_reported(self) -> None:
        report = RetrievalReport(
            retriever="stub", split="test", k=5, n_queries=61,
            recall_at_k=0.99, mrr=0.99, ndcg_at_k=0.9, recall_with_links=0.10,
        )
        with pytest.raises(EvalError, match=r"recall\+links"):
            assert_gates(report)

    def test_every_failure_is_listed_at_once(self) -> None:
        """One run should tell you everything that is broken, not the first thing."""
        report = RetrievalReport(
            retriever="stub", split="test", k=5, n_queries=61,
            recall_at_k=0.1, mrr=0.1, ndcg_at_k=0.1, recall_with_links=0.1,
        )
        with pytest.raises(EvalError) as caught:
            assert_gates(report)
        message = str(caught.value)
        assert "recall@5" in message and "mrr" in message and "recall+links" in message

    def test_gate_values_leave_headroom_but_are_not_vacuous(self) -> None:
        """A gate below the floor a broken retriever would hit catches nothing."""
        assert 0.5 < GATE_RECALL_AT_5 < 0.95
        assert 0.5 < GATE_MRR < 0.99
        assert GATE_RECALL_WITH_LINKS > GATE_RECALL_AT_5

    @pytest.mark.slow
    def test_every_component_retriever_is_measured_not_assumed(
        self, kb: KnowledgeBase
    ) -> None:
        """Records the honest comparison: the fusion's margin here is under one query.

        On the held-out split the fused retriever and LSA alone are within noise of
        each other, and LSA alone is *better* on linked recall. The fusion ships
        because it was the tuning split's choice and because it does not depend on one
        method being right — not because a held-out measurement established it.
        """
        from sentinel.kb.index import LexicalIndex, LsaIndex

        test = load_eval_queries(split="test")
        bm25 = LexicalIndex(encoder=kb.lexical.encoder, scorer="bm25")
        components = {
            "bm25": bm25,
            "tfidf": LexicalIndex(encoder=kb.lexical.encoder, scorer="tfidf"),
            "lsa": LsaIndex.build(bm25),
        }
        scores = {
            name: evaluate_retrieval(kb.with_retriever(r), test, k=5)
            for name, r in components.items()
        }
        fused = evaluate_retrieval(kb, test, k=5)
        # Every component must be individually competent; a component that is useless
        # alone is a component whose weight is noise.
        for name, report in scores.items():
            assert report.recall_at_k > 0.6, f"{name} recall {report.recall_at_k:.3f}"
            assert report.mrr > 0.8, f"{name} mrr {report.mrr:.3f}"
        # And the fusion must be at least as good as the *worst* component, which is
        # the only thing rank fusion actually guarantees.
        assert fused.recall_at_k >= min(r.recall_at_k for r in scores.values())
