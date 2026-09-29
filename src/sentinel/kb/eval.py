"""Retrieval evaluation: labelled queries, ranking metrics, acceptance gates.

Why a labelled query set exists at all
--------------------------------------
"The lateral-movement query returns T1021" is a demo, not a measurement. It is one
query, chosen after the fact, and it will pass for any retriever that does anything
at all. A knowledge base whose quality is unmeasured is a knowledge base whose
quality is unknown, and the Investigation Agent's entire F-05 guarantee rests on it:
a report can be perfectly grounded in the wrong chunks.

So ``data/eval_queries.jsonl`` holds 95 queries written in analyst language, each
labelled with the corpus documents that *should* be retrieved. They are deliberately
not paraphrases of the corpus text — several describe a technique without using any
of its title words ("many failed logins for one password across hundreds of
accounts" for password spraying), because that is the case that separates a real
retriever from a keyword echo.

The tune/test split is the point
--------------------------------
Every query carries ``split``: ``tune`` (34) or ``test`` (61). Fusion weights, BM25's
``k1`` and ``b``, the LSA dimension and the RRF constant were all selected by grid
search on the ``tune`` half. Every number reported in ``docs/BUILD_PLAN.md`` and
every threshold in :func:`assert_gates` is computed on the ``test`` half, which no
configuration decision ever saw.

This discipline is here because Part 2 already paid for the lesson: a 27-configuration
grid for the anomaly ensemble scored 0.825 on its tuning seeds and 0.740 held out.
Retrieval metrics on ~90 queries are noisier than they look — one query is over a
percentage point of recall — so a threshold set on the data used to choose the
configuration measures the choosing, not the retriever.

Metrics, and what each one hides
--------------------------------
*   **Recall@k** — the fraction of a query's gold documents appearing in the top
    ``k``. The headline number, because grounding needs the right *source* present,
    not necessarily first. It ignores ordering entirely.
*   **MRR** — reciprocal rank of the *first* gold hit, averaged. Captures whether the
    right answer is at the top, and ignores everything after it.
*   **nDCG@k** — discounted gain over all gold hits, normalised by the best possible
    ordering. The only one of the three that notices both how many gold documents
    were found and where. Reported because it is the one that can fall while the
    other two hold steady, which happens when diversification reshuffles a correct
    result set.

All three are computed at the *document* level, not the chunk level: a query is
answered by finding the right document, and whether that arrived via its description
chunk or its detection chunk is not something an analyst cares about.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np

from sentinel.core.errors import SentinelError
from sentinel.kb.corpus import DEFAULT_CORPUS_DIR
from sentinel.kb.retrieve import KnowledgeBase

__all__ = [
    "EVAL_QUERIES_PATH",
    "GATE_MRR",
    "GATE_RECALL_AT_5",
    "GATE_RECALL_WITH_LINKS",
    "EvalQuery",
    "RetrievalReport",
    "assert_gates",
    "evaluate_retrieval",
    "load_eval_queries",
]

EVAL_QUERIES_PATH: Final[Path] = DEFAULT_CORPUS_DIR / "eval_queries.jsonl"

#: Acceptance gate on recall@5 over the held-out split. Measured 0.768 with the
#: shipped configuration, so the gate sits below that with room for the variation a
#: 61-query split carries -- one query is 1.6 percentage points here, so a gate set
#: at the measured value would fail on a change that moved nothing real.
GATE_RECALL_AT_5: Final[float] = 0.72

#: Mean reciprocal rank gate on the held-out split. Measured 0.930: the first correct
#: document is usually rank 1. A first hit at rank 2 on every query would score 0.50,
#: so 0.85 still requires it to be first most of the time.
GATE_MRR: Final[float] = 0.85

#: Gate on recall@5 after link-following, which appends cross-linked documents without
#: displacing any ranked result. Measured 0.896. Higher than :data:`GATE_RECALL_AT_5`
#: because the relation graph is authored and validated, so this number *should* be
#: high; if it falls, either the graph lost edges or the ranking stopped surfacing the
#: documents that carry them.
GATE_RECALL_WITH_LINKS: Final[float] = 0.84


class EvalError(SentinelError):
    """The evaluation set is malformed or references documents that do not exist."""


@dataclass(frozen=True, slots=True)
class EvalQuery:
    """One labelled retrieval query."""

    query: str
    gold: tuple[str, ...]
    split: str
    note: str = ""

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise EvalError("empty query")
        if not self.gold:
            raise EvalError(f"{self.query!r}: no gold documents")
        if self.split not in ("tune", "test"):
            raise EvalError(f"{self.query!r}: split must be 'tune' or 'test'")
        if len(set(self.gold)) != len(self.gold):
            raise EvalError(f"{self.query!r}: duplicate gold ids")


def load_eval_queries(
    path: Path | str = EVAL_QUERIES_PATH, *, split: str | None = None
) -> tuple[EvalQuery, ...]:
    """Load the labelled query set, optionally filtered to one split."""
    source = Path(path)
    if not source.exists():
        raise EvalError(f"evaluation query file missing: {source}")
    queries: list[EvalQuery] = []
    seen: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise EvalError(f"{source.name}:{number}: {exc}") from exc
            query = EvalQuery(
                query=record["query"],
                gold=tuple(record["gold"]),
                split=record.get("split", "test"),
                note=record.get("note", ""),
            )
            if query.query in seen:
                raise EvalError(f"{source.name}:{number}: duplicate query text")
            seen.add(query.query)
            if split is None or query.split == split:
                queries.append(query)
    if not queries:
        raise EvalError(f"no queries for split {split!r}")
    return tuple(queries)


@dataclass(frozen=True, slots=True)
class RetrievalReport:
    """Measured retrieval quality for one retriever on one query split."""

    retriever: str
    split: str
    k: int
    n_queries: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    #: Queries where no gold document appeared in the top ``k``. Kept in full rather
    #: than counted, because the interesting engineering is always in reading them:
    #: the failures on this corpus cluster into "two near-identical CVEs" and
    #: "procedural question answered by a technique instead of a playbook", and
    #: neither is visible in an aggregate.
    misses: tuple[str, ...] = ()
    per_kind_recall: dict[str, float] = field(default_factory=dict)
    #: Recall@k after :meth:`~sentinel.kb.retrieve.KnowledgeBase.follow_links`, which
    #: appends cross-linked documents *without* displacing any ranked result. This is
    #: the number that describes what the Investigation Agent can actually assemble,
    #: as opposed to what one ranking pass returns, and the gap between it and
    #: :attr:`recall_at_k` is the value of the corpus's relation graph — bought here
    #: with no ordering cost, unlike folding links into the ranking.
    recall_with_links: float = 0.0

    def summary(self) -> str:
        return (
            f"{self.retriever:<22} {self.split:<5} n={self.n_queries:<3} "
            f"recall@{self.k}={self.recall_at_k:.3f} mrr={self.mrr:.3f} "
            f"ndcg@{self.k}={self.ndcg_at_k:.3f} "
            f"recall+links={self.recall_with_links:.3f} misses={len(self.misses)}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "retriever": self.retriever,
            "split": self.split,
            "k": self.k,
            "n_queries": self.n_queries,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
            "misses": list(self.misses),
            "per_kind_recall": self.per_kind_recall,
            "recall_with_links": self.recall_with_links,
        }


def _dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain with the standard ``log2(rank + 1)`` discount."""
    return float(
        sum(gain / np.log2(position + 1) for position, gain in enumerate(gains, start=1))
    )


def evaluate_retrieval(
    kb: KnowledgeBase,
    queries: Sequence[EvalQuery],
    *,
    k: int = 5,
    diversify: bool = True,
) -> RetrievalReport:
    """Measure ``kb`` on ``queries`` at cut-off ``k``.

    Gold ids must exist in the knowledge base. A gold label naming a document that
    was never indexed would silently depress recall and look like a retrieval
    failure, so it is an error instead — the same reasoning as
    :meth:`sentinel.kb.corpus.Corpus.validate` rejecting a dangling cross-link.
    """
    if not queries:
        raise EvalError("no queries to evaluate")
    indexed = {chunk.doc_id for chunk in kb.chunks}
    dangling = sorted({g for q in queries for g in q.gold if g not in indexed})
    if dangling:
        raise EvalError(f"gold ids not present in the knowledge base: {dangling}")

    recalls: list[float] = []
    recalls_linked: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    misses: list[str] = []
    # Per-kind recall is tracked by the kind of each *gold* document, so "are
    # playbooks retrievable for procedural questions" is answerable separately from
    # the aggregate, where 12 playbooks cannot move a number dominated by techniques.
    kind_hits: dict[str, list[float]] = {}

    for query in queries:
        hits = kb.search(query.query, k=k, diversify=diversify)
        # Deduplicate to document level, keeping each document's best rank.
        ranked: list[str] = []
        for hit in hits:
            if hit.doc_id not in ranked:
                ranked.append(hit.doc_id)

        gold = set(query.gold)
        found = [doc_id for doc_id in ranked if doc_id in gold]
        recalls.append(len(found) / len(gold))

        # Same k, same ordering, plus whatever the declared links reach. Measured
        # separately so the ranking's own quality is never flattered by it.
        with_links = {hit.doc_id for hit in kb.follow_links(hits, per_hit=2)}
        recalls_linked.append(len(gold & with_links) / len(gold))

        first = next(
            (position for position, doc_id in enumerate(ranked, start=1) if doc_id in gold),
            None,
        )
        reciprocal_ranks.append(0.0 if first is None else 1.0 / first)
        if first is None:
            misses.append(query.query)

        gains = [1.0 if doc_id in gold else 0.0 for doc_id in ranked]
        ideal = [1.0] * min(len(gold), max(len(ranked), 1))
        ideal_dcg = _dcg(ideal)
        ndcgs.append(0.0 if ideal_dcg == 0.0 else _dcg(gains) / ideal_dcg)

        for doc_id in query.gold:
            document_kind = next(
                (c.kind for c in kb.chunks if c.doc_id == doc_id), "unknown"
            )
            kind_hits.setdefault(document_kind, []).append(
                1.0 if doc_id in ranked else 0.0
            )

    splits = {query.split for query in queries}
    return RetrievalReport(
        retriever=kb.retriever.name,
        split="+".join(sorted(splits)),
        k=k,
        n_queries=len(queries),
        recall_at_k=float(np.mean(recalls)),
        recall_with_links=float(np.mean(recalls_linked)),
        mrr=float(np.mean(reciprocal_ranks)),
        ndcg_at_k=float(np.mean(ndcgs)),
        misses=tuple(misses),
        per_kind_recall={
            kind: float(np.mean(values)) for kind, values in sorted(kind_hits.items())
        },
    )


def assert_gates(report: RetrievalReport) -> None:
    """Raise unless ``report`` clears the F-05 retrieval gates.

    Called by ``scripts/evaluate.py --kb`` so the gate fails a build rather than
    being a number in a document nobody re-reads.
    """
    if report.split != "test":
        raise EvalError(
            f"gates are defined on the held-out split only, got {report.split!r}; "
            "asserting them on tuning queries would measure the tuning"
        )
    failures: list[str] = []
    if report.recall_at_k < GATE_RECALL_AT_5:
        failures.append(
            f"recall@{report.k} {report.recall_at_k:.3f} < gate {GATE_RECALL_AT_5:.2f}"
        )
    if report.mrr < GATE_MRR:
        failures.append(f"mrr {report.mrr:.3f} < gate {GATE_MRR:.2f}")
    if report.recall_with_links < GATE_RECALL_WITH_LINKS:
        failures.append(
            f"recall+links {report.recall_with_links:.3f} < "
            f"gate {GATE_RECALL_WITH_LINKS:.2f}"
        )
    if failures:
        raise EvalError("retrieval gates failed: " + "; ".join(failures))
