"""Retrievers: BM25, TF-IDF cosine, LSA, and rank fusion over them.

The seam
--------
:class:`Retriever` is the one abstraction the knowledge base depends on. Everything
above it — :class:`~sentinel.kb.retrieve.KnowledgeBase`, the typed
:class:`~sentinel.core.schemas.Evidence` it emits, the Investigation Agent that will
consume it — sees only ``retrieve(text, k) -> tuple[Scored, ...]``. That is what
makes the PRD's stated FAISS-and-transformer stack a swap rather than a rewrite:
a FAISS flat index computes the same inner products :class:`LsaIndex` computes, and
a sentence-transformer encoder would be a third :class:`Retriever` fused in beside
the other two. ``test_kb_index.py::TestRetrieverSeam`` drives the whole knowledge
base through a stub retriever to prove the abstraction is load-bearing rather than
decorative.

Why fuse ranks rather than scores
---------------------------------
BM25 scores are unbounded sums of per-term contributions; cosine scores live in
``[-1, 1]``. Combining them numerically requires normalising two distributions whose
shape depends on the query, and the usual fixes (min-max over the returned window,
z-scoring) are unstable precisely where it matters — when one retriever returns a
single strong hit and the other returns a flat spread, min-max inflates the flat
one's top result to 1.0.

Reciprocal rank fusion sidesteps this by discarding magnitudes and combining
*positions*: each retriever contributes ``weight / (k_rrf + rank)``. It cannot be
destabilised by scale, it needs no per-query calibration, and it degrades gracefully
when one component is useless for a given query — that component's contribution is
simply spread thin. The cost is real and worth stating: a retriever that is *very*
confident cannot express it, so fusion slightly blunts the queries where one method
is decisively right. ``eval.py`` measures fusion against each component alone on a
held-out query split, so this trade is a number rather than an opinion.

Diversification
---------------
:func:`mmr_select` exists because relevance-only ranking retrieves five chunks of
the same document, which is the single most common way a RAG citation list manages
to be both accurate and useless: five citations, one source, no corroboration. An
:class:`~sentinel.core.schemas.InvestigationReport` grounded that way satisfies its
validator and still tells the analyst nothing new.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from sentinel.core.errors import SentinelError
from sentinel.kb.chunk import Chunk
from sentinel.kb.corpus import Corpus
from sentinel.kb.embed import LexicalEncoder, LsaEncoder
from sentinel.kb.sparse import DTYPE, InvertedIndex, SparseMatrix

__all__ = [
    "DEFAULT_B",
    "DEFAULT_EXPANSION_ALPHA",
    "DEFAULT_K1",
    "DEFAULT_RRF_K",
    "HybridIndex",
    "IndexError_",
    "LexicalIndex",
    "LinkExpansionIndex",
    "LsaIndex",
    "Retriever",
    "Scored",
    "mmr_select",
    "reciprocal_rank_fusion",
]

#: BM25 term-frequency saturation. 1.2 is the long-standing default from the
#: original TREC work and is what the tuning split selected here too; see
#: ``docs/BUILD_PLAN.md`` for the grid that was searched.
DEFAULT_K1: Final[float] = 1.2

#: BM25 length normalisation. 0.75 is the standard default. Chunks here are
#: deliberately near-uniform in length (see :mod:`sentinel.kb.chunk`), which is
#: precisely the regime where ``b`` matters least — a useful reminder that a
#: hyper-parameter's importance depends on the data, not on the algorithm.
DEFAULT_B: Final[float] = 0.75

#: Reciprocal rank fusion constant. 60 is the value from the original paper; it
#: controls how quickly a retriever's contribution decays with rank. Lower values
#: make the fusion behave more like "trust whoever ranked it first".
DEFAULT_RRF_K: Final[int] = 60

#: Fraction of a hit's score propagated to each cross-linked document. **Zero by
#: default: measurement rejected this.** The hypothesis was good and the data said
#: no. On the tuning split, score-space expansion cost recall outright (0.922 ->
#: 0.858 at alpha 0.55) and wrecked ordering (MRR 1.000 -> 0.838), because lifted
#: non-gold documents displace lower-ranked *relevant* ones. Re-cast as a fourth
#: fused rank list, where RRF's rank decay bounds the displacement, it became a
#: genuine trade rather than a loss: recall 0.892 -> 0.936 at weight 0.1, but MRR
#: 1.000 -> 0.895 and nDCG 0.896 -> 0.859.
#:
#: A trade is not an improvement, so the ranking stays pure and the relation graph is
#: exposed as a deliberate second step instead --
#: :meth:`~sentinel.kb.retrieve.KnowledgeBase.related` and
#: :meth:`~sentinel.kb.retrieve.KnowledgeBase.follow_links`. That gets the technique
#: mapping F-05 wants into the evidence list *additively*, with no reordering to pay
#: for. The class is kept, tested and measurable because the option is real; the
#: default is off because the number said so.
DEFAULT_EXPANSION_ALPHA: Final[float] = 0.0


class IndexError_(SentinelError):
    """An index was queried before being built, or with an invalid argument.

    Named with a trailing underscore to avoid shadowing the builtin ``IndexError``,
    which :class:`~sentinel.kb.sparse.SparseMatrix` still raises for genuine
    out-of-range row access.
    """


@dataclass(frozen=True, slots=True, order=True)
class Scored:
    """One retrieval result: a row position in the chunk list, and its score.

    ``order=True`` with ``score`` first makes a list of these sort by score, which
    is the only ordering anything here wants.
    """

    score: float
    row: int


@runtime_checkable
class Retriever(Protocol):
    """Anything that can rank chunk rows for a query string."""

    @property
    def name(self) -> str:
        """Short identifier used in fusion weights and in evaluation reports."""
        ...

    @property
    def n_rows(self) -> int:
        """How many chunks this retriever ranks over."""
        ...

    def retrieve(self, text: str, k: int) -> tuple[Scored, ...]:
        """The top ``k`` rows for ``text``, best first, ties broken by row order."""
        ...


def _top_k(scores: npt.NDArray[np.float64], k: int) -> tuple[Scored, ...]:
    """Top ``k`` by score, dropping non-positive scores, deterministic on ties.

    Non-positive scores are dropped rather than padded: returning a chunk that
    matched nothing would produce an ``Evidence`` object whose relevance is zero and
    whose only effect is to make an ungrounded claim look grounded.

    Ties break by ascending row index, which makes every ranking reproducible.
    ``argpartition`` alone does not guarantee that, so the partitioned window is
    re-sorted on the composite key.
    """
    if k <= 0:
        raise IndexError_(f"k must be positive, got {k}")
    positive = np.flatnonzero(scores > 0.0)
    if positive.shape[0] == 0:
        return ()
    if positive.shape[0] > k:
        window = np.argpartition(-scores[positive], k - 1)[:k]
        positive = positive[window]
    order = np.lexsort((positive, -scores[positive]))
    return tuple(Scored(score=float(scores[row]), row=int(row)) for row in positive[order])


@dataclass(slots=True)
class LexicalIndex:
    """Sparse lexical retrieval: BM25 or TF-IDF cosine over the same term matrix.

    Both scorers share the count matrix and the inverted indices, so switching
    ``scorer`` costs nothing and the comparison in ``eval.py`` is genuinely
    apples-to-apples: same tokenization, same chunks, same features, one different
    weighting function.
    """

    encoder: LexicalEncoder
    scorer: str = "bm25"
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B
    _tfidf: SparseMatrix | None = None
    _counts_inverted: InvertedIndex | None = None
    _tfidf_inverted: InvertedIndex | None = None

    def __post_init__(self) -> None:
        if self.scorer not in ("bm25", "tfidf"):
            raise IndexError_(f"unknown scorer {self.scorer!r}; use 'bm25' or 'tfidf'")
        if not self.encoder.is_fitted:
            raise IndexError_("LexicalIndex requires a fitted LexicalEncoder")
        assert self.encoder.counts is not None
        self._tfidf = self.encoder.tfidf_matrix()
        self._counts_inverted = InvertedIndex.from_matrix(self.encoder.counts)
        self._tfidf_inverted = InvertedIndex.from_matrix(self._tfidf)

    @property
    def name(self) -> str:
        return self.scorer

    @property
    def n_rows(self) -> int:
        assert self.encoder.counts is not None
        return self.encoder.counts.n_rows

    @property
    def tfidf(self) -> SparseMatrix:
        assert self._tfidf is not None
        return self._tfidf

    @property
    def tfidf_inverted(self) -> InvertedIndex:
        assert self._tfidf_inverted is not None
        return self._tfidf_inverted

    def score_all(self, text: str) -> npt.NDArray[np.float64]:
        """Raw scores for every chunk. Exposed because fusion and LSA both need it."""
        if self.scorer == "tfidf":
            return self.tfidf_inverted.score(self.encoder.encode_tfidf(text))
        assert self._counts_inverted is not None
        assert self.encoder.idf_bm25 is not None
        assert self.encoder.doc_len is not None
        return self._counts_inverted.bm25(
            self.encoder.encode_counts(text),
            idf=self.encoder.idf_bm25,
            doc_len=self.encoder.doc_len,
            k1=self.k1,
            b=self.b,
        )

    def cosine_scores(self, text: str) -> npt.NDArray[np.float64]:
        """TF-IDF cosine against every chunk, regardless of ``scorer``.

        Needed by :class:`LsaIndex` for its query fold-in and by
        :func:`mmr_select` for chunk-to-chunk similarity, both of which want the
        cosine geometry specifically and must not silently get BM25 instead.
        """
        return self.tfidf_inverted.score(self.encoder.encode_tfidf(text))

    def retrieve(self, text: str, k: int) -> tuple[Scored, ...]:
        return _top_k(self.score_all(text), k)


@dataclass(slots=True)
class LsaIndex:
    """Dense retrieval over latent semantic vectors.

    Composed from a :class:`LexicalIndex` rather than duplicating tokenization,
    because the LSA space is *defined* by that TF-IDF matrix. Fitting on a different
    tokenization from the one used at query time is the classic way a dense index
    ends up subtly misaligned with its own training data.
    """

    lexical: LexicalIndex
    encoder: LsaEncoder

    @classmethod
    def build(cls, lexical: LexicalIndex, *, n_components: int = 192) -> LsaIndex:
        encoder = LsaEncoder(n_components=n_components).fit(lexical.tfidf)
        return cls(lexical=lexical, encoder=encoder)

    @property
    def name(self) -> str:
        return "lsa"

    @property
    def n_rows(self) -> int:
        return self.lexical.n_rows

    def score_all(self, text: str) -> npt.NDArray[np.float64]:
        latent = self.encoder.project_query(self.lexical.cosine_scores(text))
        return self.encoder.similarities(latent)

    def retrieve(self, text: str, k: int) -> tuple[Scored, ...]:
        return _top_k(self.score_all(text), k)


def reciprocal_rank_fusion(
    rankings: Sequence[tuple[Sequence[Scored], float]],
    *,
    n_rows: int,
    rrf_k: int = DEFAULT_RRF_K,
) -> npt.NDArray[np.float64]:
    """Fuse ranked lists into one score vector by reciprocal rank.

    ``rankings`` pairs each retriever's ordered results with its weight. A row
    absent from a retriever's list contributes nothing from that retriever, which is
    the intended behaviour: absence is weak evidence against, not evidence for.
    """
    if rrf_k < 1:
        raise IndexError_(f"rrf_k must be at least 1, got {rrf_k}")
    fused = np.zeros(n_rows, dtype=DTYPE)
    for results, weight in rankings:
        if weight == 0.0:
            continue
        for rank, scored in enumerate(results, start=1):
            if not 0 <= scored.row < n_rows:
                raise IndexError_(f"row {scored.row} outside [0, {n_rows})")
            fused[scored.row] += weight / (rrf_k + rank)
    return fused


@dataclass(slots=True)
class HybridIndex:
    """Reciprocal-rank fusion over several retrievers.

    ``candidate_depth`` is how deep each component is asked before fusing. It must
    exceed the final ``k``: a document ranked eighth by both components should be
    able to win against one ranked first by a single component, and it cannot if the
    components were only asked for their top five.
    """

    components: tuple[tuple[Retriever, float], ...]
    candidate_depth: int = 30
    rrf_k: int = DEFAULT_RRF_K

    def __post_init__(self) -> None:
        if not self.components:
            raise IndexError_("HybridIndex needs at least one component retriever")
        widths = {retriever.n_rows for retriever, _ in self.components}
        if len(widths) != 1:
            raise IndexError_(f"components disagree on corpus size: {sorted(widths)}")
        if any(weight < 0.0 for _, weight in self.components):
            raise IndexError_("component weights must be non-negative")
        if not any(weight > 0.0 for _, weight in self.components):
            raise IndexError_("at least one component weight must be positive")
        if self.candidate_depth < 1:
            raise IndexError_("candidate_depth must be at least 1")

    @property
    def name(self) -> str:
        return "hybrid(" + "+".join(r.name for r, w in self.components if w > 0.0) + ")"

    @property
    def n_rows(self) -> int:
        return self.components[0][0].n_rows

    def score_all(self, text: str) -> npt.NDArray[np.float64]:
        rankings = [
            (retriever.retrieve(text, self.candidate_depth), weight)
            for retriever, weight in self.components
            if weight > 0.0
        ]
        return reciprocal_rank_fusion(rankings, n_rows=self.n_rows, rrf_k=self.rrf_k)

    def retrieve(self, text: str, k: int) -> tuple[Scored, ...]:
        return _top_k(self.score_all(text), k)


@dataclass(slots=True)
class LinkExpansionIndex:
    """Propagate score across the corpus's authored relation graph. One round.

    The problem this solves was found by reading the tuning split's failures rather
    than by theory. The dominant failure had a consistent shape: a query names a
    vulnerability, the right CVE is retrieved at rank 1, and the ATT&CK technique the
    CVE document *explicitly declares a link to* is absent from the top five, with
    slots 2 to 5 filled by lexically similar but topically unrelated CVEs. Five of
    six tuning-split misses had that shape.

    So the corpus already contained the missing answer as structured data, and the
    retriever was throwing it away. ``Corpus.validate`` guarantees every cross-link
    resolves, which makes the relation graph trustworthy input rather than a hint.
    Using it is the same argument Part 2 made for the supply-chain GNN — risk (here,
    relevance) propagates along declared edges — applied one layer up.

    Edges, all treated as undirected
    --------------------------------
    *   ``document.techniques`` — a CVE or playbook to the techniques it maps to.
        Traversed in both directions, because "what technique is this CVE" and "which
        CVEs realise this technique" are both questions an investigation asks.
    *   sub-technique to parent technique, both ways. This is what lets a query about
        password spraying surface ``T1110.003`` *and* ``T1110``, which
        :func:`mmr_select` otherwise actively suppresses: the two documents are
        lexically similar, so diversification reads corroboration as redundancy.

    One round, not two
    ------------------
    Two rounds would let a CVE reach another CVE through a shared technique, and
    ``T1190`` alone links thirteen CVEs, so a second hop floods the ranking with
    everything remotely related to "a web application was exploited". This is the
    same finding Part 2 recorded for the GNN's layer count, arrived at from the
    opposite direction: there, depth bought reach at the cost of over-smoothing;
    here, depth destroys precision through a hub node. The single round is a
    measured choice, not a simplification.

    What measurement said about it
    ------------------------------
    Expansion is safe for the *top* hit and damaging below it, and the distinction is
    worth stating precisely because the obvious guess is wrong in both directions.

    Because the bonus is ``alpha`` times a neighbour's score, ``max``-aggregated, and
    the edges are symmetric, the highest-scoring document provably cannot be
    overtaken: the best score any document can receive is ``alpha`` times the global
    maximum, and the global maximum receives at least ``alpha`` times whatever it is
    linked to. So the primary answer stays primary.

    What expansion does damage is everything below it. Lifted non-gold documents
    displace *lower-ranked relevant* ones, and on the tuning split that moved the
    first correct answer down often enough to take mean reciprocal rank from 1.000 to
    0.838. Recall and MRR therefore move in opposite directions, which is the whole
    reason this is a trade rather than an improvement.
    ``test_kb_index.py::TestLinkExpansionMeasuredHonestly`` pins both halves, and
    :data:`DEFAULT_EXPANSION_ALPHA` records what replaced it.
    """

    base: Retriever
    #: doc_id -> the doc_ids it is linked to (symmetric closure, no self-edges).
    neighbours: dict[str, tuple[str, ...]]
    #: doc_id -> the chunk rows belonging to it.
    rows_by_doc: dict[str, tuple[int, ...]]
    #: row -> doc_id, for provenance.
    doc_by_row: tuple[str, ...]
    alpha: float = DEFAULT_EXPANSION_ALPHA
    source_depth: int = 20

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha < 1.0:
            raise IndexError_(f"alpha must be in [0, 1), got {self.alpha}")
        if self.source_depth < 1:
            raise IndexError_("source_depth must be at least 1")
        if len(self.doc_by_row) != self.base.n_rows:
            raise IndexError_(
                f"doc_by_row has {len(self.doc_by_row)} entries for "
                f"{self.base.n_rows} rows"
            )

    @classmethod
    def build(
        cls,
        base: Retriever,
        *,
        chunks: Sequence[Chunk],
        corpus: Corpus,
        alpha: float = DEFAULT_EXPANSION_ALPHA,
        source_depth: int = 20,
    ) -> LinkExpansionIndex:
        """Derive the relation graph from ``corpus`` and the row map from ``chunks``."""
        edges: dict[str, set[str]] = {}

        def link(left: str, right: str) -> None:
            if left == right:
                return
            edges.setdefault(left, set()).add(right)
            edges.setdefault(right, set()).add(left)

        for document in corpus:
            for technique in document.techniques:
                if technique in corpus:
                    link(document.doc_id, technique)
            parent = document.parent_id
            if parent is not None and parent in corpus:
                link(document.doc_id, parent)

        rows_by_doc: dict[str, list[int]] = {}
        for row, chunk in enumerate(chunks):
            rows_by_doc.setdefault(chunk.doc_id, []).append(row)

        return cls(
            base=base,
            neighbours={doc: tuple(sorted(linked)) for doc, linked in edges.items()},
            rows_by_doc={doc: tuple(rows) for doc, rows in rows_by_doc.items()},
            doc_by_row=tuple(chunk.doc_id for chunk in chunks),
            alpha=alpha,
            source_depth=source_depth,
        )

    @property
    def name(self) -> str:
        return f"{self.base.name}+links"

    @property
    def n_rows(self) -> int:
        return self.base.n_rows

    def score_with_provenance(
        self, text: str
    ) -> tuple[npt.NDArray[np.float64], dict[int, tuple[str, ...]]]:
        """Scores plus, for each lifted row, which documents lifted it.

        Provenance is returned rather than folded away because a citation the analyst
        cannot account for is the thing this project refuses to ship. "``T1190`` is
        here because ``CVE-2017-5638`` maps to it" is a reviewable statement;
        a technique appearing in the list for no visible reason is not.
        """
        scores = np.asarray(self.base.score_all(text), dtype=DTYPE).copy()
        if self.alpha == 0.0:
            return scores, {}

        # Document score is its best chunk's score: a document is as relevant as its
        # most relevant passage, not as its average, which would penalise long
        # documents for having sections the query does not ask about.
        best: dict[str, float] = {}
        for row, score in enumerate(scores):
            if score <= 0.0:
                continue
            doc_id = self.doc_by_row[row]
            if score > best.get(doc_id, 0.0):
                best[doc_id] = float(score)
        if not best:
            return scores, {}

        sources = sorted(best.items(), key=lambda item: (-item[1], item[0]))
        sources = sources[: self.source_depth]

        bonus: dict[str, float] = {}
        provenance: dict[str, set[str]] = {}
        for doc_id, score in sources:
            for neighbour in self.neighbours.get(doc_id, ()):
                contribution = self.alpha * score
                # max, not sum: three CVEs mapping to T1190 is not three times the
                # evidence that the query is about T1190, and summing would let a hub
                # technique outrank the specific vulnerability that named it.
                if contribution > bonus.get(neighbour, 0.0):
                    bonus[neighbour] = contribution
                provenance.setdefault(neighbour, set()).add(doc_id)

        lifted: dict[int, tuple[str, ...]] = {}
        for doc_id, extra in bonus.items():
            for row in self.rows_by_doc.get(doc_id, ()):
                # Added, so a neighbour that also matched the query text directly
                # keeps that advantage over one reached only structurally.
                scores[row] += extra
                lifted[row] = tuple(sorted(provenance.get(doc_id, ())))
        return scores, lifted

    def score_all(self, text: str) -> npt.NDArray[np.float64]:
        return self.score_with_provenance(text)[0]

    def retrieve(self, text: str, k: int) -> tuple[Scored, ...]:
        return _top_k(self.score_all(text), k)


def mmr_select(
    candidates: Sequence[Scored],
    *,
    similarity: SparseMatrix,
    k: int,
    lambda_: float = 0.7,
) -> tuple[Scored, ...]:
    """Maximal marginal relevance: pick ``k`` results that are relevant *and* distinct.

    At each step the chosen item maximises
    ``lambda_ * relevance - (1 - lambda_) * max_similarity_to_already_chosen``.
    ``lambda_ = 1`` reduces to plain relevance ranking; lower values trade relevance
    for coverage.

    Relevance is min-max scaled across the candidate window first. Without it the
    trade-off is meaningless, because BM25 relevance spans tens while cosine
    similarity spans ``[0, 1]`` — ``lambda_`` would then mean something different for
    every query, and a single knob that means different things per query is worse
    than no knob.

    ``similarity`` must be the L2-normalized TF-IDF matrix, so a row dot product is a
    cosine. Similarity is measured on lexical overlap rather than in the latent
    space on purpose: two chunks of the *same document* share their header and most
    of their vocabulary, which is exactly the redundancy this is meant to catch.
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise IndexError_(f"lambda_ must be in [0, 1], got {lambda_}")
    if k <= 0:
        raise IndexError_(f"k must be positive, got {k}")
    if not candidates:
        return ()

    scores = np.asarray([c.score for c in candidates], dtype=DTYPE)
    span = float(scores.max() - scores.min())
    relevance = (
        np.ones_like(scores) if span == 0.0 else (scores - scores.min()) / span
    )

    dense = np.asarray(
        [_dense_row(similarity, c.row) for c in candidates], dtype=DTYPE
    )
    pairwise = dense @ dense.T

    chosen: list[int] = []
    remaining = set(range(len(candidates)))
    while remaining and len(chosen) < k:
        best_position = -1
        best_value = -np.inf
        for position in sorted(remaining):
            penalty = max((pairwise[position, other] for other in chosen), default=0.0)
            value = lambda_ * relevance[position] - (1.0 - lambda_) * penalty
            if value > best_value:
                best_value, best_position = value, position
        chosen.append(best_position)
        remaining.discard(best_position)
    return tuple(candidates[position] for position in chosen)


def _dense_row(matrix: SparseMatrix, row: int) -> npt.NDArray[np.float64]:
    dense = np.zeros(matrix.n_cols, dtype=DTYPE)
    columns, values = matrix.row(row)
    dense[columns] = values
    return dense
