"""The knowledge base: build, search, resolve, persist (PRD F-05, Section 5.5.6).

F-05's acceptance criterion is *"every factual claim traces to a retrieved KB chunk
or raw log line"*. Part 1 already made an ungrounded
:class:`~sentinel.core.schemas.InvestigationReport` unconstructible — its validator
rejects a claim citing a ref that is not in the evidence list. This module closes
the other half of the same guarantee: that a ref in the evidence list points at real
indexed content. :meth:`KnowledgeBase.resolve` is what makes that checkable, and
``test_kb_retrieve.py`` asserts it for every ref the knowledge base ever returns.

Retrieved evidence is untrusted input
-------------------------------------
This is the part that is easy to get wrong. A knowledge base is normally treated as
the trusted side of a RAG system — it is "our data", so it feels safe. But the
corpus contains vulnerability descriptions and advisories, and those come from public
feeds that accept third-party submissions. Text in a CVE description reaches the
Investigation Agent's prompt through precisely the channel that is supposed to
*ground* the model, which makes a poisoned advisory a higher-quality injection vector
than a poisoned alert body: the alert is already fenced as hostile, while the
citation arrives wearing the authority of evidence.

So: every chunk is scanned at build time with the same
:func:`~sentinel.core.untrusted.scan_for_injection` used on alert payloads, chunks
that look like instructions are excluded from retrieval by default and reported for
audit, and :attr:`~sentinel.core.schemas.Evidence.excerpt` is typed untrusted even
for content this repository ships itself. ``PB-INJECTION-DEFENSE`` in the corpus is
the operator-facing version of the same rule.

Ranking, honestly
-----------------
The default retriever is a reciprocal-rank fusion of BM25, TF-IDF cosine and LSA.
Fusion weights and ``k`` come from a grid searched on the *tuning* half of the
labelled query set; every number reported in ``docs/BUILD_PLAN.md`` is from the
held-out half. That split exists because Part 2 already produced the lesson: a
27-configuration grid scored 0.825 on its tuning seeds and 0.740 held out.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Final

import numpy as np

from sentinel.core.errors import SentinelError
from sentinel.core.schemas import Evidence, EvidenceKind
from sentinel.core.untrusted import InjectionScan, InjectionVerdict, scan_for_injection
from sentinel.kb.chunk import Chunk, chunk_corpus
from sentinel.kb.corpus import Corpus, DocumentKind, load_default_corpus
from sentinel.kb.embed import ANALYZER_VERSION, DEFAULT_N_FEATURES, LexicalEncoder, LsaEncoder
from sentinel.kb.index import (
    DEFAULT_B,
    DEFAULT_EXPANSION_ALPHA,
    DEFAULT_K1,
    DEFAULT_RRF_K,
    HybridIndex,
    LexicalIndex,
    LinkExpansionIndex,
    LsaIndex,
    Retriever,
    Scored,
    mmr_select,
)
from sentinel.kb.sparse import DTYPE, SparseMatrix
from sentinel.kb.text import IDENTIFIER_PATTERN

__all__ = [
    "ARTIFACT_VERSION",
    "DEFAULT_WEIGHTS",
    "KnowledgeBase",
    "KnowledgeBaseError",
    "RetrievalHit",
]

#: Persisted-artifact format version. Bumped on any change to what
#: :meth:`KnowledgeBase.save` writes; an older artifact is refused rather than
#: misread.
ARTIFACT_VERSION: Final[str] = "kb-artifact-1"

#: Fusion weights, selected on the tuning query split. BM25 leads because analyst
#: queries are keyword-dense and full of identifiers; LSA contributes least but
#: contributes on exactly the paraphrase queries the other two miss, which is why
#: its weight is non-zero despite the lowest standalone score.
DEFAULT_WEIGHTS: Final[dict[str, float]] = {"bm25": 1.0, "tfidf": 0.5, "lsa": 0.5}

#: How much a citation reached by following a declared link is discounted relative
#: to one that matched the query text. Not a tuned number and not pretending to be:
#: it encodes the judgement that inherited relevance is real but secondary, the same
#: judgement ``graph/explain.py`` makes about an inherited exposure path.
LINK_RELEVANCE_DISCOUNT: Final[float] = 0.6

#: Chunks at or above this injection verdict are excluded from retrieval by default.
_EXCLUDE_AT: Final[InjectionVerdict] = InjectionVerdict.LIKELY_INJECTION


class KnowledgeBaseError(SentinelError):
    """The knowledge base could not be built, loaded, or queried as asked."""


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """One retrieved chunk, with everything needed to cite or audit it."""

    chunk: Chunk
    rank: int
    raw_score: float
    relevance: float
    retriever: str
    injection: InjectionScan
    #: Documents whose cross-links lifted this chunk into the result set, empty when
    #: it matched the query text directly. This is the citation's own explanation:
    #: "T1190 is cited because CVE-2017-5638 maps to it" is reviewable, and an
    #: unexplained entry in an evidence list is exactly what F-05 exists to prevent.
    via: tuple[str, ...] = ()

    @property
    def is_structural(self) -> bool:
        """True when this chunk was reached through the relation graph, not the text."""
        return bool(self.via)

    @property
    def ref(self) -> str:
        """The citation ref. Resolvable via :meth:`KnowledgeBase.resolve`."""
        return self.chunk.chunk_id

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id

    @property
    def looks_like_injection(self) -> bool:
        return self.injection.verdict is not InjectionVerdict.CLEAN

    def as_evidence(self, *, retrieved_at: datetime | None = None) -> Evidence:
        """Typed :class:`~sentinel.core.schemas.Evidence` for an agent report."""
        return self.chunk.as_evidence(
            evidence_kind=_EVIDENCE_KIND_FOR[self.chunk.kind],
            relevance=self.relevance,
            retrieved_at=retrieved_at,
        )


_EVIDENCE_KIND_FOR: Final[dict[str, EvidenceKind]] = {
    kind.value: kind.evidence_kind for kind in DocumentKind
}


@dataclass(slots=True)
class KnowledgeBase:
    """A built, queryable knowledge base over a chunked corpus."""

    chunks: tuple[Chunk, ...]
    retriever: Retriever
    lexical: LexicalIndex
    scans: tuple[InjectionScan, ...]
    corpus: Corpus | None = None
    #: ref -> row. Built in ``__post_init__``; declared as a field because
    #: ``slots=True`` means an attribute that is not declared cannot be assigned.
    _by_ref: dict[str, int] = field(default_factory=dict, repr=False, compare=False)
    #: doc_id -> its chunk rows. Used to price a structurally-reached citation
    #: against the strongest text match its source document had.
    _rows_by_doc: dict[str, tuple[int, ...]] = field(
        default_factory=dict, repr=False, compare=False
    )
    #: doc_id -> cross-linked doc_ids (symmetric). Populated from the corpus when one
    #: is available; empty for a knowledge base loaded from an artifact, which stores
    #: chunks rather than documents.
    _links: dict[str, tuple[str, ...]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if len(self.chunks) != self.retriever.n_rows:
            raise KnowledgeBaseError(
                f"{len(self.chunks)} chunks but retriever ranks "
                f"{self.retriever.n_rows} rows"
            )
        if len(self.scans) != len(self.chunks):
            raise KnowledgeBaseError("one injection scan per chunk is required")
        self._by_ref.clear()
        self._by_ref.update(
            {chunk.chunk_id: index for index, chunk in enumerate(self.chunks)}
        )
        if len(self._by_ref) != len(self.chunks):
            raise KnowledgeBaseError("duplicate chunk ids; refs would be ambiguous")
        grouped: dict[str, list[int]] = {}
        for row, chunk in enumerate(self.chunks):
            grouped.setdefault(chunk.doc_id, []).append(row)
        self._rows_by_doc.clear()
        self._rows_by_doc.update({doc: tuple(rows) for doc, rows in grouped.items()})
        self._links.clear()
        if self.corpus is not None:
            self._links.update(_relation_graph(self.corpus, set(self._rows_by_doc)))

    # --- construction --------------------------------------------------------- #

    @classmethod
    def build(
        cls,
        corpus: Corpus | None = None,
        *,
        n_features: int = DEFAULT_N_FEATURES,
        weights: dict[str, float] | None = None,
        n_components: int = 192,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        rrf_k: int = DEFAULT_RRF_K,
        candidate_depth: int = 30,
        target_words: int | None = None,
        expansion_alpha: float = DEFAULT_EXPANSION_ALPHA,
    ) -> KnowledgeBase:
        """Chunk, embed and index a corpus. This is the PRD's build-time step."""
        source = corpus if corpus is not None else load_default_corpus()
        kwargs = {} if target_words is None else {"target_words": target_words}
        chunks = chunk_corpus(source, **kwargs)  # type: ignore[arg-type]
        if not chunks:
            raise KnowledgeBaseError("corpus produced no chunks")

        encoder = LexicalEncoder(n_features=n_features).fit(
            [chunk.embed_text for chunk in chunks]
        )
        bm25 = LexicalIndex(encoder=encoder, scorer="bm25", k1=k1, b=b)
        tfidf = LexicalIndex(encoder=encoder, scorer="tfidf", k1=k1, b=b)
        lsa = LsaIndex.build(bm25, n_components=n_components)

        chosen = dict(DEFAULT_WEIGHTS if weights is None else weights)
        unknown = sorted(set(chosen) - {"bm25", "tfidf", "lsa"})
        if unknown:
            raise KnowledgeBaseError(f"unknown retriever weights: {unknown}")
        retriever: Retriever = HybridIndex(
            components=(
                (bm25, chosen.get("bm25", 0.0)),
                (tfidf, chosen.get("tfidf", 0.0)),
                (lsa, chosen.get("lsa", 0.0)),
            ),
            candidate_depth=candidate_depth,
            rrf_k=rrf_k,
        )
        if expansion_alpha > 0.0:
            retriever = LinkExpansionIndex.build(
                retriever, chunks=chunks, corpus=source, alpha=expansion_alpha
            )
        return cls(
            chunks=chunks,
            retriever=retriever,
            lexical=bm25,
            scans=tuple(scan_for_injection(chunk.excerpt) for chunk in chunks),
            corpus=source,
        )

    def with_retriever(self, retriever: Retriever) -> KnowledgeBase:
        """The same chunks and scans, ranked by a different retriever.

        This is how ``eval.py`` compares BM25, TF-IDF, LSA and the fusion without
        re-chunking or re-fitting, and it is the seam a FAISS or transformer backend
        would enter through.
        """
        return KnowledgeBase(
            chunks=self.chunks,
            retriever=retriever,
            lexical=self.lexical,
            scans=self.scans,
            corpus=self.corpus,
        )

    # --- query ---------------------------------------------------------------- #

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        kinds: Collection[DocumentKind] | None = None,
        diversify: bool = True,
        mmr_lambda: float = 0.7,
        include_suspicious: bool = False,
        require_known_terms: bool = True,
        retrieved_at: datetime | None = None,
    ) -> tuple[RetrievalHit, ...]:
        """Retrieve the ``k`` best chunks for ``query``.

        ``kinds`` filters *after* scoring rather than before, so a filtered search
        returns the best ``k`` of the requested kinds instead of whatever survived a
        pre-filter — the two differ whenever the unfiltered top ``k`` is dominated by
        one kind, which for "what do I do about X" queries it usually is.

        ``diversify`` applies :func:`~sentinel.kb.index.mmr_select` over a candidate
        window, so the returned set spans documents instead of returning five chunks
        of one. Turn it off to inspect pure relevance ordering.

        ``include_suspicious`` is the escape hatch for auditing poisoned content.
        Leaving it false is what keeps a chunk that reads like an instruction out of
        the Investigation Agent's prompt.
        """
        if not query.strip():
            raise KnowledgeBaseError("empty query")
        if k <= 0:
            raise KnowledgeBaseError(f"k must be positive, got {k}")

        # Nothing in the query occurs in the corpus, so every score it could earn
        # comes from a feature-hash collision. Returning those would hand the
        # Investigation Agent citations that look thin but real, and F-05's whole
        # point is that a citation means something. See :func:`kb.embed.term_digest`
        # for why relevance magnitude cannot answer this and an exact check can.
        if require_known_terms and not self.lexical.encoder.known_terms(query):
            return ()

        # Over-fetch: filtering and diversification both discard candidates, and a
        # window of exactly k would return fewer than k results whenever either
        # fires. Four times k with a floor is enough at this corpus size.
        depth = max(4 * k, 40)
        # ``LinkExpansionIndex`` can report *why* a row is in the window, which is
        # worth one isinstance check: the alternative is an evidence list containing
        # entries whose presence nobody can account for.
        provenance: dict[int, tuple[str, ...]] = {}
        if isinstance(self.retriever, LinkExpansionIndex):
            scores, provenance = self.retriever.score_with_provenance(query)
            ranked = _rank_scores(scores, depth)
        else:
            ranked = self.retriever.retrieve(query, depth)
        candidates = [
            scored
            for scored in ranked
            if self._admissible(scored.row, kinds, include_suspicious)
        ]
        # An identifier is a lookup, not a similarity question. A query naming
        # ``CVE-2021-44228`` is not asking which document is *most like* that string,
        # and the dense component has no way to know the difference: on a one-token
        # query it folds a single feature into latent space and confidently returns a
        # neighbour. Measured, ``search("CVE-2021-44228")`` put an unrelated technique
        # first. Named documents are resolved directly and take the leading slots.
        #
        # They are held out of diversification rather than merely prepended to it.
        # Prepending is not enough -- MMR re-orders whatever it is given by relevance,
        # so a promoted document with an ordinary score sinks straight back down, which
        # is exactly how the first attempt at this failed. Nor is it right to hand them
        # an artificially huge score to win that comparison: min-max scaling inside MMR
        # would then flatten every real candidate to zero relevance and the tail would
        # be ordered by diversity alone. Keeping them out of the comparison entirely
        # leaves both behaviours intact.
        promoted = self._named_documents(query, kinds, include_suspicious)
        scored_by_row = {scored.row: scored for scored in candidates}
        remaining = [scored for scored in candidates if scored.row not in promoted]
        if not promoted and not remaining:
            return ()

        slots = max(0, k - len(promoted))
        diversified = (
            mmr_select(
                remaining, similarity=self.lexical.tfidf, k=slots, lambda_=mmr_lambda
            )
            if diversify
            else tuple(remaining[:slots])
        ) if slots else ()

        selected = (
            *(
                # ``raw_score`` is the retriever's own score when it had one and 0.0
                # when the ranking never surfaced this document at all. The zero is
                # accurate rather than a placeholder: the document is present because
                # the query named it, not because anything ranked it. ``relevance``
                # stays meaningful either way, being a cosine against the query.
                scored_by_row.get(row, Scored(score=0.0, row=row))
                for row in promoted
            ),
            *diversified,
        )[:k]
        if not selected:
            return ()

        # Relevance is the TF-IDF cosine of the chunk against the query, not a
        # rescaled fusion score. Reciprocal-rank fusion scores are almost useless as
        # a magnitude: with the default rrf_k the gap between rank 1 and rank 5 is
        # 1/61 versus 1/65, so every hit reported a relevance between 0.89 and 1.00
        # and the number carried no information at all. A cosine between two unit,
        # non-negative vectors is genuinely bounded in [0, 1], means the same thing
        # for every query, and is low exactly when the match is thin. Fusion still
        # decides the ordering, which is what it is good at.
        cosine = self.lexical.cosine_scores(query)
        lifted = provenance.get if provenance else _no_provenance
        return tuple(
            RetrievalHit(
                chunk=self.chunks[scored.row],
                rank=position,
                raw_score=scored.score,
                relevance=_relevance(
                    own=float(cosine[scored.row]),
                    via=lifted(scored.row, ()),
                    cosine=cosine,
                    rows_by_doc=self._rows_by_doc,
                    alpha=_expansion_alpha(self.retriever),
                ),
                retriever=self.retriever.name,
                injection=self.scans[scored.row],
                via=lifted(scored.row, ()),
            )
            for position, scored in enumerate(selected, start=1)
        )

    def _named_documents(
        self,
        query: str,
        kinds: Collection[DocumentKind] | None,
        include_suspicious: bool,
    ) -> list[int]:
        """Rows of the documents ``query`` names by identifier, in the order named.

        One row per named document -- its first chunk, which is the description. A
        query naming a CVE wants that CVE, not all four of its chunks crowding out
        everything else.
        """
        rows: list[int] = []
        seen: set[str] = set()
        for match in IDENTIFIER_PATTERN.finditer(query):
            doc_id = _canonical_doc_id(match.group(0), self._rows_by_doc)
            if doc_id is None or doc_id in seen:
                continue
            seen.add(doc_id)
            for row in self._rows_by_doc.get(doc_id, ()):
                if self._admissible(row, kinds, include_suspicious):
                    rows.append(row)
                    break
        return rows

    def _admissible(
        self,
        row: int,
        kinds: Collection[DocumentKind] | None,
        include_suspicious: bool,
    ) -> bool:
        """Whether ``row`` may be returned at all.

        Two independent reasons to refuse, kept as separate guard clauses rather than
        one negated disjunction: a caller debugging why a chunk vanished needs to know
        *which* rule dropped it, and the two rules have very different consequences --
        one is a caller's filter, the other is a security control.
        """
        if kinds is not None and self.chunks[row].kind not in {k.value for k in kinds}:
            return False
        if not include_suspicious and self.scans[row].verdict is _EXCLUDE_AT:  # noqa: SIM103
            return False
        return True

    def evidence_for(
        self,
        query: str,
        *,
        k: int = 5,
        kinds: Collection[DocumentKind] | None = None,
        diversify: bool = True,
        retrieved_at: datetime | None = None,
    ) -> tuple[Evidence, ...]:
        """Search and return typed :class:`~sentinel.core.schemas.Evidence` directly.

        The convenience the Investigation Agent will actually call. It exists so the
        agent layer never constructs ``Evidence`` by hand — a hand-built citation is
        a citation whose ref nobody checked. Note there is no ``include_suspicious``
        parameter: the one caller that must never see a chunk which reads like an
        instruction is the one building a prompt.
        """
        hits = self.search(query, k=k, kinds=kinds, diversify=diversify)
        return tuple(hit.as_evidence(retrieved_at=retrieved_at) for hit in hits)

    # --- resolution ----------------------------------------------------------- #

    def resolve(self, ref: str) -> Chunk | None:
        """The chunk a citation ref names, or ``None`` if it resolves to nothing."""
        index = self._by_ref.get(ref)
        return None if index is None else self.chunks[index]

    def resolves_all(self, refs: Sequence[str]) -> bool:
        """True when every ref names real indexed content."""
        return all(ref in self._by_ref for ref in refs)

    def unresolved(self, refs: Sequence[str]) -> tuple[str, ...]:
        """The refs that do not resolve — for a targeted error message."""
        return tuple(ref for ref in refs if ref not in self._by_ref)

    def chunks_for(self, doc_id: str) -> tuple[Chunk, ...]:
        return tuple(chunk for chunk in self.chunks if chunk.doc_id == doc_id)

    # --- the relation graph, as an explicit second step ----------------------- #

    def related(self, doc_id: str) -> tuple[str, ...]:
        """Documents ``doc_id`` is cross-linked to: mapped techniques, parent, children.

        Separate from :meth:`search` on the strength of a measurement. Folding these
        links into the ranking was tried both in score space and as a fused rank list;
        see :data:`~sentinel.kb.index.DEFAULT_EXPANSION_ALPHA` for the numbers. The
        short version is that it buys recall and pays for it in ordering, which is a
        trade rather than a win, so relevance ranking stays relevance ranking and
        link-following becomes something a caller does on purpose.

        That split is also the better interface. The Investigation Agent's actual
        need is "I retrieved ``CVE-2017-5638``, now give me the technique it realises"
        — a lookup with a known answer, not a guess dressed as a search result.
        """
        return self._links.get(doc_id, ())

    def follow_links(
        self,
        hits: Sequence[RetrievalHit],
        *,
        per_hit: int = 2,
        retrieved_at: datetime | None = None,
    ) -> tuple[RetrievalHit, ...]:
        """Append the best chunk of each document the ``hits`` link to.

        Appended, never interleaved: the relevance-ranked results keep their order and
        their ranks, and enrichment lands after them carrying ``via`` so the analyst
        can see it arrived structurally. This is the whole reason the ranking damage
        was avoidable — nothing is displaced.

        ``relevance`` for an appended hit is discounted by
        :data:`~sentinel.kb.index.DEFAULT_EXPANSION_ALPHA`'s sibling constant
        :data:`LINK_RELEVANCE_DISCOUNT`, mirroring how ``graph/explain.py`` prices an
        inherited exposure path below a first-hand one.
        """
        if per_hit < 1:
            raise KnowledgeBaseError("per_hit must be at least 1")
        already = {hit.doc_id for hit in hits}
        appended: list[RetrievalHit] = []
        rank = len(hits)
        for hit in hits:
            taken = 0
            for doc_id in self.related(hit.doc_id):
                if taken >= per_hit:
                    break
                if doc_id in already:
                    continue
                rows = self._rows_by_doc.get(doc_id, ())
                if not rows:
                    continue
                already.add(doc_id)
                taken += 1
                rank += 1
                # The document's first chunk is its description, which is what a
                # "related technique" citation should quote; its detection guidance
                # answers a question nobody asked here.
                row = rows[0]
                appended.append(
                    RetrievalHit(
                        chunk=self.chunks[row],
                        rank=rank,
                        raw_score=hit.raw_score * LINK_RELEVANCE_DISCOUNT,
                        relevance=float(
                            min(1.0, hit.relevance * LINK_RELEVANCE_DISCOUNT)
                        ),
                        retriever=f"{self.retriever.name}+follow_links",
                        injection=self.scans[row],
                        via=(hit.doc_id,),
                    )
                )
        return (*hits, *appended)

    # --- audit ---------------------------------------------------------------- #

    @property
    def suspicious_chunks(self) -> tuple[tuple[Chunk, InjectionScan], ...]:
        """Chunks whose text looks like an attempt to instruct the model.

        Reported rather than silently dropped: a poisoned knowledge-base entry is an
        incident in its own right, and a build that quietly discards it leaves nobody
        to tell.
        """
        return tuple(
            (chunk, scan)
            for chunk, scan in zip(self.chunks, self.scans, strict=True)
            if scan.verdict is not InjectionVerdict.CLEAN
        )

    def stats(self) -> dict[str, object]:
        """Shape of the built index, for the evaluation report and the README."""
        assert self.lexical.encoder.counts is not None
        counts = self.lexical.encoder.counts
        by_kind: dict[str, int] = {}
        for chunk in self.chunks:
            by_kind[chunk.kind] = by_kind.get(chunk.kind, 0) + 1
        return {
            "documents": len(self.corpus) if self.corpus is not None else None,
            "chunks": len(self.chunks),
            "chunks_by_kind": by_kind,
            "features": counts.n_cols,
            "nonzeros": counts.nnz,
            "density": counts.nnz / (counts.n_rows * counts.n_cols),
            "mean_chunk_words": float(
                np.mean([chunk.word_count for chunk in self.chunks])
            ),
            "retriever": self.retriever.name,
            "suspicious_chunks": len(self.suspicious_chunks),
        }

    # --- persistence ---------------------------------------------------------- #

    def save(self, path: Path | str) -> Path:
        """Write the built index to a single ``.npz`` artifact.

        Only the *fitted* state is written: chunk metadata, term counts, both IDF
        vectors and the LSA factorisation. The TF-IDF matrix and both inverted
        indices are derived deterministically on load, because deriving them costs
        one pass over the non-zeros and storing them would let the stored copies
        drift out of agreement with the counts they came from.
        """
        assert self.lexical.encoder.counts is not None
        encoder = self.lexical.encoder
        counts = encoder.counts
        lsa = _lsa_encoder_of(self.retriever)

        meta = {
            "artifact_version": ARTIFACT_VERSION,
            "analyzer_version": ANALYZER_VERSION,
            "encoder_fingerprint": encoder.fingerprint,
            "n_features": encoder.n_features,
            "n_documents": encoder.n_documents,
            "k1": self.lexical.k1,
            "b": self.lexical.b,
            "weights": _weights_of(self.retriever),
            "rrf_k": getattr(self.retriever, "rrf_k", DEFAULT_RRF_K),
            "candidate_depth": getattr(self.retriever, "candidate_depth", 30),
            "lsa_components": 0 if lsa is None else lsa.n_components,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "kind": c.kind,
                    "title": c.title,
                    "section": c.section,
                    "ordinal": c.ordinal,
                    "excerpt": c.excerpt,
                    "embed_text": c.embed_text,
                }
                for c in self.chunks
            ],
        }
        arrays = {
            "counts_indptr": counts.indptr,
            "counts_indices": counts.indices,
            "counts_data": counts.data,
            "idf_tfidf": encoder.idf_tfidf,
            "idf_bm25": encoder.idf_bm25,
            "doc_len": encoder.doc_len,
            # Sorted so the artifact is byte-stable across builds, which a
            # fingerprint over a set would not otherwise be.
            "vocabulary": np.asarray(sorted(encoder.vocabulary), dtype=np.uint64),
        }
        if lsa is not None and lsa.components is not None and lsa.singular_values is not None:
            arrays["lsa_components"] = lsa.components
            arrays["lsa_singular"] = lsa.singular_values

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, meta=json.dumps(meta), **arrays)
        return target if target.suffix else target.with_suffix(".npz")

    @classmethod
    def load(cls, path: Path | str) -> KnowledgeBase:
        """Read an artifact written by :meth:`save`, refusing an incompatible one.

        Three checks, each protecting against a different silent failure:

        1.  ``artifact_version`` — the file layout changed.
        2.  ``analyzer_version`` — the tokenizer changed, so queries would be
            analyzed differently from the indexed text. This is the train/serve skew
            case, and it is invisible at runtime: the index still returns results,
            they are just quietly worse.
        3.  ``encoder_fingerprint`` — the IDF vectors do not hash to what was
            written, so the file is corrupt or was edited.
        """
        source = Path(path)
        if not source.exists() and source.with_suffix(".npz").exists():
            source = source.with_suffix(".npz")
        with np.load(source, allow_pickle=False) as handle:
            meta = json.loads(str(handle["meta"]))
            if meta.get("artifact_version") != ARTIFACT_VERSION:
                raise KnowledgeBaseError(
                    f"artifact version {meta.get('artifact_version')!r} != "
                    f"{ARTIFACT_VERSION!r}; rebuild the index"
                )
            if meta.get("analyzer_version") != ANALYZER_VERSION:
                raise KnowledgeBaseError(
                    f"artifact was built with analyzer {meta.get('analyzer_version')!r} "
                    f"but this build analyzes as {ANALYZER_VERSION!r}; the index and "
                    "the query path would disagree on tokenization. Rebuild."
                )
            arrays = {name: handle[name] for name in handle.files if name != "meta"}

        encoder = LexicalEncoder(n_features=int(meta["n_features"]))
        encoder.counts = SparseMatrix(
            indptr=arrays["counts_indptr"],
            indices=arrays["counts_indices"],
            data=arrays["counts_data"],
            n_cols=int(meta["n_features"]),
        )
        encoder.counts.validate()
        encoder.idf_tfidf = np.asarray(arrays["idf_tfidf"], dtype=DTYPE)
        encoder.idf_bm25 = np.asarray(arrays["idf_bm25"], dtype=DTYPE)
        encoder.doc_len = np.asarray(arrays["doc_len"], dtype=DTYPE)
        encoder.n_documents = int(meta["n_documents"])
        encoder.vocabulary = frozenset(int(v) for v in arrays["vocabulary"])
        if encoder.fingerprint != meta["encoder_fingerprint"]:
            raise KnowledgeBaseError(
                "encoder fingerprint mismatch: the artifact's IDF vectors do not hash "
                "to the value recorded when it was written"
            )

        chunks = tuple(
            Chunk(
                chunk_id=record["chunk_id"],
                doc_id=record["doc_id"],
                kind=record["kind"],
                title=record["title"],
                section=record["section"],
                ordinal=int(record["ordinal"]),
                excerpt=record["excerpt"],
                embed_text=record["embed_text"],
            )
            for record in meta["chunks"]
        )

        bm25 = LexicalIndex(
            encoder=encoder, scorer="bm25", k1=float(meta["k1"]), b=float(meta["b"])
        )
        tfidf = LexicalIndex(
            encoder=encoder, scorer="tfidf", k1=float(meta["k1"]), b=float(meta["b"])
        )
        weights: dict[str, float] = dict(meta["weights"])
        components: list[tuple[Retriever, float]] = [
            (bm25, weights.get("bm25", 0.0)),
            (tfidf, weights.get("tfidf", 0.0)),
        ]
        if "lsa_components" in arrays and "lsa_singular" in arrays:
            lsa_encoder = LsaEncoder(n_components=int(meta["lsa_components"]))
            lsa_encoder.components = np.asarray(arrays["lsa_components"], dtype=DTYPE)
            lsa_encoder.singular_values = np.asarray(arrays["lsa_singular"], dtype=DTYPE)
            lsa_encoder.doc_vectors = _unit_rows(
                lsa_encoder.components * lsa_encoder.singular_values
            )
            components.append(
                (LsaIndex(lexical=bm25, encoder=lsa_encoder), weights.get("lsa", 0.0))
            )

        retriever = HybridIndex(
            components=tuple(components),
            candidate_depth=int(meta["candidate_depth"]),
            rrf_k=int(meta["rrf_k"]),
        )
        return cls(
            chunks=chunks,
            retriever=retriever,
            lexical=bm25,
            scans=tuple(scan_for_injection(chunk.excerpt) for chunk in chunks),
            corpus=None,
        )


#: Score assigned to a document the query named by identifier but the ranking did
#: not surface at all. Above any reciprocal-rank-fusion score, which is bounded by
#: ``sum(weights) / (rrf_k + 1)`` and in practice sits near 0.03.
_TOP_SCORE: Final[float] = 1e6


def _canonical_doc_id(token: str, indexed: dict[str, tuple[int, ...]]) -> str | None:
    """Map an identifier as written in a query onto an indexed document id.

    Case is normalised because analysts type ``cve-2021-44228`` as often as the
    upper-case form, and the corpus stores exactly one spelling of each id.
    """
    if token in indexed:
        return token
    upper = token.upper()
    if upper in indexed:
        return upper
    return None


def _relation_graph(corpus: Corpus, indexed: set[str]) -> dict[str, tuple[str, ...]]:
    """Symmetric closure of the corpus's declared links, restricted to indexed docs.

    Both directions, because an investigation asks both "what technique is this CVE"
    and "which CVEs realise this technique". Restricted to indexed documents so a
    link can never name something :meth:`KnowledgeBase.resolve` cannot produce —
    the same dangling-reference rule the corpus and the report validator both apply.
    """
    edges: dict[str, set[str]] = {}

    def link(left: str, right: str) -> None:
        if left == right or left not in indexed or right not in indexed:
            return
        edges.setdefault(left, set()).add(right)
        edges.setdefault(right, set()).add(left)

    for document in corpus:
        for technique in document.techniques:
            link(document.doc_id, technique)
        if document.parent_id is not None:
            link(document.doc_id, document.parent_id)
    return {doc: tuple(sorted(linked)) for doc, linked in edges.items()}


def _no_provenance(_row: int, default: tuple[str, ...]) -> tuple[str, ...]:
    """Stand-in for a provenance lookup when the retriever cannot report one."""
    return default


def _rank_scores(scores: np.ndarray, k: int) -> tuple[Scored, ...]:
    """Top ``k`` positive scores, ties broken by row — the same rule as the indexes.

    Duplicated from :func:`sentinel.kb.index._top_k` rather than imported, because
    importing a private helper across modules is how a tie-breaking rule silently
    diverges later. ``test_kb_retrieve.py`` asserts the two agree.
    """
    positive = np.flatnonzero(scores > 0.0)
    if positive.shape[0] == 0:
        return ()
    if positive.shape[0] > k:
        window = np.argpartition(-scores[positive], k - 1)[:k]
        positive = positive[window]
    order = np.lexsort((positive, -scores[positive]))
    return tuple(Scored(score=float(scores[row]), row=int(row)) for row in positive[order])


def _expansion_alpha(retriever: Retriever) -> float:
    return retriever.alpha if isinstance(retriever, LinkExpansionIndex) else 0.0


def _relevance(
    *,
    own: float,
    via: tuple[str, ...],
    cosine: np.ndarray,
    rows_by_doc: dict[str, tuple[int, ...]],
    alpha: float,
) -> float:
    """Calibrated relevance in ``[0, 1]`` for one hit.

    For a chunk that matched the query text, this is its cosine against the query.
    For a chunk reached only through a cross-link, the cosine is near zero and would
    understate it: the chunk is in the list because a *strongly* matching document
    declares a link to it. So a structurally-reached chunk is priced at ``alpha``
    times the best cosine among the documents that lifted it — the same
    share-of-inherited-weight rule ``graph/explain.py`` uses for an exposure path,
    and for the same reason: inherited relevance is real but is worth less than
    first-hand relevance, and the discount should be visible rather than assumed.
    """
    inherited = 0.0
    for source in via:
        rows = rows_by_doc.get(source, ())
        if rows:
            inherited = max(inherited, alpha * float(max(cosine[row] for row in rows)))
    return float(min(1.0, max(0.0, max(own, inherited))))


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0.0)


def _weights_of(retriever: Retriever) -> dict[str, float]:
    if isinstance(retriever, HybridIndex):
        return {component.name: weight for component, weight in retriever.components}
    return {retriever.name: 1.0}


def _lsa_encoder_of(retriever: Retriever) -> LsaEncoder | None:
    if isinstance(retriever, HybridIndex):
        for component, _ in retriever.components:
            if isinstance(component, LsaIndex):
                return component.encoder
    if isinstance(retriever, LsaIndex):
        return retriever.encoder
    return None


def _score_ceiling(retriever: Retriever, *, fallback: float) -> float:
    """The largest score this retriever can assign, for calibrating relevance.

    For reciprocal-rank fusion the ceiling is analytic — every component ranking the
    chunk first gives ``sum(weights) / (rrf_k + 1)`` — so ``relevance`` means
    "how close to unanimous first place", on a scale that is the same for every
    query. That is a genuine calibration, not a per-query rescaling.

    For any other retriever there is no such constant, so the top score in the
    window is used and the leading hit therefore always reports ``1.0``. That is a
    *within-query ordering*, not a confidence, and it is the reason
    :attr:`RetrievalHit.raw_score` is kept alongside it: the honest number is the one
    the retriever actually produced.
    """
    if isinstance(retriever, HybridIndex):
        total = sum(weight for _, weight in retriever.components if weight > 0.0)
        return total / (retriever.rrf_k + 1)
    return fallback if fallback > 0.0 else 1.0
