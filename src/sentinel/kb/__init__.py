"""Layer 3 — the retrieval-augmented knowledge base (PRD F-05, Section 5.5.6).

What grounds the Investigation Agent. F-05's acceptance criterion is *"every factual
claim traces to a retrieved KB chunk or raw log line"*, which needs two halves:
a report that cannot cite nothing (Part 1's
:class:`~sentinel.core.schemas.InvestigationReport` validator) and a knowledge base
whose citations resolve to real content (this package, via
:meth:`~sentinel.kb.retrieve.KnowledgeBase.resolve`).

    ``corpus``    ATT&CK techniques, CVEs, supply-chain advisories, playbooks
    ``chunk``     documents -> stable, citable, sentence-aligned chunks
    ``text``      tokenization that survives ``CVE-2021-44228`` and ``T1021.002``
    ``embed``     hashed TF-IDF and latent-semantic dense vectors
    ``sparse``    CSR term matrix and the inverted index built from it
    ``index``     BM25, TF-IDF cosine, LSA, rank fusion, MMR diversification
    ``retrieve``  the ``KnowledgeBase`` facade: search, resolve, persist
    ``eval``      95 labelled queries, ranking metrics, held-out acceptance gates

No FAISS, no sentence-transformers, no network. See :mod:`sentinel.kb.embed` for why
the PRD's named stack is a swap behind :class:`~sentinel.kb.index.Retriever` rather
than a dependency.
"""

from __future__ import annotations

from sentinel.kb.chunk import Chunk, chunk_corpus, chunk_document
from sentinel.kb.corpus import Corpus, CorpusError, DocumentKind, KBDocument, load_default_corpus
from sentinel.kb.eval import (
    EvalQuery,
    RetrievalReport,
    assert_gates,
    evaluate_retrieval,
    load_eval_queries,
)
from sentinel.kb.index import HybridIndex, LexicalIndex, LsaIndex, Retriever, Scored
from sentinel.kb.retrieve import KnowledgeBase, KnowledgeBaseError, RetrievalHit

__all__ = [
    "Chunk",
    "Corpus",
    "CorpusError",
    "DocumentKind",
    "EvalQuery",
    "HybridIndex",
    "KBDocument",
    "KnowledgeBase",
    "KnowledgeBaseError",
    "LexicalIndex",
    "LsaIndex",
    "RetrievalHit",
    "RetrievalReport",
    "Retriever",
    "Scored",
    "assert_gates",
    "chunk_corpus",
    "chunk_document",
    "evaluate_retrieval",
    "load_default_corpus",
    "load_eval_queries",
]
