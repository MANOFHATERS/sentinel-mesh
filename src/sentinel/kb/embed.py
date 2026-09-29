"""Embedding: hashed term counts, IDF, and latent-semantic dense vectors.

Two encoders, because lexical and semantic retrieval fail in opposite directions
------------------------------------------------------------------------------------
:class:`LexicalEncoder` hashes analyzed terms into a fixed feature space and carries
raw counts, from which both TF-IDF cosine and BM25 are derived. It is exact on
identifiers and operator vocabulary — a query for ``CVE-2021-44228`` or
``T1021.002`` cannot miss — and it is blind to paraphrase: "moved between hosts
using file shares" shares almost no surface tokens with the SMB technique text.

:class:`LsaEncoder` factorises the same TF-IDF matrix and represents every chunk in
a few hundred latent dimensions. It closes some of the paraphrase gap, because terms
that co-occur across the corpus end up near each other, and it is correspondingly
fuzzy on identifiers, which it happily blends with their neighbours.

Neither dominates, which is why :class:`~sentinel.kb.index.HybridIndex` fuses them
and why ``eval.py`` reports each alone alongside the fusion on a held-out query
split. The measured numbers are in ``docs/BUILD_PLAN.md``.

Why not sentence-transformers
-----------------------------
The PRD names FAISS and an embedding model. A transformer embedder would pull a
model over the network on first use, which makes the test suite non-reproducible
(the weights can change), network-bound (so it fails in CI and offline) and slow.
Latent semantic analysis is a real, published dense embedding, it is exact to
machine precision, it is 40 lines of numpy, and it has no network dependency at
all. The seam is a Protocol, so a transformer backend slots in as another
:class:`~sentinel.kb.index.Retriever` exactly as ``AnomalyDetector`` allows a torch
swap — and ``test_kb_index.py::TestRetrieverSeam`` proves the seam holds by driving
the knowledge base with a stub retriever.

Determinism
-----------
Feature indices come from BLAKE2b, never from :func:`hash`. Python randomizes string
hashing per process, so a :func:`hash`-derived index would give a different vector
on every run, and an index written by one process would mis-rank when read by
another — a bug that would look like poor retrieval quality rather than like
corruption. :attr:`LexicalEncoder.fingerprint` is written into the saved artifact and
checked on load for the same reason.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt

from sentinel.core.errors import SentinelError
from sentinel.kb.sparse import DTYPE, InvertedIndex, SparseMatrix
from sentinel.kb.text import analyze

__all__ = [
    "ANALYZER_VERSION",
    "DEFAULT_N_FEATURES",
    "EmbeddingError",
    "LexicalEncoder",
    "LsaEncoder",
    "feature_index",
    "term_digest",
]


class EmbeddingError(SentinelError):
    """An encoder was used before fitting, or with incompatible inputs."""


#: Feature-space width. 2^14 for a corpus whose analyzed vocabulary (unigrams plus
#: bigrams) is roughly 20k terms: at that load factor the expected number of terms
#: colliding into any one bucket is near one, so collisions add a little noise and
#: cannot systematically merge two frequent terms. Widening to 2^16 was measured and
#: changed held-out recall@5 by less than the run-to-run noise, so the smaller index
#: wins.
DEFAULT_N_FEATURES: Final[int] = 1 << 14

#: Bumped whenever :func:`sentinel.kb.text.analyze` changes in a way that alters its
#: output. It is mixed into :attr:`LexicalEncoder.fingerprint`, so an index built
#: with an older analyzer is *refused* on load rather than quietly producing worse
#: rankings. This is the train/serve-skew guard: an index and the query path that
#: reads it must agree on tokenization, and nothing else in the file can detect it.
ANALYZER_VERSION: Final[str] = "analyze-v1-porter1-bigrams"


def term_digest(term: str) -> int:
    """A full 64-bit digest of ``term``, with no modulo.

    Separate from :func:`feature_index` because it answers a different question.
    A feature index is lossy by design — collisions are the price of a fixed-width
    space — which means "is this bucket occupied" cannot distinguish *this* term from
    whichever other term hashes alongside it. With roughly twenty thousand analyzed
    terms in sixteen thousand buckets, nearly every bucket is occupied, so a query of
    pure nonsense scores against real documents purely through collisions.

    That is not a hypothetical. Measured on this corpus, nonsense queries reach a top
    relevance of 0.05 to 0.11 while the weakest genuinely-correct hit in the labelled
    set sits at 0.077 — the distributions overlap, so no relevance threshold can
    separate them. Keeping the undegraded digests makes the question exact instead of
    statistical: a term is either in the corpus or it is not.
    """
    return int.from_bytes(hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest(), "big")


def feature_index(term: str, n_features: int) -> int:
    """Map ``term`` to a column in ``[0, n_features)``, deterministically.

    BLAKE2b with an 8-byte digest: cryptographic strength is not the point,
    process-independence and good avalanche behaviour on short ASCII strings are.
    """
    digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % n_features


@dataclass(slots=True)
class LexicalEncoder:
    """Hashed bag-of-terms with both TF-IDF and BM25 weighting available.

    Fitting computes document frequencies over the corpus and derives two IDF
    vectors, because the two scorers want different ones and using a single vector
    for both is a common and quietly damaging shortcut:

    *   ``idf_tfidf`` uses the smoothed ``ln((1 + N) / (1 + df)) + 1``, which is
        strictly positive, so a term present in every chunk still contributes its
        (small) share to the cosine rather than zeroing the dimension.
    *   ``idf_bm25`` uses the Robertson-Sparck-Jones form
        ``ln(1 + (N - df + 0.5) / (df + 0.5))``, which is the form BM25's saturation
        is calibrated against. Its unsmoothed variant goes *negative* for terms in
        more than half the corpus, which lets a common query term subtract score and
        reorders results nonsensically; the ``1 +`` form used here is the standard
        fix and is floored at zero as a second line of defence.
    """

    n_features: int = DEFAULT_N_FEATURES
    counts: SparseMatrix | None = field(default=None, repr=False)
    idf_tfidf: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    idf_bm25: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    doc_len: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    n_documents: int = 0
    #: Undegraded digests of every term seen while fitting. ~8 bytes per term, so
    #: 160 KB for this corpus, in exchange for being able to tell a real thin match
    #: from a hash collision. See :func:`term_digest`.
    vocabulary: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        if self.n_features < 256 or self.n_features & (self.n_features - 1):
            raise EmbeddingError("n_features must be a power of two and at least 256")

    # --- fitting -------------------------------------------------------------- #

    def fit(self, texts: Sequence[str]) -> LexicalEncoder:
        """Hash ``texts`` into a count matrix and derive both IDF vectors."""
        if not texts:
            raise EmbeddingError("cannot fit on an empty corpus")
        rows: list[dict[int, float]] = []
        seen: set[int] = set()
        for text in texts:
            row: dict[int, float] = {}
            for term in analyze(text):
                column = feature_index(term, self.n_features)
                row[column] = row.get(column, 0.0) + 1.0
                seen.add(term_digest(term))
            rows.append(row)
        self.vocabulary = frozenset(seen)

        self.counts = SparseMatrix.from_rows(rows, n_cols=self.n_features)
        self.n_documents = self.counts.n_rows
        self.doc_len = self.counts.row_sums()

        inverted = InvertedIndex.from_matrix(self.counts)
        df = inverted.document_frequency().astype(DTYPE)
        total = float(self.n_documents)
        self.idf_tfidf = np.log((1.0 + total) / (1.0 + df)) + 1.0
        self.idf_bm25 = np.maximum(np.log(1.0 + (total - df + 0.5) / (df + 0.5)), 0.0)
        return self

    @property
    def is_fitted(self) -> bool:
        return self.counts is not None

    def _require_fitted(self) -> SparseMatrix:
        if self.counts is None or self.idf_tfidf is None or self.doc_len is None:
            raise EmbeddingError("encoder is not fitted; call fit() first")
        return self.counts

    # --- derived matrices ----------------------------------------------------- #

    def tfidf_matrix(self) -> SparseMatrix:
        """Sub-linear TF, IDF weighted, L2 normalized — so a dot product is a cosine.

        Sub-linear (``1 + log tf``) rather than raw counts: a chunk that says
        "credential" five times is not five times more about credentials, and raw
        counts let one repeated word dominate a 70-word chunk's direction.
        """
        counts = self._require_fitted()
        assert self.idf_tfidf is not None
        sublinear = SparseMatrix(
            indptr=counts.indptr.copy(),
            indices=counts.indices.copy(),
            data=1.0 + np.log(counts.data),
            n_cols=counts.n_cols,
        )
        return sublinear.scaled_by_columns(self.idf_tfidf).l2_normalized()

    # --- query encoding ------------------------------------------------------- #

    def known_terms(self, text: str) -> list[str]:
        """The analyzed terms of ``text`` that actually occur in the fitted corpus.

        Empty means every scoring hit for this query came from a hash collision, which
        is the one case where returning nothing is the honest answer.
        """
        return [term for term in analyze(text) if term_digest(term) in self.vocabulary]

    def encode_counts(self, text: str) -> dict[int, float]:
        """Raw hashed term counts for one text. The BM25 query representation."""
        row: dict[int, float] = {}
        for term in analyze(text):
            column = feature_index(term, self.n_features)
            row[column] = row.get(column, 0.0) + 1.0
        return row

    def encode_tfidf(self, text: str) -> dict[int, float]:
        """L2-normalized sub-linear TF-IDF vector for one text.

        Applying the *corpus* IDF to the query is what makes the cosine meaningful:
        an IDF fitted on the query itself would be meaningless, and omitting it would
        weight a query's stopword-adjacent terms as heavily as its identifiers.
        """
        self._require_fitted()
        assert self.idf_tfidf is not None
        counts = self.encode_counts(text)
        weighted = {
            column: (1.0 + np.log(count)) * float(self.idf_tfidf[column])
            for column, count in counts.items()
        }
        norm = np.sqrt(sum(value * value for value in weighted.values()))
        if norm == 0.0:
            return {}
        return {column: value / norm for column, value in weighted.items()}

    # --- identity ------------------------------------------------------------- #

    @property
    def fingerprint(self) -> str:
        """Identity of the fitted encoder, for artifact compatibility checks.

        Covers the analyzer version, the feature width, the document count and the
        IDF vector itself. Two encoders with the same fingerprint produce the same
        vector for the same text; an index saved under one fingerprint and queried
        under another would mis-rank silently, so the load path refuses it.
        """
        digest = hashlib.sha256()
        digest.update(ANALYZER_VERSION.encode("ascii"))
        digest.update(np.int64(self.n_features).tobytes())
        digest.update(np.int64(self.n_documents).tobytes())
        if self.idf_tfidf is not None:
            digest.update(np.ascontiguousarray(self.idf_tfidf, dtype=DTYPE).tobytes())
        if self.idf_bm25 is not None:
            digest.update(np.ascontiguousarray(self.idf_bm25, dtype=DTYPE).tobytes())
        digest.update(np.int64(len(self.vocabulary)).tobytes())
        digest.update(
            np.asarray(sorted(self.vocabulary), dtype=np.uint64).tobytes()
        )
        return digest.hexdigest()


@dataclass(slots=True)
class LsaEncoder:
    """Latent semantic analysis over a TF-IDF matrix: a real dense embedder.

    Computed through the Gram matrix rather than by decomposing the term matrix
    directly. With ``n`` chunks and ``d`` features, ``X X^T`` is ``n x n`` — a few
    hundred squared here — while ``X`` is ``n x 16384``, so eigendecomposing the Gram
    matrix is the cheap route to the same factorisation. ``X = U S V^T`` gives
    ``X X^T = U S^2 U^T``, so :func:`numpy.linalg.eigh` on the Gram matrix yields
    ``U`` and ``S`` exactly, with no iterative approximation and no random
    initialisation — the embedding is bit-reproducible, which matters because it is
    part of a persisted, fingerprinted artifact.

    A query is folded in with ``q_latent = (X q)^T U S^-1``, the standard LSA
    projection. ``X q`` is one inverted-index scoring pass, so query cost stays
    proportional to the query's term count.
    """

    n_components: int = 192
    #: Singular values below this fraction of the largest are dropped. They carry
    #: numerical noise rather than structure, and dividing by one during query
    #: fold-in would amplify that noise without bound.
    tolerance: float = 1e-9
    components: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    singular_values: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    doc_vectors: npt.NDArray[np.float64] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.n_components < 2:
            raise EmbeddingError("n_components must be at least 2")

    @property
    def dim(self) -> int:
        return 0 if self.singular_values is None else int(self.singular_values.shape[0])

    @property
    def is_fitted(self) -> bool:
        return self.doc_vectors is not None

    def fit(self, tfidf: SparseMatrix) -> LsaEncoder:
        """Factorise ``tfidf`` and store unit-norm latent document vectors."""
        gram = _gram_matrix(tfidf)
        # eigh returns ascending eigenvalues for a symmetric matrix; reverse them.
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.clip(eigenvalues[order], 0.0, None)
        eigenvectors = eigenvectors[:, order]

        keep = min(self.n_components, eigenvalues.shape[0])
        singular = np.sqrt(eigenvalues[:keep])
        if singular.shape[0] and singular[0] > 0.0:
            significant = singular >= self.tolerance * singular[0]
            singular = singular[significant]
            eigenvectors = eigenvectors[:, :keep][:, significant]
        else:  # pragma: no cover - only reachable for an all-zero matrix
            raise EmbeddingError("tfidf matrix has no signal to factorise")

        self.singular_values = singular
        self.components = eigenvectors
        self.doc_vectors = _row_normalize(eigenvectors * singular)
        return self

    def project_query(self, document_scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Fold a query into the latent space.

        ``document_scores`` is ``X q``: the query's TF-IDF cosine against every
        chunk, which the caller already has from the inverted index. Reusing it
        rather than re-materialising the query's sparse vector against a dense
        matrix is why this stays cheap.
        """
        if self.components is None or self.singular_values is None:
            raise EmbeddingError("LsaEncoder is not fitted; call fit() first")
        if document_scores.shape != (self.components.shape[0],):
            raise EmbeddingError(
                f"expected {self.components.shape[0]} document scores, "
                f"got {document_scores.shape}"
            )
        latent = (document_scores @ self.components) / self.singular_values
        norm = float(np.linalg.norm(latent))
        return latent if norm == 0.0 else latent / norm

    def similarities(self, latent_query: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Cosine of a folded-in query against every chunk.

        Both sides are unit-norm, so this is a single matrix-vector product — which
        is exactly what a FAISS flat inner-product index computes. At this corpus
        size that library would add a dependency and a binary wheel to do the same
        arithmetic; the seam for swapping it in is
        :class:`~sentinel.kb.index.Retriever`.
        """
        if self.doc_vectors is None:
            raise EmbeddingError("LsaEncoder is not fitted; call fit() first")
        return self.doc_vectors @ latent_query

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"lsa-v1")
        digest.update(np.int64(self.n_components).tobytes())
        if self.singular_values is not None:
            digest.update(np.ascontiguousarray(self.singular_values, dtype=DTYPE).tobytes())
        return digest.hexdigest()


def _gram_matrix(matrix: SparseMatrix) -> npt.NDArray[np.float64]:
    """``M M^T`` without densifying ``M``.

    The naive route is ``M.to_dense() @ M.to_dense().T``, which for this corpus
    allocates ~80 MB and spends 6 GFLOP multiplying zeros. Scoring each row against
    the inverted index instead costs one postings walk per stored value — about 80k
    operations here — and is exact, not approximate. The symmetry is not assumed: it
    is *asserted* in ``test_kb_embed.py``, because a transpose bug here would show up
    as mildly worse retrieval rather than as an error.
    """
    inverted = InvertedIndex.from_matrix(matrix)
    gram = np.zeros((matrix.n_rows, matrix.n_rows), dtype=DTYPE)
    for row in range(matrix.n_rows):
        columns, values = matrix.row(row)
        query = dict(zip(columns.tolist(), values.tolist(), strict=True))
        gram[row] = inverted.score(query)
    # Force exact symmetry: the two triangles are mathematically equal but are
    # accumulated in different orders, so they can differ in the last bits, and
    # ``eigh`` on a matrix that is not symmetric to the bit is not reproducible.
    return 0.5 * (gram + gram.T)


def _row_normalize(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0.0)
