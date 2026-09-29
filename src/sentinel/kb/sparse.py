"""A compressed-row term matrix and the inverted index built from it.

Why hand-rolled, and why sparse
-------------------------------
The obvious implementation of "exact cosine search" is a dense matrix and one
``matmul``. At this corpus's feature width that is ``16_384 x ~600`` in float64,
about 79 MB resident and 79 MB on disk for an index whose actual content is roughly
55 non-zeros per chunk — three orders of magnitude of padding. The sparse form is
under a megabyte, loads instantly, and scores *faster*, because query-time work
becomes proportional to the number of terms in the query rather than to the width of
the feature space.

``scipy.sparse`` would do this and is already installed transitively via
scikit-learn. It is not used, for the same reason :mod:`sentinel.ml.nn` does not use
torch: the mechanism being demonstrated is the mechanism, and a CSR layout plus a
term-at-a-time accumulator is about sixty lines of numpy that can be tested against
a dense reference. :class:`TestAgainstDenseReference` in ``test_kb_sparse.py`` does
exactly that — every score is asserted equal to the dense computation it replaces,
so the optimisation cannot silently be wrong.

The two structures are duals and both are kept
----------------------------------------------
:class:`SparseMatrix` is row-major (chunk -> its terms). It answers "what is in this
chunk", which is what chunk-to-chunk similarity needs, so it is what
:func:`~sentinel.kb.index.mmr_select` uses for diversification.

:class:`InvertedIndex` is column-major (term -> the chunks containing it). It answers
"which chunks contain this term", which is what scoring a query needs. Building it
from the row-major form is one counting sort, and keeping both makes each operation
linear in what it actually touches instead of scanning the whole matrix.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

__all__ = ["InvertedIndex", "SparseMatrix"]

DTYPE: Final = np.float64
INDEX_DTYPE: Final = np.int32


@dataclass(frozen=True, slots=True)
class SparseMatrix:
    """Compressed sparse row matrix: ``indptr``, ``indices``, ``data``.

    Row ``i`` occupies ``indices[indptr[i]:indptr[i + 1]]`` with the matching slice
    of ``data``. Column indices within a row are sorted and unique, which
    :meth:`validate` enforces, because both the dot product and the inverted-index
    build below assume it.
    """

    indptr: npt.NDArray[np.int32]
    indices: npt.NDArray[np.int32]
    data: npt.NDArray[np.float64]
    n_cols: int

    @property
    def n_rows(self) -> int:
        return int(self.indptr.shape[0]) - 1

    @property
    def nnz(self) -> int:
        return int(self.data.shape[0])

    def validate(self) -> None:
        """Check the structural invariants. Cheap, and run on every load."""
        if self.indptr.ndim != 1 or self.indptr.shape[0] < 1:
            raise ValueError("indptr must be a 1-D array of length n_rows + 1")
        if self.indices.shape != self.data.shape:
            raise ValueError("indices and data must have the same shape")
        if int(self.indptr[0]) != 0:
            raise ValueError("indptr must start at 0")
        if int(self.indptr[-1]) != self.nnz:
            raise ValueError(f"indptr ends at {int(self.indptr[-1])}, nnz is {self.nnz}")
        if np.any(np.diff(self.indptr) < 0):
            raise ValueError("indptr must be non-decreasing")
        if self.nnz and (self.indices.min() < 0 or self.indices.max() >= self.n_cols):
            raise ValueError("column index outside [0, n_cols)")
        for row in range(self.n_rows):
            start, stop = int(self.indptr[row]), int(self.indptr[row + 1])
            cols = self.indices[start:stop]
            if cols.shape[0] > 1 and np.any(np.diff(cols) <= 0):
                raise ValueError(f"row {row} column indices are not sorted and unique")

    @classmethod
    def from_rows(
        cls, rows: Sequence[dict[int, float]], *, n_cols: int
    ) -> SparseMatrix:
        """Build from per-row ``{column: value}`` mappings, dropping exact zeros.

        Zeros are dropped because a stored zero costs space and changes nothing:
        every consumer here treats absent and zero identically, and keeping them
        would make ``nnz`` a function of tokenization accidents.
        """
        indptr = np.zeros(len(rows) + 1, dtype=INDEX_DTYPE)
        indices: list[int] = []
        data: list[float] = []
        for position, row in enumerate(rows):
            for column in sorted(row):
                value = row[column]
                if value == 0.0:
                    continue
                if not 0 <= column < n_cols:
                    raise ValueError(f"column {column} outside [0, {n_cols})")
                indices.append(column)
                data.append(value)
            indptr[position + 1] = len(indices)
        matrix = cls(
            indptr=indptr,
            indices=np.asarray(indices, dtype=INDEX_DTYPE),
            data=np.asarray(data, dtype=DTYPE),
            n_cols=int(n_cols),
        )
        matrix.validate()
        return matrix

    def row(self, index: int) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]:
        """``(column_indices, values)`` for row ``index``."""
        if not 0 <= index < self.n_rows:
            raise IndexError(f"row {index} outside [0, {self.n_rows})")
        start, stop = int(self.indptr[index]), int(self.indptr[index + 1])
        return self.indices[start:stop], self.data[start:stop]

    def row_norms(self) -> npt.NDArray[np.float64]:
        """L2 norm of every row."""
        squared = np.zeros(self.n_rows, dtype=DTYPE)
        for index in range(self.n_rows):
            _, values = self.row(index)
            squared[index] = float(values @ values)
        return np.sqrt(squared)

    def row_sums(self) -> npt.NDArray[np.float64]:
        """Sum of every row — the document length when ``data`` holds term counts."""
        totals = np.zeros(self.n_rows, dtype=DTYPE)
        for index in range(self.n_rows):
            _, values = self.row(index)
            totals[index] = float(values.sum())
        return totals

    def scaled_by_columns(self, factors: npt.NDArray[np.float64]) -> SparseMatrix:
        """Multiply every stored value by ``factors[column]`` (e.g. apply IDF)."""
        if factors.shape != (self.n_cols,):
            raise ValueError(f"expected {self.n_cols} factors, got {factors.shape}")
        return SparseMatrix(
            indptr=self.indptr.copy(),
            indices=self.indices.copy(),
            data=self.data * factors[self.indices],
            n_cols=self.n_cols,
        )

    def l2_normalized(self) -> SparseMatrix:
        """Rows scaled to unit L2 norm, so a dot product is a cosine.

        An all-zero row stays all-zero rather than producing NaN. That happens for a
        chunk whose every term is corpus-universal (IDF zero), which is rare but real
        — and a NaN row would poison every subsequent ranking, so it is handled here
        rather than discovered downstream.
        """
        norms = self.row_norms()
        scale = np.ones_like(norms)
        nonzero = norms > 0.0
        scale[nonzero] = 1.0 / norms[nonzero]
        row_of_value = np.repeat(np.arange(self.n_rows), np.diff(self.indptr))
        return SparseMatrix(
            indptr=self.indptr.copy(),
            indices=self.indices.copy(),
            data=self.data * scale[row_of_value],
            n_cols=self.n_cols,
        )

    def dot_vector(self, vector: dict[int, float]) -> npt.NDArray[np.float64]:
        """``M @ v`` for a sparse ``v``, by rows. Reference path for tests."""
        scores = np.zeros(self.n_rows, dtype=DTYPE)
        for index in range(self.n_rows):
            columns, values = self.row(index)
            total = 0.0
            for column, value in zip(columns.tolist(), values.tolist(), strict=True):
                weight = vector.get(column)
                if weight is not None:
                    total += value * weight
            scores[index] = total
        return scores

    def to_dense(self) -> npt.NDArray[np.float64]:
        """Densify. Tests and diagnostics only — this is the thing being avoided."""
        dense = np.zeros((self.n_rows, self.n_cols), dtype=DTYPE)
        for index in range(self.n_rows):
            columns, values = self.row(index)
            dense[index, columns] = values
        return dense


@dataclass(frozen=True, slots=True)
class InvertedIndex:
    """Column-major view of a :class:`SparseMatrix`: term -> (rows, values).

    ``col_ptr`` has length ``n_cols + 1``; postings for column ``c`` are
    ``row_ids[col_ptr[c]:col_ptr[c + 1]]`` with the matching ``values``.
    """

    col_ptr: npt.NDArray[np.int32]
    row_ids: npt.NDArray[np.int32]
    values: npt.NDArray[np.float64]
    n_rows: int

    @property
    def n_cols(self) -> int:
        return int(self.col_ptr.shape[0]) - 1

    @classmethod
    def from_matrix(cls, matrix: SparseMatrix) -> InvertedIndex:
        """Transpose by counting sort — one pass to count, one to scatter."""
        counts = np.bincount(matrix.indices, minlength=matrix.n_cols).astype(INDEX_DTYPE)
        col_ptr = np.zeros(matrix.n_cols + 1, dtype=INDEX_DTYPE)
        np.cumsum(counts, out=col_ptr[1:])
        cursor = col_ptr[:-1].copy()
        row_ids = np.zeros(matrix.nnz, dtype=INDEX_DTYPE)
        values = np.zeros(matrix.nnz, dtype=DTYPE)
        row_of_value = np.repeat(np.arange(matrix.n_rows), np.diff(matrix.indptr))
        for position in range(matrix.nnz):
            column = int(matrix.indices[position])
            target = int(cursor[column])
            row_ids[target] = row_of_value[position]
            values[target] = matrix.data[position]
            cursor[column] = target + 1
        return cls(col_ptr=col_ptr, row_ids=row_ids, values=values, n_rows=matrix.n_rows)

    def postings(
        self, column: int
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]:
        """``(row_ids, values)`` for one term. Empty arrays for an absent term."""
        if not 0 <= column < self.n_cols:
            return (
                np.zeros(0, dtype=INDEX_DTYPE),
                np.zeros(0, dtype=DTYPE),
            )
        start, stop = int(self.col_ptr[column]), int(self.col_ptr[column + 1])
        return self.row_ids[start:stop], self.values[start:stop]

    def document_frequency(self) -> npt.NDArray[np.int32]:
        """How many rows contain each column."""
        return np.diff(self.col_ptr).astype(INDEX_DTYPE)

    def score(self, query: dict[int, float]) -> npt.NDArray[np.float64]:
        """Accumulate ``sum_t query[t] * stored[row, t]`` over the query's terms only.

        This is the whole reason the inverted index exists: cost is proportional to
        the postings of the handful of terms in the query, not to the corpus.
        """
        scores = np.zeros(self.n_rows, dtype=DTYPE)
        for column, weight in query.items():
            if weight == 0.0:
                continue
            rows, values = self.postings(column)
            if rows.shape[0]:
                np.add.at(scores, rows, weight * values)
        return scores

    def bm25(
        self,
        query: dict[int, float],
        *,
        idf: npt.NDArray[np.float64],
        doc_len: npt.NDArray[np.float64],
        k1: float,
        b: float,
    ) -> npt.NDArray[np.float64]:
        """Okapi BM25, with ``values`` interpreted as raw term counts.

        ``query`` maps a term column to its count in the query; a term repeated in
        the query contributes proportionally, which matters for the long
        natural-language alert descriptions this index is queried with.

        BM25 is here rather than assumed away because the two scorers fail
        differently. TF-IDF cosine normalises by document length exactly, which
        over-rewards very short chunks; BM25 normalises softly through ``b`` and
        saturates term frequency through ``k1``, so it is less distracted by a long
        chunk that repeats a query term. :mod:`sentinel.kb.eval` measures both on a
        held-out query split rather than picking on reputation.
        """
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b must be in [0, 1], got {b}")
        if k1 <= 0.0:
            raise ValueError(f"k1 must be positive, got {k1}")
        if idf.shape != (self.n_cols,):
            raise ValueError(f"expected {self.n_cols} idf values, got {idf.shape}")
        if doc_len.shape != (self.n_rows,):
            raise ValueError(f"expected {self.n_rows} lengths, got {doc_len.shape}")

        average = float(doc_len.mean()) if self.n_rows else 0.0
        if average <= 0.0:
            return np.zeros(self.n_rows, dtype=DTYPE)
        # Precomputed per-row denominator component; independent of the query term.
        length_term = k1 * (1.0 - b + b * (doc_len / average))

        scores = np.zeros(self.n_rows, dtype=DTYPE)
        for column, query_count in query.items():
            if query_count == 0.0:
                continue
            rows, counts = self.postings(column)
            if not rows.shape[0]:
                continue
            numerator = counts * (k1 + 1.0)
            denominator = counts + length_term[rows]
            np.add.at(scores, rows, query_count * idf[column] * (numerator / denominator))
        return scores
