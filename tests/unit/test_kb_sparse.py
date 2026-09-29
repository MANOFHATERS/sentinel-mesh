"""The CSR term matrix and inverted index (:mod:`sentinel.kb.sparse`).

This module is a hand-rolled optimisation replacing a dense matrix multiply, and an
optimisation that is quietly wrong is worse than no optimisation. So the central
class here is :class:`TestAgainstDenseReference`: every score the sparse path
produces is asserted equal to the dense computation it stands in for, and BM25 is
asserted against the formula written out by hand rather than against itself.

:class:`TestInvertedIndexIsATranspose` exists because the counting sort in
``InvertedIndex.from_matrix`` is the kind of code that is off by one in a way no
integration test notices — a transposed index with a shifted pointer array still
returns plausible rankings.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from sentinel.kb.sparse import InvertedIndex, SparseMatrix

ROWS: list[dict[int, float]] = [
    {0: 2.0, 3: 1.0, 7: 4.0},
    {1: 1.0, 3: 3.0},
    {},
    {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0},
    {7: 5.0},
]
N_COLS = 8


@pytest.fixture
def matrix() -> SparseMatrix:
    return SparseMatrix.from_rows(ROWS, n_cols=N_COLS)


@pytest.fixture
def dense() -> np.ndarray:
    out = np.zeros((len(ROWS), N_COLS))
    for index, row in enumerate(ROWS):
        for column, value in row.items():
            out[index, column] = value
    return out


class TestConstruction:
    def test_shape_and_nnz(self, matrix: SparseMatrix) -> None:
        assert matrix.n_rows == len(ROWS)
        assert matrix.n_cols == N_COLS
        assert matrix.nnz == sum(len(row) for row in ROWS)

    def test_to_dense_matches_the_source_rows(
        self, matrix: SparseMatrix, dense: np.ndarray
    ) -> None:
        np.testing.assert_array_equal(matrix.to_dense(), dense)

    def test_empty_row_is_representable(self, matrix: SparseMatrix) -> None:
        columns, values = matrix.row(2)
        assert columns.shape == (0,)
        assert values.shape == (0,)

    def test_exact_zeros_are_not_stored(self) -> None:
        stored = SparseMatrix.from_rows([{0: 0.0, 1: 5.0}], n_cols=2)
        assert stored.nnz == 1

    def test_column_indices_are_sorted_within_each_row(self) -> None:
        built = SparseMatrix.from_rows([{5: 1.0, 1: 1.0, 3: 1.0}], n_cols=8)
        columns, _ = built.row(0)
        assert list(columns) == [1, 3, 5]

    def test_out_of_range_column_rejected(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            SparseMatrix.from_rows([{99: 1.0}], n_cols=8)

    def test_row_index_out_of_range_raises(self, matrix: SparseMatrix) -> None:
        with pytest.raises(IndexError):
            matrix.row(matrix.n_rows)
        with pytest.raises(IndexError):
            matrix.row(-1)


class TestValidate:
    def test_valid_matrix_passes(self, matrix: SparseMatrix) -> None:
        matrix.validate()

    def test_indptr_not_starting_at_zero_rejected(self, matrix: SparseMatrix) -> None:
        broken = SparseMatrix(
            indptr=matrix.indptr + 1,
            indices=matrix.indices,
            data=matrix.data,
            n_cols=N_COLS,
        )
        with pytest.raises(ValueError, match="start at 0"):
            broken.validate()

    def test_mismatched_indices_and_data_rejected(self, matrix: SparseMatrix) -> None:
        broken = SparseMatrix(
            indptr=matrix.indptr,
            indices=matrix.indices,
            data=matrix.data[:-1],
            n_cols=N_COLS,
        )
        with pytest.raises(ValueError, match="same shape"):
            broken.validate()

    def test_unsorted_row_rejected(self) -> None:
        broken = SparseMatrix(
            indptr=np.asarray([0, 3], dtype=np.int32),
            indices=np.asarray([5, 1, 3], dtype=np.int32),
            data=np.asarray([1.0, 1.0, 1.0]),
            n_cols=8,
        )
        with pytest.raises(ValueError, match="not sorted and unique"):
            broken.validate()

    def test_duplicate_column_in_a_row_rejected(self) -> None:
        broken = SparseMatrix(
            indptr=np.asarray([0, 2], dtype=np.int32),
            indices=np.asarray([3, 3], dtype=np.int32),
            data=np.asarray([1.0, 1.0]),
            n_cols=8,
        )
        with pytest.raises(ValueError, match="not sorted and unique"):
            broken.validate()

    def test_column_index_beyond_n_cols_rejected(self) -> None:
        broken = SparseMatrix(
            indptr=np.asarray([0, 1], dtype=np.int32),
            indices=np.asarray([99], dtype=np.int32),
            data=np.asarray([1.0]),
            n_cols=8,
        )
        with pytest.raises(ValueError, match=r"outside \[0, n_cols\)"):
            broken.validate()


class TestRowStatistics:
    def test_row_norms_match_numpy(self, matrix: SparseMatrix, dense: np.ndarray) -> None:
        np.testing.assert_allclose(matrix.row_norms(), np.linalg.norm(dense, axis=1))

    def test_row_sums_match_numpy(self, matrix: SparseMatrix, dense: np.ndarray) -> None:
        np.testing.assert_allclose(matrix.row_sums(), dense.sum(axis=1))

    def test_l2_normalized_rows_have_unit_norm(self, matrix: SparseMatrix) -> None:
        norms = matrix.l2_normalized().row_norms()
        nonempty = matrix.row_norms() > 0
        np.testing.assert_allclose(norms[nonempty], 1.0)

    def test_all_zero_row_stays_zero_rather_than_nan(self, matrix: SparseMatrix) -> None:
        """A NaN row would poison every subsequent ranking, silently."""
        normalized = matrix.l2_normalized()
        assert np.all(np.isfinite(normalized.data))
        assert normalized.row_norms()[2] == 0.0

    def test_scaled_by_columns_matches_dense(
        self, matrix: SparseMatrix, dense: np.ndarray
    ) -> None:
        factors = np.arange(1.0, N_COLS + 1.0)
        np.testing.assert_allclose(
            matrix.scaled_by_columns(factors).to_dense(), dense * factors
        )

    def test_wrong_factor_count_rejected(self, matrix: SparseMatrix) -> None:
        with pytest.raises(ValueError, match="expected 8 factors"):
            matrix.scaled_by_columns(np.ones(3))


class TestInvertedIndexIsATranspose:
    """The counting sort is exactly the kind of code that is off by one silently."""

    def test_postings_reproduce_each_column_of_the_dense_matrix(
        self, matrix: SparseMatrix, dense: np.ndarray
    ) -> None:
        inverted = InvertedIndex.from_matrix(matrix)
        for column in range(N_COLS):
            rows, values = inverted.postings(column)
            rebuilt = np.zeros(matrix.n_rows)
            rebuilt[rows] = values
            np.testing.assert_allclose(rebuilt, dense[:, column])

    def test_document_frequency_counts_nonzero_rows_per_column(
        self, matrix: SparseMatrix, dense: np.ndarray
    ) -> None:
        inverted = InvertedIndex.from_matrix(matrix)
        np.testing.assert_array_equal(
            inverted.document_frequency(), (dense != 0).sum(axis=0)
        )

    def test_total_postings_equals_nnz(self, matrix: SparseMatrix) -> None:
        inverted = InvertedIndex.from_matrix(matrix)
        assert int(inverted.document_frequency().sum()) == matrix.nnz

    def test_absent_and_out_of_range_columns_return_empty(self, matrix: SparseMatrix) -> None:
        inverted = InvertedIndex.from_matrix(matrix)
        for column in (-1, N_COLS, N_COLS + 100):
            rows, values = inverted.postings(column)
            assert rows.shape == (0,) and values.shape == (0,)

    def test_shape_metadata(self, matrix: SparseMatrix) -> None:
        inverted = InvertedIndex.from_matrix(matrix)
        assert inverted.n_rows == matrix.n_rows
        assert inverted.n_cols == matrix.n_cols


class TestAgainstDenseReference:
    """The optimisation must be *exactly* the thing it replaces."""

    QUERIES: ClassVar[list[dict[int, float]]] = [
        {0: 1.0},
        {7: 2.0},
        {0: 1.0, 3: 1.0, 7: 1.0},
        {2: 0.5, 5: 1.5},
        {},
        {4: 1.0, 6: 1.0},
    ]

    @pytest.mark.parametrize("query_index", range(6))
    def test_inverted_score_equals_dense_matvec(
        self, matrix: SparseMatrix, dense: np.ndarray, query_index: int
    ) -> None:
        query = self.QUERIES[query_index]
        inverted = InvertedIndex.from_matrix(matrix)
        vector = np.zeros(N_COLS)
        for column, weight in query.items():
            vector[column] = weight
        np.testing.assert_allclose(inverted.score(query), dense @ vector)

    @pytest.mark.parametrize("query_index", range(6))
    def test_row_major_dot_agrees_with_inverted_score(
        self, matrix: SparseMatrix, query_index: int
    ) -> None:
        """Two independent implementations of the same product must agree."""
        query = self.QUERIES[query_index]
        inverted = InvertedIndex.from_matrix(matrix)
        np.testing.assert_allclose(matrix.dot_vector(query), inverted.score(query))

    def test_normalized_dot_is_a_cosine(self, matrix: SparseMatrix, dense: np.ndarray) -> None:
        normalized = matrix.l2_normalized()
        inverted = InvertedIndex.from_matrix(normalized)
        vector = np.zeros(N_COLS)
        vector[[0, 7]] = 1.0
        unit = vector / np.linalg.norm(vector)
        scores = inverted.score({c: float(unit[c]) for c in (0, 7)})
        norms = np.linalg.norm(dense, axis=1)
        expected = np.divide(
            dense @ unit, norms, out=np.zeros(matrix.n_rows), where=norms > 0
        )
        np.testing.assert_allclose(scores, expected)
        assert np.all(scores <= 1.0 + 1e-12)

    def test_zero_weight_query_terms_are_skipped_not_counted(
        self, matrix: SparseMatrix
    ) -> None:
        inverted = InvertedIndex.from_matrix(matrix)
        np.testing.assert_allclose(
            inverted.score({0: 1.0}), inverted.score({0: 1.0, 3: 0.0})
        )


class TestBm25:
    """Asserted against the formula written out by hand, not against itself."""

    K1 = 1.2
    B = 0.75

    def _reference(
        self,
        dense: np.ndarray,
        query: dict[int, float],
        idf: np.ndarray,
        doc_len: np.ndarray,
    ) -> np.ndarray:
        average = doc_len.mean()
        scores = np.zeros(dense.shape[0])
        for row in range(dense.shape[0]):
            total = 0.0
            for column, query_count in query.items():
                tf = dense[row, column]
                if tf == 0.0:
                    continue
                numerator = tf * (self.K1 + 1.0)
                denominator = tf + self.K1 * (
                    1.0 - self.B + self.B * doc_len[row] / average
                )
                total += query_count * idf[column] * numerator / denominator
            scores[row] = total
        return scores

    @pytest.mark.parametrize(
        "query",
        [{0: 1.0}, {7: 1.0}, {0: 1.0, 3: 2.0}, {1: 1.0, 7: 1.0}, {2: 1.0, 4: 1.0, 6: 1.0}],
    )
    def test_matches_hand_written_formula(
        self, matrix: SparseMatrix, dense: np.ndarray, query: dict[int, float]
    ) -> None:
        inverted = InvertedIndex.from_matrix(matrix)
        df = (dense != 0).sum(axis=0).astype(float)
        total = float(matrix.n_rows)
        idf = np.maximum(np.log(1.0 + (total - df + 0.5) / (df + 0.5)), 0.0)
        doc_len = matrix.row_sums()
        np.testing.assert_allclose(
            inverted.bm25(query, idf=idf, doc_len=doc_len, k1=self.K1, b=self.B),
            self._reference(dense, query, idf, doc_len),
        )

    def test_term_frequency_saturates(self, matrix: SparseMatrix) -> None:
        """The defining property of BM25 versus raw TF: doubling tf does not double score."""
        inverted = InvertedIndex.from_matrix(matrix)
        idf = np.ones(N_COLS)
        doc_len = matrix.row_sums()
        scores = inverted.bm25({7: 1.0}, idf=idf, doc_len=doc_len, k1=self.K1, b=0.0)
        # Row 0 has tf 4, row 4 has tf 5 — a 25% rise in tf must give far less than 25%.
        ratio = scores[4] / scores[0]
        assert 1.0 < ratio < 1.10

    def test_idf_of_a_universal_term_is_zero(self, matrix: SparseMatrix) -> None:
        """A term in every chunk must not be able to subtract score.

        The unsmoothed Robertson-Sparck-Jones form goes negative above df = n/2, which
        reorders results nonsensically. Both the ``1 +`` form and the floor are here to
        prevent it.
        """
        inverted = InvertedIndex.from_matrix(matrix)
        df = np.full(N_COLS, float(matrix.n_rows))
        idf = np.maximum(
            np.log(1.0 + (matrix.n_rows - df + 0.5) / (df + 0.5)), 0.0
        )
        scores = inverted.bm25(
            {0: 1.0}, idf=idf, doc_len=matrix.row_sums(), k1=self.K1, b=self.B
        )
        assert np.all(scores >= 0.0)

    @pytest.mark.parametrize("b", [-0.1, 1.1])
    def test_invalid_b_rejected(self, matrix: SparseMatrix, b: float) -> None:
        inverted = InvertedIndex.from_matrix(matrix)
        with pytest.raises(ValueError, match="b must be"):
            inverted.bm25(
                {0: 1.0}, idf=np.ones(N_COLS), doc_len=matrix.row_sums(), k1=1.2, b=b
            )

    @pytest.mark.parametrize("k1", [0.0, -1.0])
    def test_invalid_k1_rejected(self, matrix: SparseMatrix, k1: float) -> None:
        inverted = InvertedIndex.from_matrix(matrix)
        with pytest.raises(ValueError, match="k1 must be"):
            inverted.bm25(
                {0: 1.0}, idf=np.ones(N_COLS), doc_len=matrix.row_sums(), k1=k1, b=0.75
            )

    def test_wrong_sized_idf_or_lengths_rejected(self, matrix: SparseMatrix) -> None:
        inverted = InvertedIndex.from_matrix(matrix)
        with pytest.raises(ValueError, match="idf values"):
            inverted.bm25(
                {0: 1.0}, idf=np.ones(3), doc_len=matrix.row_sums(), k1=1.2, b=0.75
            )
        with pytest.raises(ValueError, match="lengths"):
            inverted.bm25({0: 1.0}, idf=np.ones(N_COLS), doc_len=np.ones(2), k1=1.2, b=0.75)


class TestScalesToTheRealCorpus:
    """Randomised cross-check at a realistic shape, since the fixture is tiny."""

    def test_random_matrix_agrees_with_dense(self) -> None:
        rng = np.random.default_rng(20260929)
        n_rows, n_cols = 60, 512
        rows = [
            {int(c): float(rng.uniform(0.5, 4.0))
             for c in rng.choice(n_cols, size=int(rng.integers(5, 40)), replace=False)}
            for _ in range(n_rows)
        ]
        matrix = SparseMatrix.from_rows(rows, n_cols=n_cols)
        matrix.validate()
        dense = matrix.to_dense()
        inverted = InvertedIndex.from_matrix(matrix)
        for _ in range(20):
            columns = rng.choice(n_cols, size=6, replace=False)
            query = {int(c): float(rng.uniform(0.1, 2.0)) for c in columns}
            vector = np.zeros(n_cols)
            for column, weight in query.items():
                vector[column] = weight
            np.testing.assert_allclose(inverted.score(query), dense @ vector, atol=1e-12)
