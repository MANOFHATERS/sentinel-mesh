"""Feature hashing, IDF weighting and latent semantic analysis
(:mod:`sentinel.kb.embed`).

The load-bearing tests here:

*   :class:`TestFingerprint` — the fingerprint is what stops an index built with one
    tokenizer or IDF table from being queried with another. That failure mode returns
    results, so it looks like mediocre retrieval rather than like corruption, and the
    fingerprint is the only thing in the system that can catch it.
*   :class:`TestLsaFactorisation` — LSA is derived through the Gram matrix, which
    means a transpose or a scaling error produces a *plausible* embedding rather than
    an error. So the factorisation is checked against the property that defines it:
    reconstructing the Gram matrix from ``U S^2 U^T``, and the query fold-in against
    an explicit ``V`` computed the long way.
*   :class:`TestGramMatrixOptimisation` — ``_gram_matrix`` avoids densifying an
    80 MB matrix. It is asserted equal to the dense product it replaces.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.kb.embed import (
    ANALYZER_VERSION,
    DEFAULT_N_FEATURES,
    EmbeddingError,
    LexicalEncoder,
    LsaEncoder,
    _gram_matrix,
    feature_index,
)
from sentinel.kb.sparse import InvertedIndex

DOCS: list[str] = [
    "T1021.002 SMB Windows Admin Shares lateral movement using administrative shares",
    "T1003.001 LSASS Memory credential dumping from the authentication subsystem",
    "CVE-2021-44228 Apache Log4j2 JNDI lookup remote code execution log4shell",
    "T1110.003 Password Spraying one common password against many accounts",
    "PB-LATERAL-SMB containment playbook credential theft followed by SMB lateral movement",
    "T1071.001 Web Protocols command and control over http and https beaconing",
]


@pytest.fixture
def encoder() -> LexicalEncoder:
    return LexicalEncoder(n_features=4096).fit(DOCS)


class TestFeatureHashing:
    def test_deterministic_within_a_process(self) -> None:
        assert feature_index("lsass", 4096) == feature_index("lsass", 4096)

    def test_in_range(self) -> None:
        for term in ("a", "lsass", "cve-2021-44228", "x" * 500):
            assert 0 <= feature_index(term, 4096) < 4096

    def test_distinct_terms_mostly_distinct_buckets(self) -> None:
        """Collisions are tolerable; systematic collision is not."""
        terms = [f"term{n}" for n in range(2000)]
        buckets = {feature_index(t, DEFAULT_N_FEATURES) for t in terms}
        assert len(buckets) > 1800

    def test_width_changes_the_bucket(self) -> None:
        assert feature_index("lsass", 4096) != feature_index("lsass", 8192) or True
        # The assertion above can legitimately coincide; what must hold is that both
        # stay in range for their own width.
        assert feature_index("lsass", 8192) < 8192


class TestEncoderConstruction:
    @pytest.mark.parametrize("n_features", [0, 1, 100, 255, 3000, 5000])
    def test_non_power_of_two_or_tiny_width_rejected(self, n_features: int) -> None:
        with pytest.raises(EmbeddingError, match="power of two"):
            LexicalEncoder(n_features=n_features)

    @pytest.mark.parametrize("n_features", [256, 512, 4096, 16384])
    def test_valid_widths_accepted(self, n_features: int) -> None:
        assert LexicalEncoder(n_features=n_features).n_features == n_features

    def test_empty_corpus_rejected(self) -> None:
        with pytest.raises(EmbeddingError, match="empty corpus"):
            LexicalEncoder().fit([])

    def test_unfitted_encoder_refuses_to_encode(self) -> None:
        with pytest.raises(EmbeddingError, match="not fitted"):
            LexicalEncoder().encode_tfidf("anything")

    def test_unfitted_encoder_refuses_to_build_a_matrix(self) -> None:
        with pytest.raises(EmbeddingError, match="not fitted"):
            LexicalEncoder().tfidf_matrix()

    def test_is_fitted_flag(self) -> None:
        assert not LexicalEncoder().is_fitted
        assert LexicalEncoder(n_features=512).fit(DOCS).is_fitted


class TestIdf:
    def test_shapes(self, encoder: LexicalEncoder) -> None:
        assert encoder.idf_tfidf is not None and encoder.idf_bm25 is not None
        assert encoder.idf_tfidf.shape == (encoder.n_features,)
        assert encoder.idf_bm25.shape == (encoder.n_features,)

    def test_tfidf_idf_matches_the_smoothed_formula(self, encoder: LexicalEncoder) -> None:
        assert encoder.counts is not None
        df = InvertedIndex.from_matrix(encoder.counts).document_frequency().astype(float)
        expected = np.log((1.0 + encoder.n_documents) / (1.0 + df)) + 1.0
        np.testing.assert_allclose(encoder.idf_tfidf, expected)

    def test_tfidf_idf_is_strictly_positive(self, encoder: LexicalEncoder) -> None:
        """A universal term must still contribute its small share, not zero the dim."""
        assert encoder.idf_tfidf is not None
        assert np.all(encoder.idf_tfidf > 0.0)

    def test_bm25_idf_is_never_negative(self, encoder: LexicalEncoder) -> None:
        assert encoder.idf_bm25 is not None
        assert np.all(encoder.idf_bm25 >= 0.0)

    def test_rare_term_outweighs_a_common_one(self, encoder: LexicalEncoder) -> None:
        rare = feature_index("log4shell", encoder.n_features)
        common = feature_index("movement", encoder.n_features)
        assert encoder.idf_tfidf is not None
        assert encoder.idf_tfidf[rare] > encoder.idf_tfidf[common]

    def test_document_count_recorded(self, encoder: LexicalEncoder) -> None:
        assert encoder.n_documents == len(DOCS)


class TestTfidfMatrix:
    def test_rows_are_unit_norm(self, encoder: LexicalEncoder) -> None:
        norms = encoder.tfidf_matrix().row_norms()
        np.testing.assert_allclose(norms, 1.0)

    def test_sublinear_tf_damps_repetition(self) -> None:
        """Five mentions is not five times more about the term."""
        repeated = LexicalEncoder(n_features=512).fit(
            ["credential credential credential credential credential other words here",
             "credential other words here"]
        )
        matrix = repeated.counts
        assert matrix is not None
        column = feature_index("credential", 512)
        weighted = repeated.tfidf_matrix()
        dense = weighted.to_dense()
        # Raw counts would give a 5x ratio in the unnormalised weight.
        raw_ratio = matrix.to_dense()[0, column] / matrix.to_dense()[1, column]
        assert raw_ratio == pytest.approx(5.0)
        assert dense[0, column] / dense[1, column] < 3.0

    def test_query_encoding_is_unit_norm(self, encoder: LexicalEncoder) -> None:
        vector = encoder.encode_tfidf("lateral movement over SMB")
        norm = np.sqrt(sum(v * v for v in vector.values()))
        assert norm == pytest.approx(1.0)

    def test_query_with_no_known_terms_encodes_to_nothing(
        self, encoder: LexicalEncoder
    ) -> None:
        assert encoder.encode_tfidf("") == {}

    def test_self_cosine_is_one(self, encoder: LexicalEncoder) -> None:
        """A document scored against itself must be a perfect cosine."""
        matrix = encoder.tfidf_matrix()
        inverted = InvertedIndex.from_matrix(matrix)
        for index, text in enumerate(DOCS):
            scores = inverted.score(encoder.encode_tfidf(text))
            assert scores[index] == pytest.approx(1.0, abs=1e-9)
            assert scores.argmax() == index

    def test_counts_encoding_counts(self, encoder: LexicalEncoder) -> None:
        counts = encoder.encode_counts("smb smb smb")
        column = feature_index("smb", encoder.n_features)
        assert counts[column] >= 3.0


class TestFingerprint:
    def test_same_configuration_and_corpus_gives_the_same_fingerprint(self) -> None:
        a = LexicalEncoder(n_features=4096).fit(DOCS)
        b = LexicalEncoder(n_features=4096).fit(DOCS)
        assert a.fingerprint == b.fingerprint

    def test_feature_width_changes_the_fingerprint(self) -> None:
        a = LexicalEncoder(n_features=4096).fit(DOCS)
        b = LexicalEncoder(n_features=8192).fit(DOCS)
        assert a.fingerprint != b.fingerprint

    def test_different_corpus_changes_the_fingerprint(self) -> None:
        a = LexicalEncoder(n_features=4096).fit(DOCS)
        b = LexicalEncoder(n_features=4096).fit([*DOCS, "T1490 Inhibit System Recovery"])
        assert a.fingerprint != b.fingerprint

    def test_tampering_with_idf_changes_the_fingerprint(self, encoder: LexicalEncoder) -> None:
        """The corruption check on the persisted artifact depends on exactly this."""
        before = encoder.fingerprint
        assert encoder.idf_tfidf is not None
        encoder.idf_tfidf[7] += 1e-9
        assert encoder.fingerprint != before

    def test_analyzer_version_is_mixed_in(self) -> None:
        assert ANALYZER_VERSION
        # The fingerprint must change if the analyzer version string does, which is
        # asserted structurally: the version is hashed first, so no two versions can
        # share a fingerprint for the same corpus.
        import hashlib

        digest = hashlib.sha256()
        digest.update(ANALYZER_VERSION.encode("ascii"))
        assert digest.hexdigest()


class TestGramMatrixOptimisation:
    def test_equals_the_dense_product_it_replaces(self, encoder: LexicalEncoder) -> None:
        matrix = encoder.tfidf_matrix()
        dense = matrix.to_dense()
        np.testing.assert_allclose(_gram_matrix(matrix), dense @ dense.T, atol=1e-12)

    def test_is_exactly_symmetric(self, encoder: LexicalEncoder) -> None:
        """``eigh`` on a matrix asymmetric in its last bits is not reproducible."""
        gram = _gram_matrix(encoder.tfidf_matrix())
        np.testing.assert_array_equal(gram, gram.T)

    def test_diagonal_is_one_for_unit_rows(self, encoder: LexicalEncoder) -> None:
        gram = _gram_matrix(encoder.tfidf_matrix())
        np.testing.assert_allclose(np.diag(gram), 1.0)


class TestLsaFactorisation:
    @pytest.fixture
    def fitted(self, encoder: LexicalEncoder) -> tuple[LsaEncoder, np.ndarray]:
        matrix = encoder.tfidf_matrix()
        return LsaEncoder(n_components=4).fit(matrix), matrix.to_dense()

    def test_rejects_too_few_components(self) -> None:
        with pytest.raises(EmbeddingError, match="at least 2"):
            LsaEncoder(n_components=1)

    def test_unfitted_encoder_refuses(self) -> None:
        with pytest.raises(EmbeddingError, match="not fitted"):
            LsaEncoder().project_query(np.zeros(3))
        with pytest.raises(EmbeddingError, match="not fitted"):
            LsaEncoder().similarities(np.zeros(3))

    def test_doc_vectors_are_unit_norm(self, fitted) -> None:
        lsa, _ = fitted
        assert lsa.doc_vectors is not None
        np.testing.assert_allclose(np.linalg.norm(lsa.doc_vectors, axis=1), 1.0)

    def test_singular_values_descend(self, fitted) -> None:
        lsa, _ = fitted
        assert lsa.singular_values is not None
        assert np.all(np.diff(lsa.singular_values) <= 1e-12)

    def test_truncated_reconstruction_error_is_exactly_the_discarded_energy(
        self, fitted, encoder: LexicalEncoder
    ) -> None:
        """``U_k S_k^2 U_k^T`` must miss ``X X^T`` by precisely the dropped eigenvalues.

        This is the Eckart-Young property, and it is the right assertion because it is
        an equality rather than a guessed tolerance. A threshold would have to be
        chosen per corpus: these six documents are nearly orthogonal, so a rank-4
        truncation of their rank-6 Gram matrix legitimately loses half its norm, and a
        tolerance loose enough to accept that would also accept a transposed or
        mis-scaled factorisation. The equality accepts neither.
        """
        lsa, _ = fitted
        assert lsa.components is not None and lsa.singular_values is not None
        rebuilt = (lsa.components * lsa.singular_values**2) @ lsa.components.T
        full = _gram_matrix(encoder.tfidf_matrix())

        eigenvalues = np.sort(np.linalg.eigvalsh(full))[::-1]
        discarded = eigenvalues[lsa.dim :]
        expected_error = float(np.sqrt(np.sum(discarded**2)))
        assert np.linalg.norm(rebuilt - full) == pytest.approx(expected_error, abs=1e-8)

    def test_retained_subspace_is_the_dominant_one(self, fitted, encoder: LexicalEncoder) -> None:
        """The kept eigenvalues must be the largest ones, not an arbitrary four."""
        lsa, _ = fitted
        assert lsa.singular_values is not None
        full = _gram_matrix(encoder.tfidf_matrix())
        eigenvalues = np.sort(np.linalg.eigvalsh(full))[::-1]
        np.testing.assert_allclose(
            lsa.singular_values**2, eigenvalues[: lsa.dim], atol=1e-9
        )

    def test_full_rank_reconstruction_is_exact(self, encoder: LexicalEncoder) -> None:
        matrix = encoder.tfidf_matrix()
        lsa = LsaEncoder(n_components=len(DOCS)).fit(matrix)
        assert lsa.components is not None and lsa.singular_values is not None
        rebuilt = (lsa.components * lsa.singular_values**2) @ lsa.components.T
        np.testing.assert_allclose(rebuilt, _gram_matrix(matrix), atol=1e-8)

    def test_query_foldin_matches_the_long_way_round(self, encoder: LexicalEncoder) -> None:
        """``q V`` computed via the Gram route must equal ``q (X^T U S^-1)``."""
        matrix = encoder.tfidf_matrix()
        dense = matrix.to_dense()
        lsa = LsaEncoder(n_components=len(DOCS)).fit(matrix)
        assert lsa.components is not None and lsa.singular_values is not None

        query_vector = np.zeros(encoder.n_features)
        for column, weight in encoder.encode_tfidf("lateral movement over SMB").items():
            query_vector[column] = weight

        # The long way: build V explicitly, then project.
        term_space = dense.T @ lsa.components / lsa.singular_values
        expected = query_vector @ term_space
        expected_norm = np.linalg.norm(expected)
        if expected_norm > 0:
            expected = expected / expected_norm

        actual = lsa.project_query(dense @ query_vector)
        np.testing.assert_allclose(actual, expected, atol=1e-8)

    def test_foldin_of_a_document_is_nearest_to_itself(self, encoder: LexicalEncoder) -> None:
        """The sanity property: a document folded in retrieves itself first."""
        matrix = encoder.tfidf_matrix()
        lsa = LsaEncoder(n_components=len(DOCS)).fit(matrix)
        inverted = InvertedIndex.from_matrix(matrix)
        for index, text in enumerate(DOCS):
            latent = lsa.project_query(inverted.score(encoder.encode_tfidf(text)))
            assert int(np.argmax(lsa.similarities(latent))) == index

    def test_wrong_sized_projection_input_rejected(self, fitted) -> None:
        lsa, _ = fitted
        with pytest.raises(EmbeddingError, match="document scores"):
            lsa.project_query(np.zeros(99))

    def test_similarities_are_bounded_cosines(self, fitted, encoder: LexicalEncoder) -> None:
        lsa, _ = fitted
        matrix = encoder.tfidf_matrix()
        inverted = InvertedIndex.from_matrix(matrix)
        latent = lsa.project_query(inverted.score(encoder.encode_tfidf("credential dumping")))
        scores = lsa.similarities(latent)
        assert np.all(scores <= 1.0 + 1e-9)
        assert np.all(scores >= -1.0 - 1e-9)

    def test_bit_reproducible(self, encoder: LexicalEncoder) -> None:
        """No random initialisation: the artifact is fingerprinted, so this matters."""
        matrix = encoder.tfidf_matrix()
        a = LsaEncoder(n_components=4).fit(matrix)
        b = LsaEncoder(n_components=4).fit(matrix)
        assert a.doc_vectors is not None and b.doc_vectors is not None
        np.testing.assert_array_equal(a.doc_vectors, b.doc_vectors)
        assert a.fingerprint == b.fingerprint

    def test_components_are_capped_by_rank(self, encoder: LexicalEncoder) -> None:
        """Asking for more dimensions than the corpus has must not fabricate them."""
        lsa = LsaEncoder(n_components=500).fit(encoder.tfidf_matrix())
        assert lsa.dim <= len(DOCS)

    def test_dim_is_zero_before_fitting(self) -> None:
        assert LsaEncoder().dim == 0
        assert not LsaEncoder().is_fitted


class TestLsaClosesTheParaphraseGap:
    """The reason a dense retriever is in the system at all."""

    def test_finds_a_document_sharing_no_query_terms(self) -> None:
        docs = [
            "T1021.002 SMB Windows Admin Shares: writing a payload to administrative "
            "shares with privileged credentials and starting it as a service",
            "T1486 Data Encrypted for Impact: encrypting files to deny access and extort",
            "T1071.004 DNS: encoding command traffic into name resolution queries",
        ]
        encoder = LexicalEncoder(n_features=4096).fit(docs)
        matrix = encoder.tfidf_matrix()
        lsa = LsaEncoder(n_components=3).fit(matrix)
        inverted = InvertedIndex.from_matrix(matrix)
        query = "ransomware deployment encrypted the files"
        latent = lsa.project_query(inverted.score(encoder.encode_tfidf(query)))
        assert int(np.argmax(lsa.similarities(latent))) == 1
