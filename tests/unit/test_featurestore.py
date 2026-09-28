"""Feature store: train/serve parity, spec fingerprinting, transform correctness.

:meth:`TestTrainServeParity.test_batch_and_single_paths_are_bit_identical` is the
reason this module exists. The classic skew bug is a training path that vectorizes
with pandas and a serving path that loops per request; the two agree to about six
decimal places, which is invisible in accuracy metrics and wrong forever. So the
assertion is bit equality, not ``approx``.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sentinel.core.errors import FeatureError, ModelNotFittedError
from sentinel.ml.featurestore import (
    DEFAULT_SPEC,
    AlertVectorizer,
    FeatureSpec,
    Identity,
    Indicator,
    Log1p,
    OneHot,
    default_spec,
)


class TestSpec:
    def test_default_spec_is_non_trivial(self):
        assert DEFAULT_SPEC.width >= 25

    def test_column_names_are_unique(self):
        names = DEFAULT_SPEC.column_names
        assert len(set(names)) == len(names)

    def test_fingerprint_is_stable_across_constructions(self):
        assert default_spec().fingerprint == default_spec().fingerprint

    def test_fingerprint_changes_when_a_column_is_added(self):
        extended = FeatureSpec(
            name=DEFAULT_SPEC.name,
            columns=(*DEFAULT_SPEC.columns, Identity("extra", "total_bytes")),
        )
        assert extended.fingerprint != DEFAULT_SPEC.fingerprint

    def test_fingerprint_changes_when_columns_are_reordered(self):
        # Reordering produces different vectors, so the specs must not be
        # interchangeable even though they contain the same columns.
        reordered = FeatureSpec(
            name=DEFAULT_SPEC.name, columns=tuple(reversed(DEFAULT_SPEC.columns))
        )
        assert reordered.fingerprint != DEFAULT_SPEC.fingerprint

    def test_fingerprint_changes_when_a_transform_changes(self):
        swapped = FeatureSpec(
            name="x",
            columns=(Log1p("v", "total_bytes"),),
        )
        other = FeatureSpec(name="x", columns=(Identity("v", "total_bytes"),))
        assert swapped.fingerprint != other.fingerprint

    def test_fingerprint_changes_when_a_threshold_changes(self):
        a = FeatureSpec(name="x", columns=(Indicator("p", "dst_port", "lt", 1024.0),))
        b = FeatureSpec(name="x", columns=(Indicator("p", "dst_port", "lt", 2048.0),))
        assert a.fingerprint != b.fingerprint

    def test_empty_spec_rejected(self):
        with pytest.raises(ValueError, match="at least one column"):
            FeatureSpec(name="x", columns=())

    def test_duplicate_column_names_rejected(self):
        with pytest.raises(ValueError, match="duplicate column names"):
            FeatureSpec(
                name="x",
                columns=(Identity("v", "total_bytes"), Log1p("v", "src_bytes")),
            )

    def test_describe_lists_every_column(self):
        described = DEFAULT_SPEC.describe()
        for name in DEFAULT_SPEC.column_names:
            assert name in described

    def test_unknown_predicate_rejected(self):
        with pytest.raises(ValueError, match="unknown predicate"):
            Indicator("x", "dst_port", "definitely_not_a_predicate")

    def test_onehot_other_requires_known_levels(self):
        with pytest.raises(ValueError, match="known levels"):
            OneHot("x", "protocol_number", None, is_other=True)

    def test_empty_column_name_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            Identity("", "total_bytes")


class TestColumnTransforms:
    def _alert(self, small_alerts, **features):
        return small_alerts[0].updated(features={**small_alerts[0].features, **features})

    def test_log1p_compresses_heavy_tails(self, small_alerts):
        column = Log1p("v", "total_bytes")
        assert column.extract(self._alert(small_alerts, total_bytes=0.0)) == 0.0
        assert column.extract(self._alert(small_alerts, total_bytes=math.e - 1)) == pytest.approx(
            1.0
        )

    def test_log1p_clips_negatives_rather_than_producing_nan(self, small_alerts):
        # A negative byte count is a source defect; a NaN would poison the batch.
        column = Log1p("v", "total_bytes")
        assert column.extract(self._alert(small_alerts, total_bytes=-500.0)) == 0.0

    def test_missing_feature_is_zero_not_an_error(self, small_alerts):
        assert Log1p("v", "not_a_feature").extract(small_alerts[0]) == 0.0

    def test_identity_passes_values_through(self, small_alerts):
        column = Identity("v", "bytes_ratio_src_to_total")
        assert column.extract(
            self._alert(small_alerts, bytes_ratio_src_to_total=0.42)
        ) == pytest.approx(0.42)

    @pytest.mark.parametrize(
        ("predicate", "threshold", "value", "expected"),
        [
            ("gt", 10.0, 11.0, 1.0),
            ("gt", 10.0, 10.0, 0.0),
            ("gte", 10.0, 10.0, 1.0),
            ("lt", 10.0, 9.0, 1.0),
            ("lte", 10.0, 10.0, 1.0),
            ("eq", 0.0, 0.0, 1.0),
            ("eq", 0.0, 1.0, 0.0),
            ("in_range_exclusive_upper", 1024.0, 443.0, 1.0),
            ("in_range_exclusive_upper", 1024.0, 1024.0, 0.0),
            ("in_range_exclusive_upper", 1024.0, -1.0, 0.0),
        ],
    )
    def test_indicator_predicates(self, small_alerts, predicate, threshold, value, expected):
        column = Indicator("v", "dst_port", predicate, threshold)
        assert column.extract(self._alert(small_alerts, dst_port=value)) == expected

    def test_onehot_fires_on_its_level_only(self, small_alerts):
        tcp = OneHot("tcp", "protocol_number", 6)
        assert tcp.extract(self._alert(small_alerts, protocol_number=6)) == 1.0
        assert tcp.extract(self._alert(small_alerts, protocol_number=17)) == 0.0

    def test_onehot_other_fires_on_unseen_levels(self, small_alerts):
        other = OneHot(
            "other", "protocol_number", None, is_other=True, known_levels=(6, 17, 1)
        )
        assert other.extract(self._alert(small_alerts, protocol_number=132)) == 1.0
        assert other.extract(self._alert(small_alerts, protocol_number=6)) == 0.0

    def test_missing_protocol_is_other_not_tcp(self, small_alerts):
        # Encoding a missing protocol as all-zeros would make it indistinguishable
        # from an unseen one; both need to reach the 'other' bucket.
        alert = self._alert(small_alerts, protocol_number=None)
        assert OneHot("tcp", "protocol_number", 6).extract(alert) == 0.0
        assert (
            OneHot("o", "protocol_number", None, is_other=True, known_levels=(6, 17, 1)).extract(
                alert
            )
            == 1.0
        )

    def test_exactly_one_protocol_column_fires(self, small_alerts):
        # A one-hot block that can fire twice or never is a silent modelling bug.
        protocol_columns = [c for c in DEFAULT_SPEC.columns if c.name.startswith("proto_")]
        for value in (6, 17, 1, 132, None, 255):
            alert = self._alert(small_alerts, protocol_number=value)
            fired = sum(column.extract(alert) for column in protocol_columns)
            assert fired == 1.0, f"protocol_number={value} fired {fired} columns"

    def test_string_feature_values_do_not_crash_numeric_columns(self, small_alerts):
        assert Log1p("v", "protocol").extract(
            small_alerts[0].updated(features={**small_alerts[0].features, "protocol": "tcp"})
        ) == 0.0


class TestTrainServeParity:
    def test_batch_and_single_paths_are_bit_identical(self, small_alerts):
        """Bit equality, not approx. A last-bit disagreement is still skew."""
        vectorizer = AlertVectorizer().fit(small_alerts)
        batch = vectorizer.transform(small_alerts)
        for index, alert in enumerate(small_alerts):
            single = vectorizer.transform_one(alert)
            assert np.array_equal(single, batch[index]), f"row {index} differs"

    def test_single_path_returns_one_dimension(self, small_alerts):
        vectorizer = AlertVectorizer().fit(small_alerts)
        assert vectorizer.transform_one(small_alerts[0]).shape == (DEFAULT_SPEC.width,)

    def test_only_canonical_alerts_are_accepted(self, small_alerts):
        # There is no fit(dataframe) overload, so training data cannot bypass the
        # normalizer and read a column the serving path never sees.
        vectorizer = AlertVectorizer()
        with pytest.raises(FeatureError, match="only accepts canonical Alert"):
            vectorizer.fit([{"total_bytes": 1.0}])  # type: ignore[list-item]

    def test_dict_is_rejected_at_serving_time_too(self, small_alerts):
        vectorizer = AlertVectorizer().fit(small_alerts)
        with pytest.raises(FeatureError, match="Route source data through a Normalizer"):
            vectorizer.transform_one({"total_bytes": 1.0})  # type: ignore[arg-type]

    def test_output_is_float64(self, small_alerts):
        vectorizer = AlertVectorizer().fit(small_alerts)
        assert vectorizer.transform(small_alerts).dtype == np.float64

    def test_no_nan_or_inf_reaches_the_matrix(self, small_alerts):
        vectorizer = AlertVectorizer().fit(small_alerts)
        matrix = vectorizer.transform(small_alerts)
        assert np.all(np.isfinite(matrix))

    def test_shape_matches_the_spec(self, small_alerts):
        vectorizer = AlertVectorizer().fit(small_alerts)
        assert vectorizer.transform(small_alerts).shape == (
            len(small_alerts),
            DEFAULT_SPEC.width,
        )


class TestStandardization:
    def test_fitted_columns_are_centred_and_scaled(self, small_alerts):
        matrix = AlertVectorizer().fit_transform(small_alerts)
        non_degenerate = matrix.std(axis=0) > 1e-9
        assert np.allclose(matrix.mean(axis=0)[non_degenerate], 0.0, atol=1e-9)
        assert np.allclose(matrix.std(axis=0)[non_degenerate], 1.0, atol=1e-9)

    def test_zero_variance_columns_do_not_divide_by_zero(self, small_alerts):
        matrix = AlertVectorizer().fit_transform(small_alerts)
        assert np.all(np.isfinite(matrix))

    def test_degenerate_columns_are_reported(self, small_alerts):
        # A constant feature in production usually means an upstream field stopped
        # being populated: a detection gap wearing a modelling costume.
        vectorizer = AlertVectorizer().fit(small_alerts)
        assert isinstance(vectorizer.degenerate_columns, tuple)

    def test_standardize_false_returns_raw_values(self, small_alerts):
        vectorizer = AlertVectorizer(standardize=False)
        matrix = vectorizer.transform(small_alerts)
        assert matrix.min() >= 0.0  # log1p and indicators are all non-negative

    def test_unfitted_transform_is_refused(self, small_alerts):
        with pytest.raises(ModelNotFittedError, match="must be fitted"):
            AlertVectorizer().transform(small_alerts)

    def test_unfitted_degenerate_columns_is_refused(self):
        with pytest.raises(ModelNotFittedError):
            _ = AlertVectorizer().degenerate_columns

    def test_fitting_on_nothing_is_refused(self):
        with pytest.raises(FeatureError, match="zero alerts"):
            AlertVectorizer().fit([])

    def test_statistics_come_only_from_the_fit_split(self, small_alerts):
        # Fitting on train and transforming test must not recentre the test data:
        # that is leakage, and it makes held-out metrics optimistic.
        train, test = small_alerts[:200], small_alerts[200:]
        vectorizer = AlertVectorizer().fit(train)
        transformed = vectorizer.transform(test)
        assert not np.allclose(transformed.mean(axis=0), 0.0, atol=1e-6)


class TestPersistence:
    def test_state_round_trips(self, small_alerts, tmp_path):
        original = AlertVectorizer().fit(small_alerts)
        path = original.save(tmp_path / "vec.json")
        restored = AlertVectorizer.load(path)
        assert np.array_equal(
            original.transform(small_alerts), restored.transform(small_alerts)
        )

    def test_saved_state_carries_the_fingerprint(self, small_alerts, tmp_path):
        path = AlertVectorizer().fit(small_alerts).save(tmp_path / "vec.json")
        state = json.loads(path.read_text())
        assert state["spec_fingerprint"] == DEFAULT_SPEC.fingerprint
        assert state["column_names"] == list(DEFAULT_SPEC.column_names)

    def test_loading_under_a_changed_spec_is_refused(self, small_alerts, tmp_path):
        """The skew guard: a stale artifact cannot be paired with a new spec."""
        path = AlertVectorizer().fit(small_alerts).save(tmp_path / "vec.json")
        changed = FeatureSpec(
            name="changed",
            columns=(*DEFAULT_SPEC.columns, Identity("new_column", "total_bytes")),
        )
        with pytest.raises(FeatureError, match="fingerprint mismatch"):
            AlertVectorizer.load(path, spec=changed)

    def test_saving_an_unfitted_vectorizer_is_refused(self, tmp_path):
        with pytest.raises(ModelNotFittedError, match="not fitted"):
            AlertVectorizer().save(tmp_path / "vec.json")

    def test_assert_compatible_accepts_a_matching_fingerprint(self, small_alerts):
        AlertVectorizer().fit(small_alerts).assert_compatible(DEFAULT_SPEC.fingerprint)

    def test_assert_compatible_rejects_a_mismatch(self, small_alerts):
        with pytest.raises(FeatureError, match="refusing to score"):
            AlertVectorizer().fit(small_alerts).assert_compatible("0" * 64)

    def test_mismatched_statistics_width_is_refused(self, small_alerts, tmp_path):
        path = AlertVectorizer().fit(small_alerts).save(tmp_path / "vec.json")
        state = json.loads(path.read_text())
        state["mean"] = state["mean"][:-1]
        with pytest.raises(FeatureError, match="do not match the spec width"):
            AlertVectorizer().load_state(state)


class TestDeterminism:
    def test_identical_inputs_give_identical_matrices(self, small_alerts):
        first = AlertVectorizer().fit_transform(small_alerts)
        second = AlertVectorizer().fit_transform(small_alerts)
        assert np.array_equal(first, second)

    def test_vectorization_is_order_independent_per_row(self, small_alerts):
        vectorizer = AlertVectorizer().fit(small_alerts)
        forward = vectorizer.transform(small_alerts)
        backward = vectorizer.transform(list(reversed(small_alerts)))
        assert np.array_equal(forward, backward[::-1])
