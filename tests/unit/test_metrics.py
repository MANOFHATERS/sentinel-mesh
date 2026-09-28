"""Metrics and splitting: correctness, leakage refusal, and honest failure modes.

The theme is that a metric which quietly returns a plausible number on degenerate
input is worse than one that raises. An AUC of 0.5 on an all-benign split looks
like "the model is no better than chance" when the truth is "this number is
meaningless" — and that difference decides whether anyone investigates.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.ml.metrics import (
    SplitIndices,
    average_precision,
    detection_report,
    precision_recall_f1,
    recall_by_class,
    roc_auc,
    three_way_split,
)


class TestRocAuc:
    def test_perfect_separation_is_one(self):
        assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0

    def test_inverted_separation_is_zero(self):
        assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0

    def test_random_scores_are_near_one_half(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, size=5000)
        assert abs(roc_auc(y, rng.random(5000)) - 0.5) < 0.05

    def test_all_ties_give_one_half(self):
        assert roc_auc([0, 1, 0, 1], [0.5] * 4) == pytest.approx(0.5)

    def test_single_class_is_refused_not_guessed(self):
        # Returning 0.5 here would let an all-benign evaluation report a normal-looking
        # number.
        with pytest.raises(ValueError, match="undefined with a single class"):
            roc_auc([0, 0, 0], [0.1, 0.2, 0.3])

    def test_empty_input_is_refused(self):
        with pytest.raises(ValueError, match="empty split"):
            roc_auc([], [])

    def test_length_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="length mismatch"):
            roc_auc([0, 1], [0.1])

    def test_non_finite_scores_refused(self):
        with pytest.raises(ValueError, match="non-finite"):
            roc_auc([0, 1], [0.1, np.nan])


class TestAveragePrecision:
    def test_perfect_ranking_is_one(self):
        assert average_precision([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)

    def test_responds_to_imbalance_more_than_auc(self):
        # 1% positives, with 20 false positives ranked above every true positive.
        y = np.concatenate([np.zeros(990, dtype=int), np.ones(10, dtype=int)])
        scores = np.concatenate([np.zeros(970), np.full(20, 0.95), np.full(10, 0.9)])
        assert roc_auc(y, scores) > 0.95
        assert average_precision(y, scores) < 0.4

    def test_single_class_refused(self):
        with pytest.raises(ValueError, match="single class"):
            average_precision([1, 1], [0.5, 0.6])


class TestPrecisionRecallF1:
    def test_perfect_prediction(self):
        assert precision_recall_f1([0, 1, 1], [0, 1, 1]) == (1.0, 1.0, 1.0)

    def test_all_negative_prediction_gives_zero_not_nan(self):
        precision, recall, f1 = precision_recall_f1([0, 1, 1], [0, 0, 0])
        assert (precision, recall, f1) == (0.0, 0.0, 0.0)

    def test_computes_from_the_attack_class(self):
        # 2 true positives, 1 false positive, 1 false negative.
        precision, recall, _ = precision_recall_f1([1, 1, 1, 0], [1, 1, 0, 1])
        assert precision == pytest.approx(2 / 3)
        assert recall == pytest.approx(2 / 3)

    def test_length_mismatch_refused(self):
        with pytest.raises(ValueError, match="length mismatch"):
            precision_recall_f1([0, 1], [1])


class TestRecallByClass:
    def test_benign_is_excluded(self):
        families = ["benign", "ddos", "ddos"]
        assert set(recall_by_class(families, [0, 1, 1])) == {"ddos"}

    def test_per_family_recall_is_computed_independently(self):
        families = ["ddos", "ddos", "botnet", "botnet"]
        result = recall_by_class(families, [1, 1, 0, 0])
        assert result == {"botnet": 0.0, "ddos": 1.0}

    def test_aggregate_can_hide_a_missed_rare_class(self):
        """Why the breakdown is reported alongside the aggregate."""
        families = ["ddos"] * 990 + ["infiltration"] * 10
        predictions = [1] * 990 + [0] * 10
        overall = sum(predictions) / len(predictions)
        by_family = recall_by_class(families, predictions)
        assert overall > 0.98  # looks excellent
        assert by_family["infiltration"] == 0.0  # the severe family is missed entirely

    def test_result_is_sorted_for_stable_reports(self):
        families = ["zeta", "alpha", "mid"]
        assert list(recall_by_class(families, [1, 1, 1])) == ["alpha", "mid", "zeta"]

    def test_length_mismatch_refused(self):
        with pytest.raises(ValueError, match="length mismatch"):
            recall_by_class(["ddos"], [1, 1])


class TestDetectionReport:
    def _report(self, **overrides):
        rng = np.random.default_rng(5)
        y = np.concatenate([np.zeros(800, dtype=int), np.ones(200, dtype=int)])
        scores = np.concatenate([rng.beta(2, 5, 800), rng.beta(5, 2, 200)])
        kwargs = {
            "split": "test",
            "y_true": y,
            "scores": scores,
            "threshold": 0.5,
            "families": ["benign"] * 800 + ["ddos"] * 200,
        }
        kwargs.update(overrides)
        return detection_report(**kwargs)

    def test_report_is_populated(self):
        report = self._report()
        assert report.n_samples == 1000
        assert report.n_positive == 200
        assert report.positive_rate == pytest.approx(0.2)
        assert 0.0 <= report.roc_auc <= 1.0

    def test_f03_gate_reflects_the_auc(self):
        assert self._report().meets_f03_auc == (self._report().roc_auc >= 0.90)

    def test_alert_reduction_is_the_fraction_not_flagged(self):
        # PRD 9.1: "ratio of alerts auto-resolved/clustered vs total ingested".
        y = np.array([0, 0, 0, 1])
        report = detection_report(
            split="t", y_true=y, scores=np.array([0.1, 0.1, 0.1, 0.9]), threshold=0.5
        )
        assert report.alert_reduction == pytest.approx(0.75)

    def test_per_detector_auc_is_included(self):
        rng = np.random.default_rng(5)
        y = np.concatenate([np.zeros(500, dtype=int), np.ones(500, dtype=int)])
        combined = np.concatenate([rng.beta(2, 5, 500), rng.beta(5, 2, 500)])
        report = detection_report(
            split="t",
            y_true=y,
            scores=combined,
            threshold=0.5,
            per_detector_scores={"a": combined, "b": 1.0 - combined},
        )
        assert set(report.per_detector_auc) == {"a", "b"}
        assert report.per_detector_auc["b"] == pytest.approx(1.0 - report.per_detector_auc["a"])

    def test_summary_states_pass_or_fail_explicitly(self):
        summary = self._report().summary()
        assert "ROC-AUC" in summary
        assert "PASS" in summary or "FAIL" in summary
        assert "F-03" in summary

    def test_summary_lists_families_worst_first(self):
        report = detection_report(
            split="t",
            y_true=np.array([0, 0, 1, 1, 1, 1]),
            scores=np.array([0.1, 0.2, 0.9, 0.9, 0.1, 0.1]),
            threshold=0.5,
            families=["benign", "benign", "ddos", "ddos", "botnet", "botnet"],
        )
        summary = report.summary()
        # Worst-performing family should appear before the better one, so an analyst
        # reading the top of the list sees the gap.
        assert summary.index("botnet") < summary.index("ddos")

    def test_families_are_optional(self):
        assert self._report(families=None).per_family_recall == {}

    def test_false_positive_rate_is_measured_on_negatives_only(self):
        # 4 benign, 1 flagged -> FPR 0.25; the flagged attack must not count.
        report = detection_report(
            split="t",
            y_true=np.array([0, 0, 0, 0, 1]),
            scores=np.array([0.1, 0.1, 0.1, 0.9, 0.9]),
            threshold=0.5,
        )
        assert report.false_positive_rate == pytest.approx(0.25)

    def test_a_single_class_split_is_refused_outright(self):
        """Why the no-negatives branch in the FPR calculation is unreachable here.

        ``detection_report`` computes ROC-AUC first, and that refuses a single-class
        split rather than returning a plausible-looking 0.5. So an all-attack split
        never reaches the FPR code. The empty-negatives guard stays as defence in
        depth — ``np.mean`` of an empty slice is nan with a warning — but this test
        records that the guard is currently unreachable, so nobody mistakes it for
        behaviour they can rely on.
        """
        with pytest.raises(ValueError, match="single class present"):
            detection_report(
                split="t",
                y_true=np.array([1, 1]),
                scores=np.array([0.9, 0.1]),
                threshold=0.5,
            )


class TestBaseRateProjection:
    """PRD 9.1's reduction target, projected off the evaluation split's 33% attack rate.

    The tension this exists to resolve: on a split that is one-third attacks,
    *improving recall lowers* the reduction number, because each extra attack caught
    is one more alert reaching a human. Measured on the real pipeline — Part 1's
    linear ensemble scored 60.8% reduction at recall 0.978, Part 2's autoencoder
    59.9% at recall 0.9975. The better detector reports the worse number.
    """

    @staticmethod
    def _report(fpr: float, recall: float) -> object:
        """A report with a chosen FPR and recall, built through the real function."""
        n = 1000
        n_attack = 200
        n_benign = n - n_attack
        scores = np.concatenate(
            [
                np.where(np.arange(n_benign) < round(fpr * n_benign), 0.9, 0.1),
                np.where(np.arange(n_attack) < round(recall * n_attack), 0.9, 0.1),
            ]
        )
        y = np.concatenate([np.zeros(n_benign, dtype=int), np.ones(n_attack, dtype=int)])
        return detection_report(split="t", y_true=y, scores=scores, threshold=0.5)

    def test_projection_matches_the_closed_form(self):
        report = self._report(fpr=0.10, recall=0.99)
        for base_rate in (0.0, 0.001, 0.01, 0.1, 0.5, 1.0):
            expected = 1.0 - ((1 - base_rate) * 0.10 + base_rate * 0.99)
            assert report.reduction_at_base_rate(base_rate) == pytest.approx(expected)

    def test_at_a_realistic_base_rate_the_fpr_budget_dominates(self):
        # 10% FPR on a 99%-benign feed: reduction is ~1 - FPR regardless of recall.
        report = self._report(fpr=0.10, recall=0.9975)
        assert report.reduction_at_base_rate(0.01) == pytest.approx(0.89, abs=0.01)

    def test_better_recall_lowers_split_reduction_but_not_projected_reduction(self):
        """The whole point, asserted directly.

        Two detectors at the same FPR budget, one with better recall. On the split
        the better one looks worse; projected onto a real feed they are within a
        rounding error, and the better one is correctly not penalised.
        """
        worse = self._report(fpr=0.10, recall=0.90)
        better = self._report(fpr=0.10, recall=1.00)

        assert better.alert_reduction < worse.alert_reduction, (
            "the split-level metric is supposed to penalise better recall here; if it "
            "no longer does, this test and reduction_at_base_rate need revisiting"
        )
        projected_gap = abs(
            better.reduction_at_base_rate(0.01) - worse.reduction_at_base_rate(0.01)
        )
        assert projected_gap < 0.002

    def test_prd_9_1_target_is_met_at_a_realistic_base_rate(self):
        report = self._report(fpr=0.10, recall=0.9975)
        assert report.reduction_at_base_rate(0.01) >= 0.60

    @pytest.mark.parametrize("bad", [-0.01, 1.01])
    def test_an_out_of_range_base_rate_is_refused(self, bad: float):
        with pytest.raises(ValueError, match="attack_base_rate"):
            self._report(fpr=0.1, recall=0.9).reduction_at_base_rate(bad)

    def test_summary_reports_both_numbers(self):
        summary = self._report(fpr=0.10, recall=0.9975).summary()
        assert "FPR on benign" in summary
        assert "base rate" in summary
        assert "PRD 9.1" in summary


class TestThreeWaySplit:
    @pytest.fixture
    def labels(self) -> np.ndarray:
        rng = np.random.default_rng(3)
        return (rng.random(4000) < 0.2).astype(int)

    def test_training_split_is_benign_only(self, labels):
        # Semi-supervised novelty detection: one attack in the training set teaches
        # the model that attacks are normal.
        split = three_way_split(labels)
        assert labels[split.train_benign].sum() == 0

    def test_all_splits_are_disjoint(self, labels):
        split = three_way_split(labels)
        train = set(split.train_benign.tolist())
        validation = set(split.validation.tolist())
        test = set(split.test.tolist())
        assert train & validation == set()
        assert train & test == set()
        assert validation & test == set()

    def test_overlapping_splits_are_rejected_at_construction(self):
        with pytest.raises(ValueError, match="overlap"):
            SplitIndices(
                train_benign=np.array([0, 1]),
                validation=np.array([1, 2]),
                test=np.array([3]),
            )

    def test_every_index_is_used_exactly_once(self, labels):
        split = three_way_split(labels)
        combined = np.concatenate([split.train_benign, split.validation, split.test])
        assert sorted(combined.tolist()) == list(range(labels.size))

    def test_both_mixed_splits_contain_positives(self, labels):
        split = three_way_split(labels)
        assert labels[split.validation].sum() > 0
        assert labels[split.test].sum() > 0

    def test_positive_rates_are_comparable_between_validation_and_test(self, labels):
        split = three_way_split(labels)
        validation_rate = labels[split.validation].mean()
        test_rate = labels[split.test].mean()
        assert abs(validation_rate - test_rate) < 0.08

    def test_is_deterministic_for_a_seed(self, labels):
        first = three_way_split(labels, seed=42)
        second = three_way_split(labels, seed=42)
        assert np.array_equal(first.train_benign, second.train_benign)
        assert np.array_equal(first.test, second.test)

    def test_different_seeds_give_different_splits(self, labels):
        assert not np.array_equal(
            three_way_split(labels, seed=1).test, three_way_split(labels, seed=2).test
        )

    def test_indices_are_sorted(self, labels):
        split = three_way_split(labels)
        for indices in (split.train_benign, split.validation, split.test):
            assert list(indices) == sorted(indices)

    def test_sizes_are_reported(self, labels):
        sizes = three_way_split(labels).sizes
        assert sum(sizes.values()) == labels.size
        assert all(value > 0 for value in sizes.values())

    def test_fractions_are_respected(self, labels):
        split = three_way_split(labels, validation_fraction=0.1, test_fraction=0.5)
        benign_total = int((labels == 0).sum())
        assert split.test.size > split.validation.size
        assert split.train_benign.size < benign_total

    def test_empty_labels_refused(self):
        with pytest.raises(ValueError, match="empty label vector"):
            three_way_split([])

    def test_fractions_leaving_no_training_data_refused(self):
        with pytest.raises(ValueError, match="leave benign data"):
            three_way_split([0] * 10 + [1] * 5, validation_fraction=0.5, test_fraction=0.6)

    @pytest.mark.parametrize(
        "kwargs",
        [{"validation_fraction": 0.0}, {"validation_fraction": 1.0}, {"test_fraction": 1.5}],
    )
    def test_invalid_fractions_refused(self, labels, kwargs):
        with pytest.raises(ValueError, match="fractions must be"):
            three_way_split(labels, **kwargs)

    def test_too_few_benign_samples_refused(self):
        with pytest.raises(ValueError, match="at least 3 benign"):
            three_way_split([0, 1, 1, 1])

    def test_too_few_attack_samples_refused(self):
        with pytest.raises(ValueError, match="at least 2 attack"):
            three_way_split([0] * 20 + [1])

    def test_tiny_but_valid_input_still_splits(self):
        split = three_way_split([0, 0, 0, 0, 1, 1])
        assert split.train_benign.size >= 1
        assert split.validation.size >= 1
        assert split.test.size >= 1
