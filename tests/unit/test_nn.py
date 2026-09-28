"""Tests for the NumPy neural-network engine.

The centre of gravity here is :func:`gradient_check`. A hand-written backward
pass that is subtly wrong still trains, still produces a falling loss curve, and
still yields a model that looks fine — it just permanently costs accuracy.
Every layer and every composition of layers is therefore checked against a
central finite difference, plus the specific bugs that hand-rolled backprop
actually ships with: non-accumulated gradients, a transposed matrix product, and
a loss-scaling factor applied on one path only.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.ml.nn import (
    Adam,
    Dense,
    Identity,
    Module,
    Parameter,
    ReLU,
    Sequential,
    Sigmoid,
    Tanh,
    TrainingError,
    gradient_check,
    mse_loss,
    train,
)

SEED = 20260928

# Tolerance for the analytic-vs-numeric comparison. A correct float64 backward
# pass with eps=1e-6 central differences lands around 1e-9; 1e-7 leaves headroom
# for conditioning without leaving room for a missing gradient term, which shows
# up at 1e-2 or worse.
GRAD_TOL = 1e-7


def rng(offset: int = 0) -> np.random.Generator:
    return np.random.default_rng(SEED + offset)


# --------------------------------------------------------------------------- #
# Parameter
# --------------------------------------------------------------------------- #


class TestParameter:
    def test_grad_starts_at_zero_and_matches_value_shape(self) -> None:
        parameter = Parameter(np.ones((3, 4)), "w")
        assert parameter.grad.shape == (3, 4)
        assert np.all(parameter.grad == 0.0)

    def test_constructor_copies_so_caller_arrays_are_not_aliased(self) -> None:
        source = np.ones((2, 2))
        parameter = Parameter(source, "w")
        source[0, 0] = 99.0
        assert parameter.value[0, 0] == 1.0

    def test_value_is_float64_even_from_an_int_array(self) -> None:
        # An int parameter array would make `value -= update` a no-op after
        # truncation, so training would silently do nothing.
        parameter = Parameter(np.ones((2, 2), dtype=np.int32), "w")
        assert parameter.value.dtype == np.float64

    def test_zero_grad_clears_accumulation(self) -> None:
        parameter = Parameter(np.ones(3), "b")
        parameter.grad += 5.0
        parameter.zero_grad()
        assert np.all(parameter.grad == 0.0)


# --------------------------------------------------------------------------- #
# Dense
# --------------------------------------------------------------------------- #


class TestDense:
    def test_forward_is_the_affine_map(self) -> None:
        layer = Dense(3, 2, rng=rng(), name="d")
        layer.weight.value = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        layer.bias.value = np.array([0.5, -0.5])
        x = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(layer.forward(x), [[1.0 + 3.0 + 0.5, 4.0 + 3.0 - 0.5]])

    def test_he_init_is_wider_than_glorot_for_a_fan_out_layer(self) -> None:
        # Not a style preference: He ahead of ReLU preserves activation variance,
        # Glorot ahead of ReLU halves it per layer. Assert they actually differ so
        # a refactor that collapses both to one formula is caught.
        relu_layer = Dense(64, 8, rng=rng(1), gain_for="relu")
        tanh_layer = Dense(64, 8, rng=rng(1), gain_for="tanh")
        assert relu_layer.weight.value.std() > tanh_layer.weight.value.std()

    def test_bias_starts_at_zero(self) -> None:
        assert np.all(Dense(4, 4, rng=rng()).bias.value == 0.0)

    def test_rejects_wrong_input_width_with_a_diagnostic_message(self) -> None:
        layer = Dense(5, 2, rng=rng(), name="enc.0")
        with pytest.raises(ValueError, match="expected 5 input features, got 3"):
            layer.forward(np.zeros((2, 3)))

    def test_rejects_non_2d_input(self) -> None:
        with pytest.raises(ValueError, match="2-D batch"):
            Dense(3, 2, rng=rng()).forward(np.zeros(3))

    def test_rejects_nonpositive_dimensions(self) -> None:
        with pytest.raises(ValueError, match="positive dimensions"):
            Dense(0, 3, rng=rng())

    def test_backward_before_forward_is_an_error_not_a_wrong_answer(self) -> None:
        with pytest.raises(TrainingError, match="before forward"):
            Dense(3, 2, rng=rng()).backward(np.zeros((1, 2)))

    def test_unknown_gain_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown gain_for"):
            Dense(3, 2, rng=rng(), gain_for="selu")

    def test_gradients_accumulate_across_two_backward_calls(self) -> None:
        """The GraphSAGE bug class: one weight reached twice per backward pass.

        If ``backward`` assigned instead of accumulating, the second call would
        overwrite the first and the summed gradient term would vanish — silently,
        with the model still training.
        """
        layer = Dense(3, 2, rng=rng(2))
        x = rng(3).normal(size=(4, 3))
        upstream = rng(4).normal(size=(4, 2))

        layer.forward(x)
        layer.backward(upstream)
        once = np.array(layer.weight.grad, copy=True)

        layer.forward(x)
        layer.backward(upstream)
        twice = layer.weight.grad

        np.testing.assert_allclose(twice, 2.0 * once, rtol=1e-12)

    def test_bias_gradient_sums_over_the_batch(self) -> None:
        layer = Dense(2, 3, rng=rng(5))
        layer.forward(np.zeros((7, 2)))
        upstream = rng(6).normal(size=(7, 3))
        layer.backward(upstream)
        np.testing.assert_allclose(layer.bias.grad, upstream.sum(axis=0), rtol=1e-12)


# --------------------------------------------------------------------------- #
# Activations
# --------------------------------------------------------------------------- #


class TestActivations:
    def test_relu_zeroes_negatives_and_passes_positives(self) -> None:
        out = ReLU().forward(np.array([[-2.0, 0.0, 3.0]]))
        np.testing.assert_allclose(out, [[0.0, 0.0, 3.0]])

    def test_relu_subgradient_at_zero_is_zero(self) -> None:
        layer = ReLU()
        layer.forward(np.array([[0.0]]))
        assert layer.backward(np.array([[1.0]]))[0, 0] == 0.0

    def test_tanh_derivative_matches_the_closed_form(self) -> None:
        layer = Tanh()
        x = np.array([[-1.0, 0.0, 2.0]])
        out = layer.forward(x)
        np.testing.assert_allclose(layer.backward(np.ones_like(x)), 1.0 - out**2, rtol=1e-12)

    def test_sigmoid_is_stable_at_extreme_inputs(self) -> None:
        """A naive ``1/(1+exp(-x))`` overflows at x=-800 and returns nan.

        Sigmoid sits on the diffusion model's output path where a diverging
        pre-activation is entirely possible, and nan there poisons the whole
        batch rather than saturating harmlessly.
        """
        out = Sigmoid().forward(np.array([[-800.0, 0.0, 800.0]]))
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out, [[0.0, 0.5, 1.0]], atol=1e-12)

    def test_sigmoid_derivative_matches_the_closed_form(self) -> None:
        layer = Sigmoid()
        x = np.array([[-1.5, 0.3, 2.2]])
        out = layer.forward(x)
        np.testing.assert_allclose(
            layer.backward(np.ones_like(x)), out * (1.0 - out), rtol=1e-12
        )

    def test_identity_is_a_passthrough_in_both_directions(self) -> None:
        layer = Identity()
        x = np.array([[1.0, -2.0]])
        np.testing.assert_allclose(layer.forward(x), x)
        np.testing.assert_allclose(layer.backward(x), x)

    @pytest.mark.parametrize("factory", [ReLU, Tanh, Sigmoid])
    def test_backward_before_forward_is_an_error(self, factory: type[Module]) -> None:
        with pytest.raises(TrainingError, match="before forward"):
            factory().backward(np.zeros((1, 1)))  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# Sequential
# --------------------------------------------------------------------------- #


class TestSequential:
    def test_parameters_are_collected_in_forward_order(self) -> None:
        net = Sequential(
            Dense(4, 3, rng=rng(), name="a"),
            Tanh(),
            Dense(3, 2, rng=rng(), name="b"),
        )
        assert [p.name for p in net.parameters()] == ["a.W", "a.b", "b.W", "b.b"]

    def test_n_parameters_counts_every_entry(self) -> None:
        net = Sequential(Dense(4, 3, rng=rng()), Tanh(), Dense(3, 2, rng=rng()))
        assert net.n_parameters() == (4 * 3 + 3) + (3 * 2 + 2)

    def test_empty_sequential_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one layer"):
            Sequential()

    def test_state_round_trips_exactly(self) -> None:
        net = Sequential(Dense(3, 3, rng=rng(7)), Tanh(), Dense(3, 3, rng=rng(8)))
        snapshot = net.state()
        for parameter in net.parameters():
            parameter.value += 1.0
        net.load_state(snapshot)
        for parameter, saved in zip(net.parameters(), snapshot, strict=True):
            np.testing.assert_array_equal(parameter.value, saved)

    def test_state_snapshot_is_a_copy_not_a_view(self) -> None:
        """Early stopping restores this snapshot after further training steps.

        If ``state()`` returned views, the "best" weights would keep tracking the
        live parameters and the restore at the end of training would be a no-op —
        early stopping would appear to work and do nothing.
        """
        net = Sequential(Dense(2, 2, rng=rng(9)))
        snapshot = net.state()
        net.parameters()[0].value += 5.0
        assert not np.allclose(snapshot[0], net.parameters()[0].value)

    def test_load_state_rejects_a_wrong_length_state(self) -> None:
        net = Sequential(Dense(2, 2, rng=rng()))
        with pytest.raises(ValueError, match="2 parameters"):
            net.load_state([np.zeros((2, 2))])

    def test_load_state_rejects_a_shape_mismatch(self) -> None:
        net = Sequential(Dense(2, 2, rng=rng()))
        with pytest.raises(ValueError, match="shape mismatch"):
            net.load_state([np.zeros((3, 3)), np.zeros(2)])


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #


class TestMseLoss:
    def test_value_is_the_mean_over_all_elements(self) -> None:
        pred = np.array([[1.0, 2.0], [3.0, 4.0]])
        target = np.array([[0.0, 0.0], [0.0, 0.0]])
        loss, _ = mse_loss(pred, target)
        assert loss == pytest.approx((1 + 4 + 9 + 16) / 4)

    def test_gradient_is_the_analytic_derivative(self) -> None:
        pred = np.array([[1.0, 2.0]])
        target = np.array([[0.5, 0.5]])
        _, grad = mse_loss(pred, target)
        np.testing.assert_allclose(grad, 2.0 * (pred - target) / pred.size, rtol=1e-12)

    def test_scale_is_independent_of_feature_count(self) -> None:
        """Averaging over elements, not summing over features.

        Otherwise a learning rate tuned on a 40-column spec becomes 80x too large
        the day someone adds a 40-column block, and the failure presents as
        "training diverged after we added features".
        """
        narrow = mse_loss(np.full((8, 5), 2.0), np.zeros((8, 5)))[0]
        wide = mse_loss(np.full((8, 500), 2.0), np.zeros((8, 500)))[0]
        assert narrow == pytest.approx(wide)

    def test_shape_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            mse_loss(np.zeros((2, 3)), np.zeros((2, 4)))

    def test_empty_batch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty batch"):
            mse_loss(np.zeros((0, 3)), np.zeros((0, 3)))


# --------------------------------------------------------------------------- #
# Gradient checking — the load-bearing tests
# --------------------------------------------------------------------------- #


def _batch(n: int, d: int, offset: int) -> np.ndarray:
    return np.random.default_rng(SEED + offset).normal(size=(n, d))


class TestGradientCheck:
    @pytest.mark.parametrize(
        ("activation", "gain"),
        [(Tanh, "tanh"), (Sigmoid, "sigmoid"), (ReLU, "relu"), (Identity, "linear")],
    )
    def test_every_activation_backprops_exactly(
        self, activation: type[Module], gain: str
    ) -> None:
        net = Sequential(
            Dense(5, 7, rng=rng(11), gain_for=gain, name="h1"),
            activation(),  # type: ignore[call-arg]
            Dense(7, 3, rng=rng(12), gain_for="linear", name="out"),
        )
        error = gradient_check(net, _batch(6, 5, 13), _batch(6, 3, 14), rng=rng(15))
        assert error < GRAD_TOL, f"{activation.__name__} gradient error {error:.3e}"

    def test_a_deep_stack_backprops_exactly(self) -> None:
        """Four hidden layers: the depth at which a missing term stops being obvious."""
        net = Sequential(
            Dense(6, 8, rng=rng(16), name="a"),
            Tanh(),
            Dense(8, 4, rng=rng(17), name="b"),
            Tanh(),
            Dense(4, 8, rng=rng(18), name="c"),
            Tanh(),
            Dense(8, 6, rng=rng(19), gain_for="linear", name="d"),
        )
        error = gradient_check(net, _batch(5, 6, 20), _batch(5, 6, 21), rng=rng(22))
        assert error < GRAD_TOL, f"deep stack gradient error {error:.3e}"

    def test_a_single_dense_layer_backprops_exactly_and_exhaustively(self) -> None:
        # Small enough to check every entry rather than a sample.
        net = Sequential(Dense(3, 2, rng=rng(23), gain_for="linear"))
        error = gradient_check(
            net, _batch(4, 3, 24), _batch(4, 2, 25), max_checks_per_param=10_000
        )
        assert error < GRAD_TOL

    def test_relu_check_avoids_the_kink_by_construction(self) -> None:
        """ReLU is non-differentiable at 0, where a finite difference straddles the kink.

        With random weights and continuous inputs the probability of a
        pre-activation landing within ``eps`` of zero is negligible, so the check
        is valid — but it is worth recording *why* it is valid rather than
        discovering later that a fixed integer input made it flaky.
        """
        net = Sequential(Dense(4, 6, rng=rng(26), gain_for="relu"), ReLU())
        x = _batch(8, 4, 27)
        pre_activation = x @ net.layers[0].parameters()[0].value  # type: ignore[attr-defined]
        assert np.min(np.abs(pre_activation)) > 1e-4, "an input landed on the ReLU kink"
        assert gradient_check(net, x, _batch(8, 6, 28), rng=rng(29)) < GRAD_TOL

    # --- the check must be able to fail ------------------------------------ #

    def test_check_catches_a_transposed_matrix_product(self) -> None:
        """A deliberately broken layer must fail the check.

        A gradient check that cannot fail is decoration. This injects the most
        common real bug — the wrong operand order in the weight-gradient product,
        which is shape-compatible for a square layer and therefore raises nothing.
        """

        class TransposedDense(Dense):
            def backward(self, grad_out: np.ndarray) -> np.ndarray:
                assert self._input is not None
                self.weight.grad += grad_out.T @ self._input  # wrong: should be _input.T @ g
                self.bias.grad += grad_out.sum(axis=0)
                return grad_out @ self.weight.value.T

        net = Sequential(TransposedDense(4, 4, rng=rng(30), gain_for="linear"))
        error = gradient_check(net, _batch(5, 4, 31), _batch(5, 4, 32), rng=rng(33))
        assert error > 1e-3, "the check failed to notice a transposed gradient product"

    def test_check_catches_a_dropped_gradient_term(self) -> None:
        """Activation derivative omitted — the "it still trains" bug."""

        class LinearisedTanh(Tanh):
            def backward(self, grad_out: np.ndarray) -> np.ndarray:
                return np.asarray(grad_out, dtype=np.float64)  # forgot (1 - tanh^2)

        net = Sequential(
            Dense(4, 5, rng=rng(34), name="h"),
            LinearisedTanh(),
            Dense(5, 2, rng=rng(35), gain_for="linear", name="o"),
        )
        error = gradient_check(net, _batch(6, 4, 36), _batch(6, 2, 37), rng=rng(38))
        assert error > 1e-3, "the check failed to notice a dropped activation derivative"

    def test_check_catches_a_missing_bias_gradient(self) -> None:
        class BiaslessGradDense(Dense):
            def backward(self, grad_out: np.ndarray) -> np.ndarray:
                assert self._input is not None
                self.weight.grad += self._input.T @ grad_out
                return grad_out @ self.weight.value.T  # bias gradient never accumulated

        net = Sequential(BiaslessGradDense(3, 3, rng=rng(39), gain_for="linear"))
        assert gradient_check(net, _batch(5, 3, 40), _batch(5, 3, 41), rng=rng(42)) > 1e-3

    def test_check_restores_every_parameter_it_perturbs(self) -> None:
        """The check must leave the model exactly as it found it.

        It perturbs weights in place. If a probe failed to restore, the model
        returned to the caller would be the perturbed one — and since the damage
        is ~1e-6 per entry, nothing downstream would ever notice.
        """
        net = Sequential(Dense(4, 4, rng=rng(43)), Tanh(), Dense(4, 4, rng=rng(44)))
        before = net.state()
        gradient_check(net, _batch(5, 4, 45), _batch(5, 4, 46), rng=rng(47))
        for parameter, original in zip(net.parameters(), before, strict=True):
            np.testing.assert_array_equal(parameter.value, original)

    def test_check_zeroes_gradients_first_so_it_is_repeatable(self) -> None:
        """Two consecutive checks must agree.

        If the check did not zero gradients, the second call would compare a
        doubly-accumulated analytic gradient against an unchanged numeric one and
        report a large error — a passing test that fails when run twice.
        """
        net = Sequential(Dense(4, 3, rng=rng(48)), Tanh(), Dense(3, 2, rng=rng(49)))
        x, y = _batch(5, 4, 50), _batch(5, 2, 51)
        first = gradient_check(net, x, y, rng=rng(52))
        second = gradient_check(net, x, y, rng=rng(52))
        assert first == pytest.approx(second)
        assert second < GRAD_TOL


# --------------------------------------------------------------------------- #
# Adam
# --------------------------------------------------------------------------- #


class TestAdam:
    def test_first_step_is_approximately_lr_in_the_descent_direction(self) -> None:
        """Bias correction makes step one ~lr regardless of gradient magnitude.

        That is the property that makes Adam's learning rate interpretable, and
        it only holds if both moment estimates are bias-corrected. Correcting
        only ``m`` (an easy omission) makes step one about ``lr * sqrt(1-b2)``,
        i.e. 30x too small, which reads as "Adam is slow" rather than as a bug.
        """
        parameter = Parameter(np.array([1.0]), "w")
        optimizer = Adam([parameter], lr=0.1)
        parameter.grad[:] = 7.0  # magnitude is irrelevant to the first step size
        optimizer.step()
        assert parameter.value[0] == pytest.approx(1.0 - 0.1, abs=1e-6)

    def test_step_descends_for_either_gradient_sign(self) -> None:
        for sign in (1.0, -1.0):
            parameter = Parameter(np.array([0.0]), "w")
            optimizer = Adam([parameter], lr=0.05)
            parameter.grad[:] = sign
            optimizer.step()
            assert np.sign(parameter.value[0]) == -sign

    def test_minimises_a_quadratic(self) -> None:
        # f(w) = (w - 3)^2, gradient 2(w - 3), minimum at 3.
        parameter = Parameter(np.array([0.0]), "w")
        optimizer = Adam([parameter], lr=0.1)
        for _ in range(500):
            optimizer.zero_grad()
            parameter.grad[:] = 2.0 * (parameter.value - 3.0)
            optimizer.step()
        assert parameter.value[0] == pytest.approx(3.0, abs=1e-3)

    @pytest.mark.parametrize("gradient", [1000.0, 1.0, 0.001])
    def test_weight_decay_is_decoupled_from_the_gradient_scale(self, gradient: float) -> None:
        """AdamW, not Adam-with-L2-in-the-gradient.

        With coupled decay the shrinkage passes through the ``1/sqrt(v)``
        normaliser, so a parameter whose gradient history is large gets almost no
        regularisation and one with a tiny history gets an enormous amount — the
        regularisation strength ends up inversely proportional to each weight's
        own gradient noise.

        Measured as the *difference* between two otherwise identical optimisers,
        one with decay and one without. That isolates the decay term exactly,
        which comparing two parameters against each other cannot: Adam's ``eps``
        makes ``m_hat/sqrt(v_hat)`` differ from 1.0 by ``eps/|g|``, so a
        large-gradient and a small-gradient parameter legitimately move by
        slightly different amounts for reasons that have nothing to do with decay.
        """
        lr, decay, start = 0.01, 0.5, 2.0

        plain = Parameter(np.array([start]), "w")
        plain.grad[:] = gradient
        Adam([plain], lr=lr, weight_decay=0.0).step()

        decayed = Parameter(np.array([start]), "w")
        decayed.grad[:] = gradient
        Adam([decayed], lr=lr, weight_decay=decay).step()

        extra_shrinkage = float(plain.value[0] - decayed.value[0])
        assert extra_shrinkage == pytest.approx(lr * decay * start, rel=1e-12)

    def test_zero_weight_decay_leaves_the_value_untouched_at_zero_gradient(self) -> None:
        parameter = Parameter(np.array([5.0]), "w")
        Adam([parameter], lr=0.1, weight_decay=0.0).step()
        assert parameter.value[0] == pytest.approx(5.0)

    def test_non_finite_gradient_raises_instead_of_poisoning_the_weights(self) -> None:
        parameter = Parameter(np.array([1.0]), "w")
        optimizer = Adam([parameter], lr=0.1)
        parameter.grad[:] = np.nan
        with pytest.raises(TrainingError, match="non-finite gradient in w"):
            optimizer.step()
        assert np.isfinite(parameter.value[0]), "weights were corrupted before the raise"

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"lr": 0.0}, "lr must be positive"),
            ({"beta1": 1.0}, "must lie in"),
            ({"beta2": -0.1}, "must lie in"),
            ({"weight_decay": -1.0}, "non-negative"),
        ],
    )
    def test_invalid_hyperparameters_are_rejected(
        self, kwargs: dict[str, float], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            Adam([Parameter(np.zeros(1), "w")], **kwargs)  # type: ignore[arg-type]

    def test_empty_parameter_list_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one parameter"):
            Adam([])


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #


def _autoencode(rows: np.ndarray, _: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Noiseless reconstruction target — deterministic, for loop-mechanics tests."""
    return rows, rows


def _noisy_autoencode(
    rows: np.ndarray, generator: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    return rows + generator.normal(0.0, 0.3, size=rows.shape), rows


def _small_net(d: int, offset: int) -> Sequential:
    return Sequential(
        Dense(d, 6, rng=rng(offset), name="enc"),
        Tanh(),
        Dense(6, d, rng=rng(offset + 1), gain_for="linear", name="dec"),
    )


class TestTrain:
    def test_loss_decreases_on_a_learnable_problem(self) -> None:
        data = rng(60).normal(size=(200, 4)) @ rng(61).normal(size=(4, 4))
        history = train(
            _small_net(4, 62), data, _autoencode, rng=rng(63), epochs=40, batch_size=32
        )
        assert history.train_loss[-1] < history.train_loss[0] * 0.5
        assert history.converged

    def test_is_bit_identical_across_runs_with_the_same_seed(self) -> None:
        """PRD Section 8.3: the demo and the recorded backup must match exactly."""
        data = rng(64).normal(size=(120, 4))
        results = []
        for _ in range(2):
            net = _small_net(4, 65)
            train(net, data, _noisy_autoencode, rng=rng(66), epochs=6, batch_size=16)
            results.append(net.state())
        for left, right in zip(results[0], results[1], strict=True):
            np.testing.assert_array_equal(left, right)

    def test_a_different_seed_gives_a_different_model(self) -> None:
        # Guards against the reproducibility test above passing because the seed
        # is ignored entirely.
        data = rng(67).normal(size=(120, 4))
        states = []
        for seed_offset in (68, 69):
            net = _small_net(4, 65)
            train(net, data, _noisy_autoencode, rng=rng(seed_offset), epochs=6)
            states.append(net.state())
        assert not np.allclose(states[0][0], states[1][0])

    def test_early_stopping_restores_the_best_weights_not_the_last(self) -> None:
        """Stopping and then keeping the final, worse weights is a classic silent bug."""
        data = rng(70).normal(size=(150, 4))
        validation = rng(71).normal(size=(60, 4))
        net = _small_net(4, 72)
        history = train(
            net,
            data,
            _noisy_autoencode,
            rng=rng(73),
            epochs=80,
            patience=3,
            validation_data=validation,
        )
        assert history.validation_loss, "no validation curve was recorded"
        best = min(history.validation_loss)
        assert history.best_validation_loss == pytest.approx(best)
        # The restored model must reproduce the best loss on the same frozen
        # validation corruption, not the final epoch's loss.
        assert history.best_epoch == history.validation_loss.index(best)

    def test_early_stopping_fires_when_improvement_drops_below_min_delta(self) -> None:
        """``min_delta`` is what makes patience fire on a slow crawl.

        Written this way because the obvious version of this test — "train on
        constant data, it will plateau" — does not plateau. Measured: 200 epochs
        on a constant 4-column row and validation loss was *still* falling at
        epoch 199 (8.12 -> 2.08, best_epoch=199). A tiny tanh net converges
        asymptotically, not to a floor, so patience alone never fires and only
        ``min_delta`` distinguishes "still learning" from "crawling".
        """
        data = np.tile(np.array([[1.0, 2.0, 3.0, 4.0]]), (80, 1))
        history = train(
            _small_net(4, 74),
            data,
            _autoencode,
            rng=rng(75),
            epochs=200,
            patience=2,
            min_delta=1.0,  # demand a full unit of improvement per epoch
            validation_data=data[:20],
        )
        assert history.stopped_early
        assert history.epochs_run < 200

    def test_a_slow_crawl_is_not_mistaken_for_a_plateau(self) -> None:
        """The complement: at the default ``min_delta`` a slow improver keeps going.

        Pins the measured behaviour above, so a future change to the stopping
        rule that starts cutting genuinely-still-learning runs short is caught.
        """
        data = np.tile(np.array([[1.0, 2.0, 3.0, 4.0]]), (80, 1))
        history = train(
            _small_net(4, 74),
            data,
            _autoencode,
            rng=rng(75),
            epochs=40,
            patience=2,
            validation_data=data[:20],
        )
        assert not history.stopped_early
        assert history.epochs_run == 40
        assert history.validation_loss[-1] < history.validation_loss[0]

    def test_validation_noise_is_frozen_across_epochs(self) -> None:
        """Early stopping on a noisy validation curve stops on a lucky draw.

        With a *frozen* untrained model, every epoch's validation loss must be
        identical — because the only thing that could vary is the corruption. If
        the loop drew fresh noise each epoch the losses would differ.
        """

        class FrozenNet(Sequential):
            def backward(self, grad_out: np.ndarray) -> np.ndarray:
                return np.zeros_like(grad_out)  # no gradients, so no weight change

        net = FrozenNet(Dense(4, 4, rng=rng(76), gain_for="linear"))
        frozen_before = net.state()
        history = train(
            net,
            rng(77).normal(size=(60, 4)),
            _noisy_autoencode,
            rng=rng(78),
            epochs=5,
            patience=99,
            validation_data=rng(79).normal(size=(40, 4)),
        )
        np.testing.assert_allclose(net.state()[0], frozen_before[0], atol=1e-12)
        assert len(set(np.round(history.validation_loss, 12))) == 1, (
            f"validation loss varied on a frozen model: {history.validation_loss}"
        )

    def test_without_validation_data_no_weights_are_restored(self) -> None:
        data = rng(80).normal(size=(100, 4))
        net = _small_net(4, 81)
        history = train(net, data, _autoencode, rng=rng(82), epochs=10)
        assert history.validation_loss == []
        assert history.best_epoch == -1

    def test_batch_size_larger_than_the_dataset_is_clamped(self) -> None:
        data = rng(83).normal(size=(7, 4))
        history = train(
            _small_net(4, 84), data, _autoencode, rng=rng(85), epochs=3, batch_size=4096
        )
        assert history.epochs_run == 3

    def test_a_ragged_final_batch_is_handled(self) -> None:
        # 130 rows at batch 32 leaves a final batch of 2.
        data = rng(86).normal(size=(130, 4))
        history = train(
            _small_net(4, 87), data, _autoencode, rng=rng(88), epochs=3, batch_size=32
        )
        assert all(np.isfinite(history.train_loss))

    def test_divergence_raises_rather_than_returning_a_nan_model(self) -> None:
        """Overflow in the loss must raise, not return a model full of nan.

        Reaching this state takes absurd inputs on purpose. An earlier version of
        this test used ``lr=1e6`` on data scaled by 1e8 and *did not* diverge,
        which is worth recording: Adam normalises its step to roughly ``lr``
        regardless of gradient magnitude, so a huge gradient cannot produce a huge
        step the way plain SGD can. Adam diverging is therefore a symptom of the
        *loss* overflowing, not of the gradient being large — so that is what this
        test provokes.
        """
        # The overflow is the point of the test, so do not let numpy warn about it.
        with np.errstate(over="ignore", invalid="ignore"), pytest.raises(
            TrainingError, match="diverged"
        ):
            train(
                _small_net(4, 89),
                rng(90).normal(size=(80, 4)) * 1e200,
                _autoencode,
                rng=rng(91),
                epochs=5,
            )

    def test_a_large_learning_rate_alone_does_not_diverge(self) -> None:
        """The measured counterpart: Adam's normalised step is hard to blow up.

        lr=1e6 on data scaled by 1e8 trains to a terrible model without ever
        producing a non-finite value. Pinned so nobody "fixes" the divergence
        guard by making it fire on large-but-finite losses, which would start
        rejecting legitimate early epochs on unscaled features.
        """
        history = train(
            _small_net(4, 89),
            rng(90).normal(size=(80, 4)) * 1e8,
            _autoencode,
            rng=rng(91),
            epochs=5,
            lr=1e6,
        )
        assert all(np.isfinite(history.train_loss))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"epochs": 0}, "epochs must be"),
            ({"batch_size": 0}, "batch_size must be"),
            ({"patience": 0}, "patience must be"),
        ],
    )
    def test_invalid_loop_settings_are_rejected(
        self, kwargs: dict[str, int], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            train(
                _small_net(4, 92),
                rng(93).normal(size=(10, 4)),
                _autoencode,
                rng=rng(94),
                **kwargs,  # type: ignore[arg-type]
            )

    def test_empty_and_wrong_shaped_data_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty data matrix"):
            train(_small_net(4, 95), np.zeros((0, 4)), _autoencode, rng=rng(96))
        with pytest.raises(ValueError, match="2-D data matrix"):
            train(_small_net(4, 97), np.zeros(4), _autoencode, rng=rng(98))

    def test_converged_requires_a_relative_improvement_not_just_any(self) -> None:
        """Float64 summation order alone makes the loss jitter by ~1e-16.

        With ``batch_size >= n_rows`` every epoch sums the same values in a
        different order (the shuffle), and float addition is not associative. A
        bare ``last < first`` convergence test therefore passes at a learning rate
        of 1e-20, where the weights are provably frozen. Measured, not theorised —
        it is how ``test_deep.py``'s non-convergence test first failed.
        """
        data = rng(103).normal(size=(40, 4))
        history = train(
            _small_net(4, 104),
            data,
            _autoencode,
            rng=rng(105),
            epochs=6,
            batch_size=10_000,  # one full batch per epoch: only the order changes
            lr=1e-20,
        )
        assert not history.converged
        # The jitter is real and tiny — confirm that is what we are rejecting,
        # rather than a genuine (if small) improvement.
        first, last = history.train_loss[0], history.train_loss[-1]
        assert abs(first - last) / first < 1e-12

    def test_converged_is_true_for_real_training(self) -> None:
        history = train(
            _small_net(4, 106),
            rng(107).normal(size=(200, 4)),
            _autoencode,
            rng=rng(108),
            epochs=30,
            batch_size=32,
        )
        assert history.converged

    def test_history_summary_is_informative(self) -> None:
        history = train(
            _small_net(4, 99),
            rng(100).normal(size=(60, 4)),
            _autoencode,
            rng=rng(101),
            epochs=4,
            validation_data=rng(102).normal(size=(20, 4)),
        )
        text = history.summary()
        assert "epoch(s)" in text and "params" in text and "best val" in text
