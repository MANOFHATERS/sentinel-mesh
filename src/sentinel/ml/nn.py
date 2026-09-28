"""A small, exact, gradient-checked neural-network engine (NumPy).

Three Part 2 components need a trainable nonlinear model: the denoising
autoencoder (PRD Section 5.5.2), the GraphSAGE risk propagator (Section 5.5.3)
and the tabular diffusion denoiser (Section 5.5.5). The PRD names PyTorch for
these. This module is the deliberate substitute, and the reasoning is worth
recording because it is a real engineering trade, not a shortcut.

Why not PyTorch
---------------
*   **The models are small.** The autoencoder is ``40 -> 64 -> 16 -> 64 -> 40``
    over ~20,000 rows; GraphSAGE is two layers over ~500 nodes. Both train in
    seconds on one CPU core. Autograd buys nothing here except a 2 GB wheel and
    a CUDA-version conversation.
*   **Part 1 installs with four light dependencies and its evaluation needs no
    downloads.** That property is worth protecting: it is what lets anyone clone
    the repo and reproduce the F-03 number in one command. ``torch-geometric``
    in particular needs compiled extensions pinned against a specific torch
    build, and on Windows that is a coin flip.
*   **Correctness is *more* provable this way.** With autograd you test the model;
    here :func:`gradient_check` compares every analytic gradient against a central
    finite difference, so the chain rule itself is under test
    (``tests/unit/test_nn.py``). A hand-written backward pass that is *nearly*
    right trains to a plausible-looking loss curve and quietly costs you accuracy
    forever — so it is checked to 1e-8 relative error, not eyeballed.

The escape hatch is the point: :class:`~sentinel.ml.anomaly.AnomalyDetector` is a
Protocol, so a torch-backed detector can be dropped in later with no caller
changes. What must not happen is the platform's reproducibility depending on a
wheel that may not install.

Determinism
-----------
Everything is ``float64`` and every stochastic decision (initialisation, shuffle,
dropout-style corruption, early-stopping snapshots) draws from an explicitly
passed :class:`numpy.random.Generator`. Two runs with the same seed produce
bit-identical weights, because a demo that is only *usually* reproducible is not
reproducible (PRD Section 8.3, hours 53-56).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from sentinel.core.errors import SentinelError

__all__ = [
    "MIN_RELATIVE_IMPROVEMENT",
    "Adam",
    "Dense",
    "Identity",
    "LossFn",
    "Module",
    "Parameter",
    "ReLU",
    "Sequential",
    "Sigmoid",
    "Tanh",
    "TrainHistory",
    "TrainingError",
    "bce_with_logits_loss",
    "gradient_check",
    "mse_loss",
    "sigmoid",
    "train",
]

DTYPE: Final = np.float64

#: Minimum *relative* fall in training loss for :attr:`TrainHistory.converged` to
#: be True. See that property for why an absolute ``last < first`` comparison is
#: not sufficient (float64 summation order alone moves the loss by ~1e-16).
MIN_RELATIVE_IMPROVEMENT: Final = 1e-6


class TrainingError(SentinelError):
    """Training diverged, or was configured in a way that cannot converge."""


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #


class Parameter:
    """One trainable array plus its accumulated gradient.

    Gradients *accumulate* into ``grad`` rather than being assigned, so a layer
    reached twice in one backward pass (which is exactly what happens in
    GraphSAGE, where the same weight matrix is applied to every node's
    neighbourhood) sums its contributions instead of keeping only the last one.
    That is the single most common hand-rolled-backprop bug, and it is silent:
    the model still trains, just down a gradient term.

    The consequence is that :meth:`Module.zero_grad` is mandatory before every
    backward pass. :func:`train` and :func:`gradient_check` both do it; anything
    else driving a module by hand must too.
    """

    __slots__ = ("grad", "name", "value")

    def __init__(self, value: np.ndarray, name: str) -> None:
        self.value = np.array(value, dtype=DTYPE, copy=True)
        self.grad = np.zeros_like(self.value)
        self.name = name

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    @property
    def size(self) -> int:
        return int(self.value.size)

    def zero_grad(self) -> None:
        self.grad.fill(0.0)

    def __repr__(self) -> str:
        return f"Parameter({self.name!r}, shape={self.value.shape})"


# --------------------------------------------------------------------------- #
# Modules
# --------------------------------------------------------------------------- #


class Module(ABC):
    """Forward/backward with explicit gradients. No autograd, no magic.

    Contract every subclass must honour:

    *   :meth:`forward` caches whatever :meth:`backward` needs, on ``self``.
    *   :meth:`backward` receives ``dL/d(output)`` and returns ``dL/d(input)``,
        accumulating into every :class:`Parameter` it owns.
    *   Both are pure with respect to parameter *values*: neither writes
        ``param.value``. Only the optimiser does that.
    """

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def backward(self, grad_out: np.ndarray) -> np.ndarray: ...

    def parameters(self) -> list[Parameter]:
        return []

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def n_parameters(self) -> int:
        return sum(p.size for p in self.parameters())

    def state(self) -> list[np.ndarray]:
        """Snapshot of every parameter value, for early-stopping restore."""
        return [np.array(p.value, copy=True) for p in self.parameters()]

    def load_state(self, state: Sequence[np.ndarray]) -> None:
        params = self.parameters()
        if len(state) != len(params):
            raise ValueError(
                f"state has {len(state)} arrays but module has {len(params)} parameters"
            )
        for parameter, value in zip(params, state, strict=True):
            if value.shape != parameter.value.shape:
                raise ValueError(
                    f"shape mismatch restoring {parameter.name}: "
                    f"{value.shape} into {parameter.value.shape}"
                )
            parameter.value = np.array(value, dtype=DTYPE, copy=True)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class Dense(Module):
    """Affine layer ``y = x @ W + b``.

    Initialisation is chosen by the *activation that follows*, passed as
    ``gain_for``. He (``2/fan_in``) for ReLU, Glorot (``2/(fan_in+fan_out)``)
    for the saturating activations. This is not decoration: with Glorot init
    ahead of a deep ReLU stack the activation variance halves per layer, and with
    He init ahead of tanh the pre-activations start saturated and the gradient is
    already dead at epoch zero. Getting it wrong costs accuracy that then looks
    like a modelling problem.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        *,
        rng: np.random.Generator,
        gain_for: str = "tanh",
        name: str = "dense",
    ) -> None:
        if n_in < 1 or n_out < 1:
            raise ValueError(f"Dense needs positive dimensions, got {n_in} -> {n_out}")
        if gain_for == "relu":
            scale = np.sqrt(2.0 / n_in)
        elif gain_for in ("tanh", "sigmoid", "linear"):
            scale = np.sqrt(2.0 / (n_in + n_out))
        else:
            raise ValueError(f"unknown gain_for {gain_for!r}")
        self.weight = Parameter(rng.normal(0.0, scale, size=(n_in, n_out)), f"{name}.W")
        self.bias = Parameter(np.zeros(n_out), f"{name}.b")
        self._input: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        matrix = np.asarray(x, dtype=DTYPE)
        if matrix.ndim != 2:
            raise ValueError(f"Dense expects a 2-D batch, got shape {matrix.shape}")
        if matrix.shape[1] != self.weight.value.shape[0]:
            raise ValueError(
                f"{self.weight.name}: expected {self.weight.value.shape[0]} input features, "
                f"got {matrix.shape[1]}"
            )
        self._input = matrix
        return matrix @ self.weight.value + self.bias.value

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._input is None:
            raise TrainingError("Dense.backward called before forward")
        grad = np.asarray(grad_out, dtype=DTYPE)
        self.weight.grad += self._input.T @ grad
        self.bias.grad += grad.sum(axis=0)
        return grad @ self.weight.value.T

    def parameters(self) -> list[Parameter]:
        return [self.weight, self.bias]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function.

    The textbook ``1 / (1 + exp(-x))`` overflows for large negative ``x`` and
    returns nan. Evaluating the algebraically equal ``exp(x) / (1 + exp(x))`` on
    that branch keeps every ``exp`` argument non-positive, so it saturates to 0.0
    instead of producing nan — and one nan in a batch poisons the whole update.
    """
    z = np.asarray(x, dtype=DTYPE)
    out = np.empty_like(z)
    positive = z >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


class _Activation(Module):
    """Elementwise activation. Caches whichever of input/output backward needs."""

    def __init__(self) -> None:
        self._cache: np.ndarray | None = None


class ReLU(_Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        matrix = np.asarray(x, dtype=DTYPE)
        self._cache = matrix
        return np.maximum(matrix, 0.0)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise TrainingError("ReLU.backward called before forward")
        # Subgradient 0 at exactly x == 0. Either convention is defensible; this
        # one is what every framework uses, and consistency matters because
        # gradient_check compares against a finite difference that straddles it.
        return np.asarray(grad_out, dtype=DTYPE) * (self._cache > 0.0)


class Tanh(_Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        out = np.tanh(np.asarray(x, dtype=DTYPE))
        self._cache = out
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise TrainingError("Tanh.backward called before forward")
        return np.asarray(grad_out, dtype=DTYPE) * (1.0 - np.square(self._cache))


class Sigmoid(_Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        out = sigmoid(x)
        self._cache = out
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise TrainingError("Sigmoid.backward called before forward")
        return np.asarray(grad_out, dtype=DTYPE) * self._cache * (1.0 - self._cache)


class Identity(_Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=DTYPE)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return np.asarray(grad_out, dtype=DTYPE)


class Sequential(Module):
    """A chain of modules. Forward in order, backward in reverse."""

    def __init__(self, *layers: Module) -> None:
        if not layers:
            raise ValueError("Sequential needs at least one layer")
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype=DTYPE)
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grad = np.asarray(grad_out, dtype=DTYPE)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self) -> list[Parameter]:
        return [p for layer in self.layers for p in layer.parameters()]

    def __iter__(self) -> Iterator[Module]:
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #

#: A loss takes ``(prediction, target)`` and returns ``(scalar_loss, dL/dpred)``.
LossFn = Callable[[np.ndarray, np.ndarray], tuple[float, np.ndarray]]


def bce_with_logits_loss(
    logits: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray]:
    """Binary cross-entropy on **logits**, with its gradient.

    Takes logits rather than probabilities on purpose. The composition
    ``sigmoid`` then ``log`` overflows in both directions — ``log(0)`` for a
    confidently wrong prediction and ``log(1 - 1)`` for a confidently right one —
    and the usual patch, clipping the probability to ``[1e-7, 1-1e-7]``, silently
    caps the gradient exactly where the model is most wrong and most needs it.

    The stable identity used here is::

        L = max(z, 0) - z * y + log(1 + exp(-|z|))

    which is algebraically equal to ``-y*log(s(z)) - (1-y)*log(1-s(z))`` and never
    evaluates ``exp`` of a positive argument. Its gradient is the clean
    ``sigmoid(z) - y``, which is finite everywhere.
    """
    z = np.asarray(logits, dtype=DTYPE)
    y = np.asarray(target, dtype=DTYPE)
    if z.shape != y.shape:
        raise ValueError(f"bce_with_logits_loss shape mismatch: {z.shape} vs {y.shape}")
    if z.size == 0:
        raise ValueError("bce_with_logits_loss on an empty batch")
    if not np.all((y >= 0.0) & (y <= 1.0)):
        raise ValueError("bce targets must lie in [0, 1]")

    per_element = np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))
    loss = float(np.mean(per_element))
    return loss, (sigmoid(z) - y) / z.size


def mse_loss(prediction: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean squared error over **all elements**, with its gradient.

    Averaging over elements rather than summing over features and averaging over
    rows keeps the effective learning rate independent of the feature count, so a
    learning rate tuned on a 40-column feature spec does not silently become 40x
    too large when a column block is added. The factor is folded into the
    returned gradient so callers never have to remember it.
    """
    pred = np.asarray(prediction, dtype=DTYPE)
    truth = np.asarray(target, dtype=DTYPE)
    if pred.shape != truth.shape:
        raise ValueError(f"mse_loss shape mismatch: {pred.shape} vs {truth.shape}")
    if pred.size == 0:
        raise ValueError("mse_loss on an empty batch")
    diff = pred - truth
    loss = float(np.mean(np.square(diff)))
    return loss, (2.0 / diff.size) * diff


# --------------------------------------------------------------------------- #
# Optimiser
# --------------------------------------------------------------------------- #


class Adam:
    """Adam with bias correction and **decoupled** weight decay (AdamW).

    Decoupled is the correct default and the difference is not cosmetic. Classic
    Adam adds ``wd * w`` to the gradient, which then passes through the
    ``1/sqrt(v)`` normalisation — so a parameter with a small gradient history
    gets a *large* effective decay and one with a large history gets almost none.
    The regularisation strength ends up inversely proportional to each weight's
    own gradient noise, which is not a regulariser, it is a bug with a knob.
    Applying decay directly to the value (Loshchilov & Hutter) keeps it uniform.
    """

    def __init__(
        self,
        parameters: Sequence[Parameter],
        *,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if not parameters:
            raise ValueError("Adam needs at least one parameter")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("beta1 and beta2 must lie in [0, 1)")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = [np.zeros_like(p.value) for p in self.parameters]
        self._v = [np.zeros_like(p.value) for p in self.parameters]
        self._t = 0

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        self._t += 1
        bias1 = 1.0 - self.beta1**self._t
        bias2 = 1.0 - self.beta2**self._t
        for index, parameter in enumerate(self.parameters):
            grad = parameter.grad
            if not np.all(np.isfinite(grad)):
                raise TrainingError(
                    f"non-finite gradient in {parameter.name}; training has diverged. "
                    "Lower the learning rate or check the input scaling."
                )
            self._m[index] = self.beta1 * self._m[index] + (1.0 - self.beta1) * grad
            self._v[index] = self.beta2 * self._v[index] + (1.0 - self.beta2) * np.square(grad)
            m_hat = self._m[index] / bias1
            v_hat = self._v[index] / bias2
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            if self.weight_decay:
                update = update + self.lr * self.weight_decay * parameter.value
            parameter.value -= update


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class TrainHistory:
    """What happened during training, so a bad fit is diagnosable after the fact."""

    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_validation_loss: float = float("inf")
    stopped_early: bool = False
    n_parameters: int = 0

    @property
    def epochs_run(self) -> int:
        return len(self.train_loss)

    @property
    def converged(self) -> bool:
        """Did the loss come down by a **meaningful** amount?

        Reported rather than asserted, because "the first epoch was already
        optimal" is legitimate for a tiny model on easy data. The detectors that
        use this treat a False here as a reason to raise.

        The relative threshold is not fussiness. The obvious test,
        ``train_loss[-1] < train_loss[0]``, passes on pure numerical noise: with a
        batch size at or above the dataset size every epoch sums the *same* values
        in a *different* order (the shuffle), and float64 addition is not
        associative, so the loss moves by ~1e-16 relative between epochs in a
        random direction. Measured: a model trained at ``lr=1e-20`` — weights
        provably frozen to float64 resolution — reported a "decreasing" loss and
        passed. Requiring a relative improvement of
        :data:`MIN_RELATIVE_IMPROVEMENT` distinguishes training from arithmetic.
        """
        if len(self.train_loss) < 2:
            return False
        first, last = self.train_loss[0], self.train_loss[-1]
        if first <= 0.0:
            # Already at or below zero loss; there is nothing left to improve, and
            # a relative test would divide by it.
            return last <= first
        return (first - last) / abs(first) >= MIN_RELATIVE_IMPROVEMENT

    def summary(self) -> str:
        line = (
            f"{self.epochs_run} epoch(s), {self.n_parameters:,} params, "
            f"train {self.train_loss[0]:.6f} -> {self.train_loss[-1]:.6f}"
            if self.train_loss
            else "no epochs run"
        )
        if self.validation_loss:
            line += (
                f", best val {self.best_validation_loss:.6f} @ epoch {self.best_epoch}"
                f"{' (early stop)' if self.stopped_early else ''}"
            )
        return line


#: Builds ``(model_input, target)`` from a batch of raw rows. The autoencoder
#: corrupts the input and targets the clean row; the diffusion denoiser builds a
#: noisy row plus timestep features and targets the noise. The trainer does not
#: need to know which.
BatchFn = Callable[[np.ndarray, np.random.Generator], tuple[np.ndarray, np.ndarray]]


def train(
    module: Module,
    data: np.ndarray,
    make_batch: BatchFn,
    *,
    rng: np.random.Generator,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    loss_fn: LossFn = mse_loss,
    validation_data: np.ndarray | None = None,
    patience: int = 8,
    min_delta: float = 1e-6,
) -> TrainHistory:
    """Minibatch AdamW with early stopping and best-weight restore.

    Two details that are easy to get wrong and expensive to debug:

    **Validation noise must be frozen.** ``make_batch`` is stochastic — that is
    the whole point of a *denoising* autoencoder. If validation loss is computed
    with fresh noise every epoch, the curve carries the noise's variance, early
    stopping fires on a lucky draw, and the restored weights are whichever epoch
    got the easiest corruption. Each evaluation therefore uses a generator seeded
    identically every time, so successive epochs are compared on *the same*
    corrupted validation set and the only thing moving is the model.

    **Best weights are restored, not just recorded.** Stopping after ``patience``
    worsening epochs and then keeping the final (worse) weights is a bug that
    looks like working early stopping. The snapshot from ``best_epoch`` is loaded
    back before returning.
    """
    matrix = np.asarray(data, dtype=DTYPE)
    if matrix.ndim != 2:
        raise ValueError(f"train expects a 2-D data matrix, got shape {matrix.shape}")
    if matrix.shape[0] == 0:
        raise ValueError("train called on an empty data matrix")
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if patience < 1:
        raise ValueError("patience must be >= 1")

    optimizer = Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    history = TrainHistory(n_parameters=module.n_parameters())
    n_rows = matrix.shape[0]
    effective_batch = min(batch_size, n_rows)

    # The seed for the frozen validation corruption. Drawn from the caller's rng
    # so it still varies with the caller's seed, but fixed for this whole run.
    validation_seed = int(rng.integers(0, 2**31 - 1))
    best_state: list[np.ndarray] | None = None
    epochs_without_improvement = 0

    for _ in range(epochs):
        order = rng.permutation(n_rows)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_rows, effective_batch):
            rows = matrix[order[start : start + effective_batch]]
            inputs, targets = make_batch(rows, rng)
            prediction = module.forward(inputs)
            loss, grad = loss_fn(prediction, targets)
            if not np.isfinite(loss):
                raise TrainingError(
                    "loss became non-finite; training diverged. Check input scaling "
                    "and lower the learning rate."
                )
            optimizer.zero_grad()
            module.backward(grad)
            optimizer.step()
            epoch_loss += loss
            n_batches += 1
        history.train_loss.append(epoch_loss / max(1, n_batches))

        if validation_data is None:
            continue

        validation_loss = _evaluate(
            module,
            np.asarray(validation_data, dtype=DTYPE),
            make_batch,
            loss_fn,
            np.random.default_rng(validation_seed),
        )
        history.validation_loss.append(validation_loss)
        if validation_loss < history.best_validation_loss - min_delta:
            history.best_validation_loss = validation_loss
            history.best_epoch = len(history.validation_loss) - 1
            best_state = module.state()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                history.stopped_early = True
                break

    if best_state is not None:
        module.load_state(best_state)
    return history


def _evaluate(
    module: Module,
    data: np.ndarray,
    make_batch: BatchFn,
    loss_fn: LossFn,
    rng: np.random.Generator,
) -> float:
    """Mean loss over ``data`` in one pass, weighted by row count.

    Chunked at 4096 rows so a large validation split does not materialise a
    single enormous activation cache, and weighted by actual chunk size so a
    ragged final chunk does not get the same weight as a full one.
    """
    total = 0.0
    seen = 0
    for start in range(0, data.shape[0], 4096):
        rows = data[start : start + 4096]
        inputs, targets = make_batch(rows, rng)
        loss, _ = loss_fn(module.forward(inputs), targets)
        total += loss * rows.shape[0]
        seen += rows.shape[0]
    return total / max(1, seen)


# --------------------------------------------------------------------------- #
# Gradient checking
# --------------------------------------------------------------------------- #


def gradient_check(
    module: Module,
    x: np.ndarray,
    target: np.ndarray,
    *,
    loss_fn: LossFn = mse_loss,
    eps: float = 1e-6,
    max_checks_per_param: int = 24,
    rng: np.random.Generator | None = None,
) -> float:
    """Max relative error between analytic and numerical gradients.

    This is the test that makes a hand-written backward pass trustworthy. For a
    sample of entries in every parameter, it compares the accumulated analytic
    gradient against the central difference
    ``(L(w+eps) - L(w-eps)) / (2*eps)``, and returns the largest relative error
    seen::

        |analytic - numeric| / max(1, |analytic|, |numeric|)

    Central (not forward) differences because the error term is ``O(eps^2)``
    rather than ``O(eps)``: at ``eps=1e-6`` in float64 that is the difference
    between agreeing to eight digits and agreeing to four, and four digits is
    not enough to distinguish a correct gradient from one missing a term.

    A correct implementation returns something around ``1e-9``. Anything above
    ``1e-5`` means a real bug — most often a missing accumulation, a transposed
    matrix product, or a loss-scaling factor applied in one path and not the
    other.

    Entries are **sampled** rather than exhaustively checked because each one
    costs two forward passes; ``max_checks_per_param`` entries per parameter,
    chosen by a seeded generator, keeps the test fast while still covering every
    parameter array. Pass ``max_checks_per_param`` larger than a parameter's size
    to check it exhaustively.

    Known blind spot
    ----------------
    This checks **parameter** gradients, not the gradient a layer returns to its
    input. Those coincide for every layer that has something below it — the returned
    gradient is what computes the lower layer's parameter gradients — but for the
    **first** layer in a stack the return value is discarded by the caller, so an
    error in it is invisible here.

    That is not hypothetical. A deliberately transposed aggregation in a
    single-layer :class:`~sentinel.graph.gnn.SageLayer` network measured 7.3e-10,
    indistinguishable from a correct implementation; the same bug in a two-layer
    stack measured ~1e-1. So check the composition you actually ship, at the depth
    you ship it, and be suspicious of a passing check on a one-layer model. Both
    behaviours are pinned in
    ``tests/unit/test_graph_gnn.py::TestSageGradients``.
    """
    generator = rng if rng is not None else np.random.default_rng(0)
    inputs = np.asarray(x, dtype=DTYPE)
    truth = np.asarray(target, dtype=DTYPE)

    module.zero_grad()
    prediction = module.forward(inputs)
    _, grad = loss_fn(prediction, truth)
    module.backward(grad)
    analytic = {id(p): np.array(p.grad, copy=True) for p in module.parameters()}

    worst = 0.0
    for parameter in module.parameters():
        values = parameter.value
        analytic_grad = analytic[id(parameter)]
        if values.size <= max_checks_per_param:
            flat_indices = np.arange(values.size)
        else:
            flat_indices = generator.choice(
                values.size, size=max_checks_per_param, replace=False
            )

        for flat_index in flat_indices:
            # Index through unravel_index rather than a flattened view: reshape(-1)
            # and ravel() return a *copy* for any non-contiguous array, so writing
            # through them would perturb nothing, the numeric gradient would come
            # back 0, and a genuinely broken backward pass whose analytic gradient
            # was also near 0 would pass this check. Every parameter here happens
            # to be contiguous today; a check that silently stops checking when
            # that changes is worse than no check.
            position = np.unravel_index(int(flat_index), values.shape)
            original = float(values[position])

            values[position] = original + eps
            loss_plus, _ = loss_fn(module.forward(inputs), truth)

            values[position] = original - eps
            loss_minus, _ = loss_fn(module.forward(inputs), truth)

            values[position] = original  # restore before the next probe, always

            numeric = (loss_plus - loss_minus) / (2.0 * eps)
            expected = float(analytic_grad[position])
            scale = max(1.0, abs(expected), abs(numeric))
            worst = max(worst, abs(expected - numeric) / scale)

    return worst
