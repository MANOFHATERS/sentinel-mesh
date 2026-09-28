"""GraphSAGE risk propagation over the supply-chain graph (PRD F-06, Section 5.5.3).

    *"a 2-layer GraphSAGE network learns to propagate risk across edges so that a
    company's exposure through a fourth-order dependency — the kind that a static
    vendor questionnaire never reaches — surfaces as a scored, explainable path
    rather than an invisible risk."*

Those two clauses are in tension, and the tension is arithmetic
-------------------------------------------------------------
A ``k``-layer message-passing network has a receptive field of exactly ``k`` hops.
After two layers, a node's representation is a function of itself, its neighbours,
and its neighbours' neighbours — and **nothing else**. A fourth-order dependency is
four hops away. It cannot reach a 2-layer node representation. Not weakly; not at
all. No amount of training changes this, because the information is never in the
input to the final layer.

So the PRD's "2-layer" and its "fourth-order dependency" cannot both hold as
written. This is not a gotcha — it is the kind of thing that gets written when an
architecture and a capability claim are drafted in different paragraphs, and it is
exactly what an implementation is for: :class:`SupplyChainGNN` takes ``n_layers``,
defaults to **2** so the specified architecture is what ships, and
``tests/unit/test_graph_gnn.py::TestReceptiveField`` measures 2 against 4 on nodes
whose *only* exposure is deep. The result is recorded in ``docs/BUILD_PLAN.md``
rather than resolved by picking whichever clause is more convenient.

The layer
---------
GraphSAGE with a mean aggregator, which is the variant the PRD names::

    h_v' = W_self . h_v  +  W_neigh . mean_{u in sources(v)} h_u

``W_self`` and ``W_neigh`` are kept separate — that is the defining feature of
GraphSAGE versus a GCN, which folds them into one normalised operator. It matters
here beyond fidelity: separating them is what lets
:mod:`sentinel.graph.explain` answer "how much of this organisation's score came
from *its vendors* rather than from its own features", which is the sentence F-06's
guardrail actually requires.

Both branches route through :class:`~sentinel.ml.nn.Dense`, so the whole stack is
checkable by :func:`~sentinel.ml.nn.gradient_check`. That is worth more than it
sounds: the neighbour branch's backward pass has to push gradient through the
aggregation matrix as ``A.T @ g``, and getting that transpose wrong produces a
model that trains happily to a worse optimum. The gradient check catches it; a
loss curve does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from sentinel.core.errors import ModelNotFittedError
from sentinel.graph.schema import FEATURE_NAMES, GraphError, SupplyChainGraph
from sentinel.ml.nn import (
    Adam,
    Dense,
    Module,
    Parameter,
    ReLU,
    TrainingError,
    sigmoid,
)

__all__ = [
    "GraphSplit",
    "SageLayer",
    "SupplyChainGNN",
    "top_k_precision",
]

DTYPE: Final = np.float64


# --------------------------------------------------------------------------- #
# Layer
# --------------------------------------------------------------------------- #


class SageLayer(Module):
    """One GraphSAGE mean-aggregator layer over a fixed aggregation matrix.

    ``aggregation`` is ``A_norm`` from
    :meth:`~sentinel.graph.schema.SupplyChainGraph.aggregation_matrix`: row ``v``
    is the normalised weights of the nodes whose risk flows *into* ``v``. It is
    held by reference and never modified.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        aggregation: np.ndarray,
        *,
        rng: np.random.Generator,
        gain_for: str = "relu",
        name: str = "sage",
    ) -> None:
        if aggregation.ndim != 2 or aggregation.shape[0] != aggregation.shape[1]:
            raise GraphError(
                f"aggregation matrix must be square, got shape {aggregation.shape}"
            )
        self.aggregation = np.asarray(aggregation, dtype=DTYPE)
        self.lin_self = Dense(n_in, n_out, rng=rng, gain_for=gain_for, name=f"{name}.self")
        self.lin_neigh = Dense(n_in, n_out, rng=rng, gain_for=gain_for, name=f"{name}.neigh")
        self._n_nodes = aggregation.shape[0]

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.asarray(x, dtype=DTYPE)
        if h.shape[0] != self._n_nodes:
            raise GraphError(
                f"{self.lin_self.weight.name}: expected one row per node "
                f"({self._n_nodes}), got {h.shape[0]}. SageLayer is full-graph; it "
                "cannot be fed a minibatch of rows."
            )
        # Aggregate first, then transform — the order the mean aggregator specifies.
        # Transforming first and aggregating after is a different (and cheaper for
        # wide layers) model, and quietly changes what the weights mean.
        self._neighbourhood = self.aggregation @ h
        return self.lin_self.forward(h) + self.lin_neigh.forward(self._neighbourhood)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grad = np.asarray(grad_out, dtype=DTYPE)
        grad_self = self.lin_self.backward(grad)
        grad_neighbourhood = self.lin_neigh.backward(grad)
        # neighbourhood = A @ h, so dL/dh picks up A.T @ dL/d(neighbourhood).
        # A is indexed [target, source]; the transpose sends each target's gradient
        # back to the sources that fed it. Using A instead of A.T here trains fine
        # and optimises the wrong objective — hence the gradient check.
        return grad_self + self.aggregation.T @ grad_neighbourhood

    def parameters(self) -> list[Parameter]:
        return self.lin_self.parameters() + self.lin_neigh.parameters()


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class GraphSplit:
    """Disjoint train / validation / test node index sets.

    Transductive, which is the honest description: the **whole graph** including
    test nodes' features and edges is visible during training, and only test
    *labels* are hidden. That is standard for node classification and it is what
    the deployment looks like — an MSSP's graph is fully known, the question is
    which nodes to flag — but it must be said out loud, because the number it
    produces is not comparable to an inductive benchmark where test nodes are
    unseen entirely.
    """

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def __post_init__(self) -> None:
        # Coerce to an integer index dtype. `np.array([])` is float64, which raises
        # "arrays used as indices must be of integer type" only at the moment it is
        # first used as a mask index — potentially deep inside a training loop, long
        # after the split was built. Normalising here makes an empty validation split
        # (entirely legitimate) work the same as a populated one.
        for field_name in ("train", "validation", "test"):
            value = np.asarray(getattr(self, field_name))
            if value.size == 0:
                value = np.empty(0, dtype=int)
            elif not np.issubdtype(value.dtype, np.integer):
                if not np.all(value == np.floor(value)):
                    raise GraphError(
                        f"{field_name} split holds non-integer indices; a fractional "
                        "node index is always a bug"
                    )
                value = value.astype(int)
            object.__setattr__(self, field_name, value)

        groups = {
            "train": set(self.train.tolist()),
            "validation": set(self.validation.tolist()),
            "test": set(self.test.tolist()),
        }
        names = list(groups)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                overlap = groups[left] & groups[right]
                if overlap:
                    raise GraphError(
                        f"{left} and {right} node splits overlap on "
                        f"{len(overlap)} node(s); every metric would be contaminated"
                    )
        for name, indices in (("train", self.train), ("test", self.test)):
            if indices.size == 0:
                raise GraphError(f"{name} split is empty")

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": int(self.train.size),
            "validation": int(self.validation.size),
            "test": int(self.test.size),
        }

    @classmethod
    def stratified(
        cls,
        labels: np.ndarray,
        *,
        seed: int = 20260928,
        validation_fraction: float = 0.15,
        test_fraction: float = 0.30,
    ) -> GraphSplit:
        """Stratified by label, so every split carries positives.

        Unstratified splitting is not survivable at this prevalence: with ~11%
        positives and a 15% validation slice, an unlucky draw hands validation zero
        high-risk nodes, early stopping then optimises a metric that is undefined,
        and nothing raises.
        """
        y = np.asarray(labels).ravel()
        rng = np.random.default_rng(seed)
        train: list[int] = []
        validation: list[int] = []
        test: list[int] = []
        for value in np.unique(y):
            members = np.flatnonzero(y == value)
            rng.shuffle(members)
            n_validation = round(members.size * validation_fraction)
            n_test = round(members.size * test_fraction)
            # Guarantee at least one of each class in train and test, even for a
            # class with only a couple of members.
            n_test = min(n_test, max(0, members.size - 1))
            if members.size >= 3:
                n_test = max(n_test, 1)
                n_validation = min(n_validation, members.size - n_test - 1)
            else:
                n_validation = 0
            validation.extend(members[:n_validation].tolist())
            test.extend(members[n_validation : n_validation + n_test].tolist())
            train.extend(members[n_validation + n_test :].tolist())
        return cls(
            train=np.sort(np.array(train, dtype=int)),
            validation=np.sort(np.array(validation, dtype=int)),
            test=np.sort(np.array(test, dtype=int)),
        )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def top_k_precision(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """Fraction of the ``k`` highest-scoring nodes that are truly high-risk.

    F-06's acceptance metric. Ties are broken by index rather than left to
    ``argsort``'s implementation detail, so the number is reproducible: with
    saturating scores several nodes genuinely tie at the top, and a stable sort
    makes the reported figure the same on every platform.
    """
    values = np.asarray(scores, dtype=DTYPE).ravel()
    truth = np.asarray(labels).ravel()
    if values.size != truth.size:
        raise GraphError(f"score/label length mismatch: {values.size} vs {truth.size}")
    if k < 1:
        raise GraphError("k must be >= 1")
    if values.size == 0:
        raise GraphError("cannot compute top-k precision on an empty node set")
    effective_k = min(k, values.size)
    ranking = np.argsort(-values, kind="stable")[:effective_k]
    return float(np.mean(truth[ranking] > 0))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


@dataclass
class GnnTrainingReport:
    """Architecture and training curve, for the F-12 evaluation artifact."""

    aggregation: str
    objective: str
    n_layers: int
    hidden: int
    n_parameters: int
    epochs_run: int
    best_epoch: int
    best_validation_loss: float
    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    stopped_early: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "aggregation": self.aggregation,
            "objective": self.objective,
            "n_layers": self.n_layers,
            "hidden": self.hidden,
            "n_parameters": self.n_parameters,
            "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "final_train_loss": self.train_loss[-1] if self.train_loss else None,
            "stopped_early": self.stopped_early,
            "receptive_field_hops": self.n_layers,
        }


class SupplyChainGNN:
    """Node-level risk scorer: ``n_layers`` SAGE layers, ReLU, then a linear head.

    Trained full-batch with a masked loss. Full-batch is not a simplification at
    500 nodes — it is the correct choice: the whole graph fits trivially, and
    neighbour sampling exists to make large graphs tractable, not to improve them.

    Only training-split nodes contribute to the loss. Their *features and edges*
    are visible throughout, which is what makes this transductive; see
    :class:`GraphSplit`.
    """

    def __init__(
        self,
        *,
        # 12, not 32. Measured: at hidden=32 the model carries 2,529 parameters
        # against 275 training nodes and 5 input features, and it overfits so fast
        # that early stopping selected epoch 2 of 63 — validation loss rose
        # monotonically from there. A 5-feature problem does not need a 32-wide
        # hidden layer, and the capacity bought nothing but variance. 12 was chosen
        # on seeds 100-119 and then evaluated once on a disjoint seed set; see
        # docs/BUILD_PLAN.md for the numbers and for why that separation mattered.
        hidden: int = 12,
        n_layers: int = 2,
        epochs: int = 400,
        lr: float = 8e-3,
        weight_decay: float = 5e-3,
        patience: int = 60,
        aggregation: str = "mean",
        objective: str = "regression",
        random_state: int = 20260928,
    ) -> None:
        if n_layers < 1:
            raise GraphError("n_layers must be >= 1")
        if hidden < 1:
            raise GraphError("hidden must be >= 1")
        if epochs < 2:
            raise GraphError("epochs must be >= 2")
        if aggregation not in ("mean", "sum"):
            raise GraphError(f"aggregation must be 'mean' or 'sum', got {aggregation!r}")
        if objective not in ("classification", "regression"):
            raise GraphError(
                f"objective must be 'classification' or 'regression', got {objective!r}"
            )
        self.aggregation = aggregation
        self.objective = objective
        self.hidden = hidden
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.random_state = random_state

        self._layers: list[Module] = []
        self._head: Dense | None = None
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._n_nodes: int | None = None
        self._target_mean: float | None = None
        self._target_scale: float | None = None
        self.report_: GnnTrainingReport | None = None

    # --- plumbing -----------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._head is not None

    def _modules(self) -> list[Module]:
        assert self._head is not None
        return [*self._layers, self._head]

    def _parameters(self) -> list[Parameter]:
        return [p for module in self._modules() for p in module.parameters()]

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        if self._mean is None or self._scale is None:
            raise ModelNotFittedError("GNN feature standardisation is not fitted")
        return (x - self._mean) / self._scale

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Logits, shape ``(n_nodes,)``."""
        h = x
        for module in self._layers:
            h = module.forward(h)
        assert self._head is not None
        return self._head.forward(h).ravel()

    def _backward(self, grad_logits: np.ndarray) -> None:
        assert self._head is not None
        grad = self._head.backward(grad_logits.reshape(-1, 1))
        for module in reversed(self._layers):
            grad = module.backward(grad)

    def _zero_grad(self) -> None:
        for module in self._modules():
            module.zero_grad()

    # --- fit ----------------------------------------------------------------

    def fit(
        self,
        graph: SupplyChainGraph,
        labels: np.ndarray,
        split: GraphSplit,
        *,
        exposure: np.ndarray | None = None,
    ) -> SupplyChainGNN:
        """Fit on ``labels`` (binary) or on ``exposure`` (graded), per ``objective``.

        ``objective="regression"`` is the default, and it is a better match to both
        the target and the product than classification is.

        The ground-truth label is a *threshold on a continuous quantity* —
        accumulated decayed exposure. Training on the boolean throws that quantity
        away: a node just over the line and a node ten times over it become the same
        training signal, and the model is asked to reconstruct an ordering from data
        that no longer contains it. F-06 then scores the model on **top-10
        precision**, which is purely a question about ordering.

        It is also what PRD Section 3.4 actually promises — a *"first-class,
        continuously scored graph"*, not a flag. An MSSP looking at 20 clients does
        not need to know which are exposed (all of them are, measurably); it needs
        them in order.

        Measured over 8 seeds, top-10 precision: classification 0.800, regression
        (see ``docs/BUILD_PLAN.md``). ``exposure`` defaults to ``labels`` when not
        supplied, so a caller with only binary truth still works — it just gets a
        regression onto a 0/1 target, which is legitimate but throws away the
        advantage.
        """
        y = np.asarray(labels, dtype=DTYPE).ravel()
        if y.size != graph.n_nodes:
            raise GraphError(
                f"labels ({y.size}) do not match node count ({graph.n_nodes})"
            )
        if np.unique(y[split.train]).size < 2:
            raise GraphError(
                "the training split contains only one class; a model fitted on it "
                "would be a constant and every score would tie"
            )

        if self.objective == "regression":
            raw_target = y if exposure is None else np.asarray(exposure, dtype=DTYPE).ravel()
            if raw_target.size != graph.n_nodes:
                raise GraphError(
                    f"exposure ({raw_target.size}) does not match node count "
                    f"({graph.n_nodes})"
                )
            if np.any(raw_target < 0):
                raise GraphError("exposure scores must be non-negative")
            # log1p first: exposure is a sum of decayed severities and is heavily
            # right-skewed (measured min 0.08, median 0.47, max 5.06). Regressing on
            # the raw value lets the handful of extreme nodes dominate the squared
            # error, and the model spends its capacity on their exact magnitudes
            # instead of on the ordering everywhere else. log1p is monotone, so the
            # ranking F-06 measures is untouched.
            target = np.log1p(raw_target)
            self._target_mean = float(target[split.train].mean())
            deviation = float(target[split.train].std())
            self._target_scale = deviation if deviation > 1e-12 else 1.0
            target = (target - self._target_mean) / self._target_scale
        else:
            target = y

        rng = np.random.default_rng(self.random_state)
        self._n_nodes = graph.n_nodes
        raw = graph.feature_matrix()

        # Standardisation statistics come from **training nodes only**. Using all
        # nodes is the classic transductive leak: it is subtle because no label is
        # involved, and it still lets test-node feature distributions inform the
        # model. Part 1 made the same discipline explicit for alerts.
        train_rows = raw[split.train]
        self._mean = train_rows.mean(axis=0)
        deviation = train_rows.std(axis=0)
        self._scale = np.where(deviation > 1e-12, deviation, 1.0)
        x = self._standardize(raw)

        aggregation = graph.aggregation_matrix(mode=self.aggregation)
        self._layers = []
        width = len(FEATURE_NAMES)
        for layer_index in range(self.n_layers):
            self._layers.append(
                SageLayer(
                    width,
                    self.hidden,
                    aggregation,
                    rng=rng,
                    gain_for="relu",
                    name=f"sage{layer_index}",
                )
            )
            self._layers.append(ReLU())
            width = self.hidden
        self._head = Dense(width, 1, rng=rng, gain_for="linear", name="head")

        optimizer = Adam(
            self._parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        train_mask = np.zeros(graph.n_nodes, dtype=bool)
        train_mask[split.train] = True
        validation_mask = np.zeros(graph.n_nodes, dtype=bool)
        validation_mask[split.validation] = True
        has_validation = bool(validation_mask.any())

        # Positive-class weighting. At ~11% prevalence an unweighted BCE minimises
        # nicely by predicting "safe" everywhere, which scores 89% accuracy and
        # ranks nothing. The weight is derived from the training split, so it adapts
        # if the generator's calibration moves.
        n_positive = float(y[split.train].sum())
        n_negative = float(split.train.size - n_positive)
        positive_weight = n_negative / max(n_positive, 1.0)

        train_loss: list[float] = []
        validation_loss: list[float] = []
        best_validation = float("inf")
        best_epoch = -1
        best_state: list[list[np.ndarray]] | None = None
        since_improvement = 0

        for _ in range(self.epochs):
            logits = self._forward(x)
            loss, grad = self._masked_loss(logits, target, train_mask, positive_weight)
            if not np.isfinite(loss):
                raise TrainingError("GNN loss became non-finite; training diverged")
            self._zero_grad()
            self._backward(grad)
            optimizer.step()
            train_loss.append(loss)

            if not has_validation:
                continue
            validation_logits = self._forward(x)
            current, _ = self._masked_loss(
                validation_logits, target, validation_mask, positive_weight
            )
            validation_loss.append(current)
            if current < best_validation - 1e-7:
                best_validation = current
                best_epoch = len(validation_loss) - 1
                best_state = [module.state() for module in self._modules()]
                since_improvement = 0
            else:
                since_improvement += 1
                if since_improvement >= self.patience:
                    break

        stopped_early = has_validation and since_improvement >= self.patience
        if best_state is not None:
            for module, state in zip(self._modules(), best_state, strict=True):
                module.load_state(state)

        if len(train_loss) >= 2 and train_loss[-1] >= train_loss[0]:
            raise TrainingError(
                f"GNN training loss did not fall ({train_loss[0]:.6g} -> "
                f"{train_loss[-1]:.6g}); the model would score noise"
            )

        self.report_ = GnnTrainingReport(
            aggregation=self.aggregation,
            objective=self.objective,
            n_layers=self.n_layers,
            hidden=self.hidden,
            n_parameters=sum(p.size for p in self._parameters()),
            epochs_run=len(train_loss),
            best_epoch=best_epoch,
            best_validation_loss=best_validation,
            train_loss=train_loss,
            validation_loss=validation_loss,
            stopped_early=stopped_early,
        )
        return self

    def _masked_loss(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
        positive_weight: float,
    ) -> tuple[float, np.ndarray]:
        """Loss over masked nodes only, plus gradient for **all** nodes.

        The gradient is zero off-mask and scaled so the loss is a mean over masked
        nodes rather than over the whole graph. Getting that scaling wrong makes
        the effective learning rate depend on the split fraction, so a model tuned
        at a 55% training split silently mistrains at 80%.

        Classification applies ``positive_weight`` to the minority class: at ~11%
        prevalence an unweighted BCE is minimised well by predicting "safe"
        everywhere, which scores 89% accuracy and ranks nothing. Regression needs no
        such weighting — a squared error on a continuous target already pays
        attention to the large values, which are the risky nodes.
        """
        selected = np.flatnonzero(mask)
        if self.objective == "regression":
            residual = prediction[selected] - target[selected]
            loss = float(np.mean(np.square(residual)))
            grad = np.zeros_like(prediction)
            grad[selected] = (2.0 / selected.size) * residual
            return loss, grad

        weights = np.where(target[selected] > 0.5, positive_weight, 1.0)
        loss_per_node = _bce_per_element(prediction[selected], target[selected])
        total_weight = float(weights.sum())
        loss = float(np.sum(loss_per_node * weights) / total_weight)

        grad = np.zeros_like(prediction)
        grad[selected] = (
            weights * (sigmoid(prediction[selected]) - target[selected]) / total_weight
        )
        return loss, grad

    # --- inference ----------------------------------------------------------

    def risk_scores(self, graph: SupplyChainGraph) -> np.ndarray:
        """Risk in ``[0, 1]`` per node, in graph node order.

        For ``objective="classification"`` this is a probability. For
        ``"regression"`` it is ``sigmoid`` of the predicted (standardised, log1p)
        exposure — a **monotone** transform, so the ranking F-06 measures is exactly
        the model's predicted ordering, but the number is not a calibrated
        probability and is not claimed to be. Use
        :meth:`predicted_exposure` when the magnitude matters.
        """
        if not self.is_fitted:
            raise ModelNotFittedError("fit the GNN before scoring")
        if graph.n_nodes != self._n_nodes:
            raise GraphError(
                f"GNN was fitted on {self._n_nodes} nodes but this graph has "
                f"{graph.n_nodes}. Node indices would not correspond; refit instead."
            )
        return sigmoid(self._forward(self._standardize(graph.feature_matrix())))

    def predicted_exposure(self, graph: SupplyChainGraph) -> np.ndarray:
        """Predicted accumulated exposure, back on its original scale.

        Inverts the standardisation and the ``log1p``, so the output is comparable
        to :attr:`~sentinel.graph.synthetic.GroundTruth.exposure_scores`. Only
        meaningful for ``objective="regression"``.
        """
        if self.objective != "regression":
            raise GraphError(
                "predicted_exposure is only defined for objective='regression'; a "
                "classifier predicts a probability, not an exposure magnitude"
            )
        if self._target_mean is None or self._target_scale is None:
            raise ModelNotFittedError("fit the GNN before scoring")
        standardized = self.logits(graph)
        return np.expm1(standardized * self._target_scale + self._target_mean)

    def logits(self, graph: SupplyChainGraph) -> np.ndarray:
        if not self.is_fitted:
            raise ModelNotFittedError("fit the GNN before scoring")
        return self._forward(self._standardize(graph.feature_matrix()))

    def training_report(self) -> dict[str, Any]:
        if self.report_ is None:
            raise ModelNotFittedError("fit the GNN before requesting a report")
        return self.report_.as_dict()


def _bce_per_element(logits: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Stable per-element BCE. Same identity as
    :func:`~sentinel.ml.nn.bce_with_logits_loss`, without the mean, because the
    masked loss needs to weight elements before reducing."""
    z = np.asarray(logits, dtype=DTYPE)
    y = np.asarray(target, dtype=DTYPE)
    return np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))
