"""The action space and its trust-tier mask (:mod:`sentinel.rl.actions`).

This is the safety-critical file in Part 2.5, so the tests are written against the
property rather than the implementation. The build plan's instruction for this part
was explicit: *"test that the action mask is correct, not that the schema holds; the
schema is already tested."* So nothing here re-checks
:class:`~sentinel.core.schemas.ActionRequest`; what is checked is that no tier ever
admits an action above it, that the mask and
:attr:`~sentinel.core.schemas.RiskTier.permits_unattended_execution` cannot drift
apart, and that adding an action or a tier without classifying it fails.

:class:`TestExhaustiveness` is the one that survives future edits. A new
:class:`~sentinel.rl.actions.ResponseAction` added without a minimum tier would
otherwise default to being permitted everywhere, which is the wrong direction for a
safety control to fail in.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from sentinel.core.schemas import ActionType, RiskTier, TriageDecision
from sentinel.rl.actions import (
    ALL_ACTIONS,
    ActionMask,
    PolicyError,
    ResponseAction,
)

TIERS: tuple[RiskTier, ...] = tuple(RiskTier)


class TestActionSpace:
    def test_exactly_the_four_actions_the_prd_names(self) -> None:
        assert {a.value for a in ALL_ACTIONS} == {
            "auto_contain",
            "escalate",
            "monitor",
            "dismiss",
        }

    def test_declaration_order_runs_most_to_least_interventionist(self) -> None:
        """Tie-breaking depends on this order, so it is part of the contract."""
        assert ALL_ACTIONS == (
            ResponseAction.AUTO_CONTAIN,
            ResponseAction.ESCALATE,
            ResponseAction.MONITOR,
            ResponseAction.DISMISS,
        )

    def test_only_auto_contain_intervenes(self) -> None:
        intervening = {a for a in ALL_ACTIONS if a.intervenes}
        assert intervening == {ResponseAction.AUTO_CONTAIN}


class TestExhaustiveness:
    """Guards against a new action or tier being added unclassified."""

    def test_every_action_has_a_minimum_tier(self) -> None:
        for action in ALL_ACTIONS:
            assert isinstance(action.minimum_tier, RiskTier)

    def test_every_action_maps_to_a_triage_decision(self) -> None:
        for action in ALL_ACTIONS:
            assert isinstance(action.triage_decision, TriageDecision)

    def test_every_action_has_an_implied_action_type_mapping(self) -> None:
        for action in ALL_ACTIONS:
            assert isinstance(action.implied_action_types, tuple)

    def test_every_tier_has_a_cached_mask(self) -> None:
        for tier in TIERS:
            assert isinstance(ActionMask.for_tier(tier), ActionMask)

    def test_every_tier_permits_at_least_one_action(self) -> None:
        """A tier that forbids everything would deadlock the orchestrator."""
        for tier in TIERS:
            assert len(ActionMask.for_tier(tier)) >= 1

    def test_allowed_and_forbidden_partition_the_action_space(self) -> None:
        for tier in TIERS:
            mask = ActionMask.for_tier(tier)
            assert set(mask.allowed) | set(mask.forbidden) == set(ALL_ACTIONS)
            assert not set(mask.allowed) & set(mask.forbidden)


class TestTierSemantics:
    def test_observe_permits_only_watching_and_telling_a_human(self) -> None:
        allowed = set(ActionMask.for_tier(RiskTier.OBSERVE).allowed)
        assert allowed == {ResponseAction.MONITOR, ResponseAction.ESCALATE}

    def test_observe_forbids_closing_an_alert(self) -> None:
        """Dismissing destroys the artifact that would let anyone audit the call."""
        assert not ActionMask.for_tier(RiskTier.OBSERVE).permits(ResponseAction.DISMISS)

    def test_recommend_adds_dismiss_but_not_containment(self) -> None:
        mask = ActionMask.for_tier(RiskTier.RECOMMEND)
        assert mask.permits(ResponseAction.DISMISS)
        assert not mask.permits(ResponseAction.AUTO_CONTAIN)

    def test_top_two_tiers_permit_everything(self) -> None:
        for tier in (RiskTier.AUTO_WITH_NOTIFY, RiskTier.AUTONOMOUS):
            assert set(ActionMask.for_tier(tier).allowed) == set(ALL_ACTIONS)

    def test_permissions_are_monotone_in_tier(self) -> None:
        """Trust is earned upward, so a higher tier can never permit less."""
        ordered = sorted(TIERS, key=lambda t: t.rank)
        for lower, higher in pairwise(ordered):
            assert set(ActionMask.for_tier(lower).allowed) <= set(
                ActionMask.for_tier(higher).allowed
            )

    def test_containment_permission_tracks_the_schema_property(self) -> None:
        """The mask and ``permits_unattended_execution`` must not drift apart.

        Part 1 put that property on the enum precisely so this relationship has one
        definition. If someone changes the tier list in either place, this fails.
        """
        for tier in TIERS:
            permitted = ActionMask.for_tier(tier).permits(ResponseAction.AUTO_CONTAIN)
            assert permitted == tier.permits_unattended_execution

    def test_no_tier_permits_an_action_above_it(self) -> None:
        """The core invariant, stated directly over the whole cross product."""
        for tier in TIERS:
            for action in ActionMask.for_tier(tier).allowed:
                assert tier.rank >= action.minimum_tier.rank


class TestMaskInterface:
    def test_indices_match_allowed_actions(self) -> None:
        """``indices`` is what the bandit actually consumes."""
        for tier in TIERS:
            mask = ActionMask.for_tier(tier)
            assert tuple(ALL_ACTIONS[i] for i in mask.indices) == mask.allowed

    def test_require_returns_a_permitted_action(self) -> None:
        mask = ActionMask.for_tier(RiskTier.OBSERVE)
        assert mask.require(ResponseAction.MONITOR) is ResponseAction.MONITOR

    @pytest.mark.parametrize(
        "tier,action",
        [
            (RiskTier.OBSERVE, ResponseAction.AUTO_CONTAIN),
            (RiskTier.OBSERVE, ResponseAction.DISMISS),
            (RiskTier.RECOMMEND, ResponseAction.AUTO_CONTAIN),
        ],
    )
    def test_require_refuses_a_forbidden_action(
        self, tier: RiskTier, action: ResponseAction
    ) -> None:
        mask = ActionMask.for_tier(tier)
        with pytest.raises(PolicyError, match="requires tier"):
            mask.require(action)

    def test_refusal_message_names_both_tiers(self) -> None:
        """An error a reader cannot act on is a log line, not an error."""
        with pytest.raises(PolicyError) as caught:
            ActionMask.for_tier(RiskTier.OBSERVE).require(ResponseAction.AUTO_CONTAIN)
        message = str(caught.value)
        assert "auto_contain" in message
        assert "auto_with_notify" in message
        assert "observe" in message

    def test_contains_and_len(self) -> None:
        mask = ActionMask.for_tier(RiskTier.RECOMMEND)
        assert ResponseAction.DISMISS in mask
        assert ResponseAction.AUTO_CONTAIN not in mask
        assert len(mask) == 3

    def test_masks_are_cached_and_identical_per_tier(self) -> None:
        """Constructed once at import: a per-call mask is one that can be built wrong."""
        assert ActionMask.for_tier(RiskTier.OBSERVE) is ActionMask.for_tier(
            RiskTier.OBSERVE
        )

    def test_equality_and_hashing_by_tier(self) -> None:
        assert ActionMask(RiskTier.OBSERVE) == ActionMask(RiskTier.OBSERVE)
        assert ActionMask(RiskTier.OBSERVE) != ActionMask(RiskTier.AUTONOMOUS)
        assert len({ActionMask(t) for t in TIERS}) == len(TIERS)

    def test_repr_names_the_tier_and_its_actions(self) -> None:
        text = repr(ActionMask.for_tier(RiskTier.OBSERVE))
        assert "observe" in text
        assert "monitor" in text

    def test_non_tier_argument_rejected(self) -> None:
        with pytest.raises(PolicyError, match="must be a RiskTier"):
            ActionMask("autonomous")  # type: ignore[arg-type]

    def test_mask_is_immutable(self) -> None:
        mask = ActionMask.for_tier(RiskTier.OBSERVE)
        with pytest.raises(AttributeError):
            mask._tier = RiskTier.AUTONOMOUS  # type: ignore[misc]


class TestActionTypeMapping:
    def test_containment_maps_only_to_destructive_action_types(self) -> None:
        """Otherwise ``auto_contain`` could be realised as something ungated."""
        for action_type in ResponseAction.AUTO_CONTAIN.implied_action_types:
            assert action_type.is_destructive

    def test_non_containment_maps_to_nothing_destructive(self) -> None:
        for action in ALL_ACTIONS:
            if action is ResponseAction.AUTO_CONTAIN:
                continue
            for action_type in action.implied_action_types:
                assert not action_type.is_destructive

    def test_dismiss_implies_no_action_at_all(self) -> None:
        assert ResponseAction.DISMISS.implied_action_types == ()

    def test_escalate_notifies_a_human(self) -> None:
        assert ResponseAction.ESCALATE.implied_action_types == (
            ActionType.NOTIFY_ANALYST,
        )

    def test_triage_decision_mapping_is_faithful(self) -> None:
        assert ResponseAction.DISMISS.triage_decision is TriageDecision.AUTO_DISMISS
        assert ResponseAction.MONITOR.triage_decision is TriageDecision.MONITOR
        assert ResponseAction.ESCALATE.triage_decision is TriageDecision.ESCALATE
        # Containing a host is treating the alert as real.
        assert ResponseAction.AUTO_CONTAIN.triage_decision is TriageDecision.ESCALATE

    def test_no_action_maps_to_a_dismissal_it_should_not(self) -> None:
        """Only ``dismiss`` may record an auto-dismissal on the alert."""
        dismissing = {
            a for a in ALL_ACTIONS if a.triage_decision is TriageDecision.AUTO_DISMISS
        }
        assert dismissing == {ResponseAction.DISMISS}
