"""Dashboard authentication: tokens, identities, tenants and roles (Part 5)."""

from __future__ import annotations

import pytest

from sentinel.dashboard.auth import (
    MIN_TOKEN_LENGTH,
    AuthError,
    Identity,
    Role,
    TokenRegistry,
)

GOOD = "a" * MIN_TOKEN_LENGTH
MAYA = Identity("maya@acme.example", "acme", Role.ANALYST)


class TestIdentity:
    @pytest.mark.parametrize(
        "principal",
        ["", " maya", "maya chen", "maya\n@acme", "x" * 129, "@acme", 'a"b'],
    )
    def test_a_principal_is_one_unambiguous_login(self, principal):
        with pytest.raises(AuthError):
            Identity(principal, "acme", Role.ANALYST)

    @pytest.mark.parametrize("tenant", ["", "ACME", "acme corp", "../acme", "-acme"])
    def test_tenant_ids_are_validated(self, tenant):
        with pytest.raises(AuthError):
            Identity("maya", tenant, Role.ANALYST)

    def test_only_analysts_act(self):
        assert MAYA.can_act
        assert not Identity("v", "acme", Role.VIEWER).can_act


class TestRegistry:
    def test_short_or_padded_tokens_are_refused(self):
        with pytest.raises(AuthError, match="shorter"):
            TokenRegistry({"a" * (MIN_TOKEN_LENGTH - 1): MAYA})
        with pytest.raises(AuthError):
            TokenRegistry({f" {GOOD}": MAYA})

    def test_lookup_by_bearer_header(self):
        registry = TokenRegistry({GOOD: MAYA})
        assert registry.authenticate(f"Bearer {GOOD}") is MAYA
        assert registry.authenticate(f"bearer {GOOD}") is MAYA

    @pytest.mark.parametrize(
        "header",
        [None, "", GOOD, f"Basic {GOOD}", "Bearer ", f"Bearer {GOOD} ", f"Bearer  {GOOD}",
         f"Bearer {GOOD}x", "Bearer " + "b" * MIN_TOKEN_LENGTH],
    )
    def test_anything_else_is_unauthenticated(self, header):
        registry = TokenRegistry({GOOD: MAYA})
        with pytest.raises(AuthError):
            registry.authenticate(header)

    def test_the_registry_holds_no_usable_secret(self):
        registry = TokenRegistry({GOOD: MAYA})
        assert GOOD not in repr(vars(registry))

    def test_tenants(self):
        registry = TokenRegistry({
            GOOD: MAYA,
            "b" * 30: Identity("sam@globex.example", "globex", Role.VIEWER),
        })
        assert registry.tenants == {"acme", "globex"}
        assert len(registry) == 2


class TestParse:
    def test_round_trip(self):
        spec = f"{GOOD}:maya@acme.example:acme:analyst, {'c' * 30}:ops:globex:viewer"
        registry = TokenRegistry.parse(spec)
        assert registry.authenticate(f"Bearer {GOOD}") == MAYA
        assert registry.authenticate(f"Bearer {'c' * 30}").role is Role.VIEWER

    @pytest.mark.parametrize(
        "spec",
        ["", "   ,  ", f"{GOOD}:maya:acme", f"{GOOD}:maya:acme:admin",
         f"{GOOD}:maya:acme:analyst:extra", f"{GOOD}:a:acme:analyst,{GOOD}:b:acme:viewer"],
    )
    def test_bad_specs_fail_loudly(self, spec):
        with pytest.raises(AuthError):
            TokenRegistry.parse(spec)


class TestGenerate:
    def test_fresh_strong_tokens_for_each_identity(self):
        viewer = Identity("viewer@acme.example", "acme", Role.VIEWER)
        registry, issued = TokenRegistry.generate([MAYA, viewer])
        assert len(issued) == 2 and len(set(issued.values())) == 2
        for token in issued.values():
            assert len(token) >= MIN_TOKEN_LENGTH
        assert registry.authenticate(f"Bearer {issued['maya@acme.example@acme']}") == MAYA
        _, again = TokenRegistry.generate([MAYA])
        assert again["maya@acme.example@acme"] != issued["maya@acme.example@acme"]
