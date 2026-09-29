"""Target validation at the connector boundary.

An :class:`~sentinel.core.schemas.ActionRequest`'s ``target`` is a free string of up
to 256 characters, and by the time it reaches a connector a human has approved it.
That is exactly why it is checked here: the approval screen shows the target, and a
reviewer who reads ``10.0.0.5`` approves blocking one host. A connector that then
passes ``10.0.0.0/8`` — or ``0.0.0.0`` — to a firewall has executed something nobody
approved, with a valid approval attached.

Three failure classes, each tested:

*   **Wrong kind.** ``_TARGET_FIELD`` in :mod:`sentinel.agents.contain` sends an
    ``isolate_host`` the asset id and a ``disable_account`` the asset id too — and on
    the network corpus the asset id is an IP address. An IP is not an account; a
    SCIM filter for ``userName eq "10.0.0.5"`` returns nothing at best and, against a
    directory that allows numeric user names, the wrong person at worst.
*   **Too wide.** CIDR ranges, the unspecified address, broadcast, multicast,
    loopback and link-local are refused as block targets; the first two are
    "everything" and the last three are the firewall blocking itself.
*   **Protected.** The customer's own infrastructure — gateways, resolvers, domain
    controllers, the SOC's jump hosts — is listed in :class:`TargetPolicy`, and an
    action aimed at it is refused whatever the approval says. Containment that cuts
    off the DNS resolver is an outage the attacker did not have to cause.

Validation also returns the *canonical* spelling (``010.000.000.005`` is refused
rather than normalised; ``::FFFF:10.0.0.5`` becomes ``10.0.0.5``), because the target
is interpolated into URLs and API bodies and two spellings of one host defeat both
de-duplication and the protected list.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field
from typing import Final

from sentinel.connectors.base import TargetRejected
from sentinel.core.schemas import ActionType

__all__ = [
    "TargetPolicy",
    "canonical_account",
    "canonical_host",
    "canonical_ip",
]

#: RFC 1123 hostname: labels of 1-63 alnum/hyphen, not starting or ending in hyphen.
_HOST_LABEL: Final[re.Pattern[str]] = re.compile(r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)$")

#: A directory account name: an email-style UPN or a sAMAccountName-style login.
#: Deliberately excludes whitespace, quotes, and SCIM filter syntax (``"``, ``(``,
#: ``)``), because the value is interpolated into ``userName eq "<value>"``.
_ACCOUNT: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._%+\-]{0,63}(@[A-Za-z0-9.\-]{1,190}\.[A-Za-z]{2,63})?$"
)


def canonical_ip(value: str) -> str:
    """One host address, canonically spelled. Refuses ranges and non-routable forms."""
    text = value.strip()
    if "/" in text:
        try:
            ipaddress.ip_network(text, strict=False)
        except ValueError as exc:
            raise TargetRejected(f"{text!r} is not an IP address") from exc
        raise TargetRejected(
            f"block target {text!r} is a range; approvals are for one address, and a "
            "range needs its own reviewed change"
        )
    # Leading-zero octets are ambiguous (octal in some stacks, decimal in others), so
    # the address one parser blocks is not the address another one routes.
    if re.search(r"(^|\.)0\d", text):
        raise TargetRejected(f"ambiguous address {text!r}: leading zeros in an octet")
    try:
        address = ipaddress.ip_address(text)
    except ValueError as exc:
        raise TargetRejected(f"{text!r} is not an IP address") from exc
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    if address.is_unspecified:
        raise TargetRejected(f"{address} is the unspecified address, i.e. everything")
    if address.is_loopback:
        raise TargetRejected(f"{address} is loopback; the enforcement point would block itself")
    if address.is_multicast:
        raise TargetRejected(f"{address} is multicast, not a host")
    if address.is_link_local:
        raise TargetRejected(f"{address} is link-local and not routable past one segment")
    if isinstance(address, ipaddress.IPv4Address) and address == ipaddress.IPv4Address(
        "255.255.255.255"
    ):
        raise TargetRejected("255.255.255.255 is the broadcast address")
    return str(address)


def canonical_host(value: str) -> str:
    """A hostname (RFC 1123) or a single IP address, canonically spelled."""
    text = value.strip()
    try:
        return canonical_ip(text)
    except TargetRejected as exc:
        if "is not an IP address" not in str(exc):
            raise
    if len(text) > 253 or not text:
        raise TargetRejected(f"hostname {text[:40]!r} is too long")
    labels = text.rstrip(".").split(".")
    if not all(_HOST_LABEL.match(label) for label in labels):
        raise TargetRejected(f"{text!r} is not a valid hostname")
    if labels[-1].isdigit():
        raise TargetRejected(f"{text!r} looks numeric but is not a valid IP address")
    return ".".join(labels).lower()


def canonical_account(value: str) -> str:
    """A directory account name. An IP address or hostname is refused, not guessed at."""
    text = value.strip()
    try:
        ipaddress.ip_address(text)
    except ValueError:
        pass
    else:
        raise TargetRejected(
            f"{text!r} is an IP address, not an account; disabling an account needs "
            "the account's identifier, and the alert carried a host"
        )
    if not _ACCOUNT.match(text):
        raise TargetRejected(f"{text!r} is not a valid account identifier")
    return text


@dataclass(frozen=True, slots=True)
class TargetPolicy:
    """The customer's protected infrastructure, plus per-type validation.

    ``protected_networks`` is a set of CIDR ranges and ``protected_hosts`` a set of
    names and accounts; anything matching is refused for every destructive action.
    Both default to empty because the right list is the customer's, not ours, and an
    invented default would be believed.
    """

    protected_networks: tuple[str, ...] = ()
    protected_hosts: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        for network in self.protected_networks:
            ipaddress.ip_network(network, strict=False)  # raises on a typo, early
        object.__setattr__(
            self,
            "protected_hosts",
            frozenset(host.strip().lower() for host in self.protected_hosts),
        )

    def validate(self, action_type: ActionType, target: str) -> str:
        """Return the canonical target, or raise :class:`TargetRejected`."""
        if action_type is ActionType.BLOCK_IP:
            canonical = canonical_ip(target)
        elif action_type in (
            ActionType.ISOLATE_HOST,
            ActionType.KILL_PROCESS,
            ActionType.QUARANTINE_FILE,
        ):
            canonical = canonical_host(target)
        elif action_type is ActionType.DISABLE_ACCOUNT:
            canonical = canonical_account(target)
        else:
            return target.strip()
        self._refuse_protected(action_type, canonical)
        return canonical

    def _refuse_protected(self, action_type: ActionType, canonical: str) -> None:
        if canonical.lower() in self.protected_hosts:
            raise TargetRejected(
                f"{canonical} is on the protected list; {action_type.value} against "
                "it is refused regardless of approval"
            )
        try:
            address = ipaddress.ip_address(canonical)
        except ValueError:
            return
        for network in self.protected_networks:
            if address in ipaddress.ip_network(network, strict=False):
                raise TargetRejected(
                    f"{canonical} is inside protected network {network}; "
                    f"{action_type.value} against it is refused regardless of approval"
                )
