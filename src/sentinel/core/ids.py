"""Identifier generation.

Two flavours, both explicit:

``new_id``
    A random UUID4 for genuinely new objects (a fresh ActionRequest).

``deterministic_id``
    A UUID5 derived from a namespace plus stable source fields. Replaying the
    same dataset row twice must produce the *same* alert_id, or the pipeline
    cannot be idempotent and the demo cannot be reproducible. This is what makes
    "run the evaluation twice, get identical output" achievable.
"""

from __future__ import annotations

import uuid

# Project-private namespace root. Derived from a fixed name so it is stable
# forever; regenerating it would silently change every deterministic id.
SENTINEL_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "sentinel-mesh.invalid")

_NAMESPACES: dict[str, uuid.UUID] = {}


def namespace(name: str) -> uuid.UUID:
    """Return (and memoize) a stable child namespace, e.g. ``namespace("alert")``."""
    if not name:
        raise ValueError("namespace name must be non-empty")
    cached = _NAMESPACES.get(name)
    if cached is None:
        cached = uuid.uuid5(SENTINEL_NAMESPACE, name)
        _NAMESPACES[name] = cached
    return cached


def new_id() -> str:
    """A fresh random identifier."""
    return str(uuid.uuid4())


def deterministic_id(kind: str, *parts: object) -> str:
    """Derive a stable identifier from ``kind`` and ``parts``.

    Parts are joined with an ASCII unit separator (0x1f) so that
    ``("a|b",)`` and ``("a", "b")`` cannot collide the way a ``"|".join``
    would allow.
    """
    if not parts:
        raise ValueError("deterministic_id requires at least one part")
    seed = "\x1f".join(str(p) for p in parts)
    return str(uuid.uuid5(namespace(kind), seed))
