"""The tenants' workspaces, some of which may still be being built.

The synthetic demo's workspaces exist before the server accepts a request. The real-data
workspace does not: it trains a model on a real capture and indexes MITRE's ATT&CK catalogue,
which takes about a minute, and holding the whole server back for that would delay everyone
who only wants the synthetic demo. So the registry declares the real tenant up front, serves
it as *preparing* (a 503 that says so, and that the page retries) until the build finishes,
and reports a build that failed as such rather than hiding it.

It is a read-only ``Mapping`` of tenant id to :class:`Workspace`, so the code that resolves
a workspace from an identity is unchanged.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from sentinel.dashboard.workspace import Workspace, WorkspaceError, workspace_features

__all__ = ["WorkspaceRegistry", "WorkspaceUnavailable"]


class WorkspaceUnavailable(WorkspaceError):
    """The tenant's workspace is still being built, or its build failed."""

    def __init__(self, message: str, *, failed: bool = False) -> None:
        super().__init__(message)
        self.failed = failed


_PRETTY = {"unsw-nb15": "UNSW-NB15", "cic-ids2017": "CIC-IDS2017"}
_ALL = {"scenarios": True, "supply_chain": True, "code_scan": True, "models": True, "kb": False}


@dataclass(slots=True)
class _Declared:
    mode: str
    dataset: str
    state: str = "preparing"  # preparing | failed
    error: str | None = None


class WorkspaceRegistry(Mapping[str, Workspace]):
    def __init__(self, ready: Mapping[str, Workspace] | None = None) -> None:
        self._lock = threading.Lock()
        self._ready: dict[str, Workspace] = dict(ready or {})
        self._declared: dict[str, _Declared] = {}

    # -- building ---------------------------------------------------------------- #

    def declare(self, tenant: str, *, mode: str, dataset: str) -> None:
        with self._lock:
            self._declared[tenant] = _Declared(mode, dataset)

    def set_ready(self, tenant: str, workspace: Workspace) -> None:
        with self._lock:
            self._ready[tenant] = workspace
            self._declared.pop(tenant, None)

    def set_failed(self, tenant: str, error: str) -> None:
        with self._lock:
            if tenant in self._declared:
                self._declared[tenant].state = "failed"
                self._declared[tenant].error = error

    # -- Mapping ----------------------------------------------------------------- #

    def __getitem__(self, tenant: str) -> Workspace:
        with self._lock:
            if tenant in self._ready:
                return self._ready[tenant]
            declared = self._declared.get(tenant)
        if declared is None:
            raise KeyError(tenant)
        if declared.state == "failed":
            raise WorkspaceUnavailable(
                f"the {declared.dataset} workspace could not be built: {declared.error}",
                failed=True,
            )
        raise WorkspaceUnavailable(
            f"the {declared.dataset} workspace is being prepared (training on the real data "
            "and indexing the ATT&CK catalogue, about a minute); this page retries on its own"
        )

    def __contains__(self, tenant: object) -> bool:
        # Mapping's default calls __getitem__, which raises for a workspace that is still being
        # built; a tenant that is declared *is* in the registry.
        with self._lock:
            return tenant in self._ready or tenant in self._declared

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter([*self._ready, *self._declared])

    def __len__(self) -> int:
        with self._lock:
            return len(self._ready) + len(self._declared)

    # -- what the API needs -------------------------------------------------------- #

    def ready_items(self) -> list[tuple[str, Workspace]]:
        with self._lock:
            return list(self._ready.items())

    def describe(self, tenant: str) -> dict[str, Any]:
        """``{"mode", "dataset", "state", "error"}`` for a tenant, without building anything."""
        with self._lock:
            workspace = self._ready.get(tenant)
            declared = self._declared.get(tenant)
        if workspace is not None:
            models = getattr(workspace, "models", None)
            report = getattr(models, "real_report", None) or {}
            return {
                "mode": getattr(models, "mode", "synthetic"),
                "dataset": _PRETTY.get(report.get("dataset"), report.get("dataset")),
                "state": "ready",
                "error": None,
                "features": workspace_features(models) if models is not None else _ALL,
            }
        if declared is not None:
            return {
                "mode": declared.mode,
                "dataset": declared.dataset,
                "state": declared.state,
                "error": declared.error,
                "features": {**_ALL, "kb": declared.mode == "real"},
            }
        return {
            "mode": "synthetic",
            "dataset": None,
            "state": "ready",
            "error": None,
            "features": _ALL,
        }
