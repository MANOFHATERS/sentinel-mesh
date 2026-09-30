"""The dashboard's "Real data" side: status, the ATT&CK knowledge base, and the run handlers.

Nothing here touches the synthetic demo. It is a second, separate way to exercise the same
components on real public data, so the two can be shown one after the other.
"""

from __future__ import annotations

import contextlib
import threading
from pathlib import Path
from typing import Any

from sentinel.kb.retrieve import KnowledgeBase
from sentinel.real.attack import AttackError, build_attack_kb, search_view
from sentinel.real.network import analyse_flows, detect_format, find_dataset
from sentinel.real.repos import scan_repository

__all__ = ["RealData"]

FETCH_COMMAND = "python scripts/fetch_real_data.py"


class RealData:
    def __init__(self, root: Path | str = ".", *, curated_kb: KnowledgeBase | None = None) -> None:
        self._root = Path(root)
        self._curated = curated_kb
        self._attack_path = self._root / "data" / "real" / "enterprise-attack.json"
        self._lock = threading.Lock()
        self._kb: KnowledgeBase | None = None
        self._kb_error: str | None = None

    # -- status ------------------------------------------------------------------ #

    def network_path(self) -> Path | None:
        return find_dataset(self._root)

    def status(self) -> dict[str, Any]:
        dataset = self.network_path()
        return {
            "network": {
                "available": dataset is not None,
                "file": None if dataset is None else dataset.name,
                "format": None if dataset is None else detect_format(dataset),
                "size_mb": None if dataset is None else round(dataset.stat().st_size / 1e6, 1),
            },
            "attack": {
                "available": self._attack_path.is_file(),
                "size_mb": round(self._attack_path.stat().st_size / 1e6, 1)
                if self._attack_path.is_file()
                else None,
                "built": self._kb is not None,
                "documents": None
                if self._kb is None or self._kb.corpus is None
                else len(self._kb.corpus),
                "error": self._kb_error,
            },
            "scan": {"enabled": True},
            "fetch_command": FETCH_COMMAND,
        }

    # -- ATT&CK knowledge base ----------------------------------------------------- #

    def attack_kb(self) -> KnowledgeBase:
        """Built once, on first use (or by :meth:`warm`), under a lock."""
        with self._lock:
            if self._kb is None:
                try:
                    self._kb = build_attack_kb(self._attack_path)
                except AttackError as exc:
                    self._kb_error = str(exc)
                    raise
            return self._kb

    def warm(self) -> None:
        """Build the index on a background thread so the first search is not slow."""
        if not self._attack_path.is_file():
            return

        def build() -> None:
            # A failure is reported through status(); it is never fatal to the server.
            with contextlib.suppress(Exception):
                self.attack_kb()

        threading.Thread(target=build, name="attack-kb", daemon=True).start()

    def kb_search(self, query: str, *, k: int = 5) -> dict[str, Any]:
        if not self._attack_path.is_file():
            return {"available": False, "real": [], "curated": [], "fetch_command": FETCH_COMMAND}
        real = search_view(self.attack_kb(), query, k=k)
        curated = search_view(self._curated, query, k=k) if self._curated is not None else []
        real_docs = len(self.attack_kb().corpus) if self.attack_kb().corpus is not None else None
        curated_docs = (
            len(self._curated.corpus)
            if self._curated is not None and self._curated.corpus
            else None
        )
        return {
            "available": True,
            "query": query,
            "real": real,
            "curated": curated,
            "real_documents": real_docs,
            "curated_documents": curated_docs,
        }

    # -- run handlers (called by the RunManager) ---------------------------------------- #

    def run_network(self, seed: int, params: dict[str, Any]) -> dict[str, Any]:
        path = self.network_path()
        if path is None:
            raise FileNotFoundError(f"no real flow dataset found; run {FETCH_COMMAND}")
        return analyse_flows(path, limit=int(params.get("limit") or 20_000), seed=seed)

    def run_scan(self, seed: int, params: dict[str, Any]) -> dict[str, Any]:
        return scan_repository(str(params.get("url", "")))
