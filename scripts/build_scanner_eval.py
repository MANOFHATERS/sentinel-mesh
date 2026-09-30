"""Measure the code scanner against real answer keys and save the result for the dashboard.

    python scripts/build_scanner_eval.py

Uses the real advisories already cached by ``build_real_supply_chain.py``: for each whose weakness
type the scanner has a rule for and that links a fix commit, the changed Python files are fetched
as they were before and at the fix and both are scanned. Also scans a real teaching project that
keeps a vulnerable ``bad/`` tree and a fixed ``good/`` tree. Needs no key; GitHub's unauthenticated
limit (60 requests an hour) is enough for a first run, and ``GITHUB_TOKEN`` (a free token with no
scopes) lifts it. Everything fetched is cached, so a re-run is offline.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from sentinel.real.scanner_eval import (
    RESULT_PATH,
    advisory_cases,
    evaluate_advisories,
    evaluate_paired_repo,
)
from sentinel.real.supplychain import CACHE_PATH


def main() -> int:
    started = time.time()
    if not CACHE_PATH.is_file():
        print("run scripts/build_real_supply_chain.py first (it caches the advisories used here)")
        return 1
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    cases = advisory_cases(cache)
    print(f"{len(cases)} real advisories with a covered weakness type and a fix commit")
    print("scanning a real project with paired vulnerable and fixed trees ...", flush=True)
    paired = evaluate_paired_repo("fportantier/vulpy")
    print(
        f"  {paired['pairs']} file pairs: {paired['bad_flagged']} vulnerable flagged, "
        f"{paired['good_flagged']} fixed flagged"
    )
    print("scanning real advisories before and after their fixes ...", flush=True)
    advisories = evaluate_advisories(cases, cache_path=Path("data/real/scanner-eval-cache.json"))
    print(
        f"  {advisories['cases']} cases: {advisories['hit']} caught before the fix, "
        f"{advisories['fixed_version_flagged']} still flagged after it"
    )
    result = {"generated_at": time.strftime("%Y-%m-%d"), "paired": paired, "advisories": advisories}
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result), encoding="utf-8")
    print(f"wrote {RESULT_PATH} in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
