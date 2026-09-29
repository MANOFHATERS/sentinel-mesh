# `vulnerable_app` — a deliberately insecure fixture

**Do not deploy, import, or copy any of this.** Every file here contains planted
security defects. It exists so PRD **F-07** — *"at least 3 seeded vulnerabilities
detected and a syntactically valid patch PR opened for each"* — is measured against
code with known answers rather than asserted.

Nothing in this directory is imported by `sentinel`. It is read as **text** by
`sentinel.scan.repo.RepoSnapshot.from_dir` and parsed with `ast`; it is never
executed, and the libraries it imports (`flask`, `yaml`, `requests`, `jinja2`) are
not dependencies of this project.

## The ground truth lives in the source

Each planted defect carries a marker comment on the line the scanner must flag:

```python
cur.execute(f"SELECT * FROM t WHERE a = '{name}'")  # SEEDED: python.sql-injection
```

and each correct construction that a naive scanner would flag anyway carries:

```python
cur.execute("SELECT * FROM t WHERE a = ?", (name,))  # SAFE: python.sql-injection
```

`sentinel.scan.seeded.load_seed_manifest` parses those markers, so the expected
result is derived from the file rather than recorded in a second place that drifts
away from it. Moving a line moves its own ground truth with it, and a planted
defect that loses its marker shows up as a false positive rather than silently
passing.

`SAFE` markers are the half that decides whether the scanner is usable. A rule that
catches every seeded case and also flags the correct version next to it has a recall
of 1.0 and no value: the first thing a team does with a scanner that cries wolf is
switch it off. `scripts/evaluate.py --codescan` reports recall **and** the
false-positive count on the controls, and fails the gate on either.
