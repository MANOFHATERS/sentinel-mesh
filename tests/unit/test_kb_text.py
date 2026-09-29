"""Tokenization and stemming (:mod:`sentinel.kb.text`).

Retrieval quality is decided here, so these tests are about the two properties the
rest of the knowledge base assumes and cannot check for itself:

*   **Identifiers survive.** ``T1021`` and ``T1021.002`` must remain distinct
    features, and ``CVE-2021-44228`` must not shatter into ``cve``/``2021``/``44228``.
    Every one of those is a silent recall failure, not an error.
*   **Output is process-independent.** A feature index derived from :func:`hash`
    would differ per process, so an index built in one process would mis-rank in
    another. :class:`TestDeterminismAcrossProcesses` runs the analyzer in a
    subprocess under a different ``PYTHONHASHSEED`` and demands byte-identical
    output, which is the only way to catch a reintroduced :func:`hash` call.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sentinel.kb.text import (
    IDENTIFIER_PATTERN,
    STOPWORDS,
    analyze,
    bigrams,
    porter_step1,
    tokenize,
)

SRC = str(Path(__file__).resolve().parents[2] / "src")


class TestIdentifiersSurvive:
    """The single most important tokenizer property for this corpus."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("CVE-2021-44228", "cve-2021-44228"),
            ("cve-2021-44228", "cve-2021-44228"),
            ("T1021.002", "t1021.002"),
            ("T1021", "t1021"),
            ("GHSA-EVENT-STREAM-2018", "ghsa-event-stream-2018"),
            ("PB-LATERAL-SMB", "pb-lateral-smb"),
            ("MS17-010", "ms17-010"),
            ("CWE-79", "cwe-79"),
            ("TA0008", "ta0008"),
        ],
    )
    def test_identifier_emitted_verbatim(self, text: str, expected: str) -> None:
        assert expected in tokenize(text)

    def test_identifier_not_shattered_into_parts(self) -> None:
        tokens = tokenize("exploited via CVE-2021-44228 yesterday")
        assert "cve-2021-44228" in tokens
        # The pieces must not leak out as separate features: "2021" would collide
        # every CVE from that year into one bucket.
        assert "cve" not in tokens
        assert "44228" not in tokens
        assert "2021" not in tokens

    def test_technique_and_subtechnique_are_distinct_features(self) -> None:
        """The one distinction an analyst query most needs preserved."""
        assert tokenize("T1021") != tokenize("T1021.002")
        assert "t1021" in tokenize("T1021")
        assert "t1021.002" in tokenize("T1021.002")
        assert "t1021" not in tokenize("T1021.002")

    def test_identifiers_are_never_stemmed(self) -> None:
        """Porter step 1a would strip the trailing 's' of an id ending in one."""
        assert "ghsa-colors-faker-2022" in tokenize("GHSA-COLORS-FAKER-2022")

    def test_pattern_prefers_the_longer_technique_form(self) -> None:
        match = IDENTIFIER_PATTERN.search("T1021.002")
        assert match is not None
        assert match.group(0) == "T1021.002"


class TestCompoundSplitting:
    def test_compound_emitted_whole_and_split(self) -> None:
        tokens = tokenize("lsass.exe", stem=False)
        assert "lsass.exe" in tokens
        assert "lsass" in tokens
        assert "exe" in tokens

    def test_hyphenated_name_matches_a_part_query(self) -> None:
        assert "parser" in tokenize("ua-parser-js", stem=False)

    def test_exact_compound_match_shares_more_features(self) -> None:
        """An exact compound match should outweigh a partial one, for free."""
        document = set(tokenize("ua-parser-js was compromised"))
        exact = set(tokenize("ua-parser-js"))
        partial = set(tokenize("parser"))
        assert len(document & exact) > len(document & partial)


class TestPorterStep1:
    @pytest.mark.parametrize(
        "word,expected",
        [
            # 1a: plurals
            ("credentials", "credential"),
            ("accesses", "access"),
            ("queries", "queri"),
            ("access", "access"),
            ("caress", "caress"),
            # 1b: -ed / -ing, with the cleanup rules
            ("encrypted", "encrypt"),
            ("scanning", "scan"),
            ("moved", "move"),
            ("hopping", "hop"),
            ("failing", "fail"),
            ("agreed", "agree"),
            # 1c: terminal y
            ("query", "queri"),
        ],
    )
    def test_known_stems(self, word: str, expected: str) -> None:
        assert porter_step1(word) == expected

    def test_plural_and_singular_converge(self) -> None:
        """This is the entire point of stemming here."""
        for singular, plural in [
            ("credential", "credentials"),
            ("query", "queries"),
            ("account", "accounts"),
            ("ticket", "tickets"),
        ]:
            assert porter_step1(singular) == porter_step1(plural)

    def test_short_words_untouched(self) -> None:
        for word in ("as", "is", "a", "os"):
            assert porter_step1(word) == word

    def test_derivational_suffixes_are_left_alone(self) -> None:
        """Steps 2-4 are deliberately not applied; assert they are not creeping in."""
        assert porter_step1("credential") == "credential"
        assert porter_step1("persistence") == "persistence"
        assert porter_step1("national") == "national"

    def test_idempotent(self) -> None:
        """Stemming a stem must not change it, or repeated analysis would drift."""
        for word in ("credentials", "scanning", "queries", "encrypted", "moved"):
            once = porter_step1(word)
            assert porter_step1(once) == once


class TestStopwords:
    def test_function_words_dropped(self) -> None:
        tokens = tokenize("the account was compromised by an attacker")
        assert "the" not in tokens
        assert "wa" not in tokens and "was" not in tokens

    def test_security_loaded_short_words_are_kept(self) -> None:
        """A generic stoplist drops these; here they carry meaning."""
        for word in ("not", "no", "one", "out", "over", "own", "all", "any", "off"):
            assert word not in STOPWORDS, f"{word!r} must not be a stopword"

    def test_stopwords_can_be_kept(self) -> None:
        assert "the" in tokenize("the account", drop_stopwords=False)


class TestBigrams:
    def test_phrase_forms_converge_after_stopword_removal(self) -> None:
        """``pass the hash`` and ``pass-the-hash`` must produce the same feature."""
        assert "pass_hash" in analyze("pass the hash")
        assert "pass_hash" in analyze("pass-the-hash")

    def test_bigrams_are_adjacent_pairs(self) -> None:
        assert bigrams(["a", "b", "c"]) == ["a_b", "b_c"]

    def test_single_token_has_no_bigrams(self) -> None:
        assert bigrams(["only"]) == []

    def test_analyze_is_unigrams_then_bigrams(self) -> None:
        unigrams = tokenize("lateral movement detected")
        assert analyze("lateral movement detected")[: len(unigrams)] == unigrams


class TestNoiseFiltering:
    def test_bare_short_numbers_dropped(self) -> None:
        tokens = tokenize("retry 3 times over 60 seconds")
        assert "3" not in tokens
        assert "60" not in tokens

    def test_years_and_ports_kept(self) -> None:
        tokens = tokenize("in 2021 on port 8080")
        assert "2021" in tokens
        assert "8080" in tokens


class TestRobustness:
    """The analyzer is fed attacker-influenced text and must never raise."""

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "!!!???",
            "\x00\x01\x02",
            "‮evil‬",
            "a" * 20_000,
            "🔥 ransomware 🔥",
            "CVE-" * 500,
        ],
    )
    def test_never_raises(self, text: str) -> None:
        assert isinstance(analyze(text), list)

    def test_empty_text_yields_no_tokens(self) -> None:
        assert analyze("") == []


class TestDeterminismAcrossProcesses:
    """The guard against a reintroduced :func:`hash` call.

    ``hash("x")`` differs between processes unless ``PYTHONHASHSEED`` is fixed, so a
    feature index built from it would change every run. That failure is invisible
    within one process, which is why this test needs a subprocess.
    """

    PROBE = (
        "import json,sys;"
        "sys.path.insert(0, sys.argv[1]);"
        "from sentinel.kb.text import analyze;"
        "print(json.dumps(analyze("
        "'attacker used CVE-2021-44228 and T1021.002 to move laterally via lsass.exe'"
        ")))"
    )

    def _run(self, seed: str) -> list[str]:
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", self.PROBE, SRC],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        return json.loads(result.stdout)

    def test_identical_under_different_hash_seeds(self) -> None:
        assert self._run("0") == self._run("12345") == analyze(
            "attacker used CVE-2021-44228 and T1021.002 to move laterally via lsass.exe"
        )
