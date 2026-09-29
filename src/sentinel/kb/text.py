"""Text analysis for the knowledge base: tokenize, stem, bigram.

Why this file is hand-written rather than borrowed
--------------------------------------------------
Retrieval quality in a security knowledge base is decided almost entirely in the
tokenizer, and every off-the-shelf tokenizer gets two things wrong for this corpus:

1.  **It destroys identifiers.** A default word tokenizer turns ``CVE-2021-44228``
    into ``cve``, ``2021``, ``44228`` and ``T1021.002`` into ``t1021``, ``002``.
    ``T1021`` and ``T1021.002`` then become the *same* token, which is the one
    distinction an analyst query most needs preserved. Identifiers are matched by a
    protected pattern pass before generic tokenization, so they survive whole.
2.  **It drops the sub-words of compounds.** ``lsass.exe`` must match a query for
    ``lsass``, and ``ua-parser-js`` must match ``parser``. So a compound token is
    emitted *both* whole and split, and the whole form carries the extra weight that
    an exact compound match deserves for free (it is one more shared feature).

Stemming is a faithful implementation of Porter *step 1 only* (1a, 1b, 1c). That is
a deliberate stopping point. Step 1 handles plurals and verb inflection, which is
where the recall is — ``credentials``/``credential``, ``scanning``/``scan``,
``encrypted``/``encrypt``. Steps 2 through 4 strip derivational suffixes and would
map ``credential`` to ``credenti`` and ``persistence`` to ``persist``, conflating
security terms whose distinctions matter while buying almost nothing on a corpus
this size. Truncating a published algorithm at a documented step is honest;
inventing a new one is not.

Determinism note
----------------
Nothing here uses :func:`hash`. Python randomizes string hashing per process
(``PYTHONHASHSEED``), so a feature index derived from :func:`hash` would produce a
different vector on every run and an index saved by one process would silently
mis-rank when loaded by another. ``test_kb_text.py`` runs the tokenizer in a
subprocess with a different hash seed and asserts byte-identical output.
"""

from __future__ import annotations

import re
from itertools import pairwise
from typing import Final

__all__ = [
    "IDENTIFIER_PATTERN",
    "STOPWORDS",
    "analyze",
    "bigrams",
    "porter_step1",
    "tokenize",
]

#: Identifiers that must survive tokenization intact. Matched before generic
#: tokenization and emitted verbatim (lowercased), so ``T1021`` and ``T1021.002``
#: stay distinct features. Ordered longest-form-first inside each alternative.
IDENTIFIER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"""
    (?:
        CVE-\d{4}-\d{4,7}           # CVE-2021-44228
      | GHSA-[A-Za-z0-9-]{4,60}     # GHSA-EVENT-STREAM-2018
      | PB-[A-Z][A-Z-]{2,40}        # PB-LATERAL-SMB
      | T\d{4}\.\d{3}               # T1021.002 (before the bare form)
      | T\d{4}                      # T1021
      | TA\d{4}                     # TA0008
      | MS\d{2}-\d{3}               # MS17-010
      | CWE-\d{1,4}                 # CWE-79
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

#: A generic token: alphanumeric runs joined by internal ``.``, ``-``, ``_`` or ``/``.
#: The internal-separator form is what keeps ``lsass.exe`` and ``ua-parser-js`` whole.
_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+(?:[._\-/][a-z0-9]+)*")

_SEPARATORS: Final[re.Pattern[str]] = re.compile(r"[._\-/]")

#: Function words carry no retrieval signal and would dominate bigram features.
#: Deliberately *excludes* security-loaded short words that a generic list drops:
#: ``no``, ``not``, ``own``, ``off``, ``out``, ``over``, ``one``, ``all``, ``any``
#: are absent because "not signed", "one account", "over the channel" and
#: "out of band" are meaningful in this corpus.
STOPWORDS: Final[frozenset[str]] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "doing", "done", "have", "has", "had", "having",
    "of", "in", "on", "at", "to", "for", "from", "by", "with", "without", "into", "onto",
    "it", "its", "as", "so", "such",
    "i", "we", "you", "they", "he", "she", "them", "us", "our", "your", "their",
    "there", "here", "where", "when", "which", "who", "whom", "whose", "what", "why", "how",
    "can", "could", "should", "would", "may", "might", "must", "will", "shall",
    "also", "very", "just", "only", "too", "more", "most", "some", "many", "much",
    "about", "after", "before", "during", "while", "until", "through",
    "because", "since", "though", "although", "however", "therefore", "thus", "hence",
})

_VOWELS: Final[frozenset[str]] = frozenset("aeiou")


def _is_consonant(word: str, index: int) -> bool:
    """Porter's consonant test: ``y`` is a consonant unless preceded by one."""
    char = word[index]
    if char in _VOWELS:
        return False
    if char != "y":
        return True
    return index == 0 or not _is_consonant(word, index - 1)


def _measure(stem: str) -> int:
    """Porter's ``m``: the number of vowel-consonant sequences in ``stem``."""
    count = 0
    previous_consonant = True
    for index in range(len(stem)):
        consonant = _is_consonant(stem, index)
        if consonant and not previous_consonant:
            count += 1
        previous_consonant = consonant
    return count


def _contains_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, index) for index in range(len(stem)))


def _ends_double_consonant(stem: str) -> bool:
    return (
        len(stem) >= 2
        and stem[-1] == stem[-2]
        and _is_consonant(stem, len(stem) - 1)
    )


def _ends_cvc(stem: str) -> bool:
    """True when ``stem`` ends consonant-vowel-consonant, last not in ``wxy``."""
    if len(stem) < 3:
        return False
    return (
        _is_consonant(stem, len(stem) - 3)
        and not _is_consonant(stem, len(stem) - 2)
        and _is_consonant(stem, len(stem) - 1)
        and stem[-1] not in "wxy"
    )


def porter_step1(word: str) -> str:
    """Porter stemmer step 1 (plurals and past/progressive inflection) only.

    Steps 2-4 are deliberately not applied; see the module docstring. Words shorter
    than three characters are returned unchanged, because the rules are unsound on
    them (``as`` would become ``a``) and nothing in this corpus needs it.
    """
    if len(word) < 3:
        return word

    # --- 1a: plurals -----------------------------------------------------------
    if word.endswith("sses") or word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("ss"):
        pass
    elif word.endswith("s"):
        word = word[:-1]

    # --- 1b: -eed / -ed / -ing -------------------------------------------------
    # The nested conditionals below are flagged as collapsible and are kept nested on
    # purpose: they transcribe the published step-1b rules one to one, and each rule's
    # suffix test and its vowel test are separate steps in the algorithm. Flattening
    # them makes the code no shorter and the correspondence to the source unreadable,
    # which for a transcribed algorithm is the property worth protecting.
    cleanup = False
    if word.endswith("eed"):
        if _measure(word[:-1]) > 0:
            word = word[:-1]
    elif word.endswith("ed"):
        if _contains_vowel(word[:-2]):
            word = word[:-2]
            cleanup = True
    elif word.endswith("ing"):  # noqa: SIM102
        if _contains_vowel(word[:-3]):
            word = word[:-3]
            cleanup = True

    if cleanup:
        if word.endswith(("at", "bl", "iz")):
            word += "e"
        elif _ends_double_consonant(word) and not word.endswith(("l", "s", "z")):
            word = word[:-1]
        elif _measure(word) == 1 and _ends_cvc(word):
            word += "e"

    # --- 1c: terminal y -> i ---------------------------------------------------
    if word.endswith("y") and len(word) > 2 and _contains_vowel(word[:-1]):
        word = word[:-1] + "i"

    return word


def tokenize(text: str, *, stem: bool = True, drop_stopwords: bool = True) -> list[str]:
    """Tokenize ``text`` into retrieval features.

    Identifiers (``CVE-2021-44228``, ``T1021.002``) are emitted verbatim and never
    stemmed. Compound tokens are emitted whole *and* split on internal separators,
    so ``lsass.exe`` yields ``lsass.exe``, ``lsass`` and ``exe``.
    """
    tokens: list[str] = []

    # Identifiers first, then blanked out of the string so the generic pass cannot
    # re-tokenize their pieces into ``cve``/``2021``/``44228``.
    def _blank(match: re.Match[str]) -> str:
        tokens.append(match.group(0))
        return " " * len(match.group(0))

    remainder = IDENTIFIER_PATTERN.sub(_blank, text.lower())

    for match in _TOKEN_PATTERN.finditer(remainder):
        raw = match.group(0)
        parts = _SEPARATORS.split(raw) if _SEPARATORS.search(raw) else []
        candidates = [raw, *parts] if parts else [raw]
        for candidate in candidates:
            if not candidate or (candidate.isdigit() and len(candidate) < 4):
                # Bare short numbers ("3", "60") are noise; years and ports are kept.
                continue
            if drop_stopwords and candidate in STOPWORDS:
                continue
            tokens.append(porter_step1(candidate) if stem else candidate)

    return tokens


def bigrams(tokens: list[str]) -> list[str]:
    """Adjacent token pairs, joined with ``_``.

    Computed *after* stopword removal, which is what makes ``pass the hash`` and
    ``pass-the-hash`` produce the same ``pass_hash`` feature. The cost is false
    adjacency across a removed word; on phrase-heavy security text that trade has
    always gone the same way, and :mod:`sentinel.kb.eval` measures it rather than
    assuming it.
    """
    return [f"{first}_{second}" for first, second in pairwise(tokens)]


def analyze(text: str) -> list[str]:
    """The full feature stream for one piece of text: unigrams then bigrams."""
    unigrams = tokenize(text)
    return [*unigrams, *bigrams(unigrams)]
