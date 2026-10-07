"""Rule-based English sentence segmentation with character offsets.

Sentence-classifier UIs need two things that `nltk.sent_tokenize` does not
give you: the *offsets* of each sentence in the original string (so a score
can be painted back onto the source without re-joining it) and stable
behaviour around abbreviations, decimals and closing quotes.

This module is a small deterministic splitter for exactly that purpose. It is
deliberately not a general-purpose tokenizer: it assumes prose, keeps the
original character positions, and folds sub-threshold fragments into the
previous sentence so a highlight range never renders as a stray period.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Tokens that end with a period without ending a sentence.
ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "eg",
        "ie", "fig", "al", "inc", "ltd", "co", "no", "approx", "dept", "est",
        "vol", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct",
        "nov", "dec", "us", "uk",
    }
)

_TERMINATORS = ".!?"


@dataclass(frozen=True)
class Sentence:
    """A sentence plus its exact [start, end) span in the source text."""

    text: str
    start: int
    end: int

    @property
    def words(self) -> int:
        return max(1, len(self.text.split()))

    @property
    def length(self) -> int:
        return self.end - self.start


def _is_abbrev_end(text: str, dot_idx: int, abbreviations: frozenset[str]) -> bool:
    """True when the '.' at ``dot_idx`` finishes an abbreviation, not a sentence."""
    i = dot_idx
    while i > 0 and (text[i - 1].isalnum() or text[i - 1] == "."):
        i -= 1
    token = text[i:dot_idx].lower().replace(".", "")
    if not token:
        return False
    if token in abbreviations:
        return True
    # single capital letters: an initial, e.g. "J. Smith" or "J.R.R. Tolkien"
    parts = text[i:dot_idx].split(".")
    return bool(parts) and all(len(p) == 1 and p.isalpha() for p in parts if p)


def split_sentences(
    text: str,
    *,
    min_fragment_words: int = 3,
    max_merge_words: int = 25,
    abbreviations: frozenset[str] = ABBREVIATIONS,
) -> list[Sentence]:
    """Split ``text`` into sentences, keeping spans into the original string.

    Args:
        text: source prose.
        min_fragment_words: a sentence shorter than this is folded into the
            previous one (``"Yes."`` after a question is not a useful highlight).
        max_merge_words: never fold into an already-long sentence; that would
            let one fragment steal a whole paragraph's score.
        abbreviations: lowercase token set treated as sentence-internal.

    Returns:
        Sentences in source order. Empty or whitespace-only input yields ``[]``.
    """
    if not text.strip():
        return []

    raw: list[Sentence] = []
    start = 0
    n = len(text)
    idx = 0
    while idx < n:
        ch = text[idx]
        if ch in _TERMINATORS:
            prev = text[idx - 1] if idx > 0 else ""
            nxt = text[idx + 1] if idx + 1 < n else ""
            if ch == "." and prev.isdigit() and nxt.isdigit():
                idx += 1  # decimal point, not a terminator
                continue
            if ch == "." and _is_abbrev_end(text, idx, abbreviations):
                idx += 1
                continue
            end = idx + 1
            while end < n and text[end] in "'\")":
                end += 1  # absorb the closing quote/paren of `"Go now."`
            if end == n or text[end].isspace():
                piece = text[start:end].strip()
                if piece:
                    s = text.index(piece, start)
                    raw.append(Sentence(piece, s, s + len(piece)))
                start = end
                while start < n and text[start].isspace():
                    start += 1
                idx = start
                continue
        idx += 1

    tail = text[start:].strip()
    if tail:
        s = text.index(tail, start)
        raw.append(Sentence(tail, s, s + len(tail)))

    merged: list[Sentence] = []
    for sent in raw:
        if merged and sent.words < min_fragment_words and merged[-1].words < max_merge_words:
            prev = merged[-1]
            merged[-1] = Sentence(text[prev.start : sent.end].strip(), prev.start, sent.end)
        else:
            merged.append(sent)
    return merged


def word_count(text: str) -> int:
    """Whitespace word count, matching the classifier's own notion of length."""
    return len(text.split())
