"""Paint span scores back onto the source text.

The point of keeping character offsets through segmentation and aggregation is
to render highlights without re-deriving them. Two renderers are provided:

* :func:`render_html` — escaped ``<span>``s with a band class and the raw
  probability in a data attribute. Untrusted text cannot inject markup.
* :func:`render_ansi` — terminal output for quick eyeballing.

Both take the *original* text plus spans, and assert the spans still land
inside it, because an off-by-one in offset bookkeeping is invisible until a
customer sees the highlight drift by a word.
"""

from __future__ import annotations

import html
from collections.abc import Sequence

from .aggregate import DocumentScore, SpanScore
from .segment import Sentence, split_sentences

_BAND_CSS = """
.ts-highlight { background: none; border-radius: 2px; padding: 0 1px; }
.ts-high { background: #fee2e2; }
.ts-medium { background: #fef3c7; }
.ts-low { background: #dcfce7; }
""".strip()

_ANSI = {"high": "\033[41;97m", "medium": "\033[43;30m", "low": "\033[42;30m"}
_ANSI_RESET = "\033[0m"


def spans_from_sentences(sentences: Sequence[Sentence], probs: Sequence[float]) -> list[SpanScore]:
    """Zip a splitter output and a model's per-sentence probabilities."""
    if len(sentences) != len(probs):
        raise ValueError(f"{len(sentences)} sentences but {len(probs)} probabilities")
    return [
        SpanScore(text=s.text, start=s.start, end=s.end, p=p, words=s.words)
        for s, p in zip(sentences, probs)
    ]


def _check_spans(text: str, spans: Sequence[SpanScore]) -> None:
    for i, s in enumerate(spans):
        if s.start < 0 or s.end > len(text) or s.start >= s.end:
            raise ValueError(f"span {i} out of range: [{s.start}, {s.end}) for len {len(text)}")
        covered = text[s.start : s.end]
        if not covered.strip():
            raise ValueError(f"span {i} covers only whitespace: {covered!r}")
        # whitespace is normalised because callers may carry a trimmed copy of the source
        if _norm(covered) != _norm(s.text):
            raise ValueError(f"span {i} text does not match source: {s.text[:20]!r} vs {covered[:20]!r}")


def _norm(value: str) -> str:
    return " ".join(value.split())


def render_html(text: str, doc: DocumentScore, *, css: bool = True, title_template: str = "{p:.0%}") -> str:
    """Return ``text`` with each scored span wrapped in a band-coloured ``<span>``.

    Gaps between spans are emitted as escaped plain text, so the output always
    reconstructs the full document.
    """
    spans = sorted(doc.spans, key=lambda s: s.start)
    _check_spans(text, spans)
    out: list[str] = []
    if css:
        out.append(f"<style>{_BAND_CSS}</style>")
    cursor = 0
    for s in spans:
        if s.start > cursor:
            out.append(html.escape(text[cursor : s.start], quote=False))
        body = text[s.start : s.end]
        out.append(
            f'<span class="ts-highlight ts-{s.band}" data-p="{s.p:.4f}" '
            f'title="{html.escape(title_template.format(p=s.p), quote=True)}">'
            f"{html.escape(body, quote=False)}</span>"
        )
        cursor = s.end
    if cursor < len(text):
        out.append(html.escape(text[cursor:], quote=False))
    return "".join(out)


def render_ansi(text: str, doc: DocumentScore) -> str:
    spans = sorted(doc.spans, key=lambda s: s.start)
    _check_spans(text, spans)
    out: list[str] = []
    cursor = 0
    for s in spans:
        if s.start > cursor:
            out.append(text[cursor : s.start])
        out.append(f"{_ANSI.get(s.band, _ANSI_RESET)}{text[s.start : s.end]}{_ANSI_RESET}")
        cursor = s.end
    if cursor < len(text):
        out.append(text[cursor:])
    return "".join(out)


def segment_and_render(text: str, probs: Sequence[float], *, renderer: str = "ansi") -> str:
    """One-liner for the README: split, score with an external model, render."""
    from .aggregate import aggregate

    doc = aggregate(spans_from_sentences(split_sentences(text), probs))
    return render_html(text, doc) if renderer == "html" else render_ansi(text, doc)
