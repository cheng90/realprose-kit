"""`textscore` CLI — inspect segmentation and aggregate pre-computed span scores.

The CLI never runs a model. Scoring is your job (see ``bench/``); this is the
glue you can poke at from a shell when a highlight looks wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .aggregate import Config, SpanScore, aggregate
from .highlight import render_ansi, render_html, spans_from_sentences
from .segment import split_sentences


def _read(path: str) -> str:
    return sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")


def cmd_split(args: argparse.Namespace) -> int:
    text = _read(args.file)
    sentences = split_sentences(text, min_fragment_words=args.min_fragment_words)
    if args.json:
        print(json.dumps([{"text": s.text, "start": s.start, "end": s.end, "words": s.words} for s in sentences], ensure_ascii=False, indent=2))
    else:
        for s in sentences:
            print(f"[{s.start:>6}:{s.end:>6}] {s.words:>3}w  {s.text}")
        print(f"\n{len(sentences)} sentences, {sum(s.words for s in sentences)} words")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Input: {"text": "...", "probs": [0.9, 0.1, ...]} aligned with split_sentences()."""
    payload = json.loads(_read(args.input))
    text = payload["text"]
    probs = payload["probs"]
    sentences = split_sentences(text)
    if len(sentences) != len(probs):
        print(
            f"error: {len(sentences)} sentences after splitting, but {len(probs)} probabilities given",
            file=sys.stderr,
        )
        return 2
    cfg = Config.from_env() if args.from_env else Config()
    doc = aggregate([SpanScore(s.text, s.start, s.end, p, s.words) for s, p in zip(sentences, probs)], cfg)
    if args.html:
        Path(args.html).write_text(render_html(text, doc), encoding="utf-8")
        print(f"wrote {args.html}", file=sys.stderr)
    elif args.render:
        print(render_ansi(text, doc))
    else:
        out = {k: v for k, v in doc.as_dict().items() if k != "spans"}
        out["flagged_ratio"] = round(doc.flagged_ratio, 3)
        out["words"] = doc.words
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_demo(_: argparse.Namespace) -> int:
    text = (
        "Dr. Chalk reviewed the draft. The value is 3.14 times larger than before, "
        "which the reviewer flagged as suspicious. Yes. He also asked for a source, "
        "and the author promised to add one before Friday."
    )
    # three spans after fragment folding -- "Yes." merged into the sentence before it
    probs = [0.91, 0.05, 0.62]
    doc = aggregate(spans_from_sentences(split_sentences(text), probs))
    print(render_ansi(text, doc))
    print(f"\npos {doc.pos_percent}% | confidence {doc.confidence} | bands {[s.band for s in doc.spans]}")
    print("confidence stays low: 3 spans is not enough evidence to claim more")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="textscore", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("split", help="show sentence spans with character offsets")
    s.add_argument("file", nargs="?", default="-", help="text file, or - for stdin")
    s.add_argument("--json", action="store_true")
    s.add_argument("--min-fragment-words", type=int, default=3, dest="min_fragment_words")
    s.set_defaults(fn=cmd_split)

    c = sub.add_parser("score", help="aggregate {text, probs} JSON into a document score")
    c.add_argument("input", nargs="?", default="-", help="json file, or - for stdin")
    c.add_argument("--render", action="store_true", help="print ANSI-highlighted source instead of JSON")
    c.add_argument("--html", help="write an HTML fragment with band classes to this path")
    c.add_argument("--from-env", action="store_true", help="read TEXTSCORE_* thresholds from the environment")
    c.set_defaults(fn=cmd_score)

    d = sub.add_parser("demo", help="run the built-in example")
    d.set_defaults(fn=cmd_demo)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
