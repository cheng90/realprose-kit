"""Build a labelled paragraph set from any Hugging Face dataset.

The harness needs aligned positives and negatives: same prompts, human answer
and model answer, so that "the model looks good" cannot be an artefact of topic
drift. That alignment is a property of the *dataset*, not of this script -- you
declare it with ``--columns name=label`` and this does the filtering, sampling
and provenance bookkeeping.

Nothing is downloaded that you did not ask for, and the output is never committed:
derived corpora inherit their upstream licence, so the recipe is public and the
artifact is not.

Example:
    python bench/build_set.py \\
      --dataset your/org --split train \\
      --columns "human=human,gpt4_output=ai,claude_output=ai" \\
      --prompt-column instruction --per-source 200 --seed 20261003 \\
      --out bench/out/evalset.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from textscore.segment import split_sentences  # noqa: E402


def parse_columns(raw: str) -> list[tuple[str, str]]:
    """``"a=human,b=ai"`` -> [("a", "human"), ("b", "ai")]."""
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        col, _, label = item.partition("=")
        if not label or label not in ("ai", "human"):
            raise SystemExit(f"--columns entries must be name=ai or name=human, got {item!r}")
        out.append((col.strip(), label))
    return out


def keep(text: object, args: argparse.Namespace) -> bool:
    if not isinstance(text, str):
        return False
    words = len(text.split())
    return args.min_words <= words <= args.max_words and len(split_sentences(text)) >= args.min_sentences


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="Hugging Face dataset id")
    p.add_argument("--config", default=None, help="dataset configuration name")
    p.add_argument("--split", default="train")
    p.add_argument("--columns", required=True, help="comma-separated column=label pairs; label is ai|human")
    p.add_argument("--prompt-column", default=None, help="optional column copied into each row as `prompt`")
    p.add_argument("--per-source", type=int, default=200, help="max paragraphs kept per column")
    p.add_argument("--min-words", type=int, default=30)
    p.add_argument("--max-words", type=int, default=250)
    p.add_argument("--min-sentences", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261003)
    p.add_argument("--out", type=Path, default=Path("bench/out/evalset.jsonl"))
    args = p.parse_args()

    from datasets import load_dataset  # imported late so the core stays dependency-free

    columns = parse_columns(args.columns)
    ds = load_dataset(args.dataset, args.config, split=args.split) if args.config else load_dataset(args.dataset, split=args.split)
    print(f"{args.dataset}: {len(ds)} rows")

    rng = random.Random(args.seed)
    rows: list[dict] = []
    for col, label in columns:
        if col not in ds.column_names:
            raise SystemExit(f"column {col!r} not in {args.dataset}; available: {ds.column_names}")
        candidates = [r for r in ds if keep(r.get(col), args)]
        rng.shuffle(candidates)
        picked = candidates[: args.per_source]
        for i, r in enumerate(picked):
            rows.append(
                {
                    "id": f"{args.dataset.replace('/', '-').replace('@', '')}-{col}-{i}",
                    "source": col,
                    "dataset": args.dataset,
                    "label": label,
                    "prompt": r.get(args.prompt_column) if args.prompt_column else None,
                    "text": r[col],
                }
            )
        print(f"  {col} ({label}): kept {len(picked)} of {len(candidates)} eligible")

    rng.shuffle(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "total": len(rows),
        "by_label": {k: sum(1 for r in rows if r["label"] == k) for k in ("human", "ai")},
        "by_source": {c: sum(1 for r in rows if r["source"] == c) for c, _ in columns},
        "dataset": args.dataset,
        "seed": args.seed,
        "filters": {"min_words": args.min_words, "max_words": args.max_words, "min_sentences": args.min_sentences},
    }
    (args.out.parent / "evalset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
