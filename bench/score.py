"""Score candidate sentence-level classifiers over a paragraph set.

Sentence-level accuracy hides what a user sees, and a single global AUC hides a
model that only works on one generator. So this reports three things at once:

* span-level metrics (accuracy / AUC / Brier / calibration direction),
* the same metrics after `aggregate()` — the document-level number a UI shows,
* a per-source table, because that is where a "good" model is usually revealed
  to be topic-matched rather than general.

Model and label direction always come from the command line. There is no
built-in list of "the good models": a harness that quietly ships a favourite is
not a harness, it is an opinion with arguments.

Example:
    python bench/score.py --data bench/out/evalset.jsonl \\
      --model e5=some/e5-detector:1 --model distil=some/distil-detector:0 \\
      --ensemble "e5=0.6,distil=0.4" --limit 120 --out bench/out/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from textscore.aggregate import Config, SpanScore, aggregate  # noqa: E402
from textscore.metrics import by_group, evaluate, project_doc_seconds  # noqa: E402
from textscore.segment import split_sentences  # noqa: E402


def parse_model_arg(raw: str) -> tuple[str, str, int | None]:
    """``name=repo:ai_index`` -> ("name", "repo", ai_index or None)."""
    name, _, rest = raw.partition("=")
    if not name or not rest:
        raise SystemExit(f"--model must look like name=repo[:ai_index], got {raw!r}")
    repo, _, idx = rest.rpartition(":")
    if not repo:  # no colon -> the whole remainder is the repo
        repo, idx = rest, ""
    return name.strip(), repo.strip(), int(idx) if idx.strip().lstrip("-").isdigit() else None


def resolve_ai_position(id2label: dict, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    # Never guess from the repo name. Read the label text, and refuse outright
    # when even that is ambiguous (generic LABEL_0/LABEL_1 tells you nothing).
    hints = ("ai", "generated", "synthetic", "machine", "gpt", "llm")
    anti = ("non", "not", "human", "real", "true", "authentic", "original")

    def matches(label: object) -> bool:
        low = str(label).lower().replace("_", " ")
        return any(h in low for h in hints) and not any(a in low for a in anti)

    hits = {int(i) for i, label in id2label.items() if matches(label)}
    if len(hits) != 1:
        raise SystemExit(
            f"cannot resolve the positive class from id2label={id2label}; "
            "pass it explicitly as name=repo:<ai_index>"
        )
    return hits.pop()


def load_rows(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if limit:
        per: dict[str, int] = {}
        kept = []
        for r in rows:
            if per.get(r["source"], 0) < limit:
                kept.append(r)
                per[r["source"]] = per.get(r["source"], 0) + 1
        rows = kept
    return rows


def expand(values: list, per_row: list[list]) -> list:
    """Repeat a per-paragraph value once per sentence of that paragraph."""
    return [v for v, sents in zip(values, per_row) for _ in sents]


def doc_spans(sents: list, probs: list[float]) -> list[SpanScore]:
    return [SpanScore(s.text, s.start, s.end, p, s.words) for s, p in zip(sents, probs)]


def score_spans(
    repo: str, ai_index: int | None, texts: list[str], batch: int, threads: int
) -> tuple[list[float], dict]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo)
    model.eval()
    ai_pos = resolve_ai_position(model.config.id2label, ai_index)

    probs: list[float] = []
    per_item_seconds: list[float] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            enc = tok(chunk, truncation=True, max_length=512, padding=True, return_tensors="pt")
            t0 = time.perf_counter()
            out = model(**enc).logits
            dt = time.perf_counter() - t0
            probs.extend(out.softmax(dim=-1)[:, ai_pos].tolist())
            per_item_seconds.extend([dt / len(chunk)] * len(chunk))

    meta = {
        "repo": repo,
        "ai_index": ai_pos,
        "id2label": {str(k): v for k, v in model.config.id2label.items()},
        "median_ms_per_sentence": round(1000 * sorted(per_item_seconds)[len(per_item_seconds) // 2], 3),
    }
    return probs, meta


def evaluate_model(name: str, probs: list[float], rows: list[dict], per_row: list[list], threshold: float) -> dict:
    span_labels = expand([1 if r["label"] == "ai" else 0 for r in rows], per_row)
    span_groups = expand([r["source"] for r in rows], per_row)
    doc_scores = []
    confidence_counts: dict[str, int] = {}
    i = 0
    for sents in per_row:
        chunk = probs[i : i + len(sents)]
        i += len(sents)
        doc = aggregate(doc_spans(sents, chunk), Config())
        doc_scores.append(doc.pos_percent / 100.0)
        confidence_counts[doc.confidence] = confidence_counts.get(doc.confidence, 0) + 1

    row_labels = [1 if r["label"] == "ai" else 0 for r in rows]
    row_groups = [r["source"] for r in rows]
    entry: dict = {
        "span": evaluate(probs, span_labels, threshold),
        "span_by_source": by_group(probs, span_labels, span_groups, threshold),
        "doc": {**evaluate(doc_scores, row_labels, threshold), "confidence_counts": confidence_counts},
        "doc_by_source": by_group(doc_scores, row_labels, row_groups, threshold),
    }
    if entry["span"].get("inverted"):
        print(f"  WARNING: {name} scores negatives above positives -- wrong ai_index?", file=sys.stderr)
    print(
        f"  span acc={entry['span']['accuracy']} auc={entry['span']['auc']} | "
        f"doc acc={entry['doc']['accuracy']} auc={entry['doc']['auc']}",
        flush=True,
    )
    return entry


def parse_ensemble(raw: str, scored: dict) -> dict[str, float]:
    weights: dict[str, float] = {}
    for part in raw.split(","):
        key, _, w = part.partition("=")
        weights[key.strip()] = float(w)
    missing = set(weights) - set(scored)
    if missing:
        raise SystemExit(f"--ensemble names models that were not scored: {sorted(missing)}")
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise SystemExit(f"--ensemble weights must sum to 1, got {sum(weights.values())}")
    return weights


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="jsonl from build_set.py")
    p.add_argument("--model", action="append", required=True, help="name=repo[:ai_index], repeatable")
    p.add_argument("--ensemble", help='weighted span-probability blend, e.g. "a=0.6,b=0.4"')
    p.add_argument("--limit", type=int, default=None, help="paragraphs kept per source")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--sentences-per-doc", type=float, default=250.0, help="document size for the latency projection")
    p.add_argument("--out", type=Path, default=Path("bench/out/metrics.json"))
    args = p.parse_args()

    specs = [parse_model_arg(m) for m in args.model]
    rows = load_rows(args.data, args.limit)
    per_row = [split_sentences(r["text"]) for r in rows]
    flat = [s.text for sents in per_row for s in sents]
    print(f"{len(rows)} paragraphs -> {len(flat)} sentences")

    report: dict[str, dict] = {}
    span_probs: dict[str, list[float]] = {}
    for name, repo, ai_index in specs:
        print(f"scoring {name} ({repo}) ...", flush=True)
        probs, meta = score_spans(repo, ai_index, flat, args.batch, args.threads)
        span_probs[name] = probs
        report[name] = {"model": meta} | evaluate_model(name, probs, rows, per_row, args.threshold)
        report[name]["cpu_seconds_per_doc"] = project_doc_seconds(
            meta["median_ms_per_sentence"] / 1000, args.sentences_per_doc
        )

    if args.ensemble:
        weights = parse_ensemble(args.ensemble, span_probs)
        blended = [sum(w * span_probs[k][i] for k, w in weights.items()) for i in range(len(flat))]
        report["ensemble"] = {"weights": weights} | evaluate_model(
            "ensemble", blended, rows, per_row, args.threshold
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
