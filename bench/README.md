# bench/

The evaluation harness. It is separate from the package on purpose: the core
never needs `torch`, and an evaluation script that can be imported by accident
is an evaluation script that will be run in production by accident.

```bash
uv sync --extra bench          # torch + transformers + datasets
uv run pytest -q tests/test_bench_args.py   # arg parsing, no model needed
```

## Two steps

```bash
# 1. any Hugging Face dataset, columns declared as name=ai|human
python bench/build_set.py \
  --dataset your/org --split train \
  --columns "human_column=human,gpt4_column=ai,claude_column=ai" \
  --prompt-column instruction --per-source 200 --seed 20261003 \
  --out bench/out/evalset.jsonl

# 2. score one or more checkpoints, optionally blended
python bench/score.py --data bench/out/evalset.jsonl \
  --model a=some/repo:1 --model b=some/other-repo:0 \
  --ensemble "a=0.6,b=0.4" --limit 120 --out bench/out/metrics.json
```

## Reading the output

For each model you get `span` and `doc` blocks plus `*_by_source` tables:

* **Always read `by_source` before the global number.** A model that scores 0.95
  overall while one generator carries it is not a model, it is a topic detector.
* **`inverted: true` means the label axis is backwards.** You picked the wrong
  logits column. The run also warns on stderr. This is the most common way to
  burn a day on a "plausible" score.
* **`span` vs `doc` disagreeing is the interesting part.** Good span scores with
  a bad `doc` number usually means short sentences are carrying errors that the
  word-weighted average should have absorbed — tune `Config`, not the model.
* **`cpu_seconds_per_doc` projects the per-item batch median onto a realistic
  document.** Batch medians and single-shot latencies are different numbers; only
  the projection is comparable to a user-facing latency budget.

## What this harness is not

No model list, no dataset list, no leaderboard, and no committed eval set. A
derived corpus inherits its upstream licence, so only the recipe is version
controlled (`bench/out/` is gitignored). Rebuilding a set with the same `--seed`
and the same upstream dataset reproduces it exactly.
