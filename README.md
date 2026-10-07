# textscore

> Repo: [`cheng90/realprose-kit`](https://github.com/cheng90/realprose-kit) · Python package: `textscore`

**Post-model plumbing for span-level text classifiers: sentence segmentation with
offsets, word-weighted aggregation, an honest confidence label, HTML highlighting
and leak-safe request logging.** Zero runtime dependencies.

You ran the model. You have one probability per sentence. Now the hard part starts:
how do those numbers become a single headline percentage, how much should that
percentage be trusted, how do you paint it back onto the exact characters it came
from, and how do you log the request without logging somebody's draft?

That is the boring, unforgiving layer every text-analysis product rebuilds from
scratch. This package is my version of it, extracted and cleaned up from
production pipeline code (a sentence-level AI-content classifier UI) and stripped
of everything model-specific.

```python
from textscore import aggregate, spans_from_sentences, split_sentences, render_html

text = open("draft.txt").read()
sentences = split_sentences(text)                       # -> spans with offsets
probs = my_model.predict([s.text for s in sentences])   # you provide this

doc = aggregate(spans_from_sentences(sentences, probs))
doc.pos_percent   # 61.3   word-weighted, so one long sentence cannot own the score
doc.confidence    # "high" | "medium" | "low" -- how much to believe the number
render_html(text, doc)  # escaped <span class="ts-high" data-p="0.9200">…</span>
```

## Install

Not published on PyPI yet; install from the repo:

```bash
git clone https://github.com/cheng90/realprose-kit.git
cd realprose-kit
uv sync            # or: pip install -e .
uv run pytest -q   # 110+ tests, no network, no model downloads
```

Requires Python 3.12+.

## What is in here

| Module | Does | Why it is not trivial |
| --- | --- | --- |
| `segment` | Rule-based English sentence splitter that keeps `[start, end)` into the original string | Abbreviations (`Dr.`, `e.g.`), decimals (`3.14`), initials (`J. R. Tolkien`), closing quotes (`"Go now."`) and sub-threshold fragments that would render as a stray highlight |
| `aggregate` | Per-span probabilities → one document number + band per span + confidence label | Averaging per sentence is wrong; and a percentage printed on two short sentences claims more than the model knows |
| `highlight` | Spans back into HTML or ANSI | The input is untrusted user text, so escaping is not optional; offsets are validated before rendering, because an off-by-one in offset bookkeeping is invisible until a customer sees the highlight drift |
| `metrics` | AUC (rank-based, ties handled), accuracy, confusion, Brier, per-group breakdown, calibration direction | Pure Python, `O(n log n)`, and `None` instead of a fabricated number when one class is absent. Also flags an inverted label axis — the easiest way to lose a day to a "plausible" score from the wrong logits column |
| `logsafe` | Whitelist-enforcing structured event logging | "Don't log the text field" is a denylist and denylists lose: fields get renamed, exception messages carry the document, nested dicts stringify. Here a prose-length value *raises* instead of being silently truncated |
| `cli` | `textscore split / score / demo` | For poking at a highlight that looks wrong, from a shell, without a notebook |

Every threshold in `aggregate.Config` is a field, and `Config.from_env()` reads
`TEXTSCORE_*` variables. Defaults are starting points for tuning against your own
eval set, not claimed truths.

## Evaluating a model with it

`bench/` holds the harness that produced the numbers behind the aggregation
choices. It is deliberately **model-agnostic and dataset-agnostic** — you pass
both in, it never ships a curated list of either:

```bash
uv sync --extra bench   # torch + transformers + datasets

# 1. build a labelled paragraph set from any Hugging Face dataset
python bench/build_set.py --dataset your/org --split train \
    --columns "prompt-aligned-human=human,generated-gpt4=ai" \
    --per-source 200 --seed 20261003 --out bench/out/evalset.jsonl

# 2. score it with your own checkpoint(s); metrics and latency are reported per source
python bench/score.py --data bench/out/evalset.jsonl \
    --model name=<repo-or-local-path>:<ai_label_index> \
    --ensemble "a=0.6,b=0.4" --out bench/out/metrics.json
```

`score.py` asserts the calibration direction before reporting anything, and prints
per-generator breakdowns — an aggregate accuracy of 0.95 that comes entirely from
one source is the failure mode this table exists to reveal.

## Design notes worth arguing about

**Word weighting, not sentence weighting.** A 200-word sentence and nine 20-word
sentences are not equal evidence. Weight by words or the longest sentence quietly
captures the document score.

**Confidence is a separate field from the percentage.** They answer different
questions and must never be conflated into "92% sure". The label considers span
count, document length, dispersion and how many spans sit near the decision
boundary — a document whose spans are *uniformly* borderline gets `low`, because
that is the model saying it does not know, averaged.

**Refuse, don't truncate.** A logging helper that clips long values to 64 chars
will happily log the first 64 chars of a private document and look clean in code
review. `EventLogger` raises `TextLeakError` and makes the caller deal with it.

**Explicit class direction.** Every model in the wild orders its labels
differently. The harness takes the positive-class index as an argument and checks
the resulting direction, instead of inferring it from a repo name.

## Non-goals

- No models, no tokenizers, no downloads. `textscore` core is standard library only.
- Not a general NLP suite: the splitter assumes English prose and is honestly worse
  on code, tables and dialogue-heavy text than a trained sentencizer.
- No statistics claiming to be calibration. The confidence label is a heuristic and
  documented as one; if you need calibrated probabilities, calibrate.

## Status

Extracted from production use, covered by tests, and still small on purpose. The
`pyproject.toml` version is `0.1.0` and the API may move.

## License

MIT.
