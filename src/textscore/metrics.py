"""Scoring metrics a classifier evaluation harness actually needs, in pure Python.

``sklearn.metrics.roc_auc_score`` is fine, but an evaluation script that only
runs with a scientific stack attached is harder to reason about and slower to
install. These helpers cover the narrow slice used to compare candidate
sentence-level models: ranking quality, threshold behaviour, calibration
direction, and a latency projection from batch timings to whole documents.

The AUC implementation is the rank-based Mann-Whitney form (O(n log n)) with
average ranks for ties, so it matches ``roc_auc_score`` on tie-heavy
probabilities where the naive pair counting is both slow and unstable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def _average_ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks over the ascending values, ties sharing the mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # positions i..j are 0-based -> ranks +1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def roc_auc(scores: Sequence[float], labels: Sequence[int | float]) -> float | None:
    """P(score of a random positive > a random negative); 0.5 means no signal.

    Returns ``None`` when one class is missing -- AUC is undefined there, and a
    fabricated 0.0 or 1.0 would silently corrupt a comparison table.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must be the same length")
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    ranks = _average_ranks(scores)
    rank_sum_pos = sum(r for r, y in zip(ranks, labels) if y)
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def accuracy_at(scores: Sequence[float], labels: Sequence[int | float], threshold: float = 0.5) -> float:
    if not scores:
        raise ValueError("no scores")
    return sum(1 for s, y in zip(scores, labels) if (s >= threshold) == bool(y)) / len(scores)


def confusion_at(scores: Sequence[float], labels: Sequence[int | float], threshold: float = 0.5) -> dict[str, int]:
    out = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for s, y in zip(scores, labels):
        pred = s >= threshold
        if y and pred:
            out["tp"] += 1
        elif y:
            out["fn"] += 1
        elif pred:
            out["fp"] += 1
        else:
            out["tn"] += 1
    return out


def brier(scores: Sequence[float], labels: Sequence[int | float]) -> float:
    """Mean squared error of the probabilities. Lower is better; 0.25 = coin flip."""
    if not scores:
        raise ValueError("no scores")
    return sum((s - y) ** 2 for s, y in zip(scores, labels)) / len(scores)


@dataclass
class ClassMeans:
    """Calibration direction: which way does the model drift?"""

    pos_mean: float | None
    neg_mean: float | None

    @property
    def separation(self) -> float | None:
        if self.pos_mean is None or self.neg_mean is None:
            return None
        return round(self.pos_mean - self.neg_mean, 4)

    @property
    def inverted(self) -> bool:
        """True when negatives score higher than positives -- the label axis is flipped.

        Worth asserting on in any harness: a mislabelled class index produces a
        plausible-looking number in the wrong direction, and it is the single
        easiest way to waste a day.
        """
        return self.separation is not None and self.separation < 0


def class_means(scores: Sequence[float], labels: Sequence[int | float]) -> ClassMeans:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    return ClassMeans(
        pos_mean=round(sum(pos) / len(pos), 4) if pos else None,
        neg_mean=round(sum(neg) / len(neg), 4) if neg else None,
    )


def evaluate(
    scores: Sequence[float],
    labels: Sequence[int | float],
    threshold: float = 0.5,
) -> dict:
    """One-call summary used by the bench scripts."""
    means = class_means(scores, labels)
    return {
        "n": len(scores),
        "accuracy": round(accuracy_at(scores, labels, threshold), 4),
        "auc": (round(v, 4) if (v := roc_auc(scores, labels)) is not None else None),
        "brier": round(brier(scores, labels), 4),
        "pos_mean": means.pos_mean,
        "neg_mean": means.neg_mean,
        "separation": means.separation,
        "inverted": means.inverted,
        **confusion_at(scores, labels, threshold),
    }


def by_group(
    scores: Sequence[float],
    labels: Sequence[int | float],
    groups: Sequence[str],
    threshold: float = 0.5,
) -> dict[str, dict]:
    """Per-source breakdown. Aggregate accuracy hides a model that only works on one generator."""
    if not (len(scores) == len(labels) == len(groups)):
        raise ValueError("scores, labels and groups must be the same length")
    out: dict[str, dict] = {}
    for group in dict.fromkeys(groups):
        sel = [i for i, g in enumerate(groups) if g == group]
        gs = [scores[i] for i in sel]
        gl = [labels[i] for i in sel]
        out[group] = {
            "n": len(sel),
            "mean_score": round(sum(gs) / len(gs), 4),
            "accuracy": round(accuracy_at(gs, gl, threshold), 4),
            "label": "pos" if all(gl) else "neg" if not any(gl) else "mixed",
        }
    return out


def project_doc_seconds(median_batch_seconds_per_item: float, sentences_per_doc: float) -> float:
    """Project a per-sentence timing onto a whole document.

    Benchmarks that only report "ms per inference call" get misread: batched CPU
    inference amortises differently per item than a single-shot call does, so the
    number that matters for a latency budget is the per-item median times the
    sentence count of a realistic document.
    """
    return round(median_batch_seconds_per_item * sentences_per_doc, 3)
