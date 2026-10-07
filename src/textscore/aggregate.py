"""Turn per-span probabilities into one document number plus an honesty label.

Two problems show up the moment a sentence-level classifier gets a UI:

1. **Averaging by sentence count is wrong.** One 200-word sentence and nine
   20-word sentences should not count the same; weight by words or the longest
   sentence silently owns the document score.
2. **A bare percentage over-claims.** 92% on a two-sentence snippet and 92% on
   a two-page essay do not mean the same thing, and a classifier that is
   *confidently inconsistent* across spans is a different signal from one that
   is *consistently* confident.

So this module pairs every aggregate number with a coarse ``confidence`` label
derived from span count, document length and probability dispersion. Every
threshold is a :class:`Config` field, not a constant: these are presentation
tunables, and the right values depend on the model behind them.
"""

from __future__ import annotations

import os
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

CONFIDENCE_LEVELS = ("high", "medium", "low")


@dataclass
class Config:
    """Aggregation and labelling thresholds. Defaults are starting points, not truths."""

    #: span probability at or above this counts as "flagged"
    flag_threshold: float = 0.5
    #: highlight bands: >= band_high is "high", >= band_low is "medium", else "low"
    band_high: float = 0.75
    band_low: float = 0.35
    #: span count required before a document may be labelled "high" confidence
    min_spans_high: int = 20
    #: below this span count the label can only ever be "low"
    min_spans_medium: int = 8
    #: same idea, measured in words
    min_words_medium: int = 150
    #: probability stdev under which *mixed* spans mean the model is conflicted
    low_dispersion: float = 0.10
    #: share of spans within +-boundary_tolerance of the flag threshold that
    #: makes the aggregate number untrustworthy
    conflict_ratio: float = 0.35
    boundary_tolerance: float = 0.15

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, prefix: str = "TEXTSCORE_") -> Config:
        """Build from ``TEXTSCORE_BAND_HIGH=0.8`` style variables.

        Unknown variables are ignored; malformed values raise, because a typo'd
        threshold silently reverting to a default is worse than a crash.
        """
        cfg = cls()
        env = os.environ if env is None else env
        for key in vars(cfg):
            raw = env.get(f"{prefix}{key.upper()}")
            if raw is not None:
                setattr(cfg, key, type(getattr(cfg, key))(raw))
        return cfg


@dataclass
class SpanScore:
    """One scored span: its text, its offsets, and the model's probability."""

    text: str
    start: int
    end: int
    p: float
    words: int = 0
    band: str = ""  # assigned by aggregate() from Config

    def __post_init__(self) -> None:
        if not self.words:
            self.words = max(1, len(self.text.split()))


@dataclass
class DocumentScore:
    #: word-weighted mean of the positive-class probability, in percent
    pos_percent: float
    neg_percent: float
    confidence: str
    spans: list[SpanScore] = field(default_factory=list)

    @property
    def flagged_ratio(self) -> float:
        """Share of spans in the top band -- a shape signal the mean hides."""
        if not self.spans:
            return 0.0
        return sum(1 for s in self.spans if s.band == "high") / len(self.spans)

    @property
    def words(self) -> int:
        return sum(s.words for s in self.spans)

    def as_dict(self) -> dict:
        return {
            "pos_percent": self.pos_percent,
            "neg_percent": self.neg_percent,
            "confidence": self.confidence,
            "spans": [
                {"text": s.text, "start": s.start, "end": s.end, "p": round(s.p, 4), "band": s.band}
                for s in self.spans
            ],
        }


def band_of(p: float, cfg: Config | None = None) -> str:
    cfg = cfg or Config()
    if p >= cfg.band_high:
        return "high"
    if p >= cfg.band_low:
        return "medium"
    return "low"


def aggregate(spans: Sequence[SpanScore], cfg: Config | None = None) -> DocumentScore:
    """Word-weighted mean of span probabilities, plus a confidence label."""
    cfg = cfg or Config()
    if not spans:
        return DocumentScore(pos_percent=0.0, neg_percent=100.0, confidence="low")

    for s in spans:
        s.band = band_of(s.p, cfg)

    total_words = sum(s.words for s in spans)
    weighted = sum(s.p * s.words for s in spans) / max(1, total_words)
    pos_percent = round(weighted * 100, 1)

    probs = [s.p for s in spans]
    dispersion = statistics.pstdev(probs) if len(probs) > 1 else 0.0
    near_boundary = sum(1 for p in probs if abs(p - cfg.flag_threshold) <= cfg.boundary_tolerance) / len(probs)

    n = len(spans)
    if n < cfg.min_spans_medium or total_words < cfg.min_words_medium:
        # too little evidence for any stable reading
        confidence = "low"
    elif near_boundary > cfg.conflict_ratio:
        # many spans sit right on the decision line. If the spread is also tight
        # the model is not disagreeing -- it is uniformly unsure, which is the
        # weakest possible signal for an aggregate number.
        confidence = "medium" if dispersion > cfg.low_dispersion else "low"
    elif n >= cfg.min_spans_high:
        confidence = "high"
    else:
        # decisive spans, but not enough of them to earn "high"
        confidence = "medium"

    return DocumentScore(
        pos_percent=pos_percent,
        neg_percent=round(100 - pos_percent, 1),
        confidence=confidence,
        spans=list(spans),
    )
