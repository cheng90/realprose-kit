import pytest

from textscore.aggregate import Config, SpanScore, aggregate, band_of


def _span(text: str, p: float, start: int = 0) -> SpanScore:
    return SpanScore(text=text, start=start, end=start + len(text), p=p)


def _many(n: int, p: float) -> list[SpanScore]:
    # ~20-word spans: enough total words that length never caps the label,
    # which isolates the span-count factor
    body = " ".join(f"w{j}" for j in range(19))
    return [_span(f"s{i} {body}.", p, i * 120) for i in range(n)]


class TestWordWeighting:
    def test_long_span_dominates_as_it_should(self):
        doc = aggregate([_span(" ".join(["x"] * 100), 0.95), _span("no way", 0.05)])
        # equal-per-sentence averaging would give 50%; words are the unit of evidence
        assert doc.pos_percent > 80

    def test_half_and_half_lands_near_50(self):
        doc = aggregate(_many(12, 0.9) + _many(12, 0.1))
        assert 45 <= doc.pos_percent <= 55
        assert doc.pos_percent + doc.neg_percent == 100.0

    def test_empty_document(self):
        doc = aggregate([])
        assert (doc.pos_percent, doc.neg_percent, doc.confidence) == (0.0, 100.0, "low")
        assert doc.flagged_ratio == 0.0

    def test_words_total(self):
        assert aggregate([_span("one two three", 0.5)]).words == 3


class TestConfidenceLabelling:
    def test_one_span_is_always_low(self):
        assert aggregate([_span("only one sentence here.", 0.9)]).confidence == "low"

    def test_many_decisive_spans_reach_high(self):
        assert aggregate(_many(30, 0.93)).confidence == "high"

    def test_few_but_decisive_spans_cap_at_medium(self):
        assert aggregate(_many(10, 0.9)).confidence == "medium"

    def test_uniformly_unsure_spans_are_low_not_medium(self):
        # every span sits exactly on the decision line: the aggregate number
        # is a coin flip wearing a percentage costume
        doc = aggregate(_many(30, 0.5))
        assert doc.confidence == "low"

    def test_conflicted_but_spread_out_spans_are_medium(self):
        cfg = Config()
        spans = [
            _span(f"s{i} " + " ".join(f"w{j}" for j in range(19)) + ".", 0.5 if i % 2 else 0.95)
            for i in range(30)
        ]
        doc = aggregate(spans, cfg)
        # half the spans hug the boundary, but dispersion > low_dispersion
        assert doc.confidence == "medium"

    def test_too_few_words_is_low(self):
        spans = [_span("a b c d.", 0.95) for _ in range(10)]  # 10 spans, 50 words
        assert aggregate(spans).confidence == "low"


class TestBands:
    def test_bands_follow_config(self):
        cfg = Config(band_high=0.6, band_low=0.3)
        doc = aggregate([_span("a b c d e f.", 0.65), _span("g h i j k l.", 0.45), _span("m n o p q r.", 0.2)], cfg)
        assert [s.band for s in doc.spans] == ["high", "medium", "low"]

    def test_band_of_matches_aggregate(self):
        cfg = Config(band_high=0.8, band_low=0.4)
        assert band_of(0.8, cfg) == "high" and band_of(0.39, cfg) == "low"

    def test_flagged_ratio_counts_top_band_only(self):
        doc = aggregate([_span("a b c.", 0.9), _span("d e f.", 0.1)])
        assert doc.flagged_ratio == 0.5


class TestConfig:
    def test_env_override_with_default_prefix(self):
        cfg = Config.from_env({"TEXTSCORE_BAND_HIGH": "0.8", "TEXTSCORE_MIN_SPANS_HIGH": "40"})
        assert cfg.band_high == 0.8 and cfg.min_spans_high == 40

    def test_custom_prefix_and_untouched_defaults(self):
        cfg = Config.from_env({"X_BAND_HIGH": "0.9"}, prefix="X_")
        assert cfg.band_high == 0.9 and cfg.band_low == Config().band_low

    def test_malformed_value_raises_instead_of_silently_defaulting(self):
        with pytest.raises(ValueError):
            Config.from_env({"TEXTSCORE_BAND_HIGH": "not-a-float"})

    def test_as_dict_roundtrip(self):
        doc = aggregate(_many(3, 0.7))
        raw = doc.as_dict()
        assert set(raw) == {"pos_percent", "neg_percent", "confidence", "spans"}
        assert set(raw["spans"][0]) == {"text", "start", "end", "p", "band"}
