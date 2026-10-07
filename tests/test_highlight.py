import re

import pytest

from textscore.aggregate import SpanScore, aggregate
from textscore.highlight import render_ansi, render_html, segment_and_render, spans_from_sentences
from textscore.segment import split_sentences

TEXT = "First sentence is here. Second one follows. Third is short."
PROBS = [0.92, 0.05, 0.6]


def _doc(text: str = TEXT, probs: list[float] | None = None):
    return aggregate(spans_from_sentences(split_sentences(text), probs or PROBS))


class TestHtml:
    def test_user_text_cannot_inject_markup(self):
        evil = '<script>alert("xss")</script> looks machine written.'
        sents = split_sentences(evil)
        doc = aggregate(spans_from_sentences(sents, [0.9] * len(sents)))
        out = render_html(evil, doc)
        assert "<script>" not in out
        assert "&lt;script&gt;" in out

    def test_output_reconstructs_the_source(self):
        out = render_html(TEXT, _doc(), css=False)
        assert re.sub(r"<[^>]+>", "", out) == TEXT

    def test_band_classes_and_probability_are_exposed(self):
        out = render_html(TEXT, _doc(), css=False)
        assert 'class="ts-highlight ts-high"' in out
        assert 'data-p="0.9200"' in out

    def test_css_block_is_optional(self):
        assert "<style>" in render_html(TEXT, _doc())
        assert "<style>" not in render_html(TEXT, _doc(), css=False)

    def test_quotes_in_title_are_escaped(self):
        doc = _doc()
        out = render_html(TEXT, doc, css=False, title_template='p="{p:.0%}"')
        assert "&quot;" in out


class TestAnsi:
    def test_spans_are_wrapped_in_colours(self):
        out = render_ansi(TEXT, _doc())
        assert "\033[41;97m" in out and out.endswith("\033[0m")

    def test_plain_text_is_preserved(self):
        out = render_ansi(TEXT, _doc())
        assert re.sub(r"\033\[[0-9;]*m", "", out) == TEXT


class TestOffsetGuardrails:
    def test_span_outside_the_source_raises(self):
        doc = aggregate([SpanScore("not in text", 500, 520, 0.9)])
        with pytest.raises(ValueError, match="out of range"):
            render_html(TEXT, doc)

    def test_span_that_does_not_match_its_offsets_raises(self):
        sents = split_sentences(TEXT)
        wrong = SpanScore("Totally different words.", sents[0].start, sents[0].end, 0.9)
        with pytest.raises(ValueError, match="does not match source"):
            render_ansi(TEXT, aggregate([wrong]))

    def test_whitespace_only_span_raises(self):
        doc = aggregate([SpanScore(" ", 23, 24, 0.9)])  # the gap between two sentences
        with pytest.raises(ValueError, match="whitespace"):
            render_html(TEXT, doc)

    def test_count_mismatch_between_spans_and_probs_raises(self):
        with pytest.raises(ValueError, match="probabilities"):
            spans_from_sentences(split_sentences(TEXT), [0.5])


def test_segment_and_render_glue():
    assert "ts-highlight" not in segment_and_render(TEXT, PROBS)  # ansi by default
    assert "ts-highlight" in segment_and_render(TEXT, PROBS, renderer="html")


def test_empty_span_list_still_renders_the_text():
    doc = aggregate([SpanScore("a b c.", 0, 6, 0.9)])
    assert re.sub(r"<[^>]+>", "", render_html("a b c. ", doc, css=False)) == "a b c. "
