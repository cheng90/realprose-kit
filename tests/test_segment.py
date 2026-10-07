import pytest

from textscore.segment import split_sentences


class TestBasics:
    def test_simple_prose(self):
        got = [s.text for s in split_sentences("Hello world. This is a test! Are you sure? Yes.")]
        # "Yes." is a 1-word fragment and folds into the question it answers
        assert got == ["Hello world.", "This is a test!", "Are you sure? Yes."]

    def test_empty_input(self):
        assert split_sentences("   \n\t ") == []
        assert split_sentences("") == []

    def test_single_sentence(self):
        assert len(split_sentences("Just one long sentence about something interesting.")) == 1

    def test_no_terminator_at_all(self):
        assert [s.text for s in split_sentences("trailing fragment without punctuation")] == [
            "trailing fragment without punctuation"
        ]


class TestGuards:
    def test_decimal_is_not_a_boundary(self):
        assert len(split_sentences("The value is 3.14 times larger.")) == 1

    def test_abbreviation_does_not_split(self):
        assert len(split_sentences("Dr. Smith went to Washington. He arrived Monday.")) == 2

    def test_initials_do_not_split(self):
        assert len(split_sentences("J. R. Tolkien wrote this. It is good.")) == 2

    def test_closing_quote_stays_with_the_sentence(self):
        assert len(split_sentences('He said "go now." Then he left quietly.')) == 2

    def test_e_g_is_not_a_boundary(self):
        sents = split_sentences("Short texts, e.g. headlines, are hard to judge.")
        assert len(sents) == 1


class TestOffsets:
    def test_offsets_slice_back_to_the_sentence(self):
        text = "First sentence here.  Second one is here. Third."
        for s in split_sentences(text):
            assert text[s.start : s.end].strip() == s.text

    def test_offsets_are_monotonic_and_inside_the_source(self):
        text = "One two three. Four five six. Seven."
        sents = split_sentences(text)
        assert all(a.end <= b.start for a, b in zip(sents, sents[1:]))
        assert sents[-1].end <= len(text)

    def test_leading_whitespace_is_not_swallowed_by_offsets(self):
        text = "   Padded start. Second sentence here."
        first = split_sentences(text)[0]
        assert text[first.start : first.end] == "Padded start."


class TestFoldTuning:
    def test_fragments_kept_when_min_is_one(self):
        assert [s.text for s in split_sentences("Are you sure? Yes.", min_fragment_words=1)] == [
            "Are you sure?",
            "Yes.",
        ]

    def test_long_sentence_does_not_absorb_a_fragment(self):
        long = " ".join(f"w{i}" for i in range(40)) + "."
        sents = split_sentences(f"{long} Yes.")
        assert len(sents) == 2  # folding into a 40-word sentence would steal its score
        assert sents[1].text == "Yes."

    def test_custom_abbreviation_set(self):
        sents = split_sentences("See ch. 3 for details.", abbreviations=frozenset({"ch"}))
        assert len(sents) == 1


def test_words_property_never_returns_zero():
    for s in split_sentences("Hmm... ok. Fine."):
        assert s.words >= 1


def test_word_count_helper():
    from textscore.segment import word_count

    assert word_count("a  b\tc\n") == 3
    assert word_count("") == 0


@pytest.mark.parametrize("text", ["x", "One. Two!", "a.b.c", "...", "! ? .", '"Done." he said.'])
def test_never_crashes_on_pathological_input(text):
    split_sentences(text)
