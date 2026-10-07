"""Argument handling of the bench harness, tested without torch.

The scoring loop needs a model and a GPU-less minute of CPU; the parsing around
it does not, and parsing is where a harness silently reports the wrong number
(a flipped class index, weights that do not sum, a limit that drops a class).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

from score import expand, parse_ensemble, parse_model_arg, resolve_ai_position  # noqa: E402


class TestParseModelArg:
    def test_name_repo_and_index(self):
        assert parse_model_arg("e5=some/e5-detector:1") == ("e5", "some/e5-detector", 1)

    def test_index_may_be_omitted(self):
        assert parse_model_arg("distil=some/distil") == ("distil", "some/distil", None)

    def test_local_path_with_colon_index(self):
        assert parse_model_arg("m=/models/checkpoint:0") == ("m", "/models/checkpoint", 0)

    def test_negative_index_is_accepted(self):
        assert parse_model_arg("m=repo:-1")[2] == -1

    @pytest.mark.parametrize("bad", ["e5", "=repo:1", "name="])
    def test_malformed_raises(self, bad):
        with pytest.raises(SystemExit):
            parse_model_arg(bad)


class TestResolveAiPosition:
    def test_explicit_index_wins_over_labels(self):
        assert resolve_ai_position({0: "human", 1: "ai"}, 0) == 0

    def test_label_text_is_used_when_index_missing(self):
        assert resolve_ai_position({0: "not_generated", 1: "generated"}, None) == 1

    def test_ambiguous_generic_labels_refuse_instead_of_guessing(self):
        # LABEL_0/LABEL_1 carries no direction; guessing here produced a real
        # wrong-way-round in the field, so this must be a hard stop
        with pytest.raises(SystemExit, match="cannot resolve"):
            resolve_ai_position({0: "LABEL_0", 1: "LABEL_1"}, None)

    def test_two_matching_labels_refuse(self):
        with pytest.raises(SystemExit):
            resolve_ai_position({0: "ai_generated", 1: "machine_made", 2: "human"}, None)

    def test_negated_label_is_not_matched(self):
        assert resolve_ai_position({0: "non-ai", 1: "ai"}, None) == 1


class TestExpandAndEnsemble:
    def test_expand_repeats_per_paragraph_values(self):
        assert expand([1, 0], [["a", "b"], ["c"]]) == [1, 1, 0]

    def test_weights_must_name_scored_models(self):
        with pytest.raises(SystemExit, match="not scored"):
            parse_ensemble("a=0.5,z=0.5", {"a": [], "b": []})

    def test_weights_must_sum_to_one(self):
        with pytest.raises(SystemExit, match="sum to 1"):
            parse_ensemble("a=0.6,b=0.6", {"a": [], "b": []})

    def test_valid_ensemble(self):
        assert parse_ensemble("a=0.6, b=0.4", {"a": [], "b": []}) == {"a": 0.6, "b": 0.4}
