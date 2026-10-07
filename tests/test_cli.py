"""CLI behaviour, driven through file inputs (stdin is covered by the pipe checks
in CI and is awkward to re-read inside one process)."""

import json

import pytest

from textscore import cli

TEXT = "First sentence is here. Second one follows. Third is short."


@pytest.fixture
def text_file(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text(TEXT, encoding="utf-8")
    return f


@pytest.fixture
def score_file(tmp_path):
    def make(probs):
        f = tmp_path / "in.json"
        f.write_text(json.dumps({"text": TEXT, "probs": probs}), encoding="utf-8")
        return f

    return make


class TestSplit:
    def test_text_mode_prints_spans_and_totals(self, text_file, capsys):
        assert cli.main(["split", str(text_file)]) == 0
        out = capsys.readouterr().out
        assert "First sentence is here." in out
        assert "[     0:" in out
        assert "3 sentences, 10 words" in out

    def test_json_mode_is_machine_readable(self, text_file, capsys):
        assert cli.main(["split", "--json", str(text_file)]) == 0
        rows = json.loads(capsys.readouterr().out)
        assert len(rows) == 3
        assert rows[0] == {"text": "First sentence is here.", "start": 0, "end": 23, "words": 4}
        assert sum(r["words"] for r in rows) == 10

    def test_min_fragment_words_keeps_short_sentences(self, tmp_path, capsys):
        f = tmp_path / "q.txt"
        f.write_text("Are you sure? Yes.", encoding="utf-8")
        assert cli.main(["split", "--min-fragment-words", "1", str(f)]) == 0
        assert "Are you sure?" in capsys.readouterr().out


class TestScore:
    def test_aggregate_json_out(self, score_file, capsys):
        f = score_file([0.9, 0.1, 0.1])
        assert cli.main(["score", str(f)]) == 0
        out = json.loads(capsys.readouterr().out)
        assert out["confidence"] == "low"  # three spans is thin evidence
        assert 25 < out["pos_percent"] < 45
        assert out["words"] == 10

    def test_env_thresholds_are_picked_up(self, score_file, capsys, monkeypatch):
        monkeypatch.setenv("TEXTSCORE_BAND_HIGH", "0.8")
        f = score_file([0.9, 0.1, 0.1])
        assert cli.main(["score", "--from-env", str(f)]) == 0
        assert json.loads(capsys.readouterr().out)["flagged_ratio"] == pytest.approx(0.333, abs=0.001)

    def test_count_mismatch_is_a_usable_error(self, score_file, capsys):
        f = score_file([0.5])
        assert cli.main(["score", str(f)]) == 2
        err = capsys.readouterr().err
        assert "3 sentences" in err and "1 probabilities" in err

    def test_html_output_is_escaped(self, tmp_path, capsys):
        payload = {"text": '<b>bold</b> claim. Second one follows.', "probs": [0.9, 0.1]}
        f = tmp_path / "xss.json"
        f.write_text(json.dumps(payload), encoding="utf-8")
        out = tmp_path / "out.html"
        assert cli.main(["score", "--html", str(out), str(f)]) == 0
        body = out.read_text(encoding="utf-8")
        assert "<b>bold</b>" not in body
        assert "&lt;b&gt;bold&lt;/b&gt;" in body
        assert "ts-highlight" in body

    def test_render_mode_emits_ansi(self, score_file, capsys):
        f = score_file([0.9, 0.1, 0.1])
        assert cli.main(["score", "--render", str(f)]) == 0
        assert "\033[" in capsys.readouterr().out


class TestEntryPoints:
    def test_demo_runs(self, capsys):
        assert cli.main(["demo"]) == 0
        assert "Dr. Chalk reviewed the draft." in capsys.readouterr().out

    @pytest.mark.parametrize("cmd", [["split", "--help"], ["score", "--help"], ["--help"]])
    def test_help_exits_cleanly(self, cmd):
        with pytest.raises(SystemExit) as exc:
            cli.main(cmd)
        assert exc.value.code == 0

    def test_no_subcommand_is_an_error(self):
        with pytest.raises(SystemExit) as exc:
            cli.main([])
        assert exc.value.code != 0
