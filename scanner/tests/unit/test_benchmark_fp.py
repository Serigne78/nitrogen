"""Pure logic of the false-positive benchmark (roadmap step 9).

The benchmark itself downloads real Hugging Face models and needs the network,
so it is opt-in and never part of this suite. What *is* tested here, offline, is
its backbone: parsing the frozen pinned model list, aggregating the scan
results, and rendering the report. A frozen list that is not actually pinned,
or an off-by-one in the false-positive rate, would quietly make the published
number meaningless — so those are the parts worth a test.

The script lives at the repository root (scripts/), outside the package; we load
it by path. Importing it must not touch the network: huggingface_hub is imported
lazily, inside the download helpers only.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "benchmark_false_positives.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("benchmark_fp", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves its own module via sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load()
SHA = "a" * 40  # a full 40-hex commit sha: what "pinned revision" means


# --- parsing the frozen list ------------------------------------------------


def test_parse_reads_repo_and_pinned_revision() -> None:
    [pin] = bench.parse_model_list(f"org/model@{SHA}\n")
    assert (pin.repo, pin.revision) == ("org/model", SHA)


def test_parse_ignores_blank_lines_and_comments() -> None:
    text = f"# a header comment\n\n  \norg/model@{SHA}  # size=12MB, safetensors\n"
    [pin] = bench.parse_model_list(text)
    assert (pin.repo, pin.revision) == ("org/model", SHA)


def test_parse_accepts_many_lines_in_order() -> None:
    text = f"a/one@{SHA}\nb/two@{'b' * 40}\n"
    pins = bench.parse_model_list(text)
    assert [p.repo for p in pins] == ["a/one", "b/two"]


@pytest.mark.parametrize(
    "line",
    [
        "org/model",  # no revision: not reproducible
        "org/model@main",  # a branch is not a pinned revision
        "org/model@v1.0",  # a tag can move
        f"org/model@{'a' * 7}",  # a short sha is ambiguous
        f"org/model@{'a' * 39}",  # 39 hex: not a full sha
        f"org/model@{'g' * 40}",  # not hexadecimal
        f"model@{SHA}",  # no org
        f"a/b/c@{SHA}",  # too many path segments
        f"org/model@{SHA}@extra",  # two revisions
        f" @{SHA}",  # empty repo
    ],
    ids=lambda s: s[:24],
)
def test_parse_rejects_anything_not_pinned(line: str) -> None:
    with pytest.raises(bench.ModelListError):
        bench.parse_model_list(line + "\n")


def test_parse_rejects_duplicate_repo() -> None:
    # A repo pinned twice (e.g. a bad merge) would double-count in the rate.
    with pytest.raises(bench.ModelListError):
        bench.parse_model_list(f"org/model@{SHA}\norg/model@{'b' * 40}\n")


# --- aggregating the results ------------------------------------------------


def _result(repo: str, status: str, rules: list[str] | None = None) -> Any:
    return bench.Result(
        repo=repo, revision=SHA, status=status, exit_code=0, rules=rules or [], size_bytes=1
    )


def test_summary_counts_each_status_and_false_positive_rate() -> None:
    results = [
        _result("a/compliant", "compliant"),
        _result("b/warn", "warning", ["archive.encrypted"]),
        _result("c/risk", "risk", ["pickle.import"]),  # a trusted model flagged: a FP
        _result("d/error", "error"),
        _result("e/skipped", "skipped"),
    ]
    s = bench.summarize(results)
    assert (s.compliant, s.warnings, s.false_positives, s.errors, s.skipped) == (1, 1, 1, 1, 1)
    # Every trusted model is benign, so a RISK verdict is a false positive.
    assert s.false_positives == 1
    # Rate is over models actually evaluated (not the skipped ones): 1 of 4.
    assert s.evaluated == 4
    assert s.false_positive_rate == pytest.approx(0.25)


def test_summary_rate_is_zero_when_nothing_evaluated() -> None:
    s = bench.summarize([_result("a/skip", "skipped")])
    assert s.evaluated == 0
    assert s.false_positive_rate == 0.0


def test_summary_clean_run_is_zero_false_positives() -> None:
    s = bench.summarize([_result(f"m/{i}", "compliant") for i in range(50)])
    assert (s.evaluated, s.false_positives, s.false_positive_rate) == (50, 0, 0.0)


# --- rendering the report ---------------------------------------------------


def test_report_states_the_headline_rate_and_every_model() -> None:
    results = [
        _result("org/clean", "compliant"),
        _result("org/flagged", "risk", ["pickle.import"]),
    ]
    report = bench.render_report(results, bench.summarize(results))
    assert "org/clean" in report and "org/flagged" in report
    assert SHA[:12] in report  # the pinned revision is shown, abbreviated
    assert "pickle.import" in report  # the finding that caused the FP is named
    assert "50" in report  # 1 of 2 evaluated -> 50% false positives


def test_report_marks_false_positives_so_they_are_not_lost() -> None:
    results = [_result("org/flagged", "risk", ["pickle.import"])]
    report = bench.render_report(results, bench.summarize(results))
    # A false positive must be visibly flagged, not just another table row.
    assert "false positive" in report.lower()


def test_report_is_deterministic() -> None:
    results = [_result("b/two", "compliant"), _result("a/one", "risk", ["r"])]
    summary = bench.summarize(results)
    assert bench.render_report(results, summary) == bench.render_report(results, summary)
