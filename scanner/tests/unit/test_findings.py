"""Bounded findings (reviewer I4, I1, I3): identical ones collapse, each rule
is capped on its own so one rule cannot evict another, and the bound holds
while findings come in, whatever the way they are added."""

from nitrogen.findings import MAX_FINDINGS, BoundedFindings, Finding, bounded


def test_identical_findings_collapse() -> None:
    sink = BoundedFindings()
    for _ in range(10_000):
        sink.append(Finding("pickle.import", "os.system"))
    assert sink.result() == [Finding("pickle.import", "os.system")]


def test_a_flood_of_one_rule_cannot_evict_another() -> None:
    # 300 distinct malformed members, then the one that matters.
    flood = [Finding("pickle.malformed", f"m{i}.pkl: broken") for i in range(300)]
    result = bounded([*flood, Finding("pickle.import", "data.pkl: os.system")])
    assert Finding("pickle.import", "data.pkl: os.system") in result
    malformed = [f for f in result if f.rule == "pickle.malformed"]
    assert len(malformed) == MAX_FINDINGS
    assert (
        Finding("findings.truncated", "200 more pickle.malformed finding(s) not listed", "info")
        in result
    )


def test_upstream_counts_are_carried_over() -> None:
    inner = BoundedFindings()
    for i in range(MAX_FINDINGS + 5):
        inner.append(Finding("pickle.import", f"m.f{i}"))
    outer = bounded([*inner.result(), *inner.result()])
    [summary] = [f for f in outer if f.rule == "findings.truncated"]
    assert summary.message == "10 more pickle.import finding(s) not listed"


def test_messages_are_truncated() -> None:
    sink = BoundedFindings()
    sink.append(Finding("pickle.import", "x" * 10_000))
    [finding] = sink.result()
    assert len(finding.message) < 400


def test_not_a_list_so_extend_cannot_bypass_the_bound() -> None:
    # Reviewer I3: a list subclass let += and extend skip the cap silently.
    sink = BoundedFindings()
    assert not isinstance(sink, list)
    assert not hasattr(sink, "extend")
    sink.absorb(Finding("pickle.import", f"m.f{i}") for i in range(MAX_FINDINGS * 3))
    assert len([f for f in sink.result() if f.rule == "pickle.import"]) == MAX_FINDINGS


def test_finding_defaults_to_risk() -> None:
    assert Finding("pickle.import", "os.system").severity == "risk"


def test_finding_can_be_warning_or_info() -> None:
    assert Finding("archive.encrypted", "x", "warning").severity == "warning"
    assert Finding("findings.truncated", "x", "info").severity == "info"


def test_truncated_summary_is_info() -> None:
    sink = BoundedFindings()
    for i in range(MAX_FINDINGS + 3):
        sink.append(Finding("pickle.import", f"m.f{i}"))
    summaries = [f for f in sink.result() if f.rule == "findings.truncated"]
    assert summaries and all(f.severity == "info" for f in summaries)
