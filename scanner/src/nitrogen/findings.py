"""What an analyzer reports about a file. Any finding makes the verdict RISK.

Findings are bounded, while they come in: identical ones collapse, messages
are truncated, and each rule keeps at most MAX_FINDINGS of its own findings,
plus one `findings.truncated` summary per rule. Otherwise a crafted pickle
repeating an import millions of times would exhaust memory (a risk turned into
an execution error, with no AI-BOM). The cap is per rule so that a flood of
one rule (100 broken members) can never evict another (the os.system import).
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

MAX_FINDINGS = 100  # per rule, per file
MAX_MESSAGE = 300  # characters
TRUNCATED = "findings.truncated"

# Severity tiers (ADR 0008), from most to least serious.
RISK = "risk"  # code execution possible, or deceptive structure — blocks
WARNING = "warning"  # blind spot / suspicious, no proven execution path
INFO = "info"  # metadata, never blocks
SEVERITIES = (RISK, WARNING, INFO)


@dataclass(frozen=True)
class Finding:
    rule: str  # stable identifier, e.g. "safetensors.layout"
    message: str  # human-readable detail
    severity: str = RISK  # default: an unqualified finding blocks (fail-closed)


def _short(finding: Finding) -> Finding:
    if len(finding.message) <= MAX_MESSAGE:
        return finding
    return Finding(finding.rule, finding.message[:MAX_MESSAGE] + "...", finding.severity)


class BoundedFindings:
    """Deduplicates, truncates and caps per rule as findings come in. Not a
    list subclass on purpose: += or extend would silently bypass the bound."""

    def __init__(self) -> None:
        self._kept: dict[str, list[Finding]] = {}
        self._seen: set[Finding] = set()
        self._dropped: dict[str, int] = {}

    def append(self, finding: Finding) -> None:
        if finding.rule == TRUNCATED:
            # A summary from an analyzer that already bounded its output.
            count, _more, rule = finding.message.split(" ", 3)[:3]
            self._dropped[rule] = self._dropped.get(rule, 0) + int(count)
            return
        finding = _short(finding)
        if finding in self._seen:
            return
        kept = self._kept.setdefault(finding.rule, [])
        if len(kept) >= MAX_FINDINGS:
            self._dropped[finding.rule] = self._dropped.get(finding.rule, 0) + 1
            return
        self._seen.add(finding)
        kept.append(finding)

    def absorb(self, findings: Iterable[Finding]) -> None:
        for finding in findings:
            self.append(finding)

    def __iter__(self) -> Iterator[Finding]:
        for kept in self._kept.values():
            yield from kept

    def __len__(self) -> int:
        return sum(len(kept) for kept in self._kept.values())

    def __bool__(self) -> bool:
        return bool(self._kept) or bool(self._dropped)

    def result(self) -> list[Finding]:
        summaries = [
            Finding(TRUNCATED, f"{count} more {rule} finding(s) not listed", INFO)
            for rule, count in self._dropped.items()
        ]
        return list(self) + summaries


def bounded(findings: Iterable[Finding]) -> tuple[Finding, ...]:
    sink = BoundedFindings()
    sink.absorb(findings)
    return tuple(sink.result())
