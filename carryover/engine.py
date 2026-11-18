"""Pure carryover decision logic.

The caller supplies injections in run order: the order in which the batch CSV
was first uploaded (later new injection numbers are appended).  A sample whose
peak area is at or above the batch high threshold starts a carryover episode.
Following samples are flagged until a blank at or below the blank limit is
seen.  A blank above the limit, or one with a missing peak area, cannot clear
the episode.  A later high sample becomes the new trigger.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Iterable, Optional


SAMPLE = "sample"
BLANK = "blank"

HIGH = "high"
CARRYOVER_RISK = "carryover_risk"
CLEAR = "clear"
BLANK_CLEAR = "blank_clear"
BLANK_EXCEEDED = "blank_exceeded"
BLANK_MISSING = "blank_missing"

STATUS_LABELS = {
    HIGH: "高值样品",
    CARRYOVER_RISK: "残留风险",
    CLEAR: "未标记",
    BLANK_CLEAR: "空白未超限",
    BLANK_EXCEEDED: "空白超限",
    BLANK_MISSING: "空白缺失",
}

KIND_LABELS = {SAMPLE: "样品", BLANK: "冲洗空白"}


@dataclass(frozen=True)
class Injection:
    injection_no: str
    kind: str
    peak_area: Optional[float]
    revision: int
    position: int = 0


@dataclass(frozen=True)
class ResultRow:
    injection_no: str
    kind: str
    peak_area: Optional[float]
    revision: int
    status: str
    trigger_injection: Optional[str]
    note: str


def natural_key(value: str) -> tuple:
    """Sort injection numbers naturally ("2" before "10"); informational only."""
    parts = re.split(r"(\d+)", value.casefold())
    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in parts)


def sort_run(rows: Iterable[Injection]) -> list[Injection]:
    """Stable sort by stored run position.

    Equal positions keep the incoming order, so callers that already hold the
    run sequence (and leave position at its default) are unaffected.
    """
    return sorted(rows, key=lambda injection: injection.position)


def compute_results(
    injections: Iterable[Injection],
    high_threshold: float,
    blank_limit: float,
) -> list[ResultRow]:
    ordered = sort_run(injections)
    results: list[ResultRow] = []
    trigger: Optional[str] = None

    for injection in ordered:
        status: str
        note = ""
        trigger_for_row = None

        if injection.kind == BLANK:
            if injection.peak_area is None:
                status = BLANK_MISSING
                note = "冲洗空白峰面积缺失，不能解除残留风险"
                if trigger is not None:
                    note += f"（仍由进样 {trigger} 触发）"
                    trigger_for_row = trigger
            elif injection.peak_area <= blank_limit:
                status = BLANK_CLEAR
                if trigger is not None:
                    note = f"冲洗空白未超限，解除进样 {trigger} 触发的残留风险"
                else:
                    note = "冲洗空白未超限"
                trigger = None
            else:
                status = BLANK_EXCEEDED
                note = "冲洗空白超出空白上限，不能解除残留风险"
                if trigger is not None:
                    note += f"（仍由进样 {trigger} 触发）"
                    trigger_for_row = trigger
        else:
            # A sample without an area is rejected during CSV validation, but
            # keep this defensive branch so malformed database rows cannot
            # silently clear a risk.
            if injection.peak_area is None:
                status = CARRYOVER_RISK if trigger is not None else CLEAR
                note = "样品峰面积缺失"
                trigger_for_row = trigger
            elif injection.peak_area >= high_threshold:
                status = HIGH
                trigger = injection.injection_no
                note = "达到高值阈值，作为后续样品的残留触发进样"
            elif trigger is not None:
                status = CARRYOVER_RISK
                trigger_for_row = trigger
                note = f"高值进样 {trigger} 之后尚未出现有效冲洗空白"
            else:
                status = CLEAR

        results.append(
            ResultRow(
                injection_no=injection.injection_no,
                kind=injection.kind,
                peak_area=injection.peak_area,
                revision=injection.revision,
                status=status,
                trigger_injection=trigger_for_row,
                note=note,
            )
        )

    return results


# --- precision (repeat-injection RSD) ---------------------------------------

PRECISION_PASS = "pass"
PRECISION_FAIL = "fail"
PRECISION_UNVERIFIED = "unverified"

PRECISION_STATUS_LABELS = {
    PRECISION_PASS: "合格",
    PRECISION_FAIL: "超限",
    PRECISION_UNVERIFIED: "无法核查",
}

# Reasons a group cannot be verified.  They are deliberately specific so the
# page can tell the operator what to fix.
UNVERIFIED_TOO_FEW = "too_few"
UNVERIFIED_MISSING = "missing"
UNVERIFIED_CARRYOVER = "carryover"
UNVERIFIED_ZERO_MEAN = "zero_mean"

UNVERIFIED_REASONS = {
    UNVERIFIED_TOO_FEW: "现行样品进样不足两支，无法核查精密度",
    UNVERIFIED_MISSING: "分组内进样缺失或峰面积缺失，无法核查精密度",
    UNVERIFIED_CARRYOVER: "分组内出现残留风险，无法核查精密度",
    UNVERIFIED_ZERO_MEAN: "峰面积均值为零，无法核查精密度",
}


@dataclass(frozen=True)
class PrecisionGroup:
    group: str
    injection_nos: tuple[str, ...]


@dataclass(frozen=True)
class PrecisionResult:
    group: str
    status: str
    count: int
    mean: Optional[float]
    stdev: Optional[float]
    rsd: Optional[float]
    reason: str
    note: str


def compute_precision(
    groups: Iterable[PrecisionGroup],
    injections: Iterable[Injection],
    carryover: Iterable[ResultRow],
    rsd_limit: float,
) -> list[PrecisionResult]:
    """Compute RSD per group using only current sample injections.

    Every injection may belong to at most one group (enforced by the caller
    when the grouping CSV is accepted).  A group is verifiable only when it
    holds at least two current sample injections with peak areas, none of those
    injections carries a carryover-risk verdict, and the mean is non-zero.
    """
    current = {row.injection_no: row for row in injections}
    risk = {
        row.injection_no
        for row in carryover
        if row.status == CARRYOVER_RISK
    }
    results: list[PrecisionResult] = []

    for group in groups:
        numbers = list(group.injection_nos)
        missing = [no for no in numbers if no not in current]
        if missing:
            results.append(
                PrecisionResult(
                    group.group, PRECISION_UNVERIFIED, 0, None, None,
                    None, UNVERIFIED_MISSING,
                    UNVERIFIED_REASONS[UNVERIFIED_MISSING]
                    + f"（缺少进样 {', '.join(missing)}）",
                )
            )
            continue

        # Only current sample injections contribute; blanks listed in the
        # grouping are not observations of the sample's repeatability.
        rows = [
            current[no] for no in numbers
            if current[no].kind == SAMPLE
        ]
        excluded = [no for no in numbers if current[no].kind != SAMPLE]
        absent_area = [row.injection_no for row in rows if row.peak_area is None]
        if absent_area:
            results.append(
                PrecisionResult(
                    group.group, PRECISION_UNVERIFIED, len(rows), None, None,
                    None, UNVERIFIED_MISSING,
                    UNVERIFIED_REASONS[UNVERIFIED_MISSING]
                    + "（峰面积缺失：" + ", ".join(absent_area) + "）",
                )
            )
            continue

        if len(rows) < 2:
            note = UNVERIFIED_REASONS[UNVERIFIED_TOO_FEW]
            if excluded:
                note += f"（已忽略非样品进样 {', '.join(excluded)}）"
            results.append(
                PrecisionResult(
                    group.group, PRECISION_UNVERIFIED, len(rows), None, None,
                    None, UNVERIFIED_TOO_FEW, note,
                )
            )
            continue

        risky = [row.injection_no for row in rows if row.injection_no in risk]
        if risky:
            results.append(
                PrecisionResult(
                    group.group, PRECISION_UNVERIFIED, len(rows), None, None,
                    None, UNVERIFIED_CARRYOVER,
                    UNVERIFIED_REASONS[UNVERIFIED_CARRYOVER]
                    + f"（进样 {', '.join(risky)}）",
                )
            )
            continue

        areas = [float(row.peak_area) for row in rows]
        mean = statistics.fmean(areas)
        if mean == 0:
            results.append(
                PrecisionResult(
                    group.group, PRECISION_UNVERIFIED, len(rows), 0.0, None,
                    None, UNVERIFIED_ZERO_MEAN,
                    UNVERIFIED_REASONS[UNVERIFIED_ZERO_MEAN],
                )
            )
            continue

        stdev = statistics.stdev(areas)  # sample standard deviation (n - 1)
        rsd = stdev / abs(mean) * 100.0
        status = PRECISION_PASS if rsd <= rsd_limit else PRECISION_FAIL
        results.append(
            PrecisionResult(
                group.group, status, len(rows), mean, stdev, rsd, "",
                f"RSD {rsd:.2f}% {'≤' if status == PRECISION_PASS else '>'} "
                f"上限 {rsd_limit:g}%",
            )
        )

    return results
