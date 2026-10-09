# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""
The statuses this suite's report tables use, and how each is coloured.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["STATUS_CLASSES", "CoverageStatus", "MetricStatus"]


class MetricStatus(StrEnum):
    """
    What happened to one expected metric across a workload.

    The three cases need different fixes, which is the whole reason the suite distinguishes them:

    * :attr:`ROSE` -- the total increased; the profiler and the pipeline both work.
    * :attr:`FLAT` -- the series exists and is being scraped, but this workload drove no NCCL
      ops through it. The exporter is alive; either the instrumentation for this metric is not
      recording, or the workload genuinely does not exercise it.
    * :attr:`NO_SERIES` -- Prometheus has no series under this name at all, so nothing was ever
      exported. Check the profiler plugin, the OTLP endpoint and the collector.
    """

    ROSE = "rose"
    FLAT = "flat"
    NO_SERIES = "no series"

    @property
    def is_failure(self) -> bool:
        return self is not MetricStatus.ROSE

    @property
    def remedy(self) -> str:
        """One line on what to look at, shown beside a metric that did not increase."""
        match self:
            case MetricStatus.ROSE:
                return ""
            case MetricStatus.FLAT:
                return "scraped but did not move -- workload drove no ops through it"
            case MetricStatus.NO_SERIES:
                return "never exported -- check the profiler plugin, OTLP endpoint and collector"

    @classmethod
    def for_metric(cls, rose: bool, current: float | None) -> MetricStatus:
        """Classify one metric from whether it rose and whether it has a series at all."""
        if rose:
            return cls.ROSE
        return cls.NO_SERIES if current is None else cls.FLAT


class CoverageStatus(StrEnum):
    """Whether a participant count met the profile's declared coverage."""

    OK = "ok"
    SHORT = "short"

    @property
    def is_failure(self) -> bool:
        return self is CoverageStatus.SHORT

    @classmethod
    def for_counts(cls, seen: int, expected: int) -> CoverageStatus:
        return cls.OK if seen >= expected else cls.SHORT


#: Passed to ``reporter.table(..., status_classes=...)`` for any table with a status column.
STATUS_CLASSES = {
    MetricStatus.ROSE.value: "ok",
    MetricStatus.FLAT.value: "warn",
    MetricStatus.NO_SERIES.value: "bad",
    CoverageStatus.OK.value: "ok",
    CoverageStatus.SHORT.value: "bad",
}
