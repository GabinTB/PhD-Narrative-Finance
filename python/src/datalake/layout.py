"""Partition layout of an artifact: which file holds which days.

A partitioned artifact stores one file per period of its ``partition_freq``
(``datalake.periods``), named by the period key (``2008-01.parquet`` for M,
``2008Q1.parquet`` for Q, ...). The layout is part of the artifact's identity
and lives in its hyperparams:

    partition_freq   'D' | 'W' | 'M' | 'Q' | 'Y'       (absent: 'M', the legacy layout)
    start, end       ISO dates bounding the content   (legacy: start_year / end_year
                                                       -> Jan 1 / Dec 31)

Readers never build file names: ``Layout.file_for(root, day)``,
``expected()`` and ``existing(root)`` are the only way a module finds a
partition, so a daily or quarterly lake reads exactly like a monthly one.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datalake.periods import (
    DEFAULT_FREQ,
    Period,
    PeriodError,
    check_freq,
    parse_key,
    partition_file,
    period_of,
    periods,
)

if TYPE_CHECKING:
    from datalake.artifact import Artifact

SUFFIX = ".parquet"


@dataclass(frozen=True)
class Layout:
    """``freq`` partitions over [start, end] (either bound may be unknown: None)."""

    freq: str = DEFAULT_FREQ
    start: date | None = None
    end: date | None = None

    def __post_init__(self) -> None:
        check_freq(self.freq)
        if self.start and self.end and self.end < self.start:
            raise PeriodError(f"layout end {self.end} before start {self.start}")

    # -- periods ------------------------------------------------------------

    def period_for(self, day: date) -> Period:
        return period_of(day, self.freq)

    def expected(self) -> list[Period]:
        """Every partition the declared range covers (needs both bounds)."""
        if self.start is None or self.end is None:
            raise PeriodError("the layout has no declared start/end")
        return periods(self.start, self.end, self.freq)

    def file_for(self, root: Path, day: date) -> Path:
        return Path(root) / partition_file(self.period_for(day).key, SUFFIX)

    def path_of(self, root: Path, period: Period) -> Path:
        if period.freq != self.freq:
            raise PeriodError(f"period {period.key} is {period.freq}, layout is {self.freq}")
        return Path(root) / partition_file(period.key, SUFFIX)

    def existing(self, root: Path) -> dict[str, Path]:
        """Partition files present under ``root`` (key -> path), in period order.

        Files whose stem is not a period key (sidecars, run metadata) are
        ignored; a period file of another frequency is an error -- the layout
        of a directory never mixes frequencies.
        """
        out: list[tuple[Period, Path]] = []
        for path in Path(root).glob(f"*{SUFFIX}"):
            try:
                period = parse_key(path.stem)
            except PeriodError:
                continue
            if period.freq != self.freq:
                raise PeriodError(f"{path.name} is a {period.freq} partition in a "
                                  f"{self.freq} layout ({root})")
            out.append((period, path))
        return {p.key: path for p, path in sorted(out)}

    def between(self, root: Path, start: date, end: date) -> list[Path]:
        """Existing partition files that intersect [start, end], in order."""
        return [path for key, path in self.existing(root).items()
                if parse_key(key).last >= start and parse_key(key).first <= end]

    # -- identity -----------------------------------------------------------

    def hyperparams(self) -> dict[str, Any]:
        hp: dict[str, Any] = {"partition_freq": self.freq}
        if self.start is not None:
            hp["start"] = self.start.isoformat()
        if self.end is not None:
            hp["end"] = self.end.isoformat()
        return hp


def layout_from_hyperparams(hp: dict[str, Any]) -> Layout:
    """The layout an artifact declares, with the legacy monthly mapping."""
    freq = hp.get("partition_freq") or DEFAULT_FREQ
    start = end = None
    if hp.get("start"):
        start = date.fromisoformat(str(hp["start"]))
    elif hp.get("start_year") is not None:
        start = date(int(hp["start_year"]), 1, 1)
    if hp.get("end"):
        end = date.fromisoformat(str(hp["end"]))
    elif hp.get("end_year") is not None:
        end = date(int(hp["end_year"]), 12, 31)
    return Layout(freq, start, end)


def layout_of(artifact: Artifact) -> Layout:
    return layout_from_hyperparams(artifact.meta.hyperparams)


def add_layout_args(parser: Any, *, default_freq: str | None = DEFAULT_FREQ,
                    required: bool = True) -> None:
    """``--partition-freq / --start / --end`` (or legacy ``--start-year / --end-year``)."""
    parser.add_argument("--partition-freq", default=default_freq, type=check_freq,
                        help="storage partition = work unit: D, W, M, Q or Y")
    parser.add_argument("--start", type=date.fromisoformat, help="first day (ISO)")
    parser.add_argument("--end", type=date.fromisoformat, help="last day (ISO)")
    parser.add_argument("--start-year", type=int, help="legacy: --start YEAR-01-01")
    parser.add_argument("--end-year", type=int, help="legacy: --end YEAR-12-31")
    parser.set_defaults(_layout_required=required)


def layout_from_args(args: Any, *, default: Layout | None = None) -> Layout:
    """The layout of ``add_layout_args`` arguments; missing pieces come from ``default``."""
    start = args.start or (date(args.start_year, 1, 1) if args.start_year else None)
    end = args.end or (date(args.end_year, 12, 31) if args.end_year else None)
    freq = args.partition_freq
    if default is not None:
        start, end = start or default.start, end or default.end
        freq = freq or default.freq
    if getattr(args, "_layout_required", True) and (start is None or end is None):
        raise PeriodError("give --start/--end (or --start-year/--end-year)")
    return Layout(freq or DEFAULT_FREQ, start, end)
