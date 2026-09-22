"""Hybrid CPU + remote-GPU scoring over a Ray cluster.

Topology
--------
* **Head** = the machine holding the datalake (this one). It streams months
  from local disk, scores its own share on the CPU with the fused kernel, and
  ships the rest as fp16 blocks to the GPU worker.
* **GPU worker** = the second machine. It never sees the datalake. It receives
  blocks, runs steps 1-2 on the GPU (``gpu_scoring.GpuScorer``) and steps 3-5
  on its own CPU with the *same* compiled kernel, and returns only the small
  per-narrative accumulators. Nothing is staged remotely.

Every headline crosses the link **once**: the worker holds all six (mode,
pooling) target matrices and evaluates them per block, so the grid costs one
stream, not six.

Reproducibility / secrecy
-------------------------
No address is hardcoded and none is ever printed. The worker joins the head at
``RAY_HEAD_ADDRESS``; when that is unset the head derives its own address on
the interface that routes to ``VERTEX_DIRECT_IP`` (both from ``.env``). Anyone
with two machines on a network sets those two variables and runs the same two
commands (see ``docs`` below).

Numerics
--------
Steps 3-5 are bit-identical on both machines (same kernel). Steps 1-2 differ
only by cuBLAS-vs-OpenBLAS reduction order in the matmul; the runner measures
that on a shared block and records it, and months are assigned to a machine
deterministically so every panel is internally consistent.

Ray is imported lazily so the rest of the package does not need it.
"""
from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import polars as pl

# The spec_pipeline module this track was built on has been removed.
raise NotImplementedError("Ray/GPU track suspended; canonical scorer is "
                          "narrative_scoring.pipeline.score_dates")
from narrative_scoring.corrections import Correction

log = logging.getLogger(__name__)

DOCS = """
Two-machine setup (all addresses from .env, never on the command line):

  # on the HEAD (holds the datalake), from python/:
  uv run python -m narrative_scoring._kernels.build
  uv run ray start --head --node-ip-address="$(python -m narrative_scoring.ray_hybrid --my-ip)" --port=6379

  # on the GPU WORKER (same repo checkout, same .env with RAY_HEAD_ADDRESS set):
  uv run python -m narrative_scoring._kernels.build
  uv run ray start --address="$RAY_HEAD_ADDRESS" --num-gpus=1

  # then run the notebooks on the head; HybridRunner picks up the worker.
"""


# ---------------------------------------------------------------------------
# Addresses -- resolved, never logged
# ---------------------------------------------------------------------------

def _parse_host(value: str) -> str:
    v = value.strip().strip("'\"")
    for prefix in ("http://", "https://"):
        if v.startswith(prefix):
            v = v[len(prefix):]
    return v.split("/", 1)[0].split(":", 1)[0]


def my_ip_towards(peer_host: str) -> str:
    """The local address on the interface that routes to ``peer_host``."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((peer_host, 9))          # no packet is sent for UDP connect
        return s.getsockname()[0]


def head_address() -> str:
    """RAY_HEAD_ADDRESS, else derived from the route to VERTEX_DIRECT_IP."""
    explicit = os.environ.get("RAY_HEAD_ADDRESS")
    if explicit:
        return explicit
    peer = os.environ.get("VERTEX_DIRECT_IP")
    if not peer:
        raise RuntimeError("set RAY_HEAD_ADDRESS or VERTEX_DIRECT_IP in .env")
    return f"{my_ip_towards(_parse_host(peer))}:6379"


# ---------------------------------------------------------------------------
# Worker-side
# ---------------------------------------------------------------------------

@dataclass
class PassSpec:
    """One (mode, pooling) pass with everything the worker needs to score it."""

    cfg: sp.ScoringConfig
    P_scoring: np.ndarray
    mu: np.ndarray | None
    mu_hat: np.ndarray | None
    tau: float
    n_eff: float
    variants: list[sp.GateVariant]


class _BlockScorer:
    """Steps 1-5 for one block, for every pass, on whatever device is local.

    Plain class so it can be tested without Ray; ``GpuWorker`` wraps it.
    """

    def __init__(self, passes: list[PassSpec], table_meta: dict[str, Any], device: str):
        from narrative_scoring._kernels import HAVE_FUSED, gate_aggregate_rowwise
        from narrative_scoring.gpu_scoring import GpuScorer

        if not HAVE_FUSED:
            raise RuntimeError("build the fused kernel on this machine first "
                               "(python -m narrative_scoring._kernels.build)")
        self._kernel = gate_aggregate_rowwise
        self.n_texts = table_meta["n_texts"]
        self.n_prim = table_meta["n_primitives"]
        self.n_narr = table_meta["n_narratives"]
        self.p2n = np.asarray(table_meta["primitive_to_narrative"], dtype=np.int32)
        self.passes = passes
        self.scorers = [
            GpuScorer(ps.P_scoring, self.n_texts, self.n_prim, ps.cfg.paraphrase_pooling,
                      ps.cfg.mode, ps.mu, ps.mu_hat, device=device)
            for ps in passes
        ]
        self.threads = os.cpu_count() or 1

    def score(self, X: np.ndarray) -> list[list[tuple]]:
        """For each pass, the kernel's per-variant (count,total,peak,unassigned,kept,floored)."""
        out = []
        for ps, scorer in zip(self.passes, self.scorers):
            S = scorer.score(X)
            tau = float(np.float32(ps.tau))
            row_qs = np.array([v.q for v in ps.variants if v.legacy_rel_floor is None
                               and v.axis is sp.PctAxis.ROW_WISE], dtype=np.float64)
            res = self._kernel(np.ascontiguousarray(S), tau, row_qs, self.p2n, self.n_narr,
                               ps.cfg.narrative_agg is sp.AggRule.MEDIAN, self.threads)
            out.append(res)
        return out

    def calibrate(self, X: np.ndarray, pass_idx: int, trim_frac: float) -> np.ndarray:
        """Trimmed null draws for one block (the head owns the reservoir)."""
        from narrative_scoring.null_model import trim_null_draws_batch

        S = self.scorers[pass_idx].score(X)
        return trim_null_draws_batch(S, trim_frac=trim_frac).ravel()


def make_gpu_worker_class():
    """The Ray actor, created lazily so ``ray`` is only needed when used."""
    import ray

    @ray.remote(num_gpus=1)
    class GpuWorker:
        def __init__(self, passes, table_meta, device=None):
            from narrative_scoring.gpu_scoring import pick_device
            self.inner = _BlockScorer(passes, table_meta, pick_device(device))
            self.device = self.inner.scorers[0].device if self.inner.scorers else "cpu"

        def device_info(self):
            try:
                import torch
                if torch.cuda.is_available():
                    return {"device": self.device, "name": torch.cuda.get_device_name(0),
                            "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9}
            except Exception:  # noqa: BLE001
                pass
            return {"device": self.device}

        def score(self, X, day_ord):
            return day_ord, X.shape[0], self.inner.score(X)

        def calibrate(self, X, pass_idx, trim_frac):
            return self.inner.calibrate(X, pass_idx, trim_frac)

        def probe(self, X):
            """Pooled scores for one block, to quantify GPU-vs-CPU matmul rounding."""
            return [sc.score(X) for sc in self.inner.scorers]

    return GpuWorker


# ---------------------------------------------------------------------------
# Head-side driver
# ---------------------------------------------------------------------------

@dataclass
class HybridPlan:
    """Deterministic month -> machine assignment, recorded with the results."""

    gpu_months: list[tuple[int, int]]
    cpu_months: list[tuple[int, int]]
    gpu_share: float

    def to_dict(self) -> dict[str, Any]:
        return {"gpu_share": self.gpu_share,
                "gpu_months": [f"{y}-{m:02d}" for y, m in self.gpu_months],
                "cpu_months": [f"{y}-{m:02d}" for y, m in self.cpu_months]}


def plan_months(start: date, end: date, gpu_share: float) -> HybridPlan:
    """Interleave so both machines get early AND late months (volume grows over time)."""
    months = sp._months(start, end)
    n_gpu = int(round(len(months) * gpu_share))
    # every k-th month to the CPU, spread evenly
    if n_gpu >= len(months):
        return HybridPlan(months, [], gpu_share)
    step = len(months) / max(len(months) - n_gpu, 1)
    cpu_idx = {int(i * step) for i in range(len(months) - n_gpu)}
    cpu = [m for i, m in enumerate(months) if i in cpu_idx]
    gpu = [m for i, m in enumerate(months) if i not in cpu_idx]
    return HybridPlan(gpu, cpu, gpu_share)


def _table_meta(table: sp.PrimitiveTable) -> dict[str, Any]:
    return {"n_texts": table.n_texts, "n_primitives": table.n_primitives,
            "n_narratives": table.narrative_frame.height,
            "primitive_to_narrative": table.primitive_to_narrative}


class HybridRunner:
    """Score a window across the head's CPU and the remote GPU worker.

    Returns the same ``dict[variant_key, SpecResult]`` per pass as
    ``score_grid``; panels for GPU-scored months are assembled by the very
    same ``_assemble_results``.
    """

    def __init__(self, headlines_dir: Path, embeddings_dir: Path, table: sp.PrimitiveTable,
                 passes: list[PassSpec], *, gpu_share: float = 0.9,
                 batch_size: int = 50_000, score_block: int = 8_192, threads: int = 8,
                 duckdb_memory_limit: str = "6GB", temp_directory: str | None = "/tmp/duckdb_spill",
                 max_in_flight: int = 8):
        import ray

        self.ray = ray
        self.headlines_dir, self.embeddings_dir = Path(headlines_dir), Path(embeddings_dir)
        self.table, self.passes = table, passes
        self.gpu_share = gpu_share
        self.batch_size, self.score_block, self.threads = batch_size, score_block, threads
        self.duckdb_memory_limit, self.temp_directory = duckdb_memory_limit, temp_directory
        self.max_in_flight = max_in_flight
        if not ray.is_initialized():
            ray.init(address=os.environ.get("RAY_ADDRESS") or head_address(),
                     log_to_driver=False)
        self.worker = make_gpu_worker_class().remote(passes, _table_meta(table))
        self.device_info = ray.get(self.worker.device_info.remote())
        log.info("GPU worker joined: %s", self.device_info)

    # -- rounding check ---------------------------------------------------
    def matmul_divergence(self, X: np.ndarray) -> list[dict[str, float]]:
        remote = self.ray.get(self.worker.probe.remote(X))
        out = []
        for ps, S_gpu in zip(self.passes, remote):
            H = sp.apply_mode(X, ps.cfg.mode, ps.mu, ps.mu_hat)
            S_cpu = sp.primitive_scores(H, ps.P_scoring, self.table, ps.cfg.paraphrase_pooling)
            d = np.abs(S_gpu - S_cpu)
            out.append({"mode": ps.cfg.mode.value, "pooling": ps.cfg.paraphrase_pooling.value,
                        "max_abs": float(d.max()), "mean_abs": float(d.mean())})
        return out

    # -- scoring ----------------------------------------------------------
    def run(self, start: date, end: date, mu_norm: float = float("nan")) -> tuple[list[dict[str, sp.SpecResult]], HybridPlan]:
        plan = plan_months(start, end, self.gpu_share)
        n_narr, n_prim = self.table.narrative_frame.height, self.table.n_primitives
        states = [
            {v.key: sp._VariantState(n_narr, n_prim, False, self.table.primitive_to_narrative,
                                     ps.cfg.narrative_agg) for v in ps.variants}
            for ps in self.passes
        ]

        # CPU share: the ordinary fused path, month by month
        for (y, m) in plan.cpu_months:
            lo, hi = date(y, m, 1), _month_end(y, m)
            lo, hi = max(lo, start), min(hi, end)
            for pi, ps in enumerate(self.passes):
                res = sp.score_grid(self.headlines_dir, self.embeddings_dir, ps.P_scoring, self.table,
                                    ps.cfg, ps.mu, ps.mu_hat, ps.tau, ps.n_eff, ps.variants,
                                    start=lo, end=hi, mu_norm=mu_norm, keep_primitive_daily=False,
                                    batch_size=self.batch_size, score_block=self.score_block,
                                    threads=self.threads, duckdb_memory_limit=self.duckdb_memory_limit,
                                    temp_directory=self.temp_directory)
                for key, r in res.items():
                    _merge_result_into_state(states[pi][key], r)

        # GPU share: stream fp16 blocks, one crossing per headline, all passes per block
        in_flight: list = []
        def drain(block_until: int) -> None:
            nonlocal in_flight
            while len(in_flight) > block_until:
                done, in_flight = self.ray.wait(in_flight, num_returns=1)
                for day_ord, n_head, per_pass in self.ray.get(done):
                    day = sp._EPOCH.__class__.fromordinal(sp._EPOCH.toordinal() + int(day_ord))
                    for pi, ps in enumerate(self.passes):
                        row_variants = [v for v in ps.variants if v.legacy_rel_floor is None
                                        and v.axis is sp.PctAxis.ROW_WISE]
                        for v, (kc, kt, kp, ku, kk, kf) in zip(row_variants, per_pass[pi]):
                            states[pi][v.key].add_kernel_block(
                                day, n_head, n_head * n_prim, kc, kt, kp, ku, kk, kf)

        for (y, m) in plan.gpu_months:
            name = f"{y}-{m:02d}.parquet"
            hl, emb = self.headlines_dir / name, self.embeddings_dir / name
            if not hl.exists() or not emb.exists():
                continue
            for batch in sp._stream_month(hl, emb, batch_size=self.batch_size, threads=self.threads,
                                          duckdb_memory_limit=self.duckdb_memory_limit,
                                          temp_directory=self.temp_directory, day_lo=start, day_hi=end):
                day_ord = (batch["TIMESTAMP_UTC"].str.slice(0, 10).str.to_date()
                           .cast(pl.Int32).to_numpy())
                order = np.argsort(day_ord, kind="stable")
                day_ord = day_ord[order]
                X16 = batch["EMBEDDING"].to_numpy()[order].astype(np.float16)   # fp16 on the wire
                starts = np.flatnonzero(np.r_[True, day_ord[1:] != day_ord[:-1]])
                ends = np.r_[starts[1:], day_ord.shape[0]]
                for a, b in zip(starts, ends):
                    for i in range(a, b, self.score_block):
                        in_flight.append(self.worker.score.remote(X16[i:min(i + self.score_block, b)],
                                                                  int(day_ord[a])))
                        drain(self.max_in_flight)
            log.info("streamed %s to the GPU worker", name)
        drain(0)

        results = []
        for pi, ps in enumerate(self.passes):
            results.append(sp._assemble_results(states[pi], ps.variants, self.table, ps.cfg, ps.tau,
                                                ps.n_eff, mu_norm, False, float("nan"), start, end))
        return results, plan


def _month_end(y: int, m: int) -> date:
    return date(y + (m == 12), 1 if m == 12 else m + 1, 1) - __import__("datetime").timedelta(days=1)


def _merge_result_into_state(state: "sp._VariantState", r: sp.SpecResult) -> None:
    """Fold a finished SpecResult (CPU-scored month) back into a running state."""
    nd = r.narrative_daily
    for day, sub in nd.group_by("DATE", maintain_order=True):
        d = day[0] if isinstance(day, tuple) else day
        acc = state.narr.setdefault(d, sp._DayAccumulator(state.n_narr))
        idx = sub["narrative_key"].to_list()
        key_to_id = {k: i for i, k in enumerate(r.narrative_daily.filter(pl.col("DATE") == d)["narrative_key"].to_list())}
        count = sub["SUPPORT"].to_numpy().astype(np.int64)
        total = sub["TOTAL"].fill_null(0.0).to_numpy().astype(np.float64)
        peak = sub["PEAK"].fill_null(-np.inf).to_numpy().astype(np.float64)
        acc.add_arrays(count, total, peak)
    diag = r.day_diagnostics
    for row in diag.iter_rows(named=True):
        acc = state.narr.setdefault(row["DATE"], sp._DayAccumulator(state.n_narr))
        acc.n_headlines += row["n_headlines"]
        acc.n_unassigned += row["n_unassigned"]
    a = state.acct
    n = r.nan_accounting
    a.n_scores += n["n_scores"]
    a.n_nan_f0 += int(round(n["share_nan_f0"] * n["n_scores"]))
    a.n_nan_pct += int(round(n["share_nan_pct"] * n["n_scores"]))
    a.n_headlines += n["n_headlines"]
    a.n_unassigned += int(round(n["unassigned_share"] * n["n_headlines"]))


if __name__ == "__main__":
    import sys
    if "--my-ip" in sys.argv:
        from dotenv import find_dotenv, load_dotenv
        load_dotenv(find_dotenv(usecwd=True))
        print(my_ip_towards(_parse_host(os.environ["VERTEX_DIRECT_IP"])))
    else:
        print(DOCS)
