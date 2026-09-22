"""Shared fixtures: a tiny primitive taxonomy on disk and a matching table."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from narrative_scoring.primitives import PATH_FIELDS, PrimitiveTable, load_primitive_table
from narrative_scoring.schema import EMBEDDING_DIM

TAX_NAME = "Toy_v1"


def _path_id(row: dict[str, str]) -> str:
    return hashlib.sha1("/".join(row[c] for c in PATH_FIELDS).encode()).hexdigest()


def toy_rows() -> list[dict[str, str]]:
    """Two reservoirs; 'liquidity' appears in both (CATEGORY collision); one bipolar pair."""
    rows = []

    def add(topic, group, typ, sub, role, chan, name):
        cat = f"{typ}-{sub}" if sub else typ
        rows.append({
            "TOPIC": topic, "GROUP": group, "TYPE": typ, "SUB_TYPE": sub, "CATEGORY": cat,
            "ROLE": role, "OBSERVABILITY_CHANNEL": chan, "DISPLAY_NAME": name,
            "DESCRIPTION": f"desc {name}", "SCHEDULED": "", "VALID_ENTITY_TYPES": "", "TAGS": "",
            "POLARITY": sub,
        })

    add("macro", "funding", "liquidity", "stress", "interbank", "market", "m-liq-stress-ib")
    add("macro", "funding", "liquidity", "stress", "interbank", "official", "m-liq-stress-ib-off")
    add("macro", "funding", "liquidity", "stress", "repo", "market", "m-liq-stress-repo")
    add("macro", "funding", "liquidity", "easing", "interbank", "market", "m-liq-easing-ib")
    add("firm", "balance-sheet", "liquidity", "stress", "cash", "corporate", "f-liq-stress")
    add("firm", "balance-sheet", "solvency", "", "leverage", "corporate", "f-solv-lev")
    add("firm", "balance-sheet", "solvency", "", "leverage", "expert", "f-solv-lev-exp")
    add("politics", "elections", "surprise", "", "outcome", "event", "p-elec-surprise")
    return rows


def write_toy_taxonomy(root: Path, k: int = 2, style: str = "headline",
                       rows: list[dict[str, str]] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows = rows or toy_rows()
    pl.DataFrame(rows).write_csv(root / f"{TAX_NAME}_taxonomy.authored.csv")
    records = [
        {"id": _path_id(r), "display_name": r["DISPLAY_NAME"],
         "master": f"master {r['DISPLAY_NAME']}",
         "paraphrases": [f"{r['DISPLAY_NAME']} para {i}" for i in range(k)]}
        for r in rows
    ]
    (root / f"{TAX_NAME}-primitive_{style}_paraphrases.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n")
    return root


@pytest.fixture
def toy_root(tmp_path: Path) -> Path:
    return write_toy_taxonomy(tmp_path / "tax")


@pytest.fixture
def toy_table(toy_root: Path) -> PrimitiveTable:
    return load_primitive_table(toy_root, TAX_NAME, "headline")


def unit_rows(rng: np.random.Generator, n: int, dim: int = EMBEDDING_DIM) -> np.ndarray:
    X = rng.normal(size=(n, dim)).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


@pytest.fixture
def toy_embeddings(toy_table: PrimitiveTable) -> np.ndarray:
    rng = np.random.default_rng(11)
    return unit_rows(rng, toy_table.n_primitives * toy_table.n_texts)


class FixedTauProvider:
    """TEST HELPER ONLY: one TauRecord for every day (after its calibration window) plus
    as-of mu rows. Production has no frozen tau; this stands in for a tau_asof series
    in unit tests that are not about the tau job."""

    tau_policy = "test-fixed"

    def __init__(self, tau, mu_df=None, mu_asof_id=None, freeze_mu_at=None):
        from narrative_scoring.calibration import LookaheadError, resolve_mu_asof

        self.tau, self.mu_df, self.mu_asof_id = tau, mu_df, mu_asof_id
        self._resolve, self._lookahead = resolve_mu_asof, LookaheadError
        self._frozen = (resolve_mu_asof(mu_df, freeze_mu_at)
                        if freeze_mu_at and mu_df is not None else None)
        self.freeze_mu_at = freeze_mu_at

    @property
    def mu_policy(self):
        if self.mu_df is None:
            return "none"
        return f"frozen@{self.freeze_mu_at}" if self._frozen is not None else "as_of_day"

    @property
    def tau_source_id(self):
        return self.tau.digest()

    def mu_for(self, day):
        if self.mu_df is None:
            return None
        if self._frozen is not None:
            if self._frozen.date > day:
                raise self._lookahead(f"frozen mu dated {self._frozen.date} used for {day}")
            return self._frozen
        return self._resolve(self.mu_df, day)

    def tau_for(self, day):
        if self.tau.calibrated_through >= day:
            raise self._lookahead(
                f"tau calibrated through {self.tau.calibrated_through} cannot score {day}")
        return self.tau
