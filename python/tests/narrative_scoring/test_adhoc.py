"""Tests for the pure-math pieces of narrative_scoring.adhoc.

calibrate_null_model / score_window need real parquet corpora and are
exercised manually via the GFC/Covid notebooks; here we cover the
description-reshaping and mu-lookup helpers, which need neither.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from narrative_scoring.adhoc import as_pooling, centroid_from_raw, resolve_frozen_mu
from narrative_scoring.descriptions import DescriptionEmbeddings, PoolingMode

_MU_SCHEMA = pl.Schema(
    {
        "DATE": pl.Date,
        "MU": pl.Array(pl.Float32, 2),
        "MU_HAT": pl.Array(pl.Float32, 2),
        "N": pl.Int64,
    }
)


def _raw_desc(n_prim=3, k=4, dim=8, seed=0) -> DescriptionEmbeddings:
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n_prim, k, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=2, keepdims=True)
    return DescriptionEmbeddings(
        primitives=[f"p{i}" for i in range(n_prim)], mode=PoolingMode.MAX, vectors=vecs, k=k
    )


class TestCentroidFromRaw:
    def test_matches_manual_centroid(self):
        desc = _raw_desc()
        out = centroid_from_raw(desc)
        assert out.mode is PoolingMode.CENTROID
        assert out.vectors.shape == (3, 8)
        norms = np.linalg.norm(out.vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

        manual = desc.vectors[0].mean(axis=0)
        manual /= np.linalg.norm(manual)
        np.testing.assert_allclose(out.vectors[0], manual, atol=1e-5)

    def test_already_centroid_is_noop(self):
        desc = DescriptionEmbeddings(
            primitives=["p0"], mode=PoolingMode.CENTROID,
            vectors=np.array([[1.0, 0.0]], dtype=np.float32), k=1,
        )
        assert centroid_from_raw(desc) is desc


class TestAsPooling:
    def test_switches_mode_keeps_vectors(self):
        desc = _raw_desc()
        out = as_pooling(desc, PoolingMode.MEDIAN)
        assert out.mode is PoolingMode.MEDIAN
        np.testing.assert_array_equal(out.vectors, desc.vectors)

    def test_rejects_centroid_endpoints(self):
        desc = _raw_desc()
        with pytest.raises(ValueError, match="MAX and MEDIAN"):
            as_pooling(desc, PoolingMode.CENTROID)
        centroid = centroid_from_raw(desc)
        with pytest.raises(ValueError, match="MAX and MEDIAN"):
            as_pooling(centroid, PoolingMode.MAX)


class TestResolveFrozenMu:
    def test_raw_needs_no_mu_df(self):
        assert resolve_frozen_mu(None, date(2005, 12, 31)) == (None, None)

    def test_picks_latest_date_at_or_before_cutoff(self):
        mu_df = pl.DataFrame(
            {
                "DATE": [date(2005, 12, 28), date(2005, 12, 30), date(2006, 1, 2)],
                "MU": [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]],
                "MU_HAT": [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                "N": [10, 10, 10],
            },
            schema=_MU_SCHEMA,
        )
        # cutoff falls on a weekend with no row -- should pick the latest date <= cutoff
        mu, mu_hat = resolve_frozen_mu(mu_df, date(2005, 12, 31))
        np.testing.assert_allclose(mu, [0.2, 0.0])

    def test_raises_when_no_date_before_cutoff(self):
        mu_df = pl.DataFrame(
            {
                "DATE": [date(2006, 1, 2)],
                "MU": [[0.3, 0.0]],
                "MU_HAT": [[1.0, 0.0]],
                "N": [10],
            },
            schema=_MU_SCHEMA,
        )
        with pytest.raises(ValueError, match="no entry at or before"):
            resolve_frozen_mu(mu_df, date(2005, 12, 31))
