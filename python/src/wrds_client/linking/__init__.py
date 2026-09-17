"""Identifier resolution: (ISIN, CIK) -> gvkey -> permno -> secid.

    comp.security.isin  \\
                          -> gvkey -> (crsp_a_ccm.ccmxpf_linktable) -> permno
    comp.company.cik    /                                                \\
                                        (wrdsapps_link_crsp_optionm.opcrsphist) -> secid

`crsp_compustat.py` covers the first hop (reusable beyond OptionMetrics --
anything else keyed by permno could build on it); `optionmetrics.py` adds the
permno->secid hop and the end-to-end `resolve_universe_secids` orchestrator.
See `optionmetrics.py`'s module docstring for the ambiguity/point-in-time
contract.
"""
from __future__ import annotations

from universe.schema import UniverseEntry
from wrds_client.linking.optionmetrics import (
    SecidResolution,
    ingest_secid_resolution,
    resolve_universe_secids,
)

__all__ = [
    "SecidResolution",
    "UniverseEntry",
    "ingest_secid_resolution",
    "resolve_universe_secids",
]
