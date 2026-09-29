"""LSEG Data Library connector: Workspace desktop session (e.g. through an SSH tunnel) first,
Data Platform (RDP) fallback. See ``lseg_client.client`` for what the entitlements give.

    from lseg_client import LsegClient
    with LsegClient.connect() as lseg:
        lseg.index_members(".STOXX")
"""
from __future__ import annotations

from lseg_client.client import LsegClient
from lseg_client.enrich import backfill_lseg_ids
from lseg_client.session import LsegConfig, SessionUnavailable, open_session

__all__ = ["LsegClient", "LsegConfig", "SessionUnavailable", "backfill_lseg_ids",
           "open_session"]
