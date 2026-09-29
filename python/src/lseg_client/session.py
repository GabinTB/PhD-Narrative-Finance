"""Open an LSEG Data Library session: Workspace desktop (e.g. through an SSH tunnel) first,
the Data Platform (RDP) as fallback.

Environment (``.env``; values are never logged):

    LSEG_APP_KEY or LSEG_CLIENT_SECRET   app key (40 hex), used by both sessions
    LSEG_DESKTOP_URL                     Workspace API proxy, e.g. http://127.0.0.1:9006 when
                                         Workspace runs on another machine and its port is
                                         forwarded here; unset = the library's own default
    LSEG_USERNAME (or LSEG_USERNAME_1, LSEG_USERNAME_2), LSEG_PASSWORD
                                         Data Platform login, tried in that order

The platform grant allows one open session per user, and a running Workspace logged in with
the same user holds it ("Session quota is reached"): the fallback is for when Workspace is
not running. Nothing here forces a session (``signon_control=True`` would close the user's
other session), and the refusal is reported as ``SessionUnavailable`` with each attempt's
reason.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

DESKTOP, PLATFORM = "desktop", "platform"


class SessionUnavailable(RuntimeError):
    """No LSEG session could be opened; the message lists every attempt."""


@dataclass(frozen=True)
class LsegConfig:
    app_key: str | None
    desktop_url: str | None = None
    usernames: tuple[str, ...] = ()
    password: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> LsegConfig:
        env = os.environ
        users = tuple(u for u in (env.get("LSEG_USERNAME"), env.get("LSEG_USERNAME_1"),
                                  env.get("LSEG_USERNAME_2")) if u)
        return cls(app_key=env.get("LSEG_APP_KEY") or env.get("LSEG_CLIENT_SECRET"),
                   desktop_url=env.get("LSEG_DESKTOP_URL"),
                   usernames=tuple(dict.fromkeys(users)), password=env.get("LSEG_PASSWORD"))


def _is_open(session: Any) -> bool:
    return str(getattr(session, "open_state", "")).endswith("Opened")


def _close_quietly(session: Any) -> None:
    try:
        session.close()
    except Exception:  # noqa: BLE001 - best effort on a session that never opened
        pass


def open_session(config: LsegConfig, *, prefer: tuple[str, ...] = (DESKTOP, PLATFORM),
                 ld: Any = None) -> tuple[Any, str]:
    """(open session, its kind), set as the library's default session. Tries ``prefer`` in
    order; raises ``SessionUnavailable`` listing why each attempt failed."""
    if ld is None:
        import lseg.data as ld
    if not config.app_key:
        raise SessionUnavailable("no app key: set LSEG_APP_KEY (or LSEG_CLIENT_SECRET)")
    failures: list[str] = []
    for kind in prefer:
        if kind == DESKTOP:
            attempts = [None]
        elif kind == PLATFORM:
            if not (config.usernames and config.password):
                failures.append("platform: no LSEG_USERNAME / LSEG_PASSWORD")
                continue
            attempts = list(config.usernames)
        else:
            raise ValueError(f"unknown session kind {kind!r}")
        for user in attempts:
            try:
                if kind == DESKTOP:
                    if config.desktop_url:
                        ld.get_config()["sessions.desktop.workspace.base-url"] = config.desktop_url
                    session = ld.session.desktop.Definition(app_key=config.app_key).get_session()
                else:
                    grant = ld.session.platform.GrantPassword(username=user,
                                                              password=config.password)
                    session = ld.session.platform.Definition(
                        app_key=config.app_key, grant=grant, signon_control=False).get_session()
                session.open()
            except Exception as exc:  # noqa: BLE001 - reported, next attempt
                failures.append(f"{kind}: {type(exc).__name__}")
                continue
            if _is_open(session):
                ld.session.set_default(session)
                log.info("LSEG %s session open", kind)
                return session, kind
            _close_quietly(session)
            failures.append(f"{kind}: not opened (a running Workspace holds the platform "
                            "session quota)" if kind == PLATFORM else f"{kind}: not opened "
                            "(Workspace not running or not reachable at LSEG_DESKTOP_URL)")
    raise SessionUnavailable("no LSEG session: " + "; ".join(failures))
