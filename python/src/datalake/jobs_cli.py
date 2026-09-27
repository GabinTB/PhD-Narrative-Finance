"""``jobs``: one command line for every long pipeline.

    jobs start <kind> [job arguments]        a fresh run (``jobs start <kind> -h``)
    jobs resume <artifact_id> [--opt k=v]    finish a partial run from the artifact alone
    jobs update <artifact_id> [--opt k=v]    add new data to a complete artifact (new execution)
    jobs pause <artifact_id>                 stop it at the next unit boundary
    jobs status [<artifact_id>]              one job, or every partial job
    jobs list [--state running|stale|paused|failed|partial]
    jobs check <artifact_id> [--opt k=v]     what would stop a resume now

``--opt key=value`` passes throughput knobs (batch_size, device, threads, ...)
to the job; everything that defines the artifact comes from the artifact.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from datalake.index import DatalakeIndex
from datalake.jobs import JobRunner, job_class, list_jobs, registered_jobs


def _value(text: str) -> Any:
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _opts(pairs: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--opt expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        out[key.strip().replace("-", "_")] = _value(value.strip())
    return out


def _print_status(st: Any) -> None:
    d = asdict(st)
    progress = (f"{d['units_done']}/{d['units_total']}" if d["units_total"] is not None else "?")
    beat = f"{d['heartbeat_age_s']:.0f}s" if d["heartbeat_age_s"] is not None else "-"
    print(f"{d['status']:<9} {progress:>9}  {d['kind']:<24} {d['artifact_id']}")
    extra = [f"unit={d['current_unit']}" if d["current_unit"] else "",
             f"heartbeat={beat}", f"host={d['host']} pid={d['pid']}" if d["host"] else "",
             f"eta={d['eta']}" if d["eta"] else "",
             f"error={d['last_error']}" if d["last_error"] else ""]
    print("          " + "  ".join(x for x in extra if x))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="jobs", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datalake-root", help="overrides $DATALAKE_ROOT")
    ap.add_argument("--env", default=None, help="extra .env file to load")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--log-json", action="store_true", help="JSON lines in job.log")
    ap.add_argument("--no-log-file", action="store_true", help="do not write job.log")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="let a non-TEMP run start from a dirty git tree (recorded)")
    sub = ap.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="start a fresh run")
    kinds = start.add_subparsers(dest="kind", required=True)
    for kind, cls in sorted(registered_jobs().items()):
        cls.add_cli_args(kinds.add_parser(kind, help=(cls.__doc__ or "").strip().split("\n")[0]))

    for name, help_ in (("resume", "finish a partial run"), ("check", "check a partial run"),
                        ("update", "add new data to a complete artifact")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("artifact_id")
        p.add_argument("--opt", action="append", metavar="KEY=VALUE",
                       help="throughput knob passed to the job (repeatable)")
    p = sub.add_parser("pause", help="stop a running job at the next unit boundary")
    p.add_argument("artifact_id")
    p = sub.add_parser("status", help="status of one job, or of every partial job")
    p.add_argument("artifact_id", nargs="?")
    p = sub.add_parser("list", help="partial jobs")
    p.add_argument("--state", choices=["running", "stale", "paused", "failed", "partial"])
    return ap


def main(argv: list[str] | None = None) -> int:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.env:
        load_dotenv(args.env, override=True)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = args.datalake_root or os.environ.get("DATALAKE_ROOT")
    if not root:
        print("DATALAKE_ROOT must be set (in .env, environment, or --datalake-root)",
              file=sys.stderr)
        return 2
    with DatalakeIndex(Path(root)) as index:
        runner = JobRunner(index, log_file=not args.no_log_file, log_level=args.log_level,
                           log_json=args.log_json, allow_dirty=args.allow_dirty)
        if args.command == "start":
            art = runner.start(job_class(args.kind).from_args(args, index))
            _print_status(runner.status(art.artifact_id))
        elif args.command == "resume":
            art = runner.resume(args.artifact_id, **_opts(args.opt))
            _print_status(runner.status(art.artifact_id))
        elif args.command == "update":
            art = runner.update(args.artifact_id, **_opts(args.opt))
            _print_status(runner.status(art.artifact_id))
        elif args.command == "pause":
            runner.pause(args.artifact_id)
            print(f"pause requested for {args.artifact_id} (effective at the next unit)")
        elif args.command == "check":
            problems = runner.check(args.artifact_id, **_opts(args.opt))
            for problem in problems:
                print(problem)
            print("ok" if not problems else f"{len(problems)} problem(s)")
            return 0 if not problems else 1
        elif args.command == "status" and args.artifact_id:
            _print_status(runner.status(args.artifact_id))
        else:                                     # status (all) / list
            state = getattr(args, "state", None)
            for st in list_jobs(index, runner):
                if state is None or st.status == state:
                    _print_status(st)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
