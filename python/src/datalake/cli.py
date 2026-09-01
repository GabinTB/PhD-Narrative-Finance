"""Datalake CLI.

    datalake init [--root PATH]
    datalake ls [--kind KIND] [--all]
    datalake show ARTIFACT_ID
    datalake verify [--no-hashes] [--artifact ID]
    datalake lineage ARTIFACT_ID [--down]
    datalake deprecate ARTIFACT_ID --reason TEXT
    datalake reindex

The datalake root comes from --root, else $DATALAKE_ROOT, else ./datalake.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from datalake.index import DatalakeError, DatalakeIndex
from datalake.verify import Severity, verify

log = logging.getLogger("datalake")


def _resolve_root(arg_root: str | None) -> Path:
    if arg_root:
        return Path(arg_root)
    env_root = os.environ.get("DATALAKE_ROOT")
    if env_root:
        return Path(env_root)
    return Path("datalake")


def _flags(artifact) -> str:
    marks = []
    if artifact.partial:
        marks.append("PARTIAL")
    if artifact.deprecated:
        marks.append("DEPRECATED")
    return f"  [{','.join(marks)}]" if marks else ""


def cmd_init(index: DatalakeIndex, args: argparse.Namespace) -> int:
    root = index.root

    # Directories: exist_ok=True means existing data is never touched.
    for subdir in ("raw", "derived", "outputs"):
        (root / subdir).mkdir(exist_ok=True)

    # Only reindex if explicitly requested or index is currently empty.
    n_existing = index._conn.execute(
        "SELECT COUNT(*) FROM artifacts"
    ).fetchone()[0]

    if args.reindex or n_existing == 0:
        count = index.reindex()
        print(f"datalake initialised at {root}")
        print(f"reindexed {count} artifact(s)")
    else:
        print(f"datalake initialised at {root}")
        print(f"{n_existing} artifact(s) already registered, index untouched")
        print("pass --reindex to force a full rebuild from sidecars")

    return 0


def cmd_ls(index: DatalakeIndex, args: argparse.Namespace) -> int:
    artifacts = index.list(
        kind=args.kind,
        model=args.model,
        include_partial=args.all,
        include_deprecated=args.all,
    )
    if not artifacts:
        print("no artifacts found")
        return 0
    for artifact in artifacts:
        n_files = len(artifact.file_hashes)
        print(
            f"{artifact.artifact_id}\n"
            f"    kind={artifact.kind}  layer={artifact.layer}  "
            f"files={n_files}  started={artifact.meta.run_start}"
            f"{_flags(artifact)}"
        )
    print(f"\n{len(artifacts)} artifact(s)")
    return 0


def cmd_show(index: DatalakeIndex, args: argparse.Namespace) -> int:
    artifact = index.get(args.artifact_id)
    meta = artifact.meta
    print(f"{artifact.artifact_id}{_flags(artifact)}\n")
    print(f"  layer      {artifact.layer}")
    print(f"  kind       {meta.kind}")
    print(f"  path       {artifact.path}")
    print(f"  pipeline   {meta.pipeline} {meta.pipeline_version}")
    print(f"  commit     {meta.pipeline_commit or '(none recorded)'}")
    print(f"  started    {meta.run_start}")
    print(f"  finished   {meta.run_end or '(incomplete)'}")
    if meta.hyperparams:
        print("  hyperparams")
        for key in sorted(meta.hyperparams):
            print(f"    {key} = {meta.hyperparams[key]!r}")
    if meta.model_card:
        card = meta.model_card
        print(f"  model      {card.model_id} {card.version}")
        print(f"    weights  {'public' if card.weights_public else 'private'}")
    if meta.sources:
        print("  inputs")
        for source in meta.sources:
            print(f"    {source}")
    if artifact.file_hashes:
        print(f"  files      {len(artifact.file_hashes)}")
    if meta.deprecated:
        print(f"  DEPRECATED {meta.deprecation_reason or '(no reason recorded)'}")
    return 0


def cmd_verify(index: DatalakeIndex, args: argparse.Namespace) -> int:
    report = verify(
        index,
        check_hashes=not args.no_hashes,
        check_lineage=not args.no_lineage,
        artifact_ids=[args.artifact] if args.artifact else None,
    )
    for finding in report.findings:
        stream = sys.stderr if finding.severity is Severity.ERROR else sys.stdout
        print(finding, file=stream)
    print(f"\n{report.summary()}")
    return 0 if report.ok else 1


def cmd_lineage(index: DatalakeIndex, args: argparse.Namespace) -> int:
    if not index.exists(args.artifact_id):
        raise DatalakeError(f"no artifact with id {args.artifact_id!r}")

    if args.down:
        ids = index.descendants(args.artifact_id)
        header = "downstream of"
    else:
        ids = index.ancestors(args.artifact_id)
        header = "upstream of"

    print(f"{header} {args.artifact_id}:\n")
    if not ids:
        print("  (none)")
        return 0
    for artifact_id in ids:
        if index.exists(artifact_id):
            print(f"  {artifact_id}{_flags(index.get(artifact_id))}")
        else:
            print(f"  {artifact_id}  [NOT REGISTERED]")
    print(f"\n{len(ids)} artifact(s)")
    return 0


def cmd_deprecate(index: DatalakeIndex, args: argparse.Namespace) -> int:
    index.deprecate(args.artifact_id, args.reason)
    print(f"deprecated {args.artifact_id}")
    downstream = index.descendants(args.artifact_id)
    if downstream:
        print(f"\n{len(downstream)} downstream artifact(s) now built on deprecated input:")
        for artifact_id in downstream:
            print(f"  {artifact_id}")
    return 0


def cmd_reindex(index: DatalakeIndex, args: argparse.Namespace) -> int:
    count = index.reindex()
    print(f"reindexed {count} artifact(s) from {index.root}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="datalake",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", help="datalake root (default: $DATALAKE_ROOT or ./datalake)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="initialise a new datalake root")
    p_init.add_argument(
        "--reindex", action="store_true",
        help="rebuild index.db from meta.json sidecars (safe, never touches data files)",
    )
    p_init.set_defaults(func=cmd_init)

    p_ls = sub.add_parser("ls", help="list artifacts")
    p_ls.add_argument("--kind")
    p_ls.add_argument("--model")
    p_ls.add_argument("--all", action="store_true",
                      help="include partial and deprecated artifacts")
    p_ls.set_defaults(func=cmd_ls)

    p_show = sub.add_parser("show", help="show one artifact in detail")
    p_show.add_argument("artifact_id")
    p_show.set_defaults(func=cmd_show)

    p_verify = sub.add_parser("verify", help="check hashes, completeness, lineage")
    p_verify.add_argument("--no-hashes", action="store_true",
                          help="skip re-hashing (fast structural check only)")
    p_verify.add_argument("--no-lineage", action="store_true")
    p_verify.add_argument("--artifact", help="verify a single artifact")
    p_verify.set_defaults(func=cmd_verify)

    p_lineage = sub.add_parser("lineage", help="trace inputs or dependants")
    p_lineage.add_argument("artifact_id")
    p_lineage.add_argument("--down", action="store_true",
                           help="show dependants instead of inputs")
    p_lineage.set_defaults(func=cmd_lineage)

    p_dep = sub.add_parser("deprecate", help="mark an artifact deprecated")
    p_dep.add_argument("artifact_id")
    p_dep.add_argument("--reason", required=True)
    p_dep.set_defaults(func=cmd_deprecate)

    p_re = sub.add_parser("reindex", help="rebuild index.db from meta.json sidecars")
    p_re.set_defaults(func=cmd_reindex)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )

    from dotenv import load_dotenv
    load_dotenv()                    # add this

    try:
        with DatalakeIndex(_resolve_root(args.root)) as index:
            return args.func(index, args)
    except DatalakeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
