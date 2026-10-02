"""Datalake CLI.

    datalake ls [--kind KIND] [--group GROUP | --no-group] [--all]
    datalake show ARTIFACT_ID
    datalake verify [--no-hashes] [--artifact ID]
    datalake lineage ARTIFACT_ID [--down]
    datalake deprecate ARTIFACT_ID --reason TEXT
    datalake reindex
    datalake move ID [ID ...] (--group GROUP | --no-group) [--allow-partial] [--dry-run]
    datalake relink ID --replace OLD=NEW [...] [--set KEY=JSON ...] [--pipeline-version V]
                    --check SPEC [...] [--notes TEXT] [--deprecate-old] [--dry-run]
    datalake relink-plan ARTIFACT_ID

Check specs (datalake.equivalence.parse_check): files[@PARENT] |
projection[@PARENT]=COL,COL,OLD:NEW[,by=all] | keys[@PARENT]=KEY[:PARENT_KEY][,by=all] |
rerun=UNIT,UNIT[,rtol=..][,atol=..]. @PARENT (an old parent id) may be omitted when a
single parent is replaced.

The datalake root comes from --root, else $DATALAKE_ROOT, else ./datalake.
"""
from __future__ import annotations

import argparse
import json
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

    # exist_ok=True: never touches existing directories or their contents.
    for subdir in ("raw", "derived", "outputs"):
        (root / subdir).mkdir(exist_ok=True)

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
        group="" if args.no_group else args.group,
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
            f"group={artifact.group or '-'}  "
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
    print(f"  group      {meta.group or '(none)'}")
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
    if meta.relink:
        print(f"  relinked   from {meta.relink.get('from')}")
        for old, new in sorted((meta.relink.get("replace") or {}).items()):
            print(f"    {old} -> {new}")
        for check in meta.relink.get("checks") or []:
            state = "pass" if check.get("passed") else "FAIL"
            print(f"    [{state}] {check.get('name')}: {check.get('details')}")
    if meta.deprecated:
        print(f"  DEPRECATED {meta.deprecation_reason or '(no reason recorded)'}")
    return 0


def cmd_verify(index: DatalakeIndex, args: argparse.Namespace) -> int:
    report = verify(
        index,
        check_hashes=not args.no_hashes,
        check_lineage=not args.no_lineage,
        check_content=not args.no_content,
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


def cmd_move(index: DatalakeIndex, args: argparse.Namespace) -> int:
    group = None if args.no_group else args.group
    for artifact_id in args.artifact_ids:
        old, new = index.move(artifact_id, group, dry_run=args.dry_run,
                              allow_partial=args.allow_partial)
        verb = "would move" if args.dry_run else "moved"
        print(f"{verb} {artifact_id}\n    {old}\n -> {new}")
    return 0


def _pairs(items: list[str], what: str) -> dict[str, str]:
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key or not value:
            raise DatalakeError(f"{what} expects KEY=VALUE, got {item!r}")
        out[key] = value
    return out


def cmd_relink(index: DatalakeIndex, args: argparse.Namespace) -> int:
    from datalake.equivalence import parse_check
    from datalake.relink import RelinkError, relink

    set_params = {}
    for key, raw in _pairs(args.set or [], "--set").items():
        try:
            set_params[key] = json.loads(raw)
        except json.JSONDecodeError:
            set_params[key] = raw                      # a bare string
    try:
        result = relink(index, args.artifact_id, replace=_pairs(args.replace or [], "--replace"),
                        set_params=set_params, pipeline_version=args.pipeline_version,
                        checks=[parse_check(c) for c in args.check or []], notes=args.notes,
                        deprecate_old=args.deprecate_old, dry_run=args.dry_run)
    except RelinkError as exc:
        for check in exc.results:
            print(f"  [{'pass' if check.passed else 'FAIL'}] {check.name}: {check.details}")
        raise
    print(f"{'would relink' if args.dry_run else 'relinked'} {result.old_id}\n"
          f"  -> {result.new_id}\n  {result.path}")
    for old, new in result.sources_diff.items():
        print(f"  source {old} -> {new}")
    for key, (old, new) in result.hyperparams_diff.items():
        print(f"  param  {key}: {old!r} -> {new!r}")
    print(f"  files  {result.n_files} ({result.n_bytes / 1e9:.2f} GB, hard links)")
    for check in result.checks:
        print(f"  [{'pass' if check.passed else 'FAIL'}] {check.name}: {check.details}")
    return 0


def cmd_relink_plan(index: DatalakeIndex, args: argparse.Namespace) -> int:
    from datalake.relink import relink_order

    ids = relink_order(index, args.artifact_id)
    print(f"downstream of {args.artifact_id}, parents first:\n")
    if not ids:
        print("  (none)")
    for artifact_id in ids:
        art = index.get(artifact_id) if index.exists(artifact_id) else None
        flags = _flags(art) if art else "  [NOT REGISTERED]"
        print(f"  {artifact_id}{flags}")
        if art:
            print(f"      inputs: {', '.join(art.meta.sources)}")
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
    g_ls = p_ls.add_mutually_exclusive_group()
    g_ls.add_argument("--group", help="only artifacts filed under this group")
    g_ls.add_argument("--no-group", action="store_true", help="only artifacts without a group")
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
    p_verify.add_argument("--no-content", action="store_true",
                          help="skip per-kind content verifiers")
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

    p_mv = sub.add_parser("move", help="re-file artifacts under a group (content untouched)")
    p_mv.add_argument("artifact_ids", nargs="+")
    g_mv = p_mv.add_mutually_exclusive_group(required=True)
    g_mv.add_argument("--group", help="target group folder, e.g. RavenPack")
    g_mv.add_argument("--no-group", action="store_true", help="back to {layer}/{kind}/")
    p_mv.add_argument("--allow-partial", action="store_true",
                      help="move a partial artifact (only when no job is writing it)")
    p_mv.add_argument("--dry-run", action="store_true")
    p_mv.set_defaults(func=cmd_move)

    p_rl = sub.add_parser("relink", help="register an artifact's files under new inputs/code")
    p_rl.add_argument("artifact_id")
    p_rl.add_argument("--replace", action="append", metavar="OLD=NEW",
                      help="replace a source (old parent id = new parent id); repeatable")
    p_rl.add_argument("--set", action="append", metavar="KEY=JSON",
                      help="set a hyperparam explicitly (JSON value, else a string)")
    p_rl.add_argument("--pipeline-version", help="record a new pipeline version")
    p_rl.add_argument("--check", action="append", metavar="SPEC",
                      help="equivalence check (see above); repeatable")
    p_rl.add_argument("--notes", default="")
    p_rl.add_argument("--deprecate-old", action="store_true",
                      help="deprecate the old artifact once the relink is registered")
    p_rl.add_argument("--dry-run", action="store_true",
                      help="run the checks against the old files, register nothing")
    p_rl.set_defaults(func=cmd_relink)

    p_rp = sub.add_parser("relink-plan", help="descendants of an artifact, parents first")
    p_rp.add_argument("artifact_id")
    p_rp.set_defaults(func=cmd_relink_plan)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )
    try:
        with DatalakeIndex(_resolve_root(args.root)) as index:
            return args.func(index, args)
    except DatalakeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
