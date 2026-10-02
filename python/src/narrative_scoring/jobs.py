"""Command-line entry points for the production scorer.

    uv run python -m narrative_scoring.jobs register-taxonomy --taxonomy Evergreen_v5
    uv run python -m narrative_scoring.jobs score    --from 2004-01-01 --to 2004-12-31 \
                                                     --taxonomy Evergreen_v5
    uv run python -m narrative_scoring.jobs score    --from 2004-01-01 --to 2004-12-31 \
                                                     --mask-bipolar \
                                                     --sentiment-source ravenbert \
                                                     --sentiment-rule mean \
                                                     --tag "neg=x <= -1/3" \
                                                     --tag "neu=-1/3 < x < 1/3" \
                                                     --tag "pos=x >= 1/3" \
                                                     --assets-universe msci_world \
                                                     --min-relevance 0.6
    uv run python -m narrative_scoring.jobs tau-asof [--window 5Y | expanding] [--today ...]
    uv run python -m narrative_scoring.jobs resume <partial narrative_daily artifact id>
    uv run jobs start narrative_daily --from ... --to ... [same flags]   (same as score)
    uv run jobs resume <partial narrative_daily artifact id>             (same as resume)
    uv run python -m narrative_scoring.jobs mark-temp <artifact_id> [...]

``register-taxonomy`` validates {NAME}_taxonomy.authored.csv + both paraphrase JSONLs from
$RAW_DATA_PATH/Narrative_Taxonomy and registers them as a ``narrative_taxonomy`` artifact.
``score`` / ``tau-asof`` load the taxonomy from a registered artifact only: ``--taxonomy NAME``
(latest registration of that name) or ``--taxonomy-artifact ID`` (an exact one, for replays).
Scoring against several taxonomies = one run per taxonomy; their outputs never mix (every
family is resolved by the taxonomy hash).

One pass writes every output. ``--tag NAME=CONDITION`` (repeatable) adds a sentiment
label: CONDITION is an interval of the headline score x (tags.py: ``x <= -1/3``,
``-1/3 < x < 1/3``, ``(x > a) & (x < b)``; disjoint, checked at load), the score being
``--sentiment-rule`` argmax | mean | median (default mean) on the grid of a
``headline_sentiment`` artifact: ``--sentiment-source`` ravenbert | finbert (latest of that
producer built on the scored headlines) or ``--sentiment-artifact ID`` (an exact one).
Without ``--tag`` the run has one label, "all". The tags enter config_id (never the null
model's digests); the sentiment artifact is cited in the lineage.

``--assets-universe ID_OR_NAME`` adds the asset layer (assets.py): asset_attention_daily and
narrative_asset_daily, over every RavenPack entity ever in that registered universe; every
headline tagging an entity counts, and ``--min-relevance`` (a fraction, default 0.6 =
RELEVANCE >= 60) adds the *_REL columns restricted to the headlines at or above it. The
universe and the threshold enter the identity of the asset outputs only.

``--chunk-size`` (default artifacts.CHUNK_SIZE) is fixed for a run: recorded, and a resume
refuses another value (a block's scores can move by ~1e-7 with its row count).

Primitive texts are embedded through ``nlp`` (``--embedding-backend`` tei | local | embedx,
``--embedding-dtype`` float16 | float32; default TEI fp16), cached locally by texts +
embedder identity; the embedding provenance is written into every artifact of the run and
the primitive-embedding digest is part of every lookup key.

``--mask-bipolar`` keeps, per headline, one pole of every bipolar pair (selection.py); it
changes config_id and the f0 / warmup digests, so the masked run builds its own null
partitions and tau_asof. ``primitive_daily``
(day x primitive diagnostics) is written by default; ``--no-primitive-daily`` skips it.

``score`` is the one scoring path: it replays the live loop over the dates
in order (monthly tau_asof job, then the days), enforcing the 1-month delay
of the mu/tau providers and feeding the f0_monthly_partitions family. A
start before earliest data + 2 months is shifted forward with a warning.
``tau-asof`` runs the monthly job on its own (e.g. from a cron at month close).
``mark-temp`` tags agent-created artifacts TEMP and deprecates them (never deletes).

Paths come from the environment (.env): DATALAKE_ROOT (where the scorer's
families are written; a sandbox root for agent runs), DATALAKE_UPSTREAM_ROOT
(optional; where headlines / embeddings / mu_asof are read from when they are
not under DATALAKE_ROOT), RAW_DATA_PATH, CACHE_PATH.
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import date
from pathlib import Path

from narrative_scoring.config import (
    ParaphraseStyle,
    PoolRule,
    ScoringConfig,
    default_pooling,
)
from narrative_scoring.tags import SentimentTags
from nlp.corrections import Correction

log = logging.getLogger("narrative_scoring.jobs")


def _env(name: str) -> Path:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"{name} is not set (see .env)")
    return Path(v)


def _tags(args: argparse.Namespace) -> SentimentTags | None:
    """The sentiment layer of the command line: --tag NAME=CONDITION (repeatable)."""
    if not args.tag:
        if args.sentiment_source or args.sentiment_artifact:
            raise SystemExit("--sentiment-source / --sentiment-artifact without any --tag")
        return None
    pairs = []
    for spec in args.tag:
        name, sep, cond = spec.partition("=")
        if not sep or not name.strip() or not cond.strip():
            raise SystemExit(f"--tag expects NAME=CONDITION, got {spec!r}")
        pairs.append((name.strip(), cond.strip()))
    source = _sentiment_source_name(args)
    if not source:
        raise SystemExit("--tag needs --sentiment-source or --sentiment-artifact")
    try:
        return SentimentTags.build(source, args.sentiment_rule, pairs)
    except ValueError as exc:
        raise SystemExit(f"--tag: {exc}") from None


def _config(args: argparse.Namespace) -> ScoringConfig:
    style = ParaphraseStyle(args.style)
    pooling = PoolRule(args.pooling) if args.pooling else default_pooling(style)
    return ScoringConfig(
        mode=Correction(args.mode), paraphrase_style=style, paraphrase_pooling=pooling,
        q=args.q, jump_cut=args.jump_cut, tags=_tags(args), mask_bipolar=args.mask_bipolar,
        min_month_draws=args.min_month_draws, gap_alert_threshold=args.gap_alert_threshold,
        label=args.label,
    )


def _sentiment_source_name(args: argparse.Namespace) -> str | None:
    """The producer name for the tags: --sentiment-source, or the pinned artifact's."""
    if args.sentiment_artifact:
        from datalake import DatalakeIndex

        for root in (os.environ.get("DATALAKE_ROOT"), os.environ.get("DATALAKE_UPSTREAM_ROOT")):
            if not root:
                continue
            with DatalakeIndex(root, create=False) as dl:
                if dl.exists(args.sentiment_artifact):
                    return str(dl.get(args.sentiment_artifact).meta.hyperparams["source"])
        raise SystemExit(f"sentiment artifact {args.sentiment_artifact} not found")
    return args.sentiment_source


def _sentiment(args: argparse.Namespace, config: ScoringConfig, dl, upstream):
    """The headline_sentiment artifact behind the tags (None without tags)."""
    if config.tags is None:
        return None
    from narrative_scoring.artifacts import KIND_HEADLINES, resolve_sentiment

    art = resolve_sentiment(
        [dl, upstream], headlines_id=upstream.latest(KIND_HEADLINES).artifact_id,
        config=config, source=None if args.sentiment_artifact else config.tags.source,
        artifact_id=args.sentiment_artifact)
    log.info("sentiment: %s (%s, rule %s, tags %s)", art.artifact_id, config.tags.source,
             config.tags.rule, config.labels)
    return art


def _assets(args: argparse.Namespace, dl, upstream):
    """(AssetUniverse, universe artifact) of --assets-universe ID_OR_NAME, or (None, None)."""
    if not args.assets_universe:
        return None, None
    from narrative_scoring.assets import AssetUniverse
    from universe.ingest import find_universe_by_name

    ref = args.assets_universe
    art = None
    for ix in (dl, upstream):
        if ix.exists(ref):
            art = ix.get(ref)
            break
    if art is None:
        for ix in (dl, upstream):
            try:
                art = find_universe_by_name(ix, ref)
                break
            except ValueError:
                continue
    if art is None:
        raise SystemExit(f"no universe artifact with id or name {ref!r}")
    assets = AssetUniverse.from_artifact(art, min_relevance=args.min_relevance)
    log.info("assets: %s (%d entities, %d snapshots, min_relevance %g)", art.artifact_id,
             assets.n_assets, len(assets.snapshots), assets.min_relevance)
    return assets, art


def _indexes(args: argparse.Namespace):
    from datalake import DatalakeIndex

    dl = DatalakeIndex(_env("DATALAKE_ROOT"))
    up = os.environ.get("DATALAKE_UPSTREAM_ROOT")
    upstream = DatalakeIndex(up, create=False) if up else dl
    return dl, upstream


def _calendar(args: argparse.Namespace):
    from narrative_scoring.tau_asof import CalibrationCalendar

    return CalibrationCalendar(args.calibration_freq, args.calibration_delay)


def _table_and_embeddings(args: argparse.Namespace, dl, upstream):
    """The chosen registered taxonomy, its primitive table, primitive-text embeddings
    and their provenance (backend, serving metadata, checks)."""
    from narrative_scoring.artifacts import load_registered_table, resolve_taxonomy
    from narrative_scoring.primitives import embed_primitive_texts

    art = resolve_taxonomy([dl, upstream], name=None if args.taxonomy_artifact else args.taxonomy,
                           artifact_id=args.taxonomy_artifact)
    log.info("taxonomy: %s", art.artifact_id)
    table = load_registered_table(art, args.style)
    cache = (Path(args.embedding_cache) if args.embedding_cache
             else _env("CACHE_PATH") / "narrative_scoring")
    P, meta = embed_primitive_texts(table, cache, backend=args.embedding_backend,
                                    dtype=args.embedding_dtype)
    return art, table, P, meta


def cmd_register_taxonomy(args: argparse.Namespace) -> int:
    from narrative_scoring.artifacts import register_taxonomy

    dl, _ = _indexes(args)
    root = Path(args.source) if args.source else _env("RAW_DATA_PATH") / "Narrative_Taxonomy"
    art = register_taxonomy(dl, root, args.taxonomy, temp=args.temp)
    hp = art.meta.hyperparams
    print(f"{art.artifact_id}\n  primitives={hp['n_primitives']} narratives={hp['n_narratives']} "
          f"K={hp['k']} observability_channel={hp['has_observability_channel']}")
    return 0


def scoring_job_from_args(args: argparse.Namespace, dl, upstream):
    """The ``ScoringJob`` a ``score`` command line describes."""
    from narrative_scoring.artifacts import ScoringJob, headline_source

    config = _config(args)
    sentiment = _sentiment(args, config, dl, upstream)
    assets, universe = _assets(args, dl, upstream)
    tax_art, table, P, p_meta = _table_and_embeddings(args, dl, upstream)
    source = headline_source(upstream, chunk_size=args.chunk_size, sentiment=sentiment,
                             config=config, assets=assets)
    return ScoringJob(
        dl, date.fromisoformat(args.start), date.fromisoformat(args.end), config,
        table=table, P=P, upstream=upstream, source=source, window=args.window,
        seed=args.seed, threads=args.threads, rss_budget_gb=args.rss_budget_gb,
        keep_primitive_daily=not args.no_primitive_daily,
        temp=args.temp, label=args.label, taxonomy_id=tax_art.artifact_id,
        sentiment=sentiment, primitive_meta=p_meta, calendar=_calendar(args),
        chunk_size=args.chunk_size, assets=assets, universe=universe)


def cmd_score(args: argparse.Namespace) -> int:
    from narrative_scoring.artifacts import _runner

    dl, upstream = _indexes(args)
    job = scoring_job_from_args(args, dl, upstream)
    _runner(dl).start(job)
    summary = job.final_summary()
    for k in ("start", "end", "n_days_scored", "n_days_null_only", "peak_rss_gb",
              "months_finalised", "narrative_daily_id", "day_diagnostics_id",
              "partitions_id", "tau_asof_id", "asset_attention_id", "narrative_asset_id"):
        print(f"{k}={summary.get(k)}")
    return 0


def cmd_tau_asof(args: argparse.Namespace) -> int:
    from narrative_scoring.artifacts import build_tau_asof

    dl, upstream = _indexes(args)
    config = _config(args)
    tax_art, table, P, p_meta = _table_and_embeddings(args, dl, upstream)
    art = build_tau_asof(dl, config, table, P, upstream=upstream, window=args.window,
                         seed=args.seed, rebuild=args.rebuild, temp=args.temp,
                         taxonomy_id=tax_art.artifact_id, primitive_meta=p_meta,
                         calendar=_calendar(args),
                         today=date.fromisoformat(args.today) if args.today else None)
    print(art.artifact_id if art else "no partition old enough yet")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    """Everything (config, dates, taxonomy, inputs, embedder) is read from the run; the
    primitive-embedding backend must still serve the recorded model."""
    from narrative_scoring.artifacts import resume_scoring

    dl, upstream = _indexes(args)
    cache = (Path(args.embedding_cache) if args.embedding_cache
             else _env("CACHE_PATH") / "narrative_scoring")
    summary = resume_scoring(dl, args.artifact_id, upstream=upstream, cache_dir=cache,
                             threads=args.threads, chunk_size=args.chunk_size,
                             rss_budget_gb=args.rss_budget_gb)
    for k in ("start", "end", "n_days_scored", "n_days_null_only", "narrative_daily_id",
              "partitions_id", "tau_asof_id", "asset_attention_id", "narrative_asset_id"):
        print(f"{k}={summary.get(k)}")
    return 0


def cmd_mark_temp(args: argparse.Namespace) -> int:
    from narrative_scoring.artifacts import mark_temp_deprecated

    dl, _ = _indexes(args)
    for aid in args.artifact_ids:
        art = mark_temp_deprecated(dl, aid, args.reason)
        print(f"{art.artifact_id}: deprecated={art.deprecated} "
              f"agent_created={art.meta.hyperparams.get('agent_created')}")
    return 0


def _default(name: str):
    from narrative_scoring import artifacts

    return getattr(artifacts, name)


def add_scoring_args(parser: argparse.ArgumentParser) -> None:
    """The run's configuration flags (``score``, ``tau-asof`` and ``jobs start
    narrative_daily``)."""
    parser.add_argument("--taxonomy", default="Evergreen_v5",
                        help="registered taxonomy name (latest registration is used)")
    parser.add_argument("--taxonomy-artifact", default=None,
                        help="exact narrative_taxonomy artifact id (overrides --taxonomy)")
    parser.add_argument("--style", default="headline", choices=[s.value for s in ParaphraseStyle])
    parser.add_argument("--mode", default="r2", choices=[m.value for m in Correction])
    parser.add_argument("--pooling", default=None, choices=[p.value for p in PoolRule])
    parser.add_argument("--q", type=float, default=0.99)
    parser.add_argument("--jump-cut", action="store_true")
    parser.add_argument("--mask-bipolar", action="store_true",
                        help="keep one pole per bipolar pair per headline (enters config_id, "
                             "f0 and warmup digests: tau and the null partitions are rebuilt)")
    parser.add_argument("--no-primitive-daily", action="store_true",
                        help="do not write the day x primitive diagnostics (written by default, "
                             "~300 MB per run)")
    parser.add_argument("--tag", action="append", default=[], metavar="NAME=CONDITION",
                        help="a sentiment label: NAME=interval of the score x, e.g. "
                             "'neg=x <= -1/3' (repeatable; tags.py)")
    parser.add_argument("--sentiment-source", default=None,
                        help="headline_sentiment grid producer: ravenbert | finbert")
    parser.add_argument("--sentiment-artifact", default=None,
                        help="exact headline_sentiment artifact id (overrides the source)")
    parser.add_argument("--sentiment-rule", default="mean", choices=["argmax", "mean", "median"],
                        help="ordinal rule giving the score from the grid (default mean)")
    parser.add_argument("--assets-universe", default=None, metavar="ID_OR_NAME",
                        help="registered universe (with RavenPack ids): adds the asset outputs")
    parser.add_argument("--min-relevance", type=float, default=0.6,
                        help="relevance threshold of the *_REL asset columns, a fraction of "
                             "the vendor's 0-100 RELEVANCE (default 0.6 = RELEVANCE >= 60)")
    parser.add_argument("--min-month-draws", type=int, default=20_000_000)
    parser.add_argument("--gap-alert-threshold", type=float, default=0.05)
    parser.add_argument("--label", default="")
    parser.add_argument("--calibration-freq", default="M", choices=["D", "W", "M", "Q", "Y"],
                        help="tau job: null-partition period (default M)")
    parser.add_argument("--calibration-delay", default="1M",
                        help="tau job: publication delay, e.g. 1M, 1Q, 7d (default 1M)")
    parser.add_argument("--embedding-backend", default="tei",
                        choices=["tei", "local", "embedx"],
                        help="engine for the primitive-text embeddings (default: tei)")
    parser.add_argument("--embedding-dtype", default="float16",
                        help="compute dtype of the primitive-text embeddings (default: float16)")
    parser.add_argument("--embedding-cache", default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=_default("CHUNK_SIZE"),
                        help="rows per scoring block, fixed for the run (recorded)")
    parser.add_argument("--rss-budget-gb", type=float, default=_default("RSS_BUDGET_GB"))
    parser.add_argument("--window", default="5Y")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temp", action="store_true",
                        help="mark everything registered as agent-created / TEMP")


def main(argv: list[str] | None = None) -> int:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    add_scoring_args(common)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("register-taxonomy", parents=[common])
    p.add_argument("--source", default=None,
                   help="directory holding the taxonomy files (default: "
                        "$RAW_DATA_PATH/Narrative_Taxonomy)")
    p.set_defaults(fn=cmd_register_taxonomy)
    p = sub.add_parser("score", parents=[common])
    p.add_argument("--from", dest="start", required=True)
    p.add_argument("--to", dest="end", required=True)
    p.set_defaults(fn=cmd_score)
    p = sub.add_parser("tau-asof", parents=[common])
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--today", default=None)
    p.set_defaults(fn=cmd_tau_asof)
    p = sub.add_parser("resume", help="continue an interrupted score run from the run alone")
    p.add_argument("artifact_id", help="the partial narrative_daily artifact of the run")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=None,
                   help="must equal the recorded one (default: the recorded one)")
    p.add_argument("--rss-budget-gb", type=float, default=_default("RSS_BUDGET_GB"))
    p.add_argument("--embedding-cache", default=None)
    p.set_defaults(fn=cmd_resume)
    p = sub.add_parser("mark-temp")
    p.add_argument("artifact_ids", nargs="+")
    p.add_argument("--reason", default="superseded by owner run")
    p.set_defaults(fn=cmd_mark_temp)
    args = ap.parse_args(argv)
    if args.cmd in ("score", "resume"):
        log.warning("deprecated: use `jobs start narrative_daily ...` / `jobs resume <id>` "
                    "(same Job: status, pause, lock, job.log)")
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
