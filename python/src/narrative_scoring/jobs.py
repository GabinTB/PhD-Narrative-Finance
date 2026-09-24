"""Command-line entry points for the production scorer.

    uv run python -m narrative_scoring.jobs register-taxonomy --taxonomy Evergreen_v5
    uv run python -m narrative_scoring.jobs score    --from 2004-01-01 --to 2004-12-31 \
                                                     --taxonomy Evergreen_v5
    uv run python -m narrative_scoring.jobs score    --from 2004-01-01 --to 2004-12-31 \
                                                     --split sign --sentiment-source finbert \
                                                     --sentiment-column SENT_BAND \
                                                     --neutral-eps 0.3333333
    uv run python -m narrative_scoring.jobs tau-asof [--window 5Y | expanding] [--today ...]
    uv run python -m narrative_scoring.jobs mark-temp <artifact_id> [...]

``register-taxonomy`` validates {NAME}_taxonomy.authored.csv + both paraphrase JSONLs from
$RAW_DATA_PATH/Narrative_Taxonomy and registers them as a ``narrative_taxonomy`` artifact.
``score`` / ``tau-asof`` load the taxonomy from a registered artifact only: ``--taxonomy NAME``
(latest registration of that name) or ``--taxonomy-artifact ID`` (an exact one, for replays).
Scoring against several taxonomies = one run per taxonomy; their outputs never mix (every
family is resolved by the taxonomy hash).

``--split sign`` adds pos/neg(/neu) rows from one SENT_* column of a ``headline_sentiment``
artifact: ``--sentiment-source NAME`` (latest of that producer built on the scored headlines)
or ``--sentiment-artifact ID`` (an exact one), plus ``--sentiment-column`` (required) and
``--neutral-eps``. Source and column enter config_id; the artifact is cited in the lineage.

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
    SENTIMENT_NONE,
    ParaphraseStyle,
    PoolRule,
    ScoringConfig,
    SentimentSplit,
    default_pooling,
)
from narrative_scoring.corrections import Correction

log = logging.getLogger("narrative_scoring.jobs")


def _env(name: str) -> Path:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"{name} is not set (see .env)")
    return Path(v)


def _config(args: argparse.Namespace) -> ScoringConfig:
    style = ParaphraseStyle(args.style)
    pooling = PoolRule(args.pooling) if args.pooling else default_pooling(style)
    return ScoringConfig(
        mode=Correction(args.mode), paraphrase_style=style, paraphrase_pooling=pooling,
        q=args.q, jump_cut=args.jump_cut, sentiment_split=SentimentSplit(args.split),
        neutral_eps=args.neutral_eps,
        sentiment_source=_sentiment_source_name(args),
        sentiment_column=args.sentiment_column or "",
        min_month_draws=args.min_month_draws, gap_alert_threshold=args.gap_alert_threshold,
        label=args.label,
    )


def _sentiment_source_name(args: argparse.Namespace) -> str:
    """The producer name for the config: --sentiment-source, or the pinned artifact's."""
    if args.sentiment_artifact:
        from datalake import DatalakeIndex

        for root in (os.environ.get("DATALAKE_ROOT"), os.environ.get("DATALAKE_UPSTREAM_ROOT")):
            if not root:
                continue
            with DatalakeIndex(root, create=False) as dl:
                if dl.exists(args.sentiment_artifact):
                    return str(dl.get(args.sentiment_artifact).meta.hyperparams["source"])
        raise SystemExit(f"sentiment artifact {args.sentiment_artifact} not found")
    return args.sentiment_source or SENTIMENT_NONE


def _sentiment(args: argparse.Namespace, config: ScoringConfig, dl, upstream):
    """The headline_sentiment artifact of a split run (None without a split)."""
    if config.sentiment_split is not SentimentSplit.SIGN:
        return None
    from narrative_scoring.artifacts import KIND_HEADLINES, resolve_sentiment

    art = resolve_sentiment(
        [dl, upstream], headlines_id=upstream.latest(KIND_HEADLINES).artifact_id,
        column=config.sentiment_column,
        source=None if args.sentiment_artifact else config.sentiment_source,
        artifact_id=args.sentiment_artifact)
    log.info("sentiment: %s column %s", art.artifact_id, config.sentiment_column)
    return art


def _indexes(args: argparse.Namespace):
    from datalake import DatalakeIndex

    dl = DatalakeIndex(_env("DATALAKE_ROOT"))
    up = os.environ.get("DATALAKE_UPSTREAM_ROOT")
    upstream = DatalakeIndex(up, create=False) if up else dl
    return dl, upstream


def _table_and_embeddings(args: argparse.Namespace, dl, upstream):
    """The chosen registered taxonomy, its primitive table and primitive-text embeddings."""
    from narrative_scoring.artifacts import load_registered_table, resolve_taxonomy
    from narrative_scoring.primitives import embed_primitive_texts

    art = resolve_taxonomy([dl, upstream], name=None if args.taxonomy_artifact else args.taxonomy,
                           artifact_id=args.taxonomy_artifact)
    log.info("taxonomy: %s", art.artifact_id)
    table = load_registered_table(art, args.style)
    cache = (Path(args.embedding_cache) if args.embedding_cache
             else _env("CACHE_PATH") / "narrative_scoring")
    return art, table, embed_primitive_texts(table, cache, device=args.device)


def cmd_register_taxonomy(args: argparse.Namespace) -> int:
    from narrative_scoring.artifacts import register_taxonomy

    dl, _ = _indexes(args)
    root = Path(args.source) if args.source else _env("RAW_DATA_PATH") / "Narrative_Taxonomy"
    art = register_taxonomy(dl, root, args.taxonomy, temp=args.temp)
    hp = art.meta.hyperparams
    print(f"{art.artifact_id}\n  primitives={hp['n_primitives']} narratives={hp['n_narratives']} "
          f"K={hp['k']} observability_channel={hp['has_observability_channel']}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from narrative_scoring.artifacts import headline_source, score_range_to_datalake

    dl, upstream = _indexes(args)
    config = _config(args)
    sentiment = _sentiment(args, config, dl, upstream)
    tax_art, table, P = _table_and_embeddings(args, dl, upstream)
    source = headline_source(upstream, chunk_size=args.chunk_size, threads=args.threads,
                             sentiment=sentiment, sentiment_column=config.sentiment_column)
    summary = score_range_to_datalake(
        dl, date.fromisoformat(args.start), date.fromisoformat(args.end), config,
        table=table, P=P, upstream=upstream, source=source, window=args.window,
        seed=args.seed, threads=args.threads, rss_budget_gb=args.rss_budget_gb,
        temp=args.temp, label=args.label, taxonomy_id=tax_art.artifact_id,
        sentiment=sentiment)
    for k in ("start", "end", "n_days_scored", "n_days_null_only", "peak_rss_gb",
              "months_finalised", "narrative_daily_id", "day_diagnostics_id",
              "partitions_id", "tau_asof_id"):
        print(f"{k}={summary[k]}")
    return 0


def cmd_tau_asof(args: argparse.Namespace) -> int:
    from narrative_scoring.artifacts import build_tau_asof

    dl, upstream = _indexes(args)
    config = _config(args)
    tax_art, table, P = _table_and_embeddings(args, dl, upstream)
    art = build_tau_asof(dl, config, table, P, upstream=upstream, window=args.window,
                         seed=args.seed, rebuild=args.rebuild, temp=args.temp,
                         taxonomy_id=tax_art.artifact_id,
                         today=date.fromisoformat(args.today) if args.today else None)
    print(art.artifact_id if art else "no partition old enough yet")
    return 0


def cmd_mark_temp(args: argparse.Namespace) -> int:
    from narrative_scoring.artifacts import mark_temp_deprecated

    dl, _ = _indexes(args)
    for aid in args.artifact_ids:
        art = mark_temp_deprecated(dl, aid, args.reason)
        print(f"{art.artifact_id}: deprecated={art.deprecated} "
              f"agent_created={art.meta.hyperparams.get('agent_created')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--taxonomy", default="Evergreen_v5",
                        help="registered taxonomy name (latest registration is used)")
    common.add_argument("--taxonomy-artifact", default=None,
                        help="exact narrative_taxonomy artifact id (overrides --taxonomy)")
    common.add_argument("--style", default="headline", choices=[s.value for s in ParaphraseStyle])
    common.add_argument("--mode", default="r2", choices=[m.value for m in Correction])
    common.add_argument("--pooling", default=None, choices=[p.value for p in PoolRule])
    common.add_argument("--q", type=float, default=0.99)
    common.add_argument("--jump-cut", action="store_true")
    common.add_argument("--split", default="none", choices=[s.value for s in SentimentSplit])
    common.add_argument("--neutral-eps", type=float, default=0.0,
                        help="|sentiment| <= eps is 'neu' (split runs)")
    common.add_argument("--sentiment-source", default=None,
                        help="headline_sentiment producer: ravenpack | ravenbert | finbert")
    common.add_argument("--sentiment-artifact", default=None,
                        help="exact headline_sentiment artifact id (overrides the source)")
    common.add_argument("--sentiment-column", default=None,
                        help="SENT_* column of the sentiment artifact (split runs)")
    common.add_argument("--min-month-draws", type=int, default=20_000_000)
    common.add_argument("--gap-alert-threshold", type=float, default=0.05)
    common.add_argument("--label", default="")
    common.add_argument("--device", default="embedx")
    common.add_argument("--embedding-cache", default=None)
    common.add_argument("--threads", type=int, default=8)
    common.add_argument("--chunk-size", type=int, default=8_192)
    common.add_argument("--rss-budget-gb", type=float, default=30.0)
    common.add_argument("--window", default="5Y")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--temp", action="store_true",
                        help="mark everything registered as agent-created / TEMP")
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
    p = sub.add_parser("mark-temp")
    p.add_argument("artifact_ids", nargs="+")
    p.add_argument("--reason", default="superseded by owner run")
    p.set_defaults(fn=cmd_mark_temp)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
