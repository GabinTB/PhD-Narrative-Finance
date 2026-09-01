"""One-off script to register manually-curated canonical narrative taxonomy versions as datalake artifacts."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from datalake import DatalakeIndex

DATALAKE_ROOT = os.environ["DATALAKE_ROOT"]
SOURCE = Path(DATALAKE_ROOT) / "derived" / "canonical_narratives"
VERSIONS = ["v1", "v2", "v3"]
PIPELINE = "PhD-Narrative-Finance"
PIPELINE_VERSION = "v0.1.0"


with DatalakeIndex(DATALAKE_ROOT) as dl:
    for version in VERSIONS:
        src = SOURCE / version
        if not src.exists():
            print(f"skip {version}: not found at {src}")
            continue

        with dl.run(
            kind="canonical_narratives",
            pipeline=PIPELINE,
            pipeline_version=PIPELINE_VERSION,
            hyperparams={"taxonomy_version": version},
            notes=(
                f"Hand-curated taxonomy {version}. No automated pipeline; "
                f"files copied from derived/canonical_narratives/{version}."
            ),
            layer="derived",
        ) as run:
            n = 0
            for f in sorted(src.iterdir()):
                if f.is_file():
                    shutil.copy2(f, run.out_dir / f.name)
                    n += 1

        artifact = dl.get(run.artifact_id)
        print(f"registered {artifact.artifact_id} ({len(artifact.file_hashes)} files)")