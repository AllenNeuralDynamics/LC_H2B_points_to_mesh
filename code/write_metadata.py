"""Write AIND metadata (data_description.json + processing.json) for the LC mesh asset.

This runs in the separate ``/opt/meta-env`` (which has ``aind-data-schema``), NOT the
geometry environment that generates the meshes: aind-data-schema is a current release and
must not be pinned into the reproducibility-frozen geometry env. It describes the mesh set
this capsule produces (new_core_mesh.obj + percentile_10..90.obj) so the saved data asset
can be published to the AIND open-data bucket.

processing.json's ``Code`` block (capsule web URL + release version) is introspected from
the Code Ocean REST API at run time, following LC-NE_BARseq_MAT-RDS_conversion. That needs
the "Code Ocean API Credentials" secret attached to the capsule (Capsule Settings ->
Credentials), which exposes ``API_KEY``; ``CO_CAPSULE_ID`` and ``CO_COMPUTATION_ID`` are set
automatically during a run. Without those credentials, processing.json is skipped with a
warning (data_description.json is still written), rather than being written with guessed
provenance.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from aind_data_schema.components.identifiers import Code, DataAsset, Person
from aind_data_schema.core.data_description import DataDescription, DataLevel, Funding, License
from aind_data_schema.core.processing import DataProcess, Processing, ProcessStage
from aind_data_schema_models.modalities import Modality
from aind_data_schema_models.organizations import Organization
from aind_data_schema_models.process_names import ProcessName

UTC = timezone.utc

# --- the asset this capsule produces --------------------------------------------------
# TODO(name): confirm the name the saved data asset will carry. The metadata `name` must
# match the asset it describes. A stable descriptive name is used here (these LC meshes are
# not stored under the AIND <label>_YYYY-MM-DD_HH-MM-SS convention); align the saved asset's
# name with this, or set this to whatever name the asset is saved under.
ASSET_NAME = "LC_percentile_meshes"

# The single upstream input: the CCF-registered nuclear point calls. Referenced by name
# (internal CO asset, no public URL).
POINTS_ASSET = "LC_H2B_trailmap_probabilities_and_point_calls"

CODE_URL_FALLBACK = "https://github.com/AllenNeuralDynamics/LC_H2B_points_to_mesh"
CO_API_BASE = "https://codeocean.allenneuraldynamics.org/api/v1"
CO_WEB_BASE = "https://codeocean.allenneuraldynamics.org/capsule"

# Best-effort placeholders (mirror the LC-NE metadata conventions); confirm before publishing.
PROJECT_NAME = "Locus coeruleus norepinephrine neurons"
INVESTIGATORS = [Person(name="Drew Friedmann")]
FUNDING_SOURCE = [Funding(funder=Organization.AI)]  # Allen Institute placeholder
LICENSE = License.CC_BY_40

DATA_SUMMARY = (
    "Percentile-threshold 3D surface meshes delineating the locus coeruleus (LC) from "
    "pooled CCF-registered nuclear point calls across 8 SmartSPIM brains. Contents: "
    "new_core_mesh.obj (67th-percentile core) and percentile_10..90.obj (density shells). "
    "Generated deterministically from the raw point calls (load -> reflect -> crop -> kNN "
    "density -> shell/interior selection -> surfel reconstruction -> watertight -> smooth -> "
    "repair) with no manual steps; every parameter is in lc_mesh.config, so the set is fully "
    "reproducible from the point-calls asset."
)


def results_dir() -> Path:
    """Where the meshes were written, so the metadata sits beside them in the same asset.

    Mirrors reproduce_meshes.py: /root/capsule/results on Code Ocean, else the local
    results/reproduced directory.
    """
    cap = Path("/root/capsule/results")
    if Path("/root/capsule/data").exists():
        cap.mkdir(parents=True, exist_ok=True)
        return cap
    local = Path(__file__).resolve().parent.parent / "results" / "reproduced"
    local.mkdir(parents=True, exist_ok=True)
    return local


def fetch_co_provenance() -> tuple[str, str]:
    """Return (capsule_url, version) for the running Code Ocean capsule via the CO REST API.

    ``version`` is like ``"v3.0"`` for a released capsule, or
    ``"from non-release editable capsule"`` for an editable run. Requires API_KEY (from the
    attached "Code Ocean API Credentials" secret) plus the auto-set CO_CAPSULE_ID and
    CO_COMPUTATION_ID. Raises RuntimeError if credentials/env are missing or the API fails.
    """
    api_key = os.environ.get("API_KEY")
    capsule_id = os.environ.get("CO_CAPSULE_ID")
    computation_id = os.environ.get("CO_COMPUTATION_ID")
    if not api_key or not capsule_id or not computation_id:
        raise RuntimeError(
            "Missing Code Ocean env vars (API_KEY / CO_CAPSULE_ID / CO_COMPUTATION_ID). "
            "Attach the 'Code Ocean API Credentials' secret (Capsule Settings -> Credentials)."
        )

    auth = base64.b64encode(f"{api_key}:".encode()).decode()
    headers = {"Authorization": f"Basic {auth}"}

    def _get(path: str) -> dict:
        req = urllib.request.Request(f"{CO_API_BASE}{path}", headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    try:
        capsule = _get(f"/capsules/{capsule_id}")
        computation = _get(f"/computations/{computation_id}")
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Code Ocean API call failed: {e}") from e

    capsule_url = f"{CO_WEB_BASE}/{capsule['slug']}/tree"
    version = f"v{computation['version']}.0" if "version" in computation else "from non-release editable capsule"
    return capsule_url, version


def write_data_description(out: Path, creation_time: datetime) -> None:
    dd = DataDescription(
        name=ASSET_NAME,
        creation_time=creation_time,
        institution=Organization.AIND,
        funding_source=FUNDING_SOURCE,
        data_level=DataLevel.DERIVED,
        investigators=INVESTIGATORS,
        project_name=PROJECT_NAME,
        modalities=[Modality.SPIM],
        license=LICENSE,
        data_summary=DATA_SUMMARY,
        source_data=[POINTS_ASSET],
    )
    json.loads(dd.model_dump_json(by_alias=True))  # validate the object graph
    dd.write_standard_file(output_directory=str(out))
    print(f"  wrote data_description.json (name: {ASSET_NAME}, source: {POINTS_ASSET})")


def write_processing(out: Path, run_time: datetime) -> None:
    """Write processing.json, introspecting the capsule URL + version from Code Ocean.

    Skipped (with a warning) if the Code Ocean API credentials are not available, since the
    capsule URL and release version can't be populated reliably without them.
    """
    try:
        capsule_url, version = fetch_co_provenance()
    except RuntimeError as e:
        print(f"  WARNING: skipping processing.json -- {e}")
        return

    code = Code(
        url=capsule_url,
        name="LC_H2B_points_to_mesh",
        version=version,
        run_script=Path("code/run"),
        language="Python",
        input_data=[DataAsset(name=POINTS_ASSET)],
    )
    process = DataProcess(
        process_type=ProcessName.OTHER,
        name="LC density meshes from CCF point calls",
        stage=ProcessStage.ANALYSIS,
        code=code,
        experimenters=["Drew Friedmann"],
        start_date_time=run_time,
        end_date_time=run_time,
        notes="Core + nine percentile LC density meshes generated deterministically from the "
              "CCF-registered point calls; no manual steps (parameters in lc_mesh.config).",
    )
    processing = Processing(data_processes=[process])
    json.loads(processing.model_dump_json(by_alias=True))  # validate
    processing.write_standard_file(output_directory=str(out))
    print(f"  wrote processing.json (capsule {capsule_url}, {version})")


def main() -> None:
    out = results_dir()
    now = datetime.now(UTC)
    print(f"Writing mesh-asset metadata to {out}")
    write_data_description(out, now)
    write_processing(out, now)
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
