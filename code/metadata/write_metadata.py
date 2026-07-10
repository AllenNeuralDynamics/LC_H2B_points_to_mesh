"""Write AIND metadata (data_description.json + processing.json) for the LC mesh asset.

Runs in the separate ``/opt/meta-env`` (which has ``aind-data-schema``), not the geometry
environment that generates the meshes, so aind-data-schema stays a current release rather than
being pinned into the reproducibility-frozen geometry env. Describes the mesh set this capsule
produces (core_mesh.obj + percentile_10..90.obj) so the saved asset can be published to the
AIND open-data bucket.

data_description.json's project-owned fields (project_name, funding_source, investigators) are
read from the committed snapshot ``project_funding.json``. That snapshot is fetched from the AIND
metadata service (the authoritative, intake-form-owned source) offline with
``python code/metadata/write_metadata.py --refresh-snapshot``, run from the AIND network. The capsule itself
never calls the internal metadata service (it can't reach it); refresh the snapshot whenever the
project's funding changes.

processing.json's ``Code`` block (capsule web URL + release version) is introspected from the
Code Ocean REST API at run time. That needs the "Code Ocean API Credentials" secret attached to
the capsule (Capsule Settings -> Credentials), which exposes ``API_KEY``; ``CO_CAPSULE_ID`` and
``CO_COMPUTATION_ID`` are set automatically during a run. Without those credentials processing.json
is skipped with a warning (data_description.json is still written) rather than written with
incomplete provenance.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.parse
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
# The saved data asset's name. code/run stamps it with the UTC run date/time (AIND convention),
# e.g. "LC_percentile_meshes_2026-07-09_14-30-05", and exports it so produce_meshes.py, the
# notebook, and this script all agree on the results/<ASSET_NAME>/ subfolder and metadata name.
# The plain fallback below is only used for standalone runs that don't go through code/run.
ASSET_NAME = os.environ.get("ASSET_NAME", "LC_percentile_meshes")

# The single upstream input: the CCF-registered nuclear point calls (referenced by name).
POINTS_ASSET = "LC_H2B_trailmap_probabilities_and_point_calls"

# Provenance sources for processing.json's Code block.
CODE_URL_FALLBACK = "https://github.com/AllenNeuralDynamics/LC_H2B_points_to_mesh"
CO_API_BASE = "https://codeocean.allenneuraldynamics.org/api/v1"
CO_WEB_BASE = "https://codeocean.allenneuraldynamics.org/capsule"

# The AIND project this asset belongs to. project_name, funding_source, and investigators are
# owned by the project and read from the committed snapshot below (edited via the AIND intake
# form, refreshed with --refresh-snapshot). Institution, modality, data level, and license are
# intrinsic to this derived SPIM asset published to the AIND open-data bucket.
PROJECT_NAME = "Discovery Neuromodulation - Subproject 2 Molecular Anatomy Cell Types"
SNAPSHOT_PATH = Path(__file__).resolve().parent / "project_funding.json"
LICENSE = License.CC_BY_40

# Metadata-service base, used ONLY by --refresh-snapshot (run offline from the AIND network).
# The capsule never calls it at run time.
METADATA_SERVICE = "http://aind-metadata-service/api/v2"

DATA_SUMMARY = (
    "Percentile-threshold 3D surface meshes delineating the locus coeruleus (LC) from "
    "pooled CCF-registered nuclear point calls across 8 SmartSPIM brains. Contents: "
    "core_mesh.obj (67th-percentile core) and percentile_10..90.obj (density shells). "
    "Generated deterministically from the raw point calls (load -> reflect -> crop -> kNN "
    "density -> shell/interior selection -> surfel reconstruction -> watertight -> smooth -> "
    "repair) with no manual steps; every parameter is in lc_mesh.config, so the set is fully "
    "reproducible from the point-calls asset."
)


def asset_dir() -> Path:
    """The asset subfolder results/<ASSET_NAME>/, where produce_meshes.py wrote the meshes;
    the metadata is written here too so the whole subfolder is one standalone data asset.

    Mirrors produce_meshes.py: results root is /root/capsule/results on Code Ocean, else
    the local results/ directory.
    """
    root = (Path("/root/capsule/results") if Path("/root/capsule/data").exists()
            else Path(__file__).resolve().parent.parent.parent / "results")
    out = root / ASSET_NAME
    out.mkdir(parents=True, exist_ok=True)
    return out


def _fetch_project_from_service() -> dict:
    """Fetch PROJECT_NAME's funding_source + investigators from the AIND metadata service.

    Returns {"project_name", "funding_source", "investigators"} with investigators de-duplicated
    by name (the service can list a name more than once). Raises on any network/HTTP/parse error.
    """
    enc = urllib.parse.quote(PROJECT_NAME)

    def _get(endpoint: str):
        with urllib.request.urlopen(f"{METADATA_SERVICE}/{endpoint}/{enc}", timeout=10) as resp:
            return json.loads(resp.read())

    funding = _get("funding")
    seen, investigators = set(), []
    for person in _get("investigators"):
        if person.get("name") not in seen:
            seen.add(person.get("name"))
            investigators.append(person)
    return {"project_name": PROJECT_NAME, "funding_source": funding, "investigators": investigators}


def project_metadata() -> dict:
    """PROJECT_NAME's funding_source + investigators, read from the committed snapshot
    ``project_funding.json``. The capsule does not call the internal metadata service; the
    snapshot is refreshed offline from the AIND network with ``--refresh-snapshot``."""
    data = json.loads(SNAPSHOT_PATH.read_text())
    print(f"  project metadata from {SNAPSHOT_PATH.name}: "
          f"{len(data['funding_source'])} funding, {len(data['investigators'])} investigator(s)")
    return data


# Env vars that processing.json's provenance introspection needs: API_KEY comes from the attached
# "Code Ocean API Credentials" secret; CO_CAPSULE_ID/CO_COMPUTATION_ID are set by the run.
# data_description.json does NOT need these.
CO_API_ENV = ("API_KEY", "CO_CAPSULE_ID", "CO_COMPUTATION_ID")


def fetch_co_provenance() -> tuple[str, str]:
    """Return (capsule_url, version) for the running Code Ocean capsule via the CO REST API.

    ``version`` is like ``"v3.0"`` for a released capsule, or
    ``"from non-release editable capsule"`` for an editable run. Requires API_KEY (from the
    attached "Code Ocean API Credentials" secret) plus the auto-set CO_CAPSULE_ID and
    CO_COMPUTATION_ID. Raises RuntimeError if credentials/env are missing or the API fails.
    """
    missing = [v for v in CO_API_ENV if not os.environ.get(v)]
    if missing:
        raise RuntimeError(
            f"Missing Code Ocean env vars ({' / '.join(missing)}). "
            "Attach the 'Code Ocean API Credentials' secret (Capsule Settings -> Credentials)."
        )
    api_key = os.environ["API_KEY"]
    capsule_id = os.environ["CO_CAPSULE_ID"]
    computation_id = os.environ["CO_COMPUTATION_ID"]

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
    project = project_metadata()
    dd = DataDescription(
        name=ASSET_NAME,
        creation_time=creation_time,
        institution=Organization.AIND,
        funding_source=[Funding.model_validate(f) for f in project["funding_source"]],
        data_level=DataLevel.DERIVED,
        investigators=[Person.model_validate(p) for p in project["investigators"]],
        project_name=project["project_name"],
        modalities=[Modality.SPIM],
        license=LICENSE,
        data_summary=DATA_SUMMARY,
        source_data=[POINTS_ASSET],
    )
    json.loads(dd.model_dump_json(by_alias=True))  # validate the object graph
    dd.write_standard_file(output_directory=str(out))
    investigators = ", ".join(p["name"] for p in project["investigators"])
    print(f"  wrote data_description.json (name: {ASSET_NAME}, project: {project['project_name']}, "
          f"investigators: {investigators})")


def write_processing(out: Path, run_time: datetime) -> None:
    """Write processing.json, introspecting the capsule URL + version from Code Ocean.

    Skipped (with a warning) if the Code Ocean API credentials are not available, since the
    capsule URL and release version can't be populated reliably without them.

    Drew Friedmann is recorded as the experimenter (who performed this work); the project's
    investigators live in data_description.json and are a separate set of people.
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
        process_type=ProcessName.ANALYSIS,
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


def refresh_snapshot() -> None:
    """Re-fetch the project's funding/investigators from the metadata service and rewrite the
    committed snapshot. Run from the AIND network whenever the project's funding changes."""
    data = _fetch_project_from_service()
    SNAPSHOT_PATH.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Wrote {SNAPSHOT_PATH} ({len(data['funding_source'])} funding, "
          f"{len(data['investigators'])} investigator(s))")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-snapshot", action="store_true",
        help="re-fetch project_funding.json from the metadata service and exit (run from the "
             "AIND network when the project's funding changes)",
    )
    args = parser.parse_args()
    if args.refresh_snapshot:
        refresh_snapshot()
        return

    out = asset_dir()
    now = datetime.now(UTC)
    print(f"Writing mesh-asset metadata to {out}")
    write_data_description(out, now)
    write_processing(out, now)
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
