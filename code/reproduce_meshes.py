"""Generate the LC core mesh and the nine percentile meshes from the raw cell point calls.

This is the entry point for the capsule's Reproducible Run. It runs the full pipeline
with no manual steps and no reference to any previously generated mesh: load the
CCF-registered point calls -> reflect across the midline -> crop to the LC -> map local
kNN density -> then, for the core mesh and each of the nine percentile thresholds
(10..90), select the shell/interior populations, reconstruct the surface, and repair it
into a watertight solid. Every parameter lives in `lc_mesh.config`.

It writes ten `.obj` files (`new_core_mesh.obj` and `percentile_{10..90}.obj`) and a
`reproduction_report.json` recording each mesh's own statistics (vertex/face counts,
volume, watertightness, Euler number).

Usage (on CodeOcean the raw dataset is mounted and the default resolves automatically):
    python code/reproduce_meshes.py
    python code/reproduce_meshes.py --from-raw DIR --out DIR
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))  # make lc_mesh importable

import lc_mesh  # noqa: E402

# The only input is the raw point-call dataset. On CodeOcean it is mounted under
# /root/capsule/data; locally, pass --from-raw.
_CAP = Path("/root/capsule/data")
RAW_DIR = (_CAP / "LC_H2B_trailmap_probabilities_and_point_calls"
           / "segmentation_and_quantification")
_OUT = (Path("/root/capsule/results") if _CAP.exists()
        else REPO / "results" / "reproduced")


def _build_and_save(name, mesh, out, report):
    path = out / f"{name}.obj"
    mesh.export(path)
    report["meshes"][name] = lc_mesh.mesh_stats(mesh)
    print(f"  {name} -> {path.name}: {len(mesh.vertices)} verts, "
          f"{len(mesh.faces)} faces, watertight={mesh.is_watertight}, "
          f"euler={mesh.euler_number}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-raw", default=None,
                    help="data_root of the raw .npy point calls "
                         "(default: the mounted trailmap segmentation_and_quantification dir)")
    ap.add_argument("--out", default=str(_OUT), help="output directory")
    args = ap.parse_args()

    raw_dir = Path(args.from_raw) if args.from_raw else RAW_DIR
    if not raw_dir.exists():
        sys.exit(f"Raw point calls not found at {raw_dir}. Mount the "
                 f"LC_H2B_trailmap_probabilities_and_point_calls dataset, or pass --from-raw.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. build LC points from the raw upstream point calls
    print(f"Building LC points from raw .npy under {raw_dir} ...")
    df = lc_mesh.build_lc_points(str(raw_dir))
    print(f"  {len(df)} LC points")

    report = {"n_lc_points": int(len(df)), "meshes": {}}
    t0 = time.time()

    # 2. core mesh
    print("Generating core mesh ...")
    core, _ = lc_mesh.make_core_mesh(df, verbose=True)
    _build_and_save("new_core_mesh", core, out, report)

    # 3. the nine percentile meshes
    for thresh in lc_mesh.config.PERCENTILE_THRESHOLDS:
        print(f"Generating percentile mesh {thresh} ...")
        mesh, _ = lc_mesh.make_percentile_mesh(df, thresh, verbose=True)
        _build_and_save(f"percentile_{thresh}", mesh, out, report)

    report["total_seconds"] = round(time.time() - t0, 1)
    report_path = out / "reproduction_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    n = len(report["meshes"])
    print(f"\nGenerated {n} meshes in {report['total_seconds']}s. "
          f"Wrote meshes + report -> {out}")


if __name__ == "__main__":
    main()
