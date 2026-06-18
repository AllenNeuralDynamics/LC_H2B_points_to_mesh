# /// script
# requires-python = "==3.10.*"
# dependencies = [
#   "numpy==1.26.4",
#   "pandas==2.3.3",
#   "scipy==1.15.3",
#   "scikit-learn==1.7.2",
#   "trimesh==4.11.1",
#   "networkx==3.4.2",
#   "rtree==1.4.1",
#   "open3d==0.18.0",
#   "point-cloud-utils==0.34.0",
#   "matplotlib==3.10.8",
#   "plotly==6.5.2",
#   "pot==0.9.6.post1",
#   "tqdm==4.67.3",
# ]
# ///
"""Fit per-percentile isosurface-extraction parameters to the published meshes.

The paper's Methods state the percentile meshes were generated "using dynamic
radius and resolution parameters to compensate for point density and mesh size
across the displayed percentiles." Those per-percentile values were never recorded.
This script recovers them: for each percentile (10..90) it searches the surfel
radius and watertight resolution (and, only if needed for watertightness, a minimal
repair) that bring the regenerated mesh as close as possible to the published one.

The objective is the symmetric mean surface distance to the published mesh, subject
to the mesh being watertight. The shell/interior selection is fixed at the principled
recipe (4 < kNN_percentile < threshold; interior < 4) — only the documented
radius/resolution (and minimal repair) are fit.

Outputs (to results/percentile_fit/):
  - percentile_<t>.obj            best-fit mesh per percentile
  - fit_report.json               per-mesh best params + achieved metrics (written
                                  incrementally, so an interrupted run keeps progress)
  - percentile_params.py.txt      a config-ready PERCENTILE_PARAMS dict to paste into
                                  lc_mesh/config.py

Run (overnight; ~a few hours for all 9 on the pinned env):
    uv run code/fit_percentile_meshes.py
    uv run code/fit_percentile_meshes.py --only 90 --quick   # fast single-mesh test
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
import open3d as o3d  # noqa: E402
import point_cloud_utils as pcu  # noqa: E402

import lc_mesh  # noqa: E402
from lc_mesh import config  # noqa: E402
from lc_mesh.normals import estimate_normals, orient_complex_shape_normals  # noqa: E402
from lc_mesh.meshing import repair_mesh  # noqa: E402
from lc_mesh.analysis import compare_meshes  # noqa: E402

# --- input/output resolution (CodeOcean mounts, else local asset) ---
_CAP = Path("/root/capsule/data")
MESH_ASSET = (_CAP / "LC_percentile_meshes" if _CAP.exists()
              else REPO / "results-c712751d-f744-4fe8-9657-93a7084eab22")
RAW_DIR = (_CAP / "LC_H2B_trailmap_probabilities_and_point_calls"
           / "segmentation_and_quantification")
OUT = (Path("/root/capsule/results/percentile_fit") if _CAP.exists()
       else REPO / "results" / "percentile_fit")

PERCENTILES = list(range(10, 100, 10))

# --- search grids (multipliers relative to the notebook's cell-20 formula) ---
RADIUS_MULTIPLIERS = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.7, 2.0]
RESOLUTION_MULTIPLIERS = [0.5, 1.0, 2.0]
# repair candidates: None = pure isosurface (preferred); others add minimal cleanup
REPAIR_OPTIONS = [
    None,
    dict(pitch=3, max_distance=5, keep_distance=4, shrink=None, extra_seal_passes=0),
    dict(pitch=3, max_distance=5, keep_distance=3, shrink=None, extra_seal_passes=1),
    dict(pitch=3, max_distance=5, keep_distance=2, shrink=None, extra_seal_passes=1),
]
SEARCH_NSAMPLE = 15000   # fast surface-distance during search
FINAL_NSAMPLE = 50000    # precise metrics for the winner
MAX_RESOLUTION = 120000  # cap to keep watertighting tractable


def score_against(mesh, published, n_sample):
    """Symmetric mean surface distance (µm); lower = closer. None if not watertight."""
    if not mesh.is_watertight or len(mesh.vertices) == 0:
        return None, None
    cmp = compare_meshes(mesh, published, n_sample=n_sample)
    return cmp["mean_surface_dist_um"], cmp


def fit_one(thresh, df, published, quick=False):
    p = config.percentile_params(thresh)
    base_radius = p["surfel_radius"]
    base_res = p["watertight_resolution"]
    shell, interior = lc_mesh.select_shell_and_interior(
        df, p["shell_lo"], p["shell_hi"], p["interior_hi"])
    print(f"[p{thresh}] shell={len(shell)} interior={len(interior)} "
          f"(formula radius={base_radius:.1f}, resolution={base_res})", flush=True)

    # normals: computed ONCE (the expensive step), reused across radius/resolution
    normals = estimate_normals(shell)
    oriented = orient_complex_shape_normals(shell, normals.copy(), interior_points=interior)
    oriented = np.asfortranarray(oriented, dtype=np.float32)
    shell32 = np.asarray(shell, dtype=np.float32)

    rad_mults = [1.0, 1.5] if quick else RADIUS_MULTIPLIERS
    res_mults = [1.0] if quick else RESOLUTION_MULTIPLIERS
    repair_opts = [None, REPAIR_OPTIONS[1]] if quick else REPAIR_OPTIONS

    best = {"score": np.inf}
    n_eval = 0
    for rm in rad_mults:
        radius = float(base_radius * rm)
        v, f = pcu.pointcloud_surfel_geometry(shell32, oriented, radius)
        for resm in res_mults:
            resolution = int(min(base_res * resm, MAX_RESOLUTION))
            try:
                raw = build_surface_from_vf(v, f, resolution)
            except Exception as e:  # noqa: BLE001
                print(f"  radius={radius:.1f} res={resolution} build failed: {e}", flush=True)
                continue
            for rep in repair_opts:
                try:
                    mesh = raw if rep is None else repair_mesh(raw, **rep, verbose=False)
                except Exception:  # noqa: BLE001
                    continue
                s, cmp = score_against(mesh, published, SEARCH_NSAMPLE)
                n_eval += 1
                if s is None:
                    continue
                if s < best["score"]:
                    best = {"score": s, "radius": radius, "resolution": resolution,
                            "radius_mult": rm, "resolution_mult": resm, "repair": rep,
                            "mesh": mesh, "cmp": cmp}
                    print(f"  * better: radius={radius:.1f} res={resolution} "
                          f"repair={_rep_name(rep)} -> meanΔ={s:.3f}µm "
                          f"vol%={cmp['volume_pct_diff']:.2f} V={cmp['regenerated']['n_vertices']}",
                          flush=True)
    if not np.isfinite(best["score"]):
        print(f"[p{thresh}] NO watertight candidate found", flush=True)
        return None
    # precise metrics for the winner
    _, final_cmp = score_against(best["mesh"], published, FINAL_NSAMPLE)
    print(f"[p{thresh}] BEST radius={best['radius']:.1f} res={best['resolution']} "
          f"repair={_rep_name(best['repair'])} | meanΔ={final_cmp['mean_surface_dist_um']:.3f}µm "
          f"Hausdorff={final_cmp['hausdorff_um']:.2f} vol%={final_cmp['volume_pct_diff']:.2f} "
          f"(evaluated {n_eval} candidates)", flush=True)
    return best, final_cmp


def build_surface_from_vf(v, f, resolution, smooth=5):
    vw, fw = pcu.make_mesh_watertight(v, f, resolution)
    mesh = trimesh.Trimesh(vertices=vw, faces=fw)
    tries = 0
    while not mesh.is_watertight and tries < 3:
        vw, fw = pcu.make_mesh_watertight(vw, fw, resolution)
        mesh = trimesh.Trimesh(vertices=vw, faces=fw)
        tries += 1
    mo = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                   o3d.utility.Vector3iVector(mesh.faces))
    mo.compute_vertex_normals()
    mo = mo.filter_smooth_simple(number_of_iterations=smooth)
    return trimesh.Trimesh(np.asarray(mo.vertices), np.asarray(mo.triangles))


def _rep_name(rep):
    if rep is None:
        return "none"
    return f"keep{rep['keep_distance']}_seal{rep['extra_seal_passes']}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(MESH_ASSET / "LC_points.csv"),
                    help="point table with kNN_percentile (defines the shells to match)")
    ap.add_argument("--from-raw", default=None, help="build points from raw .npy data_root instead")
    ap.add_argument("--only", type=int, default=None, help="fit a single percentile (e.g. 90)")
    ap.add_argument("--quick", action="store_true", help="tiny grid for a smoke test")
    ap.add_argument("--force", action="store_true", help="refit percentiles already in the report")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    report_path = OUT / "fit_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    if args.from_raw:
        df = lc_mesh.build_lc_points(args.from_raw)
    else:
        df = lc_mesh.load_lc_points_csv(args.csv)
    published, _core = lc_mesh.load_published_meshes(MESH_ASSET)

    todo = [args.only] if args.only else PERCENTILES
    for thresh in todo:
        if str(thresh) in report and not args.force:
            print(f"[p{thresh}] already in report; skipping (use --force to refit)", flush=True)
            continue
        t0 = time.time()
        out = fit_one(thresh, df, published[thresh], quick=args.quick)
        if out is None:
            report[str(thresh)] = {"status": "no_watertight_candidate"}
            report_path.write_text(json.dumps(report, indent=2))
            continue
        best, final_cmp = out
        best["mesh"].export(OUT / f"percentile_{thresh}.obj")
        report[str(thresh)] = {
            "surfel_radius": round(best["radius"], 3),
            "watertight_resolution": best["resolution"],
            "radius_mult_vs_formula": best["radius_mult"],
            "resolution_mult_vs_formula": best["resolution_mult"],
            "repair": best["repair"],
            "shell_lo": 4, "shell_hi": thresh, "interior_hi": 4,
            "metrics": {k: v for k, v in final_cmp.items() if not isinstance(v, dict)},
            "n_vertices": int(len(best["mesh"].vertices)),
            "published_n_vertices": int(len(published[thresh].vertices)),
            "fit_seconds": round(time.time() - t0, 1),
        }
        report_path.write_text(json.dumps(report, indent=2))  # incremental save
        print(f"[p{thresh}] saved mesh + report ({report[str(thresh)]['fit_seconds']}s)\n", flush=True)

    _emit_config(report, OUT / "percentile_params.py.txt")
    print(f"\nDone. Report: {report_path}\nConfig snippet: {OUT/'percentile_params.py.txt'}", flush=True)
    # summary
    print("\nthresh  radius   resolution  repair        meanΔµm  vol%   V(fit/pub)")
    for t in PERCENTILES:
        r = report.get(str(t))
        if not r or "metrics" not in r:
            continue
        m = r["metrics"]
        print(f"  {t:<5} {r['surfel_radius']:<8} {r['watertight_resolution']:<11} "
              f"{_rep_name(r['repair']):<13} {m['mean_surface_dist_um']:.2f}    "
              f"{m['volume_pct_diff']:.2f}   {r['n_vertices']}/{r['published_n_vertices']}")


def _emit_config(report, path):
    lines = ["# Recovered per-percentile isosurface-extraction parameters",
             "# (fit to the published meshes; paste into lc_mesh/config.py).",
             "PERCENTILE_PARAMS = {"]
    for t in PERCENTILES:
        r = report.get(str(t))
        if not r or "surfel_radius" not in r:
            continue
        lines.append(f"    {t}: dict(shell_lo=4, shell_hi={t}, interior_hi=4, normals_k=80, "
                     f"surfel_radius={r['surfel_radius']}, watertight_resolution={r['watertight_resolution']}, "
                     f"smooth_iterations=5, repair={r['repair']}),")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
