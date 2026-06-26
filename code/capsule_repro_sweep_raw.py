#!/usr/bin/env python3
"""Independent (from-RAW) reproduction search: can we rebuild Drew's published core
mesh from the raw point calls, with NO dependence on the LC_points.csv intermediate?

Run INSIDE the CodeOcean capsule (Linux x86_64), where the raw point calls are
mounted at data/LC_H2B_trailmap_probabilities_and_point_calls/. The CSV is used
ONLY as ground truth to measure divergence -- it is never fed into the pipeline.

The from-raw path has TWO places it can diverge from Drew's run:
  (A) the kNN-density step: NearestNeighbors (k=100) + rankdata -> kNN_percentile,
      which drives the shell/interior selection. Borderline points whose density
      sits at a percentile threshold (10 or 67) flip in/out -> a different point set
      (locally this gave 21209 selected verts vs Drew's 21219). Levers: scikit-learn,
      scipy, numpy.
  (B) the geometry step (surfel + watertight + smooth): the ~98-vertex residual the
      pcu/open3d sweep is chasing separately.

A byte-exact from-raw reproduction needs BOTH (A) and (B) to hit zero. This script
attacks (A): it diagnoses the kNN/selection divergence against Drew's CSV, then
sweeps scikit-learn x scipy to try to reproduce his exact point selection.

  Stage A: detailed diagnostic with the current pins (point-set match, kNN-percentile
           diffs, selection flips, and the final from-raw mesh vs published).
  Stage B: sweep scikit-learn x scipy; report selection flips + vertex-count delta +
           geometry unmatched per combo; stop early on a full (delta=0, unmatched=0) hit.

Launch (in the capsule, with uv on PATH):
    python3 code/capsule_repro_sweep_raw.py
"""
import json
import os
import subprocess
import tempfile

# Held fixed (geometry libs at current pins; numpy fixed so the compiled wheels'
# ABI stays valid). lc_mesh.__init__ also needs matplotlib/plotly/pot present.
BASE_PINS = [
    "numpy==1.26.4", "pandas==2.3.3", "point-cloud-utils==0.34.0", "open3d==0.18.0",
    "trimesh==4.11.1", "networkx==3.4.2", "rtree==1.4.1", "matplotlib==3.10.8",
    "plotly==6.5.2", "pot==0.9.6.post1", "tqdm==4.67.3",
]
# kNN-determining libraries to sweep (newest first), all compatible with numpy 1.26 / py3.10
SKLEARN_VERSIONS = ["1.7.2", "1.6.1", "1.5.2", "1.4.2", "1.3.2"]
SCIPY_VERSIONS = ["1.15.3", "1.13.1", "1.11.4"]

INNER = r'''
import sys, os, json, warnings
warnings.filterwarnings("ignore")
CAP = "/root/capsule"
code = os.path.join(CAP, "code") if os.path.isdir(os.path.join(CAP, "code")) else "code"
sys.path.insert(0, code)
import numpy as np, trimesh, lc_mesh
from lc_mesh import config
from scipy.spatial import cKDTree
asset = os.path.join(CAP, "data", "LC_percentile_meshes")
if not os.path.isdir(asset):
    asset = "results-c712751d-f744-4fe8-9657-93a7084eab22"
raw = os.path.join(CAP, "data", "LC_H2B_trailmap_probabilities_and_point_calls",
                   "segmentation_and_quantification")

def in_selection(p):  # shell (shell_lo<p<shell_hi) OR interior (p<interior_hi)
    c = config.CORE
    return (p < c["interior_hi"]) | ((p > c["shell_lo"]) & (p < c["shell_hi"]))

try:
    ours = lc_mesh.build_lc_points(raw)                       # recompute kNN from RAW
    drew = lc_mesh.load_lc_points_csv(os.path.join(asset, "LC_points.csv"))  # ground truth
    oc = ours[["x", "y", "z"]].values
    dc = drew[["x", "y", "z"]].values
    # align our raw-derived points to Drew's points by nearest coordinate
    dist, idx = cKDTree(dc).query(oc, k=1)
    matched = int((dist < 1e-3).sum())
    op = ours["kNN_percentile"].values
    dp = drew["kNN_percentile"].values[idx]
    perc_diff = int((np.abs(op - dp) > 1e-6).sum())
    sel_flips = int((in_selection(op) != in_selection(dp)).sum())
    our_sel = int(in_selection(op).sum())
    drew_sel = int(in_selection(dp).sum())
    # build the mesh from RAW and compare to the published mesh
    reg, _ = lc_mesh.make_core_mesh(ours, verbose=False)
    pub = trimesh.load(os.path.join(asset, "new_core_mesh.obj"), process=False)
    d2, _ = cKDTree(reg.vertices).query(pub.vertices, k=1)
    unmatched = int((d2 > 1e-3).sum())
    print("RESULT " + json.dumps(dict(ok=True,
        n_ours=int(len(ours)), n_drew=int(len(drew)), pts_matched=matched,
        perc_diff=perc_diff, sel_flips=sel_flips, our_sel=our_sel, drew_sel=drew_sel,
        vcount=int(len(reg.vertices)), pub_vcount=int(len(pub.vertices)),
        vcount_delta=int(len(reg.vertices) - len(pub.vertices)), unmatched=unmatched)))
except Exception as e:
    print("RESULT " + json.dumps(dict(ok=False, err=str(e)[:200])))
'''


def run(sklearn_v, scipy_v, inner_path, timeout=900):
    cmd = ["uv", "run", "--python", "3.10",
           "--with", f"scikit-learn=={sklearn_v}", "--with", f"scipy=={scipy_v}"]
    for p in BASE_PINS:
        cmd += ["--with", p]
    cmd += ["python", inner_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return dict(ok=False, err="timeout")
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT"):
            return json.loads(line[len("RESULT "):])
    return dict(ok=False, err="no result (install/build failed)")


def main():
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(INNER)
        inner = f.name

    print("=" * 70)
    print("STAGE A — from-RAW diagnostic with current pins (scikit-learn 1.7.2, scipy 1.15.3)")
    print("=" * 70)
    r = run("1.7.2", "1.15.3", inner)
    print(json.dumps(r, indent=2))
    if r.get("ok"):
        print(f"\n  raw points: {r['n_ours']} (Drew: {r['n_drew']}, matched {r['pts_matched']})")
        print(f"  kNN_percentile differs on {r['perc_diff']} points")
        print(f"  selection flips: {r['sel_flips']}  (ours {r['our_sel']} vs Drew {r['drew_sel']} selected)")
        print(f"  from-raw mesh: {r['vcount']} verts (Δ {r['vcount_delta']:+d} vs published), "
              f"geometry unmatched: {r['unmatched']}")
        if r["vcount_delta"] == 0 and r["unmatched"] == 0:
            print("\n*** FULL from-raw byte-match already! ***")
            return
    print("\nProceeding to STAGE B — scikit-learn x scipy sweep...\n")

    best = None
    for skl in SKLEARN_VERSIONS:
        for sp in SCIPY_VERSIONS:
            r = run(skl, sp, inner)
            r["sklearn"], r["scipy"] = skl, sp
            if r.get("ok"):
                tag = (f"sel_flips={r['sel_flips']:<4} vΔ={r['vcount_delta']:<+5} "
                       f"unmatched={r['unmatched']}")
                # rank by selection flips first, then geometry residual
                keyf = (r["sel_flips"], r["unmatched"])
                if best is None or keyf < (best["sel_flips"], best["unmatched"]):
                    best = r
            else:
                tag = f"FAIL: {r.get('err', '')[:40]}"
            print(f"sklearn {skl:7} scipy {sp:8} -> {tag}", flush=True)
            if r.get("ok") and r["vcount_delta"] == 0 and r["unmatched"] == 0:
                print(f"\n*** FULL from-raw byte-match: scikit-learn=={skl} scipy=={sp} ***")
                return

    print("\n" + "=" * 70)
    print("Best from-raw combo (fewest selection flips, then fewest geometry-unmatched):")
    print(json.dumps(best) if best else "(all combos failed)")
    print("=" * 70)
    print("sel_flips -> 0 means the kNN/selection (A) is reproduced from raw.")
    print("unmatched is the geometry residual (B), handled by the pcu/open3d sweep;")
    print("a byte-exact from-raw mesh needs BOTH to reach 0.")


if __name__ == "__main__":
    main()
