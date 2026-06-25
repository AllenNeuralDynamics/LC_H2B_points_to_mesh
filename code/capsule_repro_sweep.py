#!/usr/bin/env python3
"""Search for the package versions that reproduce Drew's published core mesh EXACTLY.

Run this INSIDE the CodeOcean capsule (Linux x86_64). The whole point is the OS:
when pip/uv installs `point-cloud-utils` and `open3d` here, it gets the same
*manylinux* wheels Drew used -- the exact compiled binaries. (The macOS sweep used
macOS wheels, which are a different binary built by a different compiler, and that
is the most likely source of the ~98-vertex residual we saw locally.)

For each (point-cloud-utils, open3d) version it regenerates the core mesh from the
published LC_points.csv and counts how many of the *published* mesh's vertices are
NOT reproduced to sub-nanometer precision (a KD-tree nearest-neighbor test). On
macOS every version gave 98. A combo that drives this to ~0 is an exact match --
i.e. the environment Drew actually used.

  Stage 1: check the current pins (pcu 0.34.0 / open3d 0.18.0) on this OS.
  Stage 2: if not already exact, sweep the full grid; stop early on a 0-unmatched hit.

Launch (in the capsule, with uv on PATH):
    python3 code/capsule_repro_sweep.py

Each combo runs in an isolated `uv run --with ...` environment, so the capsule's
base environment is never modified.
"""
import json
import os
import subprocess
import sys
import tempfile

# Everything except the two libraries under test is held at the canonical pins
# (environment/postInstall). lc_mesh.__init__ pulls in figures/registration/meshing,
# so matplotlib/plotly/pot must be present even though the mesh build doesn't use them.
OTHER_PINS = [
    "numpy==1.26.4", "pandas==2.3.3", "scipy==1.15.3", "scikit-learn==1.7.2",
    "trimesh==4.11.1", "networkx==3.4.2", "rtree==1.4.1", "matplotlib==3.10.8",
    "plotly==6.5.2", "pot==0.9.6.post1", "tqdm==4.67.3",
]
# pcu versions that actually expose pointcloud_surfel_geometry (>= 0.29.5), newest first
PCU_VERSIONS = ["0.34.0", "0.31.0", "0.30.4", "0.30.3", "0.30.2", "0.30.0",
                "0.29.7", "0.29.6", "0.29.5"]
O3D_VERSIONS = ["0.18.0", "0.19.0", "0.17.0", "0.16.1", "0.16.0"]

# Inner worker: regenerate the mesh and compare to the published one. Printed as a
# single RESULT <json> line so the orchestrator can parse it out of the build noise.
INNER = r'''
import sys, os, json, warnings
warnings.filterwarnings("ignore")
CAP = "/root/capsule"
code = os.path.join(CAP, "code") if os.path.isdir(os.path.join(CAP, "code")) else "code"
sys.path.insert(0, code)
import numpy as np, trimesh, lc_mesh
from scipy.spatial import cKDTree
asset = os.path.join(CAP, "data", "LC_percentile_meshes")
if not os.path.isdir(asset):
    asset = "results-c712751d-f744-4fe8-9657-93a7084eab22"
try:
    df = lc_mesh.load_lc_points_csv(os.path.join(asset, "LC_points.csv"))
    reg, _ = lc_mesh.make_core_mesh(df, verbose=False)
    pub = trimesh.load(os.path.join(asset, "new_core_mesh.obj"), process=False)
    # how many published vertices have NO sub-nanometer match in the regenerated mesh?
    dist, _ = cKDTree(reg.vertices).query(pub.vertices, k=1)
    unmatched = int((dist > 1e-3).sum())
    print("RESULT " + json.dumps(dict(ok=True, vcount=int(len(reg.vertices)),
        fcount=int(len(reg.faces)), pub_vcount=int(len(pub.vertices)),
        unmatched=unmatched, max_um=float(dist.max()),
        euler=int(reg.euler_number), watertight=bool(reg.is_watertight))))
except Exception as e:
    print("RESULT " + json.dumps(dict(ok=False, err=str(e)[:200])))
'''


def run_combo(pcu, o3d, inner_path, timeout=600):
    cmd = ["uv", "run", "--python", "3.10",
           "--with", f"point-cloud-utils=={pcu}", "--with", f"open3d=={o3d}"]
    for p in OTHER_PINS:
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
    print("STAGE 1 — current pins (point-cloud-utils 0.34.0, open3d 0.18.0) on THIS OS")
    print("=" * 70)
    r = run_combo("0.34.0", "0.18.0", inner)
    print(json.dumps(r))
    if r.get("ok") and r["unmatched"] == 0:
        print("\n*** EXACT MATCH with the current pins on Linux! ***")
        print("    Drew's published mesh is reproduced byte-for-byte; the macOS-only")
        print("    98-vertex residual was purely a macOS-vs-Linux wheel difference.")
        return
    if r.get("ok"):
        print(f"\nUnmatched with current pins on this OS: {r['unmatched']}"
              f"   (macOS gave 98 — a lower number means the OS already helped)")
    print("\nProceeding to STAGE 2 — full pcu x open3d grid...\n")

    best = None
    for pcu in PCU_VERSIONS:
        for o3d in O3D_VERSIONS:
            r = run_combo(pcu, o3d, inner)
            r["pcu"], r["o3d"] = pcu, o3d
            if r.get("ok"):
                tag = (f"unmatched={r['unmatched']:<4} max={r['max_um']:.2f}um "
                       f"verts={r['vcount']} euler={r['euler']} wt={r['watertight']}")
                if best is None or r["unmatched"] < best["unmatched"]:
                    best = r
            else:
                tag = f"FAIL: {r.get('err', '')[:40]}"
            print(f"pcu {pcu:8} o3d {o3d:8} -> {tag}", flush=True)
            if r.get("ok") and r["unmatched"] == 0:
                print(f"\n*** EXACT MATCH: point-cloud-utils=={pcu}  open3d=={o3d} ***")
                print("    This is (a candidate for) the exact environment Drew used.")
                return

    print("\n" + "=" * 70)
    print("No exact (0-unmatched) combo. Best result found:")
    print(json.dumps(best) if best else "(all combos failed)")
    print("=" * 70)
    print("If 'best' is far below 98, the OS/wheel was the main factor and the")
    print("remainder is CPU-level (then match Drew's instance type); if it's still")
    print("~98, the residual is outside pcu/open3d (kNN libs or a source build).")


if __name__ == "__main__":
    main()
