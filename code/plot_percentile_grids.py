# /// script
# requires-python = "==3.10.*"
# dependencies = [
#   "numpy==1.26.4", "pandas==2.3.3", "scipy==1.15.3", "scikit-learn==1.7.2",
#   "trimesh==4.11.1", "networkx==3.4.2", "rtree==1.4.1", "open3d==0.18.0",
#   "point-cloud-utils==0.34.0", "matplotlib==3.10.8", "plotly==6.5.2",
#   "pot==0.9.6.post1", "tqdm==4.67.3",
# ]
# ///
"""Survey figures comparing the published vs fitted percentile meshes.

Produces one PNG per projection (XY, XZ, YZ); each is a 3-row x 9-column grid:
row 1 published, row 2 fitted (from results/percentile_fit/), row 3 the pointwise
difference (fitted vertices colored by distance to the published surface, shared
color scale across percentiles). Lets you eyeball where each fit succeeds or fails.

    uv run code/plot_percentile_grids.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
import lc_mesh  # noqa: E402
from lc_mesh.analysis import nearest_surface_distances  # noqa: E402

_CAP = Path("/root/capsule/data")
MESH_ASSET = (_CAP / "LC_percentile_meshes" if _CAP.exists()
              else REPO / "results-c712751d-f744-4fe8-9657-93a7084eab22")
FIT_DIR = (Path("/root/capsule/results/percentile_fit") if _CAP.exists()
           else REPO / "results" / "percentile_fit")
PERCENTILES = list(range(10, 100, 10))
PROJECTIONS = [(0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ")]


def main():
    published, _ = lc_mesh.load_published_meshes(MESH_ASSET)
    fitted = {}
    for t in PERCENTILES:
        p = FIT_DIR / f"percentile_{t}.obj"
        if p.exists():
            fitted[t] = trimesh.load(p, process=False)
        else:
            print(f"  (no fitted mesh for p{t}; skipping)")

    # per-vertex distance fitted->published surface, computed once (3D, projection-free)
    diffs = {}
    for t in fitted:
        diffs[t] = nearest_surface_distances(fitted[t].vertices, published[t])
        print(f"p{t}: mean Δ {diffs[t].mean():.2f} µm, max {diffs[t].max():.2f} µm")
    vmax = float(np.percentile(np.concatenate(list(diffs.values())), 98))
    print(f"shared color scale vmax = {vmax:.2f} µm")

    for proj in PROJECTIONS:
        out = FIT_DIR / f"grid_{proj[2]}.png"
        lc_mesh.figures.plot_percentile_grid(published, fitted, diffs, proj,
                                             PERCENTILES, vmax=vmax, save_path=str(out))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
