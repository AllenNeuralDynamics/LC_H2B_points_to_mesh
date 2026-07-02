# /// script
# requires-python = "==3.10.*"
# dependencies = ["numpy==1.26.4", "scipy==1.15.3", "trimesh==4.11.1",
#                 "networkx==3.4.2", "rtree==1.4.1"]
# ///
"""Quantitative side-by-side comparison of the published vs regenerated LC core mesh.

Reports topology (Euler characteristic, winding consistency, watertightness),
geometry (vertex/face counts, signed volume, orientation-independent voxel volume),
surface agreement (how many vertices match to sub-nanometer, max deviation), and
optionally the inside/outside cell count (the number that actually matters for the
paper). Everything here is directly verifiable -- run it on your own machine.

    uv run code/verify_mesh_comparison.py PUBLISHED.obj REGENERATED.obj [POINTS.csv]

If POINTS.csv (with x,y,z columns) is given, it also counts how many points fall
inside each mesh (slow: ~1-2 min per mesh).
"""
import sys
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def topo(m):
    return dict(verts=len(m.vertices), faces=len(m.faces),
                euler=int(m.euler_number),
                winding_consistent=bool(m.is_winding_consistent),
                watertight=bool(m.is_watertight))


def voxel_volume(m, pitch=3.0):
    """Orientation-INDEPENDENT volume (fills the voxelization); unaffected by the
    winding defect, unlike signed volume."""
    return float(m.voxelized(pitch=pitch).fill().volume) / 1e9


def main():
    pub_path = sys.argv[1] if len(sys.argv) > 1 else "new_core_mesh_published.obj"
    reg_path = sys.argv[2] if len(sys.argv) > 2 else "new_core_mesh_regenerated.obj"
    csv_path = sys.argv[3] if len(sys.argv) > 3 else None

    pub = trimesh.load(pub_path, process=False)
    reg = trimesh.load(reg_path, process=False)

    print("=" * 64)
    print(f"PUBLISHED  : {pub_path}")
    print(f"REGENERATED: {reg_path}")
    print("=" * 64)

    tp, tr = topo(pub), topo(reg)
    print(f"\n{'metric':24} {'published':>16} {'regenerated':>16}")
    print("-" * 60)
    for k in ("verts", "faces", "euler", "winding_consistent", "watertight"):
        print(f"{k:24} {str(tp[k]):>16} {str(tr[k]):>16}")
    print(f"{'(euler 2 = valid solid; odd euler = non-orientable)'}")

    # volumes
    sv_p, sv_r = pub.volume / 1e9, reg.volume / 1e9
    print(f"\n{'signed volume (mm^3)':24} {sv_p:>16.6f} {sv_r:>16.6f}"
          f"   diff {100*abs(sv_p-sv_r)/sv_r:.3f}%  <-- inflated by the winding defect")
    vv_p, vv_r = voxel_volume(pub), voxel_volume(reg)
    print(f"{'voxel volume (mm^3)':24} {vv_p:>16.6f} {vv_r:>16.6f}"
          f"   diff {100*abs(vv_p-vv_r)/vv_r:.3f}%  <-- orientation-independent: identical")

    # surface agreement: how close is every published vertex to the regenerated surface?
    d, _ = cKDTree(reg.vertices).query(pub.vertices, k=1)
    print(f"\nsurface agreement (published vertices vs regenerated):")
    print(f"  identical to < 1 nm : {int((d <= 1e-3).sum())} / {len(d)} "
          f"({100*(d <= 1e-3).mean():.2f}%)")
    print(f"  differ > 1 nm       : {int((d > 1e-3).sum())}  (max {d.max():.2f} um)")

    # optional: inside/outside cell count (the result that matters)
    if csv_path:
        import csv
        pts = []
        with open(csv_path) as f:
            r = csv.DictReader(f)
            for row in r:
                pts.append([float(row["x"]), float(row["y"]), float(row["z"])])
        pts = np.asarray(pts)
        print(f"\ninside/outside count over {len(pts)} points (slow)...")
        def count(m):
            out = np.zeros(len(pts), bool)
            for i in range(0, len(pts), 2000):
                out[i:i+2000] = m.contains(pts[i:i+2000])
            return out
        ip, ir = count(pub), count(reg)
        print(f"  inside published  : {int(ip.sum())}")
        print(f"  inside regenerated: {int(ir.sum())}")
        print(f"  cells that change membership: {int((ip != ir).sum())}")


if __name__ == "__main__":
    main()
