# lc_mesh

Library for **reproducibly generating the LC core mesh and the nine percentile meshes**
from the raw cell point calls. All parameters live in `config.py`; the mesh functions
are pure (no notebook state).

## Layout

| File | Contents |
|---|---|
| `config.py` | All parameters: point preprocessing, kNN, the core recipe (`CORE`/`REPAIR`), and the full per-percentile recipe (`PERCENTILE_PARAMS`, including each percentile's repair settings) |
| `points.py` | Point loading, midline reflection, LC crop, kNN density, shell/interior selection |
| `normals.py` | PCA normal estimation + interior-guided orientation |
| `meshing.py` | `seal_holes`, surfel surface generation, repair (cavern strip, seal, broken/solitary-face cleanup) |
| `pipeline.py` | `make_core_mesh`, `make_percentile_mesh` (select -> generate -> repair) |
| `analysis.py` | Point-in-mesh counting; `mesh_stats` (vertex/face counts, volume, watertightness, Euler number) |

## Entry point

[`../reproduce_meshes.py`](../reproduce_meshes.py) is the entry point: it builds the LC
points from the raw point calls and generates all ten meshes, writing the `.obj` files
and a JSON report to `results/`. It runs in the capsule environment and is also
self-contained via PEP 723 for standalone use (`uv run code/reproduce_meshes.py`).

## Parameters

The core-mesh recipe is `config.CORE` + `config.REPAIR`. Each percentile mesh has its
own entry in `config.PERCENTILE_PARAMS` giving its generation parameters (shell/interior
thresholds, surfel radius, watertight resolution, smoothing) and its repair settings; a
`repair` value of `None` means no extra repair pass is applied. `make_percentile_mesh`
reads these directly, so every mesh is generated deterministically from documented values.

## Dependencies & pinning (CodeOcean-native)

The canonical pinned environment lives in
[`../../environment/postInstall`](../../environment/postInstall), the standard CodeOcean
way. It pins the geometry stack for the image's **Python 3.10** (`open3d==0.18.0`,
`trimesh==4.11.1`, `point-cloud-utils==0.34.0`, and the rest), with `--exclude-newer`
bounding transitive deps to a fixed point in time. `reproduce_meshes.py`'s PEP 723 block
mirrors the geometry-relevant subset so the script also runs standalone via `uv`.
