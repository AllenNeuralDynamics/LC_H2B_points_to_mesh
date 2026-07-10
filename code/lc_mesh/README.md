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
| `figures.py` | Paper figures (kNN heatmaps, 3D renderings, per-sample counts, coronal slices, raw-image overlay) + exploration helpers |

## Entry points

- [`../produce_meshes.py`](../produce_meshes.py) builds the LC points from the raw
  point calls and generates all ten meshes, writing the `.obj` files and a JSON report to
  `results/`. Run as `python code/produce_meshes.py` in the capsule environment.
- [`../explore_mesh.ipynb`](../explore_mesh.ipynb) is a thin notebook that reproduces the
  paper figures from those generated meshes and lets you explore the parameters. It only
  calls `lc_mesh` functions.

## Parameters

The core-mesh recipe is `config.CORE` + `config.REPAIR`. Each percentile mesh has its
own entry in `config.PERCENTILE_PARAMS` giving its generation parameters (shell/interior
thresholds, surfel radius, watertight resolution, smoothing) and its repair settings; a
`repair` value of `None` means no extra repair pass is applied. `make_percentile_mesh`
reads these directly, so every mesh is generated deterministically from documented values.

## Dependencies & pinning

Top-level (directly imported) packages are listed in
[`../../environment/requirements.in`](../../environment/requirements.in); the fully
resolved, version-pinned lock is compiled from it into
[`../../environment/requirements.txt`](../../environment/requirements.txt) with
`uv pip compile ... --python-version 3.10 --python-platform linux --exclude-newer 2026-02-03`.
The [`Dockerfile`](../../environment/Dockerfile) copies the lock into the image and
[`postInstall`](../../environment/postInstall) installs from it. To change a dependency,
edit `requirements.in` and recompile the lock.
