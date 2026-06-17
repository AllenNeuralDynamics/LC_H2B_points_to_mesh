# lc_mesh

Library extracted from `mesh_and_figure_generation.ipynb` for **reproducibly
generating the LC density meshes and the paper figures**. Parameters were recovered
from the original notebook's saved cell outputs.

## Layout

| File | Contents |
|---|---|
| `config.py` | All recovered parameters (core recipe fully known; percentile generation params known, repair steps not) |
| `points.py` | Point loading, midline reflection, LC crop, kNN density; point-table + published-mesh loaders |
| `normals.py` | PCA normal estimation + interior-guided orientation |
| `meshing.py` | `seal_holes`, surfel surface generation, repair (cavern strip, seal, broken/solitary-face cleanup) |
| `registration.py` | EMD-based self-registration of the 16 hemispheres (slow; optional) |
| `pipeline.py` | `make_core_mesh`, `make_percentile_mesh`, `make_self_registered_core_mesh` |
| `analysis.py` | point-in-mesh counting; `compare_meshes` (volume / surface-distance / topology) |
| `figures.py` | 1:1 reproductions of the paper figures + lightweight exploration helpers |

## The pieces

1. **Library** — this package (pure functions, no notebook state).
2. **Reproduction script** — [`../reproduce_core_mesh.py`](../reproduce_core_mesh.py):
   the entry point for regenerating the core mesh from the cell points and comparing
   it to the published mesh. Runs in the capsule environment; also self-contained
   via PEP 723 for standalone use (`uv run code/reproduce_core_mesh.py`).
3. **Notebook** — [`../explore_mesh.ipynb`](../explore_mesh.ipynb): thin; regenerates
   the paper figures from the canonical published meshes and lets you explore mesh
   generation. Only calls library functions.

## Dependencies & pinning (CodeOcean-native)

The canonical pinned environment lives in
[`../../environment/postInstall`](../../environment/postInstall) — the standard
CodeOcean way. It pins every package (plus `aind-zarr-utils`'s git commit) to a
**maximum-fidelity reconstruction** of the environment the published meshes were
generated in: the latest releases on/before the mesh-generation date
(**2026-02-02**), resolved for the image's **Python 3.10** (e.g. `open3d==0.18.0`,
`trimesh==4.11.1`, `point-cloud-utils==0.34.0`), with `--exclude-newer 2026-02-03`
bounding transitive deps to the same point in time. (Cross-check: this resolution
yields `zarr==2.18.3`, matching the capsule's committed base freeze.)
`reproduce_core_mesh.py`'s PEP-723 block mirrors the geometry-relevant subset so the
script also runs standalone via `uv`.

## Reproduction fidelity (core mesh)

`reproduce_core_mesh.py` prints the difference between the regenerated and published
core mesh on each run. Measured during local development:

| metric | result |
|---|---|
| vertices | 21219 (identical to published) |
| faces | 42434 vs 42436 (−2) |
| volume | 0.125318 vs 0.125421 mm³ (**0.082%**) |
| surface distance | mean 3.19 µm, median 2.99 µm, Hausdorff 12.4 µm |

This result is identical across library versions (reconstructed mesh-gen stack vs a
newer one), so the residual ~0.08% is **not** library-version drift. The vertex
count is identical and only the coordinates differ, at the floating-point level —
the residual comes from the surface reconstruction (surfel fitting + watertight
remeshing), not the recipe. These numbers were measured off the original platform;
run inside the capsule for the definitive on-platform comparison. Either way the core
mesh regenerates to within ~0.1% / a few microns — below any biologically meaningful
scale.

## Note on the percentile meshes

`make_percentile_mesh(df, thresh)` reproduces the generation parameters, but the
per-mesh repair steps (optional shrink / extra hole-seal) were applied interactively
by visual inspection and were never recorded. The published percentile `.obj` files
are therefore distributed as data artifacts. If the repair steps are recovered, pass
them via `repair_overrides={'shrink': ..., 'extra_seal_passes': ...}`.
