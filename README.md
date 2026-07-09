# Locus Coeruleus (LC) 3D Mesh Generation

This capsule reproducibly generates the locus coeruleus (LC) density meshes shown in figures 1 and S1 of the paper: one core mesh plus nine percentile meshes (10th through 90th), built directly from the raw cell point calls.

**GitHub:** https://github.com/AllenNeuralDynamics/LC_H2B_points_to_mesh
**Code Ocean:** https://codeocean.allenneuraldynamics.org/capsule/8736312/tree
**Full collection:** https://codeocean.allenneuraldynamics.org/collections/9cf044ce-93c7-4c7e-bfa1-5d8c37aa42ec

## Reproducibility

All ten meshes are generated end to end from the raw point calls, with no manual steps and no reference to any previously generated mesh. The Reproducible Run (`code/run`) executes `code/reproduce_meshes.py`, which:

1. loads the CCF-registered point calls for all 8 samples,
2. reflects them across the midline and crops to the LC,
3. maps local kNN density, then
4. for the core mesh and each percentile threshold, selects the shell/interior populations, reconstructs the surface, and repairs it into a watertight solid.

Every parameter (point ordering, density k, per-mesh shell/interior thresholds, surfel radius, watertight resolution, smoothing, and per-mesh repair settings) is documented in [`code/lc_mesh/config.py`](code/lc_mesh/config.py). See [Outputs](#outputs) for what the run produces and how it is laid out.

## Outputs

The Reproducible Run writes two things under `results/`:

- **`results/LC_percentile_meshes/`** is the mesh **data asset**: `new_core_mesh.obj`, `percentile_{10..90}.obj`, `reproduction_report.json` (per-mesh vertex/face counts, volume, watertightness, Euler number), and the AIND metadata `data_description.json` + `processing.json`. This subfolder is self-contained, so it can be saved directly as a standalone Code Ocean data asset. The folder name is `ASSET_NAME` (default `LC_percentile_meshes`, set in `code/run`) and is shared by the mesh, metadata, and figure steps so they agree.
- **`results/figures/`** holds the paper figures (kNN heatmaps, 3D mesh HTMLs, per-sample counts, coronal-slice SVGs) plus the executed notebook. These are run outputs, not part of the mesh asset. Included is a csv with every coordinate of every cell, each tagged with kNN density value, kNN percentile, and flags for inclusion in each percentile mesh as well as the core mesh. Interactive 3D visualizations are saved as html files. Figures in the manuscript are extracted from these visualizations.

The metadata is written by [`code/write_metadata.py`](code/write_metadata.py). `processing.json`'s `Code` block (capsule web URL + release version) is introspected from the Code Ocean REST API at run time, which requires the **"Code Ocean API Credentials"** secret to be attached (Capsule Settings -> Credentials). Without it, `processing.json` is skipped with a warning and only `data_description.json` is written.

## Input data

The capsule attaches three datasets (see [`.codeocean/datasets.json`](.codeocean/datasets.json)):

- **`LC_H2B_trailmap_probabilities_and_point_calls`**: per-mouse CCF-registered nuclear point calls for all 8 SmartSPIM samples (IDs `798571`, `798573`, `798576`, `807322`, `807324`, `807325`, `807326`, `807327`). The meshes are built entirely from these.
- **`ccf_meshes`**: Allen CCF brain-structure reference mesh, used only as the brain-outline backdrop in the 3D figures.
- **`SmartSPIM_807324_...`**: one raw stitched light-sheet volume, used only for the raw-image max-projection overlay figure.

No previously generated LC mesh is attached; the meshes are produced from the point calls.

## Pipeline

1. **Point loading & preprocessing**: load CCF-registered `.npy` point clouds, apply the coordinate transform (scale to µm, axis flips), reflect across the midline (x = 5700 µm) to pool both hemispheres, and crop to the LC bounding box.
2. **Local density mapping**: mean distance to the k = 100 nearest neighbours per cell, converted to percentile ranks.
3. **Mesh generation** (per mesh): select shell points (between a low and a high kNN percentile) and interior points (below a low percentile, used to orient normals); estimate normals via local PCA; reconstruct the surface with `point_cloud_utils.pointcloud_surfel_geometry`; convert to watertight with `pcu.make_mesh_watertight`; Laplacian-smooth (Open3D).
4. **Mesh repair**: voxelize + distance-transform to detect near-surface caverns, drop deep internal vertices, seal holes by centroid fan-triangulation, clean broken/solitary faces, and verify watertightness. The core mesh and each percentile mesh use their own repair settings (due largely to density differences at the varying percentile cutoffs) from `config.py`.

## Library layout (`code/lc_mesh/`)

| File | Contents |
|---|---|
| `config.py` | All mesh parameters: point preprocessing, kNN, the core recipe, and the full per-percentile recipe (`PERCENTILE_PARAMS`) |
| `points.py` | Point loading, midline reflection, LC crop, kNN density, shell/interior selection |
| `normals.py` | PCA normal estimation + interior-guided orientation |
| `meshing.py` | Surfel surface generation and repair (hole seal, cavern strip, broken/solitary-face cleanup) |
| `pipeline.py` | `make_core_mesh` and `make_percentile_mesh` (select -> generate -> repair) |
| `analysis.py` | Point-in-mesh counting and basic mesh descriptors (`mesh_stats`) |
| `figures.py` | The paper figures (kNN heatmaps, 3D mesh renderings, per-sample counts, coronal slices, raw-image overlay) + exploration helpers |

## Figures

[`code/explore_mesh.ipynb`](code/explore_mesh.ipynb) is a thin notebook that reproduces the paper figures **from the meshes this capsule generates**, and provides interactive controls to explore the mesh-generation parameters. Run `code/reproduce_meshes.py` first so the meshes exist in `results/<asset>/`, then run the notebook; it writes figures to `results/figures/` and all real logic lives in `lc_mesh.figures`.

## Environment

The mesh-generation environment is pinned as a lockfile: [`environment/requirements.in`](environment/requirements.in) lists the top-level (directly imported) packages, and [`environment/requirements.txt`](environment/requirements.txt) is the fully-resolved, version-pinned set compiled from it (`uv pip compile ... --python-platform linux`). Every package is pinned to an exact version, and those pins (not any dependency-resolution date bound) are what make the meshes reproducible: the generated meshes are identical to well below a nanometer regardless of which transitive versions resolve. [`environment/Dockerfile`](environment/Dockerfile) copies the lock into the image and [`environment/postInstall`](environment/postInstall) installs from it (Python 3.10). To change a dependency, edit `requirements.in` and recompile `requirements.txt`.

Core packages: `numpy`, `scipy`, `pandas`, `scikit-learn` (kNN/PCA), `trimesh` (mesh ops, repair, voxelization), `open3d` (smoothing), `point_cloud_utils` (surface reconstruction, watertight conversion), `matplotlib`/`plotly`/`ipywidgets` (figures + notebook), `zarr`/`tifffile` (raw-image overlay).

Metadata generation uses a **separate** environment: `aind-data-schema` is a current release and must not be pinned into the reproducibility-frozen geometry env, so `postInstall` provisions it into `/opt/meta-env` from [`environment/metadata-requirements.txt`](environment/metadata-requirements.txt), and `code/write_metadata.py` runs there.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
