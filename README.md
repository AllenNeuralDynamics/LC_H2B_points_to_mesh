# Locus Coeruleus (LC) 3D Point Cloud Analysis and Mesh Generation

This capsule contains the notebook used to generate and visualize the LC density meshes shown in figures 1 and S1 of the paper.

**GitHub:** https://github.com/AllenNeuralDynamics/LC_H2B_points_to_mesh  
**Code Ocean:** https://codeocean.allenneuraldynamics.org/capsule/8736312/tree  
**Full collection:** https://codeocean.allenneuraldynamics.org/collections/9cf044ce-93c7-4c7e-bfa1-5d8c37aa42ec

## Note on reproducibility

This capsule is an **interactive notebook** and cannot be run reproducibly end to end. Two stages require manual judgement: the density-threshold selection (an interactive Plotly slider) and the mesh-repair steps, which are enabled selectively per mesh based on visual inspection (see **Usage**). Re-running the notebook therefore will not automatically reproduce the published meshes, and the canonical meshes published with the paper were not produced by this exact released version (the input point-call version that fed them may also differ).

It is shared for transparency: it documents the exact analysis, mesh-generation, and figure code behind the published LC meshes and figures, even though those assets cannot be regenerated automatically from it.

## Overview

This notebook performs spatial analysis of locus coeruleus (LC) neurons across 8 whole-brain light-sheet microscopy samples registered to the Allen CCF (Common Coordinate Framework). It constructs density-based 3D surface meshes at multiple percentile thresholds to delineate the LC boundary, performs inter-sample rigid registration, and generates publication-ready visualizations.

## Data

- **Input**: CCF-registered cell coordinates (`.npy`) from 8 SmartSPIM brain samples, identified by IDs: `798571`, `798573`, `798576`, `807322`, `807324`, `807325`, `807326`, `807327`
- **Raw imagery**: Zarr-formatted light-sheet volumes for overlay validation
- **Reference meshes**: Allen CCF brain structure meshes (`.obj`)
- **Output**: Percentile-based LC surface meshes (`.obj`), registered point data (`.csv`), figures (`.svg`, `.png`, `.html`)

### Input data assets

The data assets attached to the capsule in `.codeocean/datasets.json` are the
inputs this pipeline runs on. Only the four that are actually used are attached
(the original capsule had attached several additional raw-imagery volumes that the
notebook never referenced):

- **Nuclear point calls (primary input):** `LC_H2B_trailmap_probabilities_and_point_calls` — per-mouse CCF-registered point calls for all 8 samples; the meshes are built from these.
- **Pre-computed meshes:** `LC_percentile_meshes` — the canonical, distributed percentile/core meshes (plus `LC_points.csv`) consumed for the downstream counts, figures, and self-registration, and used as the comparison target when regenerating the core mesh.
- **Allen CCF reference:** `ccf_meshes` — CCF brain-structure reference mesh for the 3D figure backdrop.
- **One SmartSPIM whole-brain sample** (raw imagery, used only for the figure overlay): `SmartSPIM_807324_2025-08-25_11-34-40_stitched_2025-10-23_17-35-23`.

## Pipeline

### 1. Point Loading & Preprocessing
- Loads CCF-registered point clouds from all 8 samples
- Applies coordinate transforms (scaling to µm, axis flips)
- Reflects points across the midline (x = 5700 µm) to pool both hemispheres
- Crops to LC bounding box in CCF space

### 2. Local Density Mapping
- Computes k-nearest neighbor (k=100) mean distances per cell
- Converts to percentile ranks for density-based filtering
- Interactive threshold explorer (Plotly slider) to identify shell vs. core populations

### 3. Mesh Generation
- **Shell points**: Cells between the 10th and Nth kNN percentile (surface layer)
- **Interior points**: Cells below the 10th percentile (dense core, used for normal orientation)
- Normal estimation via local PCA (k=80 neighbors)
- Normal orientation using interior point guidance + iterative consistency smoothing
- Surface reconstruction via `point_cloud_utils.pointcloud_surfel_geometry`
- Watertight mesh conversion via `pcu.make_mesh_watertight`
- Laplacian smoothing (Open3D, 5 iterations)
- Meshes generated at percentile thresholds: 10, 20, 30, ..., 90

### 4. Mesh Repair
- Voxelization + distance transform to detect caverns/tunnels near the surface
- Removal of vertices with distance-to-surface threshold (irregular concavities)
- Hole sealing via centroid fan-triangulation
- Broken face removal and solitary edge cleanup
- Final watertight verification

### 5. ROI Counting
- Point-in-mesh containment testing (batched) for core mesh and all percentile meshes
- Per-hemisphere, per-sample cell counts at each percentile level

### 6. Self-Registration
- Selects reference hemisphere closest to mean voxel occupancy (L2 distance)
- Rigid registration (rotation + translation) of all 16 hemispheres to reference
- Objective: Earth Mover's Distance (EMD) via the POT library
- Optimization: L-BFGS-B with angular and translational bounds
- Stores registered coordinates (`reg_x`, `reg_y`, `reg_z`) and per-point registration error

### 7. Post-Self-Registration Mesh Reconstruction
- Recomputes kNN density on registered coordinates
- Rebuilds core mesh from self-registered point cloud
- Same repair pipeline as step 4

### 8. Visualization & Figure Generation
- 3D interactive Plotly figures (meshes, points, brain outline)
- 2D minimum-projection heatmaps of kNN density (matplotlib)
- Coronal slice plots with per-sample PC2 histograms overlaid
- Max-intensity projections from raw Zarr volumes with point overlays
- All figures saved as `.html`, `.svg`, or `.png`

## Dependencies

| Package | Purpose |
|---|---|
| `numpy`, `scipy`, `pandas` | Core computation |
| `scikit-learn` | kNN, PCA, KMeans |
| `trimesh` | Mesh operations, repair, voxelization |
| `open3d` | Mesh smoothing |
| `point_cloud_utils` | Surface reconstruction, watertight conversion |
| `plotly` | Interactive 3D visualization |
| `matplotlib` | 2D figures |
| `POT` (`ot`) | Optimal transport / EMD for registration |
| `zarr`, `tifffile` | Raw image I/O |
| `SimpleITK` | Image processing utilities |
| `aind_zarr_utils`, `aind_registration_utils` | AIND pipeline transforms |

## Usage

Run cells sequentially in a Jupyter environment. The notebook includes interactive widgets (sliders) for threshold selection. Mesh repair cells are semi-manual — uncomment specific repair steps as needed based on visual inspection of each mesh.

Set `thresh = None` to operate on the core mesh (67th percentile), or set `thresh` to a specific integer (10–90) to operate on a percentile mesh.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
