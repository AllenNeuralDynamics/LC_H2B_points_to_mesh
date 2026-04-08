This capsule contains a notebook used to generate and visualize the meshes described in figure 1 and S1.

https://github.com/AllenNeuralDynamics/LC_H2B_points_to_mesh.git

# Locus Coeruleus (LC) 3D Point Cloud Analysis and Mesh Generation

## Overview

This notebook performs spatial analysis of locus coeruleus (LC) neurons across 8 whole-brain light-sheet microscopy samples registered to the Allen CCF (Common Coordinate Framework). It constructs density-based 3D surface meshes at multiple percentile thresholds to delineate the LC boundary, performs inter-sample rigid registration, and generates publication-ready visualizations.

## Data

- **Input**: CCF-registered cell coordinates (`.npy`) from 8 SmartSPIM brain samples, identified by IDs: `798571`, `798573`, `798576`, `807322`, `807324`, `807325`, `807326`, `807327`
- **Raw imagery**: Zarr-formatted light-sheet volumes for overlay validation
- **Reference meshes**: Allen CCF brain structure meshes (`.obj`)
- **Output**: Percentile-based LC surface meshes (`.obj`), registered point data (`.csv`), figures (`.svg`, `.png`, `.html`)

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