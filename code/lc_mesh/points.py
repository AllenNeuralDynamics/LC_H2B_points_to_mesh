"""Point loading, hemisphere reflection, LC cropping, and kNN-density mapping.

Faithful extraction of notebook cells 2, 7, 9. Two entry points:

- `build_lc_points()` runs the full pipeline from the raw per-sample `.npy`
  files (needs the CodeOcean data mount).
- `load_lc_points_csv()` loads the published `LC_points.csv`, which already
  carries `kNN_percentile` — letting you reproduce meshes locally without the
  raw data or recomputing kNN.
"""
import glob
import os

import numpy as np
import pandas as pd
import trimesh
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors

from . import config


def load_ccf_points(data_root, keywords=None):
    """Load CCF-registered points from per-sample `.npy` files (cell 2).

    Applies the original coordinate transform: scale to microns, flip z and x.
    """
    keywords = keywords or config.SAMPLE_KEYWORDS
    ccf_files = []
    for kw in keywords:
        pattern = os.path.join(data_root, f'*{kw}*', '*ccf*.npy')
        ccf_files.extend(glob.glob(pattern, recursive=False))
    if not ccf_files:
        raise FileNotFoundError(f"No matching ccf .npy files found under {data_root}")

    rows = []
    for fpath in ccf_files:
        fname = os.path.basename(fpath)
        pts = np.load(fpath).copy()
        pts *= config.SCALE_TO_UM
        pts[:, 2] *= -1
        pts[:, 0] *= -1
        for row in pts:
            rows.append({'file': fname, 'x': row[0], 'y': row[1], 'z': row[2]})
    return pd.DataFrame(rows)


def reflect_and_crop(df_all_points):
    """Reflect across the midline and crop to the LC bounding box (cell 7)."""
    df = df_all_points.copy()
    border = config.MIDLINE_X
    reflected_mask = df['x'] > border
    df.loc[reflected_mask, 'x'] = 2 * border - df.loc[reflected_mask, 'x']
    df['reflected'] = reflected_mask.astype(int)

    c = config.LC_CROP
    keep = (
        (df['y'] > c['y_min']) & (df['y'] < c['y_max']) &
        (df['x'] > c['x_min']) & (df['z'] > c['z_min']) &
        ~((df['z'] > c['exclude_z_gt']) & (df['y'] > c['exclude_y_gt']))
    )
    return df[keep].copy()


def compute_knn_density(df, coord_cols=('x', 'y', 'z'), k=None):
    """Add `kNN` (mean distance to k nearest neighbours) and `kNN_percentile`
    columns (cell 9). Operates in-place on a copy and returns it."""
    k = k or config.KNN_K
    df = df.copy()
    coords = df[list(coord_cols)].values
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    df['kNN'] = distances[:, 1:k + 1].mean(axis=1)          # exclude self
    df['kNN_percentile'] = rankdata(df['kNN'], method='average') / len(df) * 100
    return df


def build_lc_points(data_root, keywords=None):
    """Full pipeline: load -> reflect+crop -> kNN density. Returns LC_only_points."""
    df = load_ccf_points(data_root, keywords)
    df = reflect_and_crop(df)
    return compute_knn_density(df)


def load_lc_points_csv(csv_path):
    """Load the published `LC_points.csv` (already carries `kNN_percentile`)."""
    return pd.read_csv(csv_path)


def load_published_meshes(mesh_dir):
    """Load the canonical published meshes from the LC_percentile_meshes asset
    (cell 36/39): returns (percentile_meshes dict keyed by int threshold,
    core_mesh). These are the distributed meshes used in the paper's figures."""
    percentile_meshes = {}
    for fname in os.listdir(mesh_dir):
        if fname.startswith('percentile') and fname.endswith('.obj'):
            key = int(os.path.splitext(fname)[0].split('_')[1])
            percentile_meshes[key] = trimesh.load(os.path.join(mesh_dir, fname))
    core_mesh = trimesh.load(os.path.join(mesh_dir, 'new_core_mesh.obj'))
    return percentile_meshes, core_mesh


def select_shell_and_interior(df, shell_lo, shell_hi, interior_hi,
                              coord_cols=('x', 'y', 'z')):
    """Select shell points (shell_lo < kNN_percentile < shell_hi) and interior
    points (kNN_percentile < interior_hi). Returns (shell, interior) float32 arrays.
    Mirrors notebook cells 13 / 20."""
    p = df['kNN_percentile']
    shell = df.loc[(p > shell_lo) & (p < shell_hi), list(coord_cols)].values.astype(np.float32)
    interior = df.loc[p < interior_hi, list(coord_cols)].values.astype(np.float32)
    return shell, interior
