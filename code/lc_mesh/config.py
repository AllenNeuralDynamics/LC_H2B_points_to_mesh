"""Pipeline parameters for generating the LC core mesh and the nine percentile meshes.

Every parameter needed to build all ten meshes is specified here: the core-mesh recipe
(CORE / REPAIR) and the full per-percentile recipe (PERCENTILE_PARAMS), including each
percentile's own repair settings. The meshes are generated deterministically from these
values and the raw point calls; no previously generated mesh is required.
"""

# --- Point preprocessing (notebook cells 2, 7) ---
MIDLINE_X = 5700           # reflection plane (microns), cell 7
SCALE_TO_UM = 1000.0       # raw .npy coords are multiplied by 1000, cell 2
# LC crop box in reflected CCF space (cell 7)
LC_CROP = dict(y_min=9000, y_max=12000, x_min=3000, z_min=3000,
               exclude_z_gt=4800, exclude_y_gt=10800)
# Sample load ORDER is load-bearing: the surfel reconstruction is order-sensitive, so
# the per-sample .npy files must be concatenated in a fixed order. reflect_and_crop and
# compute_knn_density both preserve row order, so this list pins the point order
# end-to-end. The order below is the order used to define the recipe (a sorted order
# would change ~5000 mesh vertices); load_ccf_points sorts the glob within each keyword
# so filesystem order cannot perturb it.
SAMPLE_KEYWORDS = ['807324', '807322', '798576', '807326',
                   '798571', '807325', '798573', '807327']

# --- kNN density (cell 9) ---
KNN_K = 100                # neighbours (excluding self) averaged for local density

# --- Core mesh (cells 13, 19) ---
CORE = dict(
    shell_lo=10,           # shell = 10 < kNN_percentile < 67
    shell_hi=67,           # the interactive slider value, left at its default
    interior_hi=10,        # interior (normal-orientation guide) = kNN_percentile < 10
    normals_k=80,          # passed to estimate_normals (note: see normals.py)
    surfel_radius=30,
    watertight_resolution=10000,
    smooth_iterations=5,
)

# --- Core mesh repair (cells 25, 27, 29, 31) ---
REPAIR = dict(
    pitch=3,               # voxel pitch, cell 25
    max_distance=5,        # surface distance-transform horizon (voxels), cell 25
    keep_distance=2,       # drop vertices with distance-to-surface > 2, cell 27
    shrink=None,           # optional shrink_mesh_along_normals distance (cell 32); off for core
    extra_seal_passes=0,   # optional extra hole-seal pass (cell 33); off for core
)

# --- Percentile meshes (thresholds 10..90) ---
# Full per-percentile recipe: generation parameters plus that percentile's own repair
# settings. `repair=None` means no extra repair pass is applied (the watertight surfel
# mesh is used as-is). Each mesh is generated deterministically from these values.
PERCENTILE_PARAMS = {
    10: dict(shell_lo=4, shell_hi=10, interior_hi=4, normals_k=80, surfel_radius=30.0, watertight_resolution=10000, smooth_iterations=5, repair={'pitch': 3, 'max_distance': 5, 'keep_distance': 3, 'shrink': None, 'extra_seal_passes': 1}),
    20: dict(shell_lo=4, shell_hi=20, interior_hi=4, normals_k=80, surfel_radius=32.5, watertight_resolution=7500, smooth_iterations=5, repair={'pitch': 3, 'max_distance': 5, 'keep_distance': 3, 'shrink': None, 'extra_seal_passes': 1}),
    30: dict(shell_lo=4, shell_hi=30, interior_hi=4, normals_k=80, surfel_radius=35.0, watertight_resolution=10000, smooth_iterations=5, repair=None),
    40: dict(shell_lo=4, shell_hi=40, interior_hi=4, normals_k=80, surfel_radius=37.5, watertight_resolution=12500, smooth_iterations=5, repair={'pitch': 3, 'max_distance': 5, 'keep_distance': 2, 'shrink': None, 'extra_seal_passes': 1}),
    50: dict(shell_lo=4, shell_hi=50, interior_hi=4, normals_k=80, surfel_radius=40.0, watertight_resolution=15000, smooth_iterations=5, repair={'pitch': 3, 'max_distance': 5, 'keep_distance': 2, 'shrink': None, 'extra_seal_passes': 1}),
    60: dict(shell_lo=4, shell_hi=60, interior_hi=4, normals_k=80, surfel_radius=42.5, watertight_resolution=17500, smooth_iterations=5, repair={'pitch': 3, 'max_distance': 5, 'keep_distance': 4, 'shrink': None, 'extra_seal_passes': 0}),
    70: dict(shell_lo=4, shell_hi=70, interior_hi=4, normals_k=80, surfel_radius=51.75, watertight_resolution=40000, smooth_iterations=5, repair={'pitch': 3, 'max_distance': 5, 'keep_distance': 2, 'shrink': None, 'extra_seal_passes': 1}),
    80: dict(shell_lo=4, shell_hi=80, interior_hi=4, normals_k=80, surfel_radius=47.5, watertight_resolution=22500, smooth_iterations=5, repair={'pitch': 3, 'max_distance': 5, 'keep_distance': 3, 'shrink': None, 'extra_seal_passes': 1}),
    90: dict(shell_lo=4, shell_hi=90, interior_hi=4, normals_k=80, surfel_radius=50.0, watertight_resolution=50000, smooth_iterations=5, repair={'pitch': 3, 'max_distance': 5, 'keep_distance': 4, 'shrink': None, 'extra_seal_passes': 0}),
}

PERCENTILE_THRESHOLDS = sorted(PERCENTILE_PARAMS)


def percentile_params(thresh):
    """Full recipe (generation params + `repair`) for one percentile mesh."""
    return dict(PERCENTILE_PARAMS[thresh])
