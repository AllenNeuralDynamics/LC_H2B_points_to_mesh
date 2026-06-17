"""Pipeline parameters that produced the published LC meshes.

Recovered from the original notebook's saved cell outputs. The core-mesh recipe is
fully specified here. For the percentile meshes the *generation* parameters are
known, but the per-mesh interactive repair steps were performed by hand and never
recorded, so they are not encoded here (see the PERCENTILE note below).
"""

# --- Point preprocessing (notebook cells 2, 7) ---
MIDLINE_X = 5700           # reflection plane (microns), cell 7
SCALE_TO_UM = 1000.0       # raw .npy coords are multiplied by 1000, cell 2
# LC crop box in reflected CCF space (cell 7)
LC_CROP = dict(y_min=9000, y_max=12000, x_min=3000, z_min=3000,
               exclude_z_gt=4800, exclude_y_gt=10800)
SAMPLE_KEYWORDS = ['798571', '798573', '798576', '807322',
                   '807324', '807325', '807326', '807327']

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

# --- Mesh repair (cells 25, 27, 29, 31) ---
REPAIR = dict(
    pitch=3,               # voxel pitch, cell 25
    max_distance=5,        # surface distance-transform horizon (voxels), cell 25
    keep_distance=2,       # drop vertices with distance-to-surface > 2, cell 27
    shrink=None,           # optional shrink_mesh_along_normals distance (cell 32) — OFF for core
    extra_seal_passes=0,   # optional extra hole-seal pass (cell 33) — OFF for core
)


def percentile_params(thresh):
    """Generation parameters for a percentile mesh (notebook cell 20).

    NOTE: the per-mesh interactive *repair* steps for the percentile meshes were
    performed by hand and never recorded; only the generation parameters below are
    known. Repair defaults to the core recipe, which will NOT in general reproduce
    the published percentile meshes.
    """
    radius = 30 + (thresh - 10) * (50 - 30) / (90 - 10)
    watertight_resolution = int(10000 + (thresh - 10) * (50000 - 10000) / (90 - 10))
    return dict(
        shell_lo=4,                 # shell = 4 < kNN_percentile < thresh
        shell_hi=thresh,
        interior_hi=4,              # interior = kNN_percentile < 4
        normals_k=80,
        surfel_radius=radius,
        watertight_resolution=watertight_resolution,
        smooth_iterations=5,
    )


PERCENTILE_THRESHOLDS = list(range(10, 100, 10))
