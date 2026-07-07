"""lc_mesh: library for reproducibly generating the LC core mesh and the nine
percentile meshes from the raw cell point calls.

Pure functions for: loading/preprocessing the CCF-registered points, kNN density
mapping, surfel surface reconstruction + repair, point-in-mesh counting, and the
paper figures. All mesh parameters live in `config.py`; see `code/lc_mesh/README.md`.
"""
from . import config, figures
from .points import (
    load_ccf_points, reflect_and_crop, compute_knn_density,
    build_lc_points, select_shell_and_interior,
)
from .normals import (
    estimate_normals, orient_normals_with_interior, orient_complex_shape_normals,
)
from .meshing import (
    seal_holes, shrink_mesh_along_normals, generate_surface_mesh, repair_mesh,
)
from .analysis import (
    count_points_in_mesh, count_points_in_meshes, mesh_stats,
)
from .pipeline import make_core_mesh, make_percentile_mesh

__all__ = [
    "config", "figures",
    "load_ccf_points", "reflect_and_crop", "compute_knn_density",
    "build_lc_points", "select_shell_and_interior",
    "estimate_normals", "orient_normals_with_interior", "orient_complex_shape_normals",
    "seal_holes", "shrink_mesh_along_normals", "generate_surface_mesh", "repair_mesh",
    "count_points_in_mesh", "count_points_in_meshes", "mesh_stats",
    "make_core_mesh", "make_percentile_mesh",
]
