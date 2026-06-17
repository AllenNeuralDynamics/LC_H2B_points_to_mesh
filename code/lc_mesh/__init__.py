"""lc_mesh — library for reproducing the LC density meshes and figures.

Pure functions for: loading/preprocessing the CCF-registered points, kNN density
mapping, surfel surface reconstruction + repair, point-in-mesh counting, EMD
self-registration, mesh comparison, and the paper figures. See `code/lc_mesh/
README.md` for usage and reproduction fidelity.
"""
from . import config, figures, registration
from .points import (
    load_ccf_points, reflect_and_crop, compute_knn_density,
    build_lc_points, load_lc_points_csv, load_published_meshes,
    select_shell_and_interior,
)
from .normals import (
    estimate_normals, orient_normals_with_interior, orient_complex_shape_normals,
)
from .meshing import (
    seal_holes, shrink_mesh_along_normals, generate_surface_mesh, repair_mesh,
)
from .analysis import (
    count_points_in_mesh, count_points_in_meshes, mesh_stats, compare_meshes,
)
from .pipeline import (
    make_core_mesh, make_percentile_mesh, make_self_registered_core_mesh,
)
from .registration import self_register

__all__ = [
    "config", "figures", "registration",
    "load_ccf_points", "reflect_and_crop", "compute_knn_density",
    "build_lc_points", "load_lc_points_csv", "load_published_meshes",
    "select_shell_and_interior",
    "estimate_normals", "orient_normals_with_interior", "orient_complex_shape_normals",
    "seal_holes", "shrink_mesh_along_normals", "generate_surface_mesh", "repair_mesh",
    "count_points_in_mesh", "count_points_in_meshes", "mesh_stats", "compare_meshes",
    "make_core_mesh", "make_percentile_mesh", "make_self_registered_core_mesh",
    "self_register",
]
