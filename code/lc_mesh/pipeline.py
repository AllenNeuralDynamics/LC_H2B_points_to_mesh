"""High-level mesh builders that compose the lower-level modules into the exact
sequences used in the notebook (select -> generate -> repair)."""
from . import config
from .points import select_shell_and_interior, compute_knn_density
from .meshing import generate_surface_mesh, repair_mesh


def make_core_mesh(df, coord_cols=('x', 'y', 'z'), verbose=True):
    """Build the LC core mesh (67th-percentile) from an LC_only_points frame.

    Uses the fully-recovered core recipe in config.CORE / config.REPAIR. This is
    the mesh that, in the original April rerun, regenerated to within ~0.08% of
    the published `new_core_mesh.obj`.
    """
    c = config.CORE
    shell, interior = select_shell_and_interior(
        df, c['shell_lo'], c['shell_hi'], c['interior_hi'], coord_cols)
    if verbose:
        print(f"shell points: {len(shell)}, interior points: {len(interior)}")
    raw = generate_surface_mesh(
        shell, interior,
        surfel_radius=c['surfel_radius'],
        watertight_resolution=c['watertight_resolution'],
        smooth_iterations=c['smooth_iterations'],
        normals_k=c['normals_k'], verbose=verbose)
    return repair_mesh(raw, **config.REPAIR, verbose=verbose), raw


def make_percentile_mesh(df, thresh, coord_cols=('x', 'y', 'z'),
                         repair_overrides=None, verbose=True):
    """Build a percentile mesh. Generation params are recovered (config), but the
    per-mesh repair steps are NOT — pass `repair_overrides` (e.g. {'shrink': 5,
    'extra_seal_passes': 1}) if you have them from the author; otherwise the core
    repair recipe is used as a (non-authoritative) default."""
    p = config.percentile_params(thresh)
    shell, interior = select_shell_and_interior(
        df, p['shell_lo'], p['shell_hi'], p['interior_hi'], coord_cols)
    if verbose:
        print(f"[thresh={thresh}] shell: {len(shell)}, interior: {len(interior)}, "
              f"radius={p['surfel_radius']}, res={p['watertight_resolution']}")
    raw = generate_surface_mesh(
        shell, interior,
        surfel_radius=p['surfel_radius'],
        watertight_resolution=p['watertight_resolution'],
        smooth_iterations=p['smooth_iterations'],
        normals_k=p['normals_k'], verbose=verbose)
    repair_kwargs = dict(config.REPAIR)
    if repair_overrides:
        repair_kwargs.update(repair_overrides)
    return repair_mesh(raw, **repair_kwargs, verbose=verbose), raw


def make_self_registered_core_mesh(df, verbose=True):
    """Rebuild the core mesh from self-registered coordinates (cells 63-67).

    Expects `df` to already carry `reg_x/reg_y/reg_z` (run registration first).
    Recomputes kNN density on the registered coords, then applies the core recipe.
    """
    df = compute_knn_density(df, coord_cols=('reg_x', 'reg_y', 'reg_z'))
    return make_core_mesh(df, coord_cols=('reg_x', 'reg_y', 'reg_z'), verbose=verbose)
