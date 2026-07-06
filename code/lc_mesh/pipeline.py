"""High-level mesh builders that compose the lower-level modules into the
select -> generate -> repair sequence for the core and percentile meshes."""
from . import config
from .points import select_shell_and_interior
from .meshing import generate_surface_mesh, repair_mesh


def make_core_mesh(df, coord_cols=('x', 'y', 'z'), verbose=True):
    """Build the LC core mesh (67th-percentile shell) from an LC_only_points frame,
    using the core recipe in config.CORE / config.REPAIR. Returns (repaired, raw)."""
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
    """Build a percentile mesh from the full recipe in config.PERCENTILE_PARAMS
    (generation params plus that percentile's own repair settings). Returns
    (repaired, raw); when the recipe's `repair` is None, no extra repair pass is
    applied. `repair_overrides` replaces the recipe's repair settings if given."""
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
    repair = repair_overrides if repair_overrides is not None else p.get('repair')
    if repair is None:
        return raw, raw
    return repair_mesh(raw, **repair, verbose=verbose), raw
