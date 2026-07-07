"""Point-in-mesh counting (cell 42) and basic mesh descriptors."""
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **k):
        return x


def count_points_in_mesh(mesh, coords, batch_size=1000, verbose=False):
    """Return a boolean array marking which `coords` (N x 3) are inside `mesh`
    (batched `mesh.contains`, cell 42)."""
    coords = np.asarray(coords)
    n = len(coords)
    inside = np.zeros(n, dtype=bool)
    it = range(0, n, batch_size)
    for start in (tqdm(it) if verbose else it):
        end = min(start + batch_size, n)
        inside[start:end] = mesh.contains(coords[start:end])
    return inside


def count_points_in_meshes(df, meshes, coord_cols=('x', 'y', 'z'),
                           col_prefix='in_', batch_size=1000, verbose=False):
    """Add an ``in_<key>`` column for each mesh in ``meshes`` (dict), marking which
    points fall inside. Returns the modified copy.

    Integer-keyed meshes are treated as the nested percentile family (percentile_10 is
    contained in ... is contained in percentile_90) and use the original notebook's fast
    scheme (cell 55): walk them smallest-to-largest, and for each mesh only ray-test the
    still-unplaced points whose coordinates fall inside the mesh's bounding box; once a
    point is found inside a mesh, mark it inside that mesh and every larger one and stop
    testing it. This avoids running the expensive ``mesh.contains`` ray test on every point
    against every mesh. Any non-integer key (e.g. the string ``'new_core_mesh'``) is tested
    directly with batched ``mesh.contains`` (cell 51), since it is not part of that nested
    chain.
    """
    df = df.copy()
    coords = df[list(coord_cols)].values
    n = len(coords)

    pct_keys = sorted(k for k in meshes if isinstance(k, (int, np.integer)))
    pct_set = set(pct_keys)
    plain_keys = [k for k in meshes if k not in pct_set]

    # Plain meshes (e.g. the core): direct batched containment, all points.
    for key in plain_keys:
        df[f'{col_prefix}{key}'] = count_points_in_mesh(
            meshes[key], coords, batch_size=batch_size, verbose=verbose).astype(int)

    # Nested percentile meshes: bounding-box pre-filter + ascending membership propagation.
    inside = {key: np.zeros(n, dtype=bool) for key in pct_keys}
    assigned = np.zeros(n, dtype=bool)
    for i, key in enumerate(pct_keys):
        idx = np.where(~assigned)[0]  # points not yet placed inside any smaller mesh
        if idx.size == 0:
            break
        mesh = meshes[key]
        lo, hi = mesh.bounds
        in_bbox = np.all((coords[idx] >= lo) & (coords[idx] <= hi), axis=1)
        cand = idx[in_bbox]  # only these can be inside; ray-test just them
        hit_mask = np.zeros(len(cand), dtype=bool)
        it = range(0, len(cand), batch_size)
        for start in (tqdm(it, desc=f'{col_prefix}{key}') if verbose else it):
            end = min(start + batch_size, len(cand))
            hit_mask[start:end] = mesh.contains(coords[cand[start:end]])
        hit = cand[hit_mask]
        for higher in pct_keys[i:]:  # inside key => inside every larger mesh
            inside[higher][hit] = True
        assigned[hit] = True
    for key in pct_keys:
        df[f'{col_prefix}{key}'] = inside[key].astype(int)

    return df


def mesh_stats(mesh):
    """Basic descriptors of a mesh."""
    return dict(
        n_vertices=int(len(mesh.vertices)),
        n_faces=int(len(mesh.faces)),
        volume_mm3=float(mesh.volume / 1e9),
        watertight=bool(mesh.is_watertight),
        euler_number=int(mesh.euler_number),
        bounds=mesh.bounds.tolist(),
    )
