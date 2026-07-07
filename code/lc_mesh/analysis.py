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
    """Add an `in_<key>` column for each mesh in `meshes` (dict), marking which
    points fall inside (cell 46). Returns the modified copy."""
    df = df.copy()
    coords = df[list(coord_cols)].values
    for key, mesh in meshes.items():
        df[f'{col_prefix}{key}'] = count_points_in_mesh(
            mesh, coords, batch_size=batch_size, verbose=verbose).astype(int)
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
