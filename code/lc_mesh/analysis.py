"""Point-in-mesh counting (cell 42) and mesh-to-mesh comparison metrics.

`compare_meshes` is the core of the reproducibility check: it quantifies how far
a freshly regenerated mesh is from the published one (volume, surface distance,
topology), which is what a regenerate-and-revalidate (B') claim rests on.
"""
import numpy as np
import trimesh

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
        bounds=mesh.bounds.tolist(),
    )


def _surface_distances(a, b, n_sample=50000, seed=0):
    """One-directional surface distances: points sampled on `a`'s surface to the
    EXACT nearest point on `b`'s surface (trimesh proximity). Sampling the *other*
    surface and using a KDTree would impose a floor of ~the sample spacing (a few
    µm here), which badly overstates the distance between near-identical meshes."""
    rng = np.random.RandomState(seed)
    pts_a, _ = _sample_surface(a, n_sample, rng)
    return trimesh.proximity.ProximityQuery(b).on_surface(pts_a)[1]


def _sample_surface(mesh, n, rng):
    """Deterministic uniform surface sampling (area-weighted)."""
    areas = mesh.area_faces
    probs = areas / areas.sum()
    face_idx = rng.choice(len(mesh.faces), size=n, p=probs)
    tris = mesh.triangles[face_idx]
    u = rng.random_sample((n, 1))
    v = rng.random_sample((n, 1))
    over = (u + v) > 1
    u[over] = 1 - u[over]
    v[over] = 1 - v[over]
    pts = tris[:, 0] + u * (tris[:, 1] - tris[:, 0]) + v * (tris[:, 2] - tris[:, 0])
    return pts, face_idx


def nearest_surface_distances(points, mesh, n_sample=80000, seed=0):
    """EXACT distance from each of `points` (N x 3) to the surface of `mesh`
    (trimesh proximity). Used to color the comparison figure, so it reflects true
    geometric deviation rather than surface-sampling resolution."""
    return trimesh.proximity.ProximityQuery(mesh).on_surface(np.asarray(points))[1]


def compare_meshes(regenerated, published, n_sample=50000, seed=0):
    """Compare a regenerated mesh against the published reference.

    Returns volume comparison, symmetric surface distances (microns), and
    topology deltas. Distances are in the mesh's native units (microns here).
    """
    d_rp = _surface_distances(regenerated, published, n_sample, seed)
    d_pr = _surface_distances(published, regenerated, n_sample, seed)

    vr = regenerated.volume / 1e9
    vp = published.volume / 1e9
    return dict(
        regenerated=mesh_stats(regenerated),
        published=mesh_stats(published),
        volume_mm3_regenerated=float(vr),
        volume_mm3_published=float(vp),
        volume_abs_diff_mm3=float(abs(vr - vp)),
        volume_pct_diff=float(100 * abs(vr - vp) / vp),
        hausdorff_um=float(max(d_rp.max(), d_pr.max())),
        mean_surface_dist_um=float((d_rp.mean() + d_pr.mean()) / 2),
        median_surface_dist_um=float((np.median(d_rp) + np.median(d_pr)) / 2),
        p95_surface_dist_um=float((np.percentile(d_rp, 95) + np.percentile(d_pr, 95)) / 2),
        n_vertices_delta=int(len(regenerated.vertices) - len(published.vertices)),
        n_faces_delta=int(len(regenerated.faces) - len(published.faces)),
    )
