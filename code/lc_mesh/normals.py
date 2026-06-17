"""Normal estimation and orientation for the point-cloud shell.

Faithful extraction of notebook cell 16. The mesh pipeline always supplies
interior points, so `orient_complex_shape_normals` uses the interior-guided path
(`orient_normals_with_interior`); the patch/MST fallback is preserved for parity
but is not exercised by the published pipeline.

NOTE (preserved from the original): `estimate_normals` ignores its `k` argument
and always uses 40 neighbours. Kept as-is so results match the published meshes.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **k):
        return x


def estimate_normals(shell_points, k=40):
    """Per-point normal via PCA (SVD) of the local 40-neighbour neighbourhood."""
    k_normals = 40  # NB: the original hard-codes 40 regardless of `k`
    nbrs = NearestNeighbors(n_neighbors=k_normals + 1, algorithm='kd_tree').fit(shell_points)
    normals = np.zeros((len(shell_points), 3))
    for i in tqdm(range(len(shell_points))):
        _, indices = nbrs.kneighbors([shell_points[i]])
        neighbors = shell_points[indices[0]]
        centered = neighbors - np.mean(neighbors, axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normals[i] = vh[2, :]  # eigenvector of smallest eigenvalue
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def orient_normals_with_interior(shell_points, normals, interior_points,
                                 k_interior=30, k_smooth=20):
    """Orient normals outward using nearby interior points, then smooth for
    local consistency (3 iterations)."""
    oriented = normals.copy()
    n = len(shell_points)
    print(f"Orienting {n} normals using {len(interior_points)} interior points...")

    nbrs_interior = NearestNeighbors(n_neighbors=k_interior, algorithm='kd_tree').fit(interior_points)
    distances, indices = nbrs_interior.kneighbors(shell_points)

    inside_to_outside = np.zeros_like(shell_points)
    for i in range(n):
        nearby = interior_points[indices[i]]
        weights = 1.0 / (distances[i] + 1e-10)
        weights = weights / np.sum(weights)
        weighted_interior = np.sum(nearby * weights.reshape(-1, 1), axis=0)
        inside_to_outside[i] = shell_points[i] - weighted_interior

    norms = np.linalg.norm(inside_to_outside, axis=1).reshape(-1, 1)
    inside_to_outside = np.divide(inside_to_outside, norms,
                                  out=np.zeros_like(inside_to_outside), where=norms != 0)

    dots = np.sum(oriented * inside_to_outside, axis=1)
    flip = dots < 0
    oriented[flip] *= -1
    print(f"Flipped {np.sum(flip)} normals based on interior point orientation")

    nbrs_shell = NearestNeighbors(n_neighbors=k_smooth + 1, algorithm='kd_tree').fit(shell_points)
    _, indices = nbrs_shell.kneighbors(shell_points)
    for iteration in range(3):
        changes = 0
        for i in range(n):
            neighbors = indices[i, 1:]
            dp = np.sum(oriented[i] * oriented[neighbors], axis=1)
            if (len(neighbors) - np.sum(dp > 0)) > np.sum(dp > 0):
                oriented[i] *= -1
                changes += 1
        print(f"Iteration {iteration + 1}: Flipped {changes} normals for consistency")
        if changes == 0:
            break
    return oriented


def propagate_through_patches(shell_points, normals, patch_size=100, overlap=0.5):
    """Patch/MST-based orientation fallback (used only when no interior points
    are available). Preserved for parity; not part of the published path."""
    n_points = len(shell_points)
    oriented = normals.copy()
    n_patches = max(1, int(n_points / (patch_size * (1 - overlap / 2))))
    kmeans = KMeans(n_clusters=n_patches, random_state=0).fit(shell_points)
    labels = kmeans.labels_

    for patch_id in range(n_patches):
        mask = labels == patch_id
        pts = shell_points[mask]
        pn = oriented[mask]
        k = min(30, max(10, len(pts) // 10))
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(pts)
        distances, indices = nbrs.kneighbors(pts)
        rows, cols, weights = [], [], []
        for i in range(len(pts)):
            for j_idx, j in enumerate(indices[i, 1:]):
                dp = abs(np.dot(pn[i], pn[j]))
                rows.append(i); cols.append(j)
                weights.append(distances[i, j_idx + 1] * (2.0 - dp))
        graph = csr_matrix((weights, (rows, cols)), shape=(len(pts), len(pts)))
        mst = minimum_spanning_tree(graph).tocsr()
        center = np.mean(pts, axis=0)
        root = np.argmax(np.linalg.norm(pts - center, axis=1))
        visited = np.zeros(len(pts), dtype=bool)

        def dfs(node):
            visited[node] = True
            _, nb = mst[node].nonzero()
            ri, ci = mst.nonzero()
            rev = ri[ci == node]
            alln = np.concatenate([nb, rev]) if len(nb) and len(rev) else (nb if len(nb) else rev)
            for neighbor in alln:
                if not visited[neighbor]:
                    if np.dot(pn[node], pn[neighbor]) < 0:
                        pn[neighbor] = -pn[neighbor]
                    dfs(neighbor)
        dfs(root)
        oriented[mask] = pn

    nbrs = NearestNeighbors(n_neighbors=15, algorithm='kd_tree').fit(shell_points)
    _, indices = nbrs.kneighbors(shell_points)
    for i in range(n_points):
        neighbors = indices[i, 1:10]
        dp = np.sum(oriented[i] * oriented[neighbors], axis=1)
        if np.sum(dp < 0) > len(neighbors) / 2:
            oriented[i] *= -1
    return oriented


def orient_complex_shape_normals(shell_points, normals, interior_points=None):
    """Dispatch: interior-guided orientation when interior points exist, else
    patch-based (cell 16)."""
    if interior_points is not None and len(interior_points) > 0:
        return orient_normals_with_interior(shell_points, normals.copy(), interior_points)
    return propagate_through_patches(shell_points, normals.copy())
