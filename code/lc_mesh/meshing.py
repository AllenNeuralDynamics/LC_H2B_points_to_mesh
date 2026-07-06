"""Surface reconstruction and repair.

Faithful extraction of notebook cells 17 (seal_holes), 16 (shrink), 19 (surface
generation), and 25/27/29/31 (repair). The repair `shrink` / `extra_seal_passes`
options correspond to the optional cells 32/33 that were toggled per-mesh by
visual inspection in the original notebook.
"""
from collections import defaultdict

import numpy as np
import trimesh
import open3d as o3d
import point_cloud_utils as pcu
from scipy import ndimage
from scipy.spatial import cKDTree

from .normals import estimate_normals, orient_complex_shape_normals

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **k):
        return x


# --------------------------------------------------------------------------- #
# Hole sealing (cell 17)
# --------------------------------------------------------------------------- #
def seal_holes(mesh):
    """Seal all boundary loops with a centroid fan-triangulation, in place."""
    V = mesh.vertices
    F = mesh.faces

    edge_count = dict()
    for face in F:
        for a, b in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
            e = tuple(sorted((a, b)))
            edge_count[e] = edge_count.get(e, 0) + 1
    boundary_edges = [e for e, count in edge_count.items() if count == 1]

    loops = []
    edges_left = set(boundary_edges)
    while edges_left:
        loop = []
        e = edges_left.pop()
        loop.extend(e)
        while True:
            last = loop[-1]
            found = None
            for candidate in edges_left:
                if last in candidate:
                    found = candidate
                    break
            if found is None:
                break
            edges_left.remove(found)
            next_v = found[0] if found[1] == last else found[1]
            loop.append(next_v)
            if next_v == loop[0]:
                break
        seen = set()
        loop_clean = []
        for v in loop:
            if v not in seen:
                loop_clean.append(v)
                seen.add(v)
        loops.append(loop_clean)

    new_vertices, new_faces = [], []
    for loop in loops:
        coords = V[loop]
        centroid = coords.mean(axis=0)
        centroid_idx = len(V) + len(new_vertices)
        new_vertices.append(centroid)
        N = len(loop)
        for i in range(N):
            new_faces.append([centroid_idx, loop[i], loop[(i + 1) % N]])

    if new_vertices:
        mesh.vertices = np.vstack([V, np.array(new_vertices)])
        mesh.faces = np.vstack([F, np.array(new_faces)])


def shrink_mesh_along_normals(mesh, distance):
    """Move each vertex inward along its normal (cell 16)."""
    shrunk = mesh.copy()
    vn = mesh.vertex_normals
    norm = np.linalg.norm(vn, axis=1).reshape(-1, 1)
    norm[norm == 0] = 1.0
    shrunk.vertices = mesh.vertices - (vn / norm) * distance
    shrunk.fix_normals()
    return shrunk


# --------------------------------------------------------------------------- #
# Surface generation (cell 19)
# --------------------------------------------------------------------------- #
def generate_surface_mesh(shell_points, interior_points, surfel_radius,
                          watertight_resolution, smooth_iterations=5,
                          normals_k=80, verbose=True):
    """Estimate + orient normals, reconstruct a surfel surface, make watertight,
    and Laplacian-smooth (Open3D). Returns a trimesh.Trimesh."""
    shell_points = np.asarray(shell_points, dtype=np.float32)
    interior_points = np.asarray(interior_points, dtype=np.float32)

    normals = estimate_normals(shell_points, k=normals_k)
    oriented = orient_complex_shape_normals(shell_points, normals.copy(),
                                            interior_points=interior_points)
    oriented = np.asfortranarray(oriented, dtype=np.float32)

    vertices, faces = pcu.pointcloud_surfel_geometry(shell_points, oriented, surfel_radius)
    vertices_wt, faces_wt = pcu.make_mesh_watertight(vertices, faces, watertight_resolution)
    mesh = trimesh.Trimesh(vertices=vertices_wt, faces=faces_wt)
    while not mesh.is_watertight:
        if verbose:
            print("Mesh is not watertight, retrying...")
        vertices_wt, faces_wt = pcu.make_mesh_watertight(vertices_wt, faces_wt,
                                                         watertight_resolution)
        mesh = trimesh.Trimesh(vertices=vertices_wt, faces=faces_wt)

    mesh_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces),
    )
    mesh_o3d.compute_vertex_normals()
    mesh_o3d = mesh_o3d.filter_smooth_simple(number_of_iterations=smooth_iterations)
    return trimesh.Trimesh(np.asarray(mesh_o3d.vertices), np.asarray(mesh_o3d.triangles))


# --------------------------------------------------------------------------- #
# Repair (cells 25, 27, 29, 31, + optional 32/33)
# --------------------------------------------------------------------------- #
def _vertex_surface_distance(mesh, pitch, max_distance, verbose=True):
    """Voxelize, run a bounded distance-transform from the surface, and return
    each vertex's distance-to-surface in voxels (cell 25)."""
    voxelized = mesh.voxelized(pitch=pitch).fill()
    voxel_matrix = voxelized.matrix
    voxel_coords = voxelized.points

    distance = np.zeros_like(voxel_matrix, dtype=np.uint8)
    surface = ndimage.binary_erosion(voxel_matrix) ^ voxel_matrix
    frontier = surface.copy()
    it = range(1, max_distance + 1)
    for d in (tqdm(it, desc="distance transform") if verbose else it):
        distance[frontier] = d
        frontier = ndimage.binary_dilation(frontier) & voxel_matrix & (distance == 0)
    distance[distance == 0] = max_distance + 1

    tree = cKDTree(voxel_coords)
    _, nearest_idx = tree.query(mesh.vertices, k=1)
    distance_flat = distance[voxel_matrix]
    return distance_flat[nearest_idx]


def repair_mesh(mesh, pitch=3, max_distance=5, keep_distance=2,
                shrink=None, extra_seal_passes=0, verbose=True):
    """Detect/strip near-surface caverns, seal holes, and clean broken/solitary
    faces. Returns a new (ideally watertight) trimesh.Trimesh.

    `shrink` (float) and `extra_seal_passes` (int) reproduce the optional cells
    32/33; both default off, which is the core mesh recipe.
    """
    vertex_distance = _vertex_surface_distance(mesh, pitch, max_distance, verbose=verbose)

    # cell 27: drop vertices beyond keep_distance, remap faces
    keep = vertex_distance <= keep_distance
    old_to_new = -np.ones(len(mesh.vertices), dtype=int)
    old_to_new[keep] = np.arange(np.sum(keep))
    filtered_vertices = mesh.vertices[keep]
    face_mask = keep[mesh.faces].all(axis=1)
    filtered_faces = old_to_new[mesh.faces[face_mask]]

    # cell 29: seal
    sealed = trimesh.Trimesh(vertices=filtered_vertices, faces=filtered_faces)
    trimesh.repair.fix_winding(sealed)
    seal_holes(sealed)
    trimesh.repair.fix_winding(sealed)
    trimesh.repair.fill_holes(sealed)
    trimesh.repair.fix_normals(sealed, multibody=True)

    # cell 31: broken faces + solitary edges
    broken = np.asarray(trimesh.repair.broken_faces(sealed))
    if broken.size:
        keep_idx = np.setdiff1d(np.arange(len(sealed.faces)),
                                broken.astype(int), assume_unique=True)
        sealed.faces = sealed.faces[keep_idx]
        sealed.remove_unreferenced_vertices()

        edge_count = defaultdict(int)
        for face in sealed.faces:
            for a, b in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
                edge_count[tuple(sorted((a, b)))] += 1
        solitary_mask = np.array([
            not any(edge_count[tuple(sorted((a, b)))] == 1
                    for a, b in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])])
            for f in sealed.faces
        ])
        sealed.faces = sealed.faces[solitary_mask]
        sealed.remove_unreferenced_vertices()
    elif verbose:
        print("No broken faces.")

    # optional cell 32
    if shrink:
        sealed = shrink_mesh_along_normals(sealed, distance=shrink)

    # optional cell 33
    for _ in range(extra_seal_passes):
        trimesh.repair.fix_winding(sealed)
        seal_holes(sealed)
        trimesh.repair.fix_winding(sealed)
        trimesh.repair.fill_holes(sealed)
        trimesh.repair.fix_normals(sealed, multibody=True)

    return sealed
