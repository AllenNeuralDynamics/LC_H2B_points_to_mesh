# The LC core mesh: a small non-orientable fold, and why the results are unchanged

## Short version

We compared the distributed LC core mesh (the "original") with a mesh we regenerated from scratch from the raw cell points (the "new" one). They describe the **same shape** and enclose the **same cells**: every one of the 34,998 points is on the same side (inside or outside) of both meshes, and both contain exactly **24,111 cells**. So the cell counts and densities they yield are identical.

The two meshes differ in exactly one way: the original contains a single **non-orientable fold** near the ventral tip, about 20 microns across. The new mesh is an orientable closed solid without it. This difference is in the surface's topology (how the triangles are connected), not in the shape, the enclosed region, or any cell count.

An interactive 3D view of the two meshes side by side (rotate and zoom into the fold) is here:
<https://dougollerenshaw.github.io/LC_H2B_points_to_mesh/results/nonorientable_fold.html>

## What we actually measured

Each mesh is a surface built from triangles: it has corners (**V**, the vertices), **E** edges between them, and **F** triangular faces.

| | Original (distributed) | New (regenerated) |
|---|---|---|
| Vertices (V) | 21,219 | 21,219 |
| Edges (E) | 63,654 | 63,651 |
| Faces (F) | 42,436 | 42,434 |
| Euler number (V − E + F) | **1** | **2** |
| Cells enclosed | **24,111** | **24,111** |

The two meshes share the same vertices to within less than a nanometer for 21,121 of the 21,219 points. The remaining 98 differ by up to about 8 microns, scattered over the surface, and come from floating-point differences in the surface-reconstruction step.

## What the Euler number is, in plain terms

The **Euler number** is a single whole number that acts like a "shape fingerprint." You compute it by counting the corners, edges, and faces of a surface and combining them:

> Euler number = V − E + F

This number is unaffected by size, stretching, or bending. It only cares about the *type* of shape:

- Anything shaped like a **ball** (a single closed blob with no holes and no handles, whether it's a sphere, a potato, or an LC) always gives **2**. You can dent it, inflate it, or re-triangulate it however you like, and it stays 2.
- Add a **handle** (turn it into a donut or a coffee mug) and it drops to **0**. Each extra handle subtracts 2.

So for any ordinary solid object, the surface that wraps it has an Euler number of **2**. Equivalently, a closed triangle mesh that correctly seals a solid must have exactly **F = 2V − 4** faces. For V = 21,219 that is 42,434 faces, which is exactly what the new mesh has.

The original mesh has an Euler number of **1**, and two more faces than the formula allows. A value of 1 cannot happen for the normal surface of a solid. It is the signature of a surface that is **non-orientable**.

## What "non-orientable" means physically

Imagine tiling the surface so that every tile is laid face-up, with "outside" painted on the top of each tile. On any real solid, you can do this consistently: neighboring tiles always agree on which side is out.

At three edges of the original mesh, the two triangles meeting at the edge are laid the *same* way rather than back-to-back, like two floor tiles meeting at a seam where one is face-up and the other face-down. If you walk across that seam and come back around, your sense of "outside" has flipped. This is the same idea as a **Mobius strip**, the paper loop with a half-twist that has only one side.

The boundary surface of any solid region in three dimensions is necessarily orientable. This fold is therefore a property of how the mesh triangles were connected during surface reconstruction, not of the underlying cell positions.

## Why it has no effect on results

This is the key point: **the fold changes a topological property of the surface, not the region it encloses.**

- **Same cells.** Point-by-point, both meshes classify all 34,998 points identically, and both enclose exactly 24,111 cells. Cell counts and densities are unaffected.
- **Same enclosed space.** Measured in a way that does not depend on orientation (filling the meshes on a grid and comparing the filled volumes), the two enclosed volumes are identical.
- **The only number that moves is a computational artifact.** The standard formula for a mesh's "signed volume" assumes the surface is consistently oriented. The fold breaks that assumption, so that formula reports a volume about 0.08% larger for the original mesh. This difference is due to the orientation inconsistency, not to a difference in enclosed volume. If volume is ever reported, it should be measured on the new (orientable) mesh, or with an orientation-independent method.

## Recommendation

Both meshes give the same cell counts and densities. For distribution we recommend the **new, regenerated mesh**, because it is an orientable closed solid (Euler number 2), its volume can be computed with the standard formula without the artifact, and, unlike the original mesh, it can be generated reproducibly with known parameters from the code in this repo. The original mesh remains usable for any result that depends only on which cells are inside, which is unchanged.

## How to verify this yourself

Everything above is reproducible from the two mesh files:

- `code/verify_mesh_comparison.py` prints the topology (Euler number, orientability), the orientation-independent volume, the surface agreement, and the inside/outside cell counts for the two meshes.
- `code/visualize_fold.py` locates the fold from the raw triangle list (no hand-picked coordinates) and writes the static figure and the interactive 3D page linked above.
