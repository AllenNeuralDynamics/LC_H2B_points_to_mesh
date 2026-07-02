# /// script
# requires-python = "==3.10.*"
# dependencies = ["numpy==1.26.4", "trimesh==4.11.1", "scipy==1.15.3",
#                 "matplotlib==3.10.8", "plotly==6.5.2"]
# ///
"""Show WHERE the published LC core mesh has a non-orientable fold, and that the
regenerated one does not.

The fold location is COMPUTED here, not hand-picked. A watertight mesh that bounds
a solid must be orientable: every interior edge is shared by two triangles that
traverse it in OPPOSITE directions (like floor tiles all laid face-up). The published
mesh has a few edges where the two triangles traverse the edge the SAME way -- a fold
the surface can't consistently orient. Those triangles ARE the fold; `fold_faces`
finds them from the raw triangle list. That is why the mesh has Euler number 1
(an orientable closed solid has 2) and two more triangles than such a solid has.

Colors, kept identical in the static PNG and the interactive HTML:
  red   = the fold triangles (published only)
  green = triangles sharing a vertex with the fold, matched by CONNECTIVITY across the
          two meshes so the SAME triangle is green in both panels (its shape may shift)
  blue  = the regenerated triangles over the same spot (no published counterpart)

    uv run code/visualize_fold.py     # writes results/nonorientable_fold.{png,html}
"""
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import trimesh
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO = Path(__file__).resolve().parent.parent
PUBLISHED = REPO / "results-c712751d-f744-4fe8-9657-93a7084eab22" / "new_core_mesh.obj"
REGENERATED = REPO / "results" / "reproduced" / "new_core_mesh.obj"
PNG_OUT = REPO / "results" / "nonorientable_fold.png"
HTML_OUT = REPO / "results" / "nonorientable_fold.html"

RED, GREEN, BLUE, GRAY = "red", "#2ca02c", "#1f77b4", "lightgray"


def topo(mesh):
    """(V, E, F, Euler number) from a mesh's OWN counts. An orientable closed solid has
    Euler = V - E + F = 2 and F = 2V - 4; a different value is a property of the mesh
    itself (here, a non-orientable fold), needing no comparison to another mesh."""
    V, F = len(mesh.vertices), len(mesh.faces)
    E = len(mesh.edges_unique)
    return V, E, F, V - E + F


def fold_faces(mesh):
    """Indices of triangles touching a winding-inconsistent edge (the fold).

    For every interior edge the two triangles sharing it should use it in opposite
    directions: (u->v) in one, (v->u) in the other. If a direction is used twice, the
    two triangles are wound the same way -- a non-orientable fold. Return those faces.
    """
    faces = np.asarray(mesh.faces)
    directed = defaultdict(int)        # (u, v) -> how many times used in that direction
    edge_to_faces = defaultdict(list)  # sorted (u, v) -> face indices sharing the edge
    for fi, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            directed[(int(u), int(v))] += 1
            edge_to_faces[tuple(sorted((int(u), int(v))))].append(fi)
    bad = set()
    for (u, v), fs in edge_to_faces.items():
        if len(fs) == 2 and (directed[(u, v)] != 1 or directed[(v, u)] != 1):
            bad.update(fs)
    return sorted(bad)


def faces_sharing_vertex(faces, seed_faces):
    """All faces that share at least one vertex with any of `seed_faces`."""
    seed_verts = np.unique(faces[list(seed_faces)])
    return np.nonzero(np.isin(faces, seed_verts).any(axis=1))[0].tolist()


def tri_key(face, vmap=None):
    """A triangle's identity by CONNECTIVITY: the sorted triple of vertex indices it
    joins (optionally remapped through `vmap`). Matching by connectivity, not by corner
    coordinates, keeps a triangle "the same" when a vertex merely shifts a little."""
    return tuple(sorted(int(vmap[v]) if vmap is not None else int(v) for v in face))


def classify(pub, reg):
    """Compute the colored face sets for both meshes (shared by the PNG and the HTML)."""
    pub_v, pub_f = np.asarray(pub.vertices), np.asarray(pub.faces)
    reg_v, reg_f = np.asarray(reg.vertices), np.asarray(reg.faces)

    fold = fold_faces(pub)                                     # red (published)
    center = pub_v[pub_f[fold]].reshape(-1, 3).mean(axis=0)
    neighbours = sorted(set(faces_sharing_vertex(pub_f, fold)) - set(fold))  # green (published)

    # Follow triangles by connectivity across meshes: map each published vertex to its
    # nearest regenerated vertex (they agree to sub-micron except ~98 near the fold),
    # then a triangle "survives" if the same vertex-triple is also a regenerated face.
    pub2reg = cKDTree(reg_v).query(pub_v)[1]
    reg_lookup = {tri_key(f): fi for fi, f in enumerate(reg_f)}
    reg_green = [reg_lookup[k] for fi in neighbours
                 if (k := tri_key(pub_f[fi], pub2reg)) in reg_lookup]

    # blue = every regenerated triangle touching where the fold used to be, minus the
    # surviving green neighbours. This colors the whole retriangulated patch (no gaps).
    fold_verts_reg = {int(pub2reg[v]) for fi in fold for v in pub_f[fi]}
    green_set = set(reg_green)
    reg_blue = [fi for fi in np.nonzero(np.isin(reg_f, list(fold_verts_reg)).any(axis=1))[0]
                if fi not in green_set]

    return dict(pub_v=pub_v, pub_f=pub_f, reg_v=reg_v, reg_f=reg_f, center=center,
                fold=fold, neighbours=neighbours, reg_green=reg_green, reg_blue=reg_blue,
                pub_topo=topo(pub), reg_topo=topo(reg))


# ----------------------------------------------------------------------------- static PNG
def draw_patch(ax, verts, faces, center, half, colored=()):
    """Draw the local mesh patch projected onto the y-z plane, viewed down +x (higher x =
    nearer the viewer). The fold sits ~120 um BEHIND the near wall, so we first CUT AWAY
    every face in front of the region of interest (centroid x above the colored faces);
    otherwise the near wall would sit between us and the fold and the colored faces, drawn
    on top, would look painted onto that wall. What remains is painter-sorted by depth and
    drawn opaque, with the colored faces (now the frontmost real geometry) on top."""
    yz = verts[:, 1:3]
    cy, cz = center[1], center[2]
    fc = verts[faces].mean(axis=1)
    highlight = {int(i) for idxs, _ in colored for i in idxs}
    x_cut = fc[list(highlight), 0].max() + 5.0 if highlight else fc[:, 0].max()
    keep = ((np.abs(fc[:, 1] - cy) < 1.6 * half) & (np.abs(fc[:, 2] - cz) < 1.6 * half)
            & (fc[:, 0] <= x_cut))
    gray = sorted((fi for fi in np.nonzero(keep)[0] if fi not in highlight), key=lambda fi: fc[fi, 0])
    for fi in gray:
        tri = yz[faces[fi]]
        ax.fill(tri[:, 0], tri[:, 1], facecolor="0.85", edgecolor="0.45", linewidth=0.4)
    for face_indices, color in colored:
        for fi in sorted(face_indices, key=lambda fi: fc[fi, 0]):
            tri = yz[faces[fi]]
            ax.fill(tri[:, 0], tri[:, 1], facecolor=color, edgecolor="0.15", linewidth=0.8)
    ax.set(xlim=(cy - half, cy + half), ylim=(cz - half, cz + half),
           xlabel="y (µm)", ylabel="z (µm)")
    ax.set_aspect("equal")


def write_png(d, pub):
    fold, nb, center = d["fold"], d["neighbours"], d["center"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    fig.suptitle(
        f"Published core mesh: {len(fold)} triangles form a non-orientable fold "
        f"(Euler 1; {len(d['pub_f'])} faces, 2 more than an orientable solid).  "
        f"Regenerated mesh: orientable closed solid (Euler 2).",
        fontsize=12)

    axes[0].scatter(d["pub_v"][:, 1], d["pub_v"][:, 2], s=0.5, color="0.75", linewidths=0)
    axes[0].add_patch(plt.Circle((center[1], center[2]), 70, fill=False, color="red", linewidth=2))
    axes[0].annotate("fold", (center[1], center[2]), (center[1] + 120, center[2] - 250),
                     color="red", fontsize=11, arrowprops=dict(arrowstyle="->", color="red"))
    axes[0].set(title="Whole LC mesh (y-z)\nnon-orientable fold circled", xlabel="y (µm)", ylabel="z (µm)")
    axes[0].set_aspect("equal")

    draw_patch(axes[1], d["pub_v"], d["pub_f"], center, 45, colored=[(nb, GREEN), (fold, RED)])
    axes[1].set_title(f"PUBLISHED zoom\nnon-orientable fold = {len(fold)} winding-inconsistent faces (red)\n"
                      f"neighbours share a vertex (green)", fontsize=10)
    draw_patch(axes[2], d["reg_v"], d["reg_f"], center, 45,
               colored=[(d["reg_green"], GREEN), (d["reg_blue"], BLUE)])
    axes[2].set_title(f"REGENERATED zoom\nre-triangulated patch over the same spot (blue)\n"
                      f"whole mesh: {len(d['reg_f'])} faces = 2V-4 (orientable solid)", fontsize=10)

    # Intrinsic proof: each mesh's own topology, no cross-comparison needed.
    (Vp, Ep, Fp, chip), (Vr, Er, Fr, chir) = d["pub_topo"], d["reg_topo"]
    fig.text(0.5, 0.095, "A property of each mesh on its own (no face-by-face comparison needed):",
             ha="center", fontsize=9.5, weight="bold")
    fig.text(0.5, 0.062, f"PUBLISHED  V={Vp}, E={Ep}, F={Fp}  ->  Euler = V-E+F = {chip}; "
             f"F is {Fp - (2 * Vp - 4)} more than 2V-4  ->  non-orientable fold", ha="center", fontsize=9)
    fig.text(0.5, 0.030, f"REGENERATED  V={Vr}, E={Er}, F={Fr}  ->  Euler = {chir}; "
             f"F = 2V-4 = {2 * Vr - 4}  ->  orientable closed solid", ha="center", fontsize=9)

    fig.tight_layout(rect=(0, 0.15, 1, 0.94))
    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_OUT, dpi=150)
    print(f"wrote {PNG_OUT}")


# ------------------------------------------------------------------- interactive HTML (3D)
# Camera sync: ROTATION (the eye direction + up) propagates to all four scenes, so they
# always face the same way. ZOOM (the eye's distance from centre) only syncs to the
# same-row partner -- top row shares one zoom, bottom row another, independent of each other.
SYNC_JS = """
var gd = document.getElementById('{plot_id}');
var all = ['scene', 'scene2', 'scene3', 'scene4'];
var partner = {scene: 'scene2', scene2: 'scene', scene3: 'scene4', scene4: 'scene3'};
var lock = false;
function mag(e) { return Math.sqrt(e.x * e.x + e.y * e.y + e.z * e.z); }
gd.on('plotly_relayout', function(ev) {
    if (lock) return;
    // Find which scene changed. Rotation fires 'scene.camera' (whole object); zoom often
    // fires granular keys like 'scene.camera.eye.x', so match any 'scene*.camera*' key.
    var src = null, keys = Object.keys(ev);
    for (var i = 0; i < all.length && !src; i++) {
        var pfx = all[i] + '.camera';
        for (var k = 0; k < keys.length; k++) {
            if (keys[k] === pfx || keys[k].indexOf(pfx + '.') === 0) { src = all[i]; break; }
        }
    }
    if (!src) return;
    // Read the authoritative camera from the layout (handles both whole and granular events).
    var cam = gd.layout[src] && gd.layout[src].camera;
    if (!cam || !cam.eye) return;
    lock = true;
    var m0 = mag(cam.eye);
    var dir = {x: cam.eye.x / m0, y: cam.eye.y / m0, z: cam.eye.z / m0};
    var up = cam.up || {x: 0, y: 0, z: 1};
    var ctr = cam.center || {x: 0, y: 0, z: 0};
    var u = {};
    for (var i = 0; i < all.length; i++) {
        var s = all[i];
        if (s === src) continue;
        if (s === partner[src]) {
            u[s + '.camera'] = cam;                      // same row: match rotation AND zoom
        } else {                                         // other row: match rotation, keep own zoom
            var e2 = (gd.layout[s] && gd.layout[s].camera && gd.layout[s].camera.eye) || cam.eye;
            var m = mag(e2);
            u[s + '.camera'] = {eye: {x: dir.x * m, y: dir.y * m, z: dir.z * m},
                                up: up, center: ctr, projection: cam.projection};
        }
    }
    Plotly.relayout(gd, u).then(function () { lock = false; });
});
"""


def add_mesh(fig, row, col, verts, faces, center, colored, crop_radius=None, wire=True,
             hull_opacity=0.15, color_opacity=0.55):
    """Add one mesh to a subplot. Everything is TRANSLUCENT (no opaque faces): the gray
    hull is faint and the colored patches are semi-transparent, so you can see through the
    surface to the fold inside from any angle -- opaque faces just hid it.

    crop_radius=None renders the whole mesh (the overview row); a number crops to a ball
    around the fold (the zoom row). Either way the rendered faces are reindexed into a LOCAL
    sub-mesh before plotting, because Plotly frames a scene to every point it is handed --
    passing the full vertex array for a crop would zoom the view out to the whole LC.
    wire=True overlays a dark wireframe (cheap on the small zoom, skipped on the overview)."""
    highlight = {int(i) for idxs, _ in colored for i in idxs}
    if crop_radius is None:
        render = np.arange(len(faces))
    else:
        crop = set(np.nonzero(np.linalg.norm(verts[faces].mean(axis=1) - center, axis=1)
                              < crop_radius)[0].tolist()) | highlight
        render = np.array(sorted(crop))
    used = np.unique(faces[render])
    remap = np.full(len(verts), -1)
    remap[used] = np.arange(len(used))
    lv = verts[used]
    lx, ly, lz = lv[:, 0], lv[:, 1], lv[:, 2]

    def loc(face_idx):                                  # global face indices -> local vertex triples
        return remap[faces[np.asarray(sorted(face_idx), dtype=int)]]

    gf = loc([fi for fi in render if fi not in highlight])
    fig.add_trace(go.Mesh3d(x=lx, y=ly, z=lz, i=gf[:, 0], j=gf[:, 1], k=gf[:, 2],
                            color=GRAY, opacity=hull_opacity, flatshading=True,
                            hoverinfo="skip", showscale=False), row=row, col=col)
    for face_indices, color in colored:
        if len(face_indices) == 0:
            continue
        cf = loc(face_indices)
        fig.add_trace(go.Mesh3d(x=lx, y=ly, z=lz, i=cf[:, 0], j=cf[:, 1], k=cf[:, 2],
                                color=color, opacity=color_opacity, flatshading=True,
                                lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
                                hoverinfo="skip", showscale=False), row=row, col=col)
    if wire:
        ex, ey, ez = [], [], []
        for f in loc(render):
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                ex += [lx[a], lx[b], None]
                ey += [ly[a], ly[b], None]
                ez += [lz[a], lz[b], None]
        fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                                   line=dict(color="#333333", width=1.5),
                                   hoverinfo="skip", showlegend=False), row=row, col=col)


def write_html(d):
    fold, blue = d["fold"], d["reg_blue"]
    pub_colored = [(d["neighbours"], GREEN), (fold, RED)]
    reg_colored = [(d["reg_green"], GREEN), (blue, BLUE)]
    fig = make_subplots(
        rows=2, cols=2, horizontal_spacing=0.04, vertical_spacing=0.15,
        specs=[[{"type": "scene"}, {"type": "scene"}], [{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("PUBLISHED: full mesh", "REGENERATED: full mesh",
                        f"PUBLISHED: fold zoom<br>non-orientable fold (red) = {len(fold)} faces<br>neighbours (green)",
                        "REGENERATED: fold zoom<br>re-triangulated patch (blue)<br>neighbours (green)"))
    # top row: whole mesh (translucent, no wireframe -- shows where the fold sits in the LC)
    add_mesh(fig, 1, 1, d["pub_v"], d["pub_f"], d["center"], pub_colored, crop_radius=None, wire=False)
    add_mesh(fig, 1, 2, d["reg_v"], d["reg_f"], d["center"], reg_colored, crop_radius=None, wire=False)
    # bottom row: zoom to the fold region (translucent + wireframe)
    add_mesh(fig, 2, 1, d["pub_v"], d["pub_f"], d["center"], pub_colored, crop_radius=65, wire=True)
    add_mesh(fig, 2, 2, d["reg_v"], d["reg_f"], d["center"], reg_colored, crop_radius=65, wire=True)

    # Start every scene looking down +x with z up (the same y-z view as the static image).
    # A short eye distance zooms in so the mesh fills the panel; the two rows get their own
    # start zoom (they zoom independently anyway), so both fill their panels well.
    # Perspective (not orthographic): scroll-zoom then reliably fires a camera relayout,
    # which is what the row-wise zoom sync listens for. Eye down +x, z up (y-z view).
    def _cam(eye_x):
        return dict(eye=dict(x=eye_x, y=0.0, z=0.0), up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0))
    cam_top, cam_bottom = _cam(1.7), _cam(1.05)   # top zoomed out enough to show the ventral tip

    def _scene(cam):                                    # x is edge-on in the start view; hide it
        return dict(aspectmode="data", camera=cam,
                    xaxis=dict(visible=False), yaxis=dict(title="y (µm)"), zaxis=dict(title="z (µm)"))
    for ann in fig.layout.annotations:                  # subplot titles: shrink so the 3-line ones fit
        ann.font.size = 12
    (Vp, _Ep, Fp, chip), (_Vr, _Er, _Fr, chir) = d["pub_topo"], d["reg_topo"]
    fig.add_annotation(xref="paper", yref="paper", x=0.5, y=-0.04, showarrow=False, font=dict(size=11),
                       text=f"A property of each mesh on its own: PUBLISHED Euler=V-E+F={chip} "
                            f"(F is {Fp - (2 * Vp - 4)} more than 2V-4) -> non-orientable fold;   "
                            f"REGENERATED Euler={chir}, F=2V-4 -> orientable closed solid")
    fig.update_layout(
        width=1150, height=1150, autosize=False,        # fixed, near-square panels fill the space
        title_text="LC core mesh: location of the non-orientable fold<br>"
                   "<sub>faces translucent; top = whole mesh, bottom = zoom; rotate together, zoom per row</sub>",
        margin=dict(l=0, r=0, t=70, b=40),
        scene=_scene(cam_top), scene2=_scene(cam_top),
        scene3=_scene(cam_bottom), scene4=_scene(cam_bottom))
    HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(HTML_OUT), include_plotlyjs="cdn", post_script=SYNC_JS)
    print(f"wrote {HTML_OUT}")


def main():
    pub_path = Path(sys.argv[1]) if len(sys.argv) > 1 else PUBLISHED
    reg_path = Path(sys.argv[2]) if len(sys.argv) > 2 else REGENERATED
    pub = trimesh.load(pub_path, process=False)
    reg = trimesh.load(reg_path, process=False)

    d = classify(pub, reg)
    print(f"published : Euler {pub.euler_number}, {len(d['pub_f'])} faces, "
          f"{len(d['fold'])} fold faces (+{len(d['neighbours'])} neighbours) at "
          f"(x,y,z)=({d['center'][0]:.0f}, {d['center'][1]:.0f}, {d['center'][2]:.0f}) µm")
    print(f"regenerated: Euler {reg.euler_number}, {len(d['reg_f'])} faces "
          f"(= 2V-4 = {2 * len(d['reg_v']) - 4}; published had {len(d['pub_f'])}, net {len(d['reg_f']) - len(d['pub_f']):+d}), "
          f"{len(d['reg_green'])}/{len(d['neighbours'])} neighbours survive")

    write_png(d, pub)
    write_html(d)


if __name__ == "__main__":
    main()
