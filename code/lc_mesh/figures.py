"""Plotting for the LC mesh capsule.

- 1:1 reproductions of the paper figures (fed the canonical published meshes so
  they match the distributed figures).
- A couple of lightweight exploration helpers used by the notebook.

`tifffile` is imported inside the one function that writes a TIFF, since it is
only needed on that save path; everything else is imported at module level.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

MESH_COLORS = ['royalblue', 'blue', 'darkblue', 'lightseagreen', 'seagreen',
               'darkgreen', 'orange', 'darkorange', 'darkred']


# --------------------------------------------------------------------------- #
# kNN minimum-projection heatmaps
# --------------------------------------------------------------------------- #
def min_projection_heatmap(coords, values, axis, pixel_size):
    """Project points onto the plane perpendicular to `axis`; per pixel take the
    minimum value. Returns (heatmap, extent, ax0, ax1)."""
    ax0, ax1 = [a for a in range(3) if a != axis]
    u, v = coords[:, ax0], coords[:, ax1]
    u_min, u_max, v_min, v_max = u.min(), u.max(), v.min(), v.max()
    n_u = int(np.ceil((u_max - u_min) / pixel_size)) + 1
    n_v = int(np.ceil((v_max - v_min) / pixel_size)) + 1
    u_idx = np.clip(((u - u_min) / pixel_size).astype(int), 0, n_u - 1)
    v_idx = np.clip(((v - v_min) / pixel_size).astype(int), 0, n_v - 1)
    heatmap = np.full((n_v, n_u), np.nan)
    for i in range(len(values)):
        ui, vi, val = u_idx[i], v_idx[i], values[i]
        if np.isnan(heatmap[vi, ui]) or val < heatmap[vi, ui]:
            heatmap[vi, ui] = val
    extent = [u_min, u_min + n_u * pixel_size, v_min + n_v * pixel_size, v_min]
    return heatmap, extent, ax0, ax1


def plot_knn_min_projection(df, membership_col='in_new_core_mesh', pixel_size=20,
                            save_path=None):
    """Three orthogonal kNN min-projection heatmaps over the in-core points."""
    core_pts = df[df[membership_col] == 1].copy()
    coords = core_pts[['x', 'y', 'z']].values
    knn_vals = core_pts['kNN_percentile'].values

    axis_labels = {0: 'x', 1: 'y', 2: 'z'}
    proj_names = {0: 'YZ (project along X)', 1: 'XZ (project along Y)',
                  2: 'XY (project along Z)'}
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    vmin = np.nanpercentile(knn_vals, 1)
    vmax = np.nanpercentile(knn_vals, 99)
    for i, proj_axis in enumerate([0, 1, 2]):
        heatmap, extent, ax0, ax1 = min_projection_heatmap(coords, knn_vals, proj_axis, pixel_size)
        im = axes[i].imshow(heatmap, extent=extent, origin='upper', aspect='equal',
                            cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
        axes[i].set_xlabel(f'{axis_labels[ax0]} (µm)')
        axes[i].set_ylabel(f'{axis_labels[ax1]} (µm)')
        axes[i].set_title(f'Min kNN — {proj_names[proj_axis]}')
        plt.colorbar(im, ax=axes[i], label='Min kNN', shrink=0.8)
    plt.suptitle('Minimum Projection of kNN (core mesh points only, 20 µm/px)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=150)
    return fig


# --------------------------------------------------------------------------- #
# 3D Plotly mesh / brain-outline figures
# --------------------------------------------------------------------------- #
def get_scene_layout():
    return dict(
        xaxis_title='', yaxis_title='', zaxis_title='', aspectmode='data',
        xaxis=dict(showbackground=False, gridcolor="white", showticklabels=False,
                   showspikes=False, showaxeslabels=False, visible=False),
        yaxis=dict(showbackground=False, gridcolor="white", showticklabels=False,
                   showspikes=False, showaxeslabels=False, visible=False),
        zaxis=dict(showbackground=False, gridcolor="white", showticklabels=False,
                   showspikes=False, showaxeslabels=False, visible=False),
        bgcolor='rgba(0,0,0,0)', camera=dict(projection=dict(type='orthographic')))


def add_mesh_trace(fig, mesh, color, opacity, name, reflect=False,
                   reflect_axis=0, reflect_point=5700):
    v = mesh.vertices.copy()
    f = mesh.faces
    if reflect:
        v[:, reflect_axis] = 2 * reflect_point - v[:, reflect_axis]
    fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                            i=f[:, 0], j=f[:, 1], k=f[:, 2],
                            color=color, opacity=opacity, name=name))


def add_brain_mesh_trace(fig, brain_mesh):
    v, f = brain_mesh.vertices, brain_mesh.faces
    fig.add_trace(go.Mesh3d(x=v[:, 2], y=v[:, 0], z=v[:, 1],
                            i=f[:, 0], j=f[:, 1], k=f[:, 2],
                            color='gray', opacity=0.1))


def finalize_figure(fig, save_path=None):
    fig.update_layout(showlegend=False, scene=get_scene_layout(), dragmode='orbit',
                      scene_camera=dict(eye=dict(x=-1.1, y=-0.9, z=-0.7),
                                        center=dict(x=0, y=0, z=0),
                                        up=dict(x=0, y=0, z=-1)))
    if save_path:
        fig.write_html(save_path, include_plotlyjs=True)
    return fig


def plot_percentile_and_core(percentile_meshes, core_mesh, brain_mesh=None,
                             percentile=90, save_path=None):
    """90th-percentile mesh + core mesh (both hemispheres) + brain outline."""
    fig = go.Figure()
    add_mesh_trace(fig, percentile_meshes[percentile], 'royalblue', 0.3, f'{percentile}th Percentile')
    add_mesh_trace(fig, percentile_meshes[percentile], 'royalblue', 0.3, f'{percentile}th Percentile (R)', reflect=True)
    add_mesh_trace(fig, core_mesh, 'red', 0.6, 'LC Core Mesh')
    add_mesh_trace(fig, core_mesh, 'red', 0.6, 'LC Core Mesh (R)', reflect=True)
    if brain_mesh is not None:
        add_brain_mesh_trace(fig, brain_mesh)
    return finalize_figure(fig, save_path)


def plot_all_percentile_meshes(percentile_meshes, brain_mesh=None, save_path=None):
    """All percentile meshes, both hemispheres, graded opacity."""
    fig = go.Figure()
    mesh_keys = sorted(percentile_meshes.keys(), reverse=True)  # 90..10
    for i, mesh_key in enumerate(reversed(mesh_keys)):           # 10..90
        opacity = 0.5 - ((len(mesh_keys) - 1 - i) * 0.05)
        color = MESH_COLORS[i % len(MESH_COLORS)]
        add_mesh_trace(fig, percentile_meshes[mesh_key], color, opacity, f'{mesh_key}th Percentile (R)', reflect=True)
        add_mesh_trace(fig, percentile_meshes[mesh_key], color, opacity, f'{mesh_key}th Percentile')
    if brain_mesh is not None:
        add_brain_mesh_trace(fig, brain_mesh)
    return finalize_figure(fig, save_path)


def plot_points_core_membership(df, percentile_meshes, core_mesh, brain_mesh=None,
                                membership_col='in_new_core_mesh', save_path=None):
    """Points colored by core-mesh membership + 90th percentile + core."""
    fig = go.Figure()
    out = df[df[membership_col] == 0]
    inside = df[df[membership_col] == 1]
    fig.add_trace(go.Scatter3d(x=out['x'], y=out['y'], z=out['z'], mode='markers',
                               marker=dict(size=1, color='royalblue', opacity=0.2), name='Outside Core'))
    fig.add_trace(go.Scatter3d(x=inside['x'], y=inside['y'], z=inside['z'], mode='markers',
                               marker=dict(size=1, color='crimson', opacity=0.2), name='Inside Core'))
    add_mesh_trace(fig, percentile_meshes[90], 'royalblue', 0.3, '90th Percentile', reflect=True)
    add_mesh_trace(fig, core_mesh, 'red', 0.6, 'LC Core Mesh', reflect=True)
    if brain_mesh is not None:
        add_brain_mesh_trace(fig, brain_mesh)
    return finalize_figure(fig, save_path)


# --------------------------------------------------------------------------- #
# Per-sample / per-hemisphere counts in each mesh
# --------------------------------------------------------------------------- #
def plot_percentile_counts_by_hemisphere(df, save_path=None):
    """Per-sample point counts in each `in_*` mesh column, connecting the two
    hemispheres. Requires `count_points_in_meshes` to have run first."""
    files = np.sort(df['file'].unique())
    percentile_cols = [c for c in df.columns if 'in_' in c]
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(12, 6))
    x_base = np.arange(len(percentile_cols)) * 2 + 1
    for fi, fname in enumerate(files):
        color = colors[fi % len(colors)]
        for i, col in enumerate(percentile_cols):
            grp = df[df['file'] == fname].groupby('reflected')[col].sum()
            ax.plot([x_base[i], x_base[i] + 1], [grp.get(0, 0), grp.get(1, 0)],
                    marker='o', color=color, label=fname if i == 0 else "", alpha=0.7)
    ax.set_xticks(x_base + 0.5)
    ax.set_xticklabels([c.replace('in_', '') for c in percentile_cols], rotation=45)
    ax.set_ylabel('Sum'); ax.set_xlabel('Percentile Mesh')
    ax.set_title('Sum of Points in Each Percentile Mesh by Hemisphere and Brain')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


# --------------------------------------------------------------------------- #
# Raw-image max-projection overlay
# --------------------------------------------------------------------------- #
def plot_max_proj_with_points(zarr_vol, pts, ap_range=(3100, 3200), vmax=1500,
                              save_path=None):
    """Max-intensity projection of a raw zarr slab with LC points overlaid.
    Returns (fig, max_projection_array)."""
    zarr_arr = zarr_vol[0, 0, :, ap_range[0]:ap_range[1], :]
    max_proj = zarr_arr.max(axis=1)
    mask = (pts[:, 1] >= ap_range[0]) & (pts[:, 1] < ap_range[1])
    pts_in = pts[mask]
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(max_proj, cmap='gray_r', vmin=0, vmax=vmax)
    if pts_in.shape[0] > 0:
        sc = ax.scatter(pts_in[:, 2], pts_in[:, 0], s=2, c='royalblue', alpha=0.6, edgecolors='none')
        ax.legend([sc], [f'{pts_in.shape[0]} LC points'], loc='upper left')
    ax.set_title(f'Max Projection (AP {ap_range[0]}:{ap_range[1]})')
    ax.set_xlabel('X (pixels)'); ax.set_ylabel('Z (pixels)')
    plt.tight_layout()
    if save_path:
        import tifffile  # only needed when writing the TIFF
        tifffile.imwrite(save_path, max_proj.astype(np.uint16))
    return fig, max_proj


# --------------------------------------------------------------------------- #
# Coronal slices with per-sample PC2 histograms
# --------------------------------------------------------------------------- #
def mesh_plane_intersection(vertices, faces, y_value):
    """Points where mesh edges cross the plane y = y_value."""
    points = []
    for tri in faces:
        v0, v1, v2 = vertices[tri]
        for v_start, v_end in [(v0, v1), (v1, v2), (v2, v0)]:
            y_start, y_end = v_start[1], v_end[1]
            if (y_start - y_value) * (y_end - y_value) < 0:
                t = (y_value - y_start) / (y_end - y_start)
                points.append(v_start + t * (v_end - v_start))
    return np.array(points) if points else np.empty((0, 3))


def plot_coronal_slices_with_pc2(df, core_mesh, y_planes=None, save_dir=None,
                                 sample_n=36000, seed=42):
    """Per-y coronal scatter of LC points colored by sample, with the core-mesh
    cross-section and per-sample PC2 histograms overlaid. Returns a list of
    (y_plane, fig)."""
    if y_planes is None:
        y_planes = np.arange(10000, 11001, 100)
    min_x, max_x = df['x'].min(), df['x'].max()
    min_z, max_z = df['z'].min(), df['z'].max()
    core_vertices, core_faces = core_mesh.vertices, core_mesh.faces

    file_list = df['file'].unique()
    colors_list = plt.cm.tab10.colors
    file_colors = {f: colors_list[i % len(colors_list)] for i, f in enumerate(np.sort(file_list))}

    sample = df.sample(n=sample_n, random_state=seed) if len(df) > sample_n else df
    fixed_xlim = (min_x, max_x + 500)
    fixed_ylim = (-max_z - 200, -min_z + 700)
    fixed_bin_width = 10.0

    # PASS 1: global max smoothed count + PC2 bounds
    global_max_count = 0
    baseline_length_at_max = 1.0
    global_pc2_min, global_pc2_max = np.inf, -np.inf
    for y_plane in y_planes:
        m = (sample['y'] > y_plane - 50) & (sample['y'] < y_plane + 50)
        lc = sample[m].copy()
        if len(lc) <= 10:
            continue
        xnz = lc[['x', 'z']].values.copy(); xnz[:, 1] = -xnz[:, 1]
        inl = np.all(np.abs(xnz - xnz.mean(0)) <= 3 * xnz.std(0), axis=1)
        xnz_in = xnz[inl]; files_in = lc['file'].values[inl]
        if len(xnz_in) <= 10:
            continue
        pca = PCA(n_components=2).fit(xnz_in)
        proj = (xnz_in - pca.mean_) @ pca.components_[1]
        gmin, gmax = proj.min(), proj.max()
        global_pc2_min = min(global_pc2_min, gmin)
        global_pc2_max = max(global_pc2_max, gmax)
        bin_edges = np.arange(gmin, gmax + fixed_bin_width, fixed_bin_width)
        for fname in np.sort(np.unique(files_in)):
            pf = proj[files_in == fname]
            if len(pf) < 5:
                continue
            cs = gaussian_filter1d(np.histogram(pf, bins=bin_edges)[0].astype(float), sigma=1.0)
            if cs.max() > global_max_count:
                global_max_count = cs.max()
                baseline_length_at_max = gmax - gmin

    baseline_trim = 500
    global_bin_edges = np.arange(global_pc2_min + baseline_trim,
                                 global_pc2_max - baseline_trim + fixed_bin_width, fixed_bin_width)
    global_bin_centers = 0.5 * (global_bin_edges[:-1] + global_bin_edges[1:])
    n_bins_global = len(global_bin_centers)
    global_scale_factor = 0.9 * baseline_length_at_max / (global_max_count + 1e-6)

    # PASS 2: plot
    figs = []
    for idx, y_plane in enumerate(y_planes):
        inter = mesh_plane_intersection(core_vertices, core_faces, y_plane)
        m = (sample['y'] > y_plane - 50) & (sample['y'] < y_plane + 50)
        lc = sample[m].copy().sample(frac=1, random_state=seed).reset_index(drop=True)
        pcolors = [file_colors[f] for f in lc['file']]

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(lc['x'], -lc['z'], s=5, color=pcolors, alpha=0.3, linewidths=0)
        if len(inter) > 0:
            ax.plot(inter[:, 0], -inter[:, 2], 'o', markersize=1, color='blue',
                    label=f'Core Mesh @ y={y_plane}', alpha=0.5)

        if len(lc) > 10:
            xnz = lc[['x', 'z']].values.copy(); xnz[:, 1] = -xnz[:, 1]
            inl = np.all(np.abs(xnz - xnz.mean(0)) <= 3 * xnz.std(0), axis=1)
            xnz_in = xnz[inl]; files_in = lc['file'].values[inl]
            if len(xnz_in) > 10:
                pca = PCA(n_components=2).fit(xnz_in)
                pc1_vector, pc2_vector, mean_xnz = pca.components_[0], pca.components_[1], pca.mean_
                proj = (xnz_in - mean_xnz) @ pc2_vector
                proj_pc1 = (xnz_in - mean_xnz) @ pc1_vector
                if pc1_vector[1] < 0:
                    baseline_offset, hist_sign = proj_pc1.min() - 50, -1.0
                else:
                    baseline_offset, hist_sign = proj_pc1.max() + 50, 1.0

                def pc2_hist_to_xy(centers, counts, scale):
                    base = mean_xnz[None, :] + centers[:, None] * pc2_vector[None, :]
                    base = base + baseline_offset * pc1_vector[None, :]
                    return (base + (hist_sign * counts)[:, None] * pc1_vector[None, :] * scale).T

                bx, by = pc2_hist_to_xy(global_bin_centers, np.zeros(n_bins_global), global_scale_factor)
                ax.plot(bx, by, color='black', linewidth=1, linestyle='--', alpha=0.3)
                tb = pc2_hist_to_xy(global_bin_centers[:1], np.array([0.0]), global_scale_factor)
                tt = pc2_hist_to_xy(global_bin_centers[:1], np.array([30.0]), global_scale_factor)
                ax.plot([tb[0][0], tt[0][0]], [tb[1][0], tt[1][0]], color='black',
                        linewidth=2, solid_capstyle='butt', alpha=0.3)
                for fname in np.sort(np.unique(files_in)):
                    pf = proj[files_in == fname]
                    if len(pf) < 5:
                        continue
                    cs = gaussian_filter1d(np.histogram(pf, bins=global_bin_edges)[0].astype(float), sigma=1.0)
                    lx, ly = pc2_hist_to_xy(global_bin_centers, cs, global_scale_factor)
                    ax.plot(lx, ly, color=file_colors[fname], linewidth=2, alpha=0.85)

        for f in file_list:
            if (lc['file'] == f).any():
                ax.scatter([], [], s=15, color=file_colors[f],
                           label=f.replace('points_ccf_', '').replace('.npy', ''))
        ax.set_xlabel('x'); ax.set_ylabel('z')
        ax.set_title(f'LC_only_points at y={y_plane} (±50), colored by file')
        if idx == 0:
            ax.legend(markerscale=3, loc='upper right', fontsize=8)
        ax.set_aspect('equal'); ax.set_xlim(fixed_xlim); ax.set_ylim(fixed_ylim)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f'LC_slice_y{y_plane}.svg'),
                        format='svg', bbox_inches='tight')
        figs.append((y_plane, fig))
    return figs


# --------------------------------------------------------------------------- #
# Regenerated-vs-published comparison
# --------------------------------------------------------------------------- #
def plot_mesh_comparison(regenerated, published, save_path=None, seed=0):
    """Compare a regenerated mesh against the published one.

    A 3-row layout (original mesh / regenerated mesh / pointwise difference) over the
    three orthogonal projections (xy, xz, yz). The difference row colors each
    regenerated vertex by its distance to the published surface. Axes use a single
    physical scale (equal aspect + width ratios by data span), shared-axis labels are
    not repeated, and the colorbar gets its own column so the rows stay aligned."""
    from .analysis import nearest_surface_distances

    d = nearest_surface_distances(regenerated.vertices, published, seed=seed)
    rv, pv = regenerated.vertices, published.vertices
    vr, vp = regenerated.volume / 1e9, published.volume / 1e9
    pct = 100 * abs(vr - vp) / vp

    allv = np.vstack([pv, rv])
    pad = 30
    (xmn, ymn, zmn), (xmx, ymx, zmx) = allv.min(0), allv.max(0)
    xlim, ylim, zlim = (xmn - pad, xmx + pad), (ymn - pad, ymx + pad), (zmn - pad, zmx + pad)
    Xs, Ys = xmx - xmn, ymx - ymn

    # per projection column: (horiz idx, vert idx, horiz lim, vert lim, horiz label, vert label)
    cols = [(0, 1, xlim, ylim, 'x (µm)', 'y (µm)'),   # xy
            (0, 2, xlim, zlim, 'x (µm)', 'z (µm)'),   # xz: vertical z (also shown by yz)
            (1, 2, ylim, zlim, 'y (µm)', 'z (µm)')]   # yz: shares z with xz -> hide its vertical
    width_ratios = [Xs, Xs, Ys, max(Xs, Ys) * 0.08]   # last column = colorbar
    proj_titles = ['XY projection', 'XZ projection', 'YZ projection']
    row_titles = ['Original mesh', 'Regenerated mesh',
                  'Point-wise difference between original and regenerated']

    fig = plt.figure(figsize=(15, 16))
    subfigs = fig.subfigures(3, 1, hspace=0.02)
    sc = None
    for r, sf in enumerate(subfigs):
        sf.suptitle(row_titles[r], fontweight='bold', fontsize=14)
        axs = sf.subplots(1, 4, width_ratios=width_ratios)
        bottom = (r == 2)
        for c, (hi, vi, hlim, vlim, hlab, vlab) in enumerate(cols):
            ax = axs[c]
            if r == 0:
                ax.scatter(pv[:, hi], pv[:, vi], s=3, c='0.5', alpha=0.5, linewidths=0)
            elif r == 1:
                ax.scatter(rv[:, hi], rv[:, vi], s=3, c='royalblue', alpha=0.5, linewidths=0)
            else:
                sc = ax.scatter(rv[:, hi], rv[:, vi], s=4, c=d, cmap='viridis',
                                alpha=0.85, linewidths=0, vmin=0, vmax=np.percentile(d, 99))
            ax.set_xlim(hlim); ax.set_ylim(vlim)
            ax.set_aspect('equal', adjustable='box')
            ax.spines[['top', 'right']].set_visible(False)
            if r == 0:                       # projection header labels each column
                ax.set_title(proj_titles[c], fontsize=11)
            if bottom:                       # horizontal axis shared down each column
                ax.set_xlabel(hlab)
            else:
                ax.tick_params(labelbottom=False)
            if c == 2:                       # yz shares z with xz -> drop its vertical
                ax.tick_params(labelleft=False)
            elif bottom:
                ax.set_ylabel(vlab)
        if bottom and sc is not None:
            fig.colorbar(sc, cax=axs[3],
                         label='regenerated vertex →\npublished surface (µm)')
        else:
            axs[3].set_visible(False)

    fig.suptitle(
        f'Regenerated vs published core mesh — '
        f'volume {vr:.6f} vs {vp:.6f} mm³ ({pct:.3f}%),  '
        f'vertices {len(rv)} vs {len(pv)},  '
        f'surface dist mean {d.mean():.2f} µm / max {d.max():.2f} µm',
        fontsize=12, y=1.04)
    if save_path:
        fig.savefig(save_path, format='png', bbox_inches='tight', dpi=150)
    return fig


# --------------------------------------------------------------------------- #
# Lightweight exploration helpers (used by the notebook)
# --------------------------------------------------------------------------- #
def plot_mesh_3d(mesh, points=None, color="orange", opacity=0.5, title=""):
    """Quick Plotly view of a mesh, optionally overlaid with a point cloud."""
    v, f = mesh.vertices, mesh.faces
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                            i=f[:, 0], j=f[:, 1], k=f[:, 2],
                            color=color, opacity=opacity, name="mesh"))
    if points is not None:
        p = np.asarray(points)
        fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                                   marker=dict(size=1.5, color="steelblue", opacity=0.4),
                                   name="points"))
    fig.update_layout(title=title, scene=dict(aspectmode="data"),
                      margin=dict(l=0, r=0, t=30, b=0))
    return fig


def plot_threshold_scatter(df, threshold, n_sample=10000, seed=0):
    """Three orthogonal scatter views highlighting points below a kNN-percentile
    threshold — for interactively choosing the shell threshold."""
    rng = np.random.RandomState(seed)
    n = min(n_sample, len(df))
    samp = df.iloc[rng.choice(len(df), n, replace=False)]
    hi = samp[samp["kNN_percentile"] < threshold]
    lo = samp[samp["kNN_percentile"] >= threshold]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for k, (a, b, lbl) in enumerate([("x", "y", "x vs y"), ("x", "z", "x vs z"),
                                     ("y", "z", "y vs z")]):
        ax[k].scatter(lo[a], lo[b], s=2, c="gray", alpha=0.05)
        ax[k].scatter(hi[a], hi[b], s=2, c="red", alpha=0.1)
        ax[k].set(xlabel=a, ylabel=b, title=lbl)
        ax[k].axis("equal")
    fig.suptitle(f"kNN percentile < {threshold}")
    fig.tight_layout()
    return fig
