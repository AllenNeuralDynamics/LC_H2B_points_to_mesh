"""Self-registration of the 16 hemispheres via Earth Mover's Distance.

Faithful extraction of notebook cells 56-61. Rigid (rotation+translation)
registration of each (file, reflected) group to a reference group, minimizing EMD
(POT `ot.emd2`) with L-BFGS-B. SLOW (~40 min for all 16 groups); call behind a
flag, not by default.
"""
import numpy as np
import ot
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **k):
        return x


# --------------------------------------------------------------------------- #
# EMD-based rigid registration (cell 57)
# --------------------------------------------------------------------------- #
def compute_emd(X, Y, numItermax=1000000):
    if len(X) == 0 or len(Y) == 0:
        return 1e10
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    all_pts = np.vstack([X, Y])
    shift = all_pts.min(axis=0)
    scale = (all_pts.max(axis=0) - shift)
    scale[scale == 0] = 1.0
    X_norm = (X - shift) / scale
    Y_norm = (Y - shift) / scale
    M = np.ascontiguousarray(cdist(X_norm, Y_norm, metric='euclidean'), dtype=np.float64)
    a = np.ones(len(X_norm), dtype=np.float64) / len(X_norm)
    b = np.ones(len(Y_norm), dtype=np.float64) / len(Y_norm)
    emd_val = ot.emd2(a, b, M, numItermax=numItermax)
    return emd_val * np.mean(scale)


def apply_rigid_transform(points, R, t):
    return (R @ points.T).T + t


def rotation_matrix_from_params(params):
    rx, ry, rz = params
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def register_to_reference(moving_points, ref_points, init_params=None, verbose=False):
    if init_params is None:
        init_params = np.zeros(6)
    moving_points = np.asarray(moving_points, dtype=np.float64)
    ref_points = np.asarray(ref_points, dtype=np.float64)
    eval_count = [0]

    def objective(params):
        eval_count[0] += 1
        R = rotation_matrix_from_params(params[:3])
        t = np.array(params[3:])
        emd = compute_emd(apply_rigid_transform(moving_points, R, t), ref_points)
        if verbose and eval_count[0] % 50 == 0:
            print(f"  Eval {eval_count[0]}: EMD={emd:.4f}")
        return emd

    bounds = [(-np.pi, np.pi)] * 3 + [(-5000, 5000)] * 3
    result = minimize(objective, init_params, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 3000, 'ftol': 1e-7, 'gtol': 1e-6,
                               'maxcor': 30, 'maxfun': 5000})
    return {
        'params': result.x,
        'R': rotation_matrix_from_params(result.x[:3]),
        't': result.x[3:],
        'emd': result.fun,
        'success': result.success,
        'n_evals': eval_count[0],
        'n_iters': result.nit,
    }


# --------------------------------------------------------------------------- #
# Reference selection + orchestration (cells 56, 58-61)
# --------------------------------------------------------------------------- #
def select_reference_group(df, coord_cols=('x', 'y', 'z'), res=32):
    """Pick the (file, reflected) group whose coarse voxel occupancy is closest
    to the mean occupancy across all groups (cell 56)."""
    coords_all = df[list(coord_cols)].values.astype(np.float32)
    mins, maxs = coords_all.min(axis=0), coords_all.max(axis=0)
    edges = [np.linspace(mins[d], maxs[d], res + 1) for d in range(3)]
    group_keys = list(df.groupby(['file', 'reflected']).groups.keys())
    occ_list = []
    for key in group_keys:
        g = df[(df['file'] == key[0]) & (df['reflected'] == key[1])]
        pts = g[list(coord_cols)].values
        if len(pts) == 0:
            occ = np.zeros((res, res, res), dtype=np.float32)
        else:
            h, _ = np.histogramdd(pts, bins=edges)
            occ = (h > 0).astype(np.float32)
            occ = occ / (occ.sum() + 1e-12)
        occ_list.append(occ.ravel())
    occ_stack = np.vstack(occ_list)
    dists = np.linalg.norm(occ_stack - occ_stack.mean(axis=0)[None, :], axis=1)
    return group_keys[int(np.argmin(dists))]


def register_all_groups(df, ref_key, coord_cols=('x', 'y', 'z'), verbose=False):
    """Register every (file, reflected) group to the reference (cell 58)."""
    ref_df = df[(df['file'] == ref_key[0]) & (df['reflected'] == ref_key[1])]
    ref_pts = ref_df[list(coord_cols)].values.astype(np.float32)
    results = {}
    for key in tqdm(sorted(df.groupby(['file', 'reflected']).groups.keys())):
        if key == ref_key:
            results[key] = {'R': np.eye(3), 't': np.zeros(3), 'emd': 0.0,
                            'success': True, 'n_evals': 0, 'n_iters': 0}
            continue
        g = df[(df['file'] == key[0]) & (df['reflected'] == key[1])]
        moving = g[list(coord_cols)].values.astype(np.float32)
        if len(moving) < 10:
            continue
        results[key] = register_to_reference(moving, ref_pts, verbose=verbose)
    return results


def apply_registration(df, results, coord_cols=('x', 'y', 'z')):
    """Apply per-group transforms, add reg_x/reg_y/reg_z and reg_error (cells 59-60)."""
    df = df.copy()
    cx, cy, cz = coord_cols
    df['reg_x'], df['reg_y'], df['reg_z'] = df[cx].copy(), df[cy].copy(), df[cz].copy()
    for key, res in results.items():
        mask = (df['file'] == key[0]) & (df['reflected'] == key[1])
        if mask.sum() == 0:
            continue
        pts = df.loc[mask, list(coord_cols)].values.astype(np.float64)
        transformed = ((res['R'] @ pts.T).T + res['t']).astype(np.float32)
        df.loc[mask, 'reg_x'] = transformed[:, 0]
        df.loc[mask, 'reg_y'] = transformed[:, 1]
        df.loc[mask, 'reg_z'] = transformed[:, 2]
    orig = df[list(coord_cols)].values.astype(np.float64)
    reg = df[['reg_x', 'reg_y', 'reg_z']].values.astype(np.float64)
    df['reg_error'] = np.linalg.norm(reg - orig, axis=1)
    return df


def self_register(df, coord_cols=('x', 'y', 'z'), verbose=False):
    """Full self-registration pipeline. Returns (df_with_reg_cols, results, ref_key)."""
    ref_key = select_reference_group(df, coord_cols)
    if verbose:
        print(f"Reference group: {ref_key}")
    results = register_all_groups(df, ref_key, coord_cols, verbose=verbose)
    return apply_registration(df, results, coord_cols), results, ref_key
