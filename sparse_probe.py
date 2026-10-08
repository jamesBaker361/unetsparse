# Per-latent 1D ridge logistic probes (Eq. (7) of "Rediscovering SAEs",
# arXiv:2511.17735), ported from the aesthetic repo's sparse_probe.py.
#
# fit_sparse_1d_ridge_logistic works directly on top-k sparse SAE codes and
# never materializes the dense (n_patches, n_dirs) matrix: every zero entry of
# latent j contributes to the gradient/Hessian only through expit(b_j), so
# those contributions collapse into per-latent counts. fit_dense_1d_ridge_logistic
# is the same probe, one per column of a dense matrix (raw activations, input
# patches), vectorized over columns in chunks.

import numpy as np
from scipy.special import expit


def _flatten_entries(idx: np.ndarray, val: np.ndarray):
    '''
    idx/val: (n_patches, k) top-k latent indices/values per patch. Returns
    (rows, cols, z) for the strictly positive entries only - a relu'd zero is
    identical to "not in the top k" for these probes.
    '''
    n, k = idx.shape
    rows = np.repeat(np.arange(n), k)
    cols = idx.reshape(-1).astype(np.int64)
    z = val.reshape(-1).astype(np.float64)
    keep = z > 0
    return rows[keep], cols[keep], z[keep]


def fit_sparse_1d_ridge_logistic(idx: np.ndarray, val: np.ndarray, labels: np.ndarray, n_dirs: int,
                                 ridge: float = 1e-8, n_newton_steps: int = 30):
    '''
    Returns w, b, loss (each (n_dirs,)); loss is the unregularized mean training BCE.
    '''
    y = labels.astype(bool)
    n = len(y)
    n_pos = float(y.sum())
    rows, cols, z = _flatten_entries(idx, val)
    y_e = y[rows].astype(np.float64)

    nnz = np.bincount(cols, minlength=n_dirs).astype(np.float64)
    nnz_pos = np.bincount(cols, weights=y_e, minlength=n_dirs)
    n0 = n - nnz  # patches where latent j is inactive (z=0)
    n0_pos = n_pos - nnz_pos

    w = np.zeros(n_dirs, dtype=np.float64)
    b = np.full(n_dirs, n_pos / n, dtype=np.float64)

    for _ in range(n_newton_steps):
        p_e = expit(w[cols] * z + b[cols])
        resid = p_e - y_e
        s_e = p_e * (1.0 - p_e)
        p0 = expit(b)

        g_w = np.bincount(cols, weights=resid * z, minlength=n_dirs) + 2.0 * ridge * w
        g_b = np.bincount(cols, weights=resid, minlength=n_dirs) + (n0 * p0 - n0_pos)
        h_ww = np.bincount(cols, weights=s_e * z * z, minlength=n_dirs) + 2.0 * ridge
        h_wb = np.bincount(cols, weights=s_e * z, minlength=n_dirs)
        h_bb = np.bincount(cols, weights=s_e, minlength=n_dirs) + n0 * p0 * (1.0 - p0)

        det = h_ww * h_bb - h_wb * h_wb
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        w = w - (h_bb * g_w - h_wb * g_b) / det
        b = b - (h_ww * g_b - h_wb * g_w) / det

    eps = 1e-12
    p_e = np.clip(expit(w[cols] * z + b[cols]), eps, 1 - eps)
    p0 = np.clip(expit(b), eps, 1 - eps)
    nll_e = -(y_e * np.log(p_e) + (1 - y_e) * np.log(1 - p_e))
    loss = np.bincount(cols, weights=nll_e, minlength=n_dirs)
    loss += -(n0_pos * np.log(p0) + (n0 - n0_pos) * np.log(1 - p0))
    return w, b, loss / n


def sparse_per_latent_confusion(idx: np.ndarray, val: np.ndarray, labels: np.ndarray, w: np.ndarray,
                                b: np.ndarray, threshold: float = 0.5):
    '''Per-latent tp/fp/fn of each fitted probe thresholded at `threshold`.'''
    n_dirs = len(w)
    y = labels.astype(bool)
    n = len(y)
    n_pos = float(y.sum())
    rows, cols, z = _flatten_entries(idx, val)
    y_e = y[rows]

    nnz = np.bincount(cols, minlength=n_dirs).astype(np.float64)
    nnz_pos = np.bincount(cols, weights=y_e.astype(np.float64), minlength=n_dirs)
    n0 = n - nnz
    n0_pos = n_pos - nnz_pos

    pred_e = expit(w[cols] * z + b[cols]) >= threshold
    pred0 = (expit(b) >= threshold).astype(np.float64)

    tp = np.bincount(cols, weights=(pred_e & y_e).astype(np.float64), minlength=n_dirs) + pred0 * n0_pos
    fp = np.bincount(cols, weights=(pred_e & ~y_e).astype(np.float64), minlength=n_dirs) + pred0 * (n0 - n0_pos)
    fn = n_pos - tp
    return tp, fp, fn


def fit_dense_1d_ridge_logistic(X: np.ndarray, labels: np.ndarray, ridge: float = 1e-8,
                                n_newton_steps: int = 30, chunk: int = 32):
    '''
    One 1D probe per column of X (n, d). Returns w, b, loss, tp, fp, fn
    (each (d,)), loss = unregularized mean training BCE, confusion at 0.5.
    '''
    y = labels.astype(np.float64)
    n, d = X.shape
    w, b, loss = np.zeros(d), np.zeros(d), np.zeros(d)
    tp, fp, fn = np.zeros(d), np.zeros(d), np.zeros(d)
    eps = 1e-12
    for s in range(0, d, chunk):
        x = X[:, s:s + chunk].astype(np.float64)
        wc = np.zeros(x.shape[1])
        bc = np.full(x.shape[1], y.mean())
        for _ in range(n_newton_steps):
            p = expit(x * wc + bc)
            r = p - y[:, None]
            sw = p * (1 - p)
            g_w = (r * x).sum(0) + 2 * ridge * wc
            g_b = r.sum(0)
            h_ww = (sw * x * x).sum(0) + 2 * ridge
            h_wb = (sw * x).sum(0)
            h_bb = sw.sum(0)
            det = h_ww * h_bb - h_wb * h_wb
            det = np.where(np.abs(det) < 1e-12, 1e-12, det)
            wc = wc - (h_bb * g_w - h_wb * g_b) / det
            bc = bc - (h_ww * g_b - h_wb * g_w) / det
        p = np.clip(expit(x * wc + bc), eps, 1 - eps)
        w[s:s + chunk], b[s:s + chunk] = wc, bc
        loss[s:s + chunk] = -(y[:, None] * np.log(p) + (1 - y[:, None]) * np.log(1 - p)).mean(0)
        pred, yb = p >= 0.5, y[:, None] > 0.5
        tp[s:s + chunk] = (pred & yb).sum(0)
        fp[s:s + chunk] = (pred & ~yb).sum(0)
        fn[s:s + chunk] = (~pred & yb).sum(0)
    return w, b, loss, tp, fp, fn


def precision_recall_f1(tp, fp, fn):
    tp, fp, fn = (np.asarray(a, dtype=np.float64) for a in (tp, fp, fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    return precision, recall, f1


def baseline_bce(labels: np.ndarray) -> float:
    '''BCE of the best constant predictor (the prevalence); "loss explained" is measured against it.'''
    prev = float(np.clip(np.mean(labels.astype(np.float64)), 1e-12, 1 - 1e-12))
    return float(-(prev * np.log(prev) + (1 - prev) * np.log(1 - prev)))


def latent_column(idx: np.ndarray, val: np.ndarray, latent: int) -> np.ndarray:
    '''Dense activation of one latent over all patches (0 where inactive).'''
    return np.maximum(np.where(idx == latent, val, 0).astype(np.float64).sum(1), 0.0)


def scaled_csr(idx: np.ndarray, val: np.ndarray, n_dirs: int, scale: np.ndarray = None):
    '''Top-k codes as a (n_patches, n_dirs) CSR matrix, each latent divided by scale (default: its max).'''
    from scipy.sparse import csr_matrix
    rows, cols, z = _flatten_entries(idx, val)
    if scale is None:
        scale = np.zeros(n_dirs)
        np.maximum.at(scale, cols, z)
        scale[scale == 0] = 1.0
    return csr_matrix((z / scale[cols], (rows, cols)), shape=(len(idx), n_dirs)), scale
