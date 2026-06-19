"""Cross-slice pseudo-label builders via entropic optimal transport."""

import numpy as np
import torch
from scipy.special import logsumexp
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


def zscore(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)


def fit_pca(X1, X2, n_components=50, seed=0):
    X = np.concatenate([zscore(X1), zscore(X2)], axis=0)
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(X)
    Z1 = pca.transform(zscore(X1)).astype(np.float32)
    Z2 = pca.transform(zscore(X2)).astype(np.float32)
    return Z1, Z2


def sinkhorn(C, reg=0.05, n_iters=100, eps=1e-9):
    """Entropic OT coupling between two equal-size distributions (uniform)."""
    C = np.asarray(C, dtype=np.float64)
    n, m = C.shape
    log_a = np.full(n, -np.log(n), dtype=np.float64)
    log_b = np.full(m, -np.log(m), dtype=np.float64)
    log_K = -C / reg
    log_u = np.zeros(n, dtype=np.float64)
    log_v = np.zeros(m, dtype=np.float64)
    for _ in range(n_iters):
        log_u = log_a - logsumexp(log_K + log_v[None, :], axis=1)
        log_v = log_b - logsumexp(log_K.T + log_u[None, :], axis=1)
    log_pi = log_u[:, None] + log_K + log_v[None, :]
    pi = np.exp(log_pi)
    pi /= pi.sum() + eps
    return pi.astype(np.float32)


def _nearest_landmarks(Z, landmarks):
    diff = Z[:, None, :] - landmarks[None, :, :]
    return np.argmin((diff * diff).sum(axis=2), axis=1)


def _landmark_means(y_ref, assign, n_landmarks):
    means = np.zeros((n_landmarks, y_ref.shape[1]), dtype=np.float32)
    counts = np.zeros(n_landmarks, dtype=np.float32)
    for j, lab in enumerate(assign):
        means[lab] += y_ref[j]
        counts[lab] += 1.0
    empty = counts < 1e-8
    counts = np.maximum(counts, 1.0)
    means /= counts[:, None]
    if empty.any():
        means[empty] = y_ref.mean(axis=0)
    return means


def _fallback_raw_pseudo(query, ref, y_ref, k, batch_size, device):
    from run_fig3_mnn_pseudo import build_raw_pseudo
    return build_raw_pseudo(query, ref, y_ref, k=k, batch_size=batch_size, device=device)


def build_landmark_ot_pseudo(
    query,
    ref,
    y_ref,
    k=5,
    pca_dim=50,
    n_landmarks=2048,
    reg=0.05,
    seed=0,
    batch_size=4096,
    device='cuda',
):
    """Landmark Sinkhorn OT in PCA space, then propagate to all query cells."""
    query = np.asarray(query, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    y_ref = np.asarray(y_ref, dtype=np.float32)

    Zq, Zr = fit_pca(query, ref, n_components=pca_dim, seed=seed)
    L = min(n_landmarks, Zq.shape[0], Zr.shape[0])
    print(f'Landmark OT: PCA-{pca_dim}, L={L}, query={Zq.shape[0]}, ref={Zr.shape[0]}')

    km1 = MiniBatchKMeans(n_clusters=L, random_state=seed, batch_size=4096, n_init=3)
    km2 = MiniBatchKMeans(n_clusters=L, random_state=seed + 1, batch_size=4096, n_init=3)
    landmarks1 = km1.fit(Zq).cluster_centers_.astype(np.float32)
    landmarks2 = km2.fit(Zr).cluster_centers_.astype(np.float32)

    diff = landmarks1[:, None, :] - landmarks2[None, :, :]
    C = (diff * diff).sum(axis=2)
    pi = sinkhorn(C, reg=reg)

    assign_q = _nearest_landmarks(Zq, landmarks1)
    assign_r = _nearest_landmarks(Zr, landmarks2)
    ref_means = _landmark_means(y_ref, assign_r, L)

    weights = pi[assign_q]
    wsum = weights.sum(axis=1, keepdims=True)
    pseudo = (weights @ ref_means) / np.maximum(wsum, 1e-8)

    need_fallback = (wsum[:, 0] < 1e-8)
    if need_fallback.any():
        fb = _fallback_raw_pseudo(query[need_fallback], ref, y_ref, k, batch_size, device)
        pseudo[need_fallback] = fb

    return pseudo


def batched_euclidean_topk_indices(query, ref, k=20, batch_size=4096, device='cuda'):
    """Top-k ref indices by smallest squared Euclidean distance."""
    query = torch.as_tensor(query, dtype=torch.float32, device=device)
    ref = torch.as_tensor(ref, dtype=torch.float32, device=device)
    k = min(k, ref.shape[0])
    n_query = query.shape[0]
    ref_norm = (ref * ref).sum(dim=1)
    indices = torch.zeros(n_query, k, dtype=torch.long, device='cpu')
    for start in range(0, n_query, batch_size):
        end = min(start + batch_size, n_query)
        q = query[start:end]
        dist = (q * q).sum(dim=1, keepdim=True) + ref_norm[None, :] - 2.0 * (q @ ref.T)
        indices[start:end] = torch.topk(dist, k, dim=1, largest=False).indices.cpu()
    return indices.numpy(), query.cpu().numpy(), ref.cpu().numpy()


def build_localk_ot_pseudo(
    query,
    ref,
    y_ref,
    k=20,
    pca_dim=50,
    reg=0.05,
    seed=0,
    batch_size=4096,
    device='cuda',
):
    """Per-cell 1-to-k entropic transport in PCA space."""
    query = np.asarray(query, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    y_ref = np.asarray(y_ref, dtype=np.float32)

    Zq, Zr = fit_pca(query, ref, n_components=pca_dim, seed=seed)
    topk_idx, Zq_cpu, Zr_cpu = batched_euclidean_topk_indices(
        Zq, Zr, k=k, batch_size=batch_size, device=device)

    pseudo = np.zeros((Zq.shape[0], y_ref.shape[1]), dtype=np.float32)
    for start in range(0, Zq.shape[0], batch_size):
        end = min(start + batch_size, Zq.shape[0])
        idx = topk_idx[start:end]
        neighbors = Zr_cpu[idx]
        q = Zq_cpu[start:end][:, None, :]
        dist2 = ((q - neighbors) ** 2).sum(axis=2)
        logits = -dist2 / max(reg, 1e-8)
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=1, keepdims=True) + 1e-8
        y_neighbors = y_ref[idx]
        pseudo[start:end] = (y_neighbors * weights[:, :, None]).sum(axis=1)
    return pseudo


def mean_nn_dist_pca(query, ref, k=1, pca_dim=50, seed=0, batch_size=4096, device='cuda'):
    """Mean distance to nearest ref neighbor in PCA space (diagnostic)."""
    Zq, Zr = fit_pca(query, ref, n_components=pca_dim, seed=seed)
    topk_idx, _, _ = batched_euclidean_topk_indices(Zq, Zr, k=k, batch_size=batch_size, device=device)
    dists = []
    for i in range(Zq.shape[0]):
        j = topk_idx[i, 0]
        dists.append(float(np.sqrt(((Zq[i] - Zr[j]) ** 2).sum())))
    return float(np.mean(dists))
