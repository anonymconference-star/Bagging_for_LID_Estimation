import skdim
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import numpy as np
##############################################################################################################################################################################################################################################################

def geodesic_knn(X, k_euc, n_geo):
    n_samples = X.shape[0]
    dists, knnidx = skdim._commonfuncs.get_nn(X, k=k_euc, n_jobs=1)
    row_idx = np.repeat(np.arange(n_samples), k_euc)
    col_idx = knnidx.flatten()
    data = dists.flatten()
    A = csr_matrix((data, (row_idx, col_idx)), shape=(n_samples, n_samples))
    A = A.maximum(A.T)
    dist_matrix = dijkstra(csgraph=A, directed=False, return_predecessors=False)
    geodesic_indices = []
    geodesic_dists = []
    for i in range(n_samples):
        dists_i = dist_matrix[i]
        dists_i[i] = np.inf  # ignore self
        finite_mask = np.isfinite(dists_i)
        reachable = np.where(finite_mask)[0]
        sorted_idx = np.argsort(dists_i[reachable])[:n_geo]
        selected = reachable[sorted_idx]
        selected_dists = dists_i[selected]
        geodesic_indices.append(selected.tolist())
        geodesic_dists.append(selected_dists.tolist())
    return geodesic_dists, geodesic_indices

def smoothing(X, lid_estimates, k=10, dists=None, knnidx=None, geo=None,
              smooth_style='code2', bag_indices=None):
    if smooth_style == "code1":
        if (dists is None) or (knnidx is None):
            if geo is None:
                dists, knnidx = skdim._commonfuncs.get_nn(X, k=k, n_jobs=1)
            else:
                dists, knnidx = geodesic_knn(X, k_euc=geo, n_geo=k)
        smoothed_estimates = np.empty(len(lid_estimates))
        for i in range(len(lid_estimates)):
            if np.isin(i, knnidx[i]):
                raise ValueError("KNN indices contains the query index")
            smoothed_estimates[i] = (np.sum(lid_estimates[knnidx[i,:]]) + lid_estimates[i]) / (k+1)
        return smoothed_estimates, np.mean(smoothed_estimates)
    elif smooth_style != "code1" and knnidx is None and bag_indices is None:
        if (dists is None) or (knnidx is None):
            if geo is None:
                dists, knnidx = skdim._commonfuncs.get_nn(X, k=k, n_jobs=1)
            else:
                dists, knnidx = geodesic_knn(X, k_euc=geo, n_geo=k)
        smoothed_estimates = np.empty(len(lid_estimates))
        for i in range(len(lid_estimates)):
            if np.isin(i, knnidx[i]):
                raise ValueError("KNN indices contains the query index")
            smoothed_estimates[i] = (np.sum(lid_estimates[knnidx[i, :-1]]) + lid_estimates[i])/k
        return smoothed_estimates, np.mean(smoothed_estimates)
    elif smooth_style != "code1":
        if knnidx is None:
            raise ValueError("No bag-specific knnidx are provided.")
        if bag_indices is None:
            raise ValueError("No bag indices are provided.")

        n = len(lid_estimates)
        smoothed_estimates = np.empty(n)
        in_bag = np.zeros(n, dtype=bool)
        in_bag[np.asarray(bag_indices, dtype=int)] = True
        for i in range(n):
            nbrs = knnidx[i]
            if in_bag[i]:
                idx = np.concatenate(([i], nbrs[:max(k-1, 0)]))
            else:
                idx = nbrs[:k]
            smoothed_estimates[i] = np.mean(lid_estimates[idx])
        return smoothed_estimates, np.mean(smoothed_estimates)