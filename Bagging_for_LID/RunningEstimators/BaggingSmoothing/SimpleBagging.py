from scipy.spatial.distance import squareform, pdist
###################################################OWN IMPORT###################################################
from Bagging_for_LID.RunningEstimators.BaseEstimators import *
##############################################################################################################################################################################################################################################################
def simple_bagging_skdim(estimator, Q, X, n_bags=10, k=10, sampling_rate=None,
                         progress_bar=False, estimators=None, estimator_names=None,
                         paralell_estimation=False, w=None, indexuse=None, pre_smooth=False,
                         geo=None, post_smooth=False, seed=42, smooth_style='code2'):
    rand_gen = np.random.RandomState(seed)
    def compute_distance_matrix(X):
        return squareform(pdist(X))

    def split_data_with_indices(X, n_bags=10):
        indices = np.random.permutation(len(X))
        if indexuse is not None:
            indices = np.intersect1d(indices, indexuse)
        split_indices = np.array_split(indices, n_bags)
        bags = [X[idx] for idx in split_indices]
        return bags, split_indices

    def sample_data_with_indices(X, n_bags=10, sampling_rate=0.8):
        n_samples = np.ceil(sampling_rate * len(X)).astype(int)
        indices = np.arange(len(X))
        bags = []
        selected_indices = []
        for _ in range(n_bags):
            chosen_idx = rand_gen.choice(indices, size=n_samples, replace=False)
            bags.append(X[chosen_idx])
            selected_indices.append(chosen_idx)
        return bags, selected_indices

    def k_smallest_nonzero_Q(distances, k=10, w=None):
        n = distances.shape[0]
        if w is not None:
            w = np.asarray(w)
            result_distances = []
            result_indices = []
            for q in range(n):
                nonzero_mask = distances[q] != 0
                valid_mask = nonzero_mask & (distances[q] <= w[q])
                dists = distances[q, valid_mask]
                inds = np.nonzero(valid_mask)[0]
                order = np.argsort(dists)
                result_distances.append(dists[order])
                result_indices.append(inds[order])
            return result_distances, result_indices
        else:
            result_distances = np.zeros((n, k))
            result_indices = np.zeros((n, k), dtype=int)
            for q in range(n):
                nonzero_mask = distances[q] != 0
                dists = distances[q, nonzero_mask]
                inds = np.nonzero(nonzero_mask)[0]
                if len(dists) >= k:
                    partition_indices = np.argpartition(dists, k)[:k]
                    sorted_indices = partition_indices[np.argsort(dists[partition_indices])]
                    result_distances[q, :] = dists[sorted_indices]
                    result_indices[q, :] = inds[sorted_indices]
                else:
                    raise ValueError("There are less nonzero distances than the given k")
            return result_distances, result_indices

    def k_smallest_distance_Q(distances, indices, k=10, w=None):
        considered_distances = distances[:, indices]
        smallest_distances, smallest_indices = k_smallest_nonzero_Q(considered_distances, k, w=w)
        if w is not None:
            smallest_indices = [indices[row_inds] for row_inds in smallest_indices]
        else:
            smallest_indices = indices[smallest_indices]
        return smallest_distances, smallest_indices

    n, m = Q.shape[0], Q.shape[1]
    distances = compute_distance_matrix(Q)
    #sorted_distances, sorted_indices = precompute_sorted_distances(distances)
    if sampling_rate is None:
        bags, split_indices = split_data_with_indices(X, n_bags=n_bags)
    else:
        bags, split_indices = sample_data_with_indices(X, n_bags=n_bags, sampling_rate=sampling_rate)
    if not paralell_estimation:
        estimates = np.zeros((n, n_bags))
        if progress_bar:
            for j in tqdm(range(n_bags)):
                smallest_distances, smallest_indices = k_smallest_distance_Q(distances=distances, indices=split_indices[j],
                                                                             k=k, w=w)
                #smallest_distances, smallest_indices = fast_k_smallest_in_bag(sorted_distances, sorted_indices, split_indices[j], k=k)
                estimates[:, j] = estimator(X=X, dists=smallest_distances, knnidx=smallest_indices, k=k, w=w)[0]
        else:
            for j in range(n_bags):
                smallest_distances, smallest_indices = k_smallest_distance_Q(distances=distances, indices=split_indices[j],
                                                                             k=k, w=w)
                #smallest_distances, smallest_indices = fast_k_smallest_in_bag(sorted_distances, sorted_indices,
                #                                                              split_indices[j], k=k)
                estimates[:, j] = estimator(X=X, dists=smallest_distances, knnidx=smallest_indices, k=k, w=w)[0]
        bagging_estimators = np.mean(estimates, axis=1)
        avg_bagging_estimator = np.mean(bagging_estimators)
        return bagging_estimators, avg_bagging_estimator
    else:
        estimator_dictionary = {estimator_names[i]: estimators[i] for i in range(len(estimator_names))}
        estimate_dictionary = {estimator_names[i]: np.zeros((n, n_bags)) for i in range(len(estimator_names))}
        if progress_bar:
            for j in tqdm(range(n_bags)):
                smallest_distances, smallest_indices = k_smallest_distance_Q(distances=distances, indices=split_indices[j],
                                                                             k=k, w=w)
                #smallest_distances, smallest_indices = fast_k_smallest_in_bag(sorted_distances, sorted_indices,
                #                                                              split_indices[j], k=k)
                for key in estimator_dictionary:
                    estimate_dictionary[key][:, j] = estimator_dictionary[key](X=X, dists=smallest_distances, knnidx=smallest_indices, k=k, w=w)[0]
        else:
            for j in range(n_bags):
                smallest_distances, smallest_indices = k_smallest_distance_Q(distances=distances, indices=split_indices[j],
                                                                             k=k, w=w)
                #print(n_bags)
                #print(smallest_distances.shape)
                #print(smallest_distances[0:5])
                #print(f'OG_smallest_distances_{smallest_distances.shape}')
                #print(f'OG_mean_smallest_distances_{smallest_distances.mean()}')
                #smallest_distances, smallest_indices = fast_k_smallest_in_bag(sorted_distances, sorted_indices,
                #                                                              split_indices[j], k=k)
                for key in estimator_dictionary:
                    if smooth_style != "code1":
                        estimate_dictionary[key][:, j] = estimator_dictionary[key](X=X, dists=smallest_distances, knnidx=smallest_indices, k=k, w=w, smooth=pre_smooth, geo=geo, smooth_style=smooth_style, bag_indices=split_indices[j])[0]
                    else:
                        estimate_dictionary[key][:, j] = \
                        estimator_dictionary[key](X=X, dists=smallest_distances, knnidx=smallest_indices, k=k, w=w, smooth=pre_smooth, geo=geo, smooth_style=smooth_style)[0]
                    #print(f'original_bag_estimate_for_bag_{j}: {np.mean(estimate_dictionary[key][:, j])}')
        bagging_estimator_dictionary = {estimator_names[i]: np.mean(estimate_dictionary[estimator_names[i]], axis=1) for i in range(len(estimator_names))}
        avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimator_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
        if post_smooth:
            for i in range(len(estimator_names)):
                bagging_estimator_dictionary[estimator_names[i]], _ = smoothing(X, bagging_estimator_dictionary[estimator_names[i]], k=k, dists=None, knnidx=None, geo=geo)
            avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimator_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
        return bagging_estimator_dictionary, avg_bagging_estimator_dictionary

def knn_distances(X, k=10):
    dists, knnidx = skdim._commonfuncs.get_nn(X, k=k, n_jobs=1)
    n = dists.shape[0]
    knn_distance_array = np.zeros((n, k, 1))
    knn_distance_array[:,:,0] = dists
    return knn_distance_array

def knn_distances_bagging(Q, X, n_bags=10, k=10, sampling_rate=None, w=None, indexuse=None, seed=42):
    rand_gen = np.random.RandomState(seed)

    def compute_distance_matrix(X):
        return squareform(pdist(X))

    def split_data_with_indices(X, n_bags=10):
        indices = np.random.permutation(len(X))
        if indexuse is not None:
            indices = np.intersect1d(indices, indexuse)
        split_indices = np.array_split(indices, n_bags)
        bags = [X[idx] for idx in split_indices]
        return bags, split_indices

    def sample_data_with_indices(X, n_bags=10, sampling_rate=0.8):
        n_samples = np.ceil(sampling_rate * len(X)).astype(int)
        indices = np.arange(len(X))
        bags = []
        selected_indices = []
        for _ in range(n_bags):
            chosen_idx = rand_gen.choice(indices, size=n_samples, replace=False)
            bags.append(X[chosen_idx])
            selected_indices.append(chosen_idx)
        return bags, selected_indices

    def k_smallest_nonzero_Q(distances, k=10, w=None):
        n = distances.shape[0]
        if w is not None:
            w = np.asarray(w)
            result_distances = []
            result_indices = []
            for q in range(n):
                nonzero_mask = distances[q] != 0
                valid_mask = nonzero_mask & (distances[q] <= w[q])
                dists = distances[q, valid_mask]
                inds = np.nonzero(valid_mask)[0]
                order = np.argsort(dists)
                result_distances.append(dists[order])
                result_indices.append(inds[order])
            return result_distances, result_indices
        else:
            result_distances = np.zeros((n, k))
            result_indices = np.zeros((n, k), dtype=int)
            for q in range(n):
                nonzero_mask = distances[q] != 0
                dists = distances[q, nonzero_mask]
                inds = np.nonzero(nonzero_mask)[0]
                if len(dists) >= k:
                    partition_indices = np.argpartition(dists, k)[:k]
                    sorted_indices = partition_indices[np.argsort(dists[partition_indices])]
                    result_distances[q, :] = dists[sorted_indices]
                    result_indices[q, :] = inds[sorted_indices]
                else:
                    raise ValueError("There are less nonzero distances than the given k")
            return result_distances, result_indices

    def k_smallest_distance_Q(distances, indices, k=10, w=None):
        considered_distances = distances[:, indices]
        smallest_distances, smallest_indices = k_smallest_nonzero_Q(considered_distances, k, w=w)
        if w is not None:
            smallest_indices = [indices[row_inds] for row_inds in smallest_indices]
        else:
            smallest_indices = indices[smallest_indices]
        return smallest_distances, smallest_indices

    n, m = Q.shape[0], Q.shape[1]
    distances = compute_distance_matrix(Q)

    if sampling_rate is None:
        bags, split_indices = split_data_with_indices(X, n_bags=n_bags)
    else:
        bags, split_indices = sample_data_with_indices(X, n_bags=n_bags, sampling_rate=sampling_rate)

    knn_distance_array = np.zeros((n, k, n_bags))

    for j in range(n_bags):
        smallest_distances, smallest_indices = k_smallest_distance_Q(distances=distances, indices=split_indices[j],
                                                                     k=k, w=w)
        knn_distance_array[:, :, j] = smallest_distances

    return knn_distance_array