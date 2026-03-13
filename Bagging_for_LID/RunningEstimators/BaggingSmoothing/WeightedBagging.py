from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import sys
from scipy.stats import chi2
import numpy as np
###################################################OWN IMPORT###################################################
from Bagging_for_LID.RunningEstimators.BaggingSmoothing.SimpleBagging import simple_bagging_skdim
from Bagging_for_LID.RunningEstimators.BaggingSmoothing.Smoothing import smoothing
##############################################################################################################################################################################################################################################################
#use_w = 'n', 'y'
def outofbag_weighted_bagging_skdim(estimator, Q, X, n_bags=10, k=10, sampling_rate=None, progress_bar=False, estimators=None,
                         estimator_names=None, paralell_estimation=False, weighing_type='0', use_w='n', t=2, pre_smooth=False,
                                    geo=None, post_smooth=False, error_type='diff', seed = 42):

    rand_gen = np.random.RandomState(seed)
    def compute_distance_matrix(X):
        return squareform(pdist(X))

    def sample_or_split_data_with_indices(X, n_bags=10, sampling_rate=None):
        indices = np.arange(len(X))
        if sampling_rate is None:
            split_indices = np.array_split(indices, n_bags)
            bags = [X[idx] for idx in split_indices]
            out_of_bag_indices = [
                np.concatenate([split_indices[k] for k in range(n_bags) if k != j])
                for j in range(n_bags)
            ]
        else:
            n_samples = np.ceil(sampling_rate * len(X)).astype(int)
            bags = []
            split_indices = []
            out_of_bag_indices = []

            for _ in range(n_bags):
                chosen_idx = rand_gen.choice(indices, size=n_samples, replace=False)
                bags.append(X[chosen_idx])
                split_indices.append(chosen_idx)
                oob_idx = np.setdiff1d(indices, chosen_idx)
                out_of_bag_indices.append(oob_idx)
        return bags, split_indices, out_of_bag_indices

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

    def discrepancy_pvalues(X1, X2, k1, k2, weighing_type = 'p_val_mean'):
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        k1 = np.asarray(k1)
        k2 = np.asarray(k2)
        assert X1.shape == X2.shape, "X1 and X2 must have the same shape"
        shape = X1.shape
        k1 = np.broadcast_to(k1, shape)
        k2 = np.broadcast_to(k2, shape)
        pval_weighted = np.full(shape, np.nan)
        pval_symmetric = np.full(shape, np.nan)
        pval = np.full(shape, np.nan)
        mask_k2_zero = (k2 == 0)
        mask_valid = ~mask_k2_zero
        diff_sq = (X1 - X2) ** 2
        inv_k_sum = np.zeros_like(X1)
        inv_k_sum[mask_valid] = (1.0 / k1[mask_valid]) + (1.0 / k2[mask_valid])
        if weighing_type == 'p_val_mean':
            mu_hat_weighted = np.zeros_like(X1)
            mu_hat_weighted[mask_valid] = (
                    (k1[mask_valid] * X1[mask_valid] + k2[mask_valid] * X2[mask_valid])
                    / (k1[mask_valid] + k2[mask_valid])
            )
            tau_sq_weighted = np.zeros_like(X1)
            tau_sq_weighted[mask_valid] = mu_hat_weighted[mask_valid] ** 2 * inv_k_sum[mask_valid]
            T_weighted = np.zeros_like(X1)
            T_weighted[mask_valid] = diff_sq[mask_valid] / tau_sq_weighted[mask_valid]
            pval_weighted[mask_valid] = 1 - chi2.cdf(T_weighted[mask_valid], df=1)
            pval_weighted[mask_k2_zero] = 0.0
            pval = pval_weighted
        elif weighing_type == 'p_val_symmetric':
            mu_hat_sym = (X1 ** 2 + X2 ** 2) / 2
            tau_sq_sym = mu_hat_sym * inv_k_sum
            T_sym = np.zeros_like(X1)
            T_sym[mask_valid] = diff_sq[mask_valid] / tau_sq_sym[mask_valid]
            pval_symmetric[mask_valid] = 1 - chi2.cdf(T_sym[mask_valid], df=1)
            pval_symmetric[mask_k2_zero] = 0.0
            pval = pval_symmetric
        return pval

    def results_aggregating(estimate_dictionary, estimator_names, out_of_bag_estimate_dictionary, k_1_dict, k_2_dict, weighing_type='0', t=2, error_type='diff'):
        if weighing_type == '0':
            if error_type == 'diff':
                test_errors_dictionary = {estimator_names[i]: np.abs(estimate_dictionary[estimator_names[i]] - out_of_bag_estimate_dictionary[estimator_names[i]])**t for i in range(len(estimator_names))}
            elif error_type == 'log_diff':
                test_errors_dictionary = {estimator_names[i]: np.abs(np.log(estimate_dictionary[estimator_names[i]]) - np.log(out_of_bag_estimate_dictionary[estimator_names[i]]))**t for i in range(len(estimator_names))}
            else:
                NotImplementedError()
            weights_dictionary = {estimator_names[i]: 1 / (test_errors_dictionary[estimator_names[i]] * np.sum(
                1 / test_errors_dictionary[estimator_names[i]], axis=1, keepdims=True)) for i in
                                  range(len(estimator_names))}
            bagging_estimators_dictionary = {estimator_names[i]: np.sum(
                estimate_dictionary[estimator_names[i]] * weights_dictionary[estimator_names[i]], axis=1) for i in
                                             range(len(estimator_names))}
            avg_bagging_estimator_dictionary = {
                estimator_names[i]: np.mean(bagging_estimators_dictionary[estimator_names[i]]) for i in
                range(len(estimator_names))}
        elif weighing_type == 'inf':
            if error_type == 'diff':
                test_errors_dictionary = {estimator_names[i]: np.abs(estimate_dictionary[estimator_names[i]] - out_of_bag_estimate_dictionary[estimator_names[i]]) ** t for i in range(len(estimator_names))}
            elif error_type == 'log_diff':
                test_errors_dictionary = {estimator_names[i]: np.abs(
                    np.log(estimate_dictionary[estimator_names[i]])-np.log(out_of_bag_estimate_dictionary[estimator_names[i]])) ** t
                                          for i in range(len(estimator_names))}
            weights_dictionary = {}
            for i in range(len(estimator_names)):
                mask = np.isfinite(test_errors_dictionary[estimator_names[i]])
                X_safe = np.where(mask, test_errors_dictionary[estimator_names[i]], np.nan)
                row_sums = np.nansum(1 / X_safe, axis=1, keepdims=True)
                Y = np.where(mask, 1 / (test_errors_dictionary[estimator_names[i]] * row_sums), 0)
                all_bad_rows = ~np.any(mask, axis=1, keepdims=True)
                Y[all_bad_rows.repeat(Y.shape[1], axis=1)] = 1.0 / Y.shape[1]
                weights_dictionary[estimator_names[i]] = Y
            bagging_estimators_dictionary = {estimator_names[i]: np.sum(estimate_dictionary[estimator_names[i]] * weights_dictionary[estimator_names[i]], axis=1) for i in range(len(estimator_names))}
            avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimators_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
        elif weighing_type == 'equalizing':
            test_errors_dictionary = {estimator_names[i]: 1/(1/k_1_dict[estimator_names[i]] + 1/k_2_dict[estimator_names[i]])*np.abs(estimate_dictionary[estimator_names[i]] - out_of_bag_estimate_dictionary[estimator_names[i]]) ** 2 for i in range(len(estimator_names))}
            weights_dictionary = {}
            for i in range(len(estimator_names)):
                mask = np.isfinite(test_errors_dictionary[estimator_names[i]])
                X_safe = np.where(mask, test_errors_dictionary[estimator_names[i]], np.nan)
                row_sums = np.nansum(1 / X_safe, axis=1, keepdims=True)
                Y = np.where(mask, 1 / (test_errors_dictionary[estimator_names[i]] * row_sums), 0)
                all_bad_rows = ~np.any(mask, axis=1, keepdims=True)
                Y[all_bad_rows.repeat(Y.shape[1], axis=1)] = 1.0 / Y.shape[1]
                weights_dictionary[estimator_names[i]] = Y
            bagging_estimators_dictionary = {estimator_names[i]: np.sum(estimate_dictionary[estimator_names[i]] * weights_dictionary[estimator_names[i]], axis=1) for i in range(len(estimator_names))}
            avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimators_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
        elif weighing_type == 'penalizing':
            test_errors_dictionary = {estimator_names[i]: (1 / k_1_dict[estimator_names[i]] + 1 / k_2_dict[estimator_names[i]]) * np.abs(estimate_dictionary[estimator_names[i]] - out_of_bag_estimate_dictionary[estimator_names[i]]) ** 2 for i in range(len(estimator_names))}
            weights_dictionary = {}
            for i in range(len(estimator_names)):
                mask = np.isfinite(test_errors_dictionary[estimator_names[i]])
                X_safe = np.where(mask, test_errors_dictionary[estimator_names[i]], np.nan)
                row_sums = np.nansum(1 / X_safe, axis=1, keepdims=True)
                Y = np.where(mask, 1 / (test_errors_dictionary[estimator_names[i]] * row_sums), 0)
                all_bad_rows = ~np.any(mask, axis=1, keepdims=True)
                Y[all_bad_rows.repeat(Y.shape[1], axis=1)] = 1.0 / Y.shape[1]
                weights_dictionary[estimator_names[i]] = Y
            bagging_estimators_dictionary = {estimator_names[i]: np.sum(estimate_dictionary[estimator_names[i]] * weights_dictionary[estimator_names[i]], axis=1) for i in range(len(estimator_names))}
            avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimators_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
        elif weighing_type == 'p_val_mean':
            probs_dictionary = {estimator_names[i]: -1/np.log(discrepancy_pvalues(estimate_dictionary[estimator_names[i]], out_of_bag_estimate_dictionary[estimator_names[i]], k_1_dict[estimator_names[i]], k_2_dict[estimator_names[i]], weighing_type='p_val_mean')) for i in range(len(estimator_names))}
            clipped_probs_dictionary = {estimator_names[i]: np.maximum(probs_dictionary[estimator_names[i]], np.max(probs_dictionary[estimator_names[i]], axis=1, keepdims=True)/100) for i in range(len(estimator_names))}
            weights_dictionary = {estimator_names[i]: clipped_probs_dictionary[estimator_names[i]]/np.sum(clipped_probs_dictionary[estimator_names[i]], axis=1, keepdims=True) for i in range(len(estimator_names))}
            bagging_estimators_dictionary = {estimator_names[i]: np.sum(estimate_dictionary[estimator_names[i]] * weights_dictionary[estimator_names[i]], axis=1) for i in range(len(estimator_names))}
            avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimators_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
        elif weighing_type == 'p_val_raw':
            clipped_probs_dictionary = {estimator_names[i]: discrepancy_pvalues(estimate_dictionary[estimator_names[i]], out_of_bag_estimate_dictionary[estimator_names[i]], k_1_dict[estimator_names[i]], k_2_dict[estimator_names[i]], weighing_type='p_val_mean') for i in range(len(estimator_names))}
            #clipped_probs_dictionary = {estimator_names[i]: np.maximum(probs_dictionary[estimator_names[i]], np.max(probs_dictionary[estimator_names[i]], axis=1, keepdims=True)/100) for i in range(len(estimator_names))}
            weights_dictionary = {estimator_names[i]: clipped_probs_dictionary[estimator_names[i]]/np.sum(clipped_probs_dictionary[estimator_names[i]], axis=1, keepdims=True) for i in range(len(estimator_names))}
            bagging_estimators_dictionary = {estimator_names[i]: np.sum(estimate_dictionary[estimator_names[i]] * weights_dictionary[estimator_names[i]], axis=1) for i in range(len(estimator_names))}
            avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimators_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
        elif weighing_type == 'p_val_mean2':
            clipped_probs_dictionary = {estimator_names[i]: -1/np.log(discrepancy_pvalues(estimate_dictionary[estimator_names[i]], out_of_bag_estimate_dictionary[estimator_names[i]], k_1_dict[estimator_names[i]], k_2_dict[estimator_names[i]], weighing_type='p_val_mean')) for i in range(len(estimator_names))}
            #clipped_probs_dictionary = {estimator_names[i]: probs_dictionary[estimator_names[i]], np.max(probs_dictionary[estimator_names[i]], axis=1, keepdims=True)/10) for i in range(len(estimator_names))}
            weights_dictionary = {estimator_names[i]: clipped_probs_dictionary[estimator_names[i]]/np.sum(clipped_probs_dictionary[estimator_names[i]], axis=1, keepdims=True) for i in range(len(estimator_names))}
            bagging_estimators_dictionary = {estimator_names[i]: np.sum(estimate_dictionary[estimator_names[i]] * weights_dictionary[estimator_names[i]], axis=1) for i in range(len(estimator_names))}
            avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimators_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
        elif weighing_type == 'p_val_symmetric':
            probs_dictionary = {estimator_names[i]: -1/np.log(discrepancy_pvalues(estimate_dictionary[estimator_names[i]], out_of_bag_estimate_dictionary[estimator_names[i]], k_1_dict[estimator_names[i]], k_2_dict[estimator_names[i]], weighing_type='p_val_symmetric')) for i in range(len(estimator_names))}
            clipped_probs_dictionary = {estimator_names[i]: np.maximum(probs_dictionary[estimator_names[i]], np.max(probs_dictionary[estimator_names[i]], axis=1, keepdims=True)/100) for i in range(len(estimator_names))}
            weights_dictionary = {estimator_names[i]: clipped_probs_dictionary[estimator_names[i]]/np.sum(clipped_probs_dictionary[estimator_names[i]], axis=1, keepdims=True) for i in range(len(estimator_names))}
            bagging_estimators_dictionary = {estimator_names[i]: np.sum(estimate_dictionary[estimator_names[i]] * weights_dictionary[estimator_names[i]], axis=1) for i in range(len(estimator_names))}
            avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimators_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
        elif weighing_type == 'p_val_symmetric_raw':
            clipped_probs_dictionary = {estimator_names[i]: discrepancy_pvalues(estimate_dictionary[estimator_names[i]], out_of_bag_estimate_dictionary[estimator_names[i]], k_1_dict[estimator_names[i]], k_2_dict[estimator_names[i]], weighing_type='p_val_symmetric') for i in range(len(estimator_names))}
            #clipped_probs_dictionary = {estimator_names[i]: np.maximum(probs_dictionary[estimator_names[i]], np.max(probs_dictionary[estimator_names[i]], axis=1, keepdims=True)/100) for i in range(len(estimator_names))}
            weights_dictionary = {estimator_names[i]: clipped_probs_dictionary[estimator_names[i]]/np.sum(clipped_probs_dictionary[estimator_names[i]], axis=1, keepdims=True) for i in range(len(estimator_names))}
            bagging_estimators_dictionary = {estimator_names[i]: np.sum(estimate_dictionary[estimator_names[i]] * weights_dictionary[estimator_names[i]], axis=1) for i in range(len(estimator_names))}
            avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimators_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
        return bagging_estimators_dictionary, avg_bagging_estimator_dictionary

    n, m = Q.shape[0], Q.shape[1]
    distances = compute_distance_matrix(X)
    #sorted_distances, sorted_indices = precompute_sorted_distances(distances)
    bags, split_indices, out_of_bag_indices = sample_or_split_data_with_indices(X, n_bags=n_bags, sampling_rate=sampling_rate)
    if not paralell_estimation:
        estimates = np.zeros((n, n_bags))
        out_of_bag_estimates = np.zeros((n, n_bags))
        kth_dists = np.zeros((n, n_bags))
        if progress_bar:
            for j in tqdm(range(n_bags)):
                smallest_distances, smallest_indices = k_smallest_distance_Q(distances=distances,
                                                                             indices=split_indices[j],
                                                                             k=k)
                estimates[:, j] = estimator(X=X, dists=smallest_distances, knnidx=smallest_indices, k=k)[0]
                out_of_bag_indices = np.concatenate([split_indices[k] for k in range(len(split_indices)) if k != j])
                out_of_bag_smallest_distances, out_of_bag_smallest_indices = k_smallest_distance_Q(distances=distances,
                                                                                                   indices=out_of_bag_indices,
                                                                                                   k=k)
                out_of_bag_estimates[:, j] = \
                estimator(X=X, dists=out_of_bag_smallest_distances, knnidx=out_of_bag_smallest_indices, k=k)[0]
        else:
            for j in range(n_bags):
                smallest_distances, smallest_indices = k_smallest_distance_Q(distances=distances,
                                                                             indices=split_indices[j],
                                                                             k=k)
                kth_dists[:, j] = np.max(smallest_distances, axis = 1)
                estimates[:, j] = estimator(X=X, dists=smallest_distances, knnidx=smallest_indices, k=k)[0]
                out_of_bag_indices = np.concatenate([split_indices[k] for k in range(len(split_indices)) if k != j])
                out_of_bag_smallest_distances, out_of_bag_smallest_indices = k_smallest_distance_Q(distances=distances,
                                                                                                   indices=out_of_bag_indices,
                                                                                                   k=k)
                out_of_bag_estimates[:, j] = \
                estimator(X=X, dists=out_of_bag_smallest_distances, knnidx=out_of_bag_smallest_indices, k=k)[0]
        test_errors = np.abs(estimates - out_of_bag_estimates)
        weights = 1 / (test_errors * np.sum(1 / test_errors, axis=1, keepdims=True))
        bagging_estimators = np.sum(estimates * weights, axis=1)
        avg_bagging_estimator = np.mean(bagging_estimators)
        return bagging_estimators, avg_bagging_estimator
    else:
        estimator_dictionary = {estimator_names[i]: estimators[i] for i in range(len(estimator_names))}
        estimate_dictionary = {estimator_names[i]: np.zeros((n, n_bags)) for i in range(len(estimator_names))}
        k_1_dict = {estimator_names[i]: k*np.ones((n, n_bags)) for i in range(len(estimator_names))}
        k_2_dict = {estimator_names[i]: np.zeros((n, n_bags)) for i in range(len(estimator_names))}
        out_of_bag_estimate_dictionary = {estimator_names[i]: np.zeros((n, n_bags)) for i in range(len(estimator_names))}
        if progress_bar:
            for j in tqdm(range(n_bags)):
                smallest_distances, smallest_indices = k_smallest_distance_Q(distances=distances, indices=split_indices[j],
                                                                             k=k)
                bag_ws = smallest_distances[:, -1]
                out_of_bag_smallest_distances, out_of_bag_smallest_indices = k_smallest_distance_Q(distances=distances, indices=out_of_bag_indices[j],
                                                                             k=k, w=bag_ws)
                for key in estimator_dictionary:
                    estimate_dictionary[key][:, j] = \
                    estimator_dictionary[key](X=X, dists=smallest_distances, knnidx=smallest_indices, k=k)[0]
                    out_of_bag_e = estimator_dictionary[key](X=X, dists=out_of_bag_smallest_distances, knnidx=out_of_bag_smallest_indices, k=k, w=bag_ws)[0]
                    masks = np.isnan(out_of_bag_e)
                    out_of_bag_e[masks] = 0
                    out_of_bag_estimate_dictionary[key][:, j] = out_of_bag_e
        else:
            for j in range(n_bags):
                smallest_distances, smallest_indices = k_smallest_distance_Q(distances=distances, indices=split_indices[j],
                                                                             k=k)
                if use_w == 'n':
                    out_of_bag_smallest_distances, out_of_bag_smallest_indices = k_smallest_distance_Q(distances=distances, indices=out_of_bag_indices[j], k=k)
                elif use_w == 'y':
                    bag_ws = smallest_distances[:, -1]
                    out_of_bag_smallest_distances, out_of_bag_smallest_indices = k_smallest_distance_Q(distances=distances, indices=out_of_bag_indices[j], k=k, w=bag_ws)
                else:
                    out_of_bag_smallest_distances, out_of_bag_smallest_indices = k_smallest_distance_Q(distances=distances, indices=out_of_bag_indices[j], k=int(k/sampling_rate))

                if sum([len(out_of_bag_smallest_distances[i]) < 2 for i in range(len(out_of_bag_smallest_indices))]) > 0:
                    Warning('Less than 2 samples were found in an out-of-bag, may cause estimation issues')
                for key in estimator_dictionary:
                    estimate_dictionary[key][:, j] = estimator_dictionary[key](X=X, dists=smallest_distances, knnidx=smallest_indices, k=k, smooth=pre_smooth, geo=geo)[0]
                    if use_w == 'n':
                        out_of_bag_e, _, ks = estimator_dictionary[key](X=X, dists=out_of_bag_smallest_distances, knnidx=out_of_bag_smallest_indices, k=k, return_ks=True, smooth=pre_smooth, geo=geo, w=None)
                    elif use_w == 'y':
                        out_of_bag_e, _, ks = estimator_dictionary[key](X=X, dists=out_of_bag_smallest_distances, knnidx=out_of_bag_smallest_indices, k=k, w=bag_ws, return_ks=True)
                    else:
                        out_of_bag_e, _, ks = estimator_dictionary[key](X=X, dists=out_of_bag_smallest_distances,knnidx=out_of_bag_smallest_indices, k=int(k/sampling_rate), return_ks=True, w=None)
                    k_2_dict[key][:, j] = ks
                    masks = ~np.isfinite(out_of_bag_e)
                    if weighing_type == '0':
                        out_of_bag_e[masks] = 0
                        out_of_bag_estimate_dictionary[key][:, j] = out_of_bag_e
                    else:
                        out_of_bag_estimate_dictionary[key][:, j] = out_of_bag_e
        bagging_estimators_dictionary, avg_bagging_estimator_dictionary = results_aggregating(estimate_dictionary=estimate_dictionary,
                                                                                              estimator_names=estimator_names,
                                                                                              out_of_bag_estimate_dictionary=out_of_bag_estimate_dictionary,
                                                                                              k_1_dict=k_1_dict, k_2_dict=k_2_dict, weighing_type=weighing_type, t=t, error_type=error_type)
        if post_smooth:
            for i in range(len(estimator_names)):
                bagging_estimators_dictionary[estimator_names[i]], _ = smoothing(X, bagging_estimators_dictionary[estimator_names[i]], k=k, dists=None, knnidx=None, geo=geo)
            avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimators_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
        return bagging_estimators_dictionary, avg_bagging_estimator_dictionary

def outofbag_weighted_inside_bagging_skdim(estimator, Q, X, n_bags=10, k=10, sampling_rate=None, progress_bar=False, estimators=None,
                         estimator_names=None, paralell_estimation=False, t = 2, pre_smooth=False, geo=None, post_smooth=False, error_type='diff', seed = 42):
    rand_gen = np.random.RandomState(seed)
    def compute_distance_matrix(X):
        return squareform(pdist(X))

    def sample_or_split_data_with_indices(X, n_bags=10, sampling_rate=None):
        indices = np.arange(len(X))
        if sampling_rate is None:
            split_indices = np.array_split(indices, n_bags)
            bags = [X[idx] for idx in split_indices]
            out_of_bag_indices = [
                np.concatenate([split_indices[k] for k in range(n_bags) if k != j])
                for j in range(n_bags)
            ]
        else:
            n_samples = np.ceil(sampling_rate * len(X)).astype(int)
            bags = []
            split_indices = []
            out_of_bag_indices = []
            for _ in range(n_bags):
                chosen_idx = rand_gen.choice(indices, size=n_samples, replace=False)
                bags.append(X[chosen_idx])
                split_indices.append(chosen_idx)
                oob_idx = np.setdiff1d(indices, chosen_idx)
                out_of_bag_indices.append(oob_idx)
        return bags, split_indices, out_of_bag_indices

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
    distances = compute_distance_matrix(X)
    bags, split_indices, out_of_bag_indices = sample_or_split_data_with_indices(X, n_bags=n_bags, sampling_rate=sampling_rate)
    estimator_dictionary = {estimator_names[i]: estimators[i] for i in range(len(estimator_names))}
    estimate_dictionary = {estimator_names[i]: np.zeros((n, n_bags)) for i in range(len(estimator_names))}
    out_of_bag_estimate_dictionary = {estimator_names[i]: np.zeros((n, n_bags)) for i in range(len(estimator_names))}
    for j in range(n_bags):
        smallest_distances, smallest_indices = k_smallest_distance_Q(distances=distances, indices=split_indices[j],
                                                                     k=k)
        bag_ws = smallest_distances[:, -1]
        for key in estimator_dictionary:
            estimate_dictionary[key][:, j] = \
            estimator_dictionary[key](X=X, dists=smallest_distances, knnidx=smallest_indices, k=k, smooth=pre_smooth, geo=geo)[0]
            out_of_bag_e = simple_bagging_skdim(estimator=estimator_dictionary[key], Q=X, X=X, n_bags=n_bags, k=k, sampling_rate=sampling_rate/(1-sampling_rate), progress_bar=False, estimators=None, estimator_names=None, paralell_estimation=False, w=None, indexuse=out_of_bag_indices[j], pre_smooth=pre_smooth, geo=geo)[0]
            masks = ~np.isfinite(out_of_bag_e)
            out_of_bag_e[masks] = 0
            out_of_bag_estimate_dictionary[key][:, j] = out_of_bag_e
    if error_type == 'diff':
        test_errors_dictionary = {estimator_names[i]: np.abs(estimate_dictionary[estimator_names[i]] - out_of_bag_estimate_dictionary[estimator_names[i]])**t for i in range(len(estimator_names))}
        weights_dictionary = {estimator_names[i]: 1 / (test_errors_dictionary[estimator_names[i]] * np.sum(1 / test_errors_dictionary[estimator_names[i]], axis=1, keepdims=True)) for i in range(len(estimator_names))}
        bagging_estimators_dictionary = {estimator_names[i]: np.sum(estimate_dictionary[estimator_names[i]] * weights_dictionary[estimator_names[i]], axis=1) for i in range(len(estimator_names))}
        avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimators_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
    elif error_type == 'log_diff':
        test_errors_dictionary = {estimator_names[i]: np.abs(np.log(estimate_dictionary[estimator_names[i]]) - np.log(out_of_bag_estimate_dictionary[estimator_names[i]]))**t for i in range(len(estimator_names))}
        weights_dictionary = {estimator_names[i]: 1 / (test_errors_dictionary[estimator_names[i]] * np.sum(1 / test_errors_dictionary[estimator_names[i]], axis=1, keepdims=True)) for i in range(len(estimator_names))}
        bagging_estimators_dictionary = {estimator_names[i]: np.sum(estimate_dictionary[estimator_names[i]] * weights_dictionary[estimator_names[i]], axis=1) for i in range(len(estimator_names))}
        avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimators_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
    else:
        NotImplementedError()
    if post_smooth:
        for i in range(len(estimator_names)):
            bagging_estimators_dictionary[estimator_names[i]], _ = smoothing(X, bagging_estimators_dictionary[
                estimator_names[i]], k=k, dists=None, knnidx=None, geo=geo)
        avg_bagging_estimator_dictionary = {estimator_names[i]: np.mean(bagging_estimators_dictionary[estimator_names[i]]) for i in range(len(estimator_names))}
    return bagging_estimators_dictionary, avg_bagging_estimator_dictionary

###########################################################################################LIDkit#####################################################################################################
