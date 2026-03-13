import numpy as np
###############################################################################################################################MEASURES###############################################################################################################################
def get_comparrison_measures(data_sets, estimators):
    result = dict([(key, [np.mean(estimators[key][0]), np.mean((estimators[key][0] - data_sets[key][1]) ** 2),
                          np.mean(estimators[key][0] - data_sets[key][1]) ** 2, np.var(estimators[key][0])]) for key in
                   data_sets])
    return result

def subset_estimates(data_set, data_set_name, estimators, test_types=['dim'], bounds = None):
    existing_dims = np.unique(data_set[1])
    subset_data = {data_set_name: [[data_set[0], data_set[1], data_set[2]], estimators[0]]}
    X = data_set[0]
    for test in test_types:
        if test == 'dim':
            subset = {f'{data_set_name}_dim_{dim}' : [[data_set[0][data_set[1] == dim], data_set[1][data_set[1] == dim], data_set[2]], estimators[0][data_set[1] == dim]] for dim in existing_dims}
            subset_data = subset_data | subset
        if test == 'region':
            mask = np.ones(X.shape[0], dtype=bool)
            for j in range(len(bounds)):
                for coord, (low, high) in bounds[j].items():
                    mask &= (X[:, coord] >= low) & (X[:, coord] <= high)
                inbound_indices = np.where(mask)[0]
                outbound_indices = np.where(~mask)[0]
                subset = {f'{data_set_name}_inregion_{j}': [[data_set[0][inbound_indices], data_set[1][inbound_indices], data_set[2]], estimators[0][inbound_indices]]}
                #subset[f'{data_set_name}_outregion_{j}'] = [[data_set[0][outbound_indices], data_set[1][outbound_indices], data_set[2]], estimators[0][outbound_indices]]
                subset_data = subset_data | subset
    return subset_data

def true_data_comparrison_measures(data_set, estimates, log_comparrison=False):
    n = len(data_set[1])
    existing_dims = np.unique(data_set[1])
    diffdims = len(existing_dims)
    subsets = {existing_dims[i]: [estimates[data_set[1] == existing_dims[i]], data_set[1][data_set[1] == existing_dims[i]]] for i in range(diffdims)}
    if not log_comparrison:
        subset_measures = {key: [np.mean(subsets[key][0]), np.mean((subsets[key][0]-subsets[key][1])**2), np.mean(subsets[key][0]-subsets[key][1])**2, np.var(subsets[key][0])] for key in subsets}
    else:
        subset_measures = {key: [np.mean(subsets[key][0]), np.mean((np.log(subsets[key][0])-np.log(subsets[key][1]))**2), np.mean(np.log(subsets[key][0])-np.log(subsets[key][1]))**2, np.var(np.log(subsets[key][0]))] for key in subsets}
    subset_offset = {key: len(subsets[key][0])/n for key in subsets}
    subset_corrected = {key: [subset_measures[key][0], subset_measures[key][1]*subset_offset[key], subset_measures[key][2]*subset_offset[key], subset_measures[key][3]*subset_offset[key]] for key in subsets}
    total_mse = np.sum([subset_corrected[key][1] for key in subset_corrected])
    total_bias2 = np.sum([subset_corrected[key][2] for key in subset_corrected])
    total_var = np.sum([subset_corrected[key][3] for key in subset_corrected])
    return total_mse, total_bias2, total_var, subset_measures, subsets

def add_spatial_subset(data_set, estimates, bounds, log_comparrison=False):
    n = len(data_set[1])
    existing_dims = np.unique(data_set[1])
    diffdims = len(existing_dims)
    name = 'spatial_subset_'
    for i in range(len(bounds)):
        name = name + f'_x{i}_{bounds[i][0]}-{bounds[i][1]}_'
    bounds_mask = [True for i in range(n)]
    X = data_set[0]
    for i in range(n):
        for j in range(len(bounds)):
            if X[i][j] < bounds[j][0] or X[i][j] > bounds[j][1]:
                bounds_mask[i] = False
    spatial_subset = {name: [estimates[bounds_mask], data_set[1][bounds_mask]]}
    subset_data = [data_set[0][bounds_mask], data_set[1][bounds_mask], data_set[2], data_set[3][bounds_mask]]
    total_mse, total_bias2, total_var, subset_subset_measures, subset_subsets = true_data_comparrison_measures(subset_data, estimates[bounds_mask], log_comparrison=log_comparrison)
    spatial_subset_measures = {name: [np.mean(spatial_subset[name][0]), total_mse, total_bias2, total_var, subset_subset_measures, subset_subsets]}
    return spatial_subset_measures, spatial_subset

def get_correct_measures(data_sets, estimators, log_comparrison=False, bounds = None):
    result = {}
    for key in data_sets:
        total_mse, total_bias2, total_var, subset_measures, subsets = true_data_comparrison_measures(data_sets[key], estimators[key][0], log_comparrison=log_comparrison)
        if bounds is not None:
            spatial_subset_measures, spatial_subset = add_spatial_subset(data_sets[key], estimators[key][0], bounds[key], log_comparrison=log_comparrison)
            subset_measures = subset_measures | spatial_subset_measures
            subsets = subsets | spatial_subset
        result[key] = [np.mean(estimators[key][0]), total_mse, total_bias2, total_var, subset_measures, subsets]
    return result

def get_comparrison_measures(data_sets, estimators, log_comparrison=False, bounds = None):
    result = get_correct_measures(data_sets, estimators, log_comparrison=log_comparrison, bounds=bounds)
    return result