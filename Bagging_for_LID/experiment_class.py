import itertools
from collections.abc import Iterable
###################################################OWN IMPORT###################################################
from Bagging_for_LID.Datasets.DatasetGeneration import *
from Bagging_for_LID.Helper.Other import *
from Bagging_for_LID.Helper.additional_imports import *
from Bagging_for_LID.Helper.ComparrisonMeasures import *
from Bagging_for_LID.RunningEstimators.Collecting import *
from Bagging_for_LID.RunningEstimators.Running import *
from Bagging_for_LID.Datasets.Uniform_Generator import *
###################################################################################################################

def save_dict(data, directory, filename):
    os.makedirs(directory, exist_ok=True)  # Ensure directory exists
    filepath = os.path.join(directory, filename)  # Create full path
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def expand_param_dict(param_dict):
    def to_iterable(value):
        if isinstance(value, (str, bytes)):
            return [value]
        return list(value) if isinstance(value, Iterable) and not isinstance(value, dict) else [value]
    keys = list(param_dict)
    values = [to_iterable(param_dict[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

def expand_param_dicts(param_dicts):
    if isinstance(param_dicts, list):
        param_dicts = param_dicts
    else:
        param_dicts = [param_dicts]
    params_lists = []
    for param_dict in param_dicts:
        params_list1 = expand_param_dict(param_dict)
        params_lists = params_lists + params_list1
    return params_lists

def expand_param_dict_zipped(param_dict):
    def is_seq(v):
        if isinstance(v, (str, bytes, dict)):
            return False
        return isinstance(v, Iterable)
    def to_list(v):
        return list(v) if is_seq(v) else [v]

    keys = list(param_dict)
    vals = [to_list(param_dict[k]) for k in keys]

    lengths = [len(v) for v in vals]
    n = max(lengths)

    bad = [(k, len(v)) for k, v in zip(keys, vals) if len(v) not in (1, n)]
    if bad:
        msg = ", ".join([f"{k} has length {L}" for k, L in bad])
        raise ValueError(f"Zipped expansion needs all list-like params to have same length. {msg}. Expected 1 or {n}.")

    out = []
    for i in range(n):
        d = {}
        for k, v in zip(keys, vals):
            d[k] = v[i] if len(v) == n else v[0]
        out.append(d)

    return out

class LID_experiment:

    def __init__(self, n=2500, k=10, sr=0.3, Nbag=10, dataset_name="uniform", lid=2, dim=3, pre_smooth=False,
                 post_smooth=False, t=1, estimator='mle', bagging_method=None, submethod_0=None,
                 submethod_error=None, param_string=None, params=None, set_dim_to_lid=False, data_set_function=None):
        self.used_params = data_defaults()
        self.set_dim_to_lid = set_dim_to_lid
        if params is None and param_string is None:
            self.n = n
            self.k = k
            self.sr = sr
            self.Nbag = Nbag
            self.dataset_name = dataset_name
            self.lid = lid
            self.dim = dim
            self.pre_smooth = pre_smooth
            self.post_smooth = post_smooth
            self.t = t
            self.estimator_name = estimator
            self.bagging_method = bagging_method
            self.submethod_0 = submethod_0
            self.submethod_error = submethod_error
            self.data_set_function = data_set_function
            self.params = {'dataset_name': self.dataset_name,
                           'n': self.n,
                           'lid': self.lid,
                           'dim': self.dim,
                           'estimator_name': self.estimator_name,
                           'bagging_method': self.bagging_method,
                           'submethod_0': self.submethod_0,
                           'submethod_error': self.submethod_error,
                           'k': self.k,
                           'sr': self.sr,
                           'Nbag': self.Nbag,
                           'pre_smooth': self.pre_smooth,
                           'post_smooth': self.post_smooth,
                           't': self.t}
        elif params is None and param_string is not None:
            params = self.get_params(param_string)
            self.set_class_vars(params)
        elif params is not None and param_string is None:
            self.set_class_vars(params)
        else:
            Warning('Both params and param_string is not None, params is chosen.')
            self.set_class_vars(params)

        if self.lid is None:
            self.lid = self.used_params[self.dataset_name][0]
        if self.dim is None:
            self.dim = self.used_params[self.dataset_name][1]

        self.string = self.get_character_string(self.params)

        self.data = None

        self.result_dictionary = None
        self.results = None
        self.estimators_results_dictionary = None
        self.estimators_results = None

        self.lid_estimates = None
        self.total_mse = None
        self.total_var = None
        self.total_bias2 = None
        self.subset_measures = None
        self.subsets = None
        self.total_avg_estimate = None

        self.log_estimators_results = None
        self.log_total_mse = None
        self.log_total_bias2 = None
        self.log_total_var = None
        self.log_subset_measures = None
        self.log_subsets = None

        self.knn_dists = None
        self.point_avg_knn_dists = None
        self.bag_avg_knn_dists = None
        self.point_bag_avg_knn_dists = None
        self.lid_estimates_std = None

    def get_character_string(self, params):
        string = ''
        for key, value in params.items():
            string += key + f':{value}' + '|'
        string.rstrip('|')
        return string

    def get_params(self, string):
        kvs = string.split('|')
        params = {}
        for i in range(len(kvs)):
            kv = kvs[i].split(':')
            if kv[0] in ['n', 'lid', 'dim', 'k', 'Nbag']:
                v = int(kv[1]) if kv[1] is not None else None
            elif kv[0] in ['sr', 't']:
                v = float(kv[1]) if kv[1] is not None else None
            else:
                v = str(kv[1]) if kv[1] is not None else None
            params[kv[0]] = v
        return params

    def set_class_vars(self, params):
        self.n = params['n']
        self.k = params['k']
        self.sr = params['sr']
        self.Nbag = params['Nbag']
        self.dataset_name = params['dataset_name']
        self.lid = params['lid']
        self.dim = params['dim']
        self.pre_smooth = params['pre_smooth']
        self.post_smooth = params['post_smooth']
        self.t = params['t']
        self.estimator_name = params['estimator_name']
        self.bagging_method = params['bagging_method']
        self.submethod_0 = params['submethod_0']
        self.submethod_error = params['submethod_error']
        self.params = params

    def generate_data(self, load=False, directory=r"C:\pkls", fixpoints=None):
        if not load:
            original = ['M1_Sphere', 'M2_Affine_3to5', 'M3_Nonlinear_4to6', 'M4_Nonlinear', 'M5b_Helix2d', 'M6_Nonlinear',
                        'M7_Roll', 'M8_Nonlinear', 'M9_Affine', 'M10a_Cubic', 'M10b_Cubic', 'M10c_Cubic', 'M11_Moebius',
                        'M12_Norm', 'M13a_Scurve', 'Mn1_Nonlinear', 'Mn2_Nonlinear']
            if self.dataset_name in original:
                data_gen = skdim.datasets.BenchmarkManifolds()
                data = [data_gen.dict_gen[self.dataset_name](n=self.n, d=self.lid, dim=self.dim),
                        np.repeat(self.lid, self.n),
                        self.dim,
                        np.repeat(self.lid, self.n)]
            elif self.dataset_name == "lollipop_":
                datapoints, intrinsic_dim_array = lollipop_dataset(bs=self.n)
                m_val = self.dim
                data = [datapoints, intrinsic_dim_array, m_val, intrinsic_dim_array]
            elif self.dataset_name == "uniform":
                if self.set_dim_to_lid:
                    data = [Simple_LID_data(n=self.n, lid=self.lid, dim=self.lid),
                            np.repeat(self.lid, self.n),
                            self.lid,
                            np.repeat(self.lid, self.n)]
                else:
                    data = [Simple_LID_data(n=self.n, lid=self.lid, dim=self.dim),
                            np.repeat(self.lid, self.n),
                            self.dim,
                            np.repeat(self.lid, self.n)]
            elif self.dataset_name == "ribbon":
                data = [ribbon_multi_dim_equal_density(n=self.n, dim=self.dim, d_loc=self.lid, d_glob=1, ratio=0.01)[0],
                        np.repeat(self.lid, self.n),
                        self.lid,
                        np.repeat(self.lid, self.n)]
            elif self.dataset_name == "sparse":
                data = [sparse(n=self.n, lid=self.lid, dim=self.dim, w=10, l=5, center=(0.0, 0.0), rng=None)[0],
                        np.repeat(self.lid, self.n),
                        self.lid,
                        np.repeat(self.lid, self.n)]
            elif self.dataset_name == "custom":
                data = [self.data_set_function(n=self.n, lid=self.lid, dim=self.dim),
                        np.repeat(self.lid, self.n),
                        self.lid,
                        np.repeat(self.lid, self.n)]
            else:
                raise NotImplementedError
            filename = self.dataset_name + '_' + str(self.n) + '_' + str(self.lid) + '_' + str(self.dim)

            if fixpoints is not None:
                data[0] = np.concatenate([data[0], fixpoints[0]], axis=0)
                data[1] = np.concatenate([data[1], fixpoints[1]], axis=0)
                data[3] = np.concatenate([data[3], fixpoints[3]], axis=0)
            save_dict(data, directory=directory, filename=filename)
        else:
            filename = self.dataset_name + '_' + str(self.n) + '_' + str(self.lid) + '_' + str(self.dim)
            filepath = os.path.join(directory, filename)
            data = load_dict(filepath)
            if fixpoints is not None:
                data[0] = np.concatenate([data[0], fixpoints[0]], axis=0)
                data[1] = np.concatenate([data[1], fixpoints[1]], axis=0)
                data[3] = np.concatenate([data[3], fixpoints[3]], axis=0)
        self.data = data
        return data

    def estimate(self, bounds=None):
        if isinstance(self.estimator_name, list):
            estimator_names = self.estimator_name
        elif isinstance(self.estimator_name, str):
            estimator_names = [self.estimator_name]
        else:
            raise TypeError("Estimator name incorrect")
        data_sets = {self.dataset_name: self.data}
        result_dictionary = {estimator_names[i]: {} for i in range(len(estimator_names))}
        for key in data_sets:
            estimators_dict, avg_estimator_dict = complete_estimators(data_sets[key][0], self.k, self.sr, self.Nbag,
                                                                      self.pre_smooth, self.post_smooth, self.t,
                                                                      self.estimator_name, self.bagging_method,
                                                                      self.submethod_0, self.submethod_error,
                                                                      progress_bar=False, correct=True)
            for name in estimator_names:
                result_dictionary[name][key] = (estimators_dict[name], avg_estimator_dict[name])
        self.result_dictionary = result_dictionary
        if isinstance(self.estimator_name, str):
            self.results = result_dictionary[self.estimator_name][self.dataset_name]
        estimators_results_dictionary = {estimator_names[i]: get_comparrison_measures(data_sets, result_dictionary[estimator_names[i]], log_comparrison=False, bounds=bounds) for i in range(len(estimator_names))}
        log_estimators_results_dictionary = {estimator_names[i]: get_comparrison_measures(data_sets, result_dictionary[estimator_names[i]], log_comparrison=True, bounds=bounds) for i in range(len(estimator_names))}
        self.estimators_results_dictionary = estimators_results_dictionary
        if isinstance(self.estimator_name, str):
            self.estimators_results = estimators_results_dictionary[self.estimator_name][self.dataset_name]
            self.lid_estimates = self.results[0]
            self.total_avg_estimate = self.estimators_results[0]
            self.total_mse = self.estimators_results[1]
            self.total_bias2 = self.estimators_results[2]
            self.total_var = self.estimators_results[3]
            self.subset_measures = self.estimators_results[4]
            self.subsets = self.estimators_results[5]

            self.log_estimators_results = log_estimators_results_dictionary[self.estimator_name][self.dataset_name]
            self.log_total_mse = self.log_estimators_results[1]
            self.log_total_bias2 = self.log_estimators_results[2]
            self.log_total_var = self.log_estimators_results[3]
            self.log_subset_measures = self.log_estimators_results[4]
            self.log_subsets = self.log_estimators_results[5]
        return result_dictionary, estimators_results_dictionary

    def calc_knn_dists(self):
        knn_dists = complete_knn_distances(self.data[0], k=self.k, sr=self.sr, Nbag=self.Nbag, bagging_method=self.bagging_method)
        self.knn_dists = knn_dists
        self.point_avg_knn_dists = np.mean(knn_dists, axis=0)
        self.bag_avg_knn_dists = np.mean(knn_dists, axis=2)
        self.point_bag_avg_knn_dists = np.mean(self.point_avg_knn_dists, axis=1)
        return knn_dists







