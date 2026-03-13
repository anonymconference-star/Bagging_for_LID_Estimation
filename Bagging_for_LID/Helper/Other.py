import pandas as pd
import ast
import itertools
import pickle
import os
import numpy as np
###############################################################################################################################OTHER###############################################################################################################################
def split_long_name(name, max_length=20):
    parts = name.split('_')
    lines = []
    current_line = ""
    for part in parts:
        if current_line:
            if len(current_line) + len(part) + 1 <= max_length:
                current_line += "_" + part
            else:
                lines.append(current_line)
                current_line = part
        else:
            current_line = part
    if current_line:
        lines.append(current_line)
    return "\n".join(lines)

#these both do the same reverse min-max normalization, in a slightly different way computationally
def normalize(values):
    min_val = np.min(values)
    max_val = np.max(values)
    return [(1 - (x - min_val) / (max_val - min_val)) if max_val != min_val else 0 for x in values]

def Normalize(arr):
    a = np.asarray(arr, dtype=float)
    return 1 - (a - a.min()) / (a.max() - a.min()) if np.ptp(a) else np.zeros_like(a)

def load_from_df(load_path):
    df = pd.read_csv(load_path, index_col=0)
    d = df.to_dict("split")
    d = dict(zip(d["index"], d["data"]))
    for key in d:
        value = d[key]
        if isinstance(value, str):
            sanitized_string = value.replace("\n", "").replace(" ", "")
            try:
                d[key] = ast.literal_eval(sanitized_string)
            except ValueError as e:
                continue
        elif isinstance(value, list):
            for i in range(len(value)):
                if isinstance(value[i], str):
                    try:
                        value[i] = ast.literal_eval(value[i])
                    except ValueError as e:
                        continue
            d[key] = value
    return d

def convert_results_for_plot(res):
    result_dictionaries = []
    dict_names = []
    for key in res.keys():
        result_dictionaries.append(res[key][1])
        dict_names.append(f'{key}')
    return result_dictionaries, dict_names

def generate_param_combinations(param_list):
    expanded_combinations = []
    for param_tuple in param_list:
        k, n_bags, sampling_rate, method_names, method_type = param_tuple
        k = k if isinstance(k, list) else [k]
        n_bags = n_bags if isinstance(n_bags, list) else [n_bags]
        sampling_rate = sampling_rate if isinstance(sampling_rate, list) else [sampling_rate]
        method_type = method_type if isinstance(method_type, list) else [method_type]
        for k_val, method_type_val in itertools.product(k, method_type):
            if method_type_val.startswith('bag'):
                for n_bags_val, sampling_rate_val in itertools.product(n_bags, sampling_rate):
                    expanded_combinations.append((k_val, n_bags_val, sampling_rate_val, method_names, method_type_val))
            else:
                expanded_combinations.append((k_val, None, None, method_names, method_type_val))
    return expanded_combinations

def reduce_result(result):
    reduced_result = result.copy()
    for key, value in result.items():
        for key2 in value[0].keys():
            reduced_result[key][0][key2] = result[key][0][key2][0]
    return reduced_result

def save_dict(data, directory, filename):
    os.makedirs(directory, exist_ok=True)  # Ensure directory exists
    filepath = os.path.join(directory, filename)  # Create full path
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
