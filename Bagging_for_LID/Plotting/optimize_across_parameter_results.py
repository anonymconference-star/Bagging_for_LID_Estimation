from __future__ import annotations
import pandas as pd
from Bagging_for_LID.Plotting.naming_helpers import *

def sorted_experiments(experiments, sweep_params):
    _NUMERIC_PARAMS = {"n", "k", "sr", "Nbag", "lid", "dim", "t"}
    _DATASET_SPECIFIC = {"dataset_name", "n", "lid", "dim"}
    _KNOWN_PARAMS = {"estimator_name", "bagging_method", "submethod_0", "submethod_error",
                     "k", "sr", "Nbag", "pre_smooth", "post_smooth", "t"}

    if sweep_params is None:
        sweep_params = _NUMERIC_PARAMS
    sweep_params = set(sweep_params)
    method_params = _KNOWN_PARAMS - sweep_params
    datasets = sorted({getattr(e, "dataset_name") for e in experiments})

    def method_sig(e):
        return tuple((p, getattr(e, p)) for p in sorted(method_params))

    methods = sorted({method_sig(e) for e in experiments}, key=str)

    df = pd.DataFrame(index=datasets, columns=methods, dtype=object)
    for ds in datasets:
        for m in methods:
            df.at[ds, m] = []

    for e in experiments:
        ds = getattr(e, "dataset_name")
        sig = method_sig(e)
        df.at[ds, sig].append(e)

    fixed_ds_specific = (_DATASET_SPECIFIC - sweep_params) - {"dataset_name"}
    for ds in datasets:
        for m in methods:
            runs = df.at[ds, m]
            if not runs:
                continue
            for p in fixed_ds_specific:
                vals = {getattr(r, p) for r in runs}
                if len(vals) > 1:
                    raise ValueError(
                        f"Inconsistent '{p}' within cell dataset={ds}, method={m}. "
                        f"Found values: {sorted(vals)}"
                    )
    return df

def extract_params(df, params):
    datasets = df.index
    methods = df.columns
    dummydf = pd.DataFrame(index=datasets, columns=methods, dtype=object)
    for ds in datasets:
        for m in methods:
            if m[1][0] != 'bagging_method':
                print('Problem detected with method searching code in extract_params')
            if m[1][1] is None:
                used_params = [p for p in params if p !='sr' and p !='t']
            elif m[1][1] == 'bag':
                used_params = [p for p in params if p != 't']
            else:
                used_params = params
            if isinstance(df.at[ds, m], list):
                dummydf.at[ds, m] = [{p if p !='sr' else 'r': getattr(e, p) for p in used_params} for e in df.at[ds, m]]
            else:
                dummydf.at[ds, m] = {p if p !='sr' else 'r': getattr(df.at[ds, m], p) for p in used_params}
    return dummydf

def extract_optimal(df, metric_key, return_values=False, decomposition_param=None):
    from operator import attrgetter
    datasets = df.index
    methods = df.columns
    best_df  = pd.DataFrame(index=datasets, columns=methods, dtype=object)

    if decomposition_param == 'combined':
        vals_df = pd.DataFrame(index=datasets, columns=methods, dtype=object)
    else:
        vals_df = pd.DataFrame(index=datasets, columns=methods, dtype=float)

    for ds in datasets:
        for m in methods:
            if decomposition_param is not None:
                if decomposition_param == 'combined':
                    runs = df.at[ds, m]
                    best = min(runs, key=attrgetter('total_mse'))
                    best_df.at[ds, m] = best
                    vals_df.at[ds, m] = [float(getattr(best, 'total_mse')), float(getattr(best, 'total_var')), float(getattr(best, 'total_bias2'))]
                else:
                    runs = df.at[ds, m]
                    best = min(runs, key=attrgetter('total_mse'))
                    best_df.at[ds, m] = best
                    vals_df.at[ds, m] = float(getattr(best, decomposition_param))
            else:
                runs = df.at[ds, m]
                best = min(runs, key=attrgetter(metric_key))
                best_df.at[ds, m] = best
                vals_df.at[ds, m] = float(getattr(best, metric_key))
    return (best_df, vals_df) if return_values else best_df

def extract_optimal_results(df, params, metric_key, decomposition_param=None):
    best_df, vals_df = extract_optimal(df, metric_key, return_values=True, decomposition_param=decomposition_param)
    optimal_params = extract_params(best_df, params)
    return optimal_params, vals_df

def extract_metric_results(df, params, metric_keys=None, decomposition_param=None):
    if metric_keys is None:
        metric_keys = ['total_mse', 'total_var', 'total_bias2']
    if decomposition_param=='full':
        decomposition_param = ['total_mse', 'total_var', 'total_bias2']
    out = {}
    if decomposition_param is not None:
        for decomp_key in decomposition_param:
            out[decomp_key] = extract_optimal_results(df, params, 'total_mse', decomposition_param=decomp_key)
    else:
        for key in metric_keys:
            out[key] = extract_optimal_results(df, params, key, decomposition_param=None)
    return out

def save_results2(results, directory, save_name):
    import os
    import pickle
    os.makedirs(directory, exist_ok=True)  # Ensure directory exists
    filepath = os.path.join(directory, save_name)  # Create full path
    with open(filepath, "wb") as fh:
        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)

###########################################################################

def result_extraction(experiments, sweep_params, metric_keys=None, save=False, directory=None, decomposition_param=None):
    df = sorted_experiments(experiments, sweep_params)
    df = reorder_sorted_experiments(df, sweep_params=sweep_params)
    metric_results = extract_metric_results(df, sweep_params, metric_keys=metric_keys, decomposition_param=decomposition_param)
    if save:
        save_results2(metric_results, directory=directory)
    return metric_results