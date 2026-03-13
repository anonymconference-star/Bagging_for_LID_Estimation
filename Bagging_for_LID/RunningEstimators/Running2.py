import multiprocessing as mp
from itertools import repeat
from tqdm import tqdm
import logging, traceback
###################################################OWN IMPORT###################################################
from Bagging_for_LID.experiment_class import *
###############################################################################################################################RUNNING ESTIMATORS###################################
def save_results2(results, directory, save_name):
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, save_name)
    with open(filepath, "wb") as fh:
        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)

def load_results2(directory, save_name, print_name=False):
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, save_name)
    with open(filepath, "rb") as fh:
        if print_name:
            print(fh)
        results = pickle.load(fh)
    return results

def run_experiment1(args):
    params, load, directory, fixpoints = args
    experiment = LID_experiment(params=params)
    experiment.generate_data(load=load, directory=directory, fixpoints=fixpoints)
    experiment.estimate(bounds=None)
    return experiment

def run_experiment2(params, load=False, directory=r"C:\pkls", fixpoints=None):
    experiment = LID_experiment(params=params)
    experiment.generate_data(load=load, directory=directory, fixpoints=fixpoints)
    experiment.estimate(bounds=None)
    return experiment

def run_experiment_knn_dist1(args):
    params, load, directory = args
    experiment = LID_experiment(params=params)
    experiment.generate_data(load=load, directory=directory)
    experiment.calc_knn_dists()
    return experiment

def run_experiment_lid_and_knn_dist(params, load=False, directory=r"C:\pkls", fixpoints=None):
    experiment = LID_experiment(params=params)
    experiment.generate_data(load=load, directory=directory, fixpoints=fixpoints)
    experiment.calc_knn_dists()
    experiment.estimate(bounds=None)
    return experiment

def run_experiment_lid_and_knn_dist1(args):
    params, load, directory, fixpoints = args
    experiment = LID_experiment(params=params)
    experiment.generate_data(load=load, directory=directory, fixpoints=fixpoints)
    experiment.calc_knn_dists()
    experiment.estimate(bounds=None)
    return experiment

def run_experiment_knn_dist(params, load=False, directory=r"C:\pkls"):
    experiment = LID_experiment(params=params)
    experiment.generate_data(load=load, directory=directory)
    experiment.calc_knn_dists()
    return experiment

def _run_star(args):
    try:
        exp = run_experiment2(*args)
        return {"ok": True, "value": exp}
    except Exception:
        return {"ok": False, "traceback": traceback.format_exc()}

def _run_star_knn_dist(args):
    try:
        exp = run_experiment_knn_dist(*args)
        return {"ok": True, "value": exp}
    except Exception:
        return {"ok": False, "traceback": traceback.format_exc()}

def _run_star_lid_and_knn_dist(args):
    try:
        exp = run_experiment_lid_and_knn_dist(*args)
        return {"ok": True, "value": exp}
    except Exception:
        return {"ok": False, "traceback": traceback.format_exc()}


def run_several_experiments_multiprocess(
    params_lists,
    worker_count=None,
    load=False,
    directory=r"C:\pkls",
    fixpoints=None,
):
    if worker_count is None:
        worker_count = mp.cpu_count()
    tasks = zip(params_lists, repeat(load), repeat(directory), repeat(fixpoints))
    results = []
    failures = 0
    with mp.Pool(worker_count) as pool:
        for res in tqdm(
            pool.imap_unordered(_run_star, tasks),
            total=len(params_lists),
            desc="running experiments",
            unit="exp",
        ):
            if res["ok"]:
                results.append(res["value"])
            else:
                failures += 1
                print(f"[ERROR] experiment failed ({failures} so far). See log for traceback.")
                logging.error("Experiment failed with traceback:\n%s", res["traceback"])
    logging.info("Completed %d/%d experiments (failed: %d).", len(results), len(params_lists), failures)
    return results

def run_several_knn_dist_experiments_multiprocess(
    params_lists,
    worker_count=None,
    load=False,
    directory=r"C:\pkls",
):
    if worker_count is None:
        worker_count = mp.cpu_count()
    tasks = zip(params_lists, repeat(load), repeat(directory))
    results = []
    failures = 0
    with mp.Pool(worker_count) as pool:
        for res in tqdm(
            pool.imap_unordered(_run_star_knn_dist, tasks),
            total=len(params_lists),
            desc="running experiments",
            unit="exp",
        ):
            if res["ok"]:
                results.append(res["value"])
            else:
                failures += 1
                print(f"[ERROR] experiment failed ({failures} so far). See log for traceback.")
                logging.error("Experiment failed with traceback:\n%s", res["traceback"])
    logging.info("Completed %d/%d experiments (failed: %d).", len(results), len(params_lists), failures)
    return results

def run_several_lid_and_knn_dist_experiments_multiprocess(
    params_lists,
    worker_count=None,
    load=False,
    directory=r"C:\pkls",
    fixpoints=None,
):
    if worker_count is None:
        worker_count = mp.cpu_count()
    tasks = zip(params_lists, repeat(load), repeat(directory), repeat(fixpoints))
    results = []
    failures = 0
    with mp.Pool(worker_count) as pool:
        for res in tqdm(
            pool.imap_unordered(_run_star_lid_and_knn_dist, tasks),
            total=len(params_lists),
            desc="running experiments",
            unit="exp",
        ):
            if res["ok"]:
                results.append(res["value"])
            else:
                failures += 1
                print(f"[ERROR] experiment failed ({failures} so far). See log for traceback.")
                logging.error("Experiment failed with traceback:\n%s", res["traceback"])
    logging.info("Completed %d/%d experiments (failed: %d).", len(results), len(params_lists), failures)
    return results

def run_several_experiments_sequential(params_lists, load=False, directory=r"C:\pkls", fixpoints=None):
    results = []
    for params in tqdm(params_lists):
        experiment = run_experiment1(args=(params, load, directory, fixpoints))
        results.append(experiment)
    return results

def run_several_knn_dist_experiments_sequential(params_lists, load=False, directory=r"C:\pkls"):
    results = []
    for params in tqdm(params_lists):
        experiment = run_experiment_knn_dist1(args=(params, load, directory))
        results.append(experiment)
    return results

def run_several_lid_and_knn_dist_experiments_sequential(params_lists, load=False, directory=r"C:\pkls", fixpoints=None):
    results = []
    for params in tqdm(params_lists):
        experiment = run_experiment_lid_and_knn_dist1(args=(params, load, directory, fixpoints))
        results.append(experiment)
    return results

def new_result_generator(param_dicts, multiprocess=False, load=False, load_data=False, worker_count=None, save_name='res', directory=r"C:\pkls", expand_comb=False, fixpoints=None):
    if not load:
        if expand_comb:
            params_lists = expand_param_dict_zipped(param_dicts)
        else:
            params_lists = expand_param_dicts(param_dicts)

        if multiprocess:
            results = run_several_experiments_multiprocess(params_lists, worker_count=worker_count, load=load_data, directory=directory, fixpoints=fixpoints)
        else:
            results = run_several_experiments_sequential(params_lists, load=load_data, directory=directory, fixpoints=fixpoints)
        save_results2(results, directory=directory, save_name=save_name)
    else:
        results = load_results2(directory=directory, save_name=save_name)
    return results

def new_knn_dist_result_generator(param_dicts, multiprocess=False, load=False, load_data=False, worker_count=None, save_name='res', directory=r"C:\pkls", expand_comb=False):
    if not load:
        if expand_comb:
            params_lists = expand_param_dict_zipped(param_dicts)
        else:
            params_lists = expand_param_dicts(param_dicts)
        if multiprocess:
            results = run_several_knn_dist_experiments_multiprocess(params_lists, worker_count=worker_count, load=load_data, directory=directory)
        else:
            results = run_several_knn_dist_experiments_sequential(params_lists, load=load_data, directory=directory)
        save_results2(results, directory=directory, save_name=save_name)
    else:
        results = load_results2(directory=directory, save_name=save_name)
    return results

def new_lid_and_knn_dist_result_generator(param_dicts, multiprocess=False, load=False, load_data=False, worker_count=None, save_name='res', directory=r"C:\pkls", expand_comb=False, fixpoints=None):
    if not load:
        if expand_comb:
            params_lists = expand_param_dict_zipped(param_dicts)
        else:
            params_lists = expand_param_dicts(param_dicts)

        if multiprocess:
            results = run_several_lid_and_knn_dist_experiments_multiprocess(params_lists, worker_count=worker_count, load=load_data, directory=directory, fixpoints=fixpoints)
        else:
            results = run_several_lid_and_knn_dist_experiments_sequential(params_lists, load=load_data, directory=directory, fixpoints=fixpoints)
        save_results2(results, directory=directory, save_name=save_name)
    else:
        results = load_results2(directory=directory, save_name=save_name)
    return results

def general_result_generator(param_dicts_dict, multiprocess=False, load=False, load_data=False, worker_count=None, save_name='res', directory=r"C:\pkls", expand_comb=False):
    results_dict = {}
    for key, value in param_dicts_dict.items():
        result = new_result_generator(value, multiprocess=multiprocess, load=load, load_data=load_data, worker_count=worker_count,
                             save_name=f'{save_name}_{key}', directory=directory, expand_comb=expand_comb)
        results_dict[f'{save_name}_{key}'] = result
    return results_dict

def plotting_across_results_dict(results_dict, plotting_function, **kwargs):
    for key, value in results_dict.items():
        plotting_function(experiments=value, save_prefix=key, **kwargs)

def merge_experiment_lists(directory, save_name1, save_name2, replace_dataset_key=None, save=True):
    results1 = load_results2(directory=directory, save_name=save_name1)
    results2 = load_results2(directory=directory, save_name=save_name2)
    if replace_dataset_key is None:
        results_merged = results1 + results2
    else:
        results_merged = []
        for experiment in results1:
            if experiment.dataset_name != replace_dataset_key:
                results_merged.append(experiment)
        for experiment in results2:
            if experiment.dataset_name == replace_dataset_key:
                results_merged.append(experiment)
    if save:
        save_results2(results_merged, directory=directory, save_name='merged'+save_name1)
    return results_merged









