from Bagging_for_LID.run_files.final_tasks import *
from Bagging_for_LID.RunningEstimators.Running2 import *
from Bagging_for_LID.Plotting.Plots.distance_LID import *

k_geomprog_large = geom_prog(min=5, max=200, step=50, integer=True, remove_duplicates=True)

trange = [t for t in range(50)]

directory = r'.\pkls'

save_name = 'ribbon_test_example'

multiprocess=True

load = False

query = [np.array([[0.5, 0.005]]), np.array([2]), None, np.array([2])]


def generate_all_experiments(n = 2500, krange = None, trange= None, query = None, load=False, directory=r'.\pkls', save_name='ribbon_test_example', multiprocess=True):
    all_experiments = []

    if krange is None:
        krange = geom_prog(min=5, max=200, step=50, integer=True, remove_duplicates=True)
    if trange is None:
        trange = [t for t in range(50)]

    for t in trange:
        param_dicts_data = {'dataset_name': 'ribbon',
                            'n': n,
                            'lid': None,
                            'dim': None,
                            'estimator_name': 'mle',
                            'bagging_method': None,
                            'submethod_0': '0',
                            'submethod_error': 'log_diff',
                            'k': 10,
                            'sr': 1,
                            'Nbag': 10,
                            'pre_smooth': False,
                            'post_smooth': False,
                            't': 1}

        param_dicts = {'dataset_name': 'ribbon',
                       'n': n,
                       'lid': None,
                       'dim': None,
                       'estimator_name': 'mle',
                       'bagging_method': None,
                       'submethod_0': '0',
                       'submethod_error': 'log_diff',
                       'k': krange,
                       'sr': 1,
                       'Nbag': 10,
                       'pre_smooth': False,
                       'post_smooth': False,
                       't': t}

        if query is None:
            query = [np.array([[0.5, 0.005]]), np.array([2]), None, np.array([2])]

        results_data = new_result_generator(param_dicts_data, multiprocess=False, load=load, load_data=load,
                                            worker_count=None,
                                            save_name=f'{save_name}_data_generation',
                                            directory=directory,
                                            fixpoints=query)

        experiments = new_lid_and_knn_dist_result_generator(param_dicts, multiprocess=multiprocess, load=load, load_data=True,
                                                            worker_count=5, save_name=f'{save_name}_t_{t}',
                                                            directory=directory)
        all_experiments = all_experiments + experiments

    return all_experiments

def generate_all_experiments_sparse(n = 2500, krange = None, trange= None, query = None, load=False, directory=r'.\pkls', save_name='ribbon_test_example', multiprocess=True):
    all_experiments = []

    if krange is None:
        krange = geom_prog(min=5, max=200, step=50, integer=True, remove_duplicates=True)
    if trange is None:
        trange = [t for t in range(50)]

    for t in trange:
        param_dicts_data = {'dataset_name': "sparse",
                            'n': n,
                            'lid': None,
                            'dim': None,
                            'estimator_name': 'mle',
                            'bagging_method': None,
                            'submethod_0': '0',
                            'submethod_error': 'log_diff',
                            'k': 10,
                            'sr': 1,
                            'Nbag': 10,
                            'pre_smooth': False,
                            'post_smooth': False,
                            't': 1}

        param_dicts = {'dataset_name': "sparse",
                       'n': n,
                       'lid': None,
                       'dim': None,
                       'estimator_name': 'mle',
                       'bagging_method': None,
                       'submethod_0': '0',
                       'submethod_error': 'log_diff',
                       'k': krange,
                       'sr': 1,
                       'Nbag': 10,
                       'pre_smooth': False,
                       'post_smooth': False,
                       't': t}

        if query is None:
            query = [np.array([[0, 0]]), np.array([2]), None, np.array([2])]

        results_data = new_result_generator(param_dicts_data, multiprocess=False, load=load, load_data=load,
                                            worker_count=None,
                                            save_name=f'{save_name}_data_generation',
                                            directory=directory,
                                            fixpoints=query)

        experiments = new_lid_and_knn_dist_result_generator(param_dicts, multiprocess=multiprocess, load=load, load_data=True,
                                                            worker_count=5, save_name=f'{save_name}_t_{t}',
                                                            directory=directory)
        all_experiments = all_experiments + experiments

    return all_experiments

def avg_experiments_using_t_as_index(experiments, trange, krange):
    experiments_t_k = {(t, k): None for t in trange for k in krange}
    for experiment in experiments:
        experiments_t_k[(experiment.t, experiment.k)] = experiment
    experiments_0_k_with_avg_estimates = {k: experiments_t_k[(0, k)] for k in krange}
    experiments_avg_lid_estimates = {k: np.mean([experiments_t_k[(t, k)].lid_estimates for t in trange], axis=0) for k in krange}
    experiments_std_lid_estimates = {k: np.std([experiments_t_k[(t, k)].lid_estimates for t in trange], axis=0) for k in krange}
    for k in krange:
        experiments_0_k_with_avg_estimates[k].lid_estimates = experiments_avg_lid_estimates[k]
        experiments_0_k_with_avg_estimates[k].lid_estimates_std = experiments_std_lid_estimates[k]
    exps = [experiments_0_k_with_avg_estimates[k] for k in k_geomprog_large]
    return exps, experiments_0_k_with_avg_estimates, experiments_t_k