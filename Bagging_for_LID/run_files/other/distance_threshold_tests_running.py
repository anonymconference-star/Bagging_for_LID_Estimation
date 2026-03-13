from Bagging_for_LID.run_files.final_tasks import *
from Bagging_for_LID.RunningEstimators.Running2 import *
from Bagging_for_LID.Plotting.Plots.VariableInteraction import *
from Bagging_for_LID.Plotting.Plots.other.AdditionalInteraction import *
import skdim
from matplotlib import pyplot as plt

def threshold_testing(K=10, load_data=False, load_threshold_results=False, load_lid_mse_results=False,
                      multiprocess=True, worker_count=5, directory=r"C:\pkls", save_path="./Output"):
    k_prog = [5 + k for k in range(K - 5)]
    sr_prog = [k / K for k in k_prog]
    param_dicts_none = {'dataset_name': dataset_name_strings,
                        'n': 2500,
                        'lid': None,
                        'dim': None,
                        'estimator_name': 'mle',
                        'bagging_method': None,
                        'submethod_0': '0',
                        'submethod_error': 'log_diff',
                        'k': K,
                        'sr': 1,
                        'Nbag': 10,
                        'pre_smooth': False,
                        'post_smooth': False,
                        't': 1}
    param_dicts_bag = {'dataset_name': dataset_name_strings,
                       'n': 2500,
                       'lid': None,
                       'dim': None,
                       'estimator_name': 'mle',
                       'bagging_method': 'bag',
                       'submethod_0': '0',
                       'submethod_error': 'log_diff',
                       'k': k_prog,
                       'sr': sr_prog,
                       'Nbag': 10,
                       'pre_smooth': False,
                       'post_smooth': False,
                       't': 1}

    results_data = new_result_generator(param_dicts_data, multiprocess=False, load=load_data, load_data=load_data,
                                        worker_count=None,
                                        save_name='data_generation',
                                        directory=r"C:\pkls")

    exps_baseline = new_knn_dist_result_generator(param_dicts_none.copy(), multiprocess=multiprocess, load=load_threshold_results,
                                                  load_data=True, worker_count=worker_count,
                                                  save_name=f'knn_dists_res_pairs_baseline_k_{K}',
                                                  directory=directory)

    exps_bagged = new_knn_dist_result_generator(param_dicts_bag.copy(), multiprocess=multiprocess, load=load_threshold_results, load_data=load_data,
                                                worker_count=worker_count, save_name=f'knn_dists_res_pairs_bagged_k_{K}',
                                                directory=directory)

    lid_results_baseline = new_result_generator(param_dicts_none.copy(), multiprocess=multiprocess, load=load_lid_mse_results, load_data=load_data,
                                                worker_count=worker_count, save_name=f'estimate_res_pairs_baseline_k_{K}',
                                                directory=directory)

    lid_results_bagged = new_result_generator(param_dicts_bag.copy(), multiprocess=multiprocess, load=load_lid_mse_results, load_data=load_data,
                                              worker_count=worker_count, save_name=f'estimate_res_pairs_bagged_k_{K}',
                                              directory=directory)

    plot_experiment_attr(
        exps_baseline + exps_bagged,
        x_param="sr",
        y_param="k",
        mode="difference",
        compare_param="bagging_method",
        compare_values=(None, "bag"),
        baseline_xy=(0.05, K),
        diff_kind="difference",
        plot_kind="heatmap",
        value_attr="point_bag_avg_knn_dists",
        value_index=-1,
        save_prefix=f"distance_thresholds_baseline_k_{K}",
        formats=("pdf",),
        show=False,
        figsize=(25, 25),
        base_fontsize=8,
        cmap="RdBu",
        save_dir = save_path
    )

    plot_experiment_heatmaps(
        lid_results_baseline + lid_results_bagged,
        x_param="sr",
        y_param="k",
        type="difference",
        baseline_overrides={"k": K},
        save_prefix=f"distance_thresholds_baseline_k_{K}",
        formats=("pdf",),
        show=False,
        figsize=(25, 25),
        base_fontsize=8,
        metrics=("mse", "bias2", "var"),
        label_every=1,
        grid=True,
        log=True,
        inlog=False,
        fig_title=f"k_sr_interaction_at_distance_thresholds_baseline_k_{K}",
        cmap="RdBu",
        save_dir=save_path
    )

    return exps_baseline + exps_bagged, lid_results_baseline + lid_results_bagged