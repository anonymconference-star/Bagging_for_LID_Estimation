#-----------------------------------------------------------------------------------------------------------------------
#External imports
from functools import partial
#-----------------------------------------------------------------------------------------------------------------------
#Internal Imports
from Bagging_for_LID.Plotting.Plots.SpiderCharts import *
from Bagging_for_LID.Plotting.Plots.Tables import *
from Bagging_for_LID.Plotting.Plots.VariableInteraction import *
from Bagging_for_LID.Plotting.Plots.MSEbars import *
from Bagging_for_LID.RunningEstimators.Running2 import *
from Bagging_for_LID.run_files.geom_prog import *
from Bagging_for_LID.run_files.parameter_combinations import *
from Bagging_for_LID.run_files.error_safe_running import *
#-----------------------------------------------------------------------------------------------------------------------
save_dir = r'./Output'
#-----------------------------------------------------------------------------------------------------------------------
#Function to collect test runs
def setup_tasks(task_dict, multiprocess=False, load=False, load_data=False, worker_count=1, save_name='result', directory=r"C:\pkls"):
    tasks = []
    for key, value in task_dict.items():
        tasks.append((key, partial(general_result_generator, value, multiprocess=multiprocess, load=load, load_data=load_data, worker_count=worker_count, save_name=save_name, directory=directory), (), {}))
    return tasks

#-----------------------------------------------------------------------------------------------------------------------
#Setup parameter dictionaries for main tests
effectiveness_test_general_param_dict = param_dicts_general(base_param_dict=effectiveness_test_base_param_dict,
                                                            variants_test_types=effectiveness_variants_test_types,
                                                            estimator_names=effectiveness_estimator_names,
                                                            changing_vars=['k', 'sr'],
                                                            test_name='Bagging_and_smoothing_test')

effectiveness_test_with_t_general_param_dict = param_dicts_general(base_param_dict=effectiveness_test_with_t_base_param_dict,
                                                                   variants_test_types=['weight_with_t'],
                                                                   estimator_names=['mle'],
                                                                   changing_vars=['k', 'sr', 't'],
                                                                   test_name="effectiveness_test_with_t")

Nbag_test_general_param_dict = param_dicts_general(base_param_dict=Nbag_test_base_param_dict,
                                                   variants_test_types=variable_variants_test_types,
                                                   estimator_names=variable_estimator_names,
                                                   changing_vars=['Nbag'],
                                                   test_name='Number_of_bags_test')

sr_test_general_param_dict = param_dicts_general(base_param_dict=sr_prog_test_base_param_dict,
                                                 variants_test_types=variable_variants_test_types,
                                                 estimator_names=variable_estimator_names,
                                                 changing_vars=['sr'],
                                                 test_name='Sampling_rate_test')

interaction_sr_Nbag_test_general_param_dict = param_dicts_general(base_param_dict=interaction_sr_Nbag_test_base_param_dict,
                                                                  variants_test_types=variable_variants_test_types,
                                                                  estimator_names=variable_estimator_names,
                                                                  changing_vars=['sr', 'Nbag'],
                                                                  test_name='Interaction_of_sampling_rate_and_number_of_bags_test')

interaction_sr_k_test_general_param_dict = param_dicts_general(base_param_dict=interaction_sr_k_test_base_param_dict,
                                                               variants_test_types=variable_variants_test_types,
                                                               estimator_names=variable_estimator_names,
                                                               changing_vars=['sr', 'k'],
                                                               test_name='Interaction_of_k_and_sampling_rate_test')

#-----------------------------------------------------------------------------------------------------------------------
#Setup plotting
n_rows = 3
n_cols = 7
barchart_figsize = (12*n_cols/4, 12*n_rows/5)
heatmap_figsize = (0.8*25*n_cols/4, 25*n_rows/5)
metrics = ("mse",) #, "bias2", "var"

plot_tasks_main = {

    "Bagging_and_smoothing_test": [
        (plot_radar_best_of_sweep, dict(sweep_params=['k', 'sr'], normalize_data=True, log=False,
                                        save=True, height_per_row=450, width_per_col=450,
                                        verbose=False, save_dir=f"{save_dir}/radar", decomposition_param='full',
                                        metrics=metrics)),
        (plot_table_best_of_sweep, dict(sweep_params=['k', 'sr'], mode="combined", normalize_data=False,
                                        log=False, metric_label_map=None, save_dir=f"{save_dir}/table",
                                        best_font_color = "black", heatmap_colorscale=red_blue_bright,
                                        heatmap_cells=True, decomposition_param=['combined'], combined=True)),
    ],
    "effectiveness_test_with_t": [
        (plot_radar_best_of_sweep, dict(sweep_params=['k', 'sr', 't'], normalize_data=True, log=False,
                                        save=True, height_per_row=450, width_per_col=450,
                                        verbose=False, save_dir=f"{save_dir}/radar", decomposition_param='full',
                                        metrics=metrics)),
        (plot_table_best_of_sweep, dict(sweep_params=['k', 'sr', 't'], mode="combined", normalize_data=False,
                                        log=False, metric_label_map=None, best_font_color="black",
                                        save_dir="./plots/table", heatmap_colorscale=red_blue_bright,
                                        heatmap_cells=True, decomposition_param=['combined'], combined=True)),
    ],

    "Number_of_bags_test": [
        (plot_experiment_mse_bars, dict(vary_param='Nbag', grid=True, figsize=barchart_figsize,
                                        base_fontsize=14, label_every=4, save_dir=f"{save_dir}/msebar",
                                        fig_title=False, n_rows=n_rows, n_cols=n_cols,
                                        compact=True, show_average=False)),
    ],

    "Sampling_rate_test": [
        (plot_experiment_mse_bars, dict(vary_param='sr', grid=True, figsize=barchart_figsize,
                                        base_fontsize=14, label_every=4, save_dir=f"{save_dir}/msebar",
                                        fig_title=False, n_rows=n_rows, n_cols=n_cols,
                                        compact=True, show_average=False)),
    ],

    "Interaction_of_sampling_rate_and_number_of_bags_test": [
        (plot_experiment_heatmaps, dict(x_param='sr', y_param='Nbag', reverse_x=False, reverse_y=False,
                                        metrics=metrics, label_every=4, grid=True,
                                        figsize=heatmap_figsize, base_fontsize=24, cmap="RdBu",
                                        save_dir=f"{save_dir}/interaction",
                                        log=True, type='difference', inlog=False,
                                        fig_title=False, rows=n_rows, cols=n_cols, compact=True,
                                        show_average=True, shared_colorbar=False)),
    ],

    "Interaction_of_k_and_sampling_rate_test": [
        (plot_experiment_heatmaps, dict(x_param='sr', y_param='k', reverse_x=False, reverse_y=False,
                                        metrics=metrics, label_every=4, grid=True,
                                        figsize=heatmap_figsize, base_fontsize=24, cmap="RdBu",
                                        save_dir=f"{save_dir}/interaction",
                                        log=True, type='difference', inlog=False,
                                        fig_title=False, rows=n_rows, cols=n_cols, compact=True,
                                        show_average=True, shared_colorbar=False)),
    ],
}

n_rows_appendix = 5
n_cols_appendix = 4
barchart_figsize_appendix = (12*n_cols_appendix/4, 12*n_rows_appendix/5)
heatmap_figsize_appendix = (25*n_cols_appendix/4, 25*n_rows_appendix/5)
metrics_appendix = ("mse",) #, "bias2", "var"

plot_tasks_appendix = {

    "Bagging_and_smoothing_test": [
        (plot_radar_best_of_sweep, dict(sweep_params=['k', 'sr'], normalize_data=True, log=False,
                                        save=True, height_per_row=450, width_per_col=450,
                                        verbose=False, save_dir=f"{save_dir}/radar_appendix", decomposition_param='full',
                                        metrics=metrics_appendix)),
        (plot_table_best_of_sweep, dict(sweep_params=['k', 'sr'], mode="combined", normalize_data=False,
                                        log=False, metric_label_map=None, save_dir=f"{save_dir}/table_appendix",
                                        best_font_color = "black", heatmap_colorscale=red_blue_bright,
                                        heatmap_cells=True, decomposition_param=['combined'], combined=True)),
    ],
    "effectiveness_test_with_t": [
        (plot_radar_best_of_sweep, dict(sweep_params=['k', 'sr', 't'], normalize_data=True, log=False,
                                        save=True, height_per_row=450, width_per_col=450,
                                        verbose=False, save_dir=f"{save_dir}/radar_appendix", decomposition_param='full',
                                        metrics=metrics_appendix)),
        (plot_table_best_of_sweep, dict(sweep_params=['k', 'sr', 't'], mode="combined", normalize_data=False,
                                        log=False, metric_label_map=None, best_font_color="black",
                                        save_dir="./plots/table_appendix", heatmap_colorscale=red_blue_bright,
                                        heatmap_cells=True, decomposition_param=['combined'], combined=True)),
    ],

    "Number_of_bags_test": [
        (plot_experiment_mse_bars, dict(vary_param='Nbag', grid=True, figsize=barchart_figsize_appendix,
                                        base_fontsize=6, label_every=1, save_dir=f"{save_dir}/msebar_appendix",
                                        fig_title=False, n_rows=n_rows_appendix, n_cols=n_cols_appendix,
                                        compact=False, legend_fontsize=28, tick_length=1.5)),
    ],

    "Sampling_rate_test": [
        (plot_experiment_mse_bars, dict(vary_param='sr', grid=True, figsize=barchart_figsize_appendix,
                                        base_fontsize=6, label_every=1, save_dir=f"{save_dir}/msebar_appendix",
                                        fig_title=False, n_rows=n_rows_appendix, n_cols=n_cols_appendix,
                                        compact=False, legend_fontsize=28, tick_length=1.5)),
    ],

    "Interaction_of_sampling_rate_and_number_of_bags_test": [
        (plot_experiment_heatmaps, dict(x_param='sr', y_param='Nbag', reverse_x=False, reverse_y=False,
                                        metrics=metrics_appendix, label_every=1, grid=True,
                                        figsize=heatmap_figsize_appendix, base_fontsize=10, cmap="RdBu",
                                        save_dir=f"{save_dir}/interaction_appendix",
                                        log=True, type='difference', inlog=False,
                                        fig_title=False, rows=n_rows_appendix, cols=n_cols_appendix,
                                        compact=False, cbar_label_fontsize=10, legend_fontsize=28, tick_length=1.5)),
    ],

    "Interaction_of_k_and_sampling_rate_test": [
        (plot_experiment_heatmaps, dict(x_param='sr', y_param='k', reverse_x=False, reverse_y=False,
                                        metrics=metrics_appendix, label_every=1, grid=True,
                                        figsize=heatmap_figsize_appendix, base_fontsize=10, cmap="RdBu",
                                        save_dir=f"{save_dir}/interaction_appendix",
                                        log=True, type='difference', inlog=False,
                                        fig_title=False, rows=n_rows_appendix, cols=n_cols_appendix,
                                        compact=False, cbar_label_fontsize=10, legend_fontsize=28, tick_length=1.5)),
    ],
}