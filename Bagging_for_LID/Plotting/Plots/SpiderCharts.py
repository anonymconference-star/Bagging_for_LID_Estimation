from __future__ import annotations
from typing import Mapping, Tuple
from pathlib import Path
from typing import Sequence, Iterable
import plotly.express as px
import plotly.graph_objects as go
###################################################OWN IMPORT###################################################
from Bagging_for_LID.Plotting.plotting_helpers import *
from Bagging_for_LID.Plotting.optimize_across_parameter_results import *
from Bagging_for_LID.Helper.Other import *
###############################################################################################################################SPIDER CHARTS###############################################################################################################################
_NUMERIC_PARAMS = {"n", "k", "sr", "Nbag", "lid", "dim", "t"}
_DATASET_SPECIFIC = {"dataset_name", "n", "lid", "dim"}
_KNOWN_PARAMS = {
    "estimator_name", "bagging_method", "submethod_0", "submethod_error",
    "k", "sr", "Nbag", "pre_smooth", "post_smooth", "t",
}

#As opposed to the other plotting functions, plot_radar_from_results and plot_tables_from_results does not take all the experiments as input.
#Instead, only the neceassry data extracted from optimal ones are given to it, which have been determined as such by the helper functions from optimize_across_parameter_results.
#Therefore, it doesn't perform any of the optimization steps to select the experiments (across parameter combinations) with lowest MSE/Var/Bias^2.
#This preprocessing is called by a wrapper underneath to simplify the function's use.

def plot_radar_from_results(
    results: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    *,
    normalize_data: bool = False,
    log: bool = False,
    inner_radius: float = 0.1,
    save: bool = True,
    export_format: str = "pdf",
    save_prefix: str = "radar_best",
    save_dir: Path | str = "./Output",
    fill: bool = True,
    height_per_row: int = 450,
    width_per_col: int = 450,
    verbose: bool = False,
    estimator_name: str | None = None,
    metric_label_map: Mapping[str, str] | None = None,
    metrics: Tuple[str, ...] = ("mse", "bias2", "var")
):
    """
    Plot radar charts using the output of `result_extraction`.
    """
    def pretty_metric_name(key: str) -> str:
        if metric_label_map and key in metric_label_map:
            return metric_label_map[key]
        base = key[6:] if key.startswith("total_") else key
        return {"mse": "MSE", "bias2": "Bias²", "var": "Variance"}.get(base, base.upper())
    colour_cycle = px.colors.qualitative.Plotly
    for met_key, (params_df, values_df) in results.items():
        # result_extraction with decomposition_param='full' returns keys like 'total_mse',
        # while metrics uses short names like 'mse' — normalize before matching
        met_key_short = met_key.removeprefix("total_") if met_key.startswith("total_") else met_key
        if met_key_short in metrics:
            if not isinstance(values_df, pd.DataFrame) or not isinstance(params_df, pd.DataFrame):
                raise TypeError(f"results['{met_key}'] must be a (params_df, values_df) pair of DataFrames.")
            if not values_df.index.equals(params_df.index) or not values_df.columns.equals(params_df.columns):
                raise ValueError(f"Index/columns mismatch for '{met_key}' between params_df and values_df.")
            datasets = list(values_df.index)
            method_sigs = list(values_df.columns)
            diff_params: set[str] = set()
            if len(method_sigs) > 1:
                for idx in range(len(method_sigs[0])):
                    pname = method_sigs[0][idx][0]
                    uniq_vals = {sig[idx][1] for sig in method_sigs}
                    if len(uniq_vals) > 1:
                        diff_params.add(pname)
            vals = values_df.astype(float).copy()
            if log:
                with np.errstate(divide="ignore", invalid="ignore"):
                    vals = np.log10(vals.astype(float))
            if normalize_data:
                for ds in vals.index:
                    vals.loc[ds] = Normalize(vals.loc[ds].to_numpy(dtype=float))
            fig = go.Figure()
            radial_range = [-inner_radius, 1] if normalize_data else None

            for idx, sig in enumerate(method_sigs):
                label = " | ".join(f"{k}:{fmt_val(k, v)}" for k, v in sig if k in diff_params) or "default"
                label = modify_label(label)

                colour = colour_cycle[idx % len(colour_cycle)]
                r, g, b = [int(c) for c in px.colors.hex_to_rgb(colour)]
                r_vals = vals.iloc[:, idx].to_list()
                fig.add_trace(go.Scatterpolar(
                    r=r_vals + [r_vals[0] if len(r_vals) else None],
                    theta=datasets + [datasets[0]] if datasets else [],
                    name=label,
                    mode="lines+markers",
                    line=dict(width=2.5, color=colour),
                    fill="toself" if fill else None,
                    fillcolor=f"rgba({r},{g},{b},0.15)" if fill else None,
                ))

                if verbose:
                    print(f"▶ {label}")
                    for ds in datasets:
                        v = vals.at[ds, sig]
                        p = params_df.at[ds, sig]
                        if p is None or (isinstance(v, float) and not np.isfinite(v)):
                            print(f"   {ds}: MISSING")
                        else:
                            print(f"   {ds}: metric={v:.4g}, params={p}")
            est_txt = fmt_val("estimator_name", estimator_name.upper()) if (estimator_name and hasattr(estimator_name, "upper")) else (estimator_name or "")
            met_label = pretty_metric_name(met_key)
            title_txt = f"{met_label}" + (f" | Estimator:{est_txt}" if est_txt else "")
            fig.update_layout(
                title=dict(text=title_txt, x=0.5, y=0.95),
                template="plotly_white",
                height=height_per_row,
                width=width_per_col,
                polar=dict(
                    radialaxis=dict(range=radial_range, tickfont_size=9, showline=True, linewidth=1),
                    angularaxis=dict(tickfont_size=9),
                ),
                legend=dict(orientation="h", x=0.5, y=-0.15, xanchor="center"),
                margin=dict(t=80, b=100, l=80, r=80),
            )
            if save:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                out_path = Path(save_dir) / f"{save_prefix}_{met_key}.{export_format}"
                img = fig.to_image(format=export_format, engine="kaleido",
                                   width=width_per_col, height=height_per_row)
                out_path.write_bytes(img)
            else:
                fig.show()

#Wrapper for the above function to work off of experiment lists
def plot_radar_best_of_sweep(
    experiments: Sequence[Any],
    *,
    sweep_params: Iterable[str] | None = None,
    normalize_data: bool = False,
    log: bool = False,
    inner_radius: float = 0.1,
    save: bool = True,
    export_format: str = "pdf",
    save_prefix: str = "radar_best",
    save_dir: Path | str = "./Output",
    fill: bool = True,
    height_per_row: int = 450,
    width_per_col: int = 450,
    verbose: bool = False,
    decomposition_param: str | None = None,
    metrics: Tuple[str, ...] = ("mse", "bias2", "var")
):
    experiments = reassing_placeholder_value(experiments)
    results = result_extraction(experiments, sweep_params, metric_keys=None, decomposition_param=decomposition_param)
    estimator_name = experiments[0].estimator_name
    plot_radar_from_results(results=results, normalize_data=normalize_data,
    log=log,
    inner_radius=inner_radius,
    save=save,
    export_format=export_format,
    save_prefix=save_prefix,
    save_dir=save_dir,
    fill=fill,
    height_per_row=height_per_row,
    width_per_col=width_per_col,
    verbose=verbose,
    estimator_name=estimator_name,
    metric_label_map=None,
    metrics=metrics)