from __future__ import annotations
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Sequence, Tuple, Union, Mapping
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.transforms import ScaledTranslation
from Bagging_for_LID.Plotting.plotting_helpers import *

_NUMERIC_PARAMS = {"n", "k", "sr", "Nbag", "lid", "dim", "t"} # class parameters that can change for bagged estimators
_BASELINE_PARAMS = {"n", "k", "lid", "dim"}            # class parameters that can change for baseline estimator
_BOOL_STR_PARAMS = { #these are not changable class parameters for this interaction plot
    "pre_smooth",
    "post_smooth",
    "estimator_name",
    "bagging_method",
    "submethod_0",
    "submethod_error",
}
_ALL_PARAMS = _NUMERIC_PARAMS | _BOOL_STR_PARAMS

def plot_experiment_heatmaps(
    experiments: Sequence[Any],
    *,
    x_param: str,
    y_param: str,
    baseline_xy: tuple[Any, Any] | None = None,
    baseline_overrides: Mapping[str, Any] | None = None,
    reverse_x: bool = False,
    reverse_y: bool = False,
    metrics: Tuple[str, ...] = ("mse", "bias2", "var"),
    label_every: int = 1,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    base_fontsize: int | float | None = None,
    cmap: str = "bwr",
    save_prefix: str = "heat",
    save_dir: str | Path = "./Output",
    formats: Tuple[str, ...] = ("pdf",),
    show: bool = False,
    log=False,
    type='difference',
    inlog = False,
    fig_title = False,
    rows: int | None = None,
    cols: int | None = None,
    compact: bool = False,
    shared_colorbar: bool = False,
    show_average: bool = False,
    cbar_label_fontsize: float | None = None,
    compact_label_fontsize: float | None = None,
    single_legend: bool = True,
    bold: bool = True,
    legend_fontsize: float | None = None,
    tick_length: float | None = None,
):
    """Draw baseline‑vs‑bagged metric differences as 2‑D heat‑maps, where the two axes represent varying parameters."""
    if x_param == y_param:
        raise ValueError("x_param and y_param must differ")
    for p in (x_param, y_param):
        if p not in _NUMERIC_PARAMS:
            raise ValueError(
                f"{p} must be numeric param in {sorted(_NUMERIC_PARAMS)}")
    if not experiments:
        raise ValueError("experiments list is empty")

    _base_override: dict[str, Any] = dict(baseline_overrides or {})
    if baseline_xy is not None:
        _base_override[x_param], _base_override[y_param] = baseline_xy

    if _base_override:
        unknown = set(_base_override) - set(_BASELINE_PARAMS)
        if unknown:
            raise ValueError(
                f"baseline_overrides contains params not in _BASELINE_PARAMS: {sorted(unknown)}"
            )

    ds_runs: dict[str, list[Any]] = defaultdict(list)
    for e in experiments:
        ds_runs[e.dataset_name].append(e)

    # title, somewhat hard coded to the main experiments  --------------------------------------------------------------
    def param_name(param_str):
        if param_str == 'sr':
            return 'sampling rate'
        elif param_str == 'Nbag':
            return 'number of bags'
        elif param_str == 'k':
            return 'k'
    if fig_title is None:
        fixed_global = {}
        for p in _ALL_PARAMS - {x_param, y_param, "bagging_method", "dataset_name"}:
            vals = {getattr(e, p) for e in experiments}
            if len(vals) == 1:
                fixed_global[p] = vals.pop()
        fig_title = " | ".join(f"{k}:{fmt_val(k,v)}" for k, v in fixed_global.items())
    elif fig_title == 'auto':
        x_param_name = param_name(x_param)
        y_param_name = param_name(y_param)

        fig_title = f'Interaction of {x_param_name} and {y_param_name} heatmap. \nBaseline Estimator: {experiments[0].estimator_name.upper()}'


    #automatically set up the layout, fonts -------------------------------------------------
    n_plots = len(ds_runs) + (1 if show_average else 0) + (1 if single_legend else 0)
    if rows is None and cols is None:
        rows, cols = auto_grid(n_plots) if grid and n_plots > 1 else (n_plots, 1)
    if figsize is None:
        figsize = (4 * cols, 3.5 * rows)
    bfs = auto_fontsize(figsize, base_fontsize)
    fs = {
        "title":         bfs * 1.4,
        "label":         bfs * 1.5,
        "xtick":         bfs * 1,
        "ytick":         bfs * 1.15,
        "cbar_tick":     bfs * 0.60,
        "cbar":          cbar_label_fontsize if cbar_label_fontsize is not None else bfs * 1.2 * min(rows, 3) / 2,
        "compact_label": compact_label_fontsize if compact_label_fontsize is not None else bfs * 2,
        "legend":        legend_fontsize if legend_fontsize is not None else (compact_label_fontsize if compact_label_fontsize is not None else bfs * 2),
        "suptitle":      bfs * 2,
    }
    pad = {
        "cbar_to_heatmap":           0.02,   # gap between colorbar and its heatmap
        "cbar_shrink":               0.8,    # colorbar height relative to axes
        "cbar_box_aspect":           20,     # colorbar width (higher = thinner)
        "shared_cbar_shrink":        0.8,    # shared colorbar height relative to axes
        "compact_h_between_rows":    0.005,   # tight_layout h_pad between subplot rows
        "compact_w_between_cols":    0.0,    # tight_layout w_pad between subplot columns
        "compact_wspace":            0.005,    # subplots_adjust wspace after tight_layout
        "compact_xlabel_y":          0,   # figure-wide x-label y position
        "compact_ylabel_x":          0.005,   # figure-wide y-label x position
        "compact_cbar_label_x":      0.98,   # figure-wide colorbar label x position
        "suptitle_y":                1.12,   # suptitle y position above figure
        "title_to_axes":             12,    # gap (pts) between subplot title and heatmap
        "xtick_label_hoffset":       10,     # horizontal shift (pts) for x-tick labels (positive = right)
        "major_tick_length":         10 if tick_length is None else tick_length * 10 / 3.5,
        "minor_tick_length":         5  if tick_length is None else tick_length * 5 / 3.5,
    }
    _weight = "bold" if bold else "normal"
    rc = {
        "axes.titlesize":  fs["title"],
        "axes.titlepad":   pad["title_to_axes"],
        "axes.labelsize":  fs["label"],
        "xtick.labelsize": fs["xtick"],
        "ytick.labelsize": fs["ytick"],
        "font.weight":     _weight,
        "axes.titleweight": _weight,
        "axes.labelweight": _weight,
    }
    if tick_length is not None:
        rc["xtick.major.size"] = tick_length
        rc["ytick.major.size"] = tick_length

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    _xlabel_str = 'Number of Bags (B)' if x_param == 'Nbag' else ('Sampling rate (r)' if x_param == 'sr' else x_param)
    _ylabel_str = 'Number of Bags (B)' if y_param == 'Nbag' else y_param

    key_to_label = {"mse": "MSE", "bias2": "Bias²", "var": "Variance"} #Mapping from metric to their label

    # we do this plot separately for the 3 metrics: MSE, Bias² and Variance
    for met_key in metrics:
        if met_key not in key_to_label:
            raise ValueError(f"Unknown metric '{met_key}'")
        met_label = key_to_label[met_key]

        def _fmt_cbar_label(m, inlog_, log_, type_):
            """Single-line colorbar label, e.g. log₂(MSE_baseline) – log₂(MSE_bagged)."""
            if inlog_:
                if log_:
                    if type_ == 'difference': return f"log₂(log_{m}_baseline) – log₂(log_{m}_bagged)"
                    if type_ == 'baseline':   return f"–log₂(log_{m}_baseline)"
                    if type_ == 'bagged':     return f"–log₂(log_{m}_bagged)"
                else:
                    if type_ == 'difference': return f"log_{m}_baseline – log_{m}_bagged"
                    if type_ == 'baseline':   return f"–log_{m}_baseline"
                    if type_ == 'bagged':     return f"–log_{m}_bagged"
            else:
                if log_:
                    if type_ == 'difference': return f"log₂({m}_baseline) – log₂({m}_bagged)"
                    if type_ == 'baseline':   return f"–log₂({m}_baseline)"
                    if type_ == 'bagged':     return f"–log₂({m}_bagged)"
                else:
                    if type_ == 'difference': return f"{m}_baseline – {m}_bagged"
                    if type_ == 'baseline':   return f"{m}_baseline"
                    if type_ == 'bagged':     return f"{m}_bagged"

        with plt.rc_context(rc):
            fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
            axes = axes.flatten()
            # Average goes right after the last dataset, not in the bottom-right corner
            n_used = len(ds_runs) + (1 if show_average else 0) + (1 if single_legend else 0)
            for ax in axes[n_used:]:
                ax.axis("off")

            colorbars: list[tuple[int, Any]] = []  # (subplot_index, cbar) for compact post-processing
            _ims: list = []  # (im, data) pairs collected for shared_colorbar post-processing
            _xs_sorted: list = []  # axis tick values saved from last dataset (reused for average)
            _ys_sorted: list = []
            for ax_idx, (ax, (ds_name, runs)) in enumerate(zip(axes, sorted(ds_runs.items()))):

                # we separate baseline and bagged experiments, because later we will need to compare them
                baseline_lookup: dict[tuple, Any] = {}
                bagged_list = []
                for r in runs:
                    base_key = tuple(getattr(r, p) for p in _BASELINE_PARAMS) #extract individual experiment params
                    if r.bagging_method is None: # if it's an experiment with the baseline estimator
                        baseline_lookup[base_key] = r #we save the experiment here, together with the relevant params in this dictionary
                    else:
                        bagged_list.append(r) #othrwise we save it in the bagged experiment list

                xs_sorted = sorted({getattr(r, x_param) for r in bagged_list}) #extract the relevant bagged experiment params x axis, sort them using the natural ordering of these params (usually increasing numbers)
                ys_sorted = sorted({getattr(r, y_param) for r in bagged_list}) #extract the relevant bagged experiment params y axis, sort them using the natural ordering of these params (usually increasing numbers)
                #Sometimes we want to have them in decreasing or increasing order
                if reverse_x:
                    xs_sorted = xs_sorted[::-1]
                if reverse_y:
                    ys_sorted = ys_sorted[::-1]
                xs_map = {v: i for i, v in enumerate(xs_sorted)}
                ys_map = {v: i for i, v in enumerate(ys_sorted)}
                data = np.full((len(ys_sorted), len(xs_sorted)), np.nan) #Would be function values for the map of 2d parameter value combinations

                if baseline_xy is not None and baseline_lookup:
                    bx, by = baseline_xy
                    bx_vals = {getattr(br, x_param) for br in baseline_lookup.values()}
                    by_vals = {getattr(br, y_param) for br in baseline_lookup.values()}
                    if bx not in bx_vals or by not in by_vals:
                        raise ValueError(
                            f"No baseline runs found at {x_param}={bx}, {y_param}={by} for dataset '{ds_name}'."
                        )

                #now we're ready to figure out the 2 function's values to build the color map
                for r in bagged_list:
                    # Build the baseline key, overriding x/y (and anything else in baseline_overrides)
                    base_key = tuple(
                        (_base_override[p] if p in _base_override else getattr(r, p))
                        for p in _BASELINE_PARAMS
                    )
                    base_run = baseline_lookup.get(base_key)
                    if base_run is None:
                        continue  # no matching baseline → leave NaN
                    #A lot of decisions based on function settings on how to measure difference, if we even want to measure difference. We can also just build a heatmap from the bagged or baseline results individually.
                    if inlog:
                        if log:
                            if type == 'difference':
                                diff = np.log2(getattr(base_run, f"log_total_{met_key}")) - np.log2(getattr(r, f"log_total_{met_key}"))
                            elif type == 'baseline':
                                diff = -np.log2(getattr(base_run, f"log_total_{met_key}"))
                            elif type == 'bagged':
                                diff = -np.log2(getattr(r, f"log_total_{met_key}"))
                        else:
                            if type == 'difference':
                                diff = getattr(base_run, f"log_total_{met_key}") - getattr(r, f"log_total_{met_key}")
                            elif type == 'baseline':
                                diff = -getattr(base_run, f"log_total_{met_key}")
                            elif type == 'bagged':
                                diff = -getattr(r, f"log_total_{met_key}")
                    else:
                        if log:
                            if type == 'difference':
                                diff = np.log2(getattr(base_run, f"total_{met_key}")) - np.log2(getattr(r, f"total_{met_key}"))
                            elif type == 'baseline':
                                diff = -np.log2(getattr(base_run, f"total_{met_key}"))
                            elif type == 'bagged':
                                diff = -np.log2(getattr(r, f"total_{met_key}"))
                        else:
                            if type == 'difference':
                                diff = getattr(base_run, f"total_{met_key}") - getattr(r, f"total_{met_key}")
                            elif type == 'baseline':
                                diff = getattr(base_run, f"total_{met_key}")
                            elif type == 'bagged':
                                diff = getattr(r, f"total_{met_key}")
                    xi = xs_map[getattr(r, x_param)]
                    yi = ys_map[getattr(r, y_param)]
                    data[yi, xi] = diff #fill up the function values
                _xs_sorted, _ys_sorted = xs_sorted, ys_sorted  # saved for average subplot

                #setup colormap style
                vmax = np.nanmax(np.abs(data)) or 1.0
                if type == 'difference' or log:
                    im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, origin="lower")
                else:
                    im = ax.imshow(data, cmap="Reds", vmin=0, vmax=vmax, origin="lower")
                _ims.append((im, data))

                # ticks --------------------------------------------------
                ax.set_xticks(range(len(xs_sorted)))
                ax.set_xticklabels([
                    fmt_val(x_param, v) if (i % label_every == 0 or i in {0, len(xs_sorted)-1}) else ""
                    for i, v in enumerate(xs_sorted)
                ], rotation=45, ha="right")
                if pad["xtick_label_hoffset"]:
                    dx = pad["xtick_label_hoffset"] / 72  # pts to inches
                    offset = ScaledTranslation(dx, 0, fig.dpi_scale_trans)
                    for lbl in ax.get_xticklabels():
                        lbl.set_transform(lbl.get_transform() + offset)
                ax.set_yticks(range(len(ys_sorted)))
                ax.set_yticklabels([
                    fmt_val(y_param, v) if (i % label_every == 0 or i in {0, len(ys_sorted)-1}) else ""
                    for i, v in enumerate(ys_sorted)
                ])
                emphasize_labeled_ticks(ax, major_length=pad["major_tick_length"], minor_length=pad["minor_tick_length"])
                if x_param == 'Nbag':
                    ax.set_xlabel('Number of Bags (B)')
                elif x_param == 'sr':
                    ax.set_xlabel('Sampling rate (r)')
                else:
                    ax.set_xlabel(x_param)
                if y_param == 'Nbag':
                    ax.set_ylabel('Number of Bags (B)')
                else:
                    ax.set_ylabel(y_param)
                ax.set_title(ds_name)
                if not shared_colorbar:
                    cbar = fig.colorbar(im, ax=ax, shrink=pad["cbar_shrink"], pad=pad["cbar_to_heatmap"])
                    cbar.ax.tick_params(labelsize=fs["cbar_tick"])
                    colorbars.append((ax_idx, cbar))
                    cbar.ax.set_ylabel(_fmt_cbar_label(met_label, inlog, log, type), fontsize=fs["cbar"])

            if show_average and _ims and _xs_sorted:
                avg_data = np.nanmean([d for _, d in _ims], axis=0)
                ax_avg = axes[len(ds_runs)]
                vmax_avg = np.nanmax(np.abs(avg_data)) or 1.0
                if type == 'difference' or log:
                    im_avg = ax_avg.imshow(avg_data, cmap=cmap, vmin=-vmax_avg, vmax=vmax_avg, origin="lower")
                else:
                    im_avg = ax_avg.imshow(avg_data, cmap="Reds", vmin=0, vmax=vmax_avg, origin="lower")
                _ims.append((im_avg, avg_data))
                ax_avg.set_xticks(range(len(_xs_sorted)))
                ax_avg.set_xticklabels([
                    fmt_val(x_param, v) if (i % label_every == 0 or i in {0, len(_xs_sorted)-1}) else ""
                    for i, v in enumerate(_xs_sorted)
                ], rotation=45, ha="right")
                if pad["xtick_label_hoffset"]:
                    dx = pad["xtick_label_hoffset"] / 72
                    offset = ScaledTranslation(dx, 0, fig.dpi_scale_trans)
                    for lbl in ax_avg.get_xticklabels():
                        lbl.set_transform(lbl.get_transform() + offset)
                ax_avg.set_yticks(range(len(_ys_sorted)))
                ax_avg.set_yticklabels([
                    fmt_val(y_param, v) if (i % label_every == 0 or i in {0, len(_ys_sorted)-1}) else ""
                    for i, v in enumerate(_ys_sorted)
                ])
                emphasize_labeled_ticks(ax_avg, major_length=pad["major_tick_length"], minor_length=pad["minor_tick_length"])
                if x_param == 'Nbag':
                    ax_avg.set_xlabel('Number of Bags (B)')
                elif x_param == 'sr':
                    ax_avg.set_xlabel('Sampling rate (r)')
                else:
                    ax_avg.set_xlabel(x_param)
                if y_param == 'Nbag':
                    ax_avg.set_ylabel('Number of Bags (B)')
                else:
                    ax_avg.set_ylabel(y_param)
                ax_avg.set_title("Average")
                if not shared_colorbar:
                    cbar_avg = fig.colorbar(im_avg, ax=ax_avg, shrink=pad["cbar_shrink"], pad=pad["cbar_to_heatmap"])
                    cbar_avg.ax.tick_params(labelsize=fs["cbar_tick"])
                    colorbars.append((len(ds_runs), cbar_avg))
                    cbar_avg.ax.set_ylabel(_fmt_cbar_label(met_label, inlog, log, type), fontsize=fs["cbar"])

            if shared_colorbar and _ims:
                # Compute global colour range across all datasets
                _all_data = [d for _, d in _ims]
                if type == 'difference' or log:
                    _gvmax = max(
                        (np.nanmax(np.abs(d)) for d in _all_data if np.any(~np.isnan(d))),
                        default=1.0,
                    ) or 1.0
                    _gvmin = -_gvmax
                else:
                    _gvmax = max(
                        (np.nanmax(d) for d in _all_data if np.any(~np.isnan(d))),
                        default=1.0,
                    ) or 1.0
                    _gvmin = 0.0
                for _im, _ in _ims:
                    _im.set_clim(_gvmin, _gvmax)
                _shared_cbar = fig.colorbar(
                    _ims[-1][0],
                    ax=axes[:n_used].tolist(),
                    shrink=pad["shared_cbar_shrink"],
                )
                _shared_cbar.ax.tick_params(labelsize=fs["cbar_tick"])
                _shared_cbar.ax.set_ylabel(_fmt_cbar_label(met_label, inlog, log, type), fontsize=fs["cbar"])

            if single_legend:
                ax_leg_idx = n_used - 1
                ax_leg = axes[ax_leg_idx]
                ax_leg.axis("off")
                used_cmap = plt.get_cmap(cmap if (type == 'difference' or log) else "Reds")
                handles = [
                    plt.Rectangle((0, 0), 0.6, 1.2, color=used_cmap(0.95)),
                    plt.Rectangle((0, 0), 0.6, 1.2, color=used_cmap(0.05)),
                ]
                ax_leg.legend(handles, ["Bagged\nbetter", "Baseline\nbetter"], loc="center",
                              prop={"size": fs["legend"]}, frameon=False,
                              handlelength=1.2*1.25, handleheight=1.0*1.25)

            if fig_title:
                fig.suptitle(fig_title, y=pad["suptitle_y"], fontsize=fs["suptitle"])

            if compact:
                active = list(range(n_used - (1 if single_legend else 0)))
                # Which subplots are leftmost (keep y tick labels) / bottom (keep x tick labels)
                show_yticks = {i for i in active if i % cols == 0}
                show_xticks = set()
                for i in active:
                    col_i, row_i = i % cols, i // cols
                    has_below = any(j % cols == col_i and j // cols > row_i for j in active)
                    if not has_below:
                        show_xticks.add(i)
                for i in active:
                    ax = axes[i]
                    ax.set_ylabel("")   # replaced by single figure-wide label
                    ax.set_xlabel("")   # replaced by single figure-wide label
                    if i not in show_yticks:
                        ax.tick_params(labelleft=False)
                    if i not in show_xticks:
                        ax.tick_params(labelbottom=False)
                # Single figure-wide axis labels — positioned close to the subplots
                fig.text(0.5, pad["compact_xlabel_y"], _xlabel_str, ha='center', va='top',
                         fontsize=fs["compact_label"], clip_on=False)
                fig.text(pad["compact_ylabel_x"], 0.5, _ylabel_str, ha='right', va='center',
                         rotation=90, fontsize=fs["compact_label"], clip_on=False)
                if not shared_colorbar:
                    for _, cbar in colorbars:
                        cbar.ax.set_ylabel("")
                        cbar.ax.set_box_aspect(pad["cbar_box_aspect"])
                    fig.text(
                        pad["compact_cbar_label_x"], 0.5,
                        _fmt_cbar_label(met_label, inlog, log, type),
                        rotation=90, va='center', ha='left',
                        transform=fig.transFigure, clip_on=False,
                        fontsize=fs["cbar"],
                    )
                fig.tight_layout(h_pad=pad["compact_h_between_rows"], w_pad=pad["compact_w_between_cols"])
                fig.subplots_adjust(wspace=pad["compact_wspace"])
            else:
                fig.tight_layout()
            if log:
                logsavename = '_log'
            else:
                logsavename = ''
            if inlog:
                inlogsavename = '_inlog'
            else:
                inlogsavename = ''
            for fmt in formats:
                out = save_dir / f"{save_prefix}_{met_key}_{type}{logsavename}{inlogsavename}.{fmt}"
                fig.savefig(out, bbox_inches="tight")
                print(f"[SAVED] {out}")
            if show:
                plt.show()
            else:
                plt.close(fig)