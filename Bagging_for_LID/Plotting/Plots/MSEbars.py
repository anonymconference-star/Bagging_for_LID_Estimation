import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union, Mapping, Any
import matplotlib.pyplot as plt
import numpy as np
import warnings
from matplotlib.transforms import ScaledTranslation
from matplotlib.ticker import MaxNLocator
from Bagging_for_LID.Plotting.plotting_helpers import *
##############################################################################################################################MSE BAR PLOT#############################

#allow the testing of MSE changes when one of these are changing
ALLOWED_PARAMS = {
    "n", "k", "sr", "Nbag", "lid", "dim", "pre_smooth", "post_smooth",
    "t", "estimator_name", "bagging_method", "submethod_0", "submethod_error",
}

def plot_experiment_mse_bars(
    experiments: Sequence[Any],  # LID_experiments
    *,
    vary_param: str | None = None,
    grid: bool = True,
    figsize: Tuple[float, float] | None = None,
    base_fontsize: int | float | None = None,
    colors: Tuple[str, str] = ("tab:green", "tab:red"),
    label_every: int = 1,
    save_prefix: str = "exp_mse_bar_plot",
    save_dir: str | Path = "./Output",
    formats: Tuple[str, ...] = ("pdf",),
    show: bool = False,
    xlabel: str | None = None,
    title: str | None = None,
    fig_title: bool = False,
    n_rows: int | None = None,
    n_cols: int | None = None,
    verbose: bool = False,
    compact: bool = False,
    show_average: bool = False,
    compact_label_fontsize: float | None = None,
    single_legend: bool = True,
    bold: bool = True,
    legend_fontsize: float | None = None,
    tick_length: float | None = None,
):
    # Sometimes we want to have the baseline before or after the bagged variants for reference.
    # But it's a bit annoying to handle this automatically, so we just redefine it as a bagged experiment, with 1 bag that has a sr of 1, as that would be identical.
    #for experiment in experiments:
    #    if experiment.bagging_method == None:
    #        experiment.sr = 1
    #        experiment.bagging_method = 'bag'
    #        experiment.Nbag = int(1)

    #This is just for different labeling of different varying params (where to cut the decimals)
    if vary_param == 'Nbag':
        deci = 0
    else:
        deci = 2

    #Selects a class attribute
    def _get(exp, attr, default=None):
        return getattr(exp, attr, default)

    #separate experiments based on dataset
    by_ds: dict[str, list[Any]] = defaultdict(list)
    for exp in experiments:
        by_ds[exp.dataset_name].append(exp)

    #This here automatically tries to figure out which class parameter is changing (e.g., sampling rate, number of bags, but can be something else from ALLOWED_PARAMS), in case it is not prespecified.
    if vary_param is None:
        diffs = []
        for p in ALLOWED_PARAMS:
            if any(len({_get(e, p) for e in exps}) > 1 for exps in by_ds.values()):
                diffs.append(p)
        if not diffs:
            raise ValueError("All experiments share identical parameters – nothing varies.")
        if len(diffs) > 1:
            raise ValueError(
                "More than one parameter varies across experiments. "
                "Specify which one with `vary_param=`.  Varying params: " + ", ".join(diffs)
            )
        vary_param = diffs[0]
    elif vary_param not in ALLOWED_PARAMS:
        raise ValueError(f"'{vary_param}' not in allowed parameters: {ALLOWED_PARAMS}")

    #These functions handle the annoying case where the input had multiple baseline experiments (with maybe diffrerent sr or Nbag which are also irrelevant)
    def _params_consistent(ref, e, ignore: set[str]):
        for p in (ALLOWED_PARAMS - ignore):
            ref_val = _get(ref, p)
            e_val = _get(e, p)
            if p in {"sr", "Nbag"}:
                if _get(ref, "bagging_method") is None or _get(e, "bagging_method") is None:
                    continue
            if ref_val != e_val:
                return False, p
        return True, None

    def _pick_single_baseline(baselines: list[Any]):
        if not baselines:
            return None
        sr1 = [b for b in baselines if _get(b, "sr") == 1]
        if sr1:
            return sr1[0]
        nb1 = [b for b in baselines if _get(b, "Nbag") == 1]
        if nb1:
            return nb1[0]
        return baselines[0]


    #across all the experiments, we to figure out if other than the (possibly prespecified varying parameter) are the other ones changing (which would invalidate the experimet, as this signals that the input data was wrong)
    #then we extract a more focused data dictionary, which only cares about the numbers necessary for plotting
    data_by_ds: dict[str, list[dict]] = defaultdict(list)

    for ds_name, exps in by_ds.items():
        if not exps:
            continue

        # pick a non-baseline as reference if possible, else any
        non_base = [e for e in exps if _get(e, "bagging_method") is not None]
        ref = non_base[0] if non_base else exps[0]

        # validate outside of variable consistency vs reference
        for e in exps:
            ok, p_bad = _params_consistent(ref, e, ignore={vary_param})
            if not ok:
                if verbose:
                    warnings.warn(
                        f"Dataset '{ds_name}': parameter '{p_bad}' differs while varying '{vary_param}'. "
                        "Proceeding anyway.",
                        category=UserWarning,
                        stacklevel=2,
                    )
                    print(f"Warning: Dataset '{ds_name}': parameter '{p_bad}' differs while varying '{vary_param}'. {_get(e, p_bad)} differs from {_get(ref, p_bad)}"
                        "Proceeding anyway.")

        baselines = [e for e in exps if _get(e, "bagging_method") is None]
        variants = [e for e in exps if _get(e, "bagging_method") is not None]

        # If varying Nbag or sr: include exactly one baseline (placed leftmost/rightmost)
        if vary_param in {"Nbag", "sr"}:
            chosen_base = _pick_single_baseline(baselines)
            if chosen_base is not None:
                label = "Baseline"
                # Force sort position via sentinel sort_key
                if vary_param == "Nbag":
                    sort_key = (float("-inf"), "baseline")  # leftmost
                    x_val_for_label = label
                else:  # vary_param == "sr"
                    sort_key = (float("+inf"), "baseline")  # rightmost
                    x_val_for_label = label
                if np.array([_get(e, 'sr')!=1 for e in variants]).all():
                    data_by_ds[ds_name].append({
                        "x_val": x_val_for_label,
                        "sort_key": sort_key,
                        "bias2": _get(chosen_base, "total_bias2"),
                        "var": _get(chosen_base, "total_var"),
                    })

        else:
            # Not varying Nbag/sr: include ALL baselines, ignore their sr/Nbag for labeling
            for i, b in enumerate(baselines, start=1):
                label = "Baseline" if i == 1 else f"Baseline({i})"
                # Put baselines first to the left in stable order
                data_by_ds[ds_name].append({
                    "x_val": label,
                    "sort_key": (-1, f"baseline{i:03d}"),
                    "bias2": _get(b, "total_bias2"),
                    "var": _get(b, "total_var"),
                })

        # Add the variant (bagged) experiments; sort by actual varying param
        for e in variants:
            x_val = _get(e, vary_param)
            if isinstance(x_val, (int, float)):
                sort_key = (0, float(x_val))
                x_lab = f"{x_val:.{deci}f}"
            else:
                sort_key = (0, str(x_val))
                x_lab = str(x_val)
            data_by_ds[ds_name].append({
                "x_val": x_lab,
                "sort_key": sort_key,
                "bias2": _get(e, "total_bias2"),
                "var": _get(e, "total_var"),
            })
    # Figure layout
    ds_names = sorted(data_by_ds)
    n_plots = len(ds_names) + (1 if show_average else 0) + (1 if single_legend else 0)
    if n_rows is None and n_cols is None:
        n_rows, n_cols = (auto_grid(n_plots)
                          if grid and n_plots > 1 else (n_plots, 1))
    #default figsize, fontsize, global title, label for varying parameter we try to automatically set these up well enough
    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)
    if title is None:
        if vary_param == 'sr':
            paranname = 'sampling rate'
        elif vary_param == 'Nbag':
            paranname = 'number of bags'
        else:
            paranname = vary_param
        title = f"MSE decompositions for changing {paranname}. \nBaseline Estimator: {_get(experiments[0],'estimator_name').upper()}"
    if xlabel is None:
        if vary_param == 'Nbag':
            xlabel = 'Number of Bags (B)'
        elif vary_param == 'sr':
            xlabel = 'Sampling rate (r)'
        else:
            xlabel = f"{vary_param}"
    bfs = auto_fontsize(figsize, base_fontsize)
    fs = {
        "title":         bfs * 1.4, #1.4
        "label":         bfs * 1.5,
        "xtick":         bfs * 1,
        "ytick":         bfs * 0.75,
        "legend":        bfs * 0.6,
        "compact_label": compact_label_fontsize if compact_label_fontsize is not None else bfs * 2,
        "legend":        legend_fontsize if legend_fontsize is not None else (compact_label_fontsize if compact_label_fontsize is not None else bfs * 2),
    }
    pad = {
        "xlabel_to_ticks":           0,      # labelpad on x-axis labels
        "compact_h_between_rows":    0.8,  # tight_layout h_pad between subplot rows
        "compact_w_between_cols":    0.0,    # tight_layout w_pad between subplot columns
        "compact_xlabel_y":          0,      # figure-wide x-label y position (0=bottom edge)
        "compact_ylabel_x":          0.005,  # figure-wide y-label x position (0=left edge)
        "title_to_axes":             8,     # gap (pts) between subplot title and axes
        "xtick_label_hoffset":       bfs * 10 / 14,  # horizontal shift (pts) for x-tick labels, scales with fontsize
        "ytick_nbins":               7,      # target number of y-axis (MSE) tick marks
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
        "legend.fontsize": fs["legend"],
        "font.weight":     _weight,
        "axes.titleweight": _weight,
        "axes.labelweight": _weight,
    }
    if tick_length is not None:
        rc["xtick.major.size"] = tick_length
        rc["ytick.major.size"] = tick_length

    #we will save the plot here
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)


    with plt.rc_context(rc):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=False)
        axes = np.asarray(axes).reshape(-1)
        # Average goes right after the last dataset, not in the bottom-right corner
        n_used = len(ds_names) + (1 if show_average else 0) + (1 if single_legend else 0)
        for ax in axes[n_used:]:
            ax.axis("off")

        for ax, ds in zip(axes, ds_names):
            entries = sorted(data_by_ds[ds], key=lambda d: d["sort_key"]) #We sort the results along the x-axis in increasing order of the varying parameter, (e.g.,like sr, number of bags).
            labels = [e["x_val"] for e in entries]
            b_vals = [e["bias2"] for e in entries]
            v_vals = [e["var"] for e in entries]
            x = np.arange(len(entries))
            ax.bar(x, b_vals, width=0.6, color=colors[0], label="Bias²")
            ax.bar(x, v_vals, width=0.6, bottom=b_vals, color=colors[1], label="Variance")
            ax.set_xticks(x)
            disp_lbl = [lbl if (i % label_every == 0 or i == len(labels) - 1) else "" for i, lbl in enumerate(labels)]
            disp_lbl = [lbls if lbls is not None else experiments[0].estimator_name for lbls in disp_lbl] #The none case corresponds to something that generally shouldn't happen, we will just plot the estimator name in this case
            ax.set_xticklabels(disp_lbl, rotation=45, ha="right")
            if pad["xtick_label_hoffset"]:
                dx = pad["xtick_label_hoffset"] / 72
                offset = ScaledTranslation(dx, 0, fig.dpi_scale_trans)
                for lbl in ax.get_xticklabels():
                    lbl.set_transform(lbl.get_transform() + offset)
            emphasize_labeled_ticks(ax, axis="x", major_length=pad["major_tick_length"], minor_length=pad["minor_tick_length"])
            ax.set_ylabel("MSE") #This plot is fixed to the mse
            ax.set_xlabel(f"{xlabel}", labelpad=pad["xlabel_to_ticks"]) #
            ax.set_title(f"{ds}")
            ax.yaxis.set_major_locator(MaxNLocator(nbins=pad["ytick_nbins"]))
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            if not single_legend:
                ax.legend(loc="upper right", prop={"size": fs["legend"]})

        if show_average:
            ax_avg = axes[len(ds_names)]
            all_sort_keys = sorted({e["sort_key"] for entries in data_by_ds.values() for e in entries})
            # Normalise each dataset by its own peak total-MSE bar so all contribute on [0,1]
            ds_norm = {
                ds: (max(e["bias2"] + e["var"] for e in data_by_ds[ds]) or 1.0)
                for ds in ds_names if data_by_ds[ds]
            }
            avg_b, avg_v, avg_labels = [], [], []
            for sk in all_sort_keys:
                b_list, v_list, lbl = [], [], None
                for ds in ds_names:
                    for e in data_by_ds[ds]:
                        if e["sort_key"] == sk:
                            norm = ds_norm.get(ds, 1.0)
                            b_list.append(e["bias2"] / norm)
                            v_list.append(e["var"] / norm)
                            if lbl is None:
                                lbl = e["x_val"]
                avg_b.append(float(np.nanmean(b_list)) if b_list else np.nan)
                avg_v.append(float(np.nanmean(v_list)) if v_list else np.nan)
                avg_labels.append(lbl or "")
            avg_b = np.array(avg_b)
            avg_v = np.array(avg_v)
            x_avg = np.arange(len(all_sort_keys))
            ax_avg.bar(x_avg, avg_b, width=0.6, color=colors[0], label="Bias²")
            ax_avg.bar(x_avg, avg_v, width=0.6, bottom=avg_b, color=colors[1], label="Variance")
            ax_avg.set_xticks(x_avg)
            disp_lbl = [lbl if (i % label_every == 0 or i == len(avg_labels) - 1) else "" for i, lbl in enumerate(avg_labels)]
            ax_avg.set_xticklabels(disp_lbl, rotation=45, ha="right")
            if pad["xtick_label_hoffset"]:
                dx = pad["xtick_label_hoffset"] / 72
                offset = ScaledTranslation(dx, 0, fig.dpi_scale_trans)
                for lbl in ax_avg.get_xticklabels():
                    lbl.set_transform(lbl.get_transform() + offset)
            emphasize_labeled_ticks(ax_avg, axis="x", major_length=pad["major_tick_length"], minor_length=pad["minor_tick_length"])
            ax_avg.set_ylabel("Normalised MSE")
            ax_avg.set_xlabel(f"{xlabel}", labelpad=pad["xlabel_to_ticks"])
            ax_avg.set_title("Average")
            ax_avg.yaxis.set_major_locator(MaxNLocator(nbins=pad["ytick_nbins"]))
            ax_avg.grid(axis="y", linestyle="--", alpha=0.4)
            if not single_legend:
                ax_avg.legend(loc="upper right", prop={"size": fs["legend"]})

        if single_legend:
            ax_leg = axes[n_used - 1]
            ax_leg.axis("off")
            handles = [
                plt.Rectangle((0, 0), 1, 1, color=colors[1]),
                plt.Rectangle((0, 0), 1, 1, color=colors[0]),
            ]
            ax_leg.legend(handles, ["Variance", "Bias²"], loc="center",
                          prop={"size": fs["legend"]}, frameon=False,
                          handlelength=1.2*1.25, handleheight=1.0*1.25)

        if fig_title:
            fig.suptitle(f'{title}')

        if compact:
            active = list(range(n_used - (1 if single_legend else 0)))
            # Determine which subplots sit at the bottom of their column (keep x tick labels)
            show_xticks = set()
            for i in active:
                col, row = i % n_cols, i // n_cols
                has_below = any(j % n_cols == col and j // n_cols > row for j in active)
                if not has_below:
                    show_xticks.add(i)
            for i in active:
                ax = axes[i]
                ax.set_ylabel("")   # removed; replaced by single figure-wide label
                ax.set_xlabel("")   # removed; replaced by single figure-wide label
                if i not in show_xticks:
                    ax.tick_params(labelbottom=False)
                # y-axis tick numbers kept on every subplot (point 2)
            fig.text(0.5, pad["compact_xlabel_y"], xlabel, ha='center', va='top', fontsize=fs["compact_label"], clip_on=False)
            fig.text(pad["compact_ylabel_x"], 0.5, "MSE", ha='right', va='center', rotation=90, fontsize=fs["compact_label"], clip_on=False)
            fig.tight_layout(h_pad=pad["compact_h_between_rows"], w_pad=pad["compact_w_between_cols"])
        else:
            fig.tight_layout()
        for fmt in formats:
            out = save_dir / f"{save_prefix}.{fmt}"
            fig.savefig(out, bbox_inches="tight")
            print(f"[SAVED] {out}")
        if show:
            plt.show()
        else:
            plt.close(fig)
    return fig