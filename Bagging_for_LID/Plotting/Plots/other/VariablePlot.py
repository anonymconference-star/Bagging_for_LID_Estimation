from __future__ import annotations
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from Bagging_for_LID.Plotting.plotting_helpers import *
from Bagging_for_LID.Plotting.naming_helpers import *

#define the variable class parameter, for which this function can sweep over results
_NUMERIC_PARAMS = {"n", "k", "sr", "Nbag", "lid", "dim", "t"}
_BOOL_OR_STR_PARAMS = {
    "pre_smooth",
    "post_smooth",
    "estimator_name",
    "bagging_method",
    "submethod_0",
    "submethod_error",
}
_ALL_PARAMS = _NUMERIC_PARAMS | _BOOL_OR_STR_PARAMS

def plot_experiment_metric_curves(
    experiments: Sequence[Any],
    *,
    vary_param: str | None = None,
    log: bool = False,
    label_every: int = 1,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    base_fontsize: int | float | None = None,
    save_prefix: str = "exp_metric_plot",
    save_dir: str | Path = "./plots",
    formats: tuple[str, ...] = ("pdf",),
    show: bool = False,
    xscale='log',
    fig_title = False
):
    """Plot total MSE / Bias² / Variance on the y-axis for *vary_param* on the x-axis, for each dataset.

    Produces **three** files: ``<save_name>_mse.pdf``, ``…_bias2.pdf``, ``…_var.pdf``.
    Each subplot shows one dataset, with one curve per *method signature*.
    """
    if not experiments:
        raise ValueError("experiments is empty")

    # ── bundle by dataset ───────────────────────────────────────────
    ds_map: dict[str, list[Any]] = defaultdict(list)
    for exp in experiments:
        ds_map[exp.dataset_name].append(exp)

    # ── determine varying numeric parameter ────────────────────────
    numeric_params = _NUMERIC_PARAMS
    if vary_param is None:
        candidates = []
        for p in numeric_params:
            for exps in ds_map.values():
                if len({getattr(e, p) for e in exps}) > 1:
                    candidates.append(p)
                    break
        if not candidates:
            raise ValueError("No numeric parameter varies across experiments.")
        if len(candidates) > 1:
            raise ValueError(
                "More than one numeric parameter varies. Specify vary_param="
                + ", ".join(candidates)
            )
        vary_param = candidates[0]
    elif vary_param not in numeric_params:
        raise ValueError("vary_param must be one of " + ", ".join(sorted(numeric_params)))

    # ── build figure‑level title: params identical across *all* exps ─────
    fixed_global = {}
    for p in _ALL_PARAMS - {vary_param, "dataset_name"}:
        vals = {getattr(e, p) for e in experiments}
        if len(vals) == 1:
            fixed_global[p] = vals.pop()
    if fig_title:
        fig_title = " | ".join(f"{k}:{fmt_val(k, v)}" for k, v in fixed_global.items())

    # ── helper to generate subplot curves ───────────────────────────
    def _signature(exp: Any) -> tuple[tuple[str, Any], ...]:
        """Tuple of (param,value) for all *non‑varying* params."""
        items = []
        for p in _ALL_PARAMS - {vary_param, "dataset_name"}:
            items.append((p, getattr(exp, p)))
        return tuple(sorted(items))

    # colour cycle
    prop_cycle = plt.rcParams.get("axes.prop_cycle").by_key()["color"]

    # ── iterate over metrics ────────────────────────────────────────
    metrics = [("mse", "MSE"), ("bias2", "Bias²"), ("var", "Variance")]

    rows, cols = auto_grid(len(ds_map)) if grid and len(ds_map) > 1 else (len(ds_map), 1)
    if figsize is None:
        figsize = (4 * cols, 3 * rows)
    bfs = auto_fontsize(figsize, base_fontsize)
    rc = {
        "axes.titlesize": bfs * 1.2,
        "axes.labelsize": bfs,
        "xtick.labelsize": bfs * 0.9,
        "ytick.labelsize": bfs * 0.9,
        "legend.fontsize": bfs * 0.8,
    }

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for met_key, met_label in metrics:
        with plt.rc_context(rc):
            fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=False)
            axes = np.asarray(axes).reshape(-1)
            for ax in axes[len(ds_map):]:
                ax.axis("off")

            for ax, (ds_name, exps) in zip(axes, sorted(ds_map.items())):
                # bucket by method signature
                sig_map: dict[tuple[tuple[str, Any], ...], list[Any]] = defaultdict(list)
                for e in exps:
                    sig_map[_signature(e)].append(e)

                # find params that actually differ across signatures ➔ legend
                diff_params = set()
                sig_values = list(sig_map.keys())
                if len(sig_values) > 1:
                    for p_idx in range(len(sig_values[0])):
                        p_name = sig_values[0][p_idx][0]
                        if {sig[p_idx][1] for sig in sig_values}.__len__() > 1:
                            diff_params.add(p_name)

                for i, (sig, runs) in enumerate(sorted(sig_map.items(), key=lambda item: str(item[0]))):
                    runs.sort(key=lambda r: getattr(r, vary_param))
                    xs = [float(getattr(r, vary_param)) for r in runs]
                    ys = [getattr(r, f"total_{met_key}") for r in runs]
                    if log:
                        ys = [np.log10(y) for y in ys]
                    label = " | ".join(f"{p}:{fmt_val(p, v)}" for p, v in sig if p in diff_params) or "default"
                    label = modify_label(label)
                    ax.plot(
                        xs,
                        ys,
                        marker="o",
                        markersize=3,
                        label=label,
                        color=prop_cycle[i % len(prop_cycle)]
                    )

                # x‑ticks & labels
                xs_all = sorted({float(getattr(r, vary_param)) for r in exps})
                if xscale == "ordinal":
                    x_pos = {v: i for i, v in enumerate(xs_all)}
                else:  # 'linear' or 'log'
                    x_pos = {v: v for v in xs_all}
                if xscale == "log":
                    ax.set_xscale("log", base=10)
                ax.set_xticks([x_pos[v] for v in xs_all])
                tick_lbls = []
                for idx, val in enumerate(xs_all):
                    show = idx == 0 or idx == len(xs_all) - 1 or idx % label_every == 0
                    tick_lbls.append(fmt_val(vary_param, val) if show else "")
                ax.set_xticklabels(tick_lbls)
                ax.set_xlabel(vary_param)
                ax.set_ylabel("log₁₀(" + met_label + ")" if log else met_label)
                ax.set_title(ds_name)
                ax.grid(True, linestyle="--", alpha=0.4)
                ax.legend()

            if fig_title:
                fig.suptitle(fig_title, y=1.02, fontsize=bfs * 1.2)
            fig.tight_layout()

            for fmt in formats:
                out = save_dir / f"{save_prefix}_{met_key}.{fmt}"
                fig.savefig(out, bbox_inches="tight")
                print(f"[SAVED] {out}")
            if show:
                plt.show()
            else:
                plt.close(fig)
