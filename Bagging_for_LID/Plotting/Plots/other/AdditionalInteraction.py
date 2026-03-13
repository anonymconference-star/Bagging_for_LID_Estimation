from __future__ import annotations
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Sequence, Tuple, Union, Mapping
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
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

def _get_vec_value(
    e: Any, *,
    value_attr: str,
    value_index: int,
) -> float:
    vec = getattr(e, value_attr)
    if vec is None:
        return np.nan
    # accept list/np array/etc.
    try:
        return float(vec[value_index])
    except Exception:
        return np.nan


def _default_value_label(value_attr: str, value_index: int) -> str:
    # For point_bag_avg_knn_dists, index maps to neighbor rank (index 0 -> 1-NN).
    if value_attr == "point_bag_avg_knn_dists":
        if value_index == -1:
            return "avg k-NN distance"
        return f"avg {(value_index + 1)}-NN distance"
    return f"{value_attr}[{value_index}]"


def plot_experiment_attr(
    experiments: Sequence[Any],
    *,
    # axes
    x_param: str = "k",
    y_param: str = "sr",
    reverse_x: bool = False,
    reverse_y: bool = False,

    # value
    value_attr: str = "point_bag_avg_knn_dists",
    value_index: int = -1,
    value_label: str | None = None,

    # comparison
    mode: str = "value",  # {"value","difference"}
    compare_param: str = "bagging_method",
    compare_values: Tuple[Any, Any] = (None, "bag"),  # (A,B)
    select_value: Any | None = None,  # only for mode="value"
    diff_kind: str = "difference",  # {"difference","log2"}

    # plot kind
    plot_kind: str = "heatmap",  # {"heatmap","slice1d"}
    slice_param: str | None = None,   # e.g. "k"
    slice_value: Any | None = None,   # e.g. 50

    # visuals
    label_every: int = 1,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    base_fontsize: int | float | None = None,
    cmap: str = "bwr",

    # saving
    save_prefix: str = "plot_attr",
    save_dir: str | Path = "./Output",
    formats: Tuple[str, ...] = ("pdf",),
    show: bool = False,

    fig_title: str | None = None,
    strict_other_params: bool = True,
    baseline_xy: tuple[Any, Any] | None = None,
):
    # ---- checks ----
    if not experiments:
        raise ValueError("experiments list is empty")
    if mode not in {"value", "difference"}:
        raise ValueError("mode must be 'value' or 'difference'")
    if diff_kind not in {"difference", "log2"}:
        raise ValueError("diff_kind must be 'difference' or 'log2'")
    if plot_kind not in {"heatmap", "slice1d"}:
        raise ValueError("plot_kind must be 'heatmap' or 'slice1d'")

    if value_label is None:
        value_label = _default_value_label(value_attr, value_index)

    # for slice1d, determine which parameter is the varying axis
    if plot_kind == "slice1d":
        if slice_param is None or slice_value is None:
            raise ValueError("slice1d requires slice_param and slice_value")
        if slice_param not in {x_param, y_param}:
            raise ValueError("slice_param must be either x_param or y_param")

        varying_param = y_param if slice_param == x_param else x_param

    # ---- group by dataset ----
    ds_runs: dict[str, list[Any]] = defaultdict(list)
    for e in experiments:
        ds_runs[e.dataset_name].append(e)

    # ---- title ----
    if fig_title is None:
        exclude = {x_param, y_param, "dataset_name"}
        if mode == "difference":
            exclude.add(compare_param)
        else:
            if select_value is not None:
                exclude.add(compare_param)
        if plot_kind == "slice1d":
            exclude.add(slice_param)

        fixed_global = {}
        for p in (_ALL_PARAMS - exclude):
            vals = {getattr(e, p) for e in experiments}
            if len(vals) == 1:
                fixed_global[p] = vals.pop()
        fig_title = " | ".join(f"{k}:{fmt_val(k, v)}" for k, v in fixed_global.items())

    # ---- layout/fonts (same spirit) ----
    rows, cols = auto_grid(len(ds_runs)) if grid and len(ds_runs) > 1 else (len(ds_runs), 1)
    if figsize is None:
        figsize = (4 * cols, 3.5 * rows)
    bfs = auto_fontsize(figsize, base_fontsize)
    rc = {
        "axes.titlesize": bfs * 1.8,
        "axes.labelsize": bfs * 1.6,
        "xtick.labelsize": bfs * 0.8,
        "ytick.labelsize": bfs * 1.2,
    }

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def safe_vals_repr(vals: set[Any]) -> str:
        return ", ".join(sorted(repr(v) for v in vals))

    def _get_fixed_baseline_value(runs_a: list[Any], ds_name: str) -> float:
        bx, by = baseline_xy  # type: ignore[misc]

        def is_match(r: Any) -> bool:
            # Only enforce axis coords that belong to baseline key
            if x_param in _BASELINE_PARAMS and getattr(r, x_param) != bx:
                return False
            if y_param in _BASELINE_PARAMS and getattr(r, y_param) != by:
                return False
            return True

        matches = [r for r in runs_a if is_match(r)]

        if not matches:
            # Helpful error message showing what was actually enforced
            enforced = []
            if x_param in _BASELINE_PARAMS:
                enforced.append(f"{x_param}={bx}")
            if y_param in _BASELINE_PARAMS:
                enforced.append(f"{y_param}={by}")
            enforced_txt = ", ".join(enforced) if enforced else "(no axis params; baseline treated as constant)"
            raise ValueError(
                f"[{ds_name}] No baseline run found matching {enforced_txt} "
                f"for {compare_param}={compare_values[0]!r}."
            )

        # If multiple matches (likely when sr is ignored), just pick one deterministically
        # because by definition they should be equivalent for baseline-comparison purposes.
        vals = np.array(
            [_get_vec_value(r, value_attr=value_attr, value_index=value_index) for r in matches],
            dtype=float
        )
        if not np.allclose(vals, vals[0], rtol=1e-6, atol=1e-12, equal_nan=True):
            raise ValueError(
                f"[{ds_name}] baseline_xy ignores '{x_param}' (not in _BASELINE_PARAMS), "
                f"but baseline values vary across matches. Add a baseline override or include '{x_param}' in baseline key."
            )
        return float(vals[0])

    with plt.rc_context(rc):
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        for ax in axes[len(ds_runs):]:
            ax.axis("off")

        for ax, (ds_name, runs) in zip(axes, sorted(ds_runs.items())):

            # ---- optional strict check ----
            if strict_other_params:
                exclude = {x_param, y_param, "dataset_name"}
                if mode == "difference":
                    exclude.add(compare_param)
                else:
                    if select_value is not None:
                        exclude.add(compare_param)
                if plot_kind == "slice1d":
                    exclude.add(slice_param)

                for p in (_ALL_PARAMS - exclude):
                    vals = {getattr(r, p) for r in runs}
                    if len(vals) > 1:
                        raise ValueError(
                            f"[{ds_name}] parameter '{p}' varies ({safe_vals_repr(vals)}). "
                            f"Assumes all params except '{x_param}', '{y_param}'"
                            f"{' and compare_param' if mode=='difference' else ''}"
                            f"{' and slice_param' if plot_kind=='slice1d' else ''} are fixed. "
                            "Set strict_other_params=False to disable."
                        )
            if baseline_xy is not None and mode != "difference":
                raise ValueError("baseline_xy is only supported when mode='difference'")

            # ---- filter to a fixed slice, if requested ----
            if plot_kind == "slice1d":
                runs = [r for r in runs if getattr(r, slice_param) == slice_value]

            if not runs:
                ax.set_title(ds_name)
                ax.text(0.5, 0.5, "No runs", ha="center", va="center")
                continue

            # ---- compute data depending on mode ----
            if mode == "difference":
                a_val, b_val = compare_values
                runs_a = [r for r in runs if getattr(r, compare_param, None) == a_val]
                runs_b = [r for r in runs if getattr(r, compare_param, None) == b_val]
                relevant = runs_a + runs_b
            else:
                if select_value is None:
                    relevant = runs
                else:
                    relevant = [r for r in runs if getattr(r, compare_param, None) == select_value]

            if not relevant:
                ax.set_title(ds_name)
                ax.text(0.5, 0.5, "No runs selected", ha="center", va="center")
                continue

            # ---- HEATMAP ----
            if plot_kind == "heatmap":
                grid_runs = runs_b if (mode == "difference" and baseline_xy is not None) else relevant
                xs_sorted = sorted({getattr(r, x_param) for r in grid_runs})
                ys_sorted = sorted({getattr(r, y_param) for r in grid_runs})
                if reverse_x:
                    xs_sorted = xs_sorted[::-1]
                if reverse_y:
                    ys_sorted = ys_sorted[::-1]
                xs_map = {v: i for i, v in enumerate(xs_sorted)}
                ys_map = {v: i for i, v in enumerate(ys_sorted)}

                if mode == "difference":
                    data_a = np.full((len(ys_sorted), len(xs_sorted)), np.nan)
                    data_b = np.full((len(ys_sorted), len(xs_sorted)), np.nan)

                    if baseline_xy is None:
                        # baseline varies with (x,y) as before
                        for r in runs_a:
                            xi = xs_map[getattr(r, x_param)]
                            yi = ys_map[getattr(r, y_param)]
                            data_a[yi, xi] = _get_vec_value(r, value_attr=value_attr, value_index=value_index)
                    else:
                        # NEW: baseline fixed at one (x,y) point and broadcast everywhere
                        base_val = _get_fixed_baseline_value(runs_a, ds_name)
                        data_a[:, :] = base_val

                    # bagged always varies with (x,y)
                    for r in runs_b:
                        xi = xs_map[getattr(r, x_param)]
                        yi = ys_map[getattr(r, y_param)]
                        data_b[yi, xi] = _get_vec_value(r, value_attr=value_attr, value_index=value_index)

                    if diff_kind == "difference":
                        data = data_a - data_b
                        if baseline_xy is None:
                            cbar_label = f"{value_label}\n{compare_param}={a_val} – {compare_param}={b_val}"
                        else:
                            bx, by = baseline_xy
                            cbar_label = (
                                f"{value_label}\n({compare_param}={a_val} @ {x_param}={bx}, {y_param}={by}) – "
                                f"{compare_param}={b_val}"
                            )
                    else:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            data = np.log2(data_a) - np.log2(data_b)
                        if baseline_xy is None:
                            cbar_label = f"{value_label}\nlog₂({compare_param}={a_val}) – log₂({compare_param}={b_val})"
                        else:
                            bx, by = baseline_xy
                            cbar_label = (
                                f"{value_label}\nlog₂({compare_param}={a_val} @ {x_param}={bx}, {y_param}={by}) – "
                                f"log₂({compare_param}={b_val})"
                            )

                    vmax = np.nanmax(np.abs(data)) or 1.0
                    im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, origin="lower")

                else:
                    data = np.full((len(ys_sorted), len(xs_sorted)), np.nan)
                    for r in relevant:
                        xi = xs_map[getattr(r, x_param)]
                        yi = ys_map[getattr(r, y_param)]
                        data[yi, xi] = _get_vec_value(r, value_attr=value_attr, value_index=value_index)

                    vmax = np.nanmax(np.abs(data)) or 1.0
                    im = ax.imshow(data, cmap="Reds", vmin=0, vmax=vmax, origin="lower")
                    cbar_label = value_label if select_value is None else f"{value_label}\n{compare_param}={select_value}"

                # ticks like before
                ax.set_xticks(range(len(xs_sorted)))
                ax.set_xticklabels(
                    [
                        fmt_val(x_param, v) if (i % label_every == 0 or i in {0, len(xs_sorted)-1}) else ""
                        for i, v in enumerate(xs_sorted)
                    ],
                    rotation=45, ha="right"
                )
                ax.set_yticks(range(len(ys_sorted)))
                ax.set_yticklabels(
                    [
                        fmt_val(y_param, v) if (i % label_every == 0 or i in {0, len(ys_sorted)-1}) else ""
                        for i, v in enumerate(ys_sorted)
                    ]
                )

                ax.set_xlabel(x_param)
                ax.set_ylabel(y_param)
                ax.set_title(ds_name)

                cbar = fig.colorbar(im, ax=ax, shrink=0.8)
                if baseline_xy is not None:
                    bx, by = baseline_xy
                    parts = []
                    if x_param in _BASELINE_PARAMS:
                        parts.append(f"{x_param}={bx}")
                    if y_param in _BASELINE_PARAMS:
                        parts.append(f"{y_param}={by}")
                    coord_txt = ", ".join(parts) if parts else "constant baseline"
                    cbar_label = f"{value_label}\n({compare_param}={a_val} @ {coord_txt}) – {compare_param}={b_val}"
                cbar.ax.set_ylabel(cbar_label)

            # ---- SLICE1D ----
            else:
                if mode == "difference" and baseline_xy is not None:
                    axis_runs = runs_b  # x-axis comes from bagged when baseline is fixed
                else:
                    axis_runs = relevant

                zs = sorted({getattr(r, varying_param) for r in axis_runs})
                if (varying_param == x_param and reverse_x) or (varying_param == y_param and reverse_y):
                    zs = zs[::-1]
                z_map = {v: i for i, v in enumerate(zs)}

                if mode == "difference":
                    a_val, b_val = compare_values
                    y_a = np.full(len(zs), np.nan)
                    y_b = np.full(len(zs), np.nan)

                    if baseline_xy is None:
                        for r in runs_a:
                            zi = z_map.get(getattr(r, varying_param))
                            if zi is not None:
                                y_a[zi] = _get_vec_value(r, value_attr=value_attr, value_index=value_index)
                    else:
                        base_val = _get_fixed_baseline_value(runs_a, ds_name)
                        y_a[:] = base_val

                    for r in runs_b:
                        zi = z_map.get(getattr(r, varying_param))
                        if zi is not None:
                            y_b[zi] = _get_vec_value(r, value_attr=value_attr, value_index=value_index)

                    if diff_kind == "difference":
                        y = y_a - y_b
                        ylab = f"{value_label} ({compare_param}={a_val} – {compare_param}={b_val})"
                    else:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            y = np.log2(y_a) - np.log2(y_b)
                        ylab = f"{value_label} (log₂ {compare_param}={a_val} – log₂ {compare_param}={b_val})"
                else:
                    y = np.full(len(zs), np.nan)
                    for r in relevant:
                        zi = z_map.get(getattr(r, varying_param))
                        if zi is not None:
                            y[zi] = _get_vec_value(r, value_attr=value_attr, value_index=value_index)
                    ylab = value_label if select_value is None else f"{value_label} ({compare_param}={select_value})"

                x = np.arange(len(zs))
                ax.plot(x, y)  # default matplotlib styling
                ax.set_title(ds_name)
                ax.set_xlabel(varying_param)
                ax.set_ylabel(ylab)

                ax.set_xticks(x)
                ax.set_xticklabels(
                    [
                        fmt_val(varying_param, v) if (i % label_every == 0 or i in {0, len(zs)-1}) else ""
                        for i, v in enumerate(zs)
                    ],
                    rotation=45, ha="right"
                )
                ax.grid(True, alpha=0.3)

                # annotate the slice choice
                ax.text(
                    0.01, 0.98,
                    f"{slice_param}={fmt_val(slice_param, slice_value)}",
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=bfs * 1.0
                )

        if fig_title:
            fig.suptitle(fig_title, y=1.02, fontsize=bfs * 3)
        fig.tight_layout()

        safe_idx = str(value_index).replace("-", "m")
        slice_tag = ""
        if plot_kind == "slice1d":
            slice_tag = f"_slice_{slice_param}_{fmt_val(slice_param, slice_value)}"

        for fmt in formats:
            out = save_dir / f"{save_prefix}_{value_attr}_{safe_idx}_{plot_kind}_{mode}{('_'+diff_kind) if mode=='difference' else ''}{slice_tag}.{fmt}"
            fig.savefig(out, bbox_inches="tight")
            print(f"[SAVED] {out}")

        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_experiment_heatmaps_3d(
    experiments: Sequence[Any],
    *,
    x_param: str,
    y_param: str,
    z_param: str,  # baseline-only axis (e.g. k_baseline)
    baseline_overrides: Mapping[str, Any] | None = None,  # fixed baseline params (besides z_param)
    reverse_x: bool = False,
    reverse_y: bool = False,
    reverse_z: bool = False,
    metrics: Tuple[str, ...] = ("mse", "bias2", "var"),
    label_every: int = 1,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    base_fontsize: int | float | None = None,
    cmap: str = "bwr",
    save_prefix: str = "heat3d",
    save_dir: str | Path = "./Output",
    formats: Tuple[str, ...] = ("pdf",),
    show: bool = False,
    log: bool = False,
    type: str = "difference",
    inlog: bool = False,
    fig_title: str | bool | None = False,
    alpha: float = 0.9,
    edgecolor: str | None = None,
    linewidth: float = 0.0,
    view_elev: float = 22,
    view_azim: float = -55,
):
    """
    3D "heatmap" using voxels:
      - x,y are BAGGED params (vary across bagged runs)
      - z is BASELINE param (varies only for baseline selection)
      - voxel color encodes diff/value like the 2D function.

    IMPORTANT:
      For the baseline to *only* change along z, any other baseline-key params that vary with bagged
      must be fixed via baseline_overrides (or not be in _BASELINE_PARAMS).
    """

    # --- sanity checks ----------------------------------------------------
    if x_param == y_param:
        raise ValueError("x_param and y_param must differ")
    for p in (x_param, y_param, z_param):
        if p not in _NUMERIC_PARAMS:
            raise ValueError(f"{p} must be numeric param in {sorted(_NUMERIC_PARAMS)}")
    if not experiments:
        raise ValueError("experiments list is empty")

    _base_fixed: dict[str, Any] = dict(baseline_overrides or {})
    if _base_fixed:
        unknown = set(_base_fixed) - set(_BASELINE_PARAMS)
        if unknown:
            raise ValueError(
                f"baseline_overrides contains params not in _BASELINE_PARAMS: {sorted(unknown)}"
            )

    # z_param must be in baseline lookup key, otherwise baseline can't vary along z
    if z_param not in _BASELINE_PARAMS:
        raise ValueError(
            f"z_param='{z_param}' must be in _BASELINE_PARAMS to vary baseline along z. "
            f"Current _BASELINE_PARAMS={sorted(_BASELINE_PARAMS)}"
        )

    # Warn if baseline might unintentionally vary with x/y (if x/y are in baseline key and not fixed)
    if x_param in _BASELINE_PARAMS and x_param not in _base_fixed:
        print(f"[WARN] x_param='{x_param}' is in _BASELINE_PARAMS but not fixed in baseline_overrides; "
              f"baseline may vary with x unless you set baseline_overrides[{x_param!r}] = <fixed value>.")
    if (y_param in _BASELINE_PARAMS) and (y_param != z_param) and (y_param not in _base_fixed):
        print(f"[WARN] y_param='{y_param}' is in _BASELINE_PARAMS but not fixed in baseline_overrides; "
              f"baseline may vary with y unless you set baseline_overrides[{y_param!r}] = <fixed value>.")

    # --- group by dataset -------------------------------------------------
    ds_runs: dict[str, list[Any]] = defaultdict(list)
    for e in experiments:
        ds_runs[e.dataset_name].append(e)

    # --- title helpers (kept close to your style) -------------------------
    def param_name(param_str: str) -> str:
        if param_str == "sr":
            return "sampling rate"
        if param_str == "Nbag":
            return "number of bags"
        if param_str == "k":
            return "k"
        return param_str

    if fig_title is None:
        fixed_global = {}
        for p in _ALL_PARAMS - {x_param, y_param, z_param, "bagging_method", "dataset_name"}:
            vals = {getattr(e, p) for e in experiments}
            if len(vals) == 1:
                fixed_global[p] = vals.pop()
        fig_title = " | ".join(f"{k}:{fmt_val(k,v)}" for k, v in fixed_global.items())
    elif fig_title == "auto":
        fig_title = (
            f"3D interaction: bagged({param_name(x_param)} × {param_name(y_param)}) vs baseline({param_name(z_param)})\n"
            f"Baseline Estimator: {experiments[0].estimator_name.upper()}"
        )

    # --- layout / fonts ---------------------------------------------------
    rows, cols = auto_grid(len(ds_runs)) if grid and len(ds_runs) > 1 else (len(ds_runs), 1)
    if figsize is None:
        figsize = (5 * cols, 4.2 * rows)

    bfs = auto_fontsize(figsize, base_fontsize)
    rc = {
        "axes.titlesize": bfs * 1.5,
        "axes.labelsize": bfs * 1.3,
        "xtick.labelsize": bfs * 0.7,
        "ytick.labelsize": bfs * 0.9,
    }

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    key_to_label = {"mse": "MSE", "bias2": "Bias²", "var": "Variance"}

    # --- per-metric figure ------------------------------------------------
    for met_key in metrics:
        if met_key not in key_to_label:
            raise ValueError(f"Unknown metric '{met_key}'")
        met_label = key_to_label[met_key]

        with plt.rc_context(rc):
            fig = plt.figure(figsize=figsize)

            # Create 3D subplots in the same grid arrangement
            axes = []
            for i in range(rows * cols):
                ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
                axes.append(ax)

            for ax in axes[len(ds_runs):]:
                ax.set_axis_off()

            # We'll share a single color scale across all datasets in this metric
            # (much cleaner for 3D than per-axes colorbars)
            all_vals_for_scale = []

            # First pass: compute per-dataset voxel colors and store them
            per_ds_payload = {}

            for (ds_name, runs), ax in zip(sorted(ds_runs.items()), axes):
                # separate baseline and bagged
                baseline_lookup: dict[tuple, Any] = {}
                bagged_list: list[Any] = []

                for r in runs:
                    base_key = tuple(getattr(r, p) for p in _BASELINE_PARAMS)
                    if r.bagging_method is None:
                        baseline_lookup[base_key] = r
                    else:
                        bagged_list.append(r)

                # axis values
                xs_sorted = sorted({getattr(r, x_param) for r in bagged_list})
                ys_sorted = sorted({getattr(r, y_param) for r in bagged_list})
                zs_sorted = sorted({getattr(r, z_param) for r in baseline_lookup.values()})

                if reverse_x:
                    xs_sorted = xs_sorted[::-1]
                if reverse_y:
                    ys_sorted = ys_sorted[::-1]
                if reverse_z:
                    zs_sorted = zs_sorted[::-1]

                xs_map = {v: i for i, v in enumerate(xs_sorted)}
                ys_map = {v: i for i, v in enumerate(ys_sorted)}
                zs_map = {v: i for i, v in enumerate(zs_sorted)}

                # data[x, y, z]
                data = np.full((len(xs_sorted), len(ys_sorted), len(zs_sorted)), np.nan, dtype=float)

                # Fill all voxels: for each bagged run (x,y), compare against baseline at each z
                for r in bagged_list:
                    xi = xs_map[getattr(r, x_param)]
                    yi = ys_map[getattr(r, y_param)]

                    for z_val in zs_sorted:
                        zi = zs_map[z_val]

                        _base_override = dict(_base_fixed)
                        _base_override[z_param] = z_val

                        base_key = tuple(
                            (_base_override[p] if p in _base_override else getattr(r, p))
                            for p in _BASELINE_PARAMS
                        )
                        base_run = baseline_lookup.get(base_key)
                        if base_run is None:
                            continue

                        # same diff logic as your 2D function
                        if inlog:
                            if log:
                                if type == "difference":
                                    diff = np.log2(getattr(base_run, f"log_total_{met_key}")) - np.log2(
                                        getattr(r, f"log_total_{met_key}")
                                    )
                                elif type == "baseline":
                                    diff = -np.log2(getattr(base_run, f"log_total_{met_key}"))
                                elif type == "bagged":
                                    diff = -np.log2(getattr(r, f"log_total_{met_key}"))
                                else:
                                    raise ValueError(f"Unknown type='{type}'")
                            else:
                                if type == "difference":
                                    diff = getattr(base_run, f"log_total_{met_key}") - getattr(r, f"log_total_{met_key}")
                                elif type == "baseline":
                                    diff = -getattr(base_run, f"log_total_{met_key}")
                                elif type == "bagged":
                                    diff = -getattr(r, f"log_total_{met_key}")
                                else:
                                    raise ValueError(f"Unknown type='{type}'")
                        else:
                            if log:
                                if type == "difference":
                                    diff = np.log2(getattr(base_run, f"total_{met_key}")) - np.log2(
                                        getattr(r, f"total_{met_key}")
                                    )
                                elif type == "baseline":
                                    diff = -np.log2(getattr(base_run, f"total_{met_key}"))
                                elif type == "bagged":
                                    diff = -np.log2(getattr(r, f"total_{met_key}"))
                                else:
                                    raise ValueError(f"Unknown type='{type}'")
                            else:
                                if type == "difference":
                                    diff = getattr(base_run, f"total_{met_key}") - getattr(r, f"total_{met_key}")
                                elif type == "baseline":
                                    diff = getattr(base_run, f"total_{met_key}")
                                elif type == "bagged":
                                    diff = getattr(r, f"total_{met_key}")
                                else:
                                    raise ValueError(f"Unknown type='{type}'")

                        data[xi, yi, zi] = diff

                # collect for global scaling
                if np.any(np.isfinite(data)):
                    all_vals_for_scale.append(data[np.isfinite(data)])

                per_ds_payload[ds_name] = {
                    "ax": ax,
                    "data": data,
                    "xs": xs_sorted,
                    "ys": ys_sorted,
                    "zs": zs_sorted,
                }

            # Determine global color normalization
            if all_vals_for_scale:
                flat = np.concatenate(all_vals_for_scale)
                vmax = float(np.nanmax(np.abs(flat))) if (type == "difference" or log) else float(np.nanmax(flat))
                vmax = vmax or 1.0
            else:
                vmax = 1.0

            if type == "difference" or log:
                norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
                cmap_obj = plt.get_cmap(cmap)
            else:
                norm = mcolors.Normalize(vmin=0, vmax=vmax)
                cmap_obj = plt.get_cmap("Reds")

            # Second pass: draw voxels
            for ds_name, payload in per_ds_payload.items():
                ax = payload["ax"]
                data = payload["data"]
                xs_sorted = payload["xs"]
                ys_sorted = payload["ys"]
                zs_sorted = payload["zs"]

                filled = np.isfinite(data)

                # facecolors must match filled shape with RGBA
                rgba = cmap_obj(norm(np.nan_to_num(data, nan=0.0)))
                rgba[..., 3] = np.where(filled, alpha, 0.0)

                ax.voxels(
                    filled,
                    facecolors=rgba,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                )

                # Ticks at cube centers
                ax.set_xticks(np.arange(len(xs_sorted)) + 0.5)
                ax.set_yticks(np.arange(len(ys_sorted)) + 0.5)
                ax.set_zticks(np.arange(len(zs_sorted)) + 0.5)

                ax.set_xticklabels([
                    fmt_val(x_param, v) if (i % label_every == 0 or i in {0, len(xs_sorted)-1}) else ""
                    for i, v in enumerate(xs_sorted)
                ], rotation=45, ha="right")

                ax.set_yticklabels([
                    fmt_val(y_param, v) if (i % label_every == 0 or i in {0, len(ys_sorted)-1}) else ""
                    for i, v in enumerate(ys_sorted)
                ])

                ax.set_zticklabels([
                    fmt_val(z_param, v) if (i % label_every == 0 or i in {0, len(zs_sorted)-1}) else ""
                    for i, v in enumerate(zs_sorted)
                ])

                ax.set_xlabel("Number of Bags (B)" if x_param == "Nbag" else x_param)
                ax.set_ylabel("Number of Bags (B)" if y_param == "Nbag" else y_param)
                ax.set_zlabel(f"{z_param} (baseline)")

                ax.set_title(ds_name)
                ax.view_init(elev=view_elev, azim=view_azim)

            # One shared colorbar for the whole figure
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
            sm.set_array([])

            cbar = fig.colorbar(sm, ax=axes[:len(ds_runs)], shrink=0.85, pad=0.08)

            # match your colorbar labeling scheme
            if inlog:
                if log:
                    if type == "difference":
                        cbar.ax.set_ylabel(f"{met_label}\nlog₂(log_baseline) – log₂(log_bagged)")
                    elif type == "baseline":
                        cbar.ax.set_ylabel(f"{met_label}\n-log₂(log_baseline)")
                    elif type == "bagged":
                        cbar.ax.set_ylabel(f"{met_label}\n-log₂(log_bagged)")
                else:
                    if type == "difference":
                        cbar.ax.set_ylabel(f"{met_label}\nlog_baseline – log_bagged")
                    elif type == "baseline":
                        cbar.ax.set_ylabel(f"{met_label}\n-log_baseline")
                    elif type == "bagged":
                        cbar.ax.set_ylabel(f"{met_label}\n-log_bagged")
            else:
                if log:
                    if type == "difference":
                        cbar.ax.set_ylabel(f"{met_label}\nlog₂(baseline) – log₂(bagged)")
                    elif type == "baseline":
                        cbar.ax.set_ylabel(f"{met_label}\n-log₂(baseline)")
                    elif type == "bagged":
                        cbar.ax.set_ylabel(f"{met_label}\n-log₂(bagged)")
                else:
                    if type == "difference":
                        cbar.ax.set_ylabel(f"{met_label}\nbaseline – bagged")
                    elif type == "baseline":
                        cbar.ax.set_ylabel(f"{met_label}\nbaseline")
                    elif type == "bagged":
                        cbar.ax.set_ylabel(f"{met_label}\nbagged")

            if fig_title:
                fig.suptitle(str(fig_title), y=1.02, fontsize=bfs * 2.4)

            fig.tight_layout()

            logsavename = "_log" if log else ""
            inlogsavename = "_inlog" if inlog else ""

            for fmt in formats:
                out = save_dir / f"{save_prefix}_{met_key}_{type}_3d_{x_param}_{y_param}_z{z_param}{logsavename}{inlogsavename}.{fmt}"
                fig.savefig(out, bbox_inches="tight")
                print(f"[SAVED] {out}")

            if show:
                plt.show()
            else:
                plt.close(fig)

def plot_experiment_heatmaps_slices(
    experiments: Sequence[Any],
    *,
    x_param: str,
    y_param: str,
    z_param: str,  # baseline-only axis (e.g. "k" when x="sr", y="k" for bagged)
    # baseline overrides (for params in _BASELINE_PARAMS) that are held fixed for ALL z-slices
    baseline_overrides: Mapping[str, Any] | None = None,
    # optionally restrict which baseline z values to slice over (else inferred from baseline runs)
    z_values: Sequence[Any] | None = None,
    reverse_x: bool = False,
    reverse_y: bool = False,
    reverse_z: bool = False,
    metrics: Tuple[str, ...] = ("mse", "bias2", "var"),
    label_every: int = 1,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    base_fontsize: int | float | None = None,
    cmap: str = "bwr",
    save_prefix: str = "heat3d",
    save_dir: str | Path = "./Output",
    formats: Tuple[str, ...] = ("pdf",),
    show: bool = False,
    log: bool = False,
    type: str = "difference",
    inlog: bool = False,
    fig_title: str | bool | None = False,
    share_color_scale_across_z: bool = True,
    baseline_curve: callable | None = None,  # f(x_numeric) -> y_numeric (e.g. k_baseline(x))
    curve_num: int = 300,  # smoothness of the curve
    curve_lw: float = 1.2,  # line width
    curve_ms: float = 3.0,  # marker size for dots
    curve_only_on_matching_slice: bool = True,
    overlay_k_over_x: bool = True,   # overlay y = z_val / x on every slice
):
    """
    Draw baseline-vs-bagged metric differences as "3D heatmaps" by slicing over a baseline-only z_param.

    - x_param, y_param vary over BAGGED runs (bagging_method != None)
    - z_param varies over BASELINE runs (bagging_method is None)
    - Each z-slice is a standard 2D heatmap (y,x), with baseline fixed at that z value.
    """

    def _interp_index(vals_display: list[float], v: float) -> float | None:
        """
        Map numeric v into heatmap index coordinates [0, n-1] using piecewise-linear interpolation.
        Works whether vals_display is increasing or decreasing (display order).
        Returns None if outside range.
        """
        n = len(vals_display)
        if n == 0:
            return None
        if n == 1:
            return 0.0 if np.isclose(v, vals_display[0]) else None

        inc = vals_display[0] < vals_display[-1]
        vals = vals_display if inc else vals_display[::-1]

        if v < vals[0] or v > vals[-1]:
            return None

        j = int(np.searchsorted(vals, v, side="right")) - 1
        j = max(0, min(j, n - 2))
        v0, v1 = vals[j], vals[j + 1]
        if np.isclose(v1, v0):
            idx_inc = float(j)
        else:
            t = (v - v0) / (v1 - v0)
            idx_inc = float(j + t)

        return idx_inc if inc else (n - 1 - idx_inc)

    # --- sanity checks ----------------------------------------------------
    if x_param == y_param:
        raise ValueError("x_param and y_param must differ")
    for p in (x_param, y_param, z_param):
        if p not in _NUMERIC_PARAMS:
            raise ValueError(f"{p} must be numeric param in {sorted(_NUMERIC_PARAMS)}")
    if not experiments:
        raise ValueError("experiments list is empty")

    # Baseline overrides must be relevant to baseline matching
    _base_fixed: dict[str, Any] = dict(baseline_overrides or {})
    if _base_fixed:
        unknown = set(_base_fixed) - set(_BASELINE_PARAMS)
        if unknown:
            raise ValueError(
                f"baseline_overrides contains params not in _BASELINE_PARAMS: {sorted(unknown)}"
            )

    # z_param must be matchable via _BASELINE_PARAMS, otherwise baseline can't "vary along z" in lookup
    if z_param not in _BASELINE_PARAMS:
        raise ValueError(
            f"z_param='{z_param}' must be in _BASELINE_PARAMS to vary baseline along z. "
            f"Current _BASELINE_PARAMS={sorted(_BASELINE_PARAMS)}"
        )

    # --- group by dataset -------------------------------------------------
    ds_runs: dict[str, list[Any]] = defaultdict(list)
    for e in experiments:
        ds_runs[e.dataset_name].append(e)

    # title (reuse your style lightly; keep it simple)
    def param_name(param_str: str) -> str:
        if param_str == "sr":
            return "sampling rate"
        if param_str == "Nbag":
            return "number of bags"
        if param_str == "k":
            return "k"
        return param_str

    if fig_title is None:
        fixed_global = {}
        for p in _ALL_PARAMS - {x_param, y_param, z_param, "bagging_method", "dataset_name"}:
            vals = {getattr(e, p) for e in experiments}
            if len(vals) == 1:
                fixed_global[p] = vals.pop()
        fig_title = " | ".join(f"{k}:{fmt_val(k,v)}" for k, v in fixed_global.items())
    elif fig_title == "auto":
        fig_title = (
            f"Bagged interaction: {param_name(x_param)} × {param_name(y_param)}\n"
            f"Baseline slices over {param_name(z_param)}"
        )

    # layout helpers (same feel as your 2D version)
    rows, cols = auto_grid(len(ds_runs)) if grid and len(ds_runs) > 1 else (len(ds_runs), 1)
    if figsize is None:
        figsize = (4 * cols, 3.5 * rows)

    bfs = auto_fontsize(figsize, base_fontsize)
    rc = {
        "axes.titlesize": bfs * 1.4,
        "axes.labelsize": bfs * 1.3,
        "xtick.labelsize": bfs * 0.7,
        "ytick.labelsize": bfs * 1.0,
    }

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    key_to_label = {"mse": "MSE", "bias2": "Bias²", "var": "Variance"}

    # --- main loop: metric -> z-slice figure per dataset-grid -------------
    for met_key in metrics:
        if met_key not in key_to_label:
            raise ValueError(f"Unknown metric '{met_key}'")
        met_label = key_to_label[met_key]

        # Determine z values (prefer explicit z_values; else infer from baseline runs globally)
        if z_values is None:
            z_set = {getattr(e, z_param) for e in experiments if e.bagging_method is None}
            zs_sorted = sorted(z_set)
        else:
            zs_sorted = list(z_values)

        if reverse_z:
            zs_sorted = zs_sorted[::-1]

        # We will create ONE figure PER z slice (still multi-dataset grid),
        # which is usually the cleanest way to interpret a third axis.
        # If you want a single figure with all z-slices tiled, tell me and I’ll provide that too.
        for z_val in zs_sorted:
            with plt.rc_context(rc):
                fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
                axes = axes.flatten()
                for ax in axes[len(ds_runs):]:
                    ax.axis("off")

                # First pass: compute all dataset data arrays, optionally track global vmax for shared scale
                per_ds_data: dict[str, np.ndarray] = {}
                per_ds_xy: dict[str, tuple[list[Any], list[Any]]] = {}

                global_vmax = 0.0

                for (ds_name, runs) in sorted(ds_runs.items()):
                    # separate baseline and bagged
                    baseline_lookup: dict[tuple, Any] = {}
                    bagged_list: list[Any] = []

                    for r in runs:
                        base_key = tuple(getattr(r, p) for p in _BASELINE_PARAMS)
                        if r.bagging_method is None:
                            baseline_lookup[base_key] = r
                        else:
                            bagged_list.append(r)

                    # x/y grid from BAGGED
                    xs_sorted = sorted({getattr(r, x_param) for r in bagged_list})
                    ys_sorted = sorted({getattr(r, y_param) for r in bagged_list})
                    if reverse_x:
                        xs_sorted = xs_sorted[::-1]
                    if reverse_y:
                        ys_sorted = ys_sorted[::-1]

                    xs_map = {v: i for i, v in enumerate(xs_sorted)}
                    ys_map = {v: i for i, v in enumerate(ys_sorted)}
                    data = np.full((len(ys_sorted), len(xs_sorted)), np.nan)

                    # Build baseline key override for THIS z-slice:
                    # - force z_param to z_val
                    # - apply fixed baseline_overrides (if any)
                    _base_override = dict(_base_fixed)
                    _base_override[z_param] = z_val

                    # Fill grid by comparing each bagged run to the baseline at z=z_val (and otherwise matched)
                    for r in bagged_list:
                        base_key = tuple(
                            (_base_override[p] if p in _base_override else getattr(r, p))
                            for p in _BASELINE_PARAMS
                        )
                        base_run = baseline_lookup.get(base_key)
                        if base_run is None:
                            continue

                        # compute diff exactly like your 2D code
                        if inlog:
                            if log:
                                if type == "difference":
                                    diff = np.log2(getattr(base_run, f"log_total_{met_key}")) - np.log2(
                                        getattr(r, f"log_total_{met_key}")
                                    )
                                elif type == "baseline":
                                    diff = -np.log2(getattr(base_run, f"log_total_{met_key}"))
                                elif type == "bagged":
                                    diff = -np.log2(getattr(r, f"log_total_{met_key}"))
                                else:
                                    raise ValueError(f"Unknown type='{type}'")
                            else:
                                if type == "difference":
                                    diff = getattr(base_run, f"log_total_{met_key}") - getattr(r, f"log_total_{met_key}")
                                elif type == "baseline":
                                    diff = -getattr(base_run, f"log_total_{met_key}")
                                elif type == "bagged":
                                    diff = -getattr(r, f"log_total_{met_key}")
                                else:
                                    raise ValueError(f"Unknown type='{type}'")
                        else:
                            if log:
                                if type == "difference":
                                    diff = np.log2(getattr(base_run, f"total_{met_key}")) - np.log2(getattr(r, f"total_{met_key}"))
                                elif type == "baseline":
                                    diff = -np.log2(getattr(base_run, f"total_{met_key}"))
                                elif type == "bagged":
                                    diff = -np.log2(getattr(r, f"total_{met_key}"))
                                else:
                                    raise ValueError(f"Unknown type='{type}'")
                            else:
                                if type == "difference":
                                    diff = getattr(base_run, f"total_{met_key}") - getattr(r, f"total_{met_key}")
                                elif type == "baseline":
                                    diff = getattr(base_run, f"total_{met_key}")
                                elif type == "bagged":
                                    diff = getattr(r, f"total_{met_key}")
                                else:
                                    raise ValueError(f"Unknown type='{type}'")

                        xi = xs_map[getattr(r, x_param)]
                        yi = ys_map[getattr(r, y_param)]
                        data[yi, xi] = diff

                    per_ds_data[ds_name] = data
                    per_ds_xy[ds_name] = (xs_sorted, ys_sorted)

                    if share_color_scale_across_z:
                        vmax_here = float(np.nanmax(np.abs(data))) if np.isfinite(np.nanmax(np.abs(data))) else 0.0
                        global_vmax = max(global_vmax, vmax_here)

                # Second pass: draw
                for ax, (ds_name, _) in zip(axes, sorted(ds_runs.items())):
                    data = per_ds_data[ds_name]
                    xs_sorted, ys_sorted = per_ds_xy[ds_name]

                    # color scaling
                    if share_color_scale_across_z:
                        vmax = global_vmax or 1.0
                    else:
                        vmax = np.nanmax(np.abs(data)) or 1.0

                    if type == "difference" or log:
                        im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, origin="lower")
                    else:
                        im = ax.imshow(data, cmap="Reds", vmin=0, vmax=vmax, origin="lower")

                    # ticks and labels
                    ax.set_xticks(range(len(xs_sorted)))
                    ax.set_xticklabels(
                        [
                            fmt_val(x_param, v) if (i % label_every == 0 or i in {0, len(xs_sorted) - 1}) else ""
                            for i, v in enumerate(xs_sorted)
                        ],
                        rotation=45,
                        ha="right",
                    )
                    ax.set_yticks(range(len(ys_sorted)))
                    ax.set_yticklabels(
                        [
                            fmt_val(y_param, v) if (i % label_every == 0 or i in {0, len(ys_sorted) - 1}) else ""
                            for i, v in enumerate(ys_sorted)
                        ]
                    )

                    ax.set_xlabel("Number of Bags (B)" if x_param == "Nbag" else x_param)
                    ax.set_ylabel("Number of Bags (B)" if y_param == "Nbag" else y_param)
                    ax.set_title(ds_name)

                    # ---- overlay: y = z_val / x (e.g. k_baseline / sr) ----
                    if overlay_k_over_x:
                        xs_num = [float(v) for v in xs_sorted]
                        ys_num = [float(v) for v in ys_sorted]

                        x_min, x_max = min(xs_num), max(xs_num)
                        x_dense = np.linspace(x_min, x_max, curve_num)

                        with np.errstate(divide="ignore", invalid="ignore"):
                            y_dense = float(z_val) * x_dense

                        # Map to heatmap index coords (use NaNs to break line out-of-range)
                        x_idx, y_idx = [], []
                        for xx, yy in zip(x_dense, y_dense):
                            if not np.isfinite(yy):
                                x_idx.append(np.nan);
                                y_idx.append(np.nan);
                                continue
                            xi = _interp_index(xs_num, float(xx))
                            yi = _interp_index(ys_num, float(yy))
                            if xi is None or yi is None:
                                x_idx.append(np.nan);
                                y_idx.append(np.nan)
                            else:
                                x_idx.append(xi);
                                y_idx.append(yi)

                        ax.plot(x_idx, y_idx, color="black", lw=curve_lw, zorder=5)

                        # Dots at actual x grid values
                        x_dot, y_dot = [], []
                        for xx in xs_num:
                            with np.errstate(divide="ignore", invalid="ignore"):
                                yy = float(z_val) * float(xx)
                            if not np.isfinite(yy):
                                continue
                            xi = _interp_index(xs_num, float(xx))
                            yi = _interp_index(ys_num, float(yy))
                            if xi is None or yi is None:
                                continue
                            x_dot.append(xi);
                            y_dot.append(yi)

                        ax.plot(x_dot, y_dot, "o", color="black", ms=curve_ms, zorder=6)

                    cbar = fig.colorbar(im, ax=ax, shrink=0.8)

                    # colorbar label (reuse your text, add z info)
                    z_txt = f"{z_param}={fmt_val(z_param, z_val)}"
                    if inlog:
                        if log:
                            if type == "difference":
                                cbar.ax.set_ylabel(f"{met_label}\nlog₂(log_baseline) – log₂(log_bagged)\n[{z_txt}]")
                            elif type == "baseline":
                                cbar.ax.set_ylabel(f"{met_label}\n-log₂(log_baseline)\n[{z_txt}]")
                            elif type == "bagged":
                                cbar.ax.set_ylabel(f"{met_label}\n-log₂(log_bagged)\n[{z_txt}]")
                        else:
                            if type == "difference":
                                cbar.ax.set_ylabel(f"{met_label}\nlog_baseline – log_bagged\n[{z_txt}]")
                            elif type == "baseline":
                                cbar.ax.set_ylabel(f"{met_label}\n-log_baseline\n[{z_txt}]")
                            elif type == "bagged":
                                cbar.ax.set_ylabel(f"{met_label}\n-log_bagged\n[{z_txt}]")
                    else:
                        if log:
                            if type == "difference":
                                cbar.ax.set_ylabel(f"{met_label}\nlog₂(baseline) – log₂(bagged)\n[{z_txt}]")
                            elif type == "baseline":
                                cbar.ax.set_ylabel(f"{met_label}\n-log₂(baseline)\n[{z_txt}]")
                            elif type == "bagged":
                                cbar.ax.set_ylabel(f"{met_label}\n-log₂(bagged)\n[{z_txt}]")
                        else:
                            if type == "difference":
                                cbar.ax.set_ylabel(f"{met_label}\nbaseline – bagged\n[{z_txt}]")
                            elif type == "baseline":
                                cbar.ax.set_ylabel(f"{met_label}\nbaseline\n[{z_txt}]")
                            elif type == "bagged":
                                cbar.ax.set_ylabel(f"{met_label}\nbagged\n[{z_txt}]")

                # super title
                if fig_title:
                    fig.suptitle(str(fig_title), y=1.02, fontsize=bfs * 2.4)

                fig.tight_layout()

                logsavename = "_log" if log else ""
                inlogsavename = "_inlog" if inlog else ""

                # include z in filename
                z_tag = f"_{z_param}-{fmt_val(z_param, z_val)}"
                z_tag = z_tag.replace("/", "_").replace(" ", "")  # basic filename hygiene

                for fmt in formats:
                    out = save_dir / f"{save_prefix}_{met_key}_{type}{z_tag}{logsavename}{inlogsavename}.{fmt}"
                    fig.savefig(out, bbox_inches="tight")
                    print(f"[SAVED] {out}")

                if show:
                    plt.show()
                else:
                    plt.close(fig)