from __future__ import annotations
import pathlib
from typing import Mapping, Literal
from pathlib import Path
import plotly.graph_objects as go
from multiprocess.dummy import Value
from soupsieve.css_parser import VALUE

###################################################OWN IMPORT###################################################
from Bagging_for_LID.Plotting.plotting_helpers import *
from Bagging_for_LID.Plotting.optimize_across_parameter_results import *
from Bagging_for_LID.Plotting.colormap_helpers import *
from Bagging_for_LID.Helper.Other import *

#As opposed to the other plotting functions, plot_radar_from_results and plot_tables_from_results does not take all the experiments as input.
#Instead, only the neceassry data extracted from optimal ones are given to it, which have been determined as such by the helper functions from optimize_across_parameter_results.
#Therefore, it doesn't perform any of the optimization steps to select the experiments (across parameter combinations) with lowest MSE/Var/Bias^2.
#This preprocessing is called by a wrapper underneath to simplify the function's use.

def plot_tables_from_results(
    results: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    *,
    mode: str = "combined",
    normalize_data: bool = False,
    log: bool = False,
    best_mark: bool = True,
    best_font_color: str = "green",
    best_by: Literal["min","max","auto"] = "auto",
    heatmap_cells: bool = False,
    heatmap_colorscale: str | list = "RdBu_r",
    color_scale_mode: Literal["linear","log"] = "linear",
    nan_fill_color: str = "rgba(240,240,240,0.8)",
    show_row_colorbars: bool = True,
    colorbar_len_px: int = 80,
    colorbar_gap_px: int = 10,
    colorbar_thickness_px: int = 8,
    metric_label_map: Mapping[str, str] | None = None,
    float_fmt: str = "{:.4g}",
    title_prefix: str | None = None,
    save: bool = True,
    export_format: str = "pdf",
    save_prefix: str = "table_best",
    save_dir: str | pathlib.Path = "./Output",
    combined: bool = False,
) -> dict[str, go.Figure]:

    if mode not in {"combined", "values", "params"}:
        raise ValueError("mode must be one of {'combined','values','params'}")

    def pretty_metric_name(key: str) -> str:
        if metric_label_map and key in metric_label_map:
            return metric_label_map[key]
        base = key[6:] if key.startswith("total_") else key
        return {"mse": "MSE", "bias2": "Bias²", "var": "Variance"}.get(base, base.upper())
    def to_float(x):
        if isinstance(x, list):
            return [float(v) for v in x]  # keep list, make elems float
        if isinstance(x, tuple):
            return tuple(float(v) for v in x)  # preserve tuple type, too
        if isinstance(x, np.ndarray):
            return x.astype(float)  # keep as ndarray
        return float(x)
    def logify(x):
        if isinstance(x, np.ndarray):
            return np.log10(x.astype(float))
        if isinstance(x, (list, tuple)):
            arr = np.asarray(x, dtype=float)
            logged = np.log10(arr)
            if isinstance(x, list):
                return logged.tolist()
            else:
                return tuple(logged.tolist())
        try:
            return float(np.log10(float(x)))
        except Exception:
            return np.nan
    def first_elem(x):
        if isinstance(x, list):
            return x[0] if x else np.nan
        if isinstance(x, tuple):
            return x[0] if x else np.nan
        if isinstance(x, np.ndarray):
            return x.flat[0] if x.size else np.nan
        return x
    def _lines(s: str) -> int:
        return 1 + str(s).count("<br>") + str(s).count("\n") if s is not None else 1
    figs: dict[str, go.Figure] = {}
    for met_key, (params_df, values_df) in results.items():
        if not isinstance(values_df, pd.DataFrame) or not isinstance(params_df, pd.DataFrame):
            raise TypeError(f"results['{met_key}'] must be a (params_df, values_df) pair of DataFrames.")
        if not values_df.index.equals(params_df.index) or not values_df.columns.equals(params_df.columns):
            raise ValueError(f"Index/columns mismatch for '{met_key}' between params_df and values_df.")
        datasets    = list(values_df.index)
        method_sigs = list(values_df.columns)
        diff_params: set[str] = set()
        if method_sigs and isinstance(method_sigs[0], tuple):
            for idx in range(len(method_sigs[0])):
                pname = method_sigs[0][idx][0]
                if len({sig[idx][1] for sig in method_sigs}) > 1:
                    diff_params.add(pname)
        col_headers = ["Dataset"]
        for sig in method_sigs:
            if isinstance(sig, tuple):
                lbl = " | ".join(f"{k}:{fmt_val(k, v)}" for k, v in sig if k in diff_params) or "default"
                mod = modify_label(lbl)
                col_headers.append((mod if mod is not None else str(sig)).replace(" | ", "<br>"))
            else:
                col_headers.append(str(sig))
        if combined:
            vals = values_df.applymap(to_float).copy()
            if log:
                with np.errstate(divide="ignore", invalid="ignore"):
                    vals = values_df.applymap(logify).copy()
            if normalize_data:
                NotImplementedError('The combined option does not work with normalize. The mse error decomposition would be anyway lost.')
            if met_key == 'combined':
                met_key = 'total_mse'
            else:
                ValueError('The combined option can only be used with "combined" metric key')
        else:
            vals = values_df.astype(float).copy()
            if log:
                with np.errstate(divide="ignore", invalid="ignore"):
                    vals = np.log10(vals.to_numpy(dtype=float))
                vals = pd.DataFrame(vals, index=values_df.index, columns=values_df.columns)
            if normalize_data:
                arr = vals.to_numpy(dtype=float)
                for i in range(arr.shape[0]):
                    arr[i, :] = Normalize(arr[i, :])
                vals = pd.DataFrame(arr, index=values_df.index, columns=values_df.columns)

        metric_label = pretty_metric_name(met_key)
        cell_matrix: list[list[str]] = []
        for ds in datasets:
            row = [str(ds)]
            for sig in method_sigs:
                v = vals.at[ds, sig]
                p = params_df.at[ds, sig]
                if mode == "values":
                    txt = "—" if (v is None or not np.isfinite(float(v))) else float_fmt.format(float(v))
                elif mode == "params":
                    if not isinstance(p, dict) or not p:
                        txt = "—"
                    else:
                        txt = ", ".join(f"{k}:{fmt_val(k, p[k])}" for k in sorted(p))
                elif combined:
                    top = ", ".join(f"{k}:{fmt_val(k, p[k])}" for k in sorted(p)) if isinstance(p, dict) and p else ""
                    bottom_texts = []
                    metric_labels = [pretty_metric_name('total_mse'), pretty_metric_name('total_var'), pretty_metric_name('total_bias2')]
                    for i in range(len(metric_labels)):
                        bot = f"{metric_labels[i]}:{float_fmt.format(float(v[i]))}" if (v[i] is not None and np.isfinite(float(v[i]))) else ""
                        bottom_texts.append(bot)
                    txt = f"{top}<br>{bottom_texts[0]}<br>{bottom_texts[1]}<br>{bottom_texts[2]}" if (top and bottom_texts) else (top or bottom_texts or "—")
                else:
                    top = ", ".join(f"{k}:{fmt_val(k, p[k])}" for k in sorted(p)) if isinstance(p, dict) and p else ""
                    bot = f"{metric_label}:{float_fmt.format(float(v))}" if (v is not None and np.isfinite(float(v))) else ""
                    txt = f"{top}<br>{bot}" if (top and bot) else (top or bot or "—")
                row.append(txt)
            cell_matrix.append(row)
        n_rows = len(datasets)
        font_colors_per_column: list[list[str]] = []
        font_colors_per_column.append(["black"] * n_rows)
        font_weights_per_column: list[list[str]] = []
        font_weights_per_column.append(["normal"] * n_rows)
        metric_base = met_key[6:] if met_key.startswith("total_") else met_key
        if combined:
            vals = values_df.applymap(first_elem)
            vals = vals.apply(pd.to_numeric, errors="coerce")
        if best_mark:
            if best_by == "auto":
                criterion = "min" if metric_base in {"mse", "bias2", "var"} else "min"
            else:
                criterion = best_by
            data = vals.to_numpy(dtype=float)
            for _ in method_sigs:
                font_colors_per_column.append(["black"] * n_rows)
            for r in range(n_rows):
                row_vals = data[r, :]
                finite = np.isfinite(row_vals)
                if not np.any(finite):
                    continue
                target = np.nanmin(row_vals[finite]) if criterion == "min" else np.nanmax(row_vals[finite])
                is_best = np.isclose(row_vals, target, equal_nan=False)
                for c, ok in enumerate(is_best):
                    if ok and finite[c]:
                        v = cell_matrix[r][c + 1]
                        font_colors_per_column[c + 1][r] = best_font_color
                        if isinstance(v, str):
                            cell_matrix[r][c + 1] = v if v.startswith("<b>") else f"<b>{v}</b>"
                        else:
                            cell_matrix[r][c + 1] = f"<b>{v}</b>"
        else:
            for _ in method_sigs:
                font_colors_per_column.append(["black"] * n_rows)
        fill_colors_per_column: list[list[str]] | str = "white"
        per_row_scales: list[tuple[float, float]] = []
        def _to_color(v: float, vmin: float, vmax: float) -> float:
            if not np.isfinite(v):
                return np.nan
            if vmax <= vmin:
                return 0.5
            return (v - vmin) / (vmax - vmin)
        if heatmap_cells:
            if combined:
                tbl_vals = vals.astype(float).to_numpy(copy=True)
            else:
                tbl_vals = values_df.astype(float).to_numpy(copy=True)
            if color_scale_mode == "log":
                with np.errstate(divide="ignore", invalid="ignore"):
                    tbl_vals = np.log10(tbl_vals)
            vmins = np.nanmin(np.where(np.isfinite(tbl_vals), tbl_vals, np.nan), axis=1)
            vmaxs = np.nanmax(np.where(np.isfinite(tbl_vals), tbl_vals, np.nan), axis=1)
            per_row_scales = [(vmins[i], vmaxs[i]) for i in range(tbl_vals.shape[0])]
            fill_colors_per_column = []
            fill_colors_per_column.append(["white"] * n_rows)
            from plotly.colors import get_colorscale
            def _resolve_colorscale(cs_like):
                if isinstance(cs_like, str):
                    return get_colorscale(cs_like)
                try:
                    import numpy as np
                    if isinstance(cs_like, np.ndarray):
                        cs_like = cs_like.tolist()
                except Exception:
                    pass
                if isinstance(cs_like, (list, tuple)) and cs_like:
                    first = cs_like[0]
                    if isinstance(first, (list, tuple)) and len(first) == 2 and isinstance(first[0], (int, float)):
                        out = []
                        for p, c in cs_like:
                            p = float(p)
                            p = 0.0 if p < 0 else (1.0 if p > 1 else p)
                            out.append((p, str(c)))
                        out.sort(key=lambda pc: pc[0])
                        return out
                    else:
                        n = len(cs_like)
                        if n == 1:
                            return [(0.0, str(cs_like[0])), (1.0, str(cs_like[0]))]
                        return [(i / (n - 1), str(c)) for i, c in enumerate(cs_like)]
                return get_colorscale("Reds")
            def _parse_color(col: str):
                col = str(col).strip()
                if col.startswith("#"):
                    h = col[1:]
                    if len(h) == 3:
                        r, g, b = [int(ch * 2, 16) for ch in h]
                    else:
                        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                    return (r, g, b, 1.0)
                if col.lower().startswith("rgba"):
                    nums = col[col.find("(") + 1:col.find(")")].split(",")
                    r, g, b = int(float(nums[0])), int(float(nums[1])), int(float(nums[2]))
                    a = float(nums[3])
                    return (r, g, b, a)
                if col.lower().startswith("rgb"):
                    nums = col[col.find("(") + 1:col.find(")")].split(",")
                    r, g, b = int(float(nums[0])), int(float(nums[1])), int(float(nums[2]))
                    return (r, g, b, 1.0)
                try:
                    cs_tmp = get_colorscale([(0.0, col), (1.0, col)])
                except Exception:
                    cs_tmp = [(0.0, "#000000"), (1.0, "#000000")]
                return _parse_color(cs_tmp[-1][1])
            def _lerp(a, b, t):
                return a + (b - a) * t
            def _interp_color_from_stops(stops, x: float) -> str:
                import math
                if not math.isfinite(x):
                    return nan_fill_color
                x = float(max(0.0, min(1.0, x)))
                # find segment
                for i in range(len(stops) - 1):
                    p0, c0 = stops[i]
                    p1, c1 = stops[i + 1]
                    if x >= p0 and x <= p1:
                        t = 0.0 if p1 == p0 else (x - p0) / (p1 - p0)
                        r0, g0, b0, a0 = _parse_color(c0)
                        r1, g1, b1, a1 = _parse_color(c1)
                        r = int(round(_lerp(r0, r1, t)))
                        g = int(round(_lerp(g0, g1, t)))
                        b = int(round(_lerp(b0, b1, t)))
                        a = _lerp(a0, a1, t)
                        return f"rgba({r},{g},{b},{a:.3f})"
                r, g, b, a = _parse_color(stops[-1][1])
                return f"rgba({r},{g},{b},{a:.3f})"
            cs = _resolve_colorscale(heatmap_colorscale)
            def _interp_color(x: float) -> str:
                return _interp_color_from_stops(cs, x)
            for c in range(tbl_vals.shape[1]):
                col_fill = []
                for r in range(tbl_vals.shape[0]):
                    v = tbl_vals[r, c]
                    vmin, vmax = per_row_scales[r]
                    if not np.isfinite(v):
                        col_fill.append(nan_fill_color)
                    else:
                        x = _to_color(v, vmin, vmax)
                        col_fill.append(_interp_color(x))
                fill_colors_per_column.append(col_fill)
        header_lines = max((_lines(h) for h in col_headers), default=1)
        row_lines = [max((_lines(c) for c in row), default=1) for row in cell_matrix]
        hmax = max((len(h) for h in col_headers), default=1)/10
        HEADER_LINE_PX = 28
        ROW_LINE_PX    = 20
        TOP_PX, BOT_PX = 30, 0
        SIDE_PX        = 10
        hmax_PX        = 10
        table_h = TOP_PX + header_lines * HEADER_LINE_PX + sum(max(1, n) * ROW_LINE_PX for n in row_lines) + BOT_PX + hmax_PX * hmax - 150
        n_methods = max(0, len(col_headers) - 1)
        table_w = 260 + 160 * n_methods
        extra_w = 0
        if heatmap_cells and show_row_colorbars:
            extra_w = colorbar_gap_px + colorbar_len_px
        fig_w = int(table_w + extra_w + SIDE_PX*2)
        fig_h = int(table_h)
        plot_area_w_px = fig_w - SIDE_PX - max(SIDE_PX, colorbar_gap_px)
        reserve_px = (colorbar_gap_px + colorbar_len_px) if (heatmap_cells and show_row_colorbars) else 0
        reserve_frac = min(0.5, max(0.0, reserve_px / max(1, plot_area_w_px))) if (heatmap_cells and show_row_colorbars) else 0
        table_domain_right = 1.0 - reserve_frac
        HEADER_BLUE = "paleturquoise"
        ROW_HEADER_BLUE = "aliceblue"
        header_height_px = HEADER_LINE_PX * header_lines
        columns = list(map(list, zip(*cell_matrix))) if cell_matrix else [[] for _ in col_headers]
        fig = go.Figure(
            data=[go.Table(
                header=dict(values=col_headers, align="left", font=dict(size=12), height=header_height_px, line=dict(color="darkgrey", width=1)),
                cells=dict(
                    values=columns,
                    align="left",
                    font=dict(size=11, color=font_colors_per_column),
                    line=dict(color="darkgrey", width=1),
                    fill_color=([ [ROW_HEADER_BLUE] * n_rows ] + fill_colors_per_column[1:]) if isinstance(fill_colors_per_column, list)
                   else [ [ROW_HEADER_BLUE] * n_rows ] + [fill_colors_per_column]*(len(col_headers)-1),
                    height=ROW_LINE_PX,
                ),
                domain=dict(x=[0.0, table_domain_right], y=[0.0, 1.0]),
            )]
        )
        title_txt = (", ".join([p for p in (title_prefix, f'Optimized for: MSE') if p])) if title_prefix else f'Optimized for: MSE' #set {metric_label} instead of MSE only if we optimize for something else
        fig.update_layout(
            title=title_txt,
            margin=dict(l=SIDE_PX, r=max(SIDE_PX, colorbar_gap_px), t=TOP_PX, b=BOT_PX),
            width=fig_w,
            height=fig_h,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(visible=False, showgrid=True, zeroline=False, showticklabels=False, fixedrange=True, gridcolor="darkgrey", gridwidth=1)
        fig.update_yaxes(visible=False, showgrid=True, zeroline=False, showticklabels=False, fixedrange=True, gridcolor="darkgrey", gridwidth=1)
        if heatmap_cells and show_row_colorbars and per_row_scales:
            row_mid_y_papers = []
            for r in range(len(datasets)):
                mid_global_px = TOP_PX + header_height_px + r * ROW_LINE_PX + (ROW_LINE_PX / 2.0)
                y_paper = 1.0 - (mid_global_px / fig_h)
                row_mid_y_papers.append(float(np.clip(y_paper, 0.0, 1.0)))
            left_pad_frac = SIDE_PX / fig_w
            right_pad_frac = max(SIDE_PX, colorbar_gap_px) / fig_w
            plot_area_frac = 1.0 - left_pad_frac - right_pad_frac
            table_right_paper = left_pad_frac + table_domain_right * plot_area_frac
            x_left_paper = table_right_paper + (colorbar_gap_px / fig_w)
            bar_len_px = max(20, colorbar_len_px)
            def _bar_ticks(vmin_z: float, vmax_z: float, mode: str, max_ticks: int = 9):
                if mode == "log":
                    lo, hi = float(vmin_z), float(vmax_z)
                    if not np.isfinite(lo) or not np.isfinite(hi):
                        return None, None, None, None
                    if hi <= lo:
                        hi = lo + 1e-12
                    e0, e1 = np.floor(lo), np.ceil(hi)
                    span = e1 - e0
                    if span <= 3:
                        exps = np.arange(e0, e1 + 1e-9, 1.0)
                        tickvals = []
                        ticktext = []
                        for e in exps:
                            for m in [1, 2, 5]:
                                v = np.log10(m) + e
                                if v >= lo - 1e-9 and v <= hi + 1e-9:
                                    tickvals.append(float(v))
                                    ticktext.append(f"{m * 10 ** e:.3g}")
                        return tickvals, ticktext, None, "array"
                    else:
                        step = max(1.0, np.ceil(span / (max(3, max_ticks - 1))))
                        exps = np.arange(e0, e1 + 1e-9, step)
                        tickvals = list(exps.astype(float))
                        ticktext = [f"{np.power(10.0, e):.3g}" for e in exps]
                        return tickvals, ticktext, None, "array"
                else:
                    return None, None, "g", "auto"

            for r, (vmin_col, vmax_col) in enumerate(per_row_scales):
                if not (np.isfinite(vmin_col) and np.isfinite(vmax_col)):
                    continue
                if vmax_col <= vmin_col:
                    vmax_col = vmin_col + 1e-12
                if color_scale_mode == "log":
                    vmin_z, vmax_z = vmin_col, vmax_col
                else:
                    vmin_z, vmax_z = vmin_col, vmax_col

                tickvals, ticktext, tickformat, tickmode = _bar_ticks(vmin_z, vmax_z, color_scale_mode, max_ticks=9)

                fig.add_trace(go.Heatmap(
                    z=[[vmin_z, vmax_z]],
                    zmin=vmin_z, zmax=vmax_z,
                    colorscale=heatmap_colorscale,
                    showscale=True,
                    opacity=1e-6,
                    hoverinfo="skip",
                    colorbar=dict(
                        orientation="h",
                        x=x_left_paper,
                        xanchor="left",
                        y=row_mid_y_papers[r],
                        yanchor="middle",
                        lenmode="pixels",
                        len=bar_len_px,
                        thickness=colorbar_thickness_px,
                        outlinewidth=0,
                        tickmode=tickmode,
                        tickvals=tickvals,
                        ticktext=ticktext,
                        tickformat=tickformat,
                        tickfont=dict(size=9),
                        ticks="outside",
                        title=dict(text="", side="top"),
                    ),
                ))
        if save:
            save_dir = pathlib.Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"{save_prefix}_{met_key}_{mode}.{export_format}"
            img = fig.to_image(format=export_format, engine="kaleido", width=fig_w, height=fig_h, scale=1)
            out_path.write_bytes(img)
        figs[met_key] = fig
    return figs

reds_bright = truncate_and_stretch("Reds", cut_top=0.33)

red_blue_bright = truncate_and_stretch("RdBu_r", cut_top=0.33, cut_bottom=0.33)

#Wrapper for the above function to work off of experiment lists
def plot_table_best_of_sweep(
    experiments,
    sweep_params,
    *,
    mode: str = "combined",
    normalize_data: bool = False,
    log: bool = False,
    metric_label_map: Mapping[str, str] | None = None,
    float_fmt: str = "{:.4g}",
    title_prefix: str | None = None,
    save: bool = True,
    export_format: str = "pdf",
    save_prefix: str = "table_best",
    save_dir: str = "./Output",
    font_size: int = 10,
    cell_pad: float = 0.4,
    header_pad: float = 0.6,
    max_chars = 16,
    best_mark: bool = True,
    best_font_color: str = "green",
    best_by: Literal["min","max","auto"] = "min",
    heatmap_cells: bool = True,
    heatmap_colorscale: str | list = reds_bright,
    color_scale_mode: Literal["linear","log"] = "log",
    nan_fill_color: str = "rgba(240,240,240,0.8)",
    show_row_colorbars: bool = False,
    colorbar_len_px: int = 70,
    colorbar_gap_px: int = 7,
    colorbar_thickness_px: int = 8,
    savedf: bool =False,
    decomposition_param: str | None = None,
    combined: bool = False,
):
    experiments = reassing_placeholder_value(experiments)
    results = result_extraction(experiments, sweep_params, metric_keys=None, directory=save_dir, save=savedf, decomposition_param=decomposition_param)
    estimator_name = experiments[0].estimator_name
    plot_tables_from_results(
    results = results,
    mode = mode,
    normalize_data = normalize_data,
    log = log,
    metric_label_map = metric_label_map,
    float_fmt = float_fmt,
    title_prefix = f'Estimatior: {estimator_name.upper()}',
    save = save,
    export_format = export_format,
    save_prefix = save_prefix,
    save_dir = save_dir,
    best_mark = best_mark,
    best_font_color = best_font_color,
    best_by = best_by,
    heatmap_cells = heatmap_cells,
    heatmap_colorscale = heatmap_colorscale,
    color_scale_mode = color_scale_mode,
    nan_fill_color = nan_fill_color,
    show_row_colorbars = show_row_colorbars,
    colorbar_len_px = colorbar_len_px,
    colorbar_gap_px = colorbar_gap_px,
    colorbar_thickness_px = colorbar_thickness_px,
    combined=combined,
    )