import numpy as np
import plotly.graph_objects as go
import os
from typing import Optional, Sequence
from pathlib import Path
###################################################OWN IMPORT###########################################################
from Bagging_for_LID.Plotting.naming_helpers import *
from Bagging_for_LID.Plotting.Plots.other.merge_pdf import *
from Bagging_for_LID.run_files.final_tasks import *
from Bagging_for_LID.Plotting.Plots.other.visualize_pdf import *


def e(x, y):
    return np.array(x)-np.array(y)

def I(x, y):
    return np.array(x)

def log_e(x, y):
    return np.log2(np.array(x)/np.array(y))

def mse(x, y):
    return np.array(x-y)**2

def log_mse(x, y):
    return np.log(np.array(x)/np.array(y))**2

def mae(x, y):
    return np.abs(np.array(x-y))

def log_mae(x, y):
    return np.abs(np.log(np.array(x)/np.array(y)))

def symlog_transform(x, linthresh=1e-6):
    return np.sign(x) * np.log10(1 + np.abs(x) / linthresh)

def extract_sample_objects(experiments, difference_function = log_mae, xyz = (0,1,2)):
    sample_object_list = []
    diffmax1 = np.empty(len(experiments))
    diffmin1 = np.empty(len(experiments))
    for i in range(len(experiments)):
        exp = experiments[i]
        sample_xyz = exp.data[0][:, xyz]
        difference = difference_function(exp.lid_estimates, exp.lid)
        sample_object_list.append((sample_xyz, difference, exp))
        diffmax1[i]=np.max(difference)
        diffmin1[i]=np.min(difference)
        print(np.mean(exp.lid_estimates))
    diffmax = np.max(diffmax1)
    diffmin = np.max(diffmin1)
    return sample_object_list, diffmin, diffmax

def visualize_coords_plotly(
    coords: np.ndarray,
    colors: Optional[np.ndarray] = None,
    vmax: float = 1.0,
    vmin: float = 0,
    scale: str = "linear",  # "linear" or "log"
    title: Optional[str] = None,
    marker_size: int = 3,
    opacity: float = 0.8,
    colorscale: str = "Viridis",
    save_html: Optional[str] = None,
    show_colorbar: bool = True,
):
    coords = np.asarray(coords)

    # --- Accept 2D or 3D ---
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise ValueError("coords must be an (N,2) or (N,3) array.")

    N, dim = coords.shape

    # --- Color array validation ---
    if colors is not None:
        colors = np.asarray(colors).squeeze()
        if colors.shape[0] != N:
            raise ValueError("colors must have shape (N,) matching coords length.")
    else:
        colors = None

    # --- Scale validation ---
    if scale not in ("linear", "log"):
        raise ValueError("scale must be 'linear' or 'log'.")

    if vmax is None:
        raise ValueError("vmax must be provided.")
    if scale == "log" and vmax <= 0:
        raise ValueError("vmax must be > 0 for log scale.")

    # --- Marker setup ---
    marker = dict(size=marker_size, opacity=opacity)

    if colors is None:
        marker["color"] = "rgba(50,50,200,0.8)"
        show_colorbar = False
    else:
        # -------- Linear scale --------
        if scale == "linear":
            color_for_plot = np.clip(colors, vmin, vmax)
            marker["color"] = color_for_plot
            marker["cmin"] = vmin
            marker["cmax"] = float(vmax)
            marker["colorscale"] = colorscale
            marker["showscale"] = bool(show_colorbar)

        # -------- Log scale --------
        else:
            # --- Signed log (symlog) transform ---
            # linthresh defines the region around zero that is linear
            # The smaller, the closer to a pure log
            linthresh = 1e-6

            def symlog(x):
                return np.sign(x) * np.log10(1 + np.abs(x) / linthresh)

            # Transform data
            color_for_plot = symlog(colors)

            # Transform cmin/cmax
            t_vmin = symlog(vmin)
            t_vmax = symlog(vmax)

            marker["color"] = color_for_plot
            marker["colorscale"] = colorscale
            marker["showscale"] = bool(show_colorbar)
            marker["cmin"] = t_vmin
            marker["cmax"] = t_vmax

            # Colorbar ticks (optional, but nice)
            # Use symmetric log ticks:
            tick_vals = []
            tick_text = []

            # decades: e.g. 1, 10, 100...
            for decade in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]:
                if decade <= vmax:
                    tick_vals.append(symlog(decade))
                    tick_text.append(f"{decade:g}")
                    tick_vals.append(symlog(-decade))
                    tick_text.append(f"{-decade:g}")

            # Zero tick
            tick_vals.append(0)
            tick_text.append("0")

            marker["colorbar"] = dict(
                title="value",
                tickmode="array",
                tickvals=tick_vals,
                ticktext=tick_text,
            )

    # =====================================================================
    #  Plot Creation (2D vs 3D)
    # =====================================================================
    if dim == 3:
        # --- 3D scatter ---
        fig = go.Figure(
            data=go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=marker,
            )
        )

        marker["colorbar"] = {
            **marker.get("colorbar", {}),
            "orientation": "h",
            "x": 0.5,
            "y": -0.25,
            "xanchor": "center",
            "len": 1,
            "thickness": 30,
            "title": "|Log₂(LID_Est/LID_GT)|",
            "title_side": "top"
        }

        scene = dict(
            aspectmode="data",
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
        )
        fig.update_layout(scene=scene)

    else:
        # --- 2D scatter ---
        fig = go.Figure(
            data=go.Scattergl(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                marker=marker,
            )
        )

        fig.update_layout(
            xaxis_title="x",
            yaxis_title="y",
            yaxis_scaleanchor="x",
            yaxis_scaleratio=1,
        )

    # --- Title ---
    if title is None:
        title = f"{dim}D scatter (N={N})"
    fig.update_layout(title=title, margin=dict(l=0, r=0, b=0, t=40))

    # --- Colorbar for linear case ---
    if colors is not None and scale == "linear" and show_colorbar:
        cbar = marker.get("colorbar", {})

        # --- horizontal bottom colorbar ---
        cbar.update({
            "orientation": "h",
            "x": 0.5,
            "y": -0.25,
            "xanchor": "center",
            "len": 1,
            "thickness": 30,
            "title": "|Log₂(LID_Est/LID_GT)|",
            "title_side": "top"
        })

        cbar.setdefault("title", "|Log₂(LID_Est/LID_GT)|")
        fig.data[0].marker.colorbar = cbar

    # --- Save HTML ---
    if save_html:
        fig.write_html(save_html, include_plotlyjs="cdn", full_html=True)

    fig.show()
    return fig

def compute_vmin_vmax(
    values: np.ndarray,
    method: str = "percentile",
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
    symmetric: bool = False,
    iqr_k: float = 1.5,
    mad_k: float = 3.0,
) -> Tuple[float, float]:
    """
    Compute robust vmin, vmax from 1D array `values`.

    Parameters
    ----------
    values : np.ndarray
        Input 1D array (can contain nan/inf).
    method : str
        One of {"percentile", "iqr", "mad"}.
    lower_pct, upper_pct : float
        Percentiles for "percentile" method (0-100).
    symmetric : bool
        If True, enforce symmetric bounds around zero: [-M, +M].
        For percentile method this will compute M = percentile(|values|, upper_pct).
    iqr_k : float
        Multiplier for IQR (used when method == "iqr").
    mad_k : float
        Multiplier for MAD (used when method == "mad").
    """
    v = np.asarray(values).astype(float).ravel()
    # Filter finite values
    good = np.isfinite(v)
    if not np.any(good):
        raise ValueError("No finite values in input.")
    v = v[good]

    if method == "percentile":
        if symmetric:
            # symmetric based on absolute values: use upper_pct of |v|
            M = np.percentile(np.abs(v), upper_pct)
            vmin, vmax = -float(M), float(M)
        else:
            vmin = float(np.percentile(v, lower_pct))
            vmax = float(np.percentile(v, upper_pct))

    elif method == "iqr":
        q1 = np.percentile(v, 25.0)
        q3 = np.percentile(v, 75.0)
        iqr = q3 - q1
        lower = q1 - iqr_k * iqr
        upper = q3 + iqr_k * iqr
        if symmetric:
            M = max(abs(lower), abs(upper))
            vmin, vmax = -float(M), float(M)
        else:
            vmin, vmax = float(lower), float(upper)

    elif method == "mad":
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        lower = med - mad_k * mad
        upper = med + mad_k * mad
        if symmetric:
            M = max(abs(lower - med), abs(upper - med))
            vmin, vmax = float(med - M), float(med + M)
            # For biases around zero, med ~ 0; symmetric will yield [-M, +M]
        else:
            vmin, vmax = float(lower), float(upper)

    else:
        raise ValueError("method must be 'percentile', 'iqr', or 'mad'")

    # Avoid degenerate range
    if vmax <= vmin:
        # fallback to min/max of data
        vmin, vmax = float(np.min(v)), float(np.max(v))
        if vmax == vmin:
            # extreme degenerate case
            vmax = vmin + 1.0

    return vmin, vmax


def visualize_experiment_results(experiments, difference_function = log_mae, xyz = (0,1,2), scale = "linear",
                                 marker_size=3, opacity=0.8, title_template='', save_path=None, colorscale= "inferno",
                                 vmin=None, vmax=None):
    def get_character_string2(params):
        string = ''
        for key, value in params.items():
            string += key + f'-{value}' + '_'
        string.rstrip('_')
        return string

    sample_object_list, diffmin, diffmax = extract_sample_objects(experiments, difference_function, xyz)
    save_names = []
    character_strings = []
    all_colors = np.concatenate([sample_object_list[i][1] for i in range(len(sample_object_list))])
    if vmin is None and vmax is None:
        vmin, vmax = compute_vmin_vmax(all_colors, method="percentile", lower_pct=5, upper_pct=95, symmetric=False)
    #if vmin is None:
    #    vmin= -np.maximum(diffmin, diffmax)
    #if vmax is None:
    #    vmax = np.maximum(diffmin, diffmax)
    for i in range(len(sample_object_list)):
        sample_object = sample_object_list[i]
        character_string = get_character_string2(sample_object[2].params)
        #title = f'{title_template}_{character_string}'
        title = ''
        save_name = os.path.join(save_path, f'{character_string}.html')
        show_colorbar = i == len(sample_object_list)
        visualize_coords_plotly(
        sample_object[0],
        sample_object[1],
        vmin=vmin,
        vmax=vmax,
        scale=scale,
        title=title,
        marker_size = marker_size,
        opacity = opacity,
        colorscale=colorscale,
        save_html = save_name,
        show_colorbar = show_colorbar,
        )
        save_names.append(save_name)
        character_strings.append(character_string)
    return save_names, character_strings

if __name__ == "__main__":

    param_dicts1 = {'dataset_name': 'M13a_Scurve',
                        'n': 30000,
                        'lid': None,
                        'dim': None,
                        'estimator_name': 'mle',
                        'bagging_method': None,
                        'submethod_0': '0',
                        'submethod_error': 'log_diff',
                        'k': 10,
                        'sr': 0.3,
                        'Nbag': 10,
                        'pre_smooth': False,
                        'post_smooth': [False, True],
                        't': 1}

    param_dicts2 = {'dataset_name': 'M13a_Scurve',
                        'n': 30000,
                        'lid': None,
                        'dim': None,
                        'estimator_name': 'mle',
                        'bagging_method': 'bag',
                        'submethod_0': '0',
                        'submethod_error': 'log_diff',
                        'k': 10,
                        'sr': 0.3,
                        'Nbag': 10,
                        'pre_smooth': [False, True],
                        'post_smooth': [False, True],
                        't': 1}


    param_dicts = [param_dicts1, param_dicts2]

    experiments = new_result_generator(param_dicts, multiprocess=False, load=True, load_data=True, worker_count=None,
                         save_name='res', directory=r'.\plots')

    save_path = r'.\plots'

    experiments = [experiments[1], experiments[5], experiments[0], experiments[4], experiments[3], experiments[2]]
    for experiment in experiments:
        print(f'{experiment.bagging_method}_{experiment.pre_smooth}_{experiment.post_smooth}')

    save_names, character_strings = visualize_experiment_results(experiments, difference_function = log_mae, xyz = (0,1,2), scale = "linear",
                                              marker_size=2, title_template='', save_path=save_path, opacity=0.8,
                                              colorscale='RdBu_r', vmin=None, vmax=None)

    paths = save_names
    names = [f'{character_strings[i]}.pdf' for i in range(len(character_strings))]

    for i in range(len(paths)):
        path = paths[i]
        name = names[i]

        pdf_path = html_3d_to_pdf(
            output_pdf=name,
            html_path=path,
            selector=".js-plotly-plot",
            library="plotly",
            distance=1.8,
            azimuth_deg=60,
            elevation_deg=20
        )

    save_name_path = r''
    savenames = [os.path.join(save_name_path, f'{character_strings[i]}.pdf') for i in range(len(character_strings))]

    for i in range(len(savenames)):
        if i < len(savenames):
            crop_pdf(savenames[i],
            trim_left=0.20,
            trim_right=0.20,
            trim_top=0.25,
            trim_bottom=0.05,
            overwrite=True,
            )
        else:
            crop_pdf(savenames[i],
            trim_left=0.20,
            trim_right=0.20,
            trim_top=0.25,
            trim_bottom=0,
            overwrite=True,
            )

    merge_pdfs_grid_mupdf(savenames, os.path.join(save_name_path, 'main_image.pdf'),
                          cols=3, rows=2, padding=0.0, margin=0.0,
                          order="row")