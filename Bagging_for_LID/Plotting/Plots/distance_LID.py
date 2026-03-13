import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Circle
from matplotlib.colors import TwoSlopeNorm
import os

def _extract_k_radius_for_point(exp, i, *, compute_if_missing=True, n_points=None):
    if n_points is None:
        try:
            n_points = int(np.asarray(exp.data[0]).shape[0])
        except Exception:
            n_points = None
    def pick_from(A):
        A = np.asarray(A)
        if A.ndim == 0:
            raise ValueError("distance array is 0-D (likely None).")
        if A.ndim == 1:
            if i >= A.shape[0]:
                raise IndexError(f"i={i} out of bounds for shape {A.shape}")
            return float(A[i])
        if A.ndim == 2:
            if n_points is not None:
                if A.shape[0] == n_points:
                    return float(A[i, -1])
                if A.shape[1] == n_points:
                    return float(A[-1, i])
            if i < A.shape[0]:
                return float(A[i, -1])
            if i < A.shape[1]:
                return float(A[-1, i])
            raise IndexError(f"i={i} out of bounds for shape {A.shape}")
        raise ValueError(f"Unsupported distance array ndim={A.ndim}, shape={A.shape}")
    if compute_if_missing:
        if getattr(exp, "point_avg_knn_dists", None) is None and getattr(exp, "bag_avg_knn_dists", None) is None:
            if hasattr(exp, "calc_knn_dists"):
                exp.calc_knn_dists()
    A = getattr(exp, "point_avg_knn_dists", None)
    if A is not None:
        Aarr = np.asarray(A)
        if n_points is None or (Aarr.ndim == 2 and (Aarr.shape[0] == n_points or Aarr.shape[1] == n_points)) or (Aarr.ndim == 1 and Aarr.shape[0] == n_points):
            return pick_from(A)
    B = getattr(exp, "bag_avg_knn_dists", None)
    if B is None:
        raise ValueError("No usable distance array found on experiment (point_avg_knn_dists / bag_avg_knn_dists).")
    return pick_from(B)

def _extract_point_estimate_and_err(lid_estimates, i, error="sem"):
    e = np.asarray(lid_estimates)
    if e.ndim == 1:
        return float(e[i]), 0.0
    if e.shape[0] == e.shape[1]:
        pass
    if e.shape[-1] == e.shape[-2]:
        pass
    if e.shape[-1] == 0:
        return np.nan, np.nan
    if i < e.shape[-1]:
        vals = e[..., i].ravel()
    elif i < e.shape[0]:
        vals = e[i, ...].ravel()
    else:
        raise IndexError("Index i out of bounds for lid_estimates.")
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    mean = float(vals.mean())
    if vals.size <= 1:
        return mean, 0.0
    std = float(vals.std(ddof=1))
    if error == "std":
        return mean, std
    elif error == "sem":
        return mean, std / np.sqrt(vals.size)
    else:
        raise ValueError("error must be 'std' or 'sem'")

def plot_lid_field(
    experiments,
    i,
    *,
    dims=(0, 1),
    q=None,
    extent=None,
    extent_mode="knn_radius",
    extent_pad=0.08,
    n=700,
    cmap_name="RdBu_r",
    symmetric_error_scale=True,
    error_scale=None,
    figsize=(12, 7),
    constrained_layout=True,
    show_points=True,
    points_kwargs=None,
    circle_color="k",
    circle_linestyle="--",
    circle_linewidth=1.6,
    label_angles_deg=(45, 20, -5, -30, -55),
    label_angle_offset_deg=None,
    circle_label_fmt="{k}",
    circle_mode="min_med_max",
    circle_ks=None,
    include_best_k=True,
    best_k=None,
    show_colorbar=True,
    colorbar_label=r"$\widehat{\mathrm{LID}}(q;k)$",
    title=None,
    ax=None,
    save_name='_',
    save_path=None,
    cbar_bottom = True
):
    if len(experiments) == 0:
        raise ValueError("experiments list is empty.")
    exp0 = experiments[0]
    data = np.asarray(exp0.data[0])
    if data.ndim != 2:
        raise ValueError("experiments[0].data[0] must be 2D array of shape (n, dim).")
    d0, d1 = dims
    data2 = data[:, [d0, d1]]
    if q is None:
        qx, qy = float(data2[i, 0]), float(data2[i, 1])
    else:
        qx, qy = float(q[0]), float(q[1])
    gt = float(getattr(exp0, "lid", np.nan))
    if not np.isfinite(gt):
        raise ValueError("exp.lid (ground truth) is not finite; cannot center colormap at GT.")
    ks, radii, vals = [], [], []
    n_points = data.shape[0]
    for exp in experiments:
        ks.append(int(getattr(exp, "k")))
        radii.append(_extract_k_radius_for_point(exp, i, compute_if_missing=True, n_points=n_points))
        vals.append(float(np.asarray(exp.lid_estimates)[i]))
    ks = np.asarray(ks, dtype=int)
    radii = np.asarray(radii, dtype=float)
    vals = np.asarray(vals, dtype=float)
    order = np.argsort(radii)
    ks, radii, vals = ks[order], radii[order], vals[order]
    if include_best_k:
        if best_k is None:
            best_idx = int(np.nanargmin((vals - gt) ** 2))
            best_k = int(ks[best_idx])
        else:
            best_k = int(best_k)
    if extent_mode == "knn_radius":
        r_max = float(np.nanmax(radii))
        if not np.isfinite(r_max) or r_max <= 0:
            raise ValueError("Max radius is not finite/positive; cannot set extent from kNN radii.")
        pad = extent_pad * r_max
        xmin, xmax = qx - (r_max + pad), qx + (r_max + pad)
        ymin, ymax = qy - (r_max + pad), qy + (r_max + pad)
    elif extent_mode == "data_bounds":
        xmin, xmax = float(np.min(data2[:, 0])), float(np.max(data2[:, 0]))
        ymin, ymax = float(np.min(data2[:, 1])), float(np.max(data2[:, 1]))
        pad_x = extent_pad * (xmax - xmin + 1e-12)
        pad_y = extent_pad * (ymax - ymin + 1e-12)
        xmin, xmax = xmin - pad_x, xmax + pad_x
        ymin, ymax = ymin - pad_y, ymax + pad_y
    elif extent_mode == "manual":
        if extent is None:
            raise ValueError("extent_mode='manual' requires extent=(xmin,xmax,ymin,ymax).")
        xmin, xmax, ymin, ymax = map(float, extent)
    else:
        raise ValueError("extent_mode must be 'knn_radius', 'data_bounds', or 'manual'.")
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    Xg, Yg = np.meshgrid(xs, ys)
    R = np.sqrt((Xg - qx) ** 2 + (Yg - qy) ** 2)
    idx = np.searchsorted(radii, R, side="right")
    idx = np.clip(idx, 0, len(vals) - 1)
    V = vals[idx]
    E = V - gt
    err_at_knots = vals - gt
    if error_scale is not None:
        m = float(error_scale)
    else:
        m = float(np.nanmax(np.abs(err_at_knots)))
        if not np.isfinite(m) or m <= 0:
            m = 1e-6
    if symmetric_error_scale:
        norm = TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m)
    else:
        vmin = float(np.nanmin(err_at_knots))
        vmax = float(np.nanmax(err_at_knots))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
            vmin, vmax = -1e-6, 1e-6
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=constrained_layout)
    else:
        fig = ax.figure
    im = ax.imshow(
        E,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        cmap=plt.get_cmap(cmap_name),
        norm=norm,
        interpolation="nearest",
        zorder=0,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if show_points:
        if points_kwargs is None:
            points_kwargs = dict(s=10, c="green", alpha=0.25, linewidths=0)
        P = data2
        mask = (P[:, 0] >= xmin) & (P[:, 0] <= xmax) & (P[:, 1] >= ymin) & (P[:, 1] <= ymax)
        P = P[mask]
        if P.size:
            ax.scatter(P[:, 0], P[:, 1], zorder=5, **points_kwargs)
    ax.scatter([qx], [qy], s=40, color="red", zorder=8)
    ax.text(qx, qy, "q", color="k", fontsize=18, fontweight="bold",
            ha="left", va="bottom", zorder=9)
    if circle_ks is not None:
        keep_set = set(int(k) for k in circle_ks)
    elif circle_mode == "all":
        keep_set = set(int(k) for k in ks)
    elif circle_mode == "min_med_max":
        k_sorted = np.sort(ks)
        k_min = int(k_sorted[0])
        k_med = int(k_sorted[-len(k_sorted)//3])
        k_max = int(k_sorted[-1])
        keep_set = {k_min, k_med, k_max}
    else:
        raise ValueError("circle_mode must be 'all' or 'min_med_max', or provide circle_ks.")
    if include_best_k:
        keep_set.add(int(best_k))
    draw_mask = np.array([k in keep_set for k in ks], dtype=bool)
    selected_ids = np.where(draw_mask)[0]
    for j in selected_ids:
        k = int(ks[j])
        t = float(radii[j])
        v = float(vals[j])
        if not np.isfinite(t) or t <= 0:
            continue
        ax.add_patch(Circle(
            (qx, qy),
            t,
            fill=False,
            linestyle=circle_linestyle,
            linewidth=circle_linewidth,
            edgecolor=circle_color,
            zorder=6,
        ))
        base_deg = label_angles_deg[j % len(label_angles_deg)]
        ang = np.deg2rad(base_deg + float(label_angle_offset_deg or 0.0))
        if label_angle_offset_deg is None:
            ang = np.deg2rad(45)
        else:
            ang = np.deg2rad(base_deg + float(label_angle_offset_deg or 0.0))
        lx = qx + t * np.cos(ang)
        ly = qy + t * np.sin(ang)
        ax.text(
            lx, ly,
            circle_label_fmt.format(k=k, t=t, v=v),
            color="k",
            fontsize=20,
            ha="left",
            va="bottom",
            zorder=7,
        )
    if title is None:
        title = f"Pointwise LID error rings (point i={i})"
    ax.set_title(title)
    if show_colorbar:
        if cbar_bottom:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.set_label("")
            cbar.ax.set_xlabel(colorbar_label, fontsize=18, labelpad=2)
            cbar.ax.xaxis.set_label_position("bottom")
            cbar.ax.xaxis.set_label_coords(0.60, -0.015)
        else:
            if show_colorbar:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
                cbar.set_label("")
                cbar.ax.xaxis.set_label_position("top")
                cbar.ax.xaxis.set_ticks_position("top")
                cbar.ax.set_xlabel(colorbar_label, fontsize=18, labelpad=6)
                cbar.ax.xaxis.set_label_coords(0.60, 1.02)
        selected_ids = np.where(draw_mask)[0]
        unique = {}
        for j in selected_ids:
            unique[int(ks[j])] = j
        selected_ids = np.array([unique[k] for k in sorted(unique.keys())], dtype=int)
        tick_vals = (vals[selected_ids] - gt)
        tick_lbls = []
        LIDhat = r"$\widehat{\mathrm{LID}}=$"
        for j in selected_ids:
            k = int(ks[j])
            t = float(radii[j])
            v = float(vals[j])
            dv = v - gt
            tag = " (best)" if (include_best_k and k == int(best_k)) else ""
            tick_lbls.append(f"{LIDhat}{v:.5g},\nk={k}{tag},\nt={t:.3g}")
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels(tick_lbls)
    if save_path is None:
        save_path = os.getcwd()
    out = os.path.join(save_path, f"{save_name}")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    return fig, ax, im, (ks, radii, vals, gt, best_k)

def plot_lid_curve(
    experiments,
    i,
    *,
    param="k",
    invert=False,
    show_error=True,
    n_estimates=None,
    band_alpha=0.2,
    figsize=(10, 4.8),
    title=None,
    ax=None,
    markers=False,
    save_name='_',
    save_path=None,
):
    if len(experiments) == 0:
        raise ValueError("experiments list is empty.")
    if invert:
        x = np.array([1.0 / float(getattr(exp, param)) for exp in experiments], dtype=float)
    else:
        x = np.array([float(getattr(exp, param)) for exp in experiments], dtype=float)
    ys = []
    yerr = []
    gt = []
    for exp in experiments:
        ys.append(float(np.asarray(exp.lid_estimates)[i]))
        if show_error:
            std_arr = getattr(exp, "lid_estimates_std", None)
            if std_arr is None:
                yerr.append(0.0)
            else:
                s = float(np.asarray(std_arr)[i])
                yerr.append(s)
        else:
            yerr.append(0.0)
        gt.append(float(getattr(exp, "lid", np.nan)))
    ys = np.asarray(ys, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    gt = np.asarray(gt, dtype=float)
    order = np.argsort(x)
    x, ys, yerr, gt = x[order], ys[order], yerr[order], gt[order]
    if np.all(np.isfinite(gt)):
        gt_line_val = float(np.nanmean(gt))
    else:
        gt_line_val = np.nan
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=120)
    else:
        fig = ax.figure
    if np.isfinite(gt_line_val):
        ax.axhline(gt_line_val, linewidth=2, color="blue", label="Ground Truth LID")
    if markers:
        (est_line,) = ax.plot(
            x, ys, marker="s", linewidth=2, markersize=4,
            color="red", label="Pointwise LID estimate"
        )
    else:
        (est_line,) = ax.plot(x, ys, linewidth=2, color="red", label="$\widehat{\mathrm{LID}}_{MLE}$(q;k) Estimate")
    est_color = est_line.get_color()
    if show_error and np.any(yerr > 0):
        lower = np.clip(ys - yerr, 0, None)
        upper = ys + yerr
        band_label = f"± STD ({n_estimates} datasets, fixed q)"
        ax.fill_between(x, lower, upper, alpha=band_alpha, color=est_color, label=band_label)
    ax.set_xlabel(f"1/{param}" if invert else param)
    ax.set_ylabel(r"LID")
    ax.set_title(title or f"Pointwise LID vs {param} (point i={i})")
    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=24)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    if save_path is None:
        save_path = os.getcwd()
    out = os.path.join(save_path, f"{save_name}")
    if save_name is not None:
        fig.savefig(out, dpi=300, bbox_inches="tight")
    return fig, ax

