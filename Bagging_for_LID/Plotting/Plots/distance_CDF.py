import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
import matplotlib.transforms as transforms
import os

def weighted_quantile(values, qs, sample_weight=None):
    values = np.asarray(values, dtype=float)
    qs = np.asarray(qs, dtype=float)
    if sample_weight is None:
        try:
            return np.quantile(values, qs, method="higher")
        except TypeError:
            return np.quantile(values, qs, interpolation="higher")
    w = np.asarray(sample_weight, dtype=float)
    if w.shape != values.shape:
        raise ValueError("sample_weight must have same shape as values")
    if np.any(w < 0):
        raise ValueError("sample_weight must be nonnegative")
    s = w.sum()
    if s <= 0:
        raise ValueError("sum(sample_weight) must be > 0")
    idx = np.argsort(values)
    v = values[idx]
    w = w[idx] / s
    cw = np.cumsum(w)
    cw = np.maximum.accumulate(cw)
    return np.interp(qs, cw, v, left=v[0], right=v[-1])

def distance_quantiles_from_samples(samples, q=(0.0, 0.0), probs=(0.25, 0.5, 0.75), weights=None):
    samples = np.asarray(samples, dtype=float)
    q = np.asarray(q, dtype=float).reshape(2,)
    d = np.sqrt(((samples - q) ** 2).sum(axis=1))
    return weighted_quantile(d, probs, sample_weight=weights)

def radial_cdf_from_pdf_grid(pdf, x, y, q=(0.0, 0.0), nbins=2000):
    pdf = np.asarray(pdf, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = float(np.mean(np.diff(x)))
    dy = float(np.mean(np.diff(y)))
    qx, qy = q
    X, Y = np.meshgrid(x, y)
    R = np.sqrt((X - qx) ** 2 + (Y - qy) ** 2)
    mass = pdf * dx * dy
    total = mass.sum()
    if total <= 0:
        raise ValueError("pdf integrates to zero/negative over grid.")
    mass = mass / total
    rmax = float(R.max())
    edges = np.linspace(0.0, rmax, int(nbins) + 1)
    hist, _ = np.histogram(R.ravel(), bins=edges, weights=mass.ravel())
    cdf = np.cumsum(hist)
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf = np.maximum.accumulate(cdf)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, cdf

def distance_quantiles_from_pdf_grid(pdf, x, y, q=(0.0, 0.0), probs=(0.25, 0.5, 0.75), nbins=2000):
    centers, cdf = radial_cdf_from_pdf_grid(pdf, x, y, q=q, nbins=nbins)
    probs = np.asarray(probs, dtype=float)
    return np.interp(probs, cdf, centers, left=centers[0], right=centers[-1])

def empirical_CDF(samples, q=(0.0, 0.0)):
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("samples must be shape (N,2)")
    q = np.asarray(q, dtype=float).reshape(2,)
    d = np.sqrt(((samples - q) ** 2).sum(axis=1))
    d_sorted = np.sort(d)
    n = d_sorted.size
    def F(t):
        t = np.asarray(t, dtype=float)
        return np.searchsorted(d_sorted, t, side="right") / n
    return F

def cdf_of_distance_from_pdf_grid(pdf, x, y, q=(0.0, 0.0), nbins=2000):
    pdf = np.asarray(pdf, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if pdf.shape != (y.size, x.size):
        raise ValueError("pdf must have shape (len(y), len(x))")
    dx = float(np.mean(np.diff(x)))
    dy = float(np.mean(np.diff(y)))
    if dx <= 0 or dy <= 0:
        raise ValueError("x and y must be increasing")
    qx, qy = q
    X, Y = np.meshgrid(x, y)
    R = np.sqrt((X - qx) ** 2 + (Y - qy) ** 2)
    mass = pdf * dx * dy
    total = mass.sum()
    if total <= 0:
        raise ValueError("pdf integrates to zero or negative over the grid")
    mass = mass / total
    rmax = float(R.max())
    edges = np.linspace(0.0, rmax, int(nbins) + 1)
    hist, _ = np.histogram(R.ravel(), bins=edges, weights=mass.ravel())
    cdf = np.cumsum(hist)
    cdf = np.clip(cdf, 0.0, 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    def F(t):
        t = np.asarray(t, dtype=float)
        return np.interp(t, centers, cdf, left=0.0, right=1.0)
    return F

def plot_distance_cdf_field(
    F,
    q=(0.0, 0.0),
    extent=5.0,
    grid_res=500,
    cmap_name="viridis",
    show_colorbar=True,
    ax=None,
    force_white_at_q=True,
    quantile_probs=(0.25, 0.5, 0.75),
    circle_ts=None,
    circle_linestyles=("--", "--", "--"),
    label_angles_deg=(45, 20, -5, -30, -55),
    label_angle_offset_deg=None,
    figsize=(12, 7),
    constrained_layout=True,
    sample_points=None,
    sample_kwargs=None,
    circle_label_fmt="{t:.3g}",
    colorbar_label=r"$\hat{F}_{\|X-q\|}(t)$",
    save_path=None,
    save_name='',
    cbar_bottom=False
):
    if save_path is None:
        save_path = os.getcwd()
    qx, qy = map(float, q)
    if np.isscalar(extent):
        L = float(extent)
        xmin, xmax, ymin, ymax = -L, L, -L, L
    else:
        xmin, xmax, ymin, ymax = map(float, extent)
    xs = np.linspace(xmin, xmax, grid_res)
    ys = np.linspace(ymin, ymax, grid_res)
    X, Y = np.meshgrid(xs, ys)
    R = np.sqrt((X - qx) ** 2 + (Y - qy) ** 2)
    try:
        V = np.asarray(F(R), dtype=float)
    except Exception:
        V = np.vectorize(F)(R).astype(float)
    V = np.clip(V, 0.0, 1.0)
    if force_white_at_q:
        try:
            F0 = float(np.clip(F(0.0), 0.0, 1.0))
        except Exception:
            F0 = float(np.clip(np.vectorize(F)(0.0), 0.0, 1.0))
        denom = max(1.0 - F0, 1e-12)
        V = np.clip((V - F0) / denom, 0.0, 1.0)
        def p_to_plot(p):
            return (p - F0) / denom
    else:
        def p_to_plot(p):
            return p
    base = plt.get_cmap(cmap_name)
    ncol = 256
    a = np.linspace(0, 1, ncol)
    fade = 0.18
    colors = np.zeros((ncol, 4))
    for i, ai in enumerate(a):
        if ai <= fade:
            t = ai / max(fade, 1e-12)
            colors[i] = (1 - t) * np.array([1, 1, 1, 1]) + t * np.array(base(0.0))
        else:
            t = (ai - fade) / (1 - fade)
            colors[i] = base(t)
    cmap = LinearSegmentedColormap.from_list("white_to_" + cmap_name, colors)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=constrained_layout)
    else:
        fig = ax.figure
    im = ax.imshow(
        V,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        zorder=0,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if sample_points is not None:
        P = np.asarray(sample_points, dtype=float)
        if P.ndim != 2 or P.shape[1] != 2:
            raise ValueError("sample_points must be shape (M,2)")
        if sample_kwargs is None:
            sample_kwargs = dict(s=8, c="green", alpha=0.25, linewidths=0)
        m = (
            (P[:, 0] >= xmin) & (P[:, 0] <= xmax) &
            (P[:, 1] >= ymin) & (P[:, 1] <= ymax)
        )
        P_vis = P[m]
        if P_vis.size > 0:
            ax.scatter(P_vis[:, 0], P_vis[:, 1], zorder=5, **sample_kwargs)
    ax.scatter([qx], [qy], s=40, color="red", zorder=8)
    ax.text(qx, qy, "q", color="k", fontsize=18, fontweight="bold",
            ha="left", va="bottom", zorder=9)
    if circle_ts is not None:
        ts = np.asarray(circle_ts, dtype=float).ravel()
        if quantile_probs is None:
            try:
                ps = np.asarray(F(ts), dtype=float)
            except Exception:
                ps = np.vectorize(F)(ts).astype(float)
            ps = np.clip(ps, 0.0, 1.0)
        else:
            ps = np.asarray(quantile_probs, dtype=float).ravel()
            if ps.shape != ts.shape:
                try:
                    ps = np.asarray(F(ts), dtype=float)
                except Exception:
                    ps = np.vectorize(F)(ts).astype(float)
                ps = np.clip(ps, 0.0, 1.0)
        for i, (p, t) in enumerate(zip(ps, ts)):
            if not np.isfinite(t) or t <= 0:
                continue
            ls = circle_linestyles[i % len(circle_linestyles)]
            circ = Circle(
                (qx, qy),
                float(t),
                fill=False,
                linestyle=ls,
                linewidth=2.0,
                edgecolor="k",
                zorder=6,
            )
            ax.add_patch(circ)
            base_deg = label_angles_deg[i % len(label_angles_deg)]
            if label_angle_offset_deg is None:
                ang = np.deg2rad(45)
            else:
                ang = np.deg2rad(base_deg + label_angle_offset_deg)
            lx = qx + float(t) * np.cos(ang)
            ly = qy + float(t) * np.sin(ang)
            ax.text(
                lx, ly,
                circle_label_fmt.format(p=float(p), t=float(t)),
                color="k",
                fontsize=20,
                ha="left",
                va="bottom",
                zorder=7,
            )
    else:
        ps = None
        ts = None
    if show_colorbar:
        if cbar_bottom:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label("")
            cbar.ax.set_xlabel(colorbar_label, fontsize=18, labelpad=2)
            cbar.ax.xaxis.set_label_position("bottom")
            cbar.ax.xaxis.set_label_coords(0.60, -0.015)
        else:
            if show_colorbar:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cbar.set_label("")
                cbar.ax.xaxis.set_label_position("top")
                cbar.ax.xaxis.set_ticks_position("top")

                cbar.ax.set_xlabel(colorbar_label, fontsize=18, labelpad=6)
                cbar.ax.xaxis.set_label_coords(0.60, 1.02)
        if circle_ts is not None and ts is not None and ps is not None and len(ts) > 0:
            ticks = []
            labels = []
            for p, t in zip(ps, ts):
                tp = float(p_to_plot(float(p)))
                if 0.0 <= tp <= 1.0:
                    ticks.append(tp)
                    labels.append(f"{float(p):.2f}  (t={float(t):.3g})")
            if ticks:
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(labels)
    plt.savefig(os.path.join(save_path, f'{save_name}'), dpi=300, bbox_inches="tight")
    return fig, ax, im

def plot_distance_cdf_curve(
    *,
    samples=None,
    weights=None,
    pdf=None,
    x=None,
    y=None,
    q=(0.0, 0.0),
    nbins=2000,
    quantile_probs=(0.25, 0.5, 0.75),
    circle_ts=None,
    figsize=(10, 4.8),
    ax=None,
    title=None,
    curve_color="blue",
    curve_linewidth=2.0,
    curve_label=r"$\hat{F}_{\|X-q\|}(t)$",
    marker_color="green",
    marker_linestyle="--",
    marker_linewidth=1.0,
    marker_alpha=0.6,
    marker_text_fmt="{t:.3g}",
    marker_text_color="green",
    marker_fontsize=12,
    small_p_thresh=0.0005,
    bottom_label_offset_points=(12, 4),
    right_label_offset_points=(-6, 8),
    show_grid=True,
    grid_alpha=0.35,
    save_path=None,
    save_name="",
):
    if save_path is None:
        save_path = os.getcwd()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=120)
    else:
        fig = ax.figure
    q = np.asarray(q, dtype=float).reshape(2,)
    if samples is not None:
        S = np.asarray(samples, dtype=float)
        if S.ndim != 2 or S.shape[1] != 2:
            raise ValueError("samples must be shape (N,2)")
        d = np.sqrt(((S - q) ** 2).sum(axis=1))
        if weights is None:
            d_sorted = np.sort(d)
            nS = d_sorted.size
            F_vals = np.arange(1, nS + 1) / nS
            ax.plot(
                d_sorted, F_vals,
                color=curve_color, linewidth=curve_linewidth,
                label=curve_label
            )
            def F_eval(t):
                t = np.asarray(t, dtype=float)
                return np.searchsorted(d_sorted, t, side="right") / nS
            if circle_ts is None:
                qs = np.asarray(quantile_probs, dtype=float)
                try:
                    ts = np.quantile(d, qs, method="higher")
                except TypeError:
                    ts = np.quantile(d, qs, interpolation="higher")
                ps = np.clip(qs, 0.0, 1.0)
            else:
                ts = np.asarray(circle_ts, dtype=float).ravel()
                ps = np.clip(F_eval(ts), 0.0, 1.0)
        else:
            w = np.asarray(weights, dtype=float)
            if w.shape != (S.shape[0],):
                raise ValueError("weights must be shape (N,)")
            if np.any(w < 0):
                raise ValueError("weights must be nonnegative")
            wsum = w.sum()
            if wsum <= 0:
                raise ValueError("sum(weights) must be > 0")
            idx = np.argsort(d)
            d_sorted = d[idx]
            w_sorted = w[idx] / wsum
            F_vals = np.cumsum(w_sorted)
            F_vals = np.maximum.accumulate(F_vals)
            ax.plot(
                d_sorted, F_vals,
                color=curve_color, linewidth=curve_linewidth,
                label=curve_label
            )
            def F_eval(t):
                t = np.asarray(t, dtype=float)
                j = np.searchsorted(d_sorted, t, side="right") - 1
                j = np.clip(j, -1, d_sorted.size - 1)
                out = np.zeros_like(t, dtype=float)
                mask = j >= 0
                out[mask] = F_vals[j[mask]]
                return out
            if circle_ts is None:
                qs = np.asarray(quantile_probs, dtype=float)
                cw = np.cumsum(w_sorted)
                cw = np.maximum.accumulate(cw)
                ts = np.interp(qs, cw, d_sorted, left=d_sorted[0], right=d_sorted[-1])
                ps = np.clip(qs, 0.0, 1.0)
            else:
                ts = np.asarray(circle_ts, dtype=float).ravel()
                ps = np.clip(F_eval(ts), 0.0, 1.0)
    else:
        if pdf is None or x is None or y is None:
            raise ValueError("Provide either samples=... or pdf=... with x,y.")
        pdf = np.asarray(pdf, dtype=float)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        dx = float(np.mean(np.diff(x)))
        dy = float(np.mean(np.diff(y)))
        if dx <= 0 or dy <= 0:
            raise ValueError("x and y must be increasing")
        qx, qy = q
        Xg, Yg = np.meshgrid(x, y)
        R = np.sqrt((Xg - qx) ** 2 + (Yg - qy) ** 2)
        mass = pdf * dx * dy
        total = mass.sum()
        if total <= 0:
            raise ValueError("pdf integrates to zero/negative over grid.")
        mass = mass / total
        rmax = float(R.max())
        edges = np.linspace(0.0, rmax, int(nbins) + 1)
        hist, _ = np.histogram(R.ravel(), bins=edges, weights=mass.ravel())
        cdf = np.cumsum(hist)
        cdf = np.clip(cdf, 0.0, 1.0)
        cdf = np.maximum.accumulate(cdf)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(
            centers, cdf,
            color=curve_color, linewidth=curve_linewidth,
            label=curve_label
        )
        def F_eval(t):
            t = np.asarray(t, dtype=float)
            return np.interp(t, centers, cdf, left=0.0, right=1.0)
        if circle_ts is None:
            ps = np.asarray(quantile_probs, dtype=float)
            ts = np.interp(ps, cdf, centers, left=centers[0], right=centers[-1])
            ps = np.clip(ps, 0.0, 1.0)
        else:
            ts = np.asarray(circle_ts, dtype=float).ravel()
            ps = np.clip(F_eval(ts), 0.0, 1.0)
    order = np.argsort(ts)
    ts = np.asarray(ts, dtype=float)[order]
    ps = np.asarray(ps, dtype=float)[order]
    trans_bottom = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for p, t in zip(ps, ts):
        if not np.isfinite(t) or t < 0:
            continue
        ax.vlines(
            float(t),
            ymin=0.0, ymax=float(p),
            colors=marker_color,
            linestyles=marker_linestyle,
            linewidth=marker_linewidth,
            alpha=marker_alpha,
            zorder=3,
        )
        if float(p) < float(small_p_thresh):
            ax.annotate(
                marker_text_fmt.format(p=float(p), t=float(t)),
                xy=(float(t), float(p)),
                xycoords="data",
                xytext=right_label_offset_points,
                textcoords="offset points",
                ha="left",
                va="bottom",
                color=marker_text_color,
                fontsize=marker_fontsize,
                clip_on=False,
            )
        else:
            ax.annotate(
                marker_text_fmt.format(p=float(p), t=float(t)),
                xy=(float(t), 0.0),
                xycoords=trans_bottom,
                xytext=bottom_label_offset_points,
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=marker_text_color,
                fontsize=marker_fontsize,
                clip_on=False,
            )
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$t$")
    if title is None:
        title = r"CDF of $\|X-q\|$"
    ax.set_title(title)
    if show_grid:
        ax.grid(True, which="major", linestyle="--", alpha=grid_alpha)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if curve_label:
        ax.legend(frameon=False, fontsize=24)
    plt.tight_layout()
    if save_name:
        out = os.path.join(save_path, f"{save_name}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
    return fig, ax, ts, ps

