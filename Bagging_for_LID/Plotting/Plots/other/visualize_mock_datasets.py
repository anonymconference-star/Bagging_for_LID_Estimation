# Visualize random samples of maps f: [0,1]^n_params -> R^d

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional, Sequence
import scipy

try:
    from mpl_toolkits.mplot3d import Axes3D
except Exception:
    pass

import matplotlib.colors as mcolors


def _call_map_fn(map_fn: Callable, P: np.ndarray) -> np.ndarray:
    try:
        Y = map_fn(P)
    except TypeError:
        Y = map_fn(*[P[:, i] for i in range(P.shape[1])])
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return Y


def visualize_unit_cube_map(
    map_fn: Callable,
    n_params: int,
    samples: int = 50_000,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    marker_size: float = 2.0,
    alpha: float = 0.25,
    seed: int = 0,
    equal_axes: bool = True,
    axes: Sequence[int] = (0, 1, 2),
    color_dim: Optional[int] = None,
    cmap: str = "viridis",
    colorbar: bool = True,
) -> Tuple[plt.Figure, Optional[str]]:

    rng = np.random.default_rng(seed)
    P = rng.random((samples, n_params))
    Y = _call_map_fn(map_fn, P)
    N, d = Y.shape

    axes = tuple(axes)
    if len(axes) not in (2, 3):
        raise ValueError("`axes` must have length 2 or 3 (for 2D/3D plotting).")

    if color_dim is None and d >= len(axes) + 1:

        remaining = [i for i in range(d) if i not in axes]
        color_dim = remaining[0] if remaining else d - 1

    if title is None:
        base = f"Parametric visualization (n_params={n_params}, d={d})"
        if d >= 4:
            title = f"{base} — axes={axes}, color_dim={color_dim}"
        else:
            title = base

    coords = Y[:, list(axes)]
    C = None
    if color_dim is not None and 0 <= color_dim < d:
        C = Y[:, color_dim]

    if len(axes) == 2:
        fig = plt.figure(figsize=(6, 6))
        scatter_kwargs = dict(s=marker_size, alpha=alpha)
        if C is not None:
            scatter_kwargs.update(dict(c=C, cmap=cmap))
        plt.scatter(coords[:, 0], coords[:, 1], **scatter_kwargs)
        if equal_axes:
            plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel(f"dim {axes[0]}")
        plt.ylabel(f"dim {axes[1]}")
        plt.title(title)
        if colorbar and C is not None:
            mappable = plt.cm.ScalarMappable(
                norm=mcolors.Normalize(vmin=np.min(C), vmax=np.max(C)),
                cmap=cmap,
            )
            mappable.set_array([])
            plt.colorbar(mappable, label=f"dim {color_dim}")
        plt.tight_layout()

    else:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        scatter_kwargs = dict(s=marker_size, alpha=alpha)
        if C is not None:
            scatter_kwargs.update(dict(c=C, cmap=cmap))
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], **scatter_kwargs)
        ax.set_xlabel(f"dim {axes[0]}")
        ax.set_ylabel(f"dim {axes[1]}")
        ax.set_zlabel(f"dim {axes[2]}")
        ax.set_title(title)

        if equal_axes and hasattr(ax, "set_box_aspect"):
            ranges = np.ptp(coords, axis=0)
            ranges[ranges == 0] = 1.0
            ax.set_box_aspect(ranges)

        if colorbar and C is not None:
            mappable = plt.cm.ScalarMappable(
                norm=mcolors.Normalize(vmin=np.min(C), vmax=np.max(C)),
                cmap=cmap,
            )
            mappable.set_array([])
            cb = plt.colorbar(mappable, pad=0.1)
            cb.set_label(f"dim {color_dim}")

        plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=160)
    plt.show()
    return fig, save_path

import plotly.graph_objects as go

def visualize_unit_cube_map_plotly(
    map_fn: Callable,
    n_params: int,
    samples: int = 50_000,
    title: Optional[str] = None,
    seed: int = 0,
    save_html: Optional[str] = None,
    axes: Sequence[int] = (0, 1, 2),
    color_dim: Optional[int] = None,
    marker_size: int = 2,
    opacity: float = 0.35,
    colorscale: str = "Viridis",
):

    rng = np.random.default_rng(seed)
    P = rng.random((samples, n_params))
    Y = _call_map_fn(map_fn, P)
    N, d = Y.shape

    axes = tuple(axes)
    if len(axes) not in (2, 3):
        raise ValueError("`axes` must have length 2 or 3 (for 2D/3D plotting).")

    if color_dim is None and d >= len(axes) + 1:
        remaining = [i for i in range(d) if i not in axes]
        color_dim = remaining[0] if remaining else d - 1

    if title is None:
        base = f"Parametric visualization (n_params={n_params}, d={d})"
        if d >= 4:
            title = f"{base} — axes={axes}, color_dim={color_dim}"
        else:
            title = base

    coords = Y[:, list(axes)]
    C = None if color_dim is None or not (0 <= color_dim < d) else Y[:, color_dim]

    if len(axes) == 2:
        fig = go.Figure(go.Scattergl(
            x=coords[:, 0], y=coords[:, 1],
            mode="markers",
            marker=dict(size=marker_size, opacity=opacity,
                        color=C, colorscale=colorscale, showscale=C is not None),
        ))
        fig.update_layout(
            title=title,
            xaxis_title=f"dim {axes[0]}",
            yaxis_title=f"dim {axes[1]}",
            yaxis_scaleanchor="x",
            yaxis_scaleratio=1,
        )
    else:
        fig = go.Figure(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers",
            marker=dict(size=marker_size, opacity=opacity,
                        color=C, colorscale=colorscale, showscale=C is not None),
        ))
        fig.update_layout(
            title=title,
            scene=dict(aspectmode="data",
                       xaxis_title=f"dim {axes[0]}",
                       yaxis_title=f"dim {axes[1]}",
                       zaxis_title=f"dim {axes[2]}"),
        )

    if save_html:
        fig.write_html(save_html, include_plotlyjs="cdn", full_html=True)
    fig.show()
    return fig

def map_circle(x0):
    return np.c_[np.cos(2 * np.pi * x0), np.sin(2 * np.pi * x0)]

def map_disk(x0, x1, x2):
    return np.c_[
        (x1 ** 2) * np.cos(2 * np.pi * x0),
        (x2 ** 2) * np.sin(2 * np.pi * x0),
    ]

def map_3d(x0, x1, x2, x3):
    X = (x1 ** 2) * np.cos(2 * np.pi * x0)
    Y = (x2 ** 2) * np.sin(2 * np.pi * x0)
    Z = x1 + x2 + (x1 - x3) ** 2
    return np.c_[X, Y, Z]

def map_3d2(x0, x1, x2, x3):
    X = (x1 ** 2) * np.cos(2 * np.pi * x0)
    Y = (x2 ** 2) * np.sin(2 * np.pi * x0)
    Z = x1 - 2*x2 + (x0 - x3) ** 2
    return np.c_[X, Y, Z]

def map_3d3(x0, x1, x2, x3):
    X = (x1 ** 2) * np.cos(2 * np.pi * x0)
    Y = (x2 ** 2) * np.sin(2 * np.pi * x0)
    Z = -x1 - 2*x2 + (x2-x3)**2
    return np.c_[X, Y, Z]

def map_3d4(x0, x1, x2, x3):
    X = (x1 ** 2) * np.cos(2 * np.pi * x0)
    Y = (x2 ** 2) * np.sin(2 * np.pi * x0)
    Z = x0**2 -x1**2 +x2**2-x3**2
    return np.c_[X, Y, Z]

def map_3d5(x0, x1, x2, x3):
    X = x1 - 2*x2 + (x0 - x3) ** 2
    Y = -x1 - 2*x2 + (x2-x3)**2
    Z = x0**2 -x1**2 +x2**2-x3**2
    return np.c_[X, Y, Z]

def map_3d6(x0, x1, x2, x3):
    X = x1 + x2 + (x1 - x3) ** 2
    Y = x1 - 2*x2 + (x0 - x3) ** 2
    Z = -x1 - 2*x2 + (x2-x3)**2
    return np.c_[X, Y, Z]

def map_weird1(x0, x1):
    X = x1*np.cos(2*np.pi*x0)
    Y = x1*np.sin(2*np.pi*x0)
    Z = x0*np.cos(2*np.pi*x1)
    return np.c_[X, Y, Z]

def map_weird2(x0, x1):
    X = x1*np.cos(2*np.pi*x0)
    Y = x1*np.sin(2*np.pi*x0)
    Z = x0*np.sin(2*np.pi*x1)
    return np.c_[X, Y, Z]

def map_4d(x0, x1, x2, x3):
    X = (x1 ** 2) * np.cos(2 * np.pi * x0)
    Y = (x2 ** 2) * np.sin(2 * np.pi * x0)
    Z = x1 + x2 + (x1 - x3) ** 2
    W = x0 ** 2 - x1 ** 2 + x2 ** 2 - x3 ** 2
    return np.c_[X, Y, Z, W]

def map_weird_full(x0, x1):
    X = x1*np.cos(2*np.pi*x0)
    Y = x1*np.sin(2*np.pi*x0)
    Z = x0*np.cos(2*np.pi*x1)
    W = x0*np.sin(2*np.pi*x1)
    return np.c_[X, Y, Z, W]

def map_weird_full2(x0, x1):
    X = x1*np.cos(2*np.pi*x0)
    Y = x1*np.sin(2*np.pi*x0)
    Z = x1*np.cos(2*np.pi*x0)
    W = x1*np.sin(2*np.pi*x0)
    return np.c_[X, Y, Z, W]

def map_weird_full3(x0, x1):
    X = x1*np.cos(2*np.pi*x0)
    Y = x1*np.sin(2*np.pi*x0)
    Z = x1*np.cos(2*np.pi*x0)
    W = x0*np.sin(2*np.pi*x1)
    return np.c_[X, Y, Z, W]

def helix(x0, x1):
    x0 = x0*10*np.pi
    x1 = x1*10*np.pi
    X = x0*np.cos(x1)
    Y = x0*np.sin(x1)
    Z = 0.5*x1
    W = 0*x1
    return np.c_[X, Y, Z, W]

def roll(x0, x1):
    x0 = x0*3*np.pi + 1.5*np.pi
    x1 = x1*21
    X = x0*np.cos(x0)
    Y = x1
    Z = x0*np.sin(x0)
    W = 0*x1
    return np.c_[X, Y, Z, W]

def affine3to5(x0, x1, x2):
    x0, x1, x2 = x0*4, x1*4, x2*4
    X = 1.2*x0-0.5*x1+3
    Y = 0.5*x0 + 0.9*x1 -1
    Z = -0.5*x0 -0.2*x1 +x2
    W = 0.4*x0 -0.9*x1 -  0.1*x2
    return np.c_[X, Y, Z, W]

def m9affine(x0, x1, x2, x3):
    x0, x1, x2, x3 = (x0-0.5)*5, (x1-0.5)*5, (x2-0.5)*5, (x3-0.5)*5
    X = x0
    Y = x1
    Z = x2
    W = x3
    return np.c_[X, Y, Z, W]

def moebius(x0, x1):
    x0 = x0*np.pi*2
    x1 = (x1-0.5)*2
    X = (1+0.5*x1*np.cos(5*x0))*np.cos(x0)
    Y = (1+0.5*x1*np.cos(5*x0))*np.sin(x0)
    Z = 0.5*x1*np.sin(5*x0)
    return np.c_[X, Y, Z]

def scurve(x0, x1):
    x0 = (x0-0.5)*1.5*np.pi*2
    x1 = x1*2
    X = np.sin(x0)
    Y = x1
    Z = np.sign(x0)*(np.cos(x0)-1)
    W = x0*0
    return np.c_[X, Y, Z, W]

def mn_nonlinear1(x0, x1, x2, x3):
    X = np.tan(x0*np.cos(x3))
    Y = np.arctan(x3*np.sin(x0))
    Z = np.tan(x0*np.cos(x3))
    W = np.arctan(x3*np.sin(x0))
    return np.c_[X, Y, Z, W]

def mn_nonlinear2(x0, x1, x2, x3):
    X = np.tan(x0*np.cos(x3))
    Y = np.arctan(x3*np.sin(x0))
    Z = np.tan(x1*np.cos(x2))
    W = np.arctan(x1*np.sin(x2))
    return np.c_[X, Y, Z, W]

def mn_nonlinear3(x0, x1, x2, x3):
    X = np.tan(x0*np.cos(x3))
    Y = np.arctan(x3*np.sin(x0))
    W = np.tan(x1*np.cos(x2))
    Z = np.arctan(x1*np.sin(x2))
    return np.c_[X, Y, Z, W]

def mn_nonlinear_full_1d(x0):
    X = np.tan(x0*np.cos(x0))
    Y = np.arctan(x0*np.sin(x0))
    Z = np.tan(x0*np.cos(x0))
    W = np.arctan(x0*np.sin(x0))
    return np.c_[X, Y, Z, W]

def mn_nonlinear_2d(x0, x1):
    X = np.tan(x0*np.cos(x1))
    Y = np.arctan(x0*np.sin(x1))
    Z = np.tan(x1*np.cos(x0))
    W = np.arctan(x1*np.sin(x0))
    return np.c_[X, Y, Z, W]

def mn_nonlinear_3d(x0, x1, x2):
    X = np.tan(x0*np.cos(x2))
    Y = np.tan(x1*np.cos(x1))
    Z = np.tan(x2*np.cos(x0))
    W = np.arctan(x2*np.sin(x0))
    return np.c_[X, Y, Z, W]

def mn_nonlinear_4d(x0, x1, x2, x3):
    X = np.tan(x0*np.cos(x3))
    Y = np.tan(x1*np.cos(x2))
    Z = np.tan(x2*np.cos(x1))
    W = np.tan(x3*np.cos(x0))
    return np.c_[X, Y, Z, W]

def mn_nonlinear_4d_2(x0, x1, x2, x3):
    X = np.tan(x0*np.cos(x3))
    Y = np.tan(x1*np.cos(x2))
    Z = np.arctan(x2*np.cos(x1))
    W = np.arctan(x3*np.cos(x0))
    return np.c_[X, Y, Z, W]

def m1_sphere(x0, x1, x2, x3):
    r = np.sqrt(x0**2+x1**2+x2**2+x3**2)
    X = x0/r
    Y = x1/r
    Z = x2/r
    W = x3/r
    return np.c_[X, Y, Z, W]

def M10_Cubic(x0, x1, x2, x3):
    X = 0
    Y = x1
    Z = x2
    W = x3
    return np.c_[X, Y, Z, W]

import numpy as np

def M10_Cubic(x0, x1, x2, x3):
    X = 0
    Y = x1
    Z = x2
    W = x3
    return np.c_[X + 0*np.asarray(Y), Y, Z, W]

def M11_Cubic(x0, x1, x2, x3):
    X = 1
    Y = x1
    Z = x2
    W = x3
    return np.c_[X + 0*np.asarray(Y), Y, Z, W]

def M20_Cubic(x0, x1, x2, x3):
    X = x0
    Y = 0
    Z = x2
    W = x3
    return np.c_[X, Y + 0*np.asarray(X), Z, W]

def M21_Cubic(x0, x1, x2, x3):
    X = x0
    Y = 1
    Z = x2
    W = x3
    return np.c_[X, Y + 0*np.asarray(X), Z, W]

def M30_Cubic(x0, x1, x2, x3):
    X = x0
    Y = x1
    Z = 0
    W = x3
    return np.c_[X, Y, Z + 0*np.asarray(X), W]

def M31_Cubic(x0, x1, x2, x3):
    X = x0
    Y = x1
    Z = 1
    W = x3
    return np.c_[X, Y, Z + 0*np.asarray(X), W]

def M40_Cubic(x0, x1, x2, x3):
    X = x0
    Y = x1
    Z = x2
    W = 0
    return np.c_[X, Y, Z, W + 0*np.asarray(X)]

def M41_Cubic(x0, x1, x2, x3):
    X = x0
    Y = x1
    Z = x2
    W = 1
    return np.c_[X, Y, Z, W + 0*np.asarray(X)]

def tesseract_surface_map_M10_Cubic(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P)
    if P.ndim == 1:
        P = P[None, :]
    if P.shape[1] != 4:
        raise ValueError("tesseract_surface_map expects n_params=4 (columns: s, u, v, w).")
    s = P[:, 0]
    u = P[:, 1]
    v = P[:, 2]
    w = P[:, 3]

    k = np.floor(np.clip(s, 0.0, 1.0 - 1e-12) * 8).astype(np.int64)

    X = np.empty_like(s)
    Y = np.empty_like(s)
    Z = np.empty_like(s)
    W = np.empty_like(s)

    m0 = (k == 0)
    m1 = (k == 1)
    m2 = (k == 2)
    m3 = (k == 3)
    m4 = (k == 4)
    m5 = (k == 5)
    m6 = (k == 6)
    m7 = (k == 7)

    X[m0], Y[m0], Z[m0], W[m0] = 0.0, u[m0], v[m0], w[m0]
    X[m1], Y[m1], Z[m1], W[m1] = 1.0, u[m1], v[m1], w[m1]

    X[m2], Y[m2], Z[m2], W[m2] = u[m2], 0.0, v[m2], w[m2]
    X[m3], Y[m3], Z[m3], W[m3] = u[m3], 1.0, v[m3], w[m3]

    X[m4], Y[m4], Z[m4], W[m4] = u[m4], v[m4], 0.0, w[m4]
    X[m5], Y[m5], Z[m5], W[m5] = u[m5], v[m5], 1.0, w[m5]

    X[m6], Y[m6], Z[m6], W[m6] = u[m6], v[m6], w[m6], 0.0
    X[m7], Y[m7], Z[m7], W[m7] = u[m7], v[m7], w[m7], 1.0

    return np.c_[X, Y, Z, W]

def M12_Norm(P_or_x0, x1=None, x2=None, x3=None):
    from scipy.stats import norm

    if x1 is None and x2 is None and x3 is None:
        P = np.asarray(P_or_x0)
        if P.ndim == 1:
            P = P[None, :]
        if P.shape[1] != 4:
            raise ValueError("M12_Norm expects 4 parameters (columns).")
        x0, x1, x2, x3 = P.T
    else:
        x0 = np.asarray(P_or_x0)
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        x3 = np.asarray(x3)

    eps = np.finfo(float).eps
    lo = eps
    hi = 1.0 - eps

    u0 = np.clip(x0, lo, hi)
    u1 = np.clip(x1, lo, hi)
    u2 = np.clip(x2, lo, hi)
    u3 = np.clip(x3, lo, hi)

    X = norm.ppf(u0)
    Y = norm.ppf(u1)
    Z = norm.ppf(u2)
    W = norm.ppf(u3)
    return np.c_[X, Y, Z, W]

def lollipop_map(P_or_s, r=None, u=None, t=None):
    if r is None and u is None and t is None:
        P = np.asarray(P_or_s)
        if P.ndim == 1:
            P = P[None, :]
        if P.shape[1] != 4:
            raise ValueError("lollipop_map expects 4 parameters (columns: s, r, u, t).")
        s, r, u, t = P.T
    else:
        s = np.asarray(P_or_s)
        r = np.asarray(r)
        u = np.asarray(u)
        t = np.asarray(t)

    N = s.shape[0]
    X = np.empty(N, dtype=float)
    Y = np.empty(N, dtype=float)

    two_pi = 2.0 * np.pi
    T_max = 2.0 - 1.0 / np.sqrt(2.0)

    m_candy = s < 0.5
    m_stick = ~m_candy

    if np.any(m_candy):
        R = r[m_candy]
        Phi = two_pi * u[m_candy]
        rad = np.sqrt(R)
        X[m_candy] = 2.0 + rad * np.sin(Phi)
        Y[m_candy] = 2.0 + rad * np.cos(Phi)

    if np.any(m_stick):
        T = T_max * t[m_stick]
        X[m_stick] = T
        Y[m_stick] = T

    return np.c_[X, Y]

def uniform(x0, x1, x2, x3):
    X = x0
    Y = x1
    Z = x1
    W = x1
    return np.c_[X, Y, Z, W]

if __name__ == "__main__":

    visualize_unit_cube_map_plotly(uniform, n_params=4, samples=80000, axes=(0,1,2), color_dim=3,
                               title="uniform_4d", seed=3, save_html="uniform_4d.html")