from __future__ import annotations
from typing import Any
import numpy as np

#This figures out how to arrange the per dataset subplots so that it's not too wide
def auto_grid(n: int) -> tuple[int, int]:
    cols = int(np.floor(np.sqrt(n))) or 1
    rows = int(np.ceil(n / cols))
    while rows < cols:
        cols -= 1
        rows = int(np.ceil(n / cols))
    return rows, cols

#Tries to figure out fontsize
def auto_fontsize(figsize: tuple[float, float], base: int | float | None) -> float:
    return float(base) if base is not None else max(6.0, 0.9 * min(figsize) + 2)

#This is just for different labeling of different varying params (where to cut the decimals)
def fmt_val(p: str, v: Any) -> str:
    if v is None:
        return "None"
    if p in {"sr", "t", "r"}:
        return f"{float(v):.2f}"
    if p in {"n", "k", "Nbag", "lid", "dim"}:
        return str(int(v))
    return str(v)

def emphasize_labeled_ticks(ax, axis="both", major_length=10, minor_length=5):
    """Make ticks with a visible label longer than ticks with an empty label."""
    for which in (["xaxis", "yaxis"] if axis == "both" else
                  ["xaxis"] if axis == "x" else ["yaxis"]):
        ax_obj = getattr(ax, which)
        for tick in ax_obj.get_major_ticks():
            label = tick.label1.get_text() if tick.label1 else ""
            length = major_length if label.strip() else minor_length
            tick.tick1line.set_markersize(length)
            tick.tick2line.set_markersize(length)

# is num a float or not
def isfloat(num):
    if num is not None:
        try:
            float(num)
            return True
        except ValueError:
            return False
    else:
        return False