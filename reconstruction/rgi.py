"""
saaransh-unified.reconstruction.rgi
===================================
Representation Gap Index (paper §3.2, Eq. 1).

RGI(C, P) = D_KL(f_C || f_P) = sum_d f_C(d) log( f_C(d) / f_P(d) )

Computed on the joint demographic distribution, not on independent
marginals. Smoothing constant `eps` keeps the divergence finite when a
participant cell has zero population mass (which signals demographic
mis-classification rather than a true zero-population cell).

Outputs also include:
    * per-axis marginal KL (diagnostic, not the headline RGI)
    * an intersectional heatmap over selectable two-axis projections
    * the top-N most severely under-represented cells (paper §3.4
      'Representation Audit' output)
"""

from __future__ import annotations
import math
from collections import Counter, defaultdict
from typing import Iterable

from ..population.cell_priors import (
    population_cell_weight, ELIGIBLE_POP_MARGINALS,
)
from ..population.synthetic_engine_adapter import (
    DEMOGRAPHIC_AXES, all_cells,
)


def _normalise(counts: dict, total: float | None = None) -> dict:
    s = total if total is not None else sum(counts.values())
    if s <= 0:
        return {k: 0.0 for k in counts}
    return {k: v / s for k, v in counts.items()}


def _empirical_joint(participants: list[dict]) -> dict:
    """Joint density f_C over the seven-axis cell grid."""
    from ..population.cell_priors import cell_for_demographic
    counts = Counter()
    for p in participants:
        counts[cell_for_demographic(p)] += 1
    return _normalise(counts)


def _target_joint() -> dict:
    """Joint density f_P from the independence approximation of the
    marginals. The full joint from the substrate is plug-compatible.
    """
    out = {}
    s = 0.0
    for cell in all_cells():
        w = population_cell_weight(cell)
        out[cell] = w
        s += w
    return {k: v / s for k, v in out.items()}


def compute_rgi(participants: list[dict], eps: float = 1e-6) -> dict:
    """Return the headline RGI scalar plus diagnostics."""
    fC = _empirical_joint(participants)
    fP = _target_joint()

    kl = 0.0
    n_used = 0
    for cell, p in fC.items():
        if p <= 0:
            continue
        q = fP.get(cell, eps)
        if q <= 0:
            q = eps
        kl += p * math.log(p / q)
        n_used += 1

    # Per-axis marginal KLs.
    per_axis = {}
    for ax in DEMOGRAPHIC_AXES:
        marg_c = Counter()
        marg_p = ELIGIBLE_POP_MARGINALS[ax]
        for p in participants:
            v = p.get(ax)
            if v in marg_p:
                marg_c[v] += 1
        marg_c = _normalise(dict(marg_c))
        if not marg_c:
            per_axis[ax] = 0.0
            continue
        ax_kl = 0.0
        for k, pv in marg_c.items():
            if pv <= 0:
                continue
            qv = marg_p.get(k, eps) or eps
            ax_kl += pv * math.log(pv / qv)
        per_axis[ax] = ax_kl

    # Top under-represented cells: f_P(d) >> f_C(d), ordered by
    # absolute population deficit fP(d) - fC(d).
    deficit = []
    for cell, q in fP.items():
        p = fC.get(cell, 0.0)
        deficit.append((cell, q - p, p, q))
    deficit.sort(key=lambda r: r[1], reverse=True)
    top_under = [
        {"cell": dict(zip(DEMOGRAPHIC_AXES, c)), "deficit": float(d),
         "participant_share": float(pv), "population_share": float(qv)}
        for c, d, pv, qv in deficit[:10]
    ]

    return {
        "rgi": float(kl),
        "n_participants": len(participants),
        "n_active_cells": n_used,
        "per_axis_kl": per_axis,
        "top_under_represented": top_under,
    }


def intersectional_heatmap(participants: list[dict], axis_x: str, axis_y: str) -> dict:
    """Return a two-axis projection of (population_share - participant_share)
    for use as a heatmap. Positive cells are under-represented.
    """
    fC = _empirical_joint(participants)
    fP = _target_joint()
    rows = sorted({c[DEMOGRAPHIC_AXES.index(axis_y)] for c in fP})
    cols = sorted({c[DEMOGRAPHIC_AXES.index(axis_x)] for c in fP})
    grid = {r: {c: 0.0 for c in cols} for r in rows}
    pc   = {r: {c: 0.0 for c in cols} for r in rows}
    pp   = {r: {c: 0.0 for c in cols} for r in rows}
    ix = DEMOGRAPHIC_AXES.index(axis_x)
    iy = DEMOGRAPHIC_AXES.index(axis_y)
    for cell, q in fP.items():
        p = fC.get(cell, 0.0)
        r, c = cell[iy], cell[ix]
        grid[r][c] += (q - p)
        pp[r][c]   += q
        pc[r][c]   += p
    return {
        "axis_x": axis_x, "axis_y": axis_y,
        "rows": rows, "cols": cols,
        "deficit": grid,
        "participant_share": pc,
        "population_share":  pp,
    }
