"""
saaransh-unified.reconstruction.rco
===================================
Reconstructed Citizen Opinion (paper §3.3).

Per-cell Bayesian update (Eq. 3):
    E[y | d, pi] = (n_d * y_bar_d + kappa * mu_d^(s)) / (n_d + kappa)

National-level post-stratification (Eq. 2):
    RCO(pi) = sum_d w_d * E[y | d, pi]      where w_d = f_P(d)

Uncertainty is propagated by a Beta-Binomial conjugate update on the
discretised support / oppose / neutral vote shares, with a Monte Carlo
draw across cells. A 90 % credibility interval on RCO is returned.

Both inputs (`participants` and `cell_priors`) are plain Python dicts
so the layer is decoupled from the NLP and engine implementations.
"""

from __future__ import annotations
import math
import random
from collections import defaultdict
from typing import Iterable

from ..population.cell_priors import cell_for_demographic
from ..population.synthetic_engine_adapter import all_cells, DEMOGRAPHIC_AXES


def _bin_score(y: float) -> str:
    """Discretise the continuous opinion score into three classes."""
    if y > 0.15:
        return "support"
    if y < -0.15:
        return "oppose"
    return "neutral"


def _cell_empirical(participants: list[dict]) -> dict:
    """Aggregate participants by cell into (n, y_bar, support, oppose, neutral)."""
    agg = defaultdict(lambda: {"n": 0, "sum_y": 0.0, "support": 0, "oppose": 0, "neutral": 0})
    for p in participants:
        cell = cell_for_demographic(p)
        a = agg[cell]
        a["n"] += 1
        y = float(p.get("opinion", 0.0))
        a["sum_y"] += y
        a[_bin_score(y)] += 1
    for cell, a in agg.items():
        a["y_bar"] = a["sum_y"] / max(1, a["n"])
    return agg


def compute_rco(
    participants: list[dict],
    cell_priors: dict,
    kappa: float = 20.0,
    n_mc: int = 2000,
    cri: float = 0.90,
    seed: int = 1729,
) -> dict:
    """Return RCO point estimate, posterior interval and decomposition.

    Parameters
    ----------
    participants : list of dicts, each carrying at least an `opinion`
        score in [-1, +1] and the seven demographic axes.
    cell_priors : dict produced by `population.cell_priors.build_*`.
    kappa : effective prior count. Sensitivity is exposed via
        `sweep_kappa()`.
    n_mc : Monte Carlo draws across cells for the credibility interval.
    cri  : posterior interval level (e.g. 0.90).
    """
    rng = random.Random(seed)
    emp = _cell_empirical(participants)

    # Point estimate via Eq. 3 + Eq. 2.
    rco_point = 0.0
    total_w = 0.0
    cell_rows = []
    for cell, info in cell_priors.items():
        w = info["weight"]
        mu_s = info["prior_mean"]
        e = emp.get(cell, {"n": 0, "y_bar": 0.0})
        n_d, y_bar = e["n"], e["y_bar"]
        post = (n_d * y_bar + kappa * mu_s) / (n_d + kappa)
        rco_point += w * post
        total_w += w
        cell_rows.append({
            "cell": dict(zip(DEMOGRAPHIC_AXES, cell)),
            "n_d": n_d, "y_bar": y_bar, "mu_prior": mu_s,
            "posterior": post, "weight": w,
        })
    if total_w > 0:
        rco_point /= total_w

    # Beta-Binomial posterior on support share per cell, drawn jointly.
    # The prior on the support proportion in each cell is centred on the
    # mapped prior_mean: p_s = (mu_s + 1) / 2 (transforms [-1,+1] to [0,1]).
    samples = []
    for _ in range(n_mc):
        s = 0.0
        for cell, info in cell_priors.items():
            w = info["weight"]
            mu_s = info["prior_mean"]
            e = emp.get(cell, {"n": 0, "support": 0, "oppose": 0, "neutral": 0})
            n_d = e["n"]
            sup = e.get("support", 0)

            p_prior = max(1e-3, min(1 - 1e-3, (mu_s + 1.0) / 2.0))
            # Beta(alpha, beta) with prior count = kappa.
            alpha0 = p_prior * kappa
            beta0 = (1 - p_prior) * kappa
            alpha_post = alpha0 + sup
            beta_post = beta0 + max(0, n_d - sup)
            # Sample posterior proportion.
            p_sample = rng.betavariate(alpha_post, beta_post)
            # Map back to [-1, +1] support-net score.
            score = 2.0 * p_sample - 1.0
            s += w * score
        if total_w > 0:
            s /= total_w
        samples.append(s)
    samples.sort()
    lo = samples[int((1 - cri) / 2 * n_mc)]
    hi = samples[int((1 + cri) / 2 * n_mc) - 1]

    return {
        "rco_point": float(rco_point),
        "credibility_interval": [float(lo), float(hi)],
        "credibility_level": cri,
        "kappa": kappa,
        "n_mc": n_mc,
        "n_participants": len(participants),
        "n_cells_with_data": sum(1 for c in emp if emp[c]["n"] > 0),
        "n_cells_total": len(cell_priors),
        "per_cell": cell_rows,
    }


def raw_opinion(participants: list[dict]) -> dict:
    """Headline 'naive' opinion that treats the comment stream as if it
    were a representative sample. Used for the Verdict Differential."""
    if not participants:
        return {"mean": 0.0, "n": 0, "support_share": 0.0, "oppose_share": 0.0, "neutral_share": 0.0}
    ys = [float(p.get("opinion", 0.0)) for p in participants]
    bins = [_bin_score(y) for y in ys]
    n = len(ys)
    return {
        "mean": sum(ys) / n,
        "n": n,
        "support_share": bins.count("support") / n,
        "oppose_share":  bins.count("oppose")  / n,
        "neutral_share": bins.count("neutral") / n,
    }


def verdict_differential(raw: dict, rco: dict) -> dict:
    """Compare naive comment-stream verdict against the RCO verdict."""
    def verdict(score):
        if score > 0.15:
            return "support"
        if score < -0.15:
            return "oppose"
        return "neutral"

    raw_v = verdict(raw["mean"])
    rco_v = verdict(rco["rco_point"])
    return {
        "raw_mean":      raw["mean"],
        "raw_verdict":   raw_v,
        "rco_mean":      rco["rco_point"],
        "rco_verdict":   rco_v,
        "rco_cri":       rco["credibility_interval"],
        "flipped":       raw_v != rco_v,
        "shift":         rco["rco_point"] - raw["mean"],
    }


def sweep_kappa(
    participants: list[dict],
    cell_priors: dict,
    kappas: Iterable[float] = (5, 10, 20, 50, 100),
    n_mc: int = 800,
) -> list[dict]:
    """Sensitivity analysis over the prior-strength hyperparameter."""
    rows = []
    for k in kappas:
        r = compute_rco(participants, cell_priors, kappa=float(k), n_mc=n_mc)
        rows.append({
            "kappa": float(k),
            "rco_point": r["rco_point"],
            "cri_lo": r["credibility_interval"][0],
            "cri_hi": r["credibility_interval"][1],
        })
    return rows
