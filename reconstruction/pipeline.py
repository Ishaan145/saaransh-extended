"""
saaransh-unified.reconstruction.pipeline
========================================
End-to-end pipeline glue.

run_full_pipeline(
    comments,           list of {text, demographic} dicts
    policy,             dict with `template_weights` for the substrate
                        fallback path; ignored if engine outputs exist
    engine_records=None substrate / engine outputs (paper §4.2)
    model_dir=None      transformer-ensemble weight dir (paper §4.1)
)

Returns the four artefacts of paper §3.4:
    representation_audit, raw_opinion, reconstructed_citizen_opinion,
    verdict_differential.
"""

from __future__ import annotations

from ..nlp import load_ensemble
from ..population.cell_priors import (
    build_cell_priors_from_template,
    build_cell_priors_from_engine,
)
from .rgi import compute_rgi, intersectional_heatmap
from .rco import compute_rco, raw_opinion, verdict_differential, sweep_kappa


def run_full_pipeline(
    comments: list[dict],
    policy: dict,
    engine_records: list | None = None,
    model_dir: str | None = None,
    kappa: float = 20.0,
    do_sensitivity: bool = True,
) -> dict:
    # ---- §4.1 Comment Intelligence -----------------------------------
    sentiment = load_ensemble(model_dir)
    scored = []
    for c in comments:
        s = sentiment.score(c["text"])
        rec = dict(c.get("demographic", {}))
        rec["text"] = c["text"]
        rec["opinion"] = s["score"]
        rec["sent_label"] = s["label"]
        rec["sent_backend"] = s["backend"]
        scored.append(rec)

    # ---- §4.2 Synthetic Citizen Population Prior ---------------------
    if engine_records:
        cell_priors = build_cell_priors_from_engine(engine_records, fallback_policy=policy)
    else:
        cell_priors = build_cell_priors_from_template(policy)

    # ---- §4.3 Bayesian Reconstruction Layer --------------------------
    rgi  = compute_rgi(scored)
    raw  = raw_opinion(scored)
    rco  = compute_rco(scored, cell_priors, kappa=kappa)
    verd = verdict_differential(raw, rco)

    heatmaps = {
        "tier_x_education": intersectional_heatmap(scored, "tier", "education"),
        "income_x_caste":   intersectional_heatmap(scored, "income", "caste"),
    }

    out = {
        "n_comments":            len(comments),
        "sentiment_backend":     scored[0]["sent_backend"] if scored else "n/a",
        "representation_audit":  rgi,
        "raw_opinion":           raw,
        "rco":                   rco,
        "verdict_differential":  verd,
        "heatmaps":              heatmaps,
    }
    if do_sensitivity:
        out["kappa_sweep"] = sweep_kappa(scored, cell_priors)
    return out
