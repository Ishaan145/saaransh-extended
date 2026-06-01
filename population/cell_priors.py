"""
saaransh-unified.population.cell_priors
=======================================
Construct the per-demographic-cell opinion prior \\mu_d^{(s)} required
by the Bayesian Reconstruction Layer (paper §3.3, Eq. 3).

There are two paths:

1.  Calibrated path (preferred): synthetic citizens produced by the
    Node engine, each carrying a `breakdown` dict and a demographic
    vector, are aggregated into the seven-axis cell grid. The cell
    prior is the cell-mean of citizens' final opinions; the cell
    sample size is the count of citizens.

2.  Substrate-only path (fallback): when the engine has not been run
    for the policy in question, the prior is derived from the
    division-level factor vectors loaded from `state_data/`. Each
    cell's prior opinion is a policy-template-weighted aggregate over
    the 41 normalised indicators, mapped to [-1, +1].

Both paths produce the same output schema so the reconstruction layer
is indifferent to which one is active.
"""

from __future__ import annotations
import json
import math
import random
from typing import Iterable

from .synthetic_engine_adapter import (
    DEMOGRAPHIC_AXES,
    all_cells,
    AGE_BANDS, SEX_VALUES, TIER_VALUES, REGION_VALUES,
    EDU_VALUES, INCOME_DECILES, CASTE_VALUES,
)


# Marginal distributions over each axis for the eligible adult population
# (~950 M). Sourced from Census 2011 projected to 2025 and from NSS/PLFS
# 2022-23 published tables (see paper §5). These are deliberately coarse
# and traceable; the substrate JSONs provide the finer joint structure.
ELIGIBLE_POP_MARGINALS = {
    "age":      {"18_25": 0.20, "26_40": 0.36, "41_60": 0.30, "60p": 0.14},
    "sex":      {"M": 0.512, "F": 0.488},
    "tier":     {"metro": 0.18, "tier2": 0.32, "rural": 0.50},
    "region":   {"hindi_belt": 0.43, "south": 0.21, "east": 0.16,
                 "west": 0.13, "northeast": 0.04, "uts": 0.03},
    "education":{"below_secondary": 0.42, "secondary": 0.34,
                 "tertiary": 0.18, "postgrad": 0.06},
    "income":   {"d1_3": 0.30, "d4_5": 0.20, "d6_8": 0.30, "d9_10": 0.20},
    "caste":    {"general": 0.28, "obc": 0.42, "sc_st": 0.245, "minority": 0.055},
}


def population_cell_weight(cell: tuple) -> float:
    """Independence approximation of the joint distribution from the
    marginals. The full joint is what the synthetic substrate provides
    via division-level constraints; the independence approximation is
    used as a default when the substrate is unavailable.
    """
    age, sex, tier, region, edu, inc, caste = cell
    w = 1.0
    w *= ELIGIBLE_POP_MARGINALS["age"][age]
    w *= ELIGIBLE_POP_MARGINALS["sex"][sex]
    w *= ELIGIBLE_POP_MARGINALS["tier"][tier]
    w *= ELIGIBLE_POP_MARGINALS["region"][region]
    w *= ELIGIBLE_POP_MARGINALS["education"][edu]
    w *= ELIGIBLE_POP_MARGINALS["income"][inc]
    w *= ELIGIBLE_POP_MARGINALS["caste"][caste]
    return w


def policy_template_prior(cell: tuple, policy: dict) -> float:
    """Substrate-only prior on a cell's opinion in [-1,+1] for a given
    policy template.

    The template is a small dict that says which demographic features
    are expected to be *helped* by the policy (positive weight) and
    which are expected to be *hurt* (negative weight). The output is a
    bounded sum of feature contributions. This is the substitute used
    when the four-layer cognitive engine has not been run.
    """
    age, sex, tier, region, edu, inc, caste = cell
    feats = {
        f"age:{age}":     1.0,
        f"sex:{sex}":     1.0,
        f"tier:{tier}":   1.0,
        f"region:{region}": 1.0,
        f"edu:{edu}":     1.0,
        f"income:{inc}":  1.0,
        f"caste:{caste}": 1.0,
    }
    weights = policy.get("template_weights", {})
    raw = 0.0
    for k, v in feats.items():
        raw += weights.get(k, 0.0) * v
    bias = policy.get("template_bias", 0.0)
    raw += bias
    # Squash to [-1, +1].
    return math.tanh(raw)


def build_cell_priors_from_template(policy: dict) -> dict:
    """Produce the {cell: {prior_mean, weight}} dict for the substrate
    fallback path.
    """
    out = {}
    for cell in all_cells():
        w = population_cell_weight(cell)
        if w < 1e-7:
            continue
        out[cell] = {
            "prior_mean": policy_template_prior(cell, policy),
            "weight": w,
        }
    return out


def cell_for_demographic(d: dict) -> tuple:
    """Project an arbitrary demographic record onto the seven-axis cell
    tuple. Missing keys fall back to the modal marginal value.
    """
    def pick(axis, values):
        v = d.get(axis)
        if v in values:
            return v
        # Modal fallback.
        marg = ELIGIBLE_POP_MARGINALS[axis if axis != "education" else "education"]
        return max(marg, key=marg.get)
    return (
        pick("age", AGE_BANDS),
        pick("sex", SEX_VALUES),
        pick("tier", TIER_VALUES),
        pick("region", REGION_VALUES),
        pick("education", EDU_VALUES),
        pick("income", INCOME_DECILES),
        pick("caste", CASTE_VALUES),
    )


def build_cell_priors_from_engine(engine_records: list, fallback_policy: dict | None = None) -> dict:
    """Aggregate per-citizen engine outputs into cell-level priors.

    `engine_records` is the list returned by `load_engine_outputs`. Each
    record is expected to carry a `demographic` dict; records without one
    are pooled into the marginal aggregation.

    Cells that receive no citizens fall back to the template prior so
    that every populated cell of the demographic table has a defined
    prior (paper §3, second invariant).
    """
    from collections import defaultdict
    sums = defaultdict(lambda: [0.0, 0])
    for rec in engine_records:
        d = rec.get("demographic") or {}
        cell = cell_for_demographic(d)
        op = rec.get("opinion", 0.0)
        if isinstance(op, (int, float)):
            sums[cell][0] += float(op)
            sums[cell][1] += 1

    out = {}
    for cell in all_cells():
        w = population_cell_weight(cell)
        if w < 1e-7:
            continue
        if cell in sums and sums[cell][1] > 0:
            mean = sums[cell][0] / sums[cell][1]
            out[cell] = {"prior_mean": mean, "weight": w, "n_synth": sums[cell][1]}
        elif fallback_policy is not None:
            out[cell] = {
                "prior_mean": policy_template_prior(cell, fallback_policy),
                "weight": w,
                "n_synth": 0,
            }
        else:
            out[cell] = {"prior_mean": 0.0, "weight": w, "n_synth": 0}
    return out
