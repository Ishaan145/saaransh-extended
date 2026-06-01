"""
saaransh-unified.population.synthetic_engine_adapter
====================================================
Adapts the Node.js synthetic citizen engine (paper §4.2) for use from
the Python reconstruction layer.

The original engine produces, per state, a population of synthetic
citizens, each annotated with a 41-factor demographic vector and a
policy-specific opinion through the four-layer cognitive evaluation
(economic, social, network, personal relevance). For the preliminary
results pipeline we ingest those outputs as static JSON.

When the engine has not been executed for a given policy, this module
falls back to a *demographic-only* prior derived from the per-division
factor vectors loaded from `data/state_data/`. The fallback is the
expected demographic distribution, with each cell's opinion set to a
factor-weighted aggregate of policy-relevant indicators. This is a
deliberate degradation of the full cognitive model used to keep the
pipeline runnable when only the substrate (not engine outputs) is
checked in.
"""

from __future__ import annotations
import json
import os
import glob
from typing import Iterable

# Seven demographic axes used throughout the framework (paper §3.1).
DEMOGRAPHIC_AXES = ("age", "sex", "tier", "region", "education", "income", "caste")

# Discretisations used in the joint demographic table. Coarse on purpose:
# 8064 cells (4*2*3*6*4*4*7 = 16128) is unwieldy; the reduced grid below
# yields 4 * 2 * 3 * 6 * 4 * 4 * 4 = 9216 -> we keep only cells with
# non-zero population mass.
AGE_BANDS    = ("18_25", "26_40", "41_60", "60p")
SEX_VALUES   = ("M", "F")
TIER_VALUES  = ("metro", "tier2", "rural")
REGION_VALUES = ("hindi_belt", "south", "east", "west", "northeast", "uts")
EDU_VALUES   = ("below_secondary", "secondary", "tertiary", "postgrad")
INCOME_DECILES = ("d1_3", "d4_5", "d6_8", "d9_10")
CASTE_VALUES = ("general", "obc", "sc_st", "minority")


def load_state_data(state_data_dir: str) -> dict:
    """Load every per-state JSON in `state_data_dir`.

    Files are produced by the original orchestrator; each one is a list
    of administrative divisions with a `factors` dict of 41 normalised
    indicators. Returns dict keyed by state-name.
    """
    out = {}
    for path in glob.glob(os.path.join(state_data_dir, "*.json")):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, "r", encoding="utf-8") as f:
                out[name] = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return out


def load_engine_outputs(opinion_dir: str) -> list:
    """Load any per-state engine outputs from `human_opinion_data/<state>`.

    Each output JSON has the schema produced by `run_orchestrator.js`:
    `{opinion, confidence, reasoning, breakdown, decision, metadata, state}`.
    Returns a flat list of those records.
    """
    records = []
    if not os.path.isdir(opinion_dir):
        return records
    for sub in os.listdir(opinion_dir):
        sd = os.path.join(opinion_dir, sub)
        if not os.path.isdir(sd):
            continue
        for fn in os.listdir(sd):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(sd, fn), "r", encoding="utf-8") as f:
                    rec = json.load(f)
                    rec.setdefault("state_code", sub)
                    records.append(rec)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    return records


def all_cells() -> list[tuple]:
    """Enumerate the full Cartesian product of demographic axes."""
    cells = []
    for age in AGE_BANDS:
        for sex in SEX_VALUES:
            for tier in TIER_VALUES:
                for region in REGION_VALUES:
                    for edu in EDU_VALUES:
                        for inc in INCOME_DECILES:
                            for caste in CASTE_VALUES:
                                cells.append((age, sex, tier, region, edu, inc, caste))
    return cells
