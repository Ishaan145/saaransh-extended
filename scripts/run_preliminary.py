"""
scripts/run_preliminary.py
==========================
Reproduces every preliminary result reported for Case A in the paper.

Usage:
    python3 scripts/run_preliminary.py [--n_comments 1000] [--kappa 20]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import csv

# Make the package importable when run as a bare script.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from saaransh_unified.case_studies.case_a_finfluencer import (
    POLICY, generate_comments,
)
from saaransh_unified.reconstruction.pipeline import run_full_pipeline


def _ensure_outdir(p):
    os.makedirs(p, exist_ok=True)
    return p


def _save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _plot_heatmap(hm, path, title):
    rows = hm["rows"]
    cols = hm["cols"]
    M = np.array([[hm["deficit"][r][c] for c in cols] for r in rows]) * 100.0  # percentage points
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    vmax = np.max(np.abs(M)) if M.size else 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    ax.set_xlabel(hm["axis_x"]); ax.set_ylabel(hm["axis_y"])
    ax.set_title(title, fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:+.1f}", ha="center", va="center",
                    fontsize=7, color=("white" if abs(M[i,j])>vmax*0.6 else "black"))
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("population - participant share (pp)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_kappa_sweep(rows, path):
    ks  = [r["kappa"]     for r in rows]
    pt  = [r["rco_point"] for r in rows]
    lo  = [r["cri_lo"]    for r in rows]
    hi  = [r["cri_hi"]    for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.fill_between(ks, lo, hi, alpha=0.25, color="#1F3A5F", label="90% credibility")
    ax.plot(ks, pt, "o-", color="#1F3A5F", label="RCO point")
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"prior strength $\kappa$")
    ax.set_ylabel("Reconstructed Citizen Opinion")
    ax.set_title("Sensitivity of RCO to prior strength", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_per_axis_kl(rgi, path):
    items = sorted(rgi["per_axis_kl"].items(), key=lambda kv: kv[1], reverse=True)
    axes = [k for k,_ in items]
    vals = [v for _,v in items]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    bars = ax.bar(axes, vals, color="#B85C00")
    ax.set_ylabel("marginal KL divergence")
    ax.set_title(f"Per-axis representation gap (RGI = {rgi['rgi']:.3f})", fontsize=10)
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_summary_csv(result, path):
    rgi  = result["representation_audit"]
    raw  = result["raw_opinion"]
    rco  = result["rco"]
    verd = result["verdict_differential"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric","value"])
        w.writerow(["n_comments",             result["n_comments"]])
        w.writerow(["sentiment_backend",      result["sentiment_backend"]])
        w.writerow(["RGI",                    f"{rgi['rgi']:.4f}"])
        w.writerow(["n_active_cells",         rgi["n_active_cells"]])
        w.writerow(["raw_mean_opinion",       f"{raw['mean']:+.4f}"])
        w.writerow(["raw_support_share",      f"{raw['support_share']:.3f}"])
        w.writerow(["raw_oppose_share",       f"{raw['oppose_share']:.3f}"])
        w.writerow(["raw_neutral_share",      f"{raw['neutral_share']:.3f}"])
        w.writerow(["raw_verdict",            verd["raw_verdict"]])
        w.writerow(["RCO_point",              f"{rco['rco_point']:+.4f}"])
        w.writerow(["RCO_CrI_lo",             f"{rco['credibility_interval'][0]:+.4f}"])
        w.writerow(["RCO_CrI_hi",             f"{rco['credibility_interval'][1]:+.4f}"])
        w.writerow(["RCO_verdict",            verd["rco_verdict"]])
        w.writerow(["verdict_flipped",        verd["flipped"]])
        w.writerow(["shift_rco_minus_raw",    f"{verd['shift']:+.4f}"])


def _write_under_represented_md(rgi, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Top under-represented intersectional cells (Case A)\n\n")
        f.write(f"Headline RGI: **{rgi['rgi']:.4f}** (KL divergence, joint over 7 axes)\n\n")
        f.write("| rank | cell | deficit (pp) | participant share | population share |\n")
        f.write("|------|------|--------------|-------------------|------------------|\n")
        for i, r in enumerate(rgi["top_under_represented"], 1):
            cell = ", ".join(f"{k}={v}" for k,v in r["cell"].items())
            f.write(f"| {i} | {cell} | {r['deficit']*100:+.3f} | "
                    f"{r['participant_share']*100:.4f}% | {r['population_share']*100:.4f}% |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_comments", type=int, default=1000)
    ap.add_argument("--kappa",      type=float, default=20.0)
    ap.add_argument("--outdir",     type=str,   default=None)
    ap.add_argument("--model_dir",  type=str,   default=None,
                    help="Path to dir containing xlm_roberta_sentiment/, "
                         "roberta_sentiment/, distilbert_sentiment/ subdirs "
                         "with .bin or .safetensors weights. If omitted, "
                         "tries `nlp/_original/models/`; falls back to "
                         "the heuristic backend if weights absent.")
    ap.add_argument("--engine_dir", type=str,   default=None,
                    help="Path to Node engine output dir, e.g. "
                         "`server/human_opinion_data/`. If provided, the "
                         "synthetic-prior path uses engine outputs instead "
                         "of the substrate-only template.")
    ap.add_argument("--summarizer_dir", type=str, default=None,
                    help="Path to BART-Large-CNN checkpoint dir.")
    args = ap.parse_args()

    outdir = _ensure_outdir(args.outdir or os.path.join(ROOT, "results"))

    # Auto-detect default model_dir.
    model_dir = args.model_dir
    if model_dir is None:
        default_md = os.path.join(ROOT, "nlp", "_original", "models")
        if os.path.isdir(default_md):
            model_dir = default_md
            print(f"[saaransh] model_dir auto-detected: {model_dir}")

    # Load engine outputs if pointed at one.
    engine_records = None
    if args.engine_dir:
        from saaransh_unified.population.synthetic_engine_adapter import load_engine_outputs
        engine_records = load_engine_outputs(args.engine_dir)
        print(f"[saaransh] loaded {len(engine_records)} engine records from {args.engine_dir}")

    print(f"[saaransh] outdir = {outdir}")
    print(f"[saaransh] generating {args.n_comments} synthetic comments for Case A ...")
    comments = generate_comments(args.n_comments)
    print(f"[saaransh] running full pipeline ...")
    result = run_full_pipeline(
        comments, POLICY,
        engine_records=engine_records,
        model_dir=model_dir,
        kappa=args.kappa,
    )
    print(f"[saaransh] sentiment backend: {result['sentiment_backend']}")

    rgi = result["representation_audit"]
    rco = result["rco"]
    verd = result["verdict_differential"]
    print(f"[saaransh] RGI = {rgi['rgi']:.4f}  (n_active_cells={rgi['n_active_cells']})")
    print(f"[saaransh] raw mean opinion = {result['raw_opinion']['mean']:+.4f} -> verdict={verd['raw_verdict']}")
    print(f"[saaransh] RCO point        = {rco['rco_point']:+.4f}  CrI=[{rco['credibility_interval'][0]:+.4f},{rco['credibility_interval'][1]:+.4f}]  verdict={verd['rco_verdict']}")
    print(f"[saaransh] verdict flipped? {verd['flipped']}   shift = {verd['shift']:+.4f}")

    # --- save artefacts ----------------------------------------------------
    _save_json(rgi,                    os.path.join(outdir, "case_a_rgi.json"))
    _save_json(rco,                    os.path.join(outdir, "case_a_rco.json"))
    _save_json(verd,                   os.path.join(outdir, "case_a_verdict.json"))
    _save_json(result["heatmaps"],     os.path.join(outdir, "case_a_heatmaps.json"))
    _save_json(result.get("kappa_sweep", []), os.path.join(outdir, "case_a_kappa_sweep.json"))

    _write_summary_csv(result, os.path.join(outdir, "case_a_summary_table.csv"))
    _write_under_represented_md(rgi, os.path.join(outdir, "case_a_top_undercells.md"))

    _plot_per_axis_kl(rgi, os.path.join(outdir, "case_a_per_axis_kl.png"))
    _plot_heatmap(result["heatmaps"]["tier_x_education"],
                  os.path.join(outdir, "case_a_heatmap_tier_education.png"),
                  "Under-representation: tier x education")
    _plot_heatmap(result["heatmaps"]["income_x_caste"],
                  os.path.join(outdir, "case_a_heatmap_income_caste.png"),
                  "Under-representation: income x caste")
    if result.get("kappa_sweep"):
        _plot_kappa_sweep(result["kappa_sweep"],
                          os.path.join(outdir, "case_a_kappa_sensitivity.png"))

    print("[saaransh] done. artefacts written:")
    for fn in sorted(os.listdir(outdir)):
        print(f"   - {fn}")


if __name__ == "__main__":
    main()
