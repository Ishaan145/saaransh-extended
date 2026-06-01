# Saaransh Unified — Setup and Run Guide

This guide covers three modes:

1. **Zero-config heuristic run** (works out of the box, no extra files needed)
2. **Real transformer ensemble** (adds your `.bin` / `.safetensors` weights)
3. **Real synthetic-population priors** (wires in the Node.js engine outputs)

All three modes share the same Python entry point: `scripts/run_preliminary.py`.

---

## 0. One-time install

```bash
cd saaransh_unified
python3 -m pip install --user numpy scipy matplotlib
# Only if you want the real transformer ensemble (Mode 2):
python3 -m pip install --user torch transformers sentencepiece
```

Tested on Python 3.10+. No GPU required; CPU works (transformer mode is just slower).

---

## 1. Mode A: zero-config heuristic run

This is what produced the preliminary results in the paper.

```bash
python3 scripts/run_preliminary.py --n_comments 1000 --kappa 20
```

What happens:

* synthetic SEBI-style comment stream is generated with documented participation skew
* heuristic sentiment backend scores each comment in `[-1, +1]`
* substrate-only template prior fills the synthetic-population path
* RGI, RCO, Verdict Differential, κ-sensitivity all computed

Outputs land in `results/`:

```
case_a_rgi.json          case_a_rco.json
case_a_verdict.json      case_a_heatmaps.json
case_a_kappa_sweep.json  case_a_top_undercells.md
case_a_summary_table.csv PRELIMINARY_RESULTS.md
case_a_per_axis_kl.png   case_a_kappa_sensitivity.png
case_a_heatmap_tier_education.png
case_a_heatmap_income_caste.png
```

Headline you should see (numbers vary by ±0.02 with the RNG):

```
[saaransh] sentiment backend: heuristic
[saaransh] RGI = 3.27
[saaransh] raw  mean opinion = -0.026  -> verdict = neutral
[saaransh] RCO        point  = -0.167  -> verdict = oppose
[saaransh] verdict flipped? True
```

---

## 2. Mode B: real XLM-R / RoBERTa / DistilBERT ensemble

You already have the trained weights from the original Saaransh sentiment
project (the same ones you noted are "too large for the repo"). Drop them
into the standard HuggingFace per-model layout:

```
saaransh_unified/nlp/_original/models/
├── xlm_roberta_sentiment/
│   ├── config.json                  (already in repo)
│   ├── sentencepiece.bpe.model      (already in repo)
│   ├── tokenizer.json               (already in repo)
│   ├── tokenizer_config.json        (already in repo)
│   ├── special_tokens_map.json      (already in repo)
│   └── pytorch_model.bin            ←  ADD  (or model.safetensors)
├── roberta_sentiment/
│   ├── config.json                  (already in repo)
│   ├── merges.txt                   (already in repo)
│   ├── vocab.json                   (already in repo)
│   ├── tokenizer.json               (already in repo)
│   ├── tokenizer_config.json        (already in repo)
│   ├── special_tokens_map.json      (already in repo)
│   └── pytorch_model.bin            ←  ADD
├── distilbert_sentiment/
│   ├── config.json                  (already in repo)
│   ├── vocab.txt                    (already in repo)
│   ├── tokenizer.json               (already in repo)
│   ├── tokenizer_config.json        (already in repo)
│   ├── special_tokens_map.json      (already in repo)
│   └── pytorch_model.bin            ←  ADD
└── summarizer/                      (optional, BART-Large-CNN)
    ├── config.json                  (already in repo)
    ├── generation_config.json       (already in repo)
    ├── merges.txt, vocab.json, …    (already in repo)
    └── pytorch_model.bin            ←  ADD if you want real BART
```

Then run, with **no other change** to the command line:

```bash
python3 scripts/run_preliminary.py --n_comments 1000
```

The script auto-detects `nlp/_original/models/` and the factory
`load_ensemble()` activates `TransformerEnsembleBackend` automatically.
The log line changes to:

```
[saaransh] model_dir auto-detected: …/nlp/_original/models
[saaransh] sentiment backend: transformer
```

If you keep the weights elsewhere, pass `--model_dir /path/to/your/models`.

### How auto-detection works

In `nlp/ensemble_sentiment.py`, `TransformerEnsembleBackend.is_available()`
checks that each of the three subdirs (`xlm_roberta_sentiment`,
`roberta_sentiment`, `distilbert_sentiment`) exists and contains at least
one `.bin` or `.safetensors` file. If even one is missing, the heuristic
backend is used. You can mix-and-match: e.g. put only XLM-R weights in
place to test, the heuristic still runs the other two slots.

### What about the BERT slot in `top3_models.json`?

The original FastAPI app at `nlp/_original/app.py` mentions four models
(`bert_sentiment` included). My ensemble in `nlp/ensemble_sentiment.py`
uses the three required by the paper (XLM-R, RoBERTa, DistilBERT). If you
want to add the fourth, edit the `REQUIRED` tuple in
`TransformerEnsembleBackend` and majority voting still works
(`Counter(votes).most_common(1)[0][0]`).

---

## 3. Mode C: real synthetic population engine outputs

The Node.js engine in your original repo (`server/run_orchestrator.js`)
produces per-citizen opinion files at
`server/human_opinion_data/<state>/<run>.json`. Each file has the schema
documented in `population/synthetic_engine_adapter.py:load_engine_outputs`:

```json
{
  "opinion": 0.59,
  "confidence": 0.63,
  "decision": "SUPPORT",
  "breakdown": {"economicScore": 1, "socialScore": 0.4, …},
  "metadata": {"policy": "Farmer Loan Waiver", "domain": "agriculture",
               "timestamp": "..."},
  "state": [...]
}
```

To wire this into Saaransh-unified:

```bash
# 3.1 first, generate a population for the policy
cd /path/to/your/saaransh-extended/server
node run_orchestrator.js 1000 "SEBI finfluencer regulation"

# this writes files under server/human_opinion_data/<state>/

# 3.2 then point the Python pipeline at that directory
cd /path/to/saaransh_unified
python3 scripts/run_preliminary.py \
    --n_comments 1000 \
    --engine_dir /path/to/your/saaransh-extended/server/human_opinion_data
```

You should see a log line:

```
[saaransh] loaded N engine records from .../human_opinion_data
```

When `engine_dir` is set, `reconstruction.pipeline.run_full_pipeline`
swaps the cell-prior construction from `build_cell_priors_from_template`
(substrate-only, used for the paper pilot) to
`build_cell_priors_from_engine` (uses the four-layer cognitive outputs).
Cells that the engine did not populate fall back to the template prior,
so partial coverage is fine.

### Demographic key mapping (important!)

The current engine output schema does not include a `demographic` dict
that uses the same seven-axis keys (age, sex, tier, region, education,
income, caste) that my `cell_for_demographic()` expects. Two paths:

1. **Extend the engine to emit demographics.** Modify
   `server/run_orchestrator.js` (or whichever step writes the JSON) to
   include a `demographic: {age, sex, tier, region, education, income,
   caste}` object per citizen, drawn from the state-data factor vector.
   This is the clean fix.

2. **Map post-hoc in Python.** Add a function in
   `population/synthetic_engine_adapter.py` that infers the demographic
   tuple from the `state` field of each record (the state-level
   substrate has the marginals, so a heuristic mapping is possible).
   Quicker, less faithful.

For the paper's pilot results we used path-of-least-resistance, that is,
the substrate-only template prior; both paths above are upgrades.

---

## Common pitfalls

* **"ModuleNotFoundError: No module named 'saaransh_unified'"**: run the
  script from the parent of the package directory (so that
  `saaransh_unified/` is on `sys.path`), or set `PYTHONPATH`. The script
  inserts the right path itself if invoked as
  `python3 saaransh_unified/scripts/run_preliminary.py`.

* **"OSError: Can't load tokenizer …"**: the `_original/models/<name>/`
  dir is missing tokenizer files. Make sure you kept the
  `tokenizer.json` / `vocab.*` / `merges.txt` files in place; only the
  weight `.bin` / `.safetensors` need to be added.

* **Out of memory on CPU**: pass `--n_comments 200` first to verify, then
  scale up. The transformer ensemble loads ~1.4 GB of weights total.

* **Different RGI / RCO numbers**: the synthetic comment generator is
  seeded (default `seed=20260601` in `case_studies/case_a_finfluencer.py`).
  Different `--n_comments` values move the numbers by ±0.02 in RCO and
  ±0.2 in RGI; the verdict flip is robust across `n in [400, 5000]`.

---

## Quick reference

```bash
# heuristic, no extras                       (paper pilot baseline)
python3 scripts/run_preliminary.py --n_comments 1000

# real ensemble, weights at the default path
python3 scripts/run_preliminary.py --n_comments 1000

# real ensemble, weights elsewhere
python3 scripts/run_preliminary.py --n_comments 1000 \
    --model_dir /abs/path/to/models

# real ensemble + real population engine
python3 scripts/run_preliminary.py --n_comments 1000 \
    --model_dir /abs/path/to/models \
    --engine_dir /abs/path/to/server/human_opinion_data

# sensitivity sweep at a single high-prior strength
python3 scripts/run_preliminary.py --n_comments 1000 --kappa 50
```
