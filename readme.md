# Saaransh Unified

End-to-end implementation of the framework described in *Whose Voice Counts?
Bridging the Digital Consultation Gap in India with Multilingual NLP and
Calibrated Synthetic Citizen Populations* (ICSC 2026 submission).

This repo merges two earlier prototypes (the Saaransh sentiment-ensemble
service and the synthetic citizen population engine) into a single integrated
research pipeline matching paper sections 4.1 -> 4.2 -> 4.3.

## Layout

```
saaransh-unified/
  nlp/                Comment Intelligence Pipeline   (paper §4.1)
    _original/        verbatim copy of the original FastAPI app (preserved)
    preprocessing.py
    ensemble_sentiment.py
    summarizer.py
  population/         Synthetic Citizen Population Engine (paper §4.2)
    cell_priors.py
    synthetic_engine_adapter.py
  reconstruction/     Bayesian Reconstruction Layer    (paper §4.3 / §3)
    rgi.py            Representation Gap Index
    rco.py            Reconstructed Citizen Opinion
    pipeline.py       end-to-end glue
  case_studies/
    case_a_finfluencer.py    SEBI 2024 finfluencer regulation
  data/               sample comment streams + demographic marginals
  results/            tables + figures produced by the runner
  scripts/
    run_preliminary.py       reproduces every table in §6
```

## Quick start

```
python3 scripts/run_preliminary.py
```

Outputs (`results/`):
- `case_a_rgi.json`           – Representation Gap Index + heatmap
- `case_a_rco.json`           – Reconstructed Citizen Opinion + 90 % CrI
- `case_a_verdict.json`       – Verdict Differential (raw vs RCO)
- `case_a_summary_table.csv`  – paper-ready row for Table in §6
- `case_a_heatmap.png`        – intersectional under-representation
- `case_a_sensitivity.png`    – κ ∈ {5,10,20,50,100} sensitivity sweep

## Notes on model weights

The original sentiment ensemble (`nlp/_original/`) requires the four
transformer checkpoints (`xlm_roberta_sentiment`, `roberta_sentiment`,
`distilbert_sentiment`, `bert_sentiment`). These are not in the repo
due to size constraints. The unified pipeline therefore ships with two
sentiment backends:

1. `TransformerEnsembleBackend` – loads the real checkpoints when the
   `.bin` / `.safetensors` files are present. Activated automatically.
2. `HeuristicSentimentBackend`  – a deterministic rule-based scorer used
   as a fallback so that the *integration* (preprocessing → sentiment →
   demographic inference → Bayesian fusion → RGI / RCO) runs end-to-end
   without GPU. All preliminary results in this snapshot are produced
   with this backend, which is clearly flagged in every output file.

The Bayesian Reconstruction Layer and the RGI / RCO computations are
fully implemented and are independent of which sentiment backend is
active.
