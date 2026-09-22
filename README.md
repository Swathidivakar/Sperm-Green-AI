# Green AI for Sperm Transcriptomics-Based Fertility Prediction

A lightweight ("green") machine-learning pipeline that predicts fertility status
from a public human sperm transcriptomics dataset (GEO **GSE160749**, Affymetrix
microarray, n = 24: 6 fertile, 18 infertile) and reports **carbon cost as a
first-class metric** alongside predictive accuracy.

**Thesis:** on small, expensive-to-collect biomedical cohorts, sparse feature
selection with lightweight models matches or beats heavy deep networks at a
fraction of the energy — and flexible models are statistically untrustworthy at
this sample size.

## What the script does

`sperm_green_ai_full.py` runs the whole study end to end:

1. Downloads GSE160749 (HTTPS) and builds a labelled expression matrix
2. Green (logistic regression, K=20) vs heavy (MLP, K=500): AUROC + bootstrap CI,
   MCC, and carbon (mean ± SD over repeats)
3. Label-permutation test (is the signal above chance?)
4. Panel-size sweep (K = 5…250): per-K AUROC + CI and carbon
5. Lightweight model comparison (logistic regression / SVM / LightGBM)
6. Biomarker selection-stability + gene-symbol mapping
7. Figures (accuracy-vs-carbon; sparsity curve)

Outputs (figures, `results.json`, feature CSVs) are written to `./results/`.

## Setup & run

```bash
conda create -n greenai python=3.11 -y
conda activate greenai
pip install -r requirements.txt
python sperm_green_ai_full.py
```

> numpy is pinned < 2.0 on purpose — numpy 2.x breaks the pinned pandas /
> scikit-learn builds.

## Honesty / methodology notes

- **No leakage:** scaling and feature selection are fit inside each LOOCV fold.
- **Uncertainty:** the green pipeline is deterministic (no seed variance), so
  significance rests on a permutation test and precision on a bootstrap CI —
  which is wide at n = 24, as honestly reported.
- **Carbon:** not deterministic; each measurement is repeated and reported as
  mean ± SD. Absolute CO₂e drifts between sessions, so only within-session
  comparisons are valid.
- **LightGBM** is unstable at n = 24 and is reported only with a small-sample-
  optimism flag, never as a genuine result.

## Data

GEO **GSE160749** (public). The analysed expression matrix is already
log-scaled/normalised.

## License

MIT.
