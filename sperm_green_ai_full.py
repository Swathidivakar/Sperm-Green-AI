#!/usr/bin/env python3
"""
================================================================================
Green AI for Sperm Transcriptomics-Based Fertility Prediction  —  full pipeline
================================================================================

Single-file, reproducible implementation of the whole study. Running it:

    python sperm_green_ai_full.py

downloads the public dataset (GEO GSE160749), runs every analysis, and writes
figures + a results.json summary + feature-panel CSVs into ./results/.

What it produces (maps to the paper):
  * Table 2  — green (logistic regression, K=20) vs heavy (MLP, K=500):
               AUROC + bootstrap CI, MCC, and carbon (mean ± SD over repeats)
  * Sec 6.2  — label-permutation test (is the signal above chance?)
  * Table 3  — panel-size sweep K in {5,10,20,50,100,250}: AUROC+CI and carbon
  * Table 4  — lightweight model comparison (LogReg / SVM / LightGBM) at K=20
  * Sec 6.6  — biomarker stability across folds + gene-symbol mapping
  * Figure 2 — AUROC vs CO2e scatter
  * Figure 3 — sparsity curve (AUROC with CI band + carbon vs K)

KEY HONESTY POINTS baked into the code:
  - Leakage control: scaling + feature selection are fit INSIDE each LOOCV fold,
    on the 23 training samples only.
  - The green pipeline is deterministic -> its AUROC has no seed variance;
    uncertainty is quantified with (a) a permutation test and (b) a bootstrap CI.
  - Carbon is NOT deterministic (power draw fluctuates); each measurement is
    repeated N_CARBON times and reported as mean ± SD. Absolute CO2e drifts
    between measurement sessions, so only within-session comparisons are valid.
  - LightGBM is unstable at n=24 (AUROC swings 0.0<->1.0); its score is reported
    only with an explicit small-sample-optimism flag.

Dataset note: GSE160749 is an Affymetrix microarray (values already log-scaled
and normalised); feature IDs are transcript-cluster IDs. The method is
platform-agnostic — it consumes any samples x features matrix.

Author: Swathi D.   License: MIT
================================================================================
"""

import os
import json
import random
import urllib.request

import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score

# Optional deps: script degrades gracefully if missing.
try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

try:
    from codecarbon import EmissionsTracker
    _HAS_CODECARBON = True
except Exception:
    _HAS_CODECARBON = False

import matplotlib
matplotlib.use("Agg")            # no display needed
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ config ----
SEED = 42
GEO_ID = "GSE160749"
GEO_URL = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE160nnn/"
           "GSE160749/soft/GSE160749_family.soft.gz")
KS = [5, 10, 20, 50, 100, 250]     # panel sizes for the sweep
K_MAIN = 20                        # green model panel size
K_HEAVY = 500                      # heavy-baseline feature set
N_CARBON = 10                      # carbon measurement repeats (mean ± SD)
N_BOOT = 1000                      # bootstrap resamples for AUROC CI
N_PERM = 50                        # label-permutation seeds
OUTDIR = "results"
CARBONDIR = os.path.join(OUTDIR, "carbon")


def set_seeds(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# =============================================================== data layer ===
def download_series(dest_dir="geo_data"):
    """Download the GEO SOFT family file over HTTPS (FTP is blocked on many nets)."""
    os.makedirs(dest_dir, exist_ok=True)
    local = os.path.join(dest_dir, f"{GEO_ID}_family.soft.gz")
    if not os.path.exists(local):
        print(f"[data] downloading {GEO_ID} over HTTPS ...")
        urllib.request.urlretrieve(GEO_URL, local)
    else:
        print("[data] using cached download.")
    return local


def load_data(local_path):
    """
    Parse SOFT -> X (samples x genes), y (1=infertile, 0=fertile), and the
    parsed GEO object (kept for platform annotation / symbol mapping).
    """
    import GEOparse
    gse = GEOparse.get_GEO(filepath=local_path)
    X = gse.pivot_samples("VALUE").T            # samples x genes
    X.index.name = "sample"
    labels = {n: (1 if "infertile" in g.metadata["characteristics_ch1"][0].lower()
                  else 0)
              for n, g in gse.gsms.items()}
    y = pd.Series(labels).reindex(X.index)
    # Preprocess: drop zero-variance / missing genes. Data already log-normalised.
    X = X.loc[:, X.var(axis=0) > 0].dropna(axis=1)
    print(f"[data] X = {X.shape[0]} samples x {X.shape[1]} genes | "
          f"fertile={int((y == 0).sum())} infertile={int((y == 1).sum())}")
    return X, y, gse


# ============================================================ model builders ==
def green_pipeline(k=K_MAIN, seed=SEED):
    """Scale -> top-k ANOVA-F genes -> L2 logistic regression (deterministic)."""
    return Pipeline([("scale", StandardScaler()),
                     ("select", SelectKBest(f_classif, k=k)),
                     ("clf", LogisticRegression(max_iter=1000, random_state=seed))])


def svm_pipeline(k=K_MAIN, seed=SEED):
    return Pipeline([("scale", StandardScaler()),
                     ("select", SelectKBest(f_classif, k=k)),
                     ("clf", SVC(probability=True, random_state=seed))])


def lgbm_pipeline(k=K_MAIN, seed=SEED):
    # min_child_samples lowered so trees can split on ~23-sample folds.
    return Pipeline([("scale", StandardScaler()),
                     ("select", SelectKBest(f_classif, k=k)),
                     ("clf", LGBMClassifier(n_estimators=100, random_state=seed,
                                            min_child_samples=3, verbose=-1))])


def heavy_pipeline(k=K_HEAVY, seed=SEED):
    return Pipeline([("scale", StandardScaler()),
                     ("select", SelectKBest(f_classif, k=k)),
                     ("clf", MLPClassifier(hidden_layer_sizes=(256, 128, 64),
                                           max_iter=2000, random_state=seed))])


# ============================================================ core LOOCV ======
def loocv_predict(X, y, build_pipeline):
    """
    Leave-one-out CV. `build_pipeline` -> fresh unfitted pipeline.
    Selection/scaling fit on the 23 training samples only (no leakage).
    Returns (y_true, y_prob) arrays of length n.
    """
    Xv, yv = np.asarray(X), np.asarray(y)
    trues, probs = [], []
    for tr, te in LeaveOneOut().split(Xv):
        pipe = build_pipeline()
        pipe.fit(Xv[tr], yv[tr])
        probs.append(pipe.predict_proba(Xv[te])[0, 1])
        trues.append(yv[te][0])
    return np.array(trues), np.array(probs)


def metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    return {"auroc": roc_auc_score(y_true, y_prob),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_pred)}


def bootstrap_auc(y_true, y_prob, n_boot=N_BOOT, ci=95, seed=SEED):
    """Bootstrap CI for AUROC by resampling the pooled held-out predictions."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    point = roc_auc_score(y_true, y_prob)
    rng = np.random.RandomState(seed)
    n, stats = len(y_true), []
    for _ in range(n_boot):
        i = rng.randint(0, n, n)
        if len(np.unique(y_true[i])) < 2:      # AUROC undefined on one class
            continue
        stats.append(roc_auc_score(y_true[i], y_prob[i]))
    lo = float(np.percentile(stats, (100 - ci) / 2))
    hi = float(np.percentile(stats, 100 - (100 - ci) / 2))
    return float(point), lo, hi


def carbon_repeat(run_fn, n=N_CARBON, name="run"):
    """Repeat a callable n times measuring carbon; return (mean_g, sd_g, list)."""
    if not _HAS_CODECARBON:
        return None, None, []
    os.makedirs(CARBONDIR, exist_ok=True)
    grams = []
    for i in range(n):
        t = EmissionsTracker(project_name=f"{name}_{i}", output_dir=CARBONDIR,
                             log_level="error", save_to_file=False)
        t.start()
        run_fn()
        grams.append(t.stop() * 1000)
    g = np.array(grams)
    return float(g.mean()), (float(g.std(ddof=1)) if len(g) > 1 else 0.0), grams


def permutation_test(X, y, build_pipeline, n_perm=N_PERM):
    """Shuffle labels, re-run LOOCV n_perm times -> (real_auroc, perm_aurocs)."""
    yv = np.asarray(y)
    t, p = loocv_predict(X, yv, build_pipeline)
    real = roc_auc_score(t, p)
    perm = []
    for s in range(n_perm):
        ys = np.random.RandomState(s).permutation(yv)
        tt, pp = loocv_predict(X, ys, build_pipeline)
        perm.append(roc_auc_score(tt, pp))
    return real, np.array(perm)


# ============================================================ biomarkers ======
def selection_stability(X, y, k=K_MAIN):
    """Fraction of LOOCV folds in which each gene is selected (ANOVA F)."""
    Xv, yv = np.asarray(X), np.asarray(y)
    counts = pd.Series(0, index=X.columns, dtype=int)
    n = 0
    for tr, te in LeaveOneOut().split(Xv):
        sel = SelectKBest(f_classif, k=k).fit(Xv[tr], yv[tr])
        counts[X.columns[sel.get_support()]] += 1
        n += 1
    return (counts[counts > 0].sort_values(ascending=False) / n * 100).to_frame("pct_folds")


def features_by_k(X, y, ks=KS, id2sym=None):
    """Nested per-K panels from one full-data F-score ranking."""
    sel = SelectKBest(f_classif, k="all").fit(np.asarray(X), np.asarray(y))
    r = (pd.DataFrame({"feature": X.columns.astype(str), "F_score": sel.scores_})
         .sort_values("F_score", ascending=False).reset_index(drop=True))
    r["rank"] = r.index + 1
    r["enters_at_K"] = r["rank"].apply(lambda x: next((K for K in sorted(ks) if x <= K), None))
    if id2sym is not None:
        r["gene_symbol"] = r["feature"].map(id2sym)
    return r[r["rank"] <= max(ks)].reset_index(drop=True)


def map_symbols(features, gse):
    """Map Affymetrix transcript-cluster IDs -> gene symbols via the GPL table."""
    gpl = list(gse.gpls.values())[0]
    ann = gpl.table.copy()
    idc = "ID" if "ID" in ann.columns else ann.columns[0]
    ann[idc] = ann[idc].astype(str)
    pref = ["gene_assignment", "Gene Symbol", "GENE_SYMBOL", "gene_symbol",
            "Symbol", "mrna_assignment", "GB_ACC"]
    col = next((c for c in pref if c in ann.columns), None) or ann.columns[-1]

    def extract(v):
        if pd.isna(v):
            return None
        s = str(v)
        return (s.split("//")[1].strip() if "//" in s and len(s.split("//")) > 1
                else s.strip())
    return dict(zip(ann[idc], ann[col].apply(extract)))


# ============================================================ figures =========
def figure2(points, path=os.path.join(OUTDIR, "figure2_auroc_vs_co2.png")):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for m in points:
        if m["gco2e"] is None:
            continue
        ax.scatter(m["gco2e"], m["auroc"], s=220, color=m["color"],
                   edgecolor="black", zorder=3)
        ax.annotate(m["name"] + ("  " + m.get("flag", "") if m.get("flag") else ""),
                    (m["gco2e"], m["auroc"]), textcoords="offset points",
                    xytext=(12, m.get("dy", 10)), fontsize=10)
    ax.axhline(0.5, ls="--", color="grey", lw=1)
    ax.set_xlabel(r"Carbon cost (g CO$_2$e per run) $\rightarrow$ greener is left")
    ax.set_ylabel(r"AUROC $\rightarrow$ higher is better")
    ax.set_title("Accuracy vs. carbon across models (GSE160749, LOOCV, K=20)", fontsize=10.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()


def figure3(sweep, path=os.path.join(OUTDIR, "figure3_sparsity_curve.png")):
    ks = sorted(sweep)
    au = [sweep[k]["auroc"] for k in ks]
    lo = [sweep[k]["ci"][0] for k in ks]
    hi = [sweep[k]["ci"][1] for k in ks]
    cg = [sweep[k]["gco2e"] for k in ks]
    cs = [sweep[k]["gco2e_sd"] for k in ks]
    x = np.arange(len(ks))
    fig, ax1 = plt.subplots(figsize=(7.8, 5.0))
    c1 = "#2e8b57"
    ax1.fill_between(x, lo, hi, color=c1, alpha=0.15)
    ax1.plot(x, au, "-o", color=c1, lw=2, markersize=8)
    ax1.axhline(0.5, ls="--", color="grey", lw=1)
    ax1.set_xlabel("Number of selected genes (K)")
    ax1.set_ylabel("AUROC", color=c1); ax1.tick_params(axis="y", labelcolor=c1)
    ax1.set_xticks(x); ax1.set_xticklabels(ks)
    ax2 = ax1.twinx(); c2 = "#c0392b"
    if all(v is not None for v in cg):
        ax2.errorbar(x, cg, yerr=cs, fmt="--s", color=c2, lw=1.6, markersize=6,
                     alpha=0.85, capsize=3)
    ax2.set_ylabel(r"CO$_2$e per run (g)", color=c2); ax2.tick_params(axis="y", labelcolor=c2)
    ax1.set_title("Performance (95% CI) and carbon vs. gene-panel size", fontsize=10.5)
    ax1.grid(True, alpha=0.25)
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()


# ============================================================ main ============
def main():
    set_seeds()
    os.makedirs(OUTDIR, exist_ok=True)
    X, y, gse = load_data(download_series())
    results = {"dataset": {"samples": int(X.shape[0]), "genes": int(X.shape[1]),
                           "fertile": int((y == 0).sum()),
                           "infertile": int((y == 1).sum())}}

    # ---- Table 2: green vs heavy (AUROC+CI, MCC, repeated carbon) ----
    print("\n[1/6] Green vs heavy ...")
    gt, gp = loocv_predict(X, y, lambda: green_pipeline())
    gm = metrics(gt, gp); gci = bootstrap_auc(gt, gp)
    g_cmean, g_csd, _ = carbon_repeat(lambda: loocv_predict(X, y, lambda: green_pipeline()),
                                      name="green")
    ht, hp = loocv_predict(X, y, lambda: heavy_pipeline())
    hm = metrics(ht, hp); hci = bootstrap_auc(ht, hp)
    h_cmean, h_csd, _ = carbon_repeat(lambda: loocv_predict(X, y, lambda: heavy_pipeline()),
                                      name="heavy")
    results["table2"] = {
        "green": {**gm, "ci": gci, "gco2e": g_cmean, "gco2e_sd": g_csd},
        "heavy": {**hm, "ci": hci, "gco2e": h_cmean, "gco2e_sd": h_csd}}
    print(f"  green AUROC {gm['auroc']:.3f} CI{tuple(round(v,3) for v in gci)} "
          f"carbon {g_cmean} ± {g_csd}")
    print(f"  heavy AUROC {hm['auroc']:.3f} CI{tuple(round(v,3) for v in hci)} "
          f"carbon {h_cmean} ± {h_csd}")

    # ---- Permutation test (green) ----
    print("[2/6] Permutation test ...")
    real, perm = permutation_test(X, y, lambda: green_pipeline())
    results["permutation"] = {"real": float(real), "perm_mean": float(perm.mean()),
                              "perm_sd": float(perm.std())}
    print(f"  real {real:.3f} vs shuffled {perm.mean():.3f} ± {perm.std():.3f}")

    # ---- Table 3: panel-size sweep (AUROC+CI + repeated carbon) ----
    print("[3/6] Panel-size sweep ...")
    sweep = {}
    for k in KS:
        t, p = loocv_predict(X, y, lambda k=k: green_pipeline(k))
        ci = bootstrap_auc(t, p)
        cm, cs, _ = carbon_repeat(lambda k=k: loocv_predict(X, y, lambda: green_pipeline(k)),
                                  name=f"K{k}")
        sweep[k] = {"auroc": roc_auc_score(t, p), "ci": ci, "gco2e": cm, "gco2e_sd": cs}
        print(f"  K={k:<4} AUROC {sweep[k]['auroc']:.3f} CI[{ci[0]:.3f},{ci[1]:.3f}] "
              f"carbon {cm} ± {cs}")
    results["table3_sweep"] = sweep

    # ---- Table 4: lightweight model comparison ----
    print("[4/6] Model comparison ...")
    comp = {}
    builders = [("LogReg", green_pipeline), ("SVM-RBF", svm_pipeline)]
    if _HAS_LGBM:
        builders.append(("LightGBM", lgbm_pipeline))
    for name, build in builders:
        t, p = loocv_predict(X, y, lambda build=build: build(K_MAIN))
        ci = bootstrap_auc(t, p)
        cm, cs, _ = carbon_repeat(
            lambda build=build: loocv_predict(X, y, lambda: build(K_MAIN)), name=name)
        comp[name] = {"auroc": roc_auc_score(t, p), "ci": ci, "gco2e": cm, "gco2e_sd": cs}
        print(f"  {name:<9} AUROC {comp[name]['auroc']:.3f} "
              f"CI[{ci[0]:.3f},{ci[1]:.3f}] carbon {cm} ± {cs}")
    results["table4_models"] = comp

    # ---- Sec 6.6: biomarker stability + panels ----
    print("[5/6] Biomarker stability & panels ...")
    id2sym = map_symbols(X.columns, gse)
    stab = selection_stability(X, y)
    stab["gene_symbol"] = stab.index.astype(str).map(id2sym)
    stab.to_csv(os.path.join(OUTDIR, "gene_selection_stability.csv"))
    panels = features_by_k(X, y, id2sym=id2sym)
    panels.to_csv(os.path.join(OUTDIR, "selected_features_by_K.csv"), index=False)
    results["stable_genes_top10"] = stab.head(10).reset_index().to_dict("records")
    print(f"  saved stability ({len(stab)} genes) and per-K panels")

    # ---- Figures ----
    print("[6/6] Figures ...")
    figure2([
        {"name": "LogReg (green)", "auroc": comp["LogReg"]["auroc"],
         "gco2e": comp["LogReg"]["gco2e"], "color": "#2e8b57"},
        {"name": "SVM-RBF (green)", "auroc": comp["SVM-RBF"]["auroc"],
         "gco2e": comp["SVM-RBF"]["gco2e"], "color": "#3a7bd5", "dy": -18},
    ] + ([{"name": "LightGBM", "auroc": comp["LightGBM"]["auroc"],
           "gco2e": comp["LightGBM"]["gco2e"], "color": "#e08e0b",
           "flag": "\u2021 overfit"}] if _HAS_LGBM else []) +
        [{"name": "Heavy MLP", "auroc": hm["auroc"], "gco2e": h_cmean, "color": "#c0392b"}])
    figure3(sweep)

    with open(os.path.join(OUTDIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. See ./{OUTDIR}/ for figures, results.json and feature CSVs.")


if __name__ == "__main__":
    main()
