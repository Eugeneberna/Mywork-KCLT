"""
Statistical validation for MediMultiBERT-KCLT (Stage 2, Reviewer Comment 2).

Takes the test-set predictions from every model and every seed, and produces:
  Table A  - mean +/- SD and 95% CI (t-distribution, across seeds) for each metric
  Table B  - significance tests of MediMultiBERT-KCLT against each baseline:
             paired t-test across seeds (accuracy, macro-F1), Cohen's d,
             exact McNemar test on each seed (same test samples), Holm-corrected
  Table C  - 95% bootstrap CI of the accuracy / macro-F1 difference (proposed minus baseline)

INPUT LAYOUT (one CSV per model per seed, all on the SAME fixed test set, same row order):
  preds/
    MediMultiBERT-KCLT/seed_42.csv  seed_101.csv  seed_202.csv  seed_303.csv  seed_404.csv
    XLM-RoBERTa (Base)/seed_42.csv  ...
    mBERT (Multilingual)/...
    BiLSTM + Attention/...  RCNN/...  GloVe + BiGRU/...
  Each CSV: columns  y_true, y_pred  and (optional, needed for Brier score)  p_0 ... p_23
  (class probabilities from softmax, in label-index order).

USAGE:  python stat_validation.py preds  "MediMultiBERT-KCLT"
Outputs: tableA_repeated_runs.csv, tableB_significance.csv, tableC_bootstrap_diff.csv
"""
import sys, os, glob
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

N_BOOT = 10000
RNG = np.random.default_rng(2026)


def load(root):
    data = {}
    for mdir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(mdir):
            continue
        runs = {}
        for f in sorted(glob.glob(os.path.join(mdir, "seed_*.csv"))):
            seed = os.path.basename(f)[5:-4]
            runs[seed] = pd.read_csv(f)
        if runs:
            data[os.path.basename(mdir)] = runs
    return data


def brier(df):
    pcols = [c for c in df.columns if c.startswith("p_")]
    if not pcols:
        return np.nan
    P = df[sorted(pcols, key=lambda c: int(c[2:]))].to_numpy()
    Y = np.zeros_like(P)
    Y[np.arange(len(df)), df["y_true"].to_numpy()] = 1
    return np.mean(np.sum((P - Y) ** 2, axis=1))  # multiclass Brier (sum over classes)


def metrics(df):
    yt, yp = df["y_true"], df["y_pred"]
    p, r, f, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
    return dict(Accuracy=accuracy_score(yt, yp), Precision=p, Recall=r, F1=f, Brier=brier(df))


def ci95(x):
    x = np.asarray(x, float)
    n = len(x)
    m, s = x.mean(), x.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
    return m, s, m - h, m + h


def main(root, proposed):
    data = load(root)
    assert proposed in data, f"'{proposed}' folder not found in {root}"
    seeds = sorted(data[proposed].keys(), key=int)

    # ---------- Table A: repeated-run summary ----------
    rowsA, per_run = [], {}
    for model, runs in data.items():
        missing = set(seeds) - set(runs)
        assert not missing, f"{model} missing seeds {missing}"
        M = pd.DataFrame([metrics(runs[s]) for s in seeds], index=seeds)
        per_run[model] = M
        row = {"Model": model, "Runs": len(seeds)}
        for k in M.columns:
            if M[k].isna().all():
                row[k] = "n/a"
                continue
            m, s, lo, hi = ci95(M[k])
            row[k] = f"{m:.3f} ± {s:.3f} [{lo:.3f}, {hi:.3f}]"
        rowsA.append(row)
    A = pd.DataFrame(rowsA)
    A.to_csv("tableA_repeated_runs.csv", index=False)

    # ---------- Table B: significance tests ----------
    rowsB = []
    P = per_run[proposed]
    for model in data:
        if model == proposed:
            continue
        B = per_run[model]
        r = {"Comparison": f"{proposed} vs {model}"}
        for k in ["Accuracy", "F1"]:
            d = P[k].values - B[k].values
            t, p = stats.ttest_rel(P[k].values, B[k].values)
            r[f"Δ{k} (mean)"] = round(d.mean(), 4)
            r[f"paired t p ({k})"] = p
            r[f"Cohen d ({k})"] = round(d.mean() / d.std(ddof=1), 2) if d.std(ddof=1) > 0 else np.inf
        # exact McNemar on each seed (same test samples)
        mp = []
        for s in seeds:
            a, b = data[proposed][s], data[model][s]
            assert (a["y_true"].values == b["y_true"].values).all(), f"test rows differ for seed {s}"
            ca = a["y_pred"].values == a["y_true"].values
            cb = b["y_pred"].values == b["y_true"].values
            tbl = [[np.sum(ca & cb), np.sum(ca & ~cb)], [np.sum(~ca & cb), np.sum(~ca & ~cb)]]
            mp.append(mcnemar(tbl, exact=True).pvalue)
        r["McNemar p (max over seeds)"] = max(mp)
        r["Seeds with McNemar p<0.05"] = f"{sum(p < 0.05 for p in mp)}/{len(seeds)}"
        rowsB.append(r)
    Bt = pd.DataFrame(rowsB)
    # Holm correction across the baseline comparisons
    for col in ["paired t p (Accuracy)", "paired t p (F1)", "McNemar p (max over seeds)"]:
        Bt[col + " [Holm]"] = multipletests(Bt[col], method="holm")[1]
    Bt.to_csv("tableB_significance.csv", index=False)

    # ---------- Table C: bootstrap CI of the difference ----------
    rowsC = []
    n = len(data[proposed][seeds[0]])
    for model in data:
        if model == proposed:
            continue
        dacc, df1 = [], []
        for _ in range(N_BOOT):
            s = seeds[RNG.integers(len(seeds))]          # resample a seed
            idx = RNG.integers(0, n, n)                  # resample test samples
            a = data[proposed][s].iloc[idx]
            b = data[model][s].iloc[idx]
            ma, mb = metrics(a[["y_true", "y_pred"]]), metrics(b[["y_true", "y_pred"]])
            dacc.append(ma["Accuracy"] - mb["Accuracy"])
            df1.append(ma["F1"] - mb["F1"])
        lo_a, hi_a = np.percentile(dacc, [2.5, 97.5])
        lo_f, hi_f = np.percentile(df1, [2.5, 97.5])
        rowsC.append({"Comparison": f"{proposed} vs {model}",
                      "ΔAccuracy 95% CI": f"[{lo_a:.3f}, {hi_a:.3f}]",
                      "ΔMacro-F1 95% CI": f"[{lo_f:.3f}, {hi_f:.3f}]",
                      "CI excludes 0 (Acc)": lo_a > 0, "CI excludes 0 (F1)": lo_f > 0})
    pd.DataFrame(rowsC).to_csv("tableC_bootstrap_diff.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\nTABLE A\n", A.to_string(index=False))
    print("\nTABLE B\n", Bt.to_string(index=False))
    print("\nTABLE C\n", pd.DataFrame(rowsC).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "preds",
         sys.argv[2] if len(sys.argv) > 2 else "MediMultiBERT-KCLT")
