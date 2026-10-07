"""Collect runs/*/seed_*/results.json into the manuscript tables (CSV).
  tables/main_results.csv        mean +/- SD over seeds (accuracy, P, R, macro-F1, Brier)  -> Table 12
  tables/per_language.csv        accuracy and macro-F1 per language                          -> new per-language table
  tables/complexity.csv          parameters, inference throughput, epoch time               -> Tables 13/17
  tables/epoch_log_KCLT_seed42.csv  training log of the proposed model                     -> Table 4
"""
import glob, json, os, shutil
import numpy as np, pandas as pd

os.makedirs("tables", exist_ok=True)
R = [json.load(open(f)) for f in glob.glob("runs/*/seed_*/results.json")]
if not R:
    raise SystemExit("no results found in runs/")
ms = lambda v: f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}" if len(v) > 1 else f"{v[0]:.3f}"

rows, lang_rows, cx = [], [], []
for tag in sorted({r["tag"] for r in R}):
    rr = [r for r in R if r["tag"] == tag]
    t = [r["test"] for r in rr]
    rows.append({"Model": tag, "Runs": len(rr), **{k: ms([x[k] for x in t]) for k in
                 ["accuracy", "precision", "recall", "macro_f1", "brier"]}})
    if "test_per_language" in rr[0]:
        for lg in rr[0]["test_per_language"]:
            lang_rows.append({"Model": tag, "Language": lg,
                              "accuracy": ms([r["test_per_language"][lg]["accuracy"] for r in rr]),
                              "macro_f1": ms([r["test_per_language"][lg]["macro_f1"] for r in rr])})
    cx.append({"Model": tag, "Params total (M)": round(rr[0].get("params_total_M", np.nan), 1),
               "Params at inference (M)": round(rr[0].get("params_inference_M", rr[0].get("params_total_M", np.nan)), 1),
               "Params added to encoder (M)": round(rr[0].get("params_added_to_encoder_M", np.nan), 2),
               "Inference (samples/s)": round(np.mean([r.get("inference_samples_per_sec", np.nan) for r in rr]), 1),
               "Epoch time (s)": round(np.mean([r.get("mean_epoch_seconds", np.nan) for r in rr]), 1)})
pd.DataFrame(rows).to_csv("tables/main_results.csv", index=False)
pd.DataFrame(lang_rows).to_csv("tables/per_language.csv", index=False)
pd.DataFrame(cx).to_csv("tables/complexity.csv", index=False)
src = "runs/MediMultiBERT-KCLT/seed_42/epoch_log.csv"
if os.path.exists(src): shutil.copy(src, "tables/epoch_log_KCLT_seed42.csv")
print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(cx).to_string(index=False))
