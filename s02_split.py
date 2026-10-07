"""
Step 2 - Leakage-safe split + audit.

Splits at the level of source-narrative GROUPS (identical English texts form one group),
stratified by disease label, 70/15/15. All five language versions of a narrative, and all
duplicates of it, land in the same subset. Then audits the result.

Usage:  python s02_split.py --data data/multilingual.csv --out data --seed 42
Output: data/train.csv, data/val.csv, data/test.csv, data/split_audit.json
"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from common import save_json


def main(a):
    ml = pd.read_csv(a.data)
    g = ml[ml.language == "en"].groupby("group_id")["label"].first().reset_index()
    tr_g, tmp_g = train_test_split(g, test_size=0.30, stratify=g.label, random_state=a.seed)
    va_g, te_g = train_test_split(tmp_g, test_size=0.50, stratify=tmp_g.label, random_state=a.seed)
    parts = {"train": set(tr_g.group_id), "val": set(va_g.group_id), "test": set(te_g.group_id)}
    out = {}
    for name, gs in parts.items():
        d = ml[ml.group_id.isin(gs)].reset_index(drop=True)
        d.to_csv(f"{a.out}/{name}.csv", index=False); out[name] = d

    # ---------------- audit ----------------
    audit = {"sizes": {k: len(v) for k, v in out.items()},
             "source_groups": {k: v.group_id.nunique() for k, v in out.items()},
             "per_language": {k: v.language.value_counts().to_dict() for k, v in out.items()},
             "test_per_class": out["test"].label.value_counts().to_dict()}
    names = list(out)
    audit["shared_groups"] = {f"{x}&{y}": len(set(out[x].group_id) & set(out[y].group_id))
                              for i, x in enumerate(names) for y in names[i + 1:]}
    audit["shared_source_ids"] = {f"{x}&{y}": len(set(out[x].source_id) & set(out[y].source_id))
                                  for i, x in enumerate(names) for y in names[i + 1:]}
    norm = lambda s: set(s.str.lower().str.split().str.join(" "))
    audit["identical_texts"] = {f"{x}&{y}": len(norm(out[x].text) & norm(out[y].text))
                                for i, x in enumerate(names) for y in names[i + 1:]}
    nd = {}
    for lang in sorted(ml.language.unique()):
        tr = out["train"][out["train"].language == lang].text
        te = out["test"][out["test"].language == lang].text
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(pd.concat([tr, te]))
        m = cosine_similarity(v.transform(te), v.transform(tr)).max(axis=1)
        nd[lang] = int((m >= a.near_dup).sum())
    audit[f"near_duplicates_test_vs_train_cos>={a.near_dup}"] = nd
    save_json(audit, f"{a.out}/split_audit.json")
    for k, v in audit.items():
        print(f"{k}: {v}")
    assert all(v == 0 for v in audit["shared_groups"].values()), "LEAKAGE: shared groups"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/multilingual.csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--near_dup", type=float, default=0.9)
    main(ap.parse_args())
