"""
Step 1 - Build the multilingual corpus.

Translates every English Symptom2Disease narrative into Spanish, French, Hindi and Tamil
(default: offline NLLB-200 model, facebook/nllb-200-distilled-600M; optional: --engine google),
keeps the source identifier on every translation, and exports a random 10% review sheet
per language for native-speaker checking.

Usage
  python s01_translate.py --input Symptom2Disease.csv --out data
  # after reviewers fill in review/review_<lang>.csv (column 'corrected_text', 'status'):
  python s01_translate.py --apply_review --out data

Output
  data/multilingual.csv  columns: uid, source_id, group_id, language, label, text
"""
import argparse, os, time
import pandas as pd
from tqdm import tqdm
from common import LANGS, text_group_id

TARGETS = [l for l in LANGS if l != "en"]


NLLB_CODES = {"es": "spa_Latn", "fr": "fra_Latn", "hi": "hin_Deva", "ta": "tam_Taml"}
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
_nllb = {}


def _load_cache(path):
    if os.path.exists(path):
        c = pd.read_csv(path).dropna()
        return dict(zip(c["src"], c["tgt"]))
    return {}


def _save_cache(cache, path):
    pd.DataFrame({"src": list(cache), "tgt": list(cache.values())}).to_csv(path, index=False)


def translate_nllb(texts, target, cache_path, bs=16, beams=4):
    """Offline neural MT with NLLB-200 (runs on the local GPU; no external API)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    cache = _load_cache(cache_path)
    todo = [t for t in dict.fromkeys(texts) if t not in cache]
    if todo:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if "model" not in _nllb:
            _nllb["tok"] = AutoTokenizer.from_pretrained(NLLB_MODEL, src_lang="eng_Latn")
            m = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
            _nllb["model"] = (m.half() if dev == "cuda" else m).to(dev).eval()
        tok, model = _nllb["tok"], _nllb["model"]
        tgt_id = tok.convert_tokens_to_ids(NLLB_CODES[target])
        for i in tqdm(range(0, len(todo), bs), desc=f"en->{target} (NLLB)"):
            batch = todo[i:i + bs]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(dev)
            with torch.no_grad():
                gen = model.generate(**enc, forced_bos_token_id=tgt_id, num_beams=beams, max_new_tokens=300)
            for src, out in zip(batch, tok.batch_decode(gen, skip_special_tokens=True)):
                cache[src] = out
            if (i // bs) % 10 == 0:
                _save_cache(cache, cache_path)
        _save_cache(cache, cache_path)
    return [cache[t] for t in texts]


def translate_google(texts, target, cache_path, sleep=0.2, retries=4):
    """Google Translate via deep-translator (often blocked from shared cloud IPs such as Colab)."""
    from deep_translator import GoogleTranslator
    cache = _load_cache(cache_path)
    tr = GoogleTranslator(source="en", target=target)
    out = []
    for t in tqdm(texts, desc=f"en->{target} (Google)"):
        if t not in cache:
            err = None
            for k in range(retries):
                try:
                    cache[t] = tr.translate(t); break
                except Exception as e:
                    err = e; time.sleep(2 ** k)
            if t not in cache:
                raise RuntimeError(f"Google translation failed ({type(err).__name__}: {err}). Use --engine nllb.")
            if len(cache) % 50 == 0:
                _save_cache(cache, cache_path)
            time.sleep(sleep)
        out.append(cache[t])
    _save_cache(cache, cache_path)
    return out


def translate_list(texts, target, cache_path, engine="nllb"):
    return (translate_nllb if engine == "nllb" else translate_google)(texts, target, cache_path)


def build(args):
    os.makedirs(args.out, exist_ok=True); os.makedirs(f"{args.out}/cache", exist_ok=True)
    df = pd.read_csv(args.input)
    df = df[["label", "text"]].reset_index(drop=True)
    df["source_id"] = df.index                      # one id per original English row
    df["group_id"] = df["text"].map(text_group_id)  # duplicates share a group
    rows = [df.assign(language="en")]
    for lang in TARGETS:
        t = translate_list(df["text"].tolist(), lang, f"{args.out}/cache/{args.engine}_{lang}.csv", args.engine)
        rows.append(df.assign(language=lang, text=t))
    ml = pd.concat(rows, ignore_index=True)
    ml["uid"] = ml["language"] + "_" + ml["source_id"].astype(str)
    ml = ml[["uid", "source_id", "group_id", "language", "label", "text"]]
    ml.to_csv(f"{args.out}/multilingual.csv", index=False)
    print(f"Wrote {len(ml)} rows ->", f"{args.out}/multilingual.csv")

    # 10% review sample per language (stratified by label), for native-speaker checking
    os.makedirs(f"{args.out}/review", exist_ok=True)
    for lang in TARGETS:
        sub = ml[ml.language == lang]
        samp = sub.groupby("label").sample(frac=args.review_frac, random_state=42)
        src = df.set_index("source_id")["text"]
        samp = samp.assign(english_source=samp["source_id"].map(src),
                           status="", corrected_text="", reviewer="")
        samp.to_csv(f"{args.out}/review/review_{lang}.csv", index=False)
        print(f"  review sheet: review_{lang}.csv ({len(samp)} rows)")


def apply_review(args):
    ml = pd.read_csv(f"{args.out}/multilingual.csv")
    log = []
    for lang in TARGETS:
        p = f"{args.out}/review/review_{lang}.csv"
        if not os.path.exists(p):
            continue
        r = pd.read_csv(p).fillna("")
        fix = r[r["corrected_text"].str.strip() != ""]
        for _, row in fix.iterrows():
            ml.loc[ml.uid == row["uid"], "text"] = row["corrected_text"]
        log.append({"language": lang, "reviewed": len(r), "corrected": len(fix)})
    ml.to_csv(f"{args.out}/multilingual.csv", index=False)
    pd.DataFrame(log).to_csv(f"{args.out}/review/review_summary.csv", index=False)
    print(pd.DataFrame(log))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="Symptom2Disease.csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--review_frac", type=float, default=0.10)
    ap.add_argument("--engine", choices=["nllb", "google"], default="nllb")
    ap.add_argument("--apply_review", action="store_true")
    a = ap.parse_args()
    apply_review(a) if a.apply_review else build(a)
