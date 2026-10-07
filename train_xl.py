"""
Cross-lingual transfer experiment for MediMultiBERT-KCLT.

Labels are available in ENGLISH only (zero-shot, --fewshot_k 0), or in English plus k labelled
narratives per disease in each other language (few-shot, --fewshot_k k).
  * Classification loss: labelled narratives only.
  * Contrastive loss (variants full / no_kg): each English training narrative is paired with one of
    its machine translations, which is used WITHOUT its label (unlabelled parallel text).
  * Model selection: ENGLISH validation narratives only (no target-language labels are used).
  * Test: full test set, scored once; the primary metric is accuracy on the four target languages.
Every variant sees 16 labelled English narratives per step and the same number of steps per epoch.
Translations of validation or test narratives are never used in training (grouped split).

  python train_xl.py --variant full --seed 42 --fewshot_k 0
"""
import argparse, os, time, json, random
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from common import set_seed, save_json
from data import NarrativeDataset, make_collate
from model import KCLT
from losses import supcon_loss
from train import VARIANTS, DEFAULT_TAG, to_dev, evaluate, metrics

TARGETS = ["es", "fr", "hi", "ta"]


class PairSampler(torch.utils.data.Sampler):
    """Batches of `per` units. Unit = [labelled row] or [labelled row, a translation of it]."""

    def __init__(self, df, labelled, pair, per=16, seed=0):
        self.df, self.lab, self.pair, self.per, self.seed, self.epoch = df, labelled, pair, per, seed, 0
        self.by_group = df.groupby("group_id").indices
        self.anchors = [i for i in range(len(df)) if labelled[i]]

    def set_epoch(self, e): self.epoch = e

    def __len__(self): return (len(self.anchors) + self.per - 1) // self.per

    def __iter__(self):
        rng = random.Random(self.seed * 1000 + self.epoch); lang = self.df.language.values
        units = []
        for i in self.anchors:
            u = [i]
            if self.pair:
                want_en = lang[i] != "en"      # a few-shot row is paired with its English version
                cand = [int(j) for j in self.by_group[self.df.group_id.iloc[i]]
                        if j != i and ((lang[j] == "en") == want_en)]
                if cand: u.append(rng.choice(cand))
            units.append(u)
        rng.shuffle(units)
        for k in range(0, len(units), self.per):
            yield [i for u in units[k:k + self.per] for i in u]


def fewshot_rows(df, k):
    """k labelled narratives per disease in each target language; fixed for all models and seeds."""
    if k <= 0: return set()
    rng = random.Random(42); keep = set()
    for lg in TARGETS:
        for _, g in df[df.language == lg].groupby("label"):
            idx = list(g.index); rng.shuffle(idx); keep.update(idx[:k])
    return keep


def main(a):
    set_seed(a.seed)
    use_kg, use_cl = VARIANTS[a.variant]
    tag = a.tag or DEFAULT_TAG[a.variant]
    root = a.root or f"xl_k{a.fewshot_k}"
    run_dir = os.path.join(root, "runs", tag.replace(" ", "_"), f"seed_{a.seed}"); os.makedirs(run_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = sorted(pd.read_csv(f"{a.data}/train.csv").label.unique()); label2id = {l: i for i, l in enumerate(labels)}; C = len(labels)
    graphs = node_emb = None
    if use_kg: graphs = torch.load(f"{a.kg}/graphs.pt"); node_emb = torch.load(f"{a.kg}/node_emb.pt")
    tok = AutoTokenizer.from_pretrained(a.model_name)
    ds = {s: NarrativeDataset(f"{a.data}/{s}.csv", label2id, graphs, node_emb) for s in ("train", "val", "test")}
    col = make_collate(tok, a.max_len, use_kg)
    df = ds["train"].df; few = fewshot_rows(df, a.fewshot_k)
    labelled = [(df.language.iloc[i] == "en") or (i in few) for i in range(len(df))]
    lab_t = torch.tensor(labelled)
    sampler = PairSampler(df, labelled, pair=use_cl, per=a.per, seed=a.seed)
    dl_tr = DataLoader(ds["train"], batch_sampler=sampler, collate_fn=col, num_workers=a.workers)
    val_en = [i for i in range(len(ds["val"])) if ds["val"].df.language.iloc[i] == "en"]
    dl_va = DataLoader(Subset(ds["val"], val_en), batch_size=64, collate_fn=col, num_workers=a.workers)
    dl_te = DataLoader(ds["test"], batch_size=64, collate_fn=col, num_workers=a.workers)
    uid2row = {u: i for i, u in enumerate(df.uid)}

    model = KCLT(a.model_name, C, node_dim=(node_emb.shape[1] if use_kg else 768), use_kg=use_kg, dropout=a.dropout).to(dev)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    opt = torch.optim.AdamW([{"params": list(model.encoder.parameters()), "lr": a.lr},
                             {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": a.head_lr}], weight_decay=a.wd)
    steps = len(sampler) * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(a.warmup * steps), steps)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    best, best_state, bad, log, best_epoch = float("inf"), None, 0, [], 0
    for ep in range(1, a.epochs + 1):
        model.train(); sampler.set_epoch(ep); t0 = time.time(); tl = tn = tc = tcl = nb = 0
        for b in dl_tr:
            m = lab_t[[uid2row[u] for u in b["uid"]]].to(dev)
            b = to_dev(b, dev)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                out = model(b["input_ids"], b["attention_mask"], b.get("graph"))
                ce = F.cross_entropy(out["logits"].float()[m], b["labels"][m])
                cl = supcon_loss(out["z"].float(), b["groups"], a.tau) if use_cl else torch.zeros((), device=dev)
                loss = ce + a.lam * cl
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            k = int(m.sum()); tl += ce.item() * k; tn += k; tcl += cl.item(); nb += 1
            tc += (out["logits"].argmax(-1)[m] == b["labels"][m]).sum().item()
        ev = evaluate(model, dl_va, dev, C); vm = metrics(ev["y"], ev["probs"], C)
        row = {"epoch": ep, "train_ce": tl / tn, "train_contrastive": tcl / nb, "train_acc": tc / tn,
               "val_en_loss": ev["loss"], "val_en_acc": vm["accuracy"], "epoch_seconds": time.time() - t0}
        log.append(row); print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        if ev["loss"] < best - 1e-4:
            best, bad, best_epoch = ev["loss"], 0, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience: print(f"early stop at epoch {ep}"); break
    pd.DataFrame(log).to_csv(f"{run_dir}/epoch_log.csv", index=False)

    model.load_state_dict(best_state)
    ev = evaluate(model, dl_te, dev, C); lang = np.array(ev["lang"])
    per_lang = {lg: metrics(ev["y"][lang == lg], ev["probs"][lang == lg], C) for lg in sorted(set(lang))}
    tgt = lang != "en"; m_t = metrics(ev["y"][tgt], ev["probs"][tgt], C)
    res = {"tag": tag, "variant": a.variant, "model_name": a.model_name, "seed": a.seed, "fewshot_k": a.fewshot_k,
           "labelled_train_rows": int(sum(labelled)), "best_epoch": best_epoch, "epochs_run": len(log),
           "test_target_languages": m_t, "test_per_language": per_lang, "hparams": vars(a)}
    save_json(res, f"{run_dir}/results.json")
    P = pd.DataFrame(ev["probs"], columns=[f"p_{i}" for i in range(C)])
    P.insert(0, "y_pred", ev["probs"].argmax(1)); P.insert(0, "y_true", ev["y"]); P["uid"] = ev["uid"]; P["language"] = ev["lang"]
    for sub, mask in (("preds", tgt), ("preds_all", np.ones(len(P), bool))):   # preds = target languages only
        d = os.path.join(root, sub, tag); os.makedirs(d, exist_ok=True); P[mask].to_csv(f"{d}/seed_{a.seed}.csv", index=False)
    print(json.dumps({"tag": tag, "seed": a.seed, "k": a.fewshot_k, "target_accuracy": m_t["accuracy"],
                      **{lg: round(v["accuracy"], 4) for lg, v in per_lang.items()}}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default="full")
    ap.add_argument("--model_name", default="xlm-roberta-base"); ap.add_argument("--tag")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--fewshot_k", type=int, default=0)
    ap.add_argument("--data", default="data"); ap.add_argument("--kg", default="kg"); ap.add_argument("--root")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--per", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01); ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.07); ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.1); ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    main(ap.parse_args())
