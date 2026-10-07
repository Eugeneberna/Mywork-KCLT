"""
Train and evaluate MediMultiBERT-KCLT and its ablations / transformer baselines.

Variants
  full      XLM-R + KG-GAT + cross-lingual contrastive      (MediMultiBERT-KCLT)
  no_cl     XLM-R + KG-GAT                                  (ablation)
  no_kg     XLM-R + cross-lingual contrastive               (ablation)
  backbone  encoder + classifier only                       (XLM-R baseline; with --model_name
                                                             bert-base-multilingual-cased = mBERT baseline)

Model selection uses the VALIDATION set only (early stopping on validation loss).
The TEST set is evaluated exactly once, with the selected checkpoint.

Example
  python train.py --variant full --seed 42
  python train.py --variant backbone --model_name bert-base-multilingual-cased --tag "mBERT (Multilingual)" --seed 42
"""
import argparse, os, time, copy, json
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from common import set_seed, save_json
from data import NarrativeDataset, CrossLingualBatchSampler, make_collate
from model import KCLT
from losses import supcon_loss

VARIANTS = {"full": (True, True), "no_cl": (True, False), "no_kg": (False, True), "backbone": (False, False)}
DEFAULT_TAG = {"full": "MediMultiBERT-KCLT", "no_cl": "XLM-R + KG (no CL)",
               "no_kg": "XLM-R + CL (no KG)", "backbone": "XLM-RoBERTa (Base)"}


def to_dev(b, dev):
    out = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in b.items()}
    return out


@torch.no_grad()
def evaluate(model, loader, dev, num_classes):
    model.eval(); P, Y, L, U, A = [], [], [], [], []
    loss_sum, n, t0 = 0.0, 0, time.time()
    for b in loader:
        b = to_dev(b, dev)
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            out = model(b["input_ids"], b["attention_mask"], b.get("graph"))
        logits = out["logits"].float()
        loss_sum += F.cross_entropy(logits, b["labels"], reduction="sum").item(); n += len(b["labels"])
        P.append(F.softmax(logits, -1).cpu()); Y.append(b["labels"].cpu()); L += b["language"]; U += b["uid"]
        if out["node_attention"] is not None:
            A.append((out["node_attention"].cpu(), b["graph"].batch.cpu(), len(b["labels"])))
    P = torch.cat(P).numpy(); Y = torch.cat(Y).numpy()
    return {"loss": loss_sum / n, "probs": P, "y": Y, "lang": L, "uid": U, "attn": A,
            "seconds": time.time() - t0}


def metrics(y, p, num_classes):
    yp = p.argmax(1)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yp, average="macro", zero_division=0)
    onehot = np.eye(num_classes)[y]
    return {"accuracy": accuracy_score(y, yp), "precision": pr, "recall": rc, "macro_f1": f1,
            "brier": float(np.mean(np.sum((p - onehot) ** 2, 1))), "n": int(len(y)),
            "correct": int((yp == y).sum())}


def main(a):
    set_seed(a.seed)
    use_kg, use_cl = VARIANTS[a.variant]
    tag = a.tag or DEFAULT_TAG[a.variant]
    run_dir = os.path.join(a.out, tag.replace(" ", "_").replace("/", "-"), f"seed_{a.seed}")
    os.makedirs(run_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = sorted(pd.read_csv(f"{a.data}/train.csv").label.unique())
    label2id = {l: i for i, l in enumerate(labels)}; C = len(labels)
    graphs = node_emb = None
    if use_kg:
        graphs = torch.load(f"{a.kg}/graphs.pt"); node_emb = torch.load(f"{a.kg}/node_emb.pt")
    tok = AutoTokenizer.from_pretrained(a.model_name)
    ds = {s: NarrativeDataset(f"{a.data}/{s}.csv", label2id, graphs, node_emb) for s in ("train", "val", "test")}
    col = make_collate(tok, a.max_len, use_kg)
    sampler = CrossLingualBatchSampler(ds["train"].group, a.bs, a.views, seed=a.seed)
    dl_tr = DataLoader(ds["train"], batch_sampler=sampler, collate_fn=col, num_workers=a.workers)
    dl_va = DataLoader(ds["val"], batch_size=64, collate_fn=col, num_workers=a.workers)
    dl_te = DataLoader(ds["test"], batch_size=64, collate_fn=col, num_workers=a.workers)

    model = KCLT(a.model_name, C, node_dim=(node_emb.shape[1] if use_kg else 768), use_kg=use_kg,
                 dropout=a.dropout, gat_hid=a.gat_hid, gat_heads=a.gat_heads).to(dev)
    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr},
                             {"params": head_params, "lr": a.head_lr}], weight_decay=a.wd)
    steps = len(sampler) * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(a.warmup * steps), steps)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")

    best, best_state, bad, log = float("inf"), None, 0, []
    for ep in range(1, a.epochs + 1):
        model.train(); sampler.set_epoch(ep); t0 = time.time()
        tl, tc, tn, tcl = 0.0, 0, 0, 0.0
        for b in dl_tr:
            b = to_dev(b, dev)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                out = model(b["input_ids"], b["attention_mask"], b.get("graph"))
                ce = F.cross_entropy(out["logits"].float(), b["labels"])
                key = b["groups"] if a.cl_key == "group" else b["labels"]
                cl = supcon_loss(out["z"].float(), key, a.tau) if use_cl else torch.zeros((), device=dev)
                loss = ce + a.lam * cl
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            k = len(b["labels"]); tl += ce.item() * k; tcl += cl.item() * k; tn += k
            tc += (out["logits"].argmax(-1) == b["labels"]).sum().item()
        ev = evaluate(model, dl_va, dev, C); vm = metrics(ev["y"], ev["probs"], C)
        row = {"epoch": ep, "train_ce": tl / tn, "train_contrastive": tcl / tn, "train_acc": tc / tn,
               "val_loss": ev["loss"], "val_acc": vm["accuracy"], "val_macro_f1": vm["macro_f1"],
               "epoch_seconds": time.time() - t0}
        log.append(row); print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
        if ev["loss"] < best - 1e-4:
            best, bad = ev["loss"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
        else:
            bad += 1
            if bad >= a.patience:
                print(f"early stop at epoch {ep}"); break
    pd.DataFrame(log).to_csv(f"{run_dir}/epoch_log.csv", index=False)

    # ---------------- single test evaluation with the selected checkpoint ----------------
    model.load_state_dict(best_state)
    if a.save_ckpt: torch.save(best_state, f"{run_dir}/best.pt")
    ev = evaluate(model, dl_te, dev, C); m = metrics(ev["y"], ev["probs"], C)
    per_lang = {}
    for lg in sorted(set(ev["lang"])):
        idx = [i for i, x in enumerate(ev["lang"]) if x == lg]
        per_lang[lg] = metrics(ev["y"][idx], ev["probs"][idx], C)
    rep = classification_report(ev["y"], ev["probs"].argmax(1), target_names=labels, digits=2, zero_division=0)
    n_total = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_proj = sum(p.numel() for p in model.proj.parameters())
    res = {"tag": tag, "variant": a.variant, "model_name": a.model_name, "seed": a.seed,
           "best_epoch": best_epoch, "test": m, "test_per_language": per_lang,
           "params_total_M": n_total / 1e6, "params_inference_M": (n_total - n_proj) / 1e6,
           "params_added_to_encoder_M": (n_total - n_enc - n_proj) / 1e6,
           "inference_samples_per_sec": len(ev["y"]) / ev["seconds"],
           "mean_epoch_seconds": float(np.mean([r["epoch_seconds"] for r in log])),
           "hparams": vars(a)}
    save_json(res, f"{run_dir}/results.json")
    open(f"{run_dir}/classification_report.txt", "w").write(rep)

    # predictions in the format read by stat_validation.py (preds/<Model>/seed_<s>.csv)
    pdir = os.path.join(a.preds, tag); os.makedirs(pdir, exist_ok=True)
    P = pd.DataFrame(ev["probs"], columns=[f"p_{i}" for i in range(C)])
    P.insert(0, "y_pred", ev["probs"].argmax(1)); P.insert(0, "y_true", ev["y"])
    P["uid"] = ev["uid"]; P["language"] = ev["lang"]
    P.to_csv(f"{pdir}/seed_{a.seed}.csv", index=False)

    # interpretability: top-5 attended KG concepts per test narrative
    if use_kg and ev["attn"]:
        nodes = pd.read_csv(f"{a.kg}/nodes.csv").fillna("")
        rows, k = [], 0
        for alpha, batch, nb in ev["attn"]:
            for gi in range(nb):
                uid = ev["uid"][k]; k += 1
                sel = (batch == gi).nonzero().flatten()
                if len(sel) == 0:
                    rows.append({"uid": uid, "top_concepts": ""}); continue
                g_nodes = graphs[uid]["nodes"]
                order = alpha[sel].argsort(descending=True)[:5]
                rows.append({"uid": uid, "top_concepts": " | ".join(
                    f"{nodes.name[int(g_nodes[j])]} ({float(alpha[sel][j]):.2f})" for j in order)})
        pd.DataFrame(rows).to_csv(f"{run_dir}/explanations.csv", index=False)
    print(json.dumps({"tag": tag, "seed": a.seed, **m}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default="full")
    ap.add_argument("--model_name", default="xlm-roberta-base")
    ap.add_argument("--tag"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", default="data"); ap.add_argument("--kg", default="kg")
    ap.add_argument("--out", default="runs"); ap.add_argument("--preds", default="preds")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01); ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.07); ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--cl_key", choices=["group", "label"], default="group")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--gat_hid", type=int, default=64); ap.add_argument("--gat_heads", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=128); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save_ckpt", action="store_true")
    main(ap.parse_args())
