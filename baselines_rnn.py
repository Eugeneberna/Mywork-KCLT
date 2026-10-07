"""
Conventional baselines on the SAME multilingual split: BiLSTM + Attention, RCNN, BiGRU.
They use the XLM-R SentencePiece tokenizer (so all five languages are tokenised) with
randomly initialised 128-d embeddings. (English-only GloVe cannot represent Hindi/Tamil text,
so the earlier "GloVe + BiGRU" baseline is replaced by a subword BiGRU.)

python baselines_rnn.py --arch bilstm_att --seed 42
python baselines_rnn.py --arch rcnn --seed 42
python baselines_rnn.py --arch bigru --seed 42
"""
import argparse, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer
from common import set_seed, save_json
from train import metrics

TAGS = {"bilstm_att": "BiLSTM + Attention", "rcnn": "RCNN", "bigru": "BiGRU (subword)"}


class Net(nn.Module):
    def __init__(self, arch, vocab, C, emb=128, hid=128, drop=0.3):
        super().__init__()
        self.arch = arch
        self.emb = nn.Embedding(vocab, emb, padding_idx=1)
        if arch == "bilstm_att":
            self.rnn = nn.LSTM(emb, hid, batch_first=True, bidirectional=True); self.att = nn.Linear(2 * hid, 1); d = 2 * hid
        elif arch == "rcnn":
            self.rnn = nn.LSTM(emb, hid // 2, batch_first=True); self.conv = nn.Conv1d(hid // 2, 64, 5, padding=2); d = 64
        else:
            self.rnn = nn.GRU(emb, hid // 2, batch_first=True, bidirectional=True); d = hid
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(d, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, C))

    def forward(self, ids, mask):
        h, _ = self.rnn(self.emb(ids))
        m = mask.unsqueeze(-1).float()
        if self.arch == "bilstm_att":
            s = self.att(h).masked_fill(m == 0, -1e9); z = (F.softmax(s, 1) * h).sum(1)
        elif self.arch == "rcnn":
            c = F.relu(self.conv(h.transpose(1, 2))).transpose(1, 2); z = c.masked_fill(m == 0, -1e9).max(1).values
        else:
            z = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(z)


def batches(df, tok, y, bs, shuffle, rng, max_len=128):
    idx = np.arange(len(df)); rng.shuffle(idx) if shuffle else None
    for k in range(0, len(idx), bs):
        j = idx[k:k + bs]
        e = tok(df.text.iloc[j].tolist(), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        yield e["input_ids"], e["attention_mask"], torch.tensor(y[j]), j


def main(a):
    set_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    D = {s: pd.read_csv(f"{a.data}/{s}.csv") for s in ("train", "val", "test")}
    labels = sorted(D["train"].label.unique()); l2i = {l: i for i, l in enumerate(labels)}; C = len(labels)
    Y = {s: D[s].label.map(l2i).values for s in D}
    net = Net(a.arch, tok.vocab_size, C).to(dev); opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    best, bad, state, log = 1e9, 0, None, []

    def run_eval(s):
        net.eval(); P = np.zeros((len(D[s]), C)); L = 0
        with torch.no_grad():
            for ids, m, y, j in batches(D[s], tok, Y[s], 128, False, rng):
                lo = net(ids.to(dev), m.to(dev)); L += F.cross_entropy(lo, y.to(dev), reduction="sum").item()
                P[j] = F.softmax(lo, -1).cpu().numpy()
        return L / len(D[s]), P

    for ep in range(1, a.epochs + 1):
        net.train(); t0 = time.time()
        for ids, m, y, _ in batches(D["train"], tok, Y["train"], a.bs, True, rng):
            loss = F.cross_entropy(net(ids.to(dev), m.to(dev)), y.to(dev))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        vl, vp = run_eval("val"); vm = metrics(Y["val"], vp, C)
        log.append({"epoch": ep, "val_loss": vl, "val_acc": vm["accuracy"], "epoch_seconds": time.time() - t0})
        print(log[-1])
        if vl < best - 1e-4: best, bad, state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience: break
    net.load_state_dict(state); _, tp = run_eval("test"); m = metrics(Y["test"], tp, C)
    tag = TAGS[a.arch]; rd = os.path.join(a.out, tag.replace(" ", "_"), f"seed_{a.seed}"); os.makedirs(rd, exist_ok=True)
    pd.DataFrame(log).to_csv(f"{rd}/epoch_log.csv", index=False)
    save_json({"tag": tag, "seed": a.seed, "test": m,
               "params_total_M": sum(p.numel() for p in net.parameters()) / 1e6, "hparams": vars(a)}, f"{rd}/results.json")
    pdir = os.path.join(a.preds, tag); os.makedirs(pdir, exist_ok=True)
    P = pd.DataFrame(tp, columns=[f"p_{i}" for i in range(C)]); P.insert(0, "y_pred", tp.argmax(1)); P.insert(0, "y_true", Y["test"])
    P["uid"] = D["test"].uid.values; P["language"] = D["test"].language.values
    P.to_csv(f"{pdir}/seed_{a.seed}.csv", index=False); print(tag, m)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(TAGS), default="bilstm_att"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenizer", default="xlm-roberta-base"); ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs"); ap.add_argument("--preds", default="preds")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=1e-3)
    main(ap.parse_args())
