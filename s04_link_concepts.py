"""
Step 4 - Link every narrative to knowledge-graph concepts and extract its subgraph.

Language-agnostic linking: every word n-gram (1-3 words) of the narrative and every KG concept
name/synonym is embedded with a multilingual UMLS-aligned encoder (SapBERT-XLMR by default).
Concepts whose best span similarity exceeds a threshold become seed nodes. The per-narrative
subgraph = seed nodes + their 1-hop KG neighbours (+ optionally the 24 disease nodes, which are
identical for every narrative and therefore carry no label information).

Linking uses only the narrative text, never its label.

Usage: python s04_link_concepts.py --data data --kg kg --out kg
Output: kg/node_emb.pt   (N x d node features, frozen GAT input)
        kg/graphs.pt     {uid: {"nodes": LongTensor, "edge_index": LongTensor[2,E], "seed_sim": FloatTensor}}
        kg/linking_stats.json
"""
import argparse, re, collections
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from common import save_json

PUNCT = re.compile(r"[\.,;:!\?\(\)\"'“”‘’\[\]{}।]")


@torch.no_grad()
def embed(texts, tok, enc, device, bs=256, max_len=32):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        h = enc(**b).last_hidden_state[:, 0]            # [CLS], as in SapBERT
        out.append(F.normalize(h.float(), dim=-1).cpu())
    return torch.cat(out) if out else torch.zeros(0, enc.config.hidden_size)


def spans(text, n_max=3):
    w = PUNCT.sub(" ", str(text)).split()
    return list({" ".join(w[i:i + n]) for n in range(1, n_max + 1) for i in range(len(w) - n + 1)})


def main(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.linker); enc = AutoModel.from_pretrained(a.linker).to(dev).eval()
    if dev.type == "cuda": enc.half()

    N = pd.read_csv(f"{a.kg}/nodes.csv").fillna("")
    E = pd.read_csv(f"{a.kg}/edges.csv")
    # node embedding = mean of normalised name + synonym embeddings
    names, owner = [], []
    for nid, nm, sy in zip(N.nid, N.name, N.synonyms):
        for s in list(dict.fromkeys([nm] + [x for x in sy.split("|") if x]))[: a.max_syn]:
            names.append(s); owner.append(nid)
    print(f"Embedding {len(names)} concept strings for {len(N)} nodes")
    Z = embed(names, tok, enc, dev)
    owner = torch.tensor(owner)
    node_emb = torch.zeros(len(N), Z.shape[1]).index_add_(0, owner, Z)
    node_emb = F.normalize(node_emb / torch.bincount(owner, minlength=len(N)).clamp(min=1)[:, None], dim=-1)
    torch.save(node_emb.half(), f"{a.out}/node_emb.pt")

    adj = collections.defaultdict(set)
    for s, d in zip(E.src, E.dst):
        adj[s].add(d); adj[d].add(s)
    disease_nodes = N[N.is_disease == 1].nid.tolist() if a.add_disease_nodes else []

    rows = pd.concat([pd.read_csv(f"{a.data}/{s}.csv") for s in ("train", "val", "test")])
    graphs, n_seeds, n_nodes, by_lang = {}, [], [], collections.defaultdict(list)
    NE = node_emb.float().to(dev)
    for uid, text, lang in tqdm(list(zip(rows.uid, rows.text, rows.language)), desc="linking"):
        sp = spans(text)
        S = embed(sp, tok, enc, dev, max_len=16).to(dev) if sp else torch.zeros(0, NE.shape[1], device=dev)
        sim = (S @ NE.T).max(0).values if len(sp) else torch.zeros(len(N), device=dev)
        top = torch.topk(sim, min(a.top_k, len(N)))
        keep = [(int(i), float(v)) for v, i in zip(top.values, top.indices) if v >= a.threshold]
        if len(keep) < a.min_seeds:   # always keep a few best concepts
            keep = [(int(i), float(v)) for v, i in zip(top.values[: a.min_seeds], top.indices[: a.min_seeds])]
        seeds = [i for i, _ in keep]
        nb = []
        for i in seeds:  # 1-hop neighbours, ranked by similarity to the narrative
            cand = sorted(adj[i], key=lambda j: -float(sim[j]))[: a.max_nb_per_seed]
            nb.extend(cand)
        sel = list(dict.fromkeys(seeds + nb + disease_nodes))[: a.max_nodes + len(disease_nodes)]
        loc = {g: k for k, g in enumerate(sel)}
        ei = [(loc[s], loc[d]) for s in sel for d in adj[s] if d in loc]
        ei = torch.tensor(ei, dtype=torch.long).T if ei else torch.zeros(2, 0, dtype=torch.long)
        seed_sim = torch.tensor([float(sim[g]) for g in sel])
        graphs[uid] = {"nodes": torch.tensor(sel, dtype=torch.long), "edge_index": ei, "seed_sim": seed_sim}
        n_seeds.append(len(seeds)); n_nodes.append(len(sel)); by_lang[lang].append(float(top.values[0]))
    torch.save(graphs, f"{a.out}/graphs.pt")
    stats = {"narratives": len(graphs), "mean_seed_concepts": float(np.mean(n_seeds)),
             "mean_subgraph_nodes": float(np.mean(n_nodes)),
             "mean_best_similarity_by_language": {k: float(np.mean(v)) for k, v in by_lang.items()},
             "threshold": a.threshold, "linker": a.linker}
    save_json(stats, f"{a.out}/linking_stats.json"); print(stats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--kg", default="kg"); ap.add_argument("--out", default="kg")
    ap.add_argument("--linker", default="cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR")
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--top_k", type=int, default=12)
    ap.add_argument("--min_seeds", type=int, default=3)
    ap.add_argument("--max_nb_per_seed", type=int, default=4)
    ap.add_argument("--max_nodes", type=int, default=48)
    ap.add_argument("--max_syn", type=int, default=12)
    ap.add_argument("--no_disease_nodes", dest="add_disease_nodes", action="store_false")
    main(ap.parse_args())
