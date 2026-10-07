"""Does the knowledge graph carry symptom-to-disease knowledge? Uses the graph ALONE (no training):
for each narrative, the predicted disease is the disease node with most edges to the linked concepts."""
import pandas as pd, torch, numpy as np, sys
kg, data = (sys.argv[1:3] + ["kg", "data"])[:2] if len(sys.argv) > 2 else ("kg", "data")
N = pd.read_csv(f"{kg}/nodes.csv").fillna(""); dis = dict(zip(N[N.is_disease == 1].nid, N[N.is_disease == 1].label))
G = torch.load(f"{kg}/graphs.pt"); T = pd.read_csv(f"{data}/train.csv"); rows = []
for uid, lab, lg in zip(T.uid, T.label, T.language):
    g = G[uid]; nodes = g["nodes"].tolist(); cnt = {d: 0 for d in dis.values()}
    for s, d in g["edge_index"].T.tolist():
        if nodes[s] in dis and nodes[d] not in dis: cnt[dis[nodes[s]]] += 1
    top = max(cnt.values()); best = [k for k, v in cnt.items() if v == top]
    rows.append({"language": lg, "true disease linked": cnt[lab] > 0, "graph-only correct": top > 0 and len(best) == 1 and best[0] == lab})
R = pd.DataFrame(rows); out = (100 * R.groupby("language").mean()).round(1); out.loc["all"] = (100 * R.drop(columns="language").mean()).round(1)
print("KNOWLEDGE-GRAPH CHECK on training narratives (%; chance for 24 diseases = 4.2%)"); print(out.to_string())
