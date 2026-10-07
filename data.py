"""Datasets, the cross-lingual batch sampler and the collate function (text + KG subgraph)."""
import random
import pandas as pd, torch
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Data, Batch


class NarrativeDataset(Dataset):
    def __init__(self, csv, label2id, graphs=None, node_emb=None):
        self.df = pd.read_csv(csv).reset_index(drop=True)
        self.y = self.df.label.map(label2id).tolist()
        gids = {g: i for i, g in enumerate(sorted(self.df.group_id.unique()))}
        self.group = self.df.group_id.map(gids).tolist()
        self.graphs, self.node_emb = graphs, node_emb

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        item = {"text": r.text, "label": self.y[i], "group": self.group[i], "uid": r.uid,
                "language": r.language, "idx": i}
        if self.graphs is not None:
            g = self.graphs[r.uid]
            item["graph"] = Data(x=self.node_emb[g["nodes"]].float(), edge_index=g["edge_index"],
                                 num_nodes=len(g["nodes"]))
        return item


class CrossLingualBatchSampler(Sampler):
    """Each batch holds (batch_size // views) source groups x `views` language versions each,
    so every sample has at least one translation of itself in the batch (contrastive positive).
    Every training row is visited once per epoch (a group's leftover row is paired with a
    random other version of the same group)."""

    def __init__(self, groups, batch_size=32, views=2, seed=0):
        self.by_group = {}
        for i, g in enumerate(groups):
            self.by_group.setdefault(g, []).append(i)
        self.bs, self.views, self.seed, self.epoch = batch_size, views, seed, 0

    def set_epoch(self, e):
        self.epoch = e

    def _units(self):
        rng = random.Random(self.seed + self.epoch)
        units = []
        for idx in self.by_group.values():
            idx = idx[:]; rng.shuffle(idx)
            for k in range(0, len(idx), self.views):
                u = idx[k:k + self.views]
                while len(u) < self.views and len(idx) > 1:
                    u.append(rng.choice([j for j in idx if j not in u] or idx))
                units.append(u)
        rng.shuffle(units)
        return units

    def __iter__(self):
        units, per = self._units(), max(1, self.bs // self.views)
        for k in range(0, len(units), per):
            yield [i for u in units[k:k + per] for i in u]

    def __len__(self):
        return (len(self._units()) + self.bs // self.views - 1) // (self.bs // self.views)


def make_collate(tokenizer, max_len=128, use_kg=True):
    def collate(items):
        enc = tokenizer([it["text"] for it in items], padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
        out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
               "labels": torch.tensor([it["label"] for it in items]),
               "groups": torch.tensor([it["group"] for it in items]),
               "uid": [it["uid"] for it in items], "language": [it["language"] for it in items]}
        if use_kg:
            out["graph"] = Batch.from_data_list([it["graph"] for it in items])
        return out
    return collate
