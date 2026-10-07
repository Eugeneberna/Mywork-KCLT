"""MediMultiBERT-KCLT model: multilingual encoder + GAT over a linked KG subgraph + fusion classifier
+ projection head for cross-lingual contrastive learning."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torch_geometric.nn import GATConv
from torch_geometric.utils import softmax as pyg_softmax, scatter


class KnowledgeGAT(nn.Module):
    """Two-layer multi-head GAT over the narrative's KG subgraph, followed by text-guided
    attention pooling (the [CLS] vector is the query over node states)."""

    def __init__(self, node_dim, text_dim, hid=64, heads=4, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(node_dim, hid * heads)
        self.gat1 = GATConv(hid * heads, hid, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hid * heads, hid, heads=heads, dropout=dropout)
        self.norm1, self.norm2 = nn.LayerNorm(hid * heads), nn.LayerNorm(hid * heads)
        self.query = nn.Linear(text_dim, hid * heads)
        self.drop = nn.Dropout(dropout)
        self.out_dim = hid * heads

    def forward(self, x, edge_index, batch, h_text, num_graphs):
        h = F.elu(self.inp(x))
        h = self.norm1(h + F.elu(self.gat1(self.drop(h), edge_index)))
        h = self.norm2(h + F.elu(self.gat2(self.drop(h), edge_index)))
        q = self.query(h_text)[batch]                                  # query per node from its narrative
        score = (q * h).sum(-1) / h.shape[-1] ** 0.5
        alpha = pyg_softmax(score, batch, num_nodes=h.shape[0])        # attention over each graph's nodes
        pooled = scatter(alpha[:, None] * h, batch, dim=0, dim_size=num_graphs, reduce="sum")
        return pooled, alpha


class KCLT(nn.Module):
    def __init__(self, model_name, num_classes, node_dim=768, use_kg=True, dropout=0.1,
                 gat_hid=64, gat_heads=4, proj_dim=128):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        H = self.encoder.config.hidden_size
        self.use_kg = use_kg
        self.kg = KnowledgeGAT(node_dim, H, gat_hid, gat_heads, dropout) if use_kg else None
        fused = H + (self.kg.out_dim if use_kg else 0)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(fused, 512), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(512, num_classes))
        # projection head: used only for the contrastive loss during training
        self.proj = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Linear(H, proj_dim))

    def forward(self, input_ids, attention_mask, graph=None):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        alpha = None
        if self.use_kg:
            g, alpha = self.kg(graph.x, graph.edge_index, graph.batch, h, num_graphs=h.shape[0])
            h_fused = torch.cat([h, g], dim=-1)
        else:
            h_fused = h
        return {"logits": self.classifier(h_fused), "z": F.normalize(self.proj(h), dim=-1),
                "h_cls": h, "node_attention": alpha}
