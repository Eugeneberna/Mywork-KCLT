"""Cross-lingual supervised contrastive loss (InfoNCE / SupCon form).

Positives for anchor i: every other sample in the batch that shares its key.
  key = source group  -> translations of the same narrative (cross-lingual alignment, default)
  key = label         -> any narrative with the same diagnosis (supervised contrastive)
Negatives: all other samples in the batch. Temperature tau.
Always computed in float32 (outside autocast) for numerical stability.
"""
import torch


def supcon_loss(z, keys, tau=0.07):
    # z: (B, d) L2-normalised; keys: (B,) long
    with torch.autocast(device_type=z.device.type, enabled=False):
        z = z.float()
        B = z.shape[0]
        sim = (z @ z.T) / tau
        self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(self_mask, torch.finfo(sim.dtype).min)
        pos = (keys[:, None] == keys[None, :]) & ~self_mask
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        log_prob = log_prob.masked_fill(self_mask, 0.0)
        n_pos = pos.sum(1)
        valid = n_pos > 0
        if not valid.any():
            return z.sum() * 0.0
        loss = -(log_prob * pos).sum(1)[valid] / n_pos[valid]
        return loss.mean()
