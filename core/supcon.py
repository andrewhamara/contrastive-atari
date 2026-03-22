"""Supervised Contrastive Loss for value-bin clustering.

Hypothesis: clustering hidden state representations by value bin
improves sample efficiency in model-based RL (EfficientZero/MuZero).

The SupConLoss is adapted from HobbitLong/SupContrast.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning loss.

    Adapted from: https://github.com/HobbitLong/SupContrast
    """

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """Compute SupCon loss.

        Args:
            features: [bsz, n_views, dim] — L2-normalized embeddings.
            labels: [bsz] — integer class labels (value bin indices).

        Returns:
            Scalar loss.
        """
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask for all views
        mask = mask.repeat(anchor_count, contrast_count)
        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0,
        )
        mask = mask * logits_mask

        # Log-prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Mean of log-likelihood over positives
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def discretize_values(values, num_bins=32):
    """Bin scalar values into discrete classes for SupCon labels.

    Uses quantile-based binning: sorts values and splits into equal-count
    groups. This guarantees every bin has enough samples for contrastive
    pairs, regardless of the value distribution.

    Args:
        values: [batch_size] tensor of scalar values.
        num_bins: Number of bins. Clamped to batch_size // 2 so each bin
                  has at least 2 members.

    Returns:
        [batch_size] tensor of integer bin labels.
    """
    values = values.detach()
    batch_size = values.shape[0]

    # Can't have more bins than half the batch (need >=2 per bin)
    num_bins = min(num_bins, batch_size // 2)
    if num_bins < 1:
        return torch.zeros(batch_size, dtype=torch.long, device=values.device)

    # Quantile boundaries from the actual batch distribution
    quantiles = torch.linspace(0, 1, num_bins + 1, device=values.device)
    boundaries = torch.quantile(values.float(), quantiles[1:-1])

    labels = torch.bucketize(values, boundaries)
    return labels.long()


def prepare_supcon_views(embeddings, noise_std=0.01):
    """Create two views for SupConLoss.

    Args:
        embeddings: [bsz, dim] L2-normalized embeddings.
        noise_std: Gaussian noise for second view.

    Returns:
        [bsz, 2, dim] tensor.
    """
    view1 = embeddings
    view2 = embeddings + torch.randn_like(embeddings) * noise_std
    view2 = F.normalize(view2, dim=1)
    return torch.stack([view1, view2], dim=1)


def filter_small_bins(labels, min_bin_size=2):
    """Mask out samples in bins with fewer than min_bin_size members.

    Returns boolean mask [batch_size].
    """
    unique, counts = torch.unique(labels, return_counts=True)
    valid_bins = unique[counts >= min_bin_size]
    return torch.isin(labels, valid_bins)
