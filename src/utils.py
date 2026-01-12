import numpy as np
import torch
import torch.nn.functional as F


def focal_loss_func(logits, targets, gamma=2.0, alpha=None, pos_weight=None, reduction="mean"):
    targets = targets.type_as(logits)

    # Stable BCE per element
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    # p_t = p if y=1 else (1-p)
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)

    focal_factor = (1 - p_t).pow(gamma)
    loss = focal_factor * bce

    # balancing
    if alpha is not None:
        if not torch.is_tensor(alpha):
            alpha_t = torch.tensor(alpha, device=logits.device, dtype=logits.dtype)
        else:
            alpha_t = alpha.to(device=logits.device, dtype=logits.dtype)
        alpha_factor = alpha_t * targets + (1 - alpha_t) * (1 - targets)
        loss = alpha_factor * loss

    # scales the positive terms
    if pos_weight is not None:
        pw = pos_weight.to(device=logits.device, dtype=logits.dtype).view(1, -1)
        loss = loss * (1 + (pw - 1) * targets)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def asym_loss_func(
    logits,
    targets,
    pos_weight=None,
    gamma_pos=0.0,
    gamma_neg=4.0,
    clip=0.05,
    reduction="mean",
    eps=1e-8,
):
    y = targets.float()
    p = torch.sigmoid(logits)

    # ASL clipping for negatives (only affects the negative term)
    if clip is not None and clip > 0:
        p_neg = torch.clamp(p + clip, max=1.0)
        p_clipped = torch.where(y < 0.5, p_neg, p)
    else:
        p_clipped = p

    # logs
    log_p = torch.log(p.clamp(min=eps))  # positives use original p
    log_1mp = torch.log((1.0 - p_clipped).clamp(min=eps))  # negatives use clipped p

    # focusing factors
    w_pos = (1.0 - p).clamp(min=eps).pow(gamma_pos)
    w_neg = p_clipped.clamp(min=eps).pow(gamma_neg)

    # optional positive reweighting
    if pos_weight is not None:
        pos_w = pos_weight.view(1, -1) if pos_weight.ndim == 1 else pos_weight
    else:
        pos_w = 1.0

    loss_pos = -pos_w * y * w_pos * log_p
    loss_neg = -(1.0 - y) * w_neg * log_1mp
    loss = loss_pos + loss_neg

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss
