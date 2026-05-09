"""losses — symmetric InfoNCE, hard-neg InfoNCE, phase2 combined."""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F


def _apply_pos_mask(logits: torch.Tensor, pos_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mask off-diagonal positives (false negatives) by setting their logits to -inf.
    Diagonal stays as the designated positive for cross-entropy."""
    if pos_mask is None:
        return logits
    off_diag = pos_mask.to(dtype=torch.bool)
    eye = torch.eye(off_diag.shape[0], dtype=torch.bool, device=off_diag.device)
    off_diag = off_diag & ~eye
    return logits.masked_fill(off_diag, float("-inf"))


def symmetric_infonce(
    query_emb: torch.Tensor,
    video_emb: torch.Tensor,
    temperature: float,
    pos_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Bidirectional InfoNCE: 0.5 * (L_t2v + L_v2t). Assumes normalized embeddings.

    If `pos_mask` (B, B, 1 where v_j is a positive of q_i) is given, masks off-diagonal
    positives in BOTH directions to prevent same-query / same-video false negatives."""
    logits_t2v = (query_emb @ video_emb.T) / temperature
    logits_v2t = logits_t2v.T
    if pos_mask is not None:
        logits_t2v = _apply_pos_mask(logits_t2v, pos_mask)
        # v→t mask is the transpose of (q,v) positives.
        logits_v2t = _apply_pos_mask(logits_v2t, pos_mask.T)
    labels = torch.arange(query_emb.shape[0], device=query_emb.device)
    loss_t2v = F.cross_entropy(logits_t2v, labels)
    loss_v2t = F.cross_entropy(logits_v2t, labels)
    return 0.5 * (loss_t2v + loss_v2t)


def _build_t2v_logits_with_hard_negs(
    query_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    hard_neg_emb: Optional[torch.Tensor],
    hard_neg_counts: Sequence[int],
    temperature: float,
    pos_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Logits shape (B, B + max_negs). Column 0..B-1 = in-batch positives.
    Columns B..B+max_negs = per-row hard negatives, padded with -1e4 where absent.

    If pos_mask (B, B) given, off-diagonal in-batch positives are masked to -inf."""
    inbatch = query_emb @ positive_emb.T  # (B, B)

    if hard_neg_emb is None or sum(hard_neg_counts) == 0:
        return _apply_pos_mask(inbatch, pos_mask) / temperature

    B = query_emb.shape[0]
    max_negs = max(int(c) for c in hard_neg_counts)
    extra = torch.full(
        (B, max_negs), fill_value=-1.0e4,
        dtype=inbatch.dtype, device=inbatch.device,
    )
    offset = 0
    for i, count in enumerate(hard_neg_counts):
        if count <= 0:
            continue
        seg = hard_neg_emb[offset : offset + count]
        offset += count
        extra[i, :count] = query_emb[i] @ seg.T

    inbatch = _apply_pos_mask(inbatch, pos_mask)
    return torch.cat([inbatch, extra], dim=1) / temperature


def hard_neg_infonce_t2v(
    query_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    hard_neg_emb: Optional[torch.Tensor],
    hard_neg_counts: Sequence[int],
    temperature: float,
    pos_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Text→video InfoNCE with per-row hard negatives + in-batch negatives."""
    logits = _build_t2v_logits_with_hard_negs(
        query_emb, positive_emb, hard_neg_emb, hard_neg_counts, temperature, pos_mask,
    )
    labels = torch.arange(query_emb.shape[0], device=query_emb.device)
    return F.cross_entropy(logits, labels)


def v2t_inbatch_infonce(
    query_emb: torch.Tensor,
    video_emb: torch.Tensor,
    temperature: float,
    pos_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Video→text InfoNCE, in-batch negatives only. pos_mask masked on its transpose."""
    logits = (video_emb @ query_emb.T) / temperature
    if pos_mask is not None:
        logits = _apply_pos_mask(logits, pos_mask.T)
    labels = torch.arange(video_emb.shape[0], device=video_emb.device)
    return F.cross_entropy(logits, labels)


def phase2_combined_loss(
    query_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    hard_neg_emb: Optional[torch.Tensor],
    hard_neg_counts: Sequence[int],
    temperature: float,
    alpha: float = 0.3,
    pos_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L_t2v^hard + alpha * L_v2t^in-batch.

    Primary contrastive signal is t→v with mined hard negatives; v→t keeps the
    video encoder receiving bidirectional gradient so it does not drift."""
    l_t2v = hard_neg_infonce_t2v(
        query_emb, positive_emb, hard_neg_emb, hard_neg_counts, temperature, pos_mask,
    )
    l_v2t = v2t_inbatch_infonce(query_emb, positive_emb, temperature, pos_mask)
    return l_t2v + alpha * l_v2t
