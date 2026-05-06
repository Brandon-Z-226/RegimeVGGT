"""Hybrid merge-then-subsample aggregator (Line A × Line B composition).

Pipeline per Stage 2 global layer:
  1. Standard QKV + q_norm/k_norm + RoPE (same as vanilla Attention).
  2. FastVGGT-style bipartite merge on the full token set — reduces
     Q, K, V jointly from N to N_m via cluster averaging. Top-α tokens
     by Ψ are protected from being merge sources (existing Line A logic).
  3. Phase-shifted grid subsample on the merged K/V only. Each merged
     token inherits its "parent" spatial position (dst position for
     regular merges, or the unmerged-src / protected position otherwise).
     Mask selects merged K/V tokens whose inherited (r, c) satisfies
     the per-frame phase-shift rule. Q_merged is untouched.
     **Ψ-protected tokens** (same top-α as merge protection) and frame-0
     patches are unconditionally kept in K/V — i.e. the importance set
     never gets subsampled, only the rest does. This decouples the two
     lines: Ψ tokens stay full-density across both axes; everything else
     pays the merge × subsample compression jointly.
     (Toggle: protect_psi_in_subsample, default True.)
  4. Scaled dot-product attention with unequal Q (N_m) and K/V (K_sub)
     dimensions. Output is N_m tokens.
  5. Proj + unmerge (standard Line A).

Speedup intuition: Line A already reduces attention to O(N_m²); hybrid
adds O(N_m · K_sub) where K_sub ≈ N_m / sigma_sub², i.e. ~4x attention
speedup at sigma_sub=2 versus Line A alone. Wall-clock gain is bounded
by non-attention costs (MLP on N_m tokens, unchanged from Line A).

L23 still protected (Full Global, no subsample) to match RegimeVGGT's
DPT-tap convention.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from vggt.models.aggregator import slice_expand_and_flatten
from methods.fastvggt_merge import token_merge_bipartite2d


# ─────────────────────────────────────────────────────────────────
# Flash attention with LSE export (used by _avggt_attention_full)
# ─────────────────────────────────────────────────────────────────
def _chunked_attn_with_lse(q, k, v, scale, chunk_size=4096):
    """Manual softmax attention returning (output, LSE).

    Chunks along Nq so the full [Nq, Nk] score matrix never manifests.
    Used as fallback when ``torch._scaled_dot_product_flash_attention``
    is unavailable.
    """
    B, H, Nq, D = q.shape
    out = torch.empty_like(q)
    LSE = torch.empty(B, H, Nq, dtype=torch.float32, device=q.device)
    k_t = k.float().transpose(-2, -1)
    v_f = v.float()
    for s in range(0, Nq, chunk_size):
        e = min(s + chunk_size, Nq)
        q_chunk = q[:, :, s:e, :].float()
        scores = torch.matmul(q_chunk, k_t) * scale          # [B,H,c,Nk]
        m = scores.amax(dim=-1, keepdim=True)
        exp_s = torch.exp(scores - m)
        denom = exp_s.sum(dim=-1, keepdim=True)
        out_chunk = torch.matmul(exp_s / denom, v_f)
        out[:, :, s:e, :] = out_chunk.to(q.dtype)
        LSE[:, :, s:e] = (m.squeeze(-1) + torch.log(denom.squeeze(-1)))
        del scores, m, exp_s, denom, out_chunk, q_chunk
    return out, LSE


def _flash_attn_with_lse(q, k, v, scale):
    """Attention with LSE export. Tries torch's private flash kernel
    first; falls back to ``_chunked_attn_with_lse`` on missing API or
    runtime failure.
    """
    fn = getattr(torch, "_scaled_dot_product_flash_attention", None)
    if fn is not None:
        try:
            orig_dtype = q.dtype
            q_ = q if q.dtype in (torch.bfloat16, torch.float16) else q.to(torch.bfloat16)
            k_ = k if k.dtype in (torch.bfloat16, torch.float16) else k.to(torch.bfloat16)
            v_ = v if v.dtype in (torch.bfloat16, torch.float16) else v.to(torch.bfloat16)
            result = fn(q_, k_, v_,
                        dropout_p=0.0, is_causal=False,
                        return_debug_mask=False,
                        scale=scale)
            out, lse = result[0], result[1]
            if out.dtype != orig_dtype:
                out = out.to(orig_dtype)
            return out, lse
        except Exception:
            pass
    return _chunked_attn_with_lse(q, k, v, scale)


# ─────────────────────────────────────────────────────────────────
# AVGGT-style attention — mean-fill via sdpa (flash-compatible)
# ─────────────────────────────────────────────────────────────────
def _avggt_attention(q, k_all, v_all, keeper_idx, scale):
    """AVGGT subsampled global attention with mean-fill (no diagonal).

    Three-component AVGGT formulation:
      1. Selected K/V subset at ``keeper_idx``
      2. Per-query diagonal self-preservation (q_i sees k_i)   — OMITTED here
      3. Mean-fill: single mean of dropped K/V, attended by all queries
      4. Shared softmax across [selected | mean]

    The diagonal component is omitted because per-query unique keys are
    not expressible via ``F.scaled_dot_product_attention`` and a manual
    [B, H, Nq, K+2] score matrix would OOM at long-sequence scale. Mean-
    fill + shared softmax IS expressible via sdpa: append ``k_mean`` as
    one extra key, ``v_mean`` as one extra value.

    Returns out of shape [B, H, Nq, D].
    """
    B, H, Nq, D = q.shape
    _, _, Nk, _ = k_all.shape

    # Selected K/V.
    k_sel = k_all[:, :, keeper_idx, :]        # [B, H, K, D]
    v_sel = v_all[:, :, keeper_idx, :]

    # Dropped-token mean (single mean vector, per head).
    dropped_mask = torch.ones(Nk, dtype=torch.bool, device=q.device)
    dropped_mask[keeper_idx] = False
    if bool(dropped_mask.any().item()):
        k_mean = k_all[:, :, dropped_mask, :].mean(dim=-2, keepdim=True)
        v_mean = v_all[:, :, dropped_mask, :].mean(dim=-2, keepdim=True)
        k_sel_plus = torch.cat([k_sel, k_mean], dim=-2)  # [B, H, K+1, D]
        v_sel_plus = torch.cat([v_sel, v_mean], dim=-2)
        del k_mean, v_mean, k_sel, v_sel
    else:
        k_sel_plus, v_sel_plus = k_sel, v_sel

    return F.scaled_dot_product_attention(q, k_sel_plus, v_sel_plus)


# ─────────────────────────────────────────────────────────────────
# Full AVGGT — selected + diagonal + mean-fill via LSE combine
# ─────────────────────────────────────────────────────────────────
def _avggt_attention_full(q, k_all, v_all, keeper_idx, scale):
    """Full 3-component AVGGT attention:
      1. Selected K/V subset at keeper_idx
      2. Per-query diagonal self-preservation  (q_i <-> k_i)
      3. Mean-fill: single mean of dropped K/V, attended by all queries
      4. SHARED softmax across [selected | diagonal | mean]

    Implemented without manifesting the full score matrix by splitting
    into two sub-attentions and combining them via log-sum-exp.
    Requires ``Nq == Nk`` (same token layout).
    """
    B, H, Nq, D = q.shape
    _, _, Nk, _ = k_all.shape
    assert Nq == Nk, (
        f"_avggt_attention_full needs Nq == Nk, got {Nq} vs {Nk}"
    )

    # Part A: flash attention on selected K/V with LSE.
    k_sel = k_all[:, :, keeper_idx, :]        # [B, H, K, D]
    v_sel = v_all[:, :, keeper_idx, :]
    out_A, LSE_A = _flash_attn_with_lse(q, k_sel, v_sel, scale)
    del k_sel, v_sel

    # Part B: diagonal + mean-fill, manual 2-key attention per query.
    dropped_mask = torch.ones(Nk, dtype=torch.bool, device=q.device)
    dropped_mask[keeper_idx] = False
    has_dropped = bool(dropped_mask.any().item())

    # Diagonal score in fp32 to avoid bf16 underflow on small dots.
    s_diag = (q.float() * k_all.float()).sum(dim=-1) * scale   # [B, H, Nq] fp32

    if has_dropped:
        k_mean = k_all[:, :, dropped_mask, :].mean(dim=-2, keepdim=True)
        v_mean = v_all[:, :, dropped_mask, :].mean(dim=-2, keepdim=True)
        s_mean = (torch.matmul(q.float(), k_mean.float().transpose(-2, -1))
                  .squeeze(-1)) * scale

        m_B = torch.maximum(s_diag, s_mean)
        LSE_B = m_B + torch.log(
            torch.exp(s_diag - m_B) + torch.exp(s_mean - m_B)
        )
        w_diag = torch.exp(s_diag - LSE_B).unsqueeze(-1).to(q.dtype)
        w_mean = torch.exp(s_mean - LSE_B).unsqueeze(-1).to(q.dtype)
        out_B = w_diag * v_all + w_mean * v_mean
        del k_mean, v_mean, s_mean, m_B, w_diag, w_mean
    else:
        LSE_B = s_diag
        out_B = v_all
    del s_diag, dropped_mask

    # LSE-combine Part A and Part B.
    LSE_A_f = LSE_A.float()
    M = torch.maximum(LSE_A_f, LSE_B)
    w_A = torch.exp(LSE_A_f - M)
    w_B = torch.exp(LSE_B - M)
    w_A_e = w_A.unsqueeze(-1).to(out_A.dtype)
    w_B_e = w_B.unsqueeze(-1).to(out_B.dtype)
    denom = (w_A + w_B).unsqueeze(-1).to(out_A.dtype)
    out = (w_A_e * out_A + w_B_e * out_B) / denom
    return out


# ─────────────────────────────────────────────────────────────────
# Phase-shift keeper on merged tokens
# ─────────────────────────────────────────────────────────────────
def _sigma_to_tile(sigma):
    """Map sigma -> (tile_size, kept_per_dim). Density = (kept/tile)^2 ≈ 1/sigma^2.

      sigma=1     -> (1, 1)   100%  (no subsample, phase-shift no-op)
      sigma=1.3   -> (4, 3)    9/16 ~56%   (denser than 1.5)
      sigma=1.5   -> (3, 2)    4/9  ~44%
      sigma=1.7   -> (5, 3)    9/25  36%
      sigma=N int -> (N, 1)    1/N^2
    """
    s = float(sigma)
    if abs(s - 1.3) < 1e-6:
        return 4, 3
    if abs(s - 1.5) < 1e-6:
        return 3, 2
    if abs(s - 1.7) < 1e-6:
        return 5, 3
    si = int(s)
    if si < 1 or abs(s - si) > 1e-6:
        raise ValueError(
            f"sigma must be 1, 1.3, 1.5, 1.7, or integer >= 2, got {sigma}"
        )
    return si, 1


def _build_keeper_mask_on_merged(
    merged_global_idx: torch.Tensor,
    S: int,
    Hp: int,
    Wp: int,
    tokens_per_frame: int,
    num_special: int,
    sigma_sub,                    # int or float (1, 1.5, 2, 3, 4, ...)
    use_phase_shift: bool,
    anchor_frame_idx=None,
    protected_global_idx: torch.Tensor = None,   # bool[S * tokens_per_frame] or None
) -> torch.Tensor:
    """Return bool mask of shape [N_m] marking which merged tokens to keep
    as K/V under the phase-shifted grid rule.

    Each merged token has a "parent" global index g in [0, S*tokens_per_frame).
    Decode:
      frame f      = g // tokens_per_frame
      local_idx    = g %  tokens_per_frame
      if local_idx < num_special: special token → always kept.
      else patch_idx = local_idx - num_special; r = patch_idx // Wp;
                                                 c = patch_idx %  Wp.

    Generalised phase-shift rule via (tile, kept_per_dim) from _sigma_to_tile:
      phase_r = (f // tile) % tile  if use_phase_shift else 0
      phase_c = f % tile             if use_phase_shift else 0
      keep iff ((r - phase_r) mod tile) < kept_per_dim  AND
               ((c - phase_c) mod tile) < kept_per_dim
      and r, c inside truncated grid [0, (Hp // tile) * tile).
    """
    device = merged_global_idx.device
    g = merged_global_idx.long()
    frame = g // tokens_per_frame
    local = g % tokens_per_frame
    is_special = local < num_special

    patch_idx = local - num_special
    r = patch_idx // Wp
    c = patch_idx %  Wp

    tile, kept = _sigma_to_tile(sigma_sub)
    if tile == 1:
        # No subsample; every merged token kept.
        return torch.ones(g.shape[0], dtype=torch.bool, device=device)

    Hp_use = (Hp // tile) * tile
    Wp_use = (Wp // tile) * tile
    in_grid = (r < Hp_use) & (c < Wp_use)

    if use_phase_shift:
        phase_r = (frame // tile) % tile
        phase_c = frame % tile
    else:
        phase_r = torch.zeros_like(frame)
        phase_c = torch.zeros_like(frame)

    keep_r = ((r - phase_r) % tile) < kept
    keep_c = ((c - phase_c) % tile) < kept
    keep = keep_r & keep_c & in_grid

    # Specials are always kept regardless of phase-shift.
    keep = keep | is_special

    # Anchor frame: keep ALL merged tokens whose parent lies in that frame
    # (the "global truth ref" trick from Line B C2 winner).
    if anchor_frame_idx is not None:
        keep = keep | (frame == int(anchor_frame_idx))

    # Ψ-protected tokens: never subsample. A merged token survives if its
    # parent (dst) global index is flagged in protected_global_idx — same
    # top-α set that Line A's merge protection uses, so the importance
    # tokens stay full-density on both axes.
    if protected_global_idx is not None:
        keep = keep | protected_global_idx.to(keep.device)[g]
    return keep


def _build_psi_protect_mask(
    importance_scores: torch.Tensor,   # [BS, T_patches]
    S: int,
    Hp: int,
    Wp: int,
    tokens_per_frame: int,
    num_special: int,
    alpha_kv: float,
) -> torch.Tensor:
    """Build the Ψ-protect-K/V mask. Scene-level (no per-layer dep).

    Returns bool[S * tokens_per_frame] marking which global tokens skip
    the K/V phase-shift subsample. Frame 0 patches are all protected;
    frames 1..S-1 protect the top-α-by-Ψ patches.

    Hoisted out of `_hybrid_global_forward` so the topk + bool tensor
    allocation runs ONCE per scene instead of 24× per layer.
    """
    if alpha_kv <= 0 or importance_scores is None:
        return None
    T_patches = Hp * Wp
    num_protect = max(1, int(T_patches * alpha_kv))
    dev = importance_scores.device
    protected_global = torch.zeros(
        S * tokens_per_frame, dtype=torch.bool, device=dev,
    )
    # Frame 0: protect every patch (matches Line A merge convention).
    base0 = num_special
    protected_global[base0:base0 + T_patches] = True
    # Frames 1..S-1: top-α by Ψ.
    if S > 1 and importance_scores.shape[0] >= S:
        imp_rest = importance_scores[1:S]                    # [S-1, T_patches]
        k = min(num_protect, imp_rest.shape[1])
        _, topk = imp_rest.topk(k, dim=1)                     # [S-1, k]
        frame_offsets = (
            torch.arange(1, S, device=dev) * tokens_per_frame + num_special
        ).view(-1, 1)
        flat_idx = (frame_offsets + topk).reshape(-1)
        protected_global[flat_idx] = True
    return protected_global


# ─────────────────────────────────────────────────────────────────
# Hybrid global block forward (merge + subsample K/V)
# ─────────────────────────────────────────────────────────────────
def _hybrid_global_forward(
    block,
    x_all,                       # [B, N, C] tokens for this layer
    pos,                         # [B, N, 2] RoPE positions
    merge_ratio: float,
    importance_scores,           # [S, T_patches] or None
    merge_cache_key,             # hashable or None
    S: int, Hp: int, Wp: int,
    tokens_per_frame: int,
    num_special: int,
    sigma_sub,                # int (1, 2, 3, ...) or float (1.5)
    use_phase_shift: bool,
    merge_alpha: float = 0.1,
    use_avggt_mean_fill: bool = False,
    use_avggt_full: bool = False,
    anchor_frame_idx=None,
    protect_psi_in_subsample: bool = True,
    psi_protect_alpha=None,           # default: reuse merge_alpha
    # Pre-computed by run_aggregator_regimevggt (scene-level, not per-layer).
    # When supplied, skips the per-layer topk + bool-tensor allocation.
    precomputed_psi_mask: torch.Tensor = None,
):
    """One global block forward: merge(Q, K, V) + phase-shift subsample(K, V).
    Mirrors Attention.forward's computation path, does not call block.attn.forward
    directly — we need a custom attention with unequal Q and K/V sizes.

    When protect_psi_in_subsample=True, the same top-α tokens that Line A's
    merge logic protects from being merge sources are also marked always-kept
    in the K/V subsample (frame 0 fully protected). Pass psi_protect_alpha to
    decouple the two thresholds; defaults to merge_alpha.
    """
    attn_mod = block.attn
    B, N_all, C = x_all.shape
    H = attn_mod.num_heads
    D = attn_mod.head_dim

    # ── 1. LayerNorm ──
    x_normed = block.norm1(x_all)

    # ── 2. QKV projection on full tokens ──
    qkv = attn_mod.qkv(x_normed).reshape(B, N_all, 3, H, D).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # each [B, H, N, D]
    q = attn_mod.q_norm(q)
    k = attn_mod.k_norm(k)
    if attn_mod.rope is not None and pos is not None:
        q = attn_mod.rope(q, pos)
        k = attn_mod.rope(k, pos)
    del qkv, x_normed

    # ── 3. Bipartite merge on x_all (metric), applied to Q, K, V jointly ──
    r_count = int(N_all * merge_ratio)
    generator = torch.Generator(device=x_all.device)
    generator.manual_seed(33)
    merge_fn, unmerge_fn = token_merge_bipartite2d(
        x_all, Wp, Hp, 2, 2, r_count, False, generator,
        enable_protection=True,
        importance_scores=importance_scores,
        alpha=merge_alpha,
        cache_key=merge_cache_key,
    )
    merged_global_idx = merge_fn.merged_global_idx  # [N_m] int64

    # Flatten q, k, v from [B, H, N, D] to [B, N, H*D] for merge_fn.
    q_flat = q.permute(0, 2, 1, 3).reshape(B, N_all, H * D)
    k_flat = k.permute(0, 2, 1, 3).reshape(B, N_all, H * D)
    v_flat = v.permute(0, 2, 1, 3).reshape(B, N_all, H * D)
    del q, k, v

    q_m, k_m, v_m = merge_fn(
        q_flat, mode="mean", extra_tensors=k_flat, extra_tensors_2=v_flat
    )
    del q_flat, k_flat, v_flat
    N_m = q_m.shape[1]

    # Reshape back to [B, H, N_m, D].
    q_m = q_m.reshape(B, N_m, H, D).permute(0, 2, 1, 3)
    k_m = k_m.reshape(B, N_m, H, D).permute(0, 2, 1, 3)
    v_m = v_m.reshape(B, N_m, H, D).permute(0, 2, 1, 3)

    # ── 4a. Ψ-protected global-id mask (always kept in K/V) ──
    # Hoisted out of per-layer path: scene-level mask is now built ONCE in
    # run_aggregator_regimevggt and passed in via precomputed_psi_mask.
    # (Building it per-layer was O(S * tokens_per_frame) bool allocation +
    # topk on [S-1, T_patches] × 24 layers — measurable on S=1000.)
    if precomputed_psi_mask is not None:
        protected_global = precomputed_psi_mask
    elif protect_psi_in_subsample and importance_scores is not None:
        # Fallback path (used if caller didn't pre-compute). Same logic.
        protected_global = _build_psi_protect_mask(
            importance_scores, S, Hp, Wp, tokens_per_frame, num_special,
            float(merge_alpha if psi_protect_alpha is None else psi_protect_alpha),
        )
    else:
        protected_global = None

    # ── 4b. Phase-shift subsample mask on merged K/V ──
    keep_mask = _build_keeper_mask_on_merged(
        merged_global_idx, S, Hp, Wp, tokens_per_frame,
        num_special, sigma_sub, use_phase_shift,
        anchor_frame_idx=anchor_frame_idx,
        protected_global_idx=protected_global,
    )
    keeper_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)

    # ToMe size: number of original tokens each merged-output position
    # represents (attached by token_merge_bipartite2d). After subsample,
    # only entries at keeper_idx survive.
    merge_size_full = getattr(merge_fn, "size", None)
    kept_size = (
        merge_size_full[keeper_idx] if merge_size_full is not None else None
    )

    # ── 5. Attention Q_m (N_m) × K_sub × V_sub ──
    if use_avggt_full:
        # Full AVGGT: selected + diagonal + mean-fill via LSE-combine.
        attn_out = _avggt_attention_full(q_m, k_m, v_m, keeper_idx, attn_mod.scale)
        del q_m, k_m, v_m
    elif use_avggt_mean_fill:
        # Partial AVGGT: selected + mean-fill (no diagonal).
        attn_out = _avggt_attention(q_m, k_m, v_m, keeper_idx, attn_mod.scale)
        del q_m, k_m, v_m
    else:
        k_sub = k_m[:, :, keeper_idx, :]
        v_sub = v_m[:, :, keeper_idx, :]
        if kept_size is not None:
            # ToMe size-aware bias on the (merge → subsample) K/V via
            # dim-append flash trick: encode log(size)·sqrt(d) into an extra
            # K-dim, append matching cols to Q (=1) and V (=0), pad to next
            # flash-supported head_dim. Same recipe as vggt/layers/attention.py
            # so flash sdpa is preserved (no attn_mask → no math fallback).
            import math
            d_head = q_m.shape[-1]
            flash_supported = (8, 16, 32, 64, 80, 96, 112, 128, 160, 192, 224, 256)
            target_d = next(
                (fd for fd in flash_supported if fd > d_head), d_head + 1
            )
            pad_extra = target_d - d_head - 1
            sqrt_d = math.sqrt(d_head)

            log_size_scaled = (
                (torch.log(kept_size.clamp_min(1.0)) * sqrt_d)
                .to(q_m.dtype)
                .view(1, 1, -1, 1)
                .expand(k_sub.shape[0], k_sub.shape[1], k_sub.shape[2], 1)
            )
            ones_q = torch.ones(
                *q_m.shape[:-1], 1, device=q_m.device, dtype=q_m.dtype
            )
            zero_v = torch.zeros(
                *v_sub.shape[:-1], 1, device=v_sub.device, dtype=v_sub.dtype
            )
            if pad_extra > 0:
                q_pad = torch.zeros(
                    *q_m.shape[:-1], pad_extra, device=q_m.device, dtype=q_m.dtype
                )
                k_pad = torch.zeros(
                    *k_sub.shape[:-1], pad_extra, device=k_sub.device, dtype=k_sub.dtype
                )
                v_pad = torch.zeros(
                    *v_sub.shape[:-1], pad_extra, device=v_sub.device, dtype=v_sub.dtype
                )
                q_aug = torch.cat([q_m, ones_q, q_pad], dim=-1)
                k_aug = torch.cat([k_sub, log_size_scaled, k_pad], dim=-1)
                v_aug = torch.cat([v_sub, zero_v, v_pad], dim=-1)
            else:
                q_aug = torch.cat([q_m, ones_q], dim=-1)
                k_aug = torch.cat([k_sub, log_size_scaled], dim=-1)
                v_aug = torch.cat([v_sub, zero_v], dim=-1)
            attn_aug = F.scaled_dot_product_attention(
                q_aug, k_aug, v_aug, scale=1.0 / sqrt_d
            )
            attn_out = attn_aug[..., :d_head]
            del q_aug, k_aug, v_aug, attn_aug, log_size_scaled, ones_q, zero_v
        else:
            attn_out = F.scaled_dot_product_attention(q_m, k_sub, v_sub)
        del q_m, k_m, v_m, k_sub, v_sub

    # ── 6. Output projection (on merged-token layout) ──
    attn_out = attn_out.transpose(1, 2).reshape(B, N_m, C)
    attn_out = attn_mod.proj(attn_out)
    attn_out = attn_mod.proj_drop(attn_out)

    # ── 7. Unmerge and residual ──
    attn_out = unmerge_fn(attn_out)    # [B, N_all, C]
    x = x_all + block.ls1(attn_out)
    del attn_out
    x = x + block.ls2(block.mlp(block.norm2(x)))
    return x


# ─────────────────────────────────────────────────────────────────
# Aggregator
# ─────────────────────────────────────────────────────────────────
def run_aggregator_regimevggt(
    model,
    images,
    # Token-merge axis config (rho_s / rho_m / rho_d, band-aligned cache).
    shallow_merge_ratio: float = 0.95,
    merge_ratio: float = 0.95,
    deep_merge_ratio: float = 0.99,
    middle_merge_start: int = 10,
    deep_merge_start: int = 14,
    cache_band_starts=(0, 10, 14),
    importance_method: str = "dino_attn",
    merge_alpha: float = 0.1,
    protect_last: bool = False,       # Line A default: L23 merged
    # Line B (phase-shift subsample on merged K/V) config.
    sigma_sub=2,                      # fallback for any band w/o explicit value
    sigma_shallow=None,               # L in [0, mid_lo) — defaults to sigma_sub
    sigma_deep=None,                  # L in [mid_hi, depth) — defaults to sigma_sub
    use_phase_shift: bool = True,
    use_avggt_mean_fill: bool = False,
    use_avggt_full: bool = False,
    # Angle 2: rank-aware layer selection.
    #   protect_middle=True → layers in [middle_range) do merge only
    #   (Line A path), not merge+subsample. Shallow/deep keep hybrid.
    protect_middle: bool = False,
    middle_range=(10, 18),
    # Line B C2 anchor (winning T&T config): keep ALL merged tokens whose
    # parent lies in this frame. Mirrors phase_shift's anchor_frame_idx.
    anchor_frame_idx=None,
    # Ψ-protected tokens skip the K/V subsample (importance tokens
    # stay full-density on both compression axes; everything else still
    # pays merge × subsample). Adds per-layer topk + 1M-bool tensor on
    # S=1000 — measurable overhead. Default OFF to match paper AB2
    # (commit 4f6eb022 introduced this; paper 4-28 ran without it).
    # Set to True to enable Ψ-protect-K/V on top of the AB2 baseline.
    protect_psi_in_subsample: bool = False,
    psi_protect_alpha=None,
):
    """Hybrid merge-then-subsample aggregator.

    Default config mirrors Line A `band_0_10_14_nolast` plus σ_sub=2
    phase-shift on merged K/V at every Stage 2 layer (L0-22, all 23
    layers when protect_last=False; L0-22 only if protect_last=True).
    """
    import torch.nn.functional as _F  # local alias, avoid shadowing
    agg = model.aggregator
    B, S, C_in, H, W = images.shape
    Hp = H // agg.patch_size
    Wp = W // agg.patch_size

    # ── DINOv2 CLS attention hook (for importance_method="dino_attn") ──
    psi = None
    hook_handle = None
    captured_psi = []
    if importance_method == "dino_attn" and merge_ratio > 0:
        dino_blocks = agg.patch_embed.blocks
        last_attn = dino_blocks[-1].attn

        def _psi_hook(module, inp, out):
            x_in = inp[0]
            B_h, N_h, C_h = x_in.shape
            qkv = module.qkv(x_in).reshape(
                B_h, N_h, 3, module.num_heads, C_h // module.num_heads
            )
            q, k, _ = qkv.unbind(2)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            q_cls = q[:, :, 0:1]
            attn = torch.matmul(q_cls, k.transpose(-2, -1)) * (module.head_dim ** -0.5)
            attn = _F.softmax(attn, dim=-1)
            cls_attn = attn[:, :, 0, 1:].mean(dim=1)
            captured_psi.append(cls_attn.detach())

        hook_handle = last_attn.register_forward_hook(_psi_hook)

    # Clear merge cache for this scene.
    try:
        from merging.merge import clear_merge_cache as _clear1
        _clear1()
    except ImportError:
        pass
    try:
        from methods.fastvggt_merge import clear_merge_cache as _clear2
        _clear2()
    except ImportError:
        pass

    # ── Patch embedding ──
    imgs = (images - agg._resnet_mean) / agg._resnet_std
    imgs_flat = imgs.view(B * S, C_in, H, W)
    patch_tokens = agg.patch_embed(imgs_flat)
    if isinstance(patch_tokens, dict):
        patch_tokens = patch_tokens["x_norm_patchtokens"]

    if hook_handle is not None:
        hook_handle.remove()
        if captured_psi:
            psi_raw = torch.cat(captured_psi, dim=0)  # [BS, N_dino - 1]
            T_patches = Hp * Wp
            if psi_raw.shape[1] >= T_patches:
                psi = psi_raw[:, -T_patches:].contiguous()
            else:
                pad = torch.zeros(
                    psi_raw.shape[0], T_patches - psi_raw.shape[1],
                    device=psi_raw.device, dtype=psi_raw.dtype,
                )
                psi = torch.cat([psi_raw, pad], dim=1)
        del captured_psi

    camera_token = slice_expand_and_flatten(agg.camera_token, B, S)
    register_token = slice_expand_and_flatten(agg.register_token, B, S)
    tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
    del camera_token, register_token, patch_tokens, imgs_flat, imgs

    pos = None
    if agg.rope is not None:
        pos = agg.position_getter(B * S, Hp, Wp, device=images.device)
        pos = pos + 1
        pos_special = torch.zeros(
            B * S, agg.patch_start_idx, 2, device=images.device, dtype=pos.dtype
        )
        pos = torch.cat([pos_special, pos], dim=1)

    _, P, C = tokens.shape
    patch_start = agg.patch_start_idx
    num_special = patch_start
    tokens_per_frame = P  # = num_special + Hp*Wp

    # ── Scene-level Ψ-protect-K/V mask (pre-computed once, used at every
    # global layer). Hoisted out of `_hybrid_global_forward` because the
    # mask depends only on importance_scores + S + T_patches, all of which
    # are scene-level. Building it per-layer was paying topk + 1M-bool
    # tensor allocation × 24 layers × 50 scenes on ScanNet N=1000.
    psi_protect_mask = None
    if protect_psi_in_subsample and psi is not None:
        psi_protect_mask = _build_psi_protect_mask(
            psi, S, Hp, Wp, tokens_per_frame, num_special,
            float(merge_alpha if psi_protect_alpha is None else psi_protect_alpha),
        )

    del images
    torch.cuda.empty_cache()

    depth = agg.depth
    block4DPT_idx = [4, 11, 17, 23]
    output_list = []

    if cache_band_starts is not None:
        cache_band_starts = sorted(set(cache_band_starts))

    full_layers = {depth - 1} if protect_last else set()
    mid_lo, mid_hi = middle_range  # [mid_lo, mid_hi) — merge-only if protect_middle

    # Resolve per-band sigma (fall back to sigma_sub if not explicitly set).
    sig_shallow = sigma_shallow if sigma_shallow is not None else sigma_sub
    sig_middle  = sigma_sub
    sig_deep    = sigma_deep    if sigma_deep    is not None else sigma_sub

    for i in range(depth):
        # Always run Frame block first (unchanged by hybrid).
        if tokens.shape[0] != B * S:
            tokens = tokens.view(B * S, P, C)
        frame_pos = pos.view(B * S, P, 2) if pos is not None else None
        tokens = agg.frame_blocks[i](tokens, pos=frame_pos)
        frame_inter = tokens.view(B, S, P, C) if i in block4DPT_idx else None

        tokens_global = tokens.view(B, S * P, C)
        global_pos = pos.view(B, S * P, 2) if pos is not None else None

        if i in full_layers:
            # L23 protected: Full Global, no merge, no subsample.
            tokens_global = agg.global_blocks[i](tokens_global, pos=global_pos)
        else:
            # Per-layer merge ratio (U-shape).
            if deep_merge_ratio is not None and i >= deep_merge_start:
                layer_ratio = deep_merge_ratio
            elif shallow_merge_ratio is not None and i < middle_merge_start:
                layer_ratio = shallow_merge_ratio
            else:
                layer_ratio = merge_ratio

            # Cache key (band-based; NO subsample params so merge-only
            # and merge+subsample layers can share merge indices
            # within the same band).
            merge_cache_key = None
            if layer_ratio > 0 and cache_band_starts is not None:
                band_idx = sum(1 for s in cache_band_starts if s <= i) - 1
                if band_idx >= 0:
                    merge_cache_key = (
                        "regimevggt_band", band_idx, round(layer_ratio, 4),
                    )

            in_middle = mid_lo <= i < mid_hi
            # Per-band sigma selection for the hybrid branch.
            if i < mid_lo:
                layer_sigma = sig_shallow
            elif i >= mid_hi:
                layer_sigma = sig_deep
            else:
                layer_sigma = sig_middle

            if protect_middle and in_middle:
                # Merge only (Line A path): call the global block with
                # global_merging; Attention.forward handles merge+proj+
                # unmerge internally using the pretrained weights.
                agg.global_blocks[i].attn.merge_ratio = layer_ratio
                gm = i if layer_ratio > 0 else None
                tokens_global = agg.global_blocks[i](
                    tokens_global, pos=global_pos, global_merging=gm,
                    importance_scores=psi,
                    merge_cache_key=merge_cache_key,
                    merge_alpha=merge_alpha,
                )
            else:
                # Merge + phase-shift subsample (hybrid path).
                tokens_global = _hybrid_global_forward(
                    agg.global_blocks[i], tokens_global, global_pos,
                    merge_ratio=layer_ratio,
                    importance_scores=psi,
                    merge_cache_key=merge_cache_key,
                    S=S, Hp=Hp, Wp=Wp,
                    tokens_per_frame=tokens_per_frame,
                    num_special=num_special,
                    sigma_sub=layer_sigma,
                    use_phase_shift=use_phase_shift,
                    merge_alpha=merge_alpha,
                    use_avggt_mean_fill=use_avggt_mean_fill,
                    use_avggt_full=use_avggt_full,
                    anchor_frame_idx=anchor_frame_idx,
                    protect_psi_in_subsample=protect_psi_in_subsample,
                    psi_protect_alpha=psi_protect_alpha,
                    precomputed_psi_mask=psi_protect_mask,
                )
        tokens = tokens_global.view(B * S, P, C)

        if i in block4DPT_idx:
            global_inter = tokens.view(B, S, P, C)
            concat_inter = torch.cat([frame_inter, global_inter], dim=-1)
            if concat_inter.dtype != torch.bfloat16:
                concat_inter = concat_inter.to(torch.bfloat16)
            output_list.append(concat_inter)
            del frame_inter, global_inter, concat_inter

        if os.environ.get("GEOVGGT_NO_PER_LAYER_CLEAN", "0") != "1":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if agg.rope is not None and hasattr(agg.rope, "frequency_cache"):
                agg.rope.frequency_cache.clear()

    del tokens, pos
    torch.cuda.empty_cache()
    return output_list, agg.patch_start_idx
