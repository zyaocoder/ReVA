# coding=utf-8
"""
ReMoScene wrapper for ReVA.

This module plugs a trainable video compression layer into the local
multimodal backbone implementation that ships with this repository.

Architecture (all steps follow the user spec):

  Step 1  Video sampling handled externally (processor / dataset).
  Step 2  Frozen RICE vision encoder extracts per-patch tokens Z  (T·P × Dv).
  Step 3  3-D adaptive average pooling seeds initial query tokens Q0 (Lq × Dv).
  Step 4  ReMoScene Selector: bidirectional selective-scan with interleaved queries
          compresses [Z ; Q0] → Q' (Lq × Dv).
  Step 5  Motion Branch: lightweight per-pair matching + GRU aggregation
          produces global camera-motion token g (1 × Dv).
  Step 6  Gated fusion: V = concat(α·g, Q')  (Lq+1 × Dv).
  Step 7  Trainable projector: V → H_vis  (Lq+1 × Dl).
  Step 8  H_vis replaces video placeholder tokens in inputs_embeds.
  Step 9  Frozen LLM decodes the answer.

Trainable:  ReMoSceneSelector · MotionBranch · ReMoSceneProjector · motion_gate
Frozen   :  everything inside the wrapped base model

Dependencies
------------
  Required : transformers, torch
  Optional : mamba_ssm  (pip install mamba-ssm)
             If absent, a bidirectional self-attention block is used instead.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional mamba_ssm import
# ---------------------------------------------------------------------------
try:
    from mamba_ssm import Mamba as _MambaSSMBase
    HAS_MAMBA_SSM = True
except ImportError:
    HAS_MAMBA_SSM = False

from transformers import Qwen2_5_VLForConditionalGeneration

# ============================================================================
# Utility
# ============================================================================

def pool3d_init_queries(
    Z_3d: torch.Tensor, lq_t: int, lq_h: int, lq_w: int
) -> torch.Tensor:
    """
    Seed query tokens from video features via 3-D adaptive average pooling.

    Args:
        Z_3d  : (T, H, W, Dv)  spatiotemporal pre-merger vision features.
        lq_t  : target temporal resolution.
        lq_h  : target height  resolution.
        lq_w  : target width   resolution.

    Returns:
        (lq_t * lq_h * lq_w, Dv)  flattened initial query tokens.
    """
    T, H, W, Dv = Z_3d.shape
    # Clamp targets so we never up-sample (only pool down)
    lq_t = min(T, lq_t)
    lq_h = min(H, lq_h)
    lq_w = min(W, lq_w)

    # (1, Dv, T, H, W) → adaptive 3-D pool
    x = Z_3d.permute(3, 0, 1, 2).unsqueeze(0).float()       # (1, Dv, T, H, W)
    x = F.adaptive_avg_pool3d(x, (lq_t, lq_h, lq_w))        # (1, Dv, t, h, w)
    x = x.squeeze(0).permute(1, 2, 3, 0).reshape(-1, Dv)    # (Lq_init, Dv)
    return x.to(Z_3d.dtype)


def resize_queries(Q: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Resize a 1-D sequence of tokens to *target_len* via linear interpolation.
    Q : (N, D)  →  returns (target_len, D).
    """
    if Q.shape[0] == target_len:
        return Q
    # (1, D, N) → interpolate → (1, D, target_len)
    Q_t = Q.T.unsqueeze(0).float()                           # (1, D, N)
    Q_t = F.interpolate(Q_t, size=target_len, mode="linear", align_corners=False)
    return Q_t.squeeze(0).T.to(Q.dtype)                      # (target_len, D)


# ============================================================================
# Step 4 – ReMoScene Selector
# ============================================================================

# ----------------------------------------------------------------------------
# Bidirectional block implementations
# ----------------------------------------------------------------------------

class _AttentionBlock(nn.Module):
    """
    Bidirectional self-attention + FFN block (mamba_ssm fallback).
    O(N²) but fully parallel.
    """

    def __init__(self, d_model: int, nhead: int = 8, expand: int = 2):
        super().__init__()
        # Ensure nhead divides d_model
        while d_model % nhead != 0 and nhead > 1:
            nhead //= 2
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, nhead, batch_first=True,
                                           dropout=0.0)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = d_model * expand
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, L, D)
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class _BidirMambaBlock(nn.Module):
    """
    True bidirectional Mamba block using mamba_ssm library.
    Forward + backward scans are summed and projected back.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.norm   = nn.LayerNorm(d_model)
        self.mamba_f = _MambaSSMBase(d_model, d_state=d_state,
                                     d_conv=d_conv, expand=expand)
        self.mamba_b = _MambaSSMBase(d_model, d_state=d_state,
                                     d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, L, D)
        h    = self.norm(x)
        fwd  = self.mamba_f(h)
        bwd  = self.mamba_b(h.flip(1)).flip(1)
        return x + (fwd + bwd)


def _make_block(
    d_model: int,
    nhead: int = 8,
    expand: int = 2,
    d_state: int = 16,
    d_conv: int = 4,
    prefer_mamba: bool = True,
) -> nn.Module:
    if prefer_mamba and HAS_MAMBA_SSM:
        return _BidirMambaBlock(d_model, d_state=d_state,
                                d_conv=d_conv, expand=expand)
    return _AttentionBlock(d_model, nhead=nhead, expand=expand)


# ----------------------------------------------------------------------------
# ReMoSceneSelector
# ----------------------------------------------------------------------------

class ReMoSceneSelector(nn.Module):
    """
    Selective-scan token compressor with interleaved query tokens.

    Per layer:
      1. Uniformly interleave Lq query tokens among the N video tokens.
      2. Run a bidirectional Mamba (or attention) layer over the combined
         sequence of length N + Lq.
      3. Extract the updated query positions → Q_new.
         Optionally extract updated video tokens for the next layer.

    Output: Q' of shape (Lq, Dv) – compressed spatiotemporal representation.
    """

    def __init__(
        self,
        d_model: int,
        num_queries: int,
        n_layers: int = 2,
        nhead: int = 8,
        expand: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        prefer_mamba: bool = True,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.layers = nn.ModuleList([
            _make_block(d_model, nhead=nhead, expand=expand,
                        d_state=d_state, d_conv=d_conv,
                        prefer_mamba=prefer_mamba)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------

    @staticmethod
    def _build_interleave_mask(
        N: int, Lq: int, device: torch.device
    ) -> torch.BoolTensor:
        """
        Build a boolean mask of length N+Lq with Lq True entries placed
        uniformly across the sequence (interleaved query positions).
        """
        total = N + Lq
        mask  = torch.zeros(total, dtype=torch.bool, device=device)
        # linspace gives Lq positions uniformly in [0, total-1]
        indices = torch.linspace(0, total - 1, Lq + 1,
                                 device=device, dtype=torch.long)[1:]
        mask[indices] = True
        return mask

    def forward(self, Z: torch.Tensor, Q0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z  : (N, Dv)   flat video tokens (all frames concatenated).
            Q0 : (Lq, Dv)  seed query tokens from 3-D pooling.

        Returns:
            Q' : (Lq, Dv)  compressed query tokens.
        """
        # Add batch dimension; Mamba / MHA both expect (B, L, D)
        Z = Z.unsqueeze(0)   # (1, N, Dv)
        Q = Q0.unsqueeze(0)  # (1, Lq, Dv)

        B, N, D  = Z.shape
        Lq       = Q.shape[1]

        for layer in self.layers:
            # Build interleaving mask (same positions at every layer)
            mask = self._build_interleave_mask(N, Lq, Z.device)  # (N+Lq,)

            # Assemble combined sequence
            combined = Z.new_zeros(B, N + Lq, D)
            combined[:, mask]  = Q   # (B, Lq, D)
            combined[:, ~mask] = Z   # (B, N,  D)

            # Bidirectional scan
            combined = layer(combined)  # (B, N+Lq, D)

            # Extract updated queries and video tokens
            Q = combined[:, mask]   # (B, Lq, D)
            Z = combined[:, ~mask]  # (B, N,  D)

        return self.norm_out(Q.squeeze(0))  # (Lq, Dv)


# ============================================================================
# Step 5 – Motion Branch
# ============================================================================

class MotionMatch(nn.Module):
    """
    Per-pair global cost-volume motion feature extractor.

    1. Spatially downsamples both frames to (ds_h, ds_w) via adaptive avg-pool
       so the cost volume stays at most (N', N') where N' = ds_h * ds_w.
    2. Projects and L2-normalises the downsampled tokens.
    3. Computes the full (N', N') scalar cost volume (cosine similarity).
    4. Unsqueezes to (N', N', 1) and applies an MLP → (N', N', Dv).
    5. Global average-pools over both spatial axes → (Dv,).
    """

    def __init__(self, d_model: int, downsample_hw: Tuple[int, int] = (14, 14)):
        super().__init__()
        self.downsample_hw = downsample_hw
        self.proj  = nn.Linear(d_model, d_model, bias=False)
        # MLP: 1 scalar → d_model, applied independently to each (i,j) entry
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )

    def forward(self, X_t: torch.Tensor, X_t1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X_t, X_t1 : (H, W, Dv)  spatial patch tokens for frames t, t+1.
        Returns:
            g_t : (Dv,)  global motion descriptor for this frame pair.
        """
        H, W, Dv = X_t.shape
        dh, dw = self.downsample_hw

        # Spatially downsample: (H, W, Dv) → (1, Dv, H, W) → pool → (dh, dw, Dv)
        def _pool(x: torch.Tensor) -> torch.Tensor:
            return (F.adaptive_avg_pool2d(
                        x.permute(2, 0, 1).unsqueeze(0),  # (1, Dv, H, W)
                        (dh, dw))                          # (1, Dv, dh, dw)
                    .squeeze(0)                            # (Dv, dh, dw)
                    .permute(1, 2, 0))                     # (dh, dw, Dv)

        f_t  = _pool(X_t)   # (dh, dw, Dv)
        f_t1 = _pool(X_t1)  # (dh, dw, Dv)
        N = dh * dw

        # Project then L2-normalise before computing correlations
        f1 = F.normalize(self.proj(f_t.reshape(N, Dv)),  dim=-1)  # (N, Dv)
        f2 = F.normalize(self.proj(f_t1.reshape(N, Dv)), dim=-1)  # (N, Dv)

        # Scalar cost volume at low resolution
        cost = torch.matmul(f1, f2.T)                   # (N, N)  ∈ [-1, 1]

        # Expand each scalar to a Dv-dim feature via MLP
        feat = self.mlp(cost.unsqueeze(-1))              # (N, N, Dv)

        # Global average pool over the full (N, N) cost volume
        return feat.mean(dim=(0, 1))                     # (Dv,)


class MotionBranch(nn.Module):
    """
    Full motion branch.

    1. For every consecutive frame pair, extract a motion vector via MotionMatch.
    2. Aggregate across time with a 1-layer GRU.
    3. Return a single global motion token g ∈ R^{1 × Dv}.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.motion_match = MotionMatch(d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Z_3d: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            Z_3d : (T, H, W, Dv)  per-frame pre-merger patch features.
        Returns:
            g          : (1, Dv)          global camera-motion token.
            pair_feats : List[(Dv,)]      per-pair motion vectors, length T-1.
                         Empty list when T < 2.
        """
        T, H, W, Dv = Z_3d.shape

        if T < 2:
            # Single-frame fallback: use mean of spatial features
            g = Z_3d.reshape(-1, Dv).mean(dim=0, keepdim=True)  # (1, Dv)
            return self.norm(g), []

        # Compute per-pair motion vectors via cost volume
        pair_feats = []
        for t in range(T - 1):
            g_t = self.motion_match(Z_3d[t], Z_3d[t + 1])   # (Dv,)
            pair_feats.append(g_t)

        # Stack → (1, T-1, Dv) and run GRU
        pairs = torch.stack(pair_feats, dim=0).unsqueeze(0)  # (1, T-1, Dv)
        _, h_n = self.gru(pairs)                             # h_n: (1, 1, Dv)
        g = self.norm(h_n.squeeze(0))                        # (1, Dv)
        return g, pair_feats


# ============================================================================
# Step 5b – Recurrent Object Selector
# ============================================================================

class _CrossAttentionBlock(nn.Module):
    """
    Cross-attention block: queries attend to a separate key/value sequence.
    Pre-norm on both Q and KV, residual connection, followed by FFN.
    No batch dimension — inputs are (N, D) 2-D tensors.
    """

    def __init__(self, d_model: int, nhead: int = 8, expand: int = 2):
        super().__init__()
        while d_model % nhead != 0 and nhead > 1:
            nhead //= 2
        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn    = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)
        self.norm2   = nn.LayerNorm(d_model)
        hidden = d_model * expand
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, Q: torch.Tensor, KV: torch.Tensor) -> torch.Tensor:
        """
        Q  : (N_q,  D)
        KV : (N_kv, D)
        Returns: (N_q, D)
        """
        Q_b  = Q.unsqueeze(0)    # (1, N_q,  D)
        KV_b = KV.unsqueeze(0)   # (1, N_kv, D)
        kv_n = self.norm_kv(KV_b)
        out  = self.attn(self.norm_q(Q_b), kv_n, kv_n, need_weights=False)[0]
        Q_b  = Q_b + out
        Q_b  = Q_b + self.ffn(self.norm2(Q_b))
        return Q_b.squeeze(0)    # (N_q, D)


class RecurrentObjectSelector(nn.Module):
    """
    Cross-frame recurrent object query selector.

    Learnable object queries Q are propagated across frames and updated at
    each step via cross-attention over that frame's visual (+ motion) tokens:

        Q_t = CrossAttn(Q=Q_{t-1}, KV=[motion_t | visual_t])

    The updated object tokens for each frame are concatenated with that
    frame's visual tokens, producing per-frame enriched representations
    stacked across all T frames.

    Args:
        d_model        : feature dimension (Dv).
        num_queries    : number of learnable object queries (N).
        n_layers       : depth of per-step cross-attention stack.
        nhead          : attention heads.
        expand         : FFN expansion factor.

    Forward inputs:
        Z_3d        : (T, H, W, Dv)  per-frame ViT patch features.
        pair_feats  : List[(Dv,)]    per-pair motion vectors from MotionBranch,
                      length T-1.  Empty list is accepted (zeros used).

    Forward output:
        (T * (H·W + num_queries), Dv)  per-frame enriched tokens
        [visual_t | object_queries_t] stacked across all T frames.
    """

    def __init__(
        self,
        d_model    : int,
        num_queries: int = 16,
        n_layers   : int = 2,
        chunk_size : int = 8,
        nhead      : int = 8,
        expand     : int = 2,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.chunk_size = chunk_size

        # Learnable initial object query tokens
        self.query_init = nn.Parameter(torch.zeros(num_queries, d_model))
        nn.init.trunc_normal_(self.query_init, std=0.02)

        # Per-step cross-attention: object queries attend to visual tokens
        self.cross_attn_layers = nn.ModuleList([
            _CrossAttentionBlock(d_model, nhead=nhead, expand=expand)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        Z_3d       : torch.Tensor,
        pair_feats : List[torch.Tensor],
    ) -> torch.Tensor:
        T, H, W, Dv = Z_3d.shape

        # Per-frame motion context: frame 0 → zeros, frame t > 0 → pair_feats[t-1]
        zero = Z_3d.new_zeros(Dv)
        motion_ctx = [zero] + (pair_feats if pair_feats else [zero] * (T - 1))
        while len(motion_ctx) < T:
            motion_ctx.append(zero)

        Q = self.query_init.clone()              # (N, Dv) initial object queries

        per_frame_tokens: List[torch.Tensor] = []
        for t in range(T):
            v_t  = Z_3d[t].reshape(H * W, Dv)   # (HW,    Dv)
            m_t  = motion_ctx[t].unsqueeze(0)    # (1,     Dv)
            kv_t = torch.cat([m_t, v_t], dim=0)  # (1+HW,  Dv)  motion + visual

            # Cross-attention: object queries attend to motion + visual tokens
            for layer in self.cross_attn_layers:
                Q = layer(Q, kv_t)               # (N, Dv)

            Q = self.norm_out(Q)                 # (N, Dv) → passed to next frame

            # Attach updated object tokens to this frame's visual tokens
            per_frame_tokens.append(torch.cat([v_t, Q], dim=0))  # (HW+N, Dv)

        return torch.cat(per_frame_tokens, dim=0)  # (T*(HW+N), Dv)


# ============================================================================
# Step 7 – Trainable Projector  (Dv → Dl)
# ============================================================================

class ReMoSceneProjector(nn.Module):
    """
    Two-layer MLP projector that maps compressed tokens from vision space
    (Dv) to LLM embedding space (Dl).  Structure mirrors RicePatchMerger
    but operates directly on the 1-D token sequence without spatial grouping.
    """

    def __init__(self, d_vision: int, d_llm: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_vision)
        self.mlp  = nn.Sequential(
            nn.Linear(d_vision, d_vision * 4),
            nn.GELU(),
            nn.Linear(d_vision * 4, d_llm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (N, Dv)  →  (N, Dl)"""
        return self.mlp(self.norm(x))


# ============================================================================
# ReMoSceneConfig
# ============================================================================

class ReMoSceneConfig:
    """
    Hyper-parameters for the ReMoScene extension modules.

    Attributes
    ----------
    num_queries       : int   Total number of compressed video tokens (Lq) for ReMoSceneSelector.
    n_remoscene_layers    : int   Depth of the ReMoScene selector.
    remoscene_nhead       : int   Attention heads (used in fallback attention blocks).
    remoscene_expand      : int   Inner-dimension expansion factor.
    remoscene_d_state     : int   Mamba SSM state dimension (ignored if no mamba_ssm).
    remoscene_d_conv      : int   Mamba local conv kernel size.
    lq_t, lq_h, lq_w : int   3-D pool targets for query seed initialisation.
                              lq_t * lq_h * lq_w should equal num_queries.
    num_object_queries: int   Learnable object queries per frame in RecurrentObjectSelector (N).
    rec_n_layers      : int   Depth of the per-step cross-attention stack in RecurrentObjectSelector.
    rec_nhead         : int   Attention heads for RecurrentObjectSelector.
    """

    def __init__(
        self,
        num_queries       : int = 256,
        n_remoscene_layers    : int = 2,
        remoscene_nhead       : int = 8,
        remoscene_expand      : int = 2,
        remoscene_d_state     : int = 16,
        remoscene_d_conv      : int = 4,
        lq_t              : int = 4,
        lq_h              : int = 8,
        lq_w              : int = 8,
        num_object_queries: int  = 16,
        rec_n_layers      : int  = 2,
        rec_nhead         : int  = 8,
        rec_chunk_size    : int  = 8,
        use_remoscene_selector  : bool = True,
        use_motion_branch   : bool = True,
        use_recurrent_selector: bool = True,
    ):
        self.num_queries        = num_queries
        self.n_remoscene_layers     = n_remoscene_layers
        self.remoscene_nhead        = remoscene_nhead
        self.remoscene_expand       = remoscene_expand
        self.remoscene_d_state      = remoscene_d_state
        self.remoscene_d_conv       = remoscene_d_conv
        self.lq_t = lq_t
        self.lq_h = lq_h
        self.lq_w = lq_w
        self.num_object_queries = num_object_queries
        self.rec_n_layers       = rec_n_layers
        self.rec_nhead          = rec_nhead
        self.rec_chunk_size     = rec_chunk_size
        self.use_remoscene_selector   = use_remoscene_selector
        self.use_motion_branch    = use_motion_branch
        self.use_recurrent_selector = use_recurrent_selector


# ============================================================================
# ReMoSceneWrapper  –  the main model class
# ============================================================================

class ReMoSceneWrapper(nn.Module):
    """
    Wraps a frozen multimodal backbone model and adds the ReMoScene
    video understanding pipeline.

    Trainable parameters
    --------------------
    - ``remoscene_selector``    :  ReMoSceneSelector
    - ``motion_branch``     :  MotionBranch
    - ``projector``         :  ReMoSceneProjector
    - ``motion_gate_logit`` :  scalar nn.Parameter  (α = sigmoid of this)

    Frozen parameters
    -----------------
    - All parameters of ``base_model`` (RICE vision encoder + Qwen2 LLM).

    Forward
    -------
    Accepts the exact same keyword arguments as
    the wrapped backbone model's ``forward()`` method.
    When ``pixel_values_videos`` is present the standard video embedding path
    is replaced by the ReMoScene pipeline, and ``inputs_embeds`` /
    ``attention_mask`` / ``labels`` are rebuilt to match the new token count.
    """

    def __init__(
        self,
        base_model : Qwen2_5_VLForConditionalGeneration,
        remoscene_cfg  : Optional[ReMoSceneConfig] = None,
    ):
        super().__init__()
        if remoscene_cfg is None:
            remoscene_cfg = ReMoSceneConfig()
        self.cfg = remoscene_cfg

        # ---- Frozen base model ----------------------------------------
        self.base_model = base_model
        for p in self.base_model.parameters():
            p.requires_grad_(False)

        # ---- Dimensions -----------------------------------------------
        Dv = base_model.config.vision_config.hidden_size          # Qwen2.5-VL pre-merger dim
        Dl = base_model.config.text_config.hidden_size           # 2560
        self.Dv  = Dv
        self.Dl  = Dl
        self.Lq  = remoscene_cfg.num_queries
        self.spatial_merge_size = base_model.config.vision_config.spatial_merge_size

        # Token IDs from base config
        self.video_token_id = base_model.config.video_token_id
        self.image_token_id = base_model.config.image_token_id

        # ---- Trainable modules ----------------------------------------
        self.use_remoscene_selector     = remoscene_cfg.use_remoscene_selector
        self.use_motion_branch      = remoscene_cfg.use_motion_branch
        self.use_recurrent_selector = remoscene_cfg.use_recurrent_selector

        if self.use_remoscene_selector:
            self.remoscene_selector = ReMoSceneSelector(
                d_model     = Dv,
                num_queries = remoscene_cfg.num_queries,
                n_layers    = remoscene_cfg.n_remoscene_layers,
                nhead       = remoscene_cfg.remoscene_nhead,
                expand      = remoscene_cfg.remoscene_expand,
                d_state     = remoscene_cfg.remoscene_d_state,
                d_conv      = remoscene_cfg.remoscene_d_conv,
            )
        if self.use_motion_branch:
            self.motion_branch     = MotionBranch(d_model=Dv)
            # Gate: learnable logit initialised to −5  →  α = sigmoid(−5) ≈ 0.007
            self.motion_gate_logit = nn.Parameter(torch.tensor(-5.0))
        if self.use_recurrent_selector:
            self.recurrent_selector = RecurrentObjectSelector(
                d_model     = Dv,
                num_queries = remoscene_cfg.num_object_queries,
                n_layers    = remoscene_cfg.rec_n_layers,
                chunk_size  = remoscene_cfg.rec_chunk_size,
                nhead       = remoscene_cfg.rec_nhead,
            )
        self.projector = ReMoSceneProjector(d_vision=Dv, d_llm=Dl)

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        """Yields only the parameters that should receive gradients."""
        if self.use_remoscene_selector:
            yield from self.remoscene_selector.parameters()
        if self.use_motion_branch:
            yield from self.motion_branch.parameters()
            yield self.motion_gate_logit
        if self.use_recurrent_selector:
            yield from self.recurrent_selector.parameters()
        yield from self.projector.parameters()

    def set_trainable(self):
        """Freeze base model; unfreeze new modules."""
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        for p in self.trainable_parameters():
            p.requires_grad_(True)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Delegate to base_model so HuggingFace Trainer can call this."""
        self.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs or {}
        )

    def gradient_checkpointing_disable(self):
        self.base_model.gradient_checkpointing_disable()

    def trainable_state_dict(self) -> dict:
        """State-dict containing only trainable parameters (for checkpointing)."""
        names = {"projector"}
        if self.use_remoscene_selector:
            names.add("remoscene_selector")
        if self.use_motion_branch:
            names.add("motion_branch")
        if self.use_recurrent_selector:
            names.add("recurrent_selector")
        sd = self.state_dict()
        result = {k: v for k, v in sd.items() if any(k.startswith(n) for n in names)}
        if self.use_motion_branch:
            result["motion_gate_logit"] = sd["motion_gate_logit"]
        return result

    # ------------------------------------------------------------------
    # Step 2 – Frozen vision encoder  (pre-merger)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_video_frozen(
        self,
        pixel_values_videos : torch.Tensor,
        video_grid_thw      : torch.LongTensor,
    ) -> torch.Tensor:
        """
        Run the frozen Qwen2.5-VL vision encoder *without* the merger projector.

        Qwen2.5-VL's vision encoder always applies the merger at the end of its
        forward pass.  We register a one-shot pre-hook on ``vis.merger`` to
        intercept the pre-merger features before they are consumed by the MLP,
        then remove the hook immediately.

        Returns
        -------
        Z_all : (N_total, Dv)
            where N_total = Σ_i  T_i · H_i · W_i   (sum over all videos).
        """
        vis = self.base_model.model.visual
        pre_merger: dict = {}

        def _capture_pre_merger(module, inp):
            pre_merger["x"] = inp[0]

        hook = vis.merger.register_forward_pre_hook(_capture_pre_merger)
        try:
            vis.forward(
                pixel_values_videos.to(dtype=vis.patch_embed.proj.weight.dtype),
                grid_thw=video_grid_thw,
            )
        finally:
            hook.remove()
        return pre_merger["x"]   # (N_total, Dv)  pre-merger features

    # ------------------------------------------------------------------
    # Steps 3-7 – single video compression
    # ------------------------------------------------------------------

    def _compress_single_video(
        self, Z: torch.Tensor, T: int, H: int, W: int
    ) -> torch.Tensor:
        """
        Run the full ReMoScene pipeline for one video.

        Args:
            Z : (T·H·W, Dv)  pre-merger features for this video.
            T, H, W : temporal / spatial patch counts.

        Returns:
            H_vis : (Lq+1, Dl)  compressed video tokens in LLM embedding space.
        """
        Dv  = Z.shape[-1]
        Lq  = self.Lq
        cfg = self.cfg

        # -- Step 3: 3-D pool → seed queries Q0 (from raw Z_3d) ----------
        Z_3d = Z.reshape(T, H, W, Dv)                       # (T, H, W, Dv)
        Q0   = pool3d_init_queries(Z_3d, cfg.lq_t, cfg.lq_h, cfg.lq_w)
        Q0   = resize_queries(Q0, Lq)                        # (Lq, Dv)

        # -- Step 5: Motion Branch (optional) ---------------------------
        if self.use_motion_branch:
            g, pair_feats = self.motion_branch(Z_3d)        # (1, Dv), List[(Dv,)]
        else:
            g, pair_feats = None, []

        # -- Step 5b: Recurrent Object Selector (optional) --------------
        # Produces per-frame enriched tokens [visual_t | object_t] stacked
        # across all frames → (T*(HW+N), Dv).  These replace raw Z as input
        # to ReMoSceneSelector so the compressed queries absorb object-level info.
        if self.use_recurrent_selector:
            Z_for_selector = self.recurrent_selector(Z_3d, pair_feats)  # (T*(HW+N), Dv)
        else:
            Z_for_selector = Z                                  # (T*HW, Dv)

        # -- Step 4: ReMoScene Selector (optional) --------------------------
        if self.use_remoscene_selector:
            Q_prime = self.remoscene_selector(Z_for_selector, Q0)  # (Lq, Dv)
        else:
            Q_prime = Q0                                     # (Lq, Dv)

        # -- Step 6: Gated fusion ---------------------------------------
        parts = [Q_prime]
        if self.use_motion_branch:
            alpha = torch.sigmoid(self.motion_gate_logit)   # scalar ∈ (0,1)
            parts = [alpha * g] + parts
        V = torch.cat(parts, dim=0)

        # -- Step 7: Projector ------------------------------------------
        H_vis = self.projector(V)                           # (1+Lq, Dl)
        return H_vis

    # ------------------------------------------------------------------
    # Public: encode all videos in a batch
    # ------------------------------------------------------------------

    def get_video_features_remoscene(
        self,
        pixel_values_videos : torch.Tensor,
        video_grid_thw      : torch.LongTensor,
    ) -> List[torch.Tensor]:
        """
        Encode every video in the batch through the ReMoScene pipeline.

        Returns
        -------
        List[(Lq+1, Dl)]  — one tensor per video in the batch.
        """
        Z_all = self._encode_video_frozen(pixel_values_videos, video_grid_thw)

        results = []
        offset  = 0
        for i in range(video_grid_thw.shape[0]):
            T, H, W = int(video_grid_thw[i, 0]), int(video_grid_thw[i, 1]), int(video_grid_thw[i, 2])
            n_tok   = T * H * W
            Z_i     = Z_all[offset : offset + n_tok]    # (T*H*W, Dv)
            offset += n_tok
            results.append(self._compress_single_video(Z_i, T, H, W))

        return results  # list[(Lq+1, Dl)]

    # ------------------------------------------------------------------
    # Step 8 – replace video placeholder tokens in inputs_embeds
    # ------------------------------------------------------------------

    def _replace_video_tokens(
        self,
        input_ids         : torch.LongTensor,         # (B, L_orig)
        inputs_embeds     : torch.FloatTensor,        # (B, L_orig, Dl)
        attention_mask    : Optional[torch.Tensor],   # (B, L_orig) or None
        labels            : Optional[torch.LongTensor], # (B, L_orig) or None
        video_embeds_list : List[torch.Tensor],       # list[(Lq+1, Dl)]
        video_grid_thw    : torch.LongTensor,         # (num_videos, 3)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[torch.Tensor], torch.LongTensor]:
        """
        Replace each contiguous run of video placeholder tokens in
        ``inputs_embeds`` with the corresponding compressed ReMoScene embeddings,
        then pad / stack the batch to a uniform length.

        Also updates ``video_grid_thw`` to reflect the new token count
        (Lq+1 tokens per video encoded as [[Lq+1, s, s]] where s is the
        spatial_merge_size so that get_rope_index yields Lq+1 positions).

        Returns
        -------
        new_inputs_embeds, new_attention_mask, new_labels, new_video_grid_thw
        """
        B               = input_ids.shape[0]
        # video length is now dynamic (1 + Lq + num_chunks*N); read from embed
        vid_tok         = self.video_token_id

        new_emb_list    = []
        new_mask_list   = []
        new_lbl_list    = []

        vid_embed_idx = 0

        for b in range(B):
            ids_b = input_ids[b]
            emb_b = inputs_embeds[b]
            msk_b = attention_mask[b] if attention_mask is not None else None
            lbl_b = labels[b]         if labels         is not None else None

            # Find video token positions in this sample
            vid_pos = (ids_b == vid_tok).nonzero(as_tuple=True)[0]

            if len(vid_pos) == 0:
                # No video tokens in this sample: pass through unchanged
                new_emb_list.append(emb_b)
                if msk_b is not None: new_mask_list.append(msk_b)
                if lbl_b is not None: new_lbl_list.append(lbl_b)
                continue

            # Group consecutive positions into contiguous blocks
            blocks = []
            s = int(vid_pos[0])
            p = s
            for pos in vid_pos[1:]:
                pos = int(pos)
                if pos == p + 1:
                    p = pos
                else:
                    blocks.append((s, p + 1))
                    s = p = pos
            blocks.append((s, p + 1))  # (start_incl, end_excl)

            # Build new sequence for this sample
            emb_parts  = []
            msk_parts  = []
            lbl_parts  = []
            cursor     = 0

            for (vs, ve) in blocks:
                # ---- Text before this video block ----
                emb_parts.append(emb_b[cursor:vs])
                if msk_b is not None:
                    msk_parts.append(msk_b[cursor:vs])
                if lbl_b is not None:
                    lbl_parts.append(lbl_b[cursor:vs])

                # ---- Compressed video embedding (dynamic length, Dl) ----
                v_embed = video_embeds_list[vid_embed_idx].to(emb_b.dtype)
                vid_embed_idx += 1
                new_video_len = v_embed.shape[0]   # 1 + Lq + num_chunks*N
                emb_parts.append(v_embed)
                if msk_b is not None:
                    msk_parts.append(
                        torch.ones(new_video_len,
                                   device=msk_b.device, dtype=msk_b.dtype)
                    )
                if lbl_b is not None:
                    lbl_parts.append(
                        torch.full((new_video_len,), -100,
                                   device=lbl_b.device, dtype=lbl_b.dtype)
                    )
                cursor = ve

            # ---- Text after the last video block ----
            emb_parts.append(emb_b[cursor:])
            if msk_b is not None: msk_parts.append(msk_b[cursor:])
            if lbl_b is not None: lbl_parts.append(lbl_b[cursor:])

            new_emb_list.append(torch.cat(emb_parts, dim=0))
            if msk_b is not None:
                new_mask_list.append(torch.cat(msk_parts, dim=0))
            if lbl_b is not None:
                new_lbl_list.append(torch.cat(lbl_parts, dim=0))

        # ---- Pad batch to uniform length ----------------------------
        max_len = max(t.shape[0] for t in new_emb_list)
        Dl      = inputs_embeds.shape[-1]

        def _pad(lst, fill=0.0):
            out = []
            for t in lst:
                gap = max_len - t.shape[0]
                if gap > 0:
                    pad_shape = (gap,) + t.shape[1:]
                    t = torch.cat(
                        [t, t.new_full(pad_shape, fill)], dim=0
                    )
                out.append(t)
            return torch.stack(out, dim=0)

        new_inputs_embeds = _pad(new_emb_list, 0.0)
        new_attention_mask = (
            _pad(new_mask_list, 0).to(attention_mask.dtype)
            if new_mask_list else attention_mask
        )
        new_labels = (
            _pad(new_lbl_list, -100).long()
            if new_lbl_list else labels
        )

        # ---- Update video_grid_thw so get_rope_index yields correct positions
        # Each video i now has video_embeds_list[i].shape[0] tokens.
        # Encode as [[token_count, s, s]] where s = spatial_merge_size so that
        # get_rope_index produces token_count sequential positions.
        s = self.spatial_merge_size
        new_vg = video_grid_thw.new_full(video_grid_thw.shape, s)  # [[s, s, s], ...]
        for i, ve in enumerate(video_embeds_list):
            new_vg[i, 0] = ve.shape[0]                     # actual token count

        return new_inputs_embeds, new_attention_mask, new_labels, new_vg

    # ------------------------------------------------------------------
    # Step 9 – forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids            : Optional[torch.LongTensor]  = None,
        attention_mask       : Optional[torch.Tensor]      = None,
        position_ids         : Optional[torch.LongTensor]  = None,
        past_key_values                                     = None,
        inputs_embeds        : Optional[torch.FloatTensor] = None,
        labels               : Optional[torch.LongTensor]  = None,
        use_cache            : Optional[bool]               = None,
        output_attentions    : Optional[bool]               = None,
        output_hidden_states : Optional[bool]               = None,
        return_dict          : Optional[bool]               = None,
        pixel_values         : Optional[torch.Tensor]      = None,
        pixel_values_videos  : Optional[torch.FloatTensor] = None,
        image_grid_thw       : Optional[torch.LongTensor]  = None,
        video_grid_thw       : Optional[torch.LongTensor]  = None,
        rope_deltas          : Optional[torch.LongTensor]  = None,
        cache_position       : Optional[torch.LongTensor]  = None,
        **kwargs,
    ):
        """
        Identical signature to the wrapped backbone model's ``forward()`` method.

        Video inputs are processed through the ReMoScene pipeline.
        The frozen LLM receives the final ``inputs_embeds``.
        """
        from transformers.cache_utils import DynamicCache

        # ------------------------------------------------------------------
        # Build inputs_embeds
        # ------------------------------------------------------------------
        if inputs_embeds is None:
            embed         = self.base_model.model.get_input_embeddings()
            inputs_embeds = embed(input_ids)   # (B, L_orig, Dl)

            # ---- Static images (unchanged, use frozen base pipeline) ----
            if pixel_values is not None:
                with torch.no_grad():
                    img_embeds = self.base_model.model.get_image_features(
                        pixel_values, image_grid_thw
                    )
                img_mask = (
                    (input_ids == self.image_token_id)
                    .unsqueeze(-1).expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                img_embeds = img_embeds.to(inputs_embeds.device,
                                           inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(
                    img_mask, img_embeds
                )

            # ---- Videos: ReMoScene pipeline ----
            if pixel_values_videos is not None:
                all_video_embeds = self.get_video_features_remoscene(
                    pixel_values_videos, video_grid_thw
                )
                (inputs_embeds,
                 attention_mask,
                 labels,
                 video_grid_thw) = self._replace_video_tokens(
                    input_ids,
                    inputs_embeds,
                    attention_mask,
                    labels,
                    all_video_embeds,
                    video_grid_thw,
                )
                # input_ids is now inconsistent with the new length → clear it
                input_ids = None

        # ------------------------------------------------------------------
        # Position IDs
        # ------------------------------------------------------------------
        if position_ids is None:
            if input_ids is not None:
                # Still have consistent input_ids (no video): use 3-D RoPE
                # get_rope_index returns (3, B, L); take axis-0 (temporal/sequential)
                # because this model uses standard 1-D RoPE, not MRoPE.
                position_ids_3d, rope_deltas = (
                    self.base_model.model.get_rope_index(
                        input_ids, image_grid_thw, video_grid_thw,
                        attention_mask
                    )
                )
                position_ids = position_ids_3d[0]  # (B, L)
            else:
                # Video was replaced: sequential positions, shape (B, L).
                # The LLM rotary_emb expects 2-D (B, L) position_ids.
                B, L, _ = inputs_embeds.shape
                position_ids = torch.arange(
                    L, device=inputs_embeds.device
                ).unsqueeze(0).expand(B, -1)

        # ------------------------------------------------------------------
        # KV cache
        # ------------------------------------------------------------------
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            seen = (past_key_values.get_seq_length()
                    if past_key_values is not None else 0)
            cache_position = torch.arange(
                seen, seen + inputs_embeds.shape[1],
                device=inputs_embeds.device
            )

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        # ------------------------------------------------------------------
        # Frozen LLM forward
        # ------------------------------------------------------------------
        lm_out = self.base_model.model.language_model(
            input_ids            = None,
            position_ids         = position_ids,
            attention_mask       = attention_mask,
            past_key_values      = past_key_values,
            inputs_embeds        = inputs_embeds,
            use_cache            = use_cache,
            output_attentions    = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict          = True,
            cache_position       = cache_position,
        )

        hidden_states = lm_out[0]
        logits = self.base_model.lm_head(hidden_states)  # (B, L_new, vocab)

        # ------------------------------------------------------------------
        # Loss
        # ------------------------------------------------------------------
        loss = None
        if labels is not None:
            # Shift for auto-regressive language modelling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # ------------------------------------------------------------------
        # Return
        # ------------------------------------------------------------------
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss             = loss,
            logits           = logits,
            past_key_values  = lm_out.past_key_values,
            hidden_states    = lm_out.hidden_states,
            attentions       = lm_out.attentions,
        )

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def prepare_generation_inputs(
        self,
        input_ids           : torch.LongTensor,
        pixel_values_videos : Optional[torch.Tensor]    = None,
        video_grid_thw      : Optional[torch.LongTensor] = None,
        pixel_values        : Optional[torch.Tensor]    = None,
        image_grid_thw      : Optional[torch.LongTensor] = None,
        attention_mask      : Optional[torch.Tensor]    = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Build ``inputs_embeds`` and ``attention_mask`` for greedy/beam
        generation.  The ReMoScene+Motion pipeline runs once here; subsequent
        auto-regressive decode steps use the base LLM with KV cache.

        Returns
        -------
        inputs_embeds   : (B, L_new, Dl)
        attention_mask  : (B, L_new)  or None
        """
        with torch.no_grad():
            embed         = self.base_model.model.get_input_embeddings()
            inputs_embeds = embed(input_ids)

            # Static images
            if pixel_values is not None:
                img_embeds = self.base_model.model.get_image_features(
                    pixel_values, image_grid_thw
                )
                img_mask = (
                    (input_ids == self.image_token_id)
                    .unsqueeze(-1).expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                inputs_embeds = inputs_embeds.masked_scatter(
                    img_mask,
                    img_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                )

            # Videos
            if pixel_values_videos is not None:
                all_video_embeds = self.get_video_features_remoscene(
                    pixel_values_videos, video_grid_thw
                )
                inputs_embeds, attention_mask, _, _ = self._replace_video_tokens(
                    input_ids, inputs_embeds, attention_mask, None,
                    all_video_embeds, video_grid_thw,
                )

        return inputs_embeds, attention_mask

    def generate(
        self,
        input_ids           : torch.LongTensor,
        pixel_values_videos : Optional[torch.Tensor]    = None,
        video_grid_thw      : Optional[torch.LongTensor] = None,
        pixel_values        : Optional[torch.Tensor]    = None,
        image_grid_thw      : Optional[torch.LongTensor] = None,
        attention_mask      : Optional[torch.Tensor]    = None,
        **gen_kwargs,
    ) -> torch.LongTensor:
        """
        Convenience wrapper for auto-regressive generation.

        1. Runs the ReMoScene+Motion pipeline to build ``inputs_embeds``.
        2. Delegates to the base language model's ``generate`` (which uses
           KV-cache for all subsequent decode steps).

        Returns
        -------
        Generated token IDs (shape depends on gen_kwargs).
        """
        inputs_embeds, attention_mask = self.prepare_generation_inputs(
            input_ids,
            pixel_values_videos = pixel_values_videos,
            video_grid_thw      = video_grid_thw,
            pixel_values        = pixel_values,
            image_grid_thw      = image_grid_thw,
            attention_mask      = attention_mask,
        )

        # Pass precomputed embeddings to the full model's generate method.
        # The wrapped backbone model inherits GenerationMixin.
        # and accepts inputs_embeds directly (skipping vision re-encoding).
        return self.base_model.generate(
            inputs_embeds  = inputs_embeds,
            attention_mask = attention_mask,
            **gen_kwargs,
        )


# ============================================================================
# Factory helper
# ============================================================================

def build_remoscene_model(
    base_model_path : str,
    remoscene_cfg       : Optional[ReMoSceneConfig] = None,
    torch_dtype     : torch.dtype = torch.bfloat16,
    device_map      : str = "auto",
) -> ReMoSceneWrapper:
    """
    Load a multimodal backbone checkpoint and wrap it with the
    ReMoScene extension.

    Parameters
    ----------
    base_model_path : path to a HuggingFace-format model directory.
    remoscene_cfg       : ReMoScene hyper-parameters (uses defaults if None).
    torch_dtype     : compute dtype for the base model weights.
    device_map      : HuggingFace device-map string ("auto", "cuda:0", …).

    Returns
    -------
    ReMoSceneWrapper (trainable modules on CPU by default; call
    ``.to(device)`` or use a Trainer to move everything to GPU).
    """
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype = torch_dtype,
        device_map  = device_map,
    )
    wrapper = ReMoSceneWrapper(base_model, remoscene_cfg=remoscene_cfg)
    wrapper.set_trainable()
    return wrapper
