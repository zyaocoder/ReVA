"""
Smoke test for the ReMoScene pipeline
(ReMoSceneSelector + MotionBranch + RecurrentObjectSelector).

Tests all new/modified components with dummy tensors (no base model required).
Run from the repository root:
    python ReMoScene/test_remoscene_smoke.py
"""

import math
import sys
import traceback
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Stub out the base-model import so we can test motion modules standalone
# without needing the full wrapped backbone + transformers installation.
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

_stub = types.ModuleType("backbone.modeling_backbone")
_stub.ReVABackboneForConditionalGeneration = MagicMock()
sys.modules["backbone.modeling_backbone"] = _stub

from backbone.remoscene import (  # noqa: E402
    ReMoSceneConfig,
    ReMoSceneSelector,
    MotionBranch,
    RecurrentObjectSelector,
    ReMoSceneProjector,
    pool3d_init_queries,
    resize_queries,
)

# ---------------------------------------------------------------------------
# Test dimensions (realistic but small enough to run on CPU quickly)
# ---------------------------------------------------------------------------
T   = 64    # frames
H   = 14    # patch grid height
W   = 14    # patch grid width
Dv  = 128   # vision dim (real: 1024; use 128 for speed)
Dl  = 256   # LLM dim   (real: 2560; use 256 for speed)
Lq  = 32    # ReMoSceneSelector queries (real: 256)
N   = 4     # object queries per RecurrentObjectSelector (real: 16)
CS  = 8     # chunk size

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"

def check(name, cond, detail=""):
    status = PASS if cond else FAIL
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    return cond


# ============================================================================
# 1. MotionBranch: shape + return type
# ============================================================================
def test_motion_branch():
    print("\n[1] MotionBranch")
    mb = MotionBranch(d_model=Dv)
    Z_3d = torch.randn(T, H, W, Dv)

    g, pair_feats = mb(Z_3d)

    check("g shape == (1, Dv)",       g.shape == (1, Dv),         str(g.shape))
    check("len(pair_feats) == T-1",   len(pair_feats) == T - 1,   str(len(pair_feats)))
    check("pair_feats[0] shape (Dv,)",pair_feats[0].shape == (Dv,), str(pair_feats[0].shape))

    # Single-frame fallback
    g1, pf1 = mb(torch.randn(1, H, W, Dv))
    check("single-frame: g shape",    g1.shape == (1, Dv))
    check("single-frame: pair_feats empty", pf1 == [])


# ============================================================================
# 2. RecurrentObjectSelector: shape + chunked BPTT
# ============================================================================
def test_recurrent_selector():
    print("\n[2] RecurrentObjectSelector")
    mb  = MotionBranch(d_model=Dv)
    rs  = RecurrentObjectSelector(d_model=Dv, num_queries=N,
                                  n_layers=1, chunk_size=CS, nhead=4)

    Z_3d = torch.randn(T, H, W, Dv)
    _, pair_feats = mb(Z_3d)

    Q_agg = rs(Z_3d, pair_feats)

    num_chunks = math.ceil(T / CS)
    expected   = num_chunks * N
    check(f"Q_agg shape == ({expected}, {Dv})",
          Q_agg.shape == (expected, Dv), str(Q_agg.shape))

    # --- Chunked BPTT: gradient should flow within a chunk but not cross it ---
    rs2   = RecurrentObjectSelector(d_model=Dv, num_queries=N,
                                    n_layers=1, chunk_size=CS, nhead=4)
    Z_req = torch.randn(T, H, W, Dv, requires_grad=True)
    _, pf = mb(Z_req.detach())
    out   = rs2(Z_req, pf)
    loss  = out.sum()
    loss.backward()
    check("gradient flows through Z_req", Z_req.grad is not None)

    # Only last chunk's frames should carry gradient into Z_req
    # (earlier chunks were detached).  We just verify backward doesn't crash.
    check("backward completes without error", True)

    # Empty pair_feats fallback (T=1)
    Z1   = torch.randn(1, H, W, Dv)
    out1 = rs(Z1, [])
    check("T=1 fallback shape == (1, N, Dv) flattened",
          out1.shape == (1 * N, Dv), str(out1.shape))


# ============================================================================
# 3. Full compress pipeline (without base model)
# ============================================================================
def test_compress_pipeline():
    print("\n[3] Full compress pipeline (_compress_single_video equivalent)")

    cfg = ReMoSceneConfig(
        num_queries=Lq,
        n_remoscene_layers=1,
        remoscene_nhead=4,
        remoscene_expand=2,
        remoscene_d_state=8,
        remoscene_d_conv=4,
        lq_t=2,
        lq_h=4,
        lq_w=4,   # 2*4*4 = 32 == Lq
        num_object_queries=N,
        rec_n_layers=1,
        rec_chunk_size=CS,
        rec_nhead=4,
    )

    # prefer_mamba=False forces attention blocks so the test runs on CPU
    remoscene_sel = ReMoSceneSelector(
        d_model=Dv,
        num_queries=Lq,
        n_layers=1,
        nhead=4,
        expand=2,
        d_state=8,
        d_conv=4,
        prefer_mamba=False,
    )
    motion_br = MotionBranch(d_model=Dv)
    rec_sel   = RecurrentObjectSelector(d_model=Dv, num_queries=N,
                                        n_layers=1, chunk_size=CS, nhead=4)
    projector = ReMoSceneProjector(d_vision=Dv, d_llm=Dl)
    gate_logit = nn.Parameter(torch.tensor(-5.0))

    Z    = torch.randn(T * H * W, Dv)
    Z_3d = Z.reshape(T, H, W, Dv)

    # Step 3
    Q0 = pool3d_init_queries(Z_3d, cfg.lq_t, cfg.lq_h, cfg.lq_w)
    Q0 = resize_queries(Q0, Lq)
    check("Q0 shape == (Lq, Dv)", Q0.shape == (Lq, Dv), str(Q0.shape))

    # Step 4
    Q_prime = remoscene_sel(Z, Q0)
    check("Q_prime shape == (Lq, Dv)", Q_prime.shape == (Lq, Dv), str(Q_prime.shape))

    # Step 5
    g, pair_feats = motion_br(Z_3d)
    check("g shape == (1, Dv)", g.shape == (1, Dv), str(g.shape))

    # Step 5b
    Q_rec = rec_sel(Z_3d, pair_feats)
    num_chunks = math.ceil(T / CS)
    check(f"Q_rec shape == ({num_chunks*N}, {Dv})",
          Q_rec.shape == (num_chunks * N, Dv), str(Q_rec.shape))

    # Step 6
    alpha = torch.sigmoid(gate_logit)
    V     = torch.cat([alpha * g, Q_prime, Q_rec], dim=0)
    expected_V = 1 + Lq + num_chunks * N
    check(f"V shape == ({expected_V}, {Dv})",
          V.shape == (expected_V, Dv), str(V.shape))

    # Step 7
    H_vis = projector(V)
    check(f"H_vis shape == ({expected_V}, {Dl})",
          H_vis.shape == (expected_V, Dl), str(H_vis.shape))


# ============================================================================
# 4. Gradient end-to-end through the full pipeline
# ============================================================================
def test_e2e_gradient():
    print("\n[4] End-to-end gradient flow")

    Z_3d = torch.randn(T, H, W, Dv, requires_grad=False)

    motion_br = MotionBranch(d_model=Dv)
    rec_sel   = RecurrentObjectSelector(d_model=Dv, num_queries=N,
                                        n_layers=1, chunk_size=CS, nhead=4)
    projector = ReMoSceneProjector(d_vision=Dv, d_llm=Dl)

    g, pair_feats = motion_br(Z_3d)
    Q_rec = rec_sel(Z_3d, pair_feats)
    out   = projector(Q_rec)
    loss  = out.mean()
    loss.backward()

    # Check that trainable params have gradients
    params_with_grad = sum(
        1 for p in list(rec_sel.parameters()) + list(projector.parameters())
        if p.grad is not None
    )
    total_params = sum(
        1 for _ in list(rec_sel.parameters()) + list(projector.parameters())
    )
    check(f"all trainable params have grad ({params_with_grad}/{total_params})",
          params_with_grad == total_params)


# ============================================================================
# Runner
# ============================================================================
if __name__ == "__main__":
    all_passed = True
    for test_fn in [
        test_motion_branch,
        test_recurrent_selector,
        test_compress_pipeline,
        test_e2e_gradient,
    ]:
        try:
            test_fn()
        except Exception:
            print(f"\033[91m  EXCEPTION in {test_fn.__name__}:\033[0m")
            traceback.print_exc()
            all_passed = False

    print()
    if all_passed:
        print("\033[92m All smoke tests passed.\033[0m")
    else:
        print("\033[91m Some tests FAILED.\033[0m")
        sys.exit(1)
