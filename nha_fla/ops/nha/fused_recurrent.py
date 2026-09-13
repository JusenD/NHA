# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

import os
import warnings
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from einops import rearrange

from fla.ops.common.fused_recurrent import fused_recurrent_bwd_kernel, fused_recurrent_fwd_kernel
from fla.ops.utils import chunk_global_cumsum
from fla.ops.utils.op import exp
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

try:
    from triton.language.extra.cuda import gdc_wait as _gdc_wait
    _HAS_GDC = True
except Exception:  # older Triton without PDL device intrinsics
    _HAS_GDC = False

    @triton.jit
    def _gdc_wait():
        pass


def _triton_supports_pdl() -> bool:
    try:
        from triton.backends.nvidia.compiler import CUDAOptions
        return 'launch_pdl' in CUDAOptions.__dataclass_fields__
    except Exception:
        return False


# Programmatic dependent launch master switch (default on; NHA_PDL=0 disables
# it for A/B measurement). Per-device sm_90 gating happens at launch time.
_PDL_ENV = os.environ.get("NHA_PDL", "1") != "0"
_PDL_LAUNCH = _triton_supports_pdl()


def _use_pdl(device: torch.device) -> bool:
    return (
        _HAS_GDC
        and _PDL_ENV
        and _PDL_LAUNCH
        and torch.cuda.get_device_capability(device)[0] >= 9
    )


@triton.jit
def fused_recurrent_nha_inference_kernel(
    q,
    k,
    v,
    s,
    g,
    o,
    hk0,
    hv0,
    hkt,
    hvt,
    scale,
    K: tl.constexpr,
    V: tl.constexpr,
    M: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr
):
    i_bh = tl.program_id(0)
    i_bg = i_bh // NG

    b_s = tl.load(s + i_bg * M + tl.arange(0, M)).to(tl.float32)
    b_g = tl.load(g + i_bg * M + tl.arange(0, M)).to(tl.float32)
    b_g = exp(b_g)

    b_ok = tl.zeros([M], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)

        p_hk0 = hk0 + i_bg * K * M + (o_k[None, :]) * M + tl.arange(0, M)[:, None]
        # [BK,]
        mask_k = o_k < K
        # [M, BK]
        mask_hk = (tl.arange(0, M) < M)[:, None] & mask_k[None, :]
        # [M, BK]
        b_hk = tl.load(p_hk0, mask=mask_hk, other=0.).to(tl.float32)
        # [BK,]
        b_q = tl.load(q + i_bh * K + o_k, mask=mask_k, other=0.).to(tl.float32) * scale
        b_k = tl.load(k + i_bg * K + o_k, mask=mask_k, other=0.).to(tl.float32)
        b_hk = b_hk * b_g[:, None] + b_k[None, :] * b_s[:, None]
        b_ok += tl.sum(b_hk * b_q[None, :], axis=1)

        if i_bh % NG == 0:
            p_hkt = hkt + i_bg * K * M + o_k[None, :] * M + tl.arange(0, M)[:, None]
            tl.store(p_hkt, b_hk.to(p_hkt.dtype.element_ty), mask=mask_hk)

    b_qv = tl.softmax(b_ok)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)

        p_hv0 = hv0 + i_bg * M * V + tl.arange(0, M)[None, :] * V + o_v[:, None]
        # [BV,]
        mask_v = o_v < V
        # [BV, M]
        mask_hv = mask_v[:, None] & (tl.arange(0, M) < M)[None, :]
        # [BV, M]
        b_hv = tl.load(p_hv0, mask=mask_hv, other=0).to(tl.float32)
        # [BV,]
        b_v = tl.load(v + i_bg * V + o_v, mask=mask_v, other=0).to(tl.float32)
        b_hv = b_hv * b_g[None, :] + b_s[None, :] * b_v[:, None]
        b_ov = tl.sum(b_hv * b_qv[None, :], axis=1)

        tl.store(o + i_bh * V + o_v, b_ov.to(o.dtype.element_ty), mask=mask_v)

        if i_bh % NG == 0:
            p_hvt = hvt + i_bg * M * V + tl.arange(0, M)[None, :] * V + o_v[:, None]
            tl.store(p_hvt, b_hv.to(p_hvt.dtype.element_ty), mask=mask_hv)

@triton.jit
def fused_recurrent_nha_inference_k_kernel(
    q,
    k,
    s,
    g,
    hk0,
    hkt,
    out_b_ok,
    scale,
    K: tl.constexpr,
    M: tl.constexpr,
    MS: tl.constexpr,
    BK: tl.constexpr,
    NG: tl.constexpr,
    OUTPUT_STATE: tl.constexpr
):
    i_bh = tl.program_id(0)
    i_bg = i_bh // NG

    o_m = tl.arange(0, MS)
    mask_m = o_m < M

    b_s = tl.load(s + i_bg * M + o_m, mask=mask_m, other=0.).to(tl.float32)
    b_g = tl.load(g + i_bg * M + o_m, mask=mask_m, other=0.).to(tl.float32)
    b_g = exp(b_g)

    b_ok = tl.zeros([MS], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)

        p_hk0 = hk0 + i_bg * K * M + (o_k[None, :]) * M + o_m[:, None]
        # [BK,]
        mask_k = o_k < K
        # [MS, BK]
        mask_hk = mask_m[:, None] & mask_k[None, :]
        # [MS, BK]
        b_hk = tl.load(p_hk0, mask=mask_hk, other=0.).to(tl.float32)
        # [BK,]
        b_q = tl.load(q + i_bh * K + o_k, mask=mask_k, other=0.).to(tl.float32) * scale
        b_k = tl.load(k + i_bg * K + o_k, mask=mask_k, other=0.).to(tl.float32)
        b_hk = b_hk * b_g[:, None] + b_k[None, :] * b_s[:, None]
        b_ok += tl.sum(b_hk * b_q[None, :], axis=1)

        if OUTPUT_STATE and i_bh % NG == 0:
            p_hkt = hkt + i_bg * K * M + o_k[None, :] * M + o_m[:, None]
            tl.store(p_hkt, b_hk.to(p_hkt.dtype.element_ty), mask=mask_hk)

    # per-query-head slot scores: every program owns row i_bh exactly once,
    # so all B*HQ rows are written with no cross-program race
    tl.store(out_b_ok + i_bh * M + o_m, b_ok.to(out_b_ok.dtype.element_ty), mask=mask_m)

@triton.jit
def fused_recurrent_nha_inference_v_kernel(
    qv,
    v,
    s,
    g,
    o,
    hv0,
    hvt,
    sqv,
    V: tl.constexpr,
    M: tl.constexpr,
    MS: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr,
    OUTPUT_STATE: tl.constexpr
):
    i_bh = tl.program_id(0)
    i_bg = i_bh // NG

    o_m = tl.arange(0, MS)
    mask_m = o_m < M

    b_s = tl.load(s + i_bg * M + o_m, mask=mask_m, other=0.).to(tl.float32)
    b_g = tl.load(g + i_bg * M + o_m, mask=mask_m, other=0.).to(tl.float32)
    b_g = exp(b_g)

    # per-query-head slot probabilities, matching the k-kernel's per-head store;
    # sqv is the row stride of qv (it may be a last-dim slice of the softmax output)
    b_qv = tl.load(qv + i_bh * sqv + o_m, mask=mask_m, other=0.).to(tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)

        p_hv0 = hv0 + i_bg * M * V + o_m[None, :] * V + o_v[:, None]
        # [BV,]
        mask_v = o_v < V
        # [BV, MS]
        mask_hv = mask_v[:, None] & mask_m[None, :]
        # [BV, MS]
        b_hv = tl.load(p_hv0, mask=mask_hv, other=0).to(tl.float32)
        # [BV,]
        b_v = tl.load(v + i_bg * V + o_v, mask=mask_v, other=0).to(tl.float32)
        b_hv = b_hv * b_g[None, :] + b_s[None, :] * b_v[:, None]
        b_ov = tl.sum(b_hv * b_qv[None, :], axis=1)

        tl.store(o + i_bh * V + o_v, b_ov.to(o.dtype.element_ty), mask=mask_v)

        if OUTPUT_STATE and i_bh % NG == 0:
            p_hvt = hvt + i_bg * M * V + o_m[None, :] * V + o_v[:, None]
            tl.store(p_hvt, b_hv.to(p_hvt.dtype.element_ty), mask=mask_hv)

def fused_recurrent_nha_inference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sliding_window: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    output_final_state: bool = False,
    scale: float = 1.,
    head_first: bool = False
) -> torch.Tensor:
    # T = 1
    if head_first:
        B, H, T, K, V, M = *k.shape, v.shape[-1], s.shape[-1]
    else:
        B, T, H, K, V, M = *k.shape, v.shape[-1], s.shape[-1]
    HQ = q.shape[1] if head_first else q.shape[2]
    BK, BV = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64)
    MS = triton.next_power_of_2(M)
    NG = HQ // H

    if initial_state != (None, None) and initial_state is not None:
        hk0, hv0 = initial_state
    else:
        hk0, hv0 = q.new_zeros(B, H, K, M, dtype=torch.float), q.new_zeros(B, H, M, V, dtype=torch.float)

    hkt, hvt = None, None
    if output_final_state:
        if NG == 1:
            hkt, hvt = hk0, hv0
        else:
            hkt, hvt = q.new_empty(B, H, K, M, dtype=torch.float), q.new_empty(B, H, M, V, dtype=torch.float)

    ok = v.new_empty(B, HQ, T, M) if head_first else v.new_empty(B, T, HQ, M)
    grid = (B * HQ,)
    fused_recurrent_nha_inference_k_kernel[grid](
        q,
        k,
        s,
        g,
        hk0,
        hkt,
        ok,
        scale=scale,
        K=K,
        M=M,
        MS=MS,
        BK=BK,
        NG=NG,
        OUTPUT_STATE=output_final_state,
    )

    combined = torch.cat([ok, sliding_window], dim=-1)
    combined_scores = combined.softmax(dim=-1, dtype=torch.float32)
    # a view into combined_scores: the v-kernel takes the row stride (sqv), so
    # no contiguous copy is needed here
    qv = combined_scores[..., :M]
    swa_score = combined_scores[..., M:].clone()

    o = v.new_empty(B, HQ, T, V) if head_first else v.new_empty(B, T, HQ, V)
    fused_recurrent_nha_inference_v_kernel[grid](
        qv,
        v,
        s,
        g,
        o,
        hv0,
        hvt,
        qv.stride(-2),
        V=V,
        M=M,
        MS=MS,
        BV=BV,
        NG=NG,
        OUTPUT_STATE=output_final_state,
    )

    return o, swa_score, (hkt, hvt)


@triton.jit
def fused_recurrent_nha_decode_k_kernel(
    q,
    sq,
    k,
    g,
    s,
    sk,
    hk0,
    hkt,
    qv,
    swa_p,
    cos,
    sin,
    c_sb,
    c_sw,
    scale,
    W,
    H: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    MS: tl.constexpr,
    BK: tl.constexpr,
    BW: tl.constexpr,
    HQ: tl.constexpr,
    NG: tl.constexpr,
    ROTARY: tl.constexpr,
    USE_S: tl.constexpr,
    GDC: tl.constexpr,
    OUTPUT_STATE: tl.constexpr
):
    # One program per (batch, query head).
    # Computes the GSA slot scores b_ok and the sliding-window scores b_sw,
    # then the combined softmax over [M | W] in registers, eliminating the
    # cat + softmax + clone round-trips through global memory.
    # ROTARY: apply the rotate-half embedding to the query/window keys
    # in-register (fp32) from the shared cos/sin tables instead of reading
    # materialized rope'd tensors.
    # USE_S: read the caller-computed slot residual s; otherwise recompute
    # s = 1 - exp(g) in fp32 from the log-gate.
    if GDC:
        # Programmatic dependent launch: wait for the immediately preceding
        # grid on the stream before any global read.
        _gdc_wait()
    i_bh = tl.program_id(0)
    i_bg = i_bh // NG
    i_b = i_bh // HQ
    i_h = i_bg % H

    o_m = tl.arange(0, MS)
    mask_m = o_m < M

    b_g = tl.load(g + i_bg * M + o_m, mask=mask_m, other=0.).to(tl.float32)
    b_g = exp(b_g)
    if USE_S:
        b_s = tl.load(s + i_bg * M + o_m, mask=mask_m, other=0.).to(tl.float32)
    else:
        b_s = 1. - b_g

    o_w = tl.arange(0, BW)
    m_w = o_w < W

    b_ok = tl.zeros([MS], dtype=tl.float32)
    b_sw = tl.zeros([BW], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)

        p_hk0 = hk0 + i_bg * K * M + (o_k[None, :]) * M + o_m[:, None]
        # [BK,]
        mask_k = o_k < K
        # [MS, BK]
        mask_hk = mask_m[:, None] & mask_k[None, :]
        # [MS, BK]
        b_hk = tl.load(p_hk0, mask=mask_hk, other=0.).to(tl.float32)
        # [BK,]
        b_q = tl.load(q + i_bh * K + o_k, mask=mask_k, other=0.).to(tl.float32) * scale
        b_k = tl.load(k + i_bg * K + o_k, mask=mask_k, other=0.).to(tl.float32)
        b_hk = b_hk * b_g[:, None] + b_k[None, :] * b_s[:, None]
        b_ok += tl.sum(b_hk * b_q[None, :], axis=1)

        # sliding-window scores use the rope'd query/key
        if ROTARY:
            # rotate-half pairing: column j pairs with j+K/2 (sign -1) for
            # j < K/2 and with j-K/2 (sign +1) for j >= K/2
            o_rk = tl.where(o_k < K // 2, o_k + K // 2, o_k - K // 2)
            b_sign = tl.where(o_k < K // 2, -1., 1.)
            # the query position is the last row of the window cos/sin
            b_cq = tl.load(cos + i_b * c_sb + (W - 1) * c_sw + o_k, mask=mask_k, other=0.).to(tl.float32)
            b_rq = tl.load(sin + i_b * c_sb + (W - 1) * c_sw + o_k, mask=mask_k, other=0.).to(tl.float32)
            b_qp = tl.load(q + i_bh * K + o_rk, mask=mask_k, other=0.).to(tl.float32)
            b_sq = (tl.load(q + i_bh * K + o_k, mask=mask_k, other=0.).to(tl.float32) * b_cq
                    + b_qp * b_sign * b_rq) * scale
            # sk is [B, W, H, K] (kv-head layout): element (i_b, w, i_h, k)
            p_sk = sk + ((i_b * W + o_w[:, None]) * H + i_h) * K + o_k[None, :]
            b_sk = tl.load(p_sk, mask=m_w[:, None] & mask_k[None, :], other=0.).to(tl.float32)
            p_skp = sk + ((i_b * W + o_w[:, None]) * H + i_h) * K + o_rk[None, :]
            b_skp = tl.load(p_skp, mask=m_w[:, None] & mask_k[None, :], other=0.).to(tl.float32)
            b_cw = tl.load(cos + i_b * c_sb + o_w[:, None] * c_sw + o_k[None, :],
                           mask=m_w[:, None] & mask_k[None, :], other=0.).to(tl.float32)
            b_rw = tl.load(sin + i_b * c_sb + o_w[:, None] * c_sw + o_k[None, :],
                           mask=m_w[:, None] & mask_k[None, :], other=0.).to(tl.float32)
            b_sk = b_sk * b_cw + b_skp * b_sign[None, :] * b_rw
        else:
            b_sq = tl.load(sq + i_bh * K + o_k, mask=mask_k, other=0.).to(tl.float32) * scale
            # sk is [B, W, H, K] (kv-head layout): element (i_b, w, i_h, k)
            p_sk = sk + ((i_b * W + o_w[:, None]) * H + i_h) * K + o_k[None, :]
            b_sk = tl.load(p_sk, mask=m_w[:, None] & mask_k[None, :], other=0.).to(tl.float32)
        b_sw += tl.sum(b_sk * b_sq[None, :], axis=1)

        if OUTPUT_STATE and i_bh % NG == 0:
            p_hkt = hkt + i_bg * K * M + o_k[None, :] * M + o_m[:, None]
            tl.store(p_hkt, b_hk.to(p_hkt.dtype.element_ty), mask=mask_hk)

    # mask the padded slot lanes so they cannot win the max or gain softmax mass
    b_ok = tl.where(mask_m, b_ok, float('-inf'))
    b_sw = tl.where(m_w, b_sw, float('-inf'))
    b_max = tl.maximum(tl.max(b_ok, 0), tl.max(b_sw, 0))
    b_eok = exp(b_ok - b_max)
    b_esw = exp(b_sw - b_max)
    b_den = tl.sum(b_eok, 0) + tl.sum(b_esw, 0)
    b_qv = b_eok / b_den
    b_swp = b_esw / b_den

    tl.store(qv + i_bh * M + o_m, b_qv.to(qv.dtype.element_ty), mask=mask_m)
    tl.store(swa_p + i_bh * BW + o_w, b_swp.to(swa_p.dtype.element_ty), mask=m_w)


@triton.jit
def fused_recurrent_nha_decode_v_kernel(
    qv,
    swa_p,
    v,
    g,
    s,
    vw,
    o,
    hv0,
    hvt,
    W,
    H: tl.constexpr,
    V: tl.constexpr,
    M: tl.constexpr,
    MS: tl.constexpr,
    BV: tl.constexpr,
    BW: tl.constexpr,
    HQ: tl.constexpr,
    NG: tl.constexpr,
    USE_S: tl.constexpr,
    GDC: tl.constexpr,
    OUTPUT_STATE: tl.constexpr
):
    # One program per (batch, query head).
    # o = hv_readout(qv) + swa_p @ vw, fused into a single store.
    if GDC:
        _gdc_wait()
    i_bh = tl.program_id(0)
    i_bg = i_bh // NG
    i_b = i_bh // HQ
    i_h = i_bg % H

    o_m = tl.arange(0, MS)
    mask_m = o_m < M

    b_g = tl.load(g + i_bg * M + o_m, mask=mask_m, other=0.).to(tl.float32)
    b_g = exp(b_g)
    if USE_S:
        b_s = tl.load(s + i_bg * M + o_m, mask=mask_m, other=0.).to(tl.float32)
    else:
        b_s = 1. - b_g

    b_qv = tl.load(qv + i_bh * M + o_m, mask=mask_m, other=0.).to(tl.float32)

    o_w = tl.arange(0, BW)
    m_w = o_w < W
    b_swp = tl.load(swa_p + i_bh * BW + o_w, mask=m_w, other=0.).to(tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)

        p_hv0 = hv0 + i_bg * M * V + o_m[None, :] * V + o_v[:, None]
        # [BV,]
        mask_v = o_v < V
        # [BV, MS]
        mask_hv = mask_v[:, None] & mask_m[None, :]
        # [BV, MS]
        b_hv = tl.load(p_hv0, mask=mask_hv, other=0).to(tl.float32)
        # [BV,]
        b_v = tl.load(v + i_bg * V + o_v, mask=mask_v, other=0).to(tl.float32)
        b_hv = b_hv * b_g[None, :] + b_s[None, :] * b_v[:, None]
        b_ov = tl.sum(b_hv * b_qv[None, :], axis=1)

        # vw is [B, W, H, V] (kv-head layout): element (i_b, w, i_h, v)
        p_vw = vw + ((i_b * W + o_w[:, None]) * H + i_h) * V + o_v[None, :]
        b_vw = tl.load(p_vw, mask=m_w[:, None] & mask_v[None, :], other=0.).to(tl.float32)
        b_ov += tl.sum(b_vw * b_swp[:, None], axis=0)

        tl.store(o + i_bh * V + o_v, b_ov.to(o.dtype.element_ty), mask=mask_v)

        if OUTPUT_STATE and i_bh % NG == 0:
            p_hvt = hvt + i_bg * M * V + o_m[None, :] * V + o_v[:, None]
            tl.store(p_hvt, b_hv.to(p_hvt.dtype.element_ty), mask=mask_hv)


def fused_recurrent_nha_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    sw_k: torch.Tensor,
    sw_v: torch.Tensor,
    sq: Optional[torch.Tensor] = None,
    s: Optional[torch.Tensor] = None,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    output_final_state: bool = False,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Single-token fused NHA decode step with GQA support.

    Two kernel launches: the k-kernel computes the GSA slot scores (raw `q`)
    and the sliding-window scores (rope'd query/keys), applies the combined
    softmax over `[M slots | W window]` in registers and writes the per-head
    slot/window probabilities; the v-kernel reads out the updated value state
    plus the window values in one fused store.

    Args:
        q: raw (un-rotated) queries of shape `[B, 1, HQ, K]`; used for the GSA
            slot scores and, when `cos`/`sin` are given, rotated in-kernel for
            the sliding-window scores.
        k: popped (oldest window) keys of shape `[B, 1, H, K]`.
        v: popped (oldest window) values of shape `[B, 1, H, V]`.
        g: log forget gates of the popped token of shape `[B, 1, H, M]`.
        sw_k: sliding-window keys of shape `[B, W, H, K]` (per-kv-head layout,
            window ascending in time; un-rotated when `cos`/`sin` are given,
            pre-rotated otherwise).
        sw_v: sliding-window values of shape `[B, W, H, V]`.
        sq: pre-rotated SWA queries of shape `[B, 1, HQ, K]`; required when
            `cos`/`sin` are not given.
        s: optional pre-computed slot residual `1 - exp(g)` of shape
            `[B, 1, H, M]`. When omitted, the kernels recompute it from the
            log-gate in fp32.
        cos, sin: optional rotary tables of shape `[B | 1, W2 >= W, K]`
            (duplicated halves, as produced by the HF rotary embedding); the
            last `W` rows are used and row `W-1` is the query position. When
            given, `q`/`sw_k` must be un-rotated and the kernels apply the
            rotate-half embedding in fp32.
        initial_state: optional `(hk0, hv0)` of shapes `[B, H, K, M]`/`[B, H, M, V]`.
        output_final_state: whether to emit the updated state.
        scale: score scale, defaults to `K ** -0.5`.

    Returns:
        o of shape `[B, 1, HQ, V]` and the `(hkt, hvt)` final-state tuple
        (`(None, None)` when `output_final_state=False`).
    """
    B, T, HQ, K = q.shape
    H = sw_k.shape[2]
    V = v.shape[-1]
    M = g.shape[-1]
    W = sw_k.shape[1]
    assert T == 1 and k.shape[1] == 1 and g.shape[1] == 1
    assert k.shape[2] == H and v.shape[2] == H and g.shape[2] == H, \
        f"popped-entry head count must match the kv-head count ({H}) of the window tensors"
    assert sw_v.shape[1] == W and sw_v.shape[2] == H
    assert HQ % H == 0
    assert K >= 16 and V >= 16
    NG = HQ // H
    if scale is None:
        scale = K ** -0.5

    rotary = cos is not None
    if rotary:
        assert sq is None, "pass either sq (pre-rotated) or cos/sin (in-kernel rotary), not both"
        assert sin is not None and cos.shape == sin.shape
        assert cos.shape[0] in (1, B) and cos.shape[1] >= W and cos.shape[2] == K, \
            f"rotary cos/sin must be [1|{B}, W2>={W}, K={K}], got {tuple(cos.shape)}"
        assert K % 2 == 0, "in-kernel rotate-half requires an even head dim"
        if cos.shape[1] != W:
            cos = cos[:, -W:]
            sin = sin[:, -W:]
        assert cos.stride(-1) == 1 and sin.stride(-1) == 1
        # batch-1 cos/sin (shared positions across the batch) broadcasts
        c_sb = cos.stride(0) if cos.shape[0] == B else 0
        c_sw = cos.stride(1)
        assert (sin.stride(0) if sin.shape[0] == B else 0) == c_sb and sin.stride(1) == c_sw
    else:
        assert sq is not None and sq.shape == q.shape
        c_sb, c_sw = 0, 0

    use_s = s is not None
    if use_s:
        assert s.shape == g.shape

    q = q.reshape(B, HQ, K)
    k = k.reshape(B, H, K)
    v = v.reshape(B, H, V)
    g = g.reshape(B, H, M)
    sq = q if sq is None else sq.reshape(B, HQ, K)
    s = g if s is None else s.reshape(B, H, M)
    # reshape of a strided input may return a non-contiguous view (e.g. cache
    # slices); the kernels index all of these as flat [B*, dim] matrices
    q, sq, k, v, g, s, sw_k, sw_v = (
        x.contiguous() for x in (q, sq, k, v, g, s, sw_k, sw_v)
    )

    if initial_state != (None, None) and initial_state is not None:
        hk0, hv0 = initial_state
    else:
        hk0 = q.new_zeros(B, H, K, M, dtype=torch.float)
        hv0 = q.new_zeros(B, H, M, V, dtype=torch.float)

    hkt, hvt = None, None
    if output_final_state:
        if NG == 1:
            hkt, hvt = hk0, hv0
        else:
            hkt = q.new_empty(B, H, K, M, dtype=torch.float)
            hvt = q.new_empty(B, H, M, V, dtype=torch.float)

    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    MS = triton.next_power_of_2(M)
    BW = max(triton.next_power_of_2(W), 1)

    # PDL (sm_90+ only): compile the grid-dependency wait in and let launches
    # carry the programmatic-stream-serialization attribute.
    use_gdc = _use_pdl(q.device)

    qv = q.new_empty(B, HQ, M, dtype=torch.float)
    swa_p = q.new_empty(B, HQ, BW, dtype=torch.float)
    grid = (B * HQ,)
    launch_kwargs = {'launch_pdl': use_gdc} if _PDL_LAUNCH else {}
    fused_recurrent_nha_decode_k_kernel[grid](
        q, sq, k, g, s, sw_k, hk0, hkt, qv, swa_p,
        cos if rotary else q, sin if rotary else q, c_sb, c_sw,
        scale=scale,
        W=W,
        H=H, K=K, M=M, MS=MS, BK=BK, BW=BW, HQ=HQ, NG=NG,
        ROTARY=rotary,
        USE_S=use_s,
        GDC=use_gdc,
        OUTPUT_STATE=output_final_state,
        **launch_kwargs,
    )

    o = q.new_empty(B, HQ, V)
    fused_recurrent_nha_decode_v_kernel[grid](
        qv, swa_p, v, g, s, sw_v, o, hv0, hvt,
        W=W,
        H=H, V=V, M=M, MS=MS, BV=BV, BW=BW, HQ=HQ, NG=NG,
        USE_S=use_s,
        GDC=use_gdc,
        OUTPUT_STATE=output_final_state,
        **launch_kwargs,
    )

    return o.view(B, 1, HQ, V), (hkt, hvt)


def fused_recurrent_nha_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    output_final_state: bool = False,
    scale: float = 1.,
    reverse: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
    if head_first:
        B, H, T, K, V, M = *k.shape, v.shape[-1], s.shape[-1]
    else:
        B, T, H, K, V, M = *k.shape, v.shape[-1], s.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    HQ = q.shape[1] if head_first else q.shape[2]
    if HQ != H:
        raise ValueError("GQA not supported yet.")

    BK, BV, BM = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64), min(M, 64)
    NK, NV, NM = triton.cdiv(K, BK), triton.cdiv(V, BV), triton.cdiv(M, BM)

    hk0, hv0 = None, None
    if initial_state != (None, None) and initial_state is not None:
        hk0, hv0 = initial_state
    hkt, hvt = None, None
    if output_final_state:
        hkt, hvt = q.new_empty(N, H, K, M, dtype=torch.float), q.new_empty(N, H, M, V, dtype=torch.float)

    ok = q.new_empty(NK, *s.shape, dtype=torch.float)
    gk, gv = None, g
    grid = (NM, NK, N * H)
    fused_recurrent_fwd_kernel[grid](
        q=q,
        k=k,
        v=s,
        g=None,
        gk=gk,
        gv=gv,
        o=ok,
        h0=hk0,
        ht=hkt,
        cu_seqlens=cu_seqlens,
        scale=scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=M,
        BK=BK,
        BV=BM,
        USE_G=False,
        USE_GK=False,
        USE_GV=True,
        REVERSE=reverse
    )
    ok = ok.sum(0)

    qv = ok.softmax(-1, dtype=torch.float)
    ov = q.new_empty(NM, *v.shape, dtype=torch.float)
    gk, gv = g, None
    grid = (NV, NM, N * H)
    fused_recurrent_fwd_kernel[grid](
        q=qv,
        k=s,
        v=v,
        g=None,
        gk=gk,
        gv=gv,
        o=ov,
        h0=hv0,
        ht=hvt,
        cu_seqlens=cu_seqlens,
        scale=1.,
        B=B,
        T=T,
        H=H,
        K=M,
        V=V,
        BK=BM,
        BV=BV,
        USE_G=False,
        USE_GK=True,
        USE_GV=False,
        REVERSE=reverse,
    )
    ov = ov.sum(0)
    return ok, hkt, qv, ov, hvt


def fused_recurrent_nha_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    qv: torch.Tensor,
    hk0: Optional[torch.Tensor] = None,
    hv0: Optional[torch.Tensor] = None,
    ok: Optional[torch.Tensor] = None,
    do: Optional[torch.Tensor] = None,
    dhkt: Optional[torch.Tensor] = None,
    dhvt: Optional[torch.Tensor] = None,
    scale: float = 1.,
    reverse: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor]:
    B, T, H, K, V, M = *q.shape, v.shape[-1], s.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    BK, BV, BM = min(K, 64), min(V, 64), min(M, 64)
    NK, NV, NM = triton.cdiv(K, BK), triton.cdiv(V, BV), triton.cdiv(M, BM)

    dqv = q.new_empty(NV, B, T, H, M, dtype=torch.float)
    dsv = q.new_empty(NV, B, T, H, M, dtype=torch.float)
    dv = q.new_empty(NM, B, T, H, V, dtype=torch.float)
    dhk0 = torch.empty_like(hk0)if hk0 is not None else None
    dhv0 = torch.empty_like(hv0)if hv0 is not None else None

    gk, gv = g, None
    grid = (NV, NM, N * H)
    fused_recurrent_bwd_kernel[grid](
        q=qv,
        k=s,
        v=v,
        g=None,
        gk=gk,
        gv=gv,
        h0=hv0,
        do=do,
        dq=dqv,
        dk=dsv,
        dv=dv,
        dht=dhvt,
        dh0=dhv0,
        cu_seqlens=cu_seqlens,
        scale=1.,
        B=B,
        T=T,
        H=H,
        K=M,
        V=V,
        BK=BM,
        BV=BV,
        USE_G=False,
        USE_GK=True,
        USE_GV=False,
        REVERSE=reverse,
    )
    dqv = dqv.sum(0)
    dsv = dsv.sum(0)
    dv = dv.sum(0)
    dgk = chunk_global_cumsum(dqv * qv.float() - dsv * s.float(), reverse=not reverse, cu_seqlens=cu_seqlens)

    dok = qv * (dqv - (qv * dqv).sum(-1, True))
    dq = q.new_empty(NM, B, T, H, K, dtype=torch.float)
    dk = q.new_empty(NM, B, T, H, K, dtype=torch.float)
    dsk = q.new_empty(NK, B, T, H, M, dtype=torch.float)
    gk, gv = None, g
    grid = (NM, NK, N * H)
    fused_recurrent_bwd_kernel[grid](
        q=q,
        k=k,
        v=s,
        g=None,
        gk=gk,
        gv=gv,
        h0=hk0,
        do=dok,
        dq=dq,
        dk=dk,
        dv=dsk,
        dht=dhkt,
        dh0=dhk0,
        cu_seqlens=cu_seqlens,
        scale=scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=M,
        BK=BK,
        BV=BM,
        USE_G=False,
        USE_GK=False,
        USE_GV=True,
        REVERSE=reverse,
    )
    dq = dq.sum(0)
    dk = dk.sum(0)
    dsk = dsk.sum(0)

    dgv = chunk_global_cumsum(dok.float() * ok.float() - dsk * s.float(), reverse=not reverse, cu_seqlens=cu_seqlens)

    ds = dsk.add_(dsv)
    dg = dgk.add_(dgv)

    return dq, dk, dv, ds, dg, dhk0, dhv0


class FusedRecurrentNHAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sliding_window: torch.Tensor,
        s: torch.Tensor,
        g: torch.Tensor,
        scale: Optional[float] = None,
        hk0: Optional[torch.Tensor] = None,
        hv0: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        reverse: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        T = q.shape[1]
        assert(T == 1 and not q.requires_grad)
        o, swa_score, (hkt, hvt) = fused_recurrent_nha_inference(
            q=q,
            k=k,
            v=v,
            sliding_window=sliding_window,
            s=s,
            g=g,
            initial_state=(hk0, hv0),
            output_final_state=output_final_state,
            scale=scale,
        )
        return o, swa_score, hkt, hvt

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dhkt=None, dhvt=None):
        q, k, v, s, g, qv, hk0, hv0, ok = ctx.saved_tensors
        scale = ctx.scale
        reverse = ctx.reverse
        cu_seqlens = ctx.cu_seqlens

        # not supported yet.
        if dhkt is not None or dhvt is not None:
            if g is not None:
                assert g.requires_grad is False, "Cannot load final state gradient and use gates at the same time"
        dq, dk, dv, ds, dg, dhk0, dhv0 = fused_recurrent_nha_bwd(
            q=q,
            k=k,
            v=v,
            s=s,
            g=g,
            qv=qv,
            hk0=hk0,
            hv0=hv0,
            ok=ok,
            do=do,
            dhkt=dhkt,
            dhvt=dhvt,
            scale=scale,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
        )
        return dq.to(q), dk.to(k), dv.to(v), ds.to(s), dg.to(g), None, dhk0, dhv0, None, None, None


def fused_recurrent_nha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sliding_window: torch.Tensor,
    s: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[int] = None,
    initial_state: Optional[Tuple[torch.Tensor]] = None,
    output_final_state: Optional[bool] = False,
    reverse: Optional[bool] = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        s (torch.Tensor):
            slot representations of shape `[B, T, H, M]` if `head_first=False` else `[B, H, T, M]`.
        g (torch.Tensor):
            Forget gates of shape `[B, H, T, M]` applied to keys.
        scale (Optional[int]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[Tuple[torch.Tensor]]):
            Initial state tuple having tensors of shape `[N, H, K, M]` and `[N, H, M, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]` and `[N, H, M, V]`.
            Default: `False`.
        reverse (Optional[bool]):
            If `True`, process the state passing in reverse order. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        final_state (Tuple[torch.Tensor]):
            Final state tuple having tensors of shape `[N, H, K, M]` and `[N, H, M, V]`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.nha import fused_recurrent_nha
        # inputs with equal lengths
        >>> B, T, H, K, V, M = 4, 2048, 4, 512, 512, 64
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> s = torch.randn(B, T, H, M, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, H, M, device='cuda'))
        >>> h0 = (torch.randn(B, H, K, M, device='cuda'), torch.randn(B, H, M, V, device='cuda'))
        >>> o, (hk, hv) = fused_recurrent_nha(
            q, k, v, s, g,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, s, g = map(lambda x: rearrange(x, 'b t h d -> 1 (b t) h d'), (q, k, v, s, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, (hk_var, hv_var) = fused_recurrent_nha(
            q, k, v, s, g,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
        >>> assert o.allclose(o_var.view(o.shape))
        >>> assert hk.allclose(hk_var)
        >>> assert hv.allclose(hv_var)
    """
    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
        q, k, v, s, g = map(lambda x: rearrange(x, 'b h t ... -> b t h ...'), (q, k, v, s, g))
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state[0].shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state[0].shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if initial_state is None:
        initial_state = (None, None)
    o, sliding_window_prob, *final_state = FusedRecurrentNHAFunction.apply(
        q,
        k,
        v,
        sliding_window,
        s,
        g,
        scale,
        *initial_state,
        output_final_state,
        reverse,
        cu_seqlens,
    )
    if head_first:
        o = rearrange(o, 'b t h ... -> b h t ...')
    return o, sliding_window_prob, final_state
