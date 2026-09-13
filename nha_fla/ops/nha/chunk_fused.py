# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# Fused flash-style NHA prefill (inference path).
#
# One fused state pass (GQA-native, both GLA states + chunk-local gate cumsum
# in a single sweep) plus one fused output pass (per chunk/head: k-path scores,
# sliding-window scores, in-register mixing softmax, v-path output) so the
# baseline chain's HBM intermediates (A, ok, sliding_window, mix_ok/mix_p,
# qv/swa_p slices, fp32 cumsum and per-chunk states at expanded head count)
# never touch HBM.
#
# Numerics mirror the baseline kernel chain step for step (same bf16 rounding
# points, fp32 accumulators, sub-chunk decay references and fast_expf) so the
# output matches the pristine chunk_nha within bf16 noise.

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp


@triton.jit(do_not_specialize=['T'])
def chunk_nha_fused_state_kernel(
    k,
    s,
    v,
    g,
    hk,
    hv,
    gc,
    hk0,
    hv0,
    hkt,
    hvt,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    M: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BM: tl.constexpr,
    USE_H0: tl.constexpr,
    STORE_HT: tl.constexpr,
):
    i_m, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    o_m = i_m * BM + tl.arange(0, BM)
    m_k = o_k < K
    m_v = o_v < V
    m_m = o_m < M
    o_t = tl.arange(0, BT)

    b_hk = tl.zeros([BK, BM], dtype=tl.float32)
    b_hv = tl.zeros([BM, BV], dtype=tl.float32)
    if USE_H0:
        p_hk0 = tl.make_block_ptr(hk0 + (i_b * H + i_h) * K * M, (K, M), (M, 1), (0, i_m * BM), (BK, BM), (1, 0))
        p_hv0 = tl.make_block_ptr(hv0 + (i_b * H + i_h) * M * V, (M, V), (V, 1), (i_m * BM, 0), (BM, BV), (1, 0))
        b_hk = tl.load(p_hk0, boundary_check=(0, 1)).to(tl.float32)
        b_hv = tl.load(p_hv0, boundary_check=(0, 1)).to(tl.float32)

    NT = tl.cdiv(T, BT)
    for i_t in range(NT):
        # store the chunk-start states (bf16, as the baseline chunk_fwd_h does)
        p_hk = tl.make_block_ptr(hk + ((i_b * NT + i_t) * H + i_h) * K * M, (K, M), (M, 1), (0, i_m * BM), (BK, BM), (1, 0))
        p_hv = tl.make_block_ptr(hv + ((i_b * NT + i_t) * H + i_h) * M * V, (M, V), (V, 1), (i_m * BM, 0), (BM, BV), (1, 0))
        tl.store(p_hk, b_hk.to(p_hk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_hv, b_hv.to(p_hv.dtype.element_ty), boundary_check=(0, 1))

        rows = i_t * BT + o_t
        real = rows - W
        m_r = (real >= 0) & (rows < T)
        # shifted loads (real row = GLA row - W, zero for real < 0)
        p_k = k + ((i_b * T + real[:, None]) * H + i_h) * K + o_k[None, :]
        p_s = s + ((i_b * T + real[:, None]) * H + i_h) * M + o_m[None, :]
        p_g = g + ((i_b * T + real[:, None]) * H + i_h) * M + o_m[None, :]
        p_v = v + ((i_b * T + real[:, None]) * H + i_h) * V + o_v[None, :]
        b_k = tl.load(p_k, mask=m_r[:, None] & m_k[None, :], other=0.)
        b_s = tl.load(p_s, mask=m_r[:, None] & m_m[None, :], other=0.)
        b_g = tl.load(p_g, mask=m_r[:, None] & m_m[None, :], other=0.).to(tl.float32)
        b_v = tl.load(p_v, mask=m_r[:, None] & m_v[None, :], other=0.)

        b_gc = tl.cumsum(b_g, axis=0)
        p_gc = tl.make_block_ptr(gc + (i_b * T * H + i_h) * M, (T, M), (H * M, 1), (i_t * BT, i_m * BM), (BT, BM), (1, 0))
        tl.store(p_gc, b_gc.to(p_gc.dtype.element_ty), boundary_check=(0, 1))

        last_local = min(BT, T - i_t * BT) - 1
        b_glast = tl.sum(tl.where(o_t[:, None] == last_local, b_gc, 0.), 0)

        b_hk *= exp(b_glast)[None, :]
        b_hv *= exp(b_glast)[:, None]
        b_sg = (b_s * exp(b_glast[None, :] - b_gc)).to(b_s.dtype)
        b_hk += tl.dot(tl.trans(b_k), b_sg)
        b_hv += tl.dot(tl.trans(b_sg), b_v)

    if STORE_HT:
        p_hkt = tl.make_block_ptr(hkt + (i_b * H + i_h) * K * M, (K, M), (M, 1), (0, i_m * BM), (BK, BM), (1, 0))
        p_hvt = tl.make_block_ptr(hvt + (i_b * H + i_h) * M * V, (M, V), (V, 1), (i_m * BM, 0), (BM, BV), (1, 0))
        tl.store(p_hkt, b_hk.to(p_hkt.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_hvt, b_hv.to(p_hvt.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def chunk_nha_fused_o_kernel(
    q,
    rq,
    rk,
    k,
    s,
    v,
    gc,
    hk,
    hv,
    o,
    scale,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    M: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BM: tl.constexpr,
    BWK: tl.constexpr,
    NG: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // NG
    NT = tl.cdiv(T, BT)
    NC: tl.constexpr = BT // BC

    o_r = tl.arange(0, BC)
    o_c = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    o_m = tl.arange(0, BM)
    o_w = tl.arange(0, BWK)
    m_k = o_k < K
    m_v = o_v < V
    m_m = o_m < M
    m_s = o_r[:, None] >= o_c[None, :]

    # chunk-start states for this (chunk, kv-head)
    p_hk = tl.make_block_ptr(hk + ((i_b * NT + i_t) * H + i_h) * K * M, (K, M), (M, 1), (0, 0), (BK, BM), (1, 0))
    p_hv = tl.make_block_ptr(hv + ((i_b * NT + i_t) * H + i_h) * M * V, (M, V), (V, 1), (0, 0), (BM, BV), (1, 0))
    b_hk = tl.load(p_hk, boundary_check=(0, 1))
    b_hv = tl.load(p_hv, boundary_check=(0, 1))

    base = i_t * BT
    for i_i in tl.static_range(NC):
        t0 = base + i_i * BC
        rows = t0 + o_r
        # q and rotary-q tiles for this sub-chunk
        p_q = q + ((i_b * T + rows[:, None]) * HQ + i_hq) * K + o_k[None, :]
        p_rq = rq + ((i_b * T + rows[:, None]) * HQ + i_hq) * K + o_k[None, :]
        m_rows = rows < T
        b_q = tl.load(p_q, mask=m_rows[:, None] & m_k[None, :], other=0.)
        b_rq = tl.load(p_rq, mask=m_rows[:, None] & m_k[None, :], other=0.)
        b_qs = (b_q * scale).to(b_q.dtype)
        b_rqs = (b_rq * scale).to(b_rq.dtype)
        # gate cumsum tile (GLA rows)
        p_gc = gc + ((i_b * T + rows[:, None]) * H + i_h) * M + o_m[None, :]
        b_gc = tl.load(p_gc, mask=m_rows[:, None] & m_m[None, :], other=0.)
        # decay reference: cumsum at the sub-chunk start row
        p_gn = gc + ((i_b * T + t0) * H + i_h) * M + o_m
        b_gn = tl.load(p_gn, mask=m_m & (t0 < T), other=0.)

        # ---- k-path: ok = (q*scale @ hk) * exp(gc)
        b_ok = tl.dot(b_qs, b_hk) * exp(b_gc)

        # ---- k-path intra: off-diagonal sub-chunk pairs
        for i_j in tl.static_range(0, NC):
            if i_j < i_i:
                tj = base + i_j * BC
                rj = tj + o_c
                real_j = rj - W
                m_rj = (real_j >= 0) & (rj < T)
                p_kj = k + ((i_b * T + real_j[:, None]) * H + i_h) * K + o_k[None, :]
                p_sj = s + ((i_b * T + real_j[:, None]) * H + i_h) * M + o_m[None, :]
                p_gcj = gc + ((i_b * T + rj[:, None]) * H + i_h) * M + o_m[None, :]
                b_kj = tl.load(p_kj, mask=m_rj[:, None] & m_k[None, :], other=0.)
                b_sj = tl.load(p_sj, mask=m_rj[:, None] & m_m[None, :], other=0.)
                b_gcj = tl.load(p_gcj, mask=(rj[:, None] < T) & m_m[None, :], other=0.)
                b_A = tl.dot(b_qs, tl.trans(b_kj)).to(b_kj.dtype)
                b_vg = (b_sj * exp(b_gn[None, :] - b_gcj)).to(b_sj.dtype)
                b_ok += tl.dot(b_A, b_vg) * exp(b_gc - b_gn[None, :])

        # ---- k-path intra: diagonal sub-chunk (tril block, gn-referenced dot)
        real_i = rows - W
        m_ri = (real_i >= 0) & (rows < T)
        p_ki = k + ((i_b * T + real_i[:, None]) * H + i_h) * K + o_k[None, :]
        p_si = s + ((i_b * T + real_i[:, None]) * H + i_h) * M + o_m[None, :]
        b_ki = tl.load(p_ki, mask=m_ri[:, None] & m_k[None, :], other=0.)
        b_si = tl.load(p_si, mask=m_ri[:, None] & m_m[None, :], other=0.)
        b_Aii = tl.where(m_s, tl.dot(b_qs, tl.trans(b_ki)), 0.).to(b_ki.dtype)
        b_vgi = (b_si * exp(b_gn[None, :] - b_gc)).to(b_si.dtype)
        b_ok += tl.dot(b_Aii, b_vgi) * exp(b_gc - b_gn[None, :])

        # ---- sliding-window scores (rotary)
        wrows = t0 - W + o_w
        m_w = (wrows >= 0) & (wrows < T)
        p_rw = rk + ((i_b * T + wrows[:, None]) * H + i_h) * K + o_k[None, :]
        b_rw = tl.load(p_rw, mask=m_w[:, None] & m_k[None, :], other=0.)
        b_swa = tl.dot(b_rqs, tl.trans(b_rw))
        valid = (o_w[None, :] >= o_r[:, None] + 1) & (o_w[None, :] <= o_r[:, None] + W) & m_w[None, :]
        b_swa = tl.where(valid, b_swa, float('-inf')).to(b_rw.dtype)

        # ---- in-register mixing softmax over [ok (M), swa (BWK)]
        # (baseline rounds ok to the input dtype via HBM before the fp32 softmax)
        b_okf = b_ok.to(b_qs.dtype)
        m_row = tl.maximum(tl.max(b_okf.to(tl.float32), 1), tl.max(b_swa.to(tl.float32), 1))
        d_row = tl.sum(exp(b_okf.to(tl.float32) - m_row[:, None]), 1) \
            + tl.sum(exp(b_swa.to(tl.float32) - m_row[:, None]), 1)
        b_qv = (exp(b_okf.to(tl.float32) - m_row[:, None]) / d_row[:, None]).to(b_qs.dtype)
        b_swp = (exp(b_swa.to(tl.float32) - m_row[:, None]) / d_row[:, None]).to(b_qs.dtype)

        # ---- v-path inter: (qv * exp(gc)) @ hv
        b_qvg = (b_qv * exp(b_gc)).to(b_qv.dtype)
        b_ov = tl.dot(b_qvg, b_hv)

        # ---- v-path intra: sub-chunk pairs including the tril diagonal block
        for i_j in tl.static_range(0, NC):
            if i_j <= i_i:
                tj = base + i_j * BC
                rj = tj + o_c
                real_j = rj - W
                m_rj = (real_j >= 0) & (rj < T)
                p_sj = s + ((i_b * T + real_j[:, None]) * H + i_h) * M + o_m[None, :]
                p_vj = v + ((i_b * T + real_j[:, None]) * H + i_h) * V + o_v[None, :]
                p_gcj = gc + ((i_b * T + rj[:, None]) * H + i_h) * M + o_m[None, :]
                b_sj = tl.load(p_sj, mask=m_rj[:, None] & m_m[None, :], other=0.)
                b_vj = tl.load(p_vj, mask=m_rj[:, None] & m_v[None, :], other=0.)
                b_gcj = tl.load(p_gcj, mask=(rj[:, None] < T) & m_m[None, :], other=0.)
                b_qg2 = b_qv * exp(b_gc - b_gn[None, :])
                b_sg2 = b_sj * exp(b_gn[None, :] - b_gcj)
                if i_j == i_i:
                    b_Av = tl.where(m_s, tl.dot(b_qg2, tl.trans(b_sg2)), 0.).to(b_vj.dtype)
                else:
                    b_Av = tl.dot(b_qg2, tl.trans(b_sg2)).to(b_vj.dtype)
                b_ov += tl.dot(b_Av, b_vj)

        # ---- sliding-window output
        p_wv = v + ((i_b * T + wrows[:, None]) * H + i_h) * V + o_v[None, :]
        b_wv = tl.load(p_wv, mask=m_w[:, None] & m_v[None, :], other=0.)
        b_ov += tl.dot(b_swp, b_wv)

        p_o = o + ((i_b * T + rows[:, None]) * HQ + i_hq) * V + o_v[None, :]
        tl.store(p_o, b_ov.to(p_o.dtype.element_ty), mask=m_rows[:, None] & m_v[None, :])


def _pow2(d: int) -> bool:
    return d & (d - 1) == 0 and d >= 16


@torch.no_grad()
def fused_chunk_nha_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rotary_q: torch.Tensor,
    rotary_k: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    window_size: int,
    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    output_final_state: bool = False,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Fused flash-style NHA prefill (inference only, no autograd).

    Args:
        q: [B, T, HQ, K] query (post q_norm, pre-rotary), contiguous.
        k: [B, T, H, K] key (pre-rotary), contiguous, kv heads.
        v: [B, T, H, V] value, contiguous, kv heads.
        rotary_q: [B, T, HQ, K] rotary-embedded query.
        rotary_k: [B, T, H, K] rotary-embedded key (kv heads).
        s: [B, T, H, M] slot residual (1 - exp(g)), contiguous, kv heads.
        g: [B, T, H, M] log gate (pre-cumsum), contiguous, kv heads.
        window_size: sliding window size W <= 64.
        initial_state: optional (hk0 [B, H, K, M] fp32, hv0 [B, H, M, V] fp32).
        output_final_state: whether to return the final states.
        scale: softmax scale (defaults to K ** -0.5).

    Returns:
        o: [B, T, HQ, V] output, same dtype as q.
        (hkt, hvt): fp32 final states [B, H, K, M] / [B, H, M, V] (or Nones).
    """
    B, T, HQ, K = q.shape
    H = k.shape[2]
    V = v.shape[-1]
    M = s.shape[-1]
    W = window_size
    assert k.shape[1] == T and v.shape[1] == T and s.shape[1] == T and g.shape[1] == T
    assert rotary_q.shape == q.shape and rotary_k.shape == k.shape
    assert HQ % H == 0
    assert _pow2(K) and K <= 128 and _pow2(V) and V <= 128 and _pow2(M) and M <= 128
    assert 1 <= W <= 64
    if scale is None:
        scale = K ** -0.5

    BT = 64
    BC = 16
    NT = triton.cdiv(T, BT)
    NG = HQ // H
    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)
    BM = triton.next_power_of_2(M)
    BWK = triton.next_power_of_2(BC + W)

    hk0, hv0 = (None, None) if initial_state is None else initial_state

    dtype = q.dtype
    hk = q.new_empty(B, NT, H, K, M, dtype=dtype)
    hv = q.new_empty(B, NT, H, M, V, dtype=dtype)
    gc = torch.empty(B, T, H, M, dtype=torch.float32, device=q.device)
    hkt = q.new_empty(B, H, K, M, dtype=torch.float32) if output_final_state else None
    hvt = q.new_empty(B, H, M, V, dtype=torch.float32) if output_final_state else None

    grid = (triton.cdiv(M, 16), B * H)
    chunk_nha_fused_state_kernel[grid](
        k, s, v, g, hk, hv, gc, hk0, hv0, hkt, hvt,
        T=T, H=H, K=K, V=V, M=M, W=W, BT=BT,
        BK=BK, BV=BV, BM=16,
        USE_H0=hk0 is not None, STORE_HT=output_final_state,
        num_warps=4, num_stages=3,
    )

    o = q.new_empty(B, T, HQ, V)
    grid = (NT, B * HQ)
    chunk_nha_fused_o_kernel[grid](
        q, rotary_q, rotary_k, k, s, v, gc, hk, hv, o,
        scale=scale,
        T=T, H=H, HQ=HQ, K=K, V=V, M=M, W=W, BT=BT, BC=BC,
        BK=BK, BV=BV, BM=BM, BWK=BWK, NG=NG,
        num_warps=4, num_stages=1,
    )
    return o, (hkt, hvt)
