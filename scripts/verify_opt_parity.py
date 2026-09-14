# -*- coding: utf-8 -*-
# Parity harness for the NHA operator optimizations (opt-verify branch).
#
# Two modes:
#   dump    --impl baseline|opt --out <dir>   run the suite, dump inputs+outputs
#   compare --baseline <dir> --opt <dir>      compare dumps, print err tables
#
# Baseline and opt differ only by PYTHONPATH (NHA-baseline vs NHA worktrees);
# inputs are regenerated from fixed per-case seeds (CPU randn -> deterministic).
#
# Numerical expectations encoded in the verdicts:
#   * item 1 (ok-buffer race fix): opt must match the fp64 naive reference for
#     every NG; the baseline is expected to mismatch for NG > 1 (that is the
#     bug being fixed) and to fail to compile for non-pow2 M.
#   * item 2 (fused chunk prefill): rel-L2 vs the chunk_nha chain <= 3e-3.
#   * items 3/5 (decode callsite reorder, cache cat): bit-exact vs baseline.
#   * item 4 (in-kernel rotary / in-kernel s): tiny fp32-reassociation-level
#     differences only.
#   * item 6 (PDL): bit-exact on/off.
#   * training path (chunk_nha fwd+bwd, varlen): untouched -> bit-exact.

import argparse
import inspect
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

DEVICE = 'cuda'
DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def repeat_kv(x: torch.Tensor, n: int) -> torch.Tensor:
    # transformers.models.qwen3_moe.modeling_qwen3_moe.repeat_kv semantics
    # ([B, H, T, D] -> [B, H*n, T, D], group-major order)
    return x.repeat_interleave(n, dim=1)


def eager_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # HF apply_rotary_pos_emb with unsqueeze_dim=2 on [B, T, H, K] tensors and
    # duplicated-halves cos/sin of shape [B, T, K].
    x1, x2 = x.chunk(2, dim=-1)
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos.unsqueeze(-2) + rot * sin.unsqueeze(-2)


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().flatten()
    b = b.double().flatten()
    return (a - b).norm().item() / max(b.norm().item(), 1e-30)


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.double() - b.double()).abs().max().item()


def max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a.double() - b.double()).abs()
    return (d / (b.double().abs() + 1e-6)).max().item()


# ---------------------------------------------------------------------------
# fp64 naive references (repo-independent, computed from dumped inputs)
# ---------------------------------------------------------------------------

def ref_gsa_step(q, k, v, s, g, hk0, hv0, scale):
    """One GSA recurrence step in fp64.
    q: [B,HQ,K] raw. k/v/s/g: [B,H,*]. hk0: [B,H,K,M], hv0: [B,H,M,V].
    Returns (ok [B,HQ,M], hk1, hv1)."""
    NG = q.shape[1] // k.shape[1]
    eg = g.double().exp()                                              # [B,H,M]
    hk1 = hk0.double() * eg[:, :, None, :] + k.double()[:, :, :, None] * s.double()[:, :, None, :]
    hv1 = hv0.double() * eg[:, :, :, None] + s.double()[:, :, :, None] * v.double()[:, :, None, :]
    ok = torch.einsum('bqk,bqkm->bqm', q.double() * scale, repeat_kv(hk1, NG))
    return ok, hk1, hv1


def ref_decode(inp):
    """fp64 reference for fused_recurrent_nha_decode semantics (window incl.).

    The rotary embedding is recomputed in fp64 from the raw q/window keys and
    the (bf16) cos/sin tables, so both the eager (bf16-rounded rope) and the
    in-kernel (fp32 rope) paths are measured against the same truth."""
    q = inp['q'].double()            # [B,1,HQ,K] raw
    k = inp['k'].double()            # [B,1,H,K]
    v = inp['v'].double()            # [B,1,H,V]
    s = inp['s'].double()            # [B,1,H,M]
    g = inp['g'].double()            # [B,1,H,M]
    swk = inp['sw_k_raw'].double()   # [B,W,H,K] raw, kv-head layout
    vw = inp['vw'].double()          # [B,W,H,V]
    cos = inp['cos'].double()        # [1,W,K]
    sin = inp['sin'].double()
    hk0 = inp['hk0'].double()
    hv0 = inp['hv0'].double()
    scale = inp['scale']
    B, _, HQ, K = q.shape
    H = k.shape[2]
    NG = HQ // H

    def rope(x, c, s):
        x1, x2 = x.chunk(2, dim=-1)
        rot = torch.cat([-x2, x1], dim=-1)
        return x * c.unsqueeze(-2) + rot * s.unsqueeze(-2)

    sq = rope(q, cos[:, -1:], sin[:, -1:])      # [B,1,HQ,K]
    sk = rope(swk, cos, sin)                    # [B,W,H,K]
    ok, hk1, hv1 = ref_gsa_step(q[:, 0], k[:, 0], v[:, 0], s[:, 0], g[:, 0], hk0, hv0, scale)
    sk_hq = sk.repeat_interleave(NG, dim=2)     # [B,W,HQ,K]
    vw_hq = vw.repeat_interleave(NG, dim=2)     # [B,W,HQ,V]
    sw = torch.einsum('bqk,bwqk->bqw', sq[:, 0] * scale, sk_hq)
    p = torch.softmax(torch.cat([ok, sw], dim=-1), dim=-1)
    qv, swp = p[..., :ok.shape[-1]], p[..., ok.shape[-1]:]
    o = torch.einsum('bqm,bqmv->bqv', qv, repeat_kv(hv1, NG)) \
        + torch.einsum('bqw,bwqv->bqv', swp, vw_hq)
    return {'o': o[:, None], 'hkt': hk1, 'hvt': hv1, 'qv': qv, 'swp': swp}


def ref_fused_recurrent(inp):
    """fp64 reference for fused_recurrent_nha (T=1 inference path)."""
    q = inp['q'].double()            # [B,1,HQ,K]
    k = inp['k'].double()
    v = inp['v'].double()
    s = inp['s'].double()
    g = inp['g'].double()
    sw = inp['sliding_window'].double()   # [B,1,HQ,W]
    hk0 = inp['hk0'].double()
    hv0 = inp['hv0'].double()
    scale = inp['scale']
    NG = q.shape[2] // k.shape[2]
    ok, hk1, hv1 = ref_gsa_step(q[:, 0], k[:, 0], v[:, 0], s[:, 0], g[:, 0], hk0, hv0, scale)
    p = torch.softmax(torch.cat([ok, sw[:, 0]], dim=-1), dim=-1)
    qv, swp = p[..., :ok.shape[-1]], p[..., ok.shape[-1]:]
    o = torch.einsum('bqm,bqmv->bqv', qv, repeat_kv(hv1, NG))
    return {'o': o[:, None], 'swa_score': swp[:, None], 'hkt': hk1, 'hvt': hv1}


def ref_prefill(inp):
    """fp64 naive NHA prefill: per-step GSA recurrence + clamped sliding window.

    q: [B,T,HQ,K] raw, rotary_q: [B,T,HQ,K], k/v/s/g: [B,T,H,*] kv heads,
    rotary_k: [B,T,H,K], initial states [B,H,K,M]/[B,H,M,V] or absent.
    GSA token for output t is raw token t-W (state updated before readout);
    the window covers raw tokens [max(0, t-W+1), t].
    """
    q = inp['q'].double()
    rq = inp['rotary_q'].double()
    k = inp['k'].double()
    v = inp['v'].double()
    s = inp['s'].double()
    g = inp['g'].double()
    rk = inp['rotary_k'].double()
    scale = inp['scale']
    W = inp['W']
    B, T, HQ, K = q.shape
    H = k.shape[2]
    M = s.shape[-1]
    V = v.shape[-1]
    NG = HQ // H
    if 'hk0' in inp:
        hk = inp['hk0'].double().clone()
        hv = inp['hv0'].double().clone()
    else:
        hk = torch.zeros(B, H, K, M, dtype=torch.float64)
        hv = torch.zeros(B, H, M, V, dtype=torch.float64)
    o = torch.zeros(B, T, HQ, V, dtype=torch.float64)
    for t in range(T):
        j = t - W
        if j >= 0:
            eg = g[:, j].exp()                                    # [B,H,M]
            hk = hk * eg[:, :, None, :] + k[:, j][:, :, :, None] * s[:, j][:, :, None, :]
            hv = hv * eg[:, :, :, None] + s[:, j][:, :, :, None] * v[:, j][:, :, None, :]
        ok = torch.einsum('bqk,bqkm->bqm', q[:, t] * scale, repeat_kv(hk, NG))
        lo = max(0, t - W + 1)
        win = rk[:, lo:t + 1].repeat_interleave(NG, dim=2)          # [B,w,HQ,K]
        sw = torch.einsum('bqk,bwqk->bqw', rq[:, t] * scale, win)
        p = torch.softmax(torch.cat([ok, sw], dim=-1), dim=-1)
        qv, swp = p[..., :M], p[..., M:]
        wv = v[:, lo:t + 1].repeat_interleave(NG, dim=2)
        o[:, t] = torch.einsum('bqm,bqmv->bqv', qv, repeat_kv(hv, NG)) \
            + torch.einsum('bqw,bwqv->bqv', swp, wv)
    return {'o': o, 'hkt': hk, 'hvt': hv}


# ---------------------------------------------------------------------------
# input generation (deterministic)
# ---------------------------------------------------------------------------

def _t(*shape, dtype=DTYPE, scale=1.0):
    return (torch.randn(*shape, dtype=torch.float32) * scale).to(dtype).to(DEVICE)


def gen_recurrent_inputs(seed, B, H, NG, K, V, M, W):
    torch.manual_seed(seed)
    HQ = H * NG
    inp = {
        'q': _t(B, 1, HQ, K),
        'k': _t(B, 1, H, K),
        'v': _t(B, 1, H, V),
        'sliding_window': _t(B, 1, HQ, W),
        'hk0': _t(B, H, K, M, dtype=torch.float32, scale=0.5),
        'hv0': _t(B, H, M, V, dtype=torch.float32, scale=0.5),
    }
    inp['g'] = F.logsigmoid(torch.randn(B, 1, H, M)).to(DTYPE).to(DEVICE) / 8.0
    inp['s'] = (1 - inp['g'].exp()).to(DTYPE)
    inp['scale'] = K ** -0.5
    return inp


def gen_decode_inputs(seed, B, H, NG, K, V, M, W):
    torch.manual_seed(seed)
    HQ = H * NG
    inp = gen_recurrent_inputs(seed, B, H, NG, K, V, M, W)
    # window tensors in kv-head layout + rotary tables (duplicated halves)
    inp['sw_k_raw'] = _t(B, W, H, K)
    inp['vw'] = _t(B, W, H, V)
    theta = 10000.0
    inv_freq = 1.0 / (theta ** (torch.arange(0, K, 2, dtype=torch.float32) / K))
    pos = torch.arange(100, 100 + W, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    inp['cos'] = emb.cos().to(DTYPE).to(DEVICE)[None]
    inp['sin'] = emb.sin().to(DTYPE).to(DEVICE)[None]
    # rope'd variants (eager, model-dtype rounding)
    inp['sq'] = eager_rope(inp['q'], inp['cos'][:, -1:], inp['sin'][:, -1:])
    inp['sk'] = eager_rope(inp['sw_k_raw'], inp['cos'], inp['sin'])
    return inp


def gen_prefill_inputs(seed, B, H, NG, K, V, M, W, T, with_state):
    torch.manual_seed(seed)
    HQ = H * NG
    inp = {
        'q': _t(B, T, HQ, K),
        'k': _t(B, T, H, K),
        'v': _t(B, T, H, V),
        'rotary_q': _t(B, T, HQ, K),
        'rotary_k': _t(B, T, H, K),
        'W': W,
        'scale': K ** -0.5,
    }
    inp['g'] = F.logsigmoid(torch.randn(B, T, H, M)).to(DTYPE).to(DEVICE) / 8.0
    inp['s'] = (1 - inp['g'].exp()).to(DTYPE)
    if with_state:
        inp['hk0'] = _t(B, H, K, M, dtype=torch.float32, scale=0.5)
        inp['hv0'] = _t(B, H, M, V, dtype=torch.float32, scale=0.5)
    return inp


def _cpu(d):
    out = {}
    for k, v in d.items():
        out[k] = v.detach().cpu() if torch.is_tensor(v) else v
    return out


def _save(d, path):
    torch.save(_cpu(d), path)


# ---------------------------------------------------------------------------
# dump: op-level fused_recurrent_nha (item 1)
# ---------------------------------------------------------------------------

def dump_fused_recurrent(out_dir):
    from nha_fla.ops.nha.fused_recurrent import fused_recurrent_nha
    results = {}
    seed = 1000
    for NG in (1, 4, 8):
        for M in (16, 64, 80):
            for W in (32, 64):
                for K in (64, 128):
                    for B in (1, 4):
                        V = K
                        H = 2
                        case = f'fr_ng{NG}_m{M}_w{W}_k{K}_b{B}'
                        inp = gen_recurrent_inputs(seed, B, H, NG, K, V, M, W)
                        seed += 1
                        rec = {'inputs': _cpu(inp)}
                        try:
                            with torch.no_grad():
                                o, swa, (hkt, hvt) = fused_recurrent_nha(
                                    q=inp['q'], k=inp['k'], v=inp['v'],
                                    sliding_window=inp['sliding_window'],
                                    s=inp['s'], g=inp['g'],
                                    initial_state=(inp['hk0'].clone(), inp['hv0'].clone()),
                                    output_final_state=True,
                                    scale=inp['scale'],
                                )
                            rec['outputs'] = _cpu({'o': o, 'swa_score': swa, 'hkt': hkt, 'hvt': hvt})
                        except Exception as e:
                            rec['error'] = f'{type(e).__name__}: {str(e)[:300]}'
                        results[case] = rec
                        print(f'  [fr] {case} done', flush=True)
    _save(results, os.path.join(out_dir, 'fused_recurrent.pt'))


# ---------------------------------------------------------------------------
# dump: op-level fused_recurrent_nha_decode (items 3/4/6)
# ---------------------------------------------------------------------------

def _decode_new_style():
    from nha_fla.ops.nha.fused_recurrent import fused_recurrent_nha_decode
    return 'sw_k' in inspect.signature(fused_recurrent_nha_decode).parameters


def dump_decode(out_dir):
    from nha_fla.ops.nha.fused_recurrent import fused_recurrent_nha_decode
    import nha_fla.ops.nha.fused_recurrent as fr_mod
    new_style = _decode_new_style()
    results = {}
    seed = 2000
    for NG in (1, 4, 8):
        for M in (16, 64, 80):
            for W in (32, 64):
                for K in (64, 128):
                    for B in (1, 4):
                        V = K
                        H = 2
                        case = f'dc_ng{NG}_m{M}_w{W}_k{K}_b{B}'
                        inp = gen_decode_inputs(seed, B, H, NG, K, V, M, W)
                        seed += 1
                        rec = {'inputs': _cpu(inp)}
                        outs = {}
                        try:
                            if new_style:
                                with torch.no_grad():
                                    # eager mode (pre-rotated sq/sk, caller s):
                                    # isolates the layout reorder (item 3)
                                    o, (hkt, hvt) = fused_recurrent_nha_decode(
                                        q=inp['q'], sq=inp['sq'],
                                        k=inp['k'], v=inp['v'], g=inp['g'], s=inp['s'],
                                        sw_k=inp['sk'], sw_v=inp['vw'],
                                        initial_state=(inp['hk0'].clone(), inp['hv0'].clone()),
                                        output_final_state=True, scale=inp['scale'])
                                    outs['eager'] = _cpu({'o': o, 'hkt': hkt, 'hvt': hvt})
                                    # rotary mode (in-kernel rotary + s, item 4)
                                    o, (hkt, hvt) = fused_recurrent_nha_decode(
                                        q=inp['q'],
                                        k=inp['k'], v=inp['v'], g=inp['g'],
                                        sw_k=inp['sw_k_raw'], sw_v=inp['vw'],
                                        cos=inp['cos'], sin=inp['sin'],
                                        initial_state=(inp['hk0'].clone(), inp['hv0'].clone()),
                                        output_final_state=True, scale=inp['scale'])
                                    outs['rotary'] = _cpu({'o': o, 'hkt': hkt, 'hvt': hvt})
                                    # PDL A/B on the rotary mode (item 6)
                                    if hasattr(fr_mod, '_PDL_ENV'):
                                        fr_mod._PDL_ENV = False
                                        o, (hkt, hvt) = fused_recurrent_nha_decode(
                                            q=inp['q'],
                                            k=inp['k'], v=inp['v'], g=inp['g'],
                                            sw_k=inp['sw_k_raw'], sw_v=inp['vw'],
                                            cos=inp['cos'], sin=inp['sin'],
                                            initial_state=(inp['hk0'].clone(), inp['hv0'].clone()),
                                            output_final_state=True, scale=inp['scale'])
                                        outs['rotary_nopdl'] = _cpu({'o': o, 'hkt': hkt, 'hvt': hvt})
                                        fr_mod._PDL_ENV = True
                            else:
                                # baseline signature: HQ-expanded window layout
                                sk_hq = repeat_kv(inp['sk'].transpose(1, 2), NG).transpose(1, 2)
                                vw_hq = repeat_kv(inp['vw'].transpose(1, 2), NG).transpose(1, 2)
                                with torch.no_grad():
                                    o, (hkt, hvt) = fused_recurrent_nha_decode(
                                        q=inp['q'][:, 0], sq=inp['sq'][:, 0],
                                        k=inp['k'][:, 0], v=inp['v'][:, 0],
                                        s=inp['s'][:, 0], g=inp['g'][:, 0],
                                        sk=sk_hq, vw=vw_hq,
                                        initial_state=(inp['hk0'].clone(), inp['hv0'].clone()),
                                        output_final_state=True, scale=inp['scale'])
                                outs['eager'] = _cpu({'o': o[:, None], 'hkt': hkt, 'hvt': hvt})
                        except Exception as e:
                            rec['error'] = f'{type(e).__name__}: {str(e)[:300]}'
                        rec['outputs'] = outs
                        results[case] = rec
                        print(f'  [decode] {case} done', flush=True)
    _save(results, os.path.join(out_dir, 'decode.pt'))


# ---------------------------------------------------------------------------
# dump: chunk_fused prefill (item 2) vs the chunk_nha chain
# ---------------------------------------------------------------------------

def chain_prefill(chunk_nha, inp, NG, W):
    """The qwennha fallback prefill chain (kv-head inputs, expanded inside)."""
    q, k, v, s, g = inp['q'], inp['k'], inp['v'], inp['s'], inp['g']
    rotary_q, rotary_k = inp['rotary_q'], inp['rotary_k']
    B, T, H, K = k.shape
    M = s.shape[-1]
    V = v.shape[-1]
    z = lambda *shape: torch.zeros(*shape, dtype=k.dtype, device=k.device)
    shift_k = torch.cat([z(B, W, H, K), k], dim=1)
    shift_v = torch.cat([z(B, W, H, V), v], dim=1)
    shift_s = torch.cat([z(B, W, H, M), s], dim=1)
    shift_g = torch.cat([z(B, W, H, M), g], dim=1)
    rotary_k_full = torch.cat([z(B, W, H, K), rotary_k], dim=1)
    chunk_k = repeat_kv(shift_k.transpose(1, 2), NG).transpose(1, 2)
    chunk_v = repeat_kv(shift_v.transpose(1, 2), NG).transpose(1, 2)
    chunk_s = repeat_kv(shift_s.transpose(1, 2), NG).transpose(1, 2)
    chunk_g = repeat_kv(shift_g.transpose(1, 2), NG).transpose(1, 2)
    rotary_k_hq = repeat_kv(rotary_k_full.transpose(1, 2), NG).transpose(1, 2)
    initial_state = None
    if 'hk0' in inp:
        initial_state = (
            inp['hk0'].repeat_interleave(NG, dim=1).contiguous(),
            inp['hv0'].repeat_interleave(NG, dim=1).contiguous(),
        )
    o, st = chunk_nha(
        q=q, k=chunk_k, v=chunk_v,
        rotary_q=rotary_q, rotary_k=rotary_k_hq,
        window_size=W, s=chunk_s, g=chunk_g,
        initial_state=initial_state,
        output_final_state=True,
        scale=None, head_first=False, rotary=None,
    )
    hkt, hvt = st[0][:, ::NG].contiguous(), st[1][:, ::NG].contiguous()
    return {'o': o, 'hkt': hkt, 'hvt': hvt}


def dump_chunk_fused(out_dir):
    from nha_fla.ops.nha_naive import chunk_nha
    try:
        from nha_fla.ops.nha.chunk_fused import fused_chunk_nha_prefill
        has_fused = True
    except Exception:
        has_fused = False
    results = {}
    seed = 3000
    for T in (63, 128, 257, 1000):
        for B in (1, 4):
            for W in (32, 64):
                for M in (16, 64):
                    K = V = 64
                    H, NG = 2, 4
                    case = f'pf_t{T}_b{B}_w{W}_m{M}'
                    inp = gen_prefill_inputs(seed, B, H, NG, K, V, M, W, T, with_state=(T == 128))
                    seed += 1
                    rec = {'inputs': _cpu(inp)}
                    outs = {}
                    try:
                        with torch.no_grad():
                            outs['chain'] = _cpu(chain_prefill(chunk_nha, inp, NG, W))
                    except Exception as e:
                        rec['error'] = f'chain {type(e).__name__}: {str(e)[:300]}'
                    if has_fused:
                        try:
                            with torch.no_grad():
                                o, (hkt, hvt) = fused_chunk_nha_prefill(
                                    q=inp['q'], k=inp['k'], v=inp['v'],
                                    rotary_q=inp['rotary_q'], rotary_k=inp['rotary_k'],
                                    window_size=W, s=inp['s'], g=inp['g'],
                                    initial_state=(inp['hk0'], inp['hv0']) if 'hk0' in inp else None,
                                    output_final_state=True, scale=None)
                            outs['fused'] = _cpu({'o': o, 'hkt': hkt, 'hvt': hvt})
                        except Exception as e:
                            rec['error'] = f'fused {type(e).__name__}: {str(e)[:300]}'
                    rec['outputs'] = outs
                    results[case] = rec
                    print(f'  [prefill] {case} done', flush=True)
    _save(results, os.path.join(out_dir, 'chunk_fused.pt'))


# ---------------------------------------------------------------------------
# dump: window>64 decode chain (naive_swa+einsum baseline vs flash-attn)
# ---------------------------------------------------------------------------

def dump_swa_decode_chain(out_dir):
    results = {}
    seed = 3500
    try:
        from flash_attn import flash_attn_func
        has_fa = True
    except Exception:
        has_fa = False

    def naive_swa(q, k, W):
        seq_len = q.shape[1]
        i = torch.arange(seq_len, device=q.device).view(-1, 1)
        j = torch.arange(seq_len, device=q.device).view(1, -1)
        left_bound = torch.clamp(i - W + 1, min=0)
        valid_mask = (j >= left_bound) & (j <= i)
        qk = torch.einsum('bthd,bnhd->bhtn', q, k) * (q.shape[-1] ** -0.5)
        qk = qk.masked_fill(~valid_mask.unsqueeze(0).unsqueeze(0), -1e7)
        return qk.transpose(1, 2)

    for (B, HQ, H, K, W, M) in ((1, 8, 2, 64, 96, 16), (4, 8, 2, 128, 130, 16)):
        case = f'swa_b{B}_k{K}_w{W}'
        torch.manual_seed(seed)
        seed += 1
        NG = HQ // H
        sq = _t(B, 1, HQ, K)
        sk = _t(B, W, H, K)
        v = _t(B, W, H, K)
        rec = {'inputs': _cpu({'sq': sq, 'sk': sk, 'v': v, 'W': 2048, 'M': M})}
        outs = {}
        with torch.no_grad():
            # baseline HEAD chain: zero-prefix of num_slots + naive_swa + einsum
            sk_hq = repeat_kv(sk.transpose(1, 2), NG).transpose(1, 2)
            v_hq = repeat_kv(v.transpose(1, 2), NG).transpose(1, 2)
            prefix_k = torch.zeros(B, M, HQ, K, dtype=DTYPE, device=DEVICE)
            prefix_v = torch.zeros(B, M, HQ, K, dtype=DTYPE, device=DEVICE)
            shift_k = torch.cat([prefix_k, sk_hq], dim=1)
            shift_v = torch.cat([prefix_v, v_hq], dim=1)
            sw = naive_swa(sq, shift_k, 2048)
            prob = sw.softmax(-1)
            o_base = torch.einsum('bthw,bwhd->bthd', prob, shift_v)
            outs['naive_chain'] = _cpu({'o': o_base})
            if has_fa:
                o_fa = flash_attn_func(sq, sk, v, causal=False)
                outs['flash_attn'] = _cpu({'o': o_fa})
            # fp64 true attention over the window (no prefix)
            p = torch.softmax(
                torch.einsum('bqk,bwqk->bqw', sq[:, 0].double() * (K ** -0.5),
                             repeat_kv(sk.double().transpose(1, 2), NG).transpose(1, 2)), dim=-1)
            o_ref = torch.einsum('bqw,bwqv->bqv', p, repeat_kv(v.double().transpose(1, 2), NG).transpose(1, 2))
            outs['ref_fp64'] = _cpu({'o': o_ref[:, None]})
        rec['outputs'] = outs
        results[case] = rec
        print(f'  [swa-chain] {case} done', flush=True)
    _save(results, os.path.join(out_dir, 'swa_decode_chain.pt'))


# ---------------------------------------------------------------------------
# dump: training path (chunk_nha fwd+bwd, new-style op; untouched code)
# ---------------------------------------------------------------------------

def dump_training(out_dir):
    from nha_fla.ops.nha import chunk_nha
    results = {}
    seed = 4000
    shapes = [
        # (B, T, H(hq), Hkv, K, V, M, varlen)
        (2, 128, 4, 4, 64, 64, 32, False),
        (1, 300, 8, 8, 64, 64, 64, True),   # varlen cu_seqlens [0,100,300]
        (2, 96, 4, 4, 128, 128, 16, False),
    ]
    for (B, T, HQ, H, K, V, M, varlen) in shapes:
        case = f'tr_b{B}_t{T}_h{HQ}_k{K}_v{V}_m{M}_vl{int(varlen)}'
        torch.manual_seed(seed)
        seed += 1
        NG = HQ // H
        if NG != 1:
            H = HQ  # training layer uses equal head counts; keep HQ == H here
        if varlen:
            cu = torch.tensor([0, 100, T], dtype=torch.int32, device=DEVICE)
            Bv = 1
        else:
            cu = None
            Bv = B
        q_swa = _t(Bv, T, HQ, K).requires_grad_(True)
        k_swa = _t(Bv, T, HQ, K).requires_grad_(True)
        q_gsa = _t(Bv, T, HQ, K).requires_grad_(True)
        k_gsa = _t(Bv, T, HQ, K).requires_grad_(True)
        v = _t(Bv, T, HQ, V).requires_grad_(True)
        s = _t(Bv, T, HQ, M).requires_grad_(True)
        g = (F.logsigmoid(torch.randn(Bv, T, HQ, M)).to(DTYPE).to(DEVICE) / 8.0).requires_grad_(True)
        do = _t(Bv, T, HQ, V, dtype=torch.float32)
        ins = [q_swa, k_swa, q_gsa, k_gsa, v, s, g]
        rec = {'inputs': _cpu({'q_swa': q_swa, 'k_swa': k_swa, 'q_gsa': q_gsa, 'k_gsa': k_gsa,
                               'v': v, 's': s, 'g': g, 'do': do,
                               'cu': cu.cpu() if cu is not None else None, 'T': T})}
        try:
            o, lse_swa, lse_gsa = chunk_nha(
                q_swa, k_swa, q_gsa, k_gsa, v, s, g,
                cu_seqlens=cu, max_seqlen=T if cu is not None else None,
                window_size_left=63, window_size_right=0,
            )
            o.backward(do.to(o.dtype))
            outs = {'o': o, 'lse_swa': lse_swa, 'lse_gsa': lse_gsa}
            for name, x in zip(('q_swa', 'k_swa', 'q_gsa', 'k_gsa', 'v', 's', 'g'), ins):
                outs[f'grad_{name}'] = x.grad
            rec['outputs'] = _cpu(outs)
        except Exception as e:
            rec['error'] = f'{type(e).__name__}: {str(e)[:300]}'
        results[case] = rec
        print(f'  [training] {case} done', flush=True)
    _save(results, os.path.join(out_dir, 'training.pt'))


# ---------------------------------------------------------------------------
# dump: model-level prefill+decode (items 2/3/4/5 e2e)
# ---------------------------------------------------------------------------

def _tiny_config(tag):
    from nha_fla.models.qwen3_moe_nha.configuration_qwen3_moe_nha import Qwen3MoeNHAConfig
    common = dict(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        moe_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        num_slots=16,
        window_size=32,
        max_position_embeddings=4096,
    )
    if tag == 'nha':
        common.update(block_size=None, transformer_idx=None)
    elif tag == 'mixed':
        # layers 1 and 3 become full-attention (window 2048)
        common.update(block_size=2, transformer_idx=1)
    return Qwen3MoeNHAConfig(**common)


def dump_model(out_dir):
    # transformers >= 4.57 requires Cache.__init__ arguments; NHACache (both
    # repo versions) calls the no-arg super().__init__ and is broken out of the
    # box. Patch the *base class* leniently for both impls alike (test-only
    # monkeypatch, repo code untouched).
    import transformers.cache_utils as _tcu

    def _lenient_cache_init(self, layers=None, layer_class_to_replicate=None,
                            offloading=False, offload_only_non_sliding=True):
        self.layers = layers if layers is not None else []
        self.layer_class_to_replicate = layer_class_to_replicate
        self.offloading = offloading
        self.offload_only_non_sliding = offload_only_non_sliding
    _tcu.Cache.__init__ = _lenient_cache_init

    from nha_fla.models.qwen3_moe_nha.modeling_qwen3_moe_nha import Qwen3MoeNHAForCausalLM
    from nha_fla.models.nha_cache import NHACache

    def _patch_decoder_layers(model):
        # transformers >= 4.57 decoder layers return a bare tensor while the
        # repo's modeling code unpacks `layer_outputs[0]`; wrap each layer to
        # return a tuple again (same treatment for both impls). The layers also
        # forward `past_key_values` (plural) to self_attn while the repo's
        # attention takes `past_key_value` (singular); translate the kwarg.
        for layer in model.model.layers:
                _orig_fwd = layer.forward

                def _fwd(*args, _orig=_orig_fwd, **kwargs):
                    out = _orig(*args, **kwargs)
                    return (out, None) if torch.is_tensor(out) else out
                layer.forward = _fwd

                attn = layer.self_attn
                _orig_attn = attn.forward

                def _attn_fwd(*args, _orig=_orig_attn, **kwargs):
                    if 'past_key_values' in kwargs and 'past_key_value' not in kwargs:
                        kwargs['past_key_value'] = kwargs.pop('past_key_values')
                    return _orig(*args, **kwargs)
                attn.forward = _attn_fwd
    for tag in ('nha', 'mixed'):
        for B in (1, 4):
            case = f'model_{tag}_b{B}'
            torch.manual_seed(1234)
            config = _tiny_config(tag)
            model = Qwen3MoeNHAForCausalLM(config).to(DTYPE).to(DEVICE)
            model.eval()
            _patch_decoder_layers(model)
            torch.manual_seed(99)
            pre_ids = torch.randint(0, config.vocab_size, (B, 128))
            torch.manual_seed(7)
            dec_ids = torch.randint(0, config.vocab_size, (B, 16))
            outs = {'input_ids': _cpu({'prefill': pre_ids, 'decode': dec_ids})}
            try:
                past = NHACache()
                with torch.no_grad():
                    o = model(input_ids=pre_ids.to(DEVICE), past_key_values=past, use_cache=True)
                    outs['prefill_logits'] = _cpu({'x': o.logits})['x']
                    for t in range(dec_ids.shape[1]):
                        o = model(input_ids=dec_ids[:, t:t + 1].to(DEVICE),
                                  past_key_values=past, use_cache=True)
                        outs[f'step{t:02d}_logits'] = _cpu({'x': o.logits})['x']
                    for i, st in enumerate(past.states):
                        rs = st['recurrent_state']
                        if rs is not None and rs[0] is not None:
                            outs[f'layer{i}_hk'] = _cpu({'x': rs[0]})['x']
                            outs[f'layer{i}_hv'] = _cpu({'x': rs[1]})['x']
                        wk, wv, wg = st['attn_state']
                        hd = config.head_dim
                        outs[f'layer{i}_wk'] = _cpu({'x': wk.view(*wk.shape[:-1], -1, hd)})['x']
                        outs[f'layer{i}_wv'] = _cpu({'x': wv.view(*wv.shape[:-1], -1, hd)})['x']
                rec = {'outputs': outs}
            except Exception as e:
                import traceback
                rec = {'error': f'{type(e).__name__}: {str(e)[:300]}\n{traceback.format_exc()[-1000:]}'}
            _save({'outputs': rec.get('outputs', {}), 'error': rec.get('error')},
                  os.path.join(out_dir, f'{case}.pt'))
            print(f'  [model] {case} done', flush=True)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def _bit(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a, b))


def _cmp_row(a, b):
    if a is None or b is None:
        return ('N/A', '', '', '')
    if a.shape != b.shape:
        return ('SHAPE', str(tuple(a.shape)), str(tuple(b.shape)), '')
    if _bit(a, b):
        return ('0', '0', '0', 'bit-exact')
    return (f'{max_abs(a, b):.3e}', f'{max_rel(a, b):.3e}', f'{rel_l2(a, b):.3e}', '')


def compare(args):
    rep = []
    def p(line=''):
        print(line)
        rep.append(line)

    bd, od = args.baseline, args.opt

    def load(d, name):
        path = os.path.join(d, name)
        return torch.load(path, map_location='cpu', weights_only=False) if os.path.exists(path) else None

    # ---- fused_recurrent_nha vs fp64 ref (item 1) ----
    base = load(bd, 'fused_recurrent.pt')
    opt = load(od, 'fused_recurrent.pt')
    if base and opt:
        p('\n=== [item 1] fused_recurrent_nha vs fp64 naive reference ===')
        p(f'{"case":28s} {"base relL2":>12s} {"opt relL2":>12s}  verdict')
        worst = 0.0
        for case in sorted(base):
            rb, ro = base[case], opt[case]
            ref = ref_fused_recurrent({k: v for k, v in rb['inputs'].items()})
            eb = rb.get('error')
            eo = ro.get('error')
            if eo:
                verdict = f'OPT ERROR: {eo[:80]}'
                rb64 = ro64 = float('nan')
            else:
                ro64 = max(rel_l2(ro['outputs'][k2], ref[k2]) for k2 in ('o', 'hkt', 'hvt'))
                worst = max(worst, ro64 if ro64 == ro64 else 1e9)
                if eb:
                    rb64 = float('nan')
                    verdict = 'opt OK; baseline cannot run (expected)' if 'NG' not in case else ''
                    verdict = 'opt OK; baseline N/A'
                else:
                    rb64 = max(rel_l2(rb['outputs'][k2], ref[k2]) for k2 in ('o', 'hkt', 'hvt'))
                    ng = int(case.split('_')[1][2:])
                    if ng > 1 and rb64 > 5 * max(ro64, 1e-6):
                        verdict = 'baseline WRONG (race), opt OK'
                    elif ro64 <= 2e-2:
                        verdict = 'OK'
                    else:
                        verdict = 'CHECK'
            p(f'{case:28s} {rb64:12.3e} {ro64:12.3e}  {verdict}')
        p(f'worst opt rel-L2 vs fp64: {worst:.3e}')

    # ---- decode op (items 3/4/6) ----
    base = load(bd, 'decode.pt')
    opt = load(od, 'decode.pt')
    if base and opt:
        p('\n=== [items 3/4/6] fused_recurrent_nha_decode ===')
        p(f'{"case":28s} {"i3 bit?":>8s} {"i4 relL2":>10s} {"i6 bit?":>8s} {"opt vs fp64":>12s} {"base vs fp64":>12s}')
        n_bit3 = n_bit6 = 0
        worst4 = worst_ref = 0.0
        for case in sorted(opt):
            rb, ro = base.get(case), opt[case]
            ref = ref_decode(rb['inputs'])
            ob = rb.get('outputs', {}).get('eager')
            oe = ro.get('outputs', {}).get('eager')
            orot = ro.get('outputs', {}).get('rotary')
            onopdl = ro.get('outputs', {}).get('rotary_nopdl')
            if rb.get('error') or ro.get('error'):
                p(f'{case:28s}  baseline_err={bool(rb.get("error"))} opt_err={bool(ro.get("error"))}')
                continue
            bit3 = all(_bit(ob[k2], oe[k2]) for k2 in ('o', 'hkt', 'hvt'))
            n_bit3 += bit3
            e4 = max(rel_l2(ob[k2], orot[k2]) for k2 in ('o', 'hkt', 'hvt'))
            worst4 = max(worst4, e4)
            bit6 = True
            if onopdl is not None:
                bit6 = all(_bit(onopdl[k2], orot[k2]) for k2 in ('o', 'hkt', 'hvt'))
                n_bit6 += bit6
            er = max(rel_l2(orot[k2], ref[k2]) for k2 in ('o', 'hkt', 'hvt'))
            eb = max(rel_l2(ob[k2], ref[k2]) for k2 in ('o', 'hkt', 'hvt'))
            worst_ref = max(worst_ref, er)
            p(f'{case:28s} {str(bit3):>8s} {e4:10.3e} {str(bit6):>8s} {er:12.3e} {eb:12.3e}')
        p(f'item3 bit-exact cases: {n_bit3}/{len(opt)}; item6 bit-exact: {n_bit6}; '
          f'worst item4 rel-L2: {worst4:.3e}; worst opt-vs-fp64: {worst_ref:.3e}')

    # ---- chunk_fused prefill (item 2) ----
    base = load(bd, 'chunk_fused.pt')
    opt = load(od, 'chunk_fused.pt')
    if base and opt:
        p('\n=== [item 2] fused chunk prefill vs chunk_nha chain ===')
        p(f'{"case":24s} {"chain bit?":>8s} {"fused relL2":>12s} {"fused maxrel":>12s} {"ref relL2":>10s} {"chain relL2":>11s}  verdict')
        worst = 0.0
        for case in sorted(opt):
            rb, ro = base[case], opt[case]
            oc_b = rb.get('outputs', {}).get('chain')
            oc_o = ro.get('outputs', {}).get('chain')
            of = ro.get('outputs', {}).get('fused')
            if of is None:
                p(f'{case:24s}  fused missing: {ro.get("error", "")[:80]}')
                continue
            bit_chain = all(_bit(oc_b[k2], oc_o[k2]) for k2 in ('o', 'hkt', 'hvt'))
            e = {k2: rel_l2(oc_b[k2], of[k2]) for k2 in ('o', 'hkt', 'hvt')}
            em = max(e.values())
            mr = max(max_rel(oc_b[k2], of[k2]) for k2 in ('o', 'hkt', 'hvt'))
            worst = max(worst, em)
            ref = ref_prefill(rb['inputs'])
            ef = rel_l2(of['o'], ref['o'])
            ec = rel_l2(oc_b['o'], ref['o'])
            verdict = 'OK (<=3e-3)' if em <= 3e-3 else 'CHECK (>3e-3)'
            p(f'{case:24s} {str(bit_chain):>8s} {em:12.3e} {mr:12.3e} {ef:10.3e} {ec:11.3e}  {verdict}')
        p(f'worst fused-vs-chain rel-L2: {worst:.3e} (threshold 3e-3)')

    # ---- swa decode chain (window>64, item 3 partial) ----
    base = load(bd, 'swa_decode_chain.pt')
    opt = load(od, 'swa_decode_chain.pt')
    if base and opt:
        p('\n=== [item 3] window>64 decode: naive chain vs flash-attn vs fp64 ===')
        for case in sorted(opt):
            outs = opt[case]['outputs']
            naive, fa, ref = outs['naive_chain']['o'], outs.get('flash_attn', {}).get('o'), outs['ref_fp64']['o']
            p(f'{case}:')
            p(f'   naive_chain  vs fp64: rel-L2 {rel_l2(naive, ref):.3e}  max-abs {max_abs(naive, ref):.3e}')
            if fa is not None:
                p(f'   flash_attn   vs fp64: rel-L2 {rel_l2(fa, ref):.3e}  max-abs {max_abs(fa, ref):.3e}')
                p(f'   flash_attn   vs naive_chain: rel-L2 {rel_l2(fa, naive):.3e}')

    # ---- training path ----
    base = load(bd, 'training.pt')
    opt = load(od, 'training.pt')
    if base and opt:
        p('\n=== [safety] training path chunk_nha fwd+bwd (untouched code) ===')
        allbit = True
        for case in sorted(opt):
            rb, ro = base[case], opt[case]
            if rb.get('error') or ro.get('error'):
                p(f'{case}: baseline_err={rb.get("error", "")[:60]} opt_err={ro.get("error", "")[:60]}')
                allbit = False
                continue
            keys = sorted(set(rb['outputs']) & set(ro['outputs']))
            bits = {k2: _bit(rb['outputs'][k2], ro['outputs'][k2]) for k2 in keys}
            ok = all(bits.values())
            allbit &= ok
            bad = [k2 for k2, v2 in bits.items() if not v2]
            p(f'{case}: {"bit-exact" if ok else "DIFF in " + ",".join(bad)}')
        p(f'training path: {"ALL bit-exact" if allbit else "MISMATCH FOUND"}')

    # ---- model level ----
    p('\n=== [e2e] model-level prefill + 16 decode steps ===')
    p('(note: item 3 stores cache/states at kv-head count, baseline at query-head')
    p(' count; states/windows are group-duplicated in baseline, so the opt side is')
    p(' expanded to the baseline layout before comparison)')

    def _align(a, b):
        # expand a ([B, H, ...] kv-head) to b's ([B, HQ, ...]) layout if needed
        if a.shape == b.shape:
            return a, b
        if a.dim() == 4 and a.shape[1] != b.shape[1] and b.shape[1] % a.shape[1] == 0:
            return a.repeat_interleave(b.shape[1] // a.shape[1], dim=1), b
        return a, b

    for f in sorted(os.listdir(od)):
        if not f.startswith('model_'):
            continue
        case = f[:-3]
        rb = load(bd, f)
        ro = load(od, f)
        if rb is None:
            p(f'{case}: baseline dump missing')
            continue
        if rb.get('error') or ro.get('error'):
            p(f'{case}: baseline_err={str(rb.get("error"))[:120]}')
            p(f'{case}: opt_err={str(ro.get("error"))[:120]}')
            continue
        keys = [k2 for k2 in ro['outputs'] if k2.startswith(('prefill', 'step', 'layer'))]
        worst_case = 0.0
        nbit = 0
        rows = []
        for k2 in sorted(keys):
            a, b = _align(ro['outputs'][k2], rb['outputs'][k2])
            ma, mr, rl, note = _cmp_row(a, b)
            if rl == '' and ma == '0':
                nbit += 1
            try:
                worst_case = max(worst_case, float(rl))
            except ValueError:
                pass
            rows.append((k2, ma, mr, rl))
        p(f'{case}: {nbit}/{len(keys)} tensors bit-exact, worst rel-L2 {worst_case:.3e}')
        for k2, ma, mr, rl in rows:
            if k2.startswith(('prefill', 'step00', 'step07', 'step15', 'layer0_hk', 'layer3_hk')):
                p(f'   {k2:22s} max-abs {ma:>10s} max-rel {mr:>10s} rel-L2 {rl:>10s}')

    if args.report:
        with open(args.report, 'w') as fh:
            fh.write('\n'.join(rep))


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='mode', required=True)
    d = sub.add_parser('dump')
    d.add_argument('--impl', required=True, choices=['baseline', 'opt'])
    d.add_argument('--out', required=True)
    d.add_argument('--only', default='',
                   help='comma list: fused_recurrent,decode,chunk_fused,swa,training,model')
    c = sub.add_parser('compare')
    c.add_argument('--baseline', required=True)
    c.add_argument('--opt', required=True)
    c.add_argument('--report', default='')
    args = ap.parse_args()

    if args.mode == 'compare':
        compare(args)
        return

    assert torch.cuda.is_available(), 'CUDA required'
    os.makedirs(args.out, exist_ok=True)
    only = set(x for x in args.only.split(',') if x)
    def want(name):
        return not only or name in only
    if want('fused_recurrent'):
        dump_fused_recurrent(args.out)
    if want('decode'):
        dump_decode(args.out)
    if want('chunk_fused'):
        dump_chunk_fused(args.out)
    if want('swa'):
        dump_swa_decode_chain(args.out)
    if want('training'):
        dump_training(args.out)
    if want('model'):
        dump_model(args.out)
    with open(os.path.join(args.out, 'manifest.json'), 'w') as fh:
        json.dump({'impl': args.impl, 'pythonpath': os.environ.get('PYTHONPATH', '')}, fh, indent=2)


if __name__ == '__main__':
    main()
