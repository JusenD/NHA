# Copyright (c) 2024-2026, Jusen Du, Songlin Yang, Yu Zhang

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
import triton
from einops import rearrange

from fla.ops.utils.cumsum import chunk_local_cumsum
from .chunk_gsa import chunk_gsa_fwd_with_lse, chunk_gsa_bwd_with_lse

_flash_attn_fwd_fn = None
_flash_attn_bwd_fn = None
_flash_attn_backend = None  # 'v3' | 'v2_new' | 'v2_old' | None

try:
    # flash-attn v3
    from flash_attn_interface import (
        _flash_attn_varlen_forward as _fa3_fwd,
        _flash_attn_varlen_backward as _fa3_bwd,
    )
    _flash_attn_fwd_fn = _fa3_fwd
    _flash_attn_bwd_fn = _fa3_bwd
    _flash_attn_backend = 'v3'
except ImportError:
    try:
        # flash-attn v2 (>= 2.6)
        from flash_attn.flash_attn_interface import (
            _flash_attn_varlen_forward as _fa2_fwd,
            _flash_attn_varlen_backward as _fa2_bwd,
        )
        _flash_attn_fwd_fn = _fa2_fwd
        _flash_attn_bwd_fn = _fa2_bwd
        import flash_attn as _fa_mod
        import packaging.version
        _fa_ver = packaging.version.Version(_fa_mod.__version__)
        if _fa_ver >= packaging.version.Version('2.6'):
            _flash_attn_backend = 'v2_new'  # uses window_size_left/right
        else:
            _flash_attn_backend = 'v2_old'  # uses window_size=(left, right)
    except ImportError:
        import warnings
        warnings.warn(
            "flash-attn is not installed. chunk_nha requires flash-attn. "
            "Install via: pip install flash-attn --no-build-isolation",
            category=ImportWarning,
        )


def _packed_shift_right(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    shift: int,
) -> torch.Tensor:
    """Shift packed varlen tokens to the right within each sequence.

    Args:
        x: [total_tokens, H, D]
        cu_seqlens: [num_seq + 1]
        shift: number of positions to shift right

    Returns:
        y: [total_tokens, H, D] with zero-padded leading tokens
    """
    if shift <= 0:
        return x
    y = torch.zeros_like(x)
    num_seq = cu_seqlens.numel() - 1
    for i in range(num_seq):
        st = int(cu_seqlens[i].item())
        en = int(cu_seqlens[i + 1].item())
        if en - st <= shift:
            continue
        y[st + shift:en] = x[st:en - shift]
    return y


def _packed_shift_right_backward(
    grad_y: torch.Tensor,
    cu_seqlens: torch.Tensor,
    shift: int,
) -> torch.Tensor:
    """Backward of _packed_shift_right."""
    if shift <= 0:
        return grad_y
    grad_x = torch.zeros_like(grad_y)
    num_seq = cu_seqlens.numel() - 1
    for i in range(num_seq):
        st = int(cu_seqlens[i].item())
        en = int(cu_seqlens[i + 1].item())
        if en - st <= shift:
            continue
        grad_x[st:en - shift] = grad_y[st + shift:en]
    return grad_x


def _batch_shift_right(
    x: torch.Tensor,
    shift: int,
) -> torch.Tensor:
    """Shift batched [B, T, H, D] tokens right by `shift` positions, zero-padded."""
    if shift <= 0:
        return x
    B, T = x.shape[0], x.shape[1]
    y = torch.zeros_like(x)
    if T > shift:
        y[:, shift:] = x[:, :T - shift]
    return y


def _batch_shift_right_backward(
    grad_y: torch.Tensor,
    shift: int,
) -> torch.Tensor:
    """Backward of _batch_shift_right."""
    if shift <= 0:
        return grad_y
    B, T = grad_y.shape[0], grad_y.shape[1]
    grad_x = torch.zeros_like(grad_y)
    if T > shift:
        grad_x[:, :T - shift] = grad_y[:, shift:]
    return grad_x


def _chunk_gsa_fwd_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    batched: bool = False,
) -> tuple:
    """Wrapper around chunk_gsa_fwd_with_lse with varlen convention.

    Inputs are flat [T, H, D] (or [B, T, H, D] when `batched`); we unsqueeze
    the batch dim for fla in the flat case.
    """
    if not batched:
        q, k, v, s, g = (x.unsqueeze(0) for x in [q, k, v, s, g])
        assert cu_seqlens is not None

    if g is None:
        z = s.float().logcumsumexp(2)
        g = torch.cat((z[:, :, :1], z[:, :, :-1]), 1) - z
        s = torch.exp(s - z).to(k.dtype)

    hk0, hv0 = (None, None) if initial_state is None else initial_state
    chunk_size = min(64, max(16, triton.next_power_of_2(q.shape[1])))

    g_org, g = g, chunk_local_cumsum(g, chunk_size, cu_seqlens=None if batched else cu_seqlens)

    Ak, hk, hkt, ok, p, Av, hv, hvt, ov, lse = chunk_gsa_fwd_with_lse(
        q=q, k=k, v=v, s=s, g=g,
        initial_state=(hk0, hv0),
        output_final_state=output_final_state,
        scale=scale,
        cu_seqlens=None if batched else cu_seqlens.int(),
        chunk_size=chunk_size,
    )

    if not batched:
        ov = ov.squeeze(0)
        g = g.squeeze(0)
        lse = lse.squeeze(0) if lse is not None else None

    return ov, lse, g, hkt, hvt, (ok, p, Av, hk0, hv0, hk, hv), chunk_size


def _chunk_gsa_bwd_wrapper(
    q, k, v, s, g,
    ok, p, A, h, initial_state,
    scale, do, dht, dlse,
    cu_seqlens, chunk_size,
    batched: bool = False,
):
    """Wrapper around chunk_gsa_bwd_with_lse."""
    if not batched:
        q, k, v, s, g = (x.unsqueeze(0) for x in [q, k, v, s, g])
        do = do.unsqueeze(0)

    dq, dk, dv, ds, dg, dhk0, dhv0 = chunk_gsa_bwd_with_lse(
        q=q, k=k, v=v, s=s, g=g,
        ok=ok, p=p, A=A, h=h,
        initial_state=initial_state,
        scale=scale, do=do, dht=dht, dlse=dlse,
        cu_seqlens=None if batched else cu_seqlens.int(),
        chunk_size=chunk_size,
    )

    if not batched:
        dq, dk, dv, ds, dg = (x.squeeze(0) for x in [dq, dk, dv, ds, dg])
    return dq, dk, dv, ds, dg, dhk0, dhv0


def _build_window_kwargs(window_size_left: int, window_size_right: int) -> dict:
    """Return flash-attn kwargs for window size, adapting to the installed version."""
    if _flash_attn_backend == 'v2_new' or _flash_attn_backend == 'v3':
        return {'window_size_left': window_size_left, 'window_size_right': window_size_right}
    else:
        return {'window_size': (window_size_left, window_size_right)}


def _flash_attn_fwd_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    softmax_scale: float,
    causal: bool = True,
    window_size_left: int = -1,
    window_size_right: int = -1,
):
    """Run flash_attn varlen forward, return (output, lse).

    Output shapes:
        output: [total_tokens, H, D]
        lse:    [total_tokens, H]
    """
    assert _flash_attn_fwd_fn is not None, (
        "flash-attn is required for chunk_nha. "
        "Install: pip install flash-attn --no-build-isolation"
    )

    # flash_attn requires fp16/bf16
    input_dtype = q.dtype
    if q.dtype == torch.float32:
        q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)

    win_kwargs = _build_window_kwargs(window_size_left, window_size_right)

    if _flash_attn_backend == 'v3':
        # v3: returns (_, q, k, v, out, lse, raw_max_logits)
        ret = _flash_attn_fwd_fn(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale, causal=causal,
            **win_kwargs,
        )
        _, _, _, _, out, lse, _ = ret
    else:
        # v2: returns various tuples depending on version
        extra = {
            'dropout_p': 0.0,
            'softcap': 0.0,
            'return_softmax': False,
            'alibi_slopes': None,
        }
        # v2_old uses 'logits_cap' instead of 'softcap'
        if _flash_attn_backend == 'v2_old':
            extra.pop('softcap')
            extra['logits_cap'] = 0.0
            extra.pop('alibi_slopes', None)

        ret = _flash_attn_fwd_fn(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale, causal=causal,
            **win_kwargs,
            **extra,
        )
        # Typical v2 return: (out, lse, ...) or longer tuple
        if len(ret) == 5:
            out, lse, _, _, _ = ret
        elif len(ret) == 9:
            _, _, _, _, out, lse, _, _, _ = ret
        else:
            out, lse = ret[0], ret[1]

    # lse shape: [H, total_tokens] -> [total_tokens, H]
    lse = lse.t().contiguous()
    # Cast back to original dtype if we promoted to bf16
    if out.dtype != input_dtype:
        out = out.to(input_dtype)
    return out, lse


def _flash_attn_bwd_wrapper(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    softmax_scale: float,
    causal: bool = True,
    window_size_left: int = -1,
    window_size_right: int = -1,
):
    """Run flash_attn varlen backward."""
    assert _flash_attn_bwd_fn is not None

    # flash_attn only supports fp16/bf16 — cast if needed
    input_dtype = q.dtype
    if input_dtype == torch.float32:
        compute_dtype = torch.bfloat16
        dout = dout.to(compute_dtype)
        q, k, v = q.to(compute_dtype), k.to(compute_dtype), v.to(compute_dtype)
        out = out.to(compute_dtype)

    # softmax_lse needs [H, total_tokens] for the kernel
    lse_for_kernel = softmax_lse.t().contiguous()
    win_kwargs = _build_window_kwargs(window_size_left, window_size_right)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    extra_bwd = {}
    if _flash_attn_backend in ('v2_new', 'v2_old'):
        extra_bwd = {
            'dropout_p': 0.0,
            'softcap': 0.0,
            'alibi_slopes': None,
            'deterministic': False,
        }
        if _flash_attn_backend == 'v2_old':
            extra_bwd.pop('softcap')
            extra_bwd['logits_cap'] = 0.0
            extra_bwd.pop('alibi_slopes', None)

    _flash_attn_bwd_fn(
        dout=dout,
        q=q, k=k, v=v,
        out=out,
        softmax_lse=lse_for_kernel,
        dq=dq, dk=dk, dv=dv,
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
        softmax_scale=softmax_scale, causal=causal,
        **win_kwargs,
        **extra_bwd,
    )

    # Cast gradients back to original dtype
    if input_dtype == torch.float32:
        dq, dk, dv = dq.to(input_dtype), dk.to(input_dtype), dv.to(input_dtype)
    return dq, dk, dv


class ChunkNativeHybridAttentionFunction(torch.autograd.Function):
    """Custom autograd function for chunk-level NHA.

    Fuses SWA (flash_attn) and GSA (chunk_gsa) with online softmax.

    Forward inputs (all flat varlen [T, H, D] unless noted):
        q_swa, k_swa : Q/K with RoPE for sliding-window attention
        q_gsa, k_gsa : Q/K without RoPE for gated-slot attention
        v            : shared value (may be zero-padded to match qk dim)
        s            : slot gate keys   [T, H, M]
        g            : slot gate values [T, H, M]
        cu_seqlens   : cumulative sequence lengths [num_seq + 1]
        max_seqlen   : max sequence length (int stored as tensor)
    """

    @staticmethod
    def forward(
        ctx,
        q_swa: torch.Tensor,
        k_swa: torch.Tensor,
        q_gsa: torch.Tensor,
        k_gsa: torch.Tensor,
        v: torch.Tensor,
        s: torch.Tensor,
        g: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        softmax_scale: float,
        window_size_left: int = -1,
        window_size_right: int = -1,
        head_first: bool = False,
        gsa_kv_shift: int = 0,
        hk0: torch.Tensor | None = None,
        hv0: torch.Tensor | None = None,
        output_final_state: bool = False,
        input_is_varlen: bool = True,
        batch_size: int = 1,
    ):
        # Ensure contiguous
        q_swa = q_swa.contiguous()
        k_swa = k_swa.contiguous()
        q_gsa = q_gsa.contiguous()
        k_gsa = k_gsa.contiguous()
        v_swa = v.contiguous()
        v_gsa = v_swa
        s = s.contiguous()
        g = g.contiguous()

        # Optional: shift GSA KV to focus on older context
        if gsa_kv_shift > 0:
            if input_is_varlen:
                k_gsa = _packed_shift_right(k_gsa, cu_seqlens, gsa_kv_shift)
                v_gsa = _packed_shift_right(v_gsa, cu_seqlens, gsa_kv_shift)
                s = _packed_shift_right(s, cu_seqlens, gsa_kv_shift)
                g = _packed_shift_right(g, cu_seqlens, gsa_kv_shift)
            else:
                T_batch = max_seqlen
                q_gsa = rearrange(q_gsa, '(b t) h d -> b t h d', b=batch_size, t=T_batch).contiguous()
                k_gsa = _batch_shift_right(rearrange(k_gsa, '(b t) h d -> b t h d', b=batch_size, t=T_batch).contiguous(), gsa_kv_shift)
                v_gsa = _batch_shift_right(rearrange(v_gsa, '(b t) h d -> b t h d', b=batch_size, t=T_batch).contiguous(), gsa_kv_shift)
                s = _batch_shift_right(rearrange(s, '(b t) h m -> b t h m', b=batch_size, t=T_batch).contiguous(), gsa_kv_shift)
                g = _batch_shift_right(rearrange(g, '(b t) h m -> b t h m', b=batch_size, t=T_batch).contiguous(), gsa_kv_shift)
        elif not input_is_varlen:
            T_batch = max_seqlen
            q_gsa = rearrange(q_gsa, '(b t) h d -> b t h d', b=batch_size, t=T_batch).contiguous()
            k_gsa = rearrange(k_gsa, '(b t) h d -> b t h d', b=batch_size, t=T_batch).contiguous()
            v_gsa = rearrange(v_gsa, '(b t) h d -> b t h d', b=batch_size, t=T_batch).contiguous()
            s = rearrange(s, '(b t) h m -> b t h m', b=batch_size, t=T_batch).contiguous()
            g = rearrange(g, '(b t) h m -> b t h m', b=batch_size, t=T_batch).contiguous()

        # ---- Branch 1: SWA via flash_attn ----
        output_swa, lse_swa = _flash_attn_fwd_wrapper(
            q=q_swa, k=k_swa, v=v_swa,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            softmax_scale=softmax_scale,
            causal=True,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )

        # ---- Branch 2: GSA via chunk_gsa ----
        # batched inputs stay in true batch mode: fla 0.5.1's varlen k-path
        # produces nondeterministic results for more than one sequence on
        # some stacks, while the batch kernels are exact.
        output_gsa, lse_gsa, g_cum, hkt, hvt, saved_gsa, chunk_size = _chunk_gsa_fwd_wrapper(
            q_gsa, k_gsa, v_gsa,
            s=s, g=g,
            scale=softmax_scale,
            cu_seqlens=cu_seqlens,
            initial_state=(hk0, hv0),
            output_final_state=output_final_state,
            batched=not input_is_varlen,
        )
        if not input_is_varlen:
            T_batch = max_seqlen
            output_gsa = rearrange(output_gsa, 'b t h d -> (b t) h d', b=batch_size, t=T_batch).contiguous()

        # Fix lse_gsa shape to match lse_swa: [T, H]
        if not input_is_varlen:
            lse_gsa = rearrange(lse_gsa, 'b t h -> (b t) h', b=batch_size, t=T_batch).contiguous()
        elif lse_gsa.shape[0] == q_swa.shape[1]:  # [H, T] -> [T, H]
            lse_gsa = lse_gsa.t().contiguous()
        else:
            lse_gsa = lse_gsa.contiguous()

        # ---- Online softmax fusion ----
        lse_max = torch.maximum(lse_swa, lse_gsa)
        w_swa = torch.exp(lse_swa - lse_max)
        w_gsa = torch.exp(lse_gsa - lse_max)
        sum_exp = w_swa + w_gsa
        lse_global = torch.log(sum_exp) + lse_max

        r_gsa = (w_gsa / sum_exp).unsqueeze(-1)

        output = (
            w_swa.unsqueeze(-1) * output_swa + w_gsa.unsqueeze(-1) * output_gsa
        ) / sum_exp.unsqueeze(-1)
        output = output.to(q_swa.dtype)

        # ---- Save for backward ----
        ok, p, Av, hk0, hv0, hk, hv = saved_gsa
        ctx.save_for_backward(
            q_swa, k_swa, q_gsa, k_gsa, v_swa, v_gsa, s, g_cum,
            output, output_gsa, lse_global, cu_seqlens, r_gsa,
            ok, p, Av, hk0, hv0, hk, hv,
        )
        ctx.max_seqlen = max_seqlen
        ctx.softmax_scale = softmax_scale
        ctx.window_size_left = window_size_left
        ctx.window_size_right = window_size_right
        ctx.chunk_size = chunk_size
        ctx.gsa_kv_shift = gsa_kv_shift
        ctx.input_is_varlen = input_is_varlen
        ctx.batch_size = batch_size

        return output, lse_swa, lse_gsa, hkt, hvt

    @staticmethod
    def backward(ctx, d_output, d_lse_swa, d_lse_gsa, dhkt, dhvt):
        (
            q_swa, k_swa, q_gsa, k_gsa, v_swa, v_gsa, s, g_cum,
            output, output_gsa, lse_global, cu_seqlens, r_gsa,
            ok, p, Av, hk0, hv0, hk, hv,
        ) = ctx.saved_tensors

        d_output = d_output.contiguous().to(q_swa.dtype)

        # ---- 1. SWA backward ----
        dq_swa, dk_swa, dv_swa = _flash_attn_bwd_wrapper(
            dout=d_output,
            q=q_swa, k=k_swa, v=v_swa,
            out=output,
            softmax_lse=lse_global,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            softmax_scale=ctx.softmax_scale,
            causal=True,
            window_size_left=ctx.window_size_left,
            window_size_right=ctx.window_size_right,
        )

        # ---- 2. GSA backward ----
        dov = (d_output * r_gsa).to(q_gsa.dtype)
        dlse_fusion = (dov * (output_gsa - output)).sum(dim=-1).float()

        if d_lse_gsa is not None:
            dlse_fusion = dlse_fusion + d_lse_gsa

        if not ctx.input_is_varlen:
            T_batch = ctx.max_seqlen
            dov = rearrange(dov, '(b t) h d -> b t h d', b=ctx.batch_size, t=T_batch).contiguous()
            dlse_fusion = rearrange(dlse_fusion, '(b t) h -> b t h', b=ctx.batch_size, t=T_batch).contiguous()

        dq_gsa, dk_gsa, dv_gsa, ds, dg, dhk0, dhv0 = _chunk_gsa_bwd_wrapper(
            q=q_gsa, k=k_gsa, v=v_gsa, s=s, g=g_cum,
            ok=ok, p=p, A=(None, Av), h=(hk, hv),
            initial_state=(hk0, hv0),
            scale=ctx.softmax_scale,
            do=dov, dht=(dhkt, dhvt), dlse=dlse_fusion,
            cu_seqlens=cu_seqlens,
            chunk_size=ctx.chunk_size,
            batched=not ctx.input_is_varlen,
        )

        # Undo shift in backward
        if ctx.gsa_kv_shift > 0:
            if ctx.input_is_varlen:
                dk_gsa = _packed_shift_right_backward(dk_gsa, cu_seqlens, ctx.gsa_kv_shift)
                dv_gsa = _packed_shift_right_backward(dv_gsa, cu_seqlens, ctx.gsa_kv_shift)
                ds = _packed_shift_right_backward(ds, cu_seqlens, ctx.gsa_kv_shift)
                dg = _packed_shift_right_backward(dg, cu_seqlens, ctx.gsa_kv_shift)
            else:
                dk_gsa = _batch_shift_right_backward(dk_gsa, ctx.gsa_kv_shift)
                dv_gsa = _batch_shift_right_backward(dv_gsa, ctx.gsa_kv_shift)
                ds = _batch_shift_right_backward(ds, ctx.gsa_kv_shift)
                dg = _batch_shift_right_backward(dg, ctx.gsa_kv_shift)

        if not ctx.input_is_varlen:
            T_batch = ctx.max_seqlen
            dq_gsa = rearrange(dq_gsa, 'b t h d -> (b t) h d', b=ctx.batch_size, t=T_batch).contiguous()
            dk_gsa = rearrange(dk_gsa, 'b t h d -> (b t) h d', b=ctx.batch_size, t=T_batch).contiguous()
            dv_gsa = rearrange(dv_gsa, 'b t h d -> (b t) h d', b=ctx.batch_size, t=T_batch).contiguous()
            ds = rearrange(ds, 'b t h m -> (b t) h m', b=ctx.batch_size, t=T_batch).contiguous()
            dg = rearrange(dg, 'b t h m -> (b t) h m', b=ctx.batch_size, t=T_batch).contiguous()

        # V is shared: accumulate gradients
        dv = dv_swa.add_(dv_gsa)

        # Return grads for all 19 forward args (after ctx)
        return (dq_swa, dk_swa, dq_gsa, dk_gsa, dv, ds, dg,
                None, None, None, None, None, None, None,
                dhk0, dhv0, None, None, None)


@torch.compiler.disable
def chunk_nha(
    q_swa: torch.Tensor,
    k_swa: torch.Tensor,
    q_gsa: torch.Tensor,
    k_gsa: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    scale: float | None = None,
    window_size_left: int = -1,
    window_size_right: int = -1,
    head_first: bool | None = False,
    gsa_kv_shift: int = 0,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    r"""
    Chunk-level Native Hybrid Attention.

    Fuses sliding-window attention (SWA) and gated-slot attention (GSA)
    through online softmax.

    Args:
        q_swa (torch.Tensor):
            Query with RoPE for SWA. Shape ``[B, T, H, D]``.
        k_swa (torch.Tensor):
            Key with RoPE for SWA. Shape ``[B, T, H, D]``.
        q_gsa (torch.Tensor):
            Query without RoPE for GSA. Shape ``[B, T, H, D]``.
        k_gsa (torch.Tensor):
            Feature-mapped key for GSA. Shape ``[B, T, H, D]``.
        v (torch.Tensor):
            Shared value tensor. Shape ``[B, T, H, V]``.
        s (torch.Tensor):
            Slot representations. Shape ``[B, T, H, M]``.
        g (torch.Tensor):
            Forget gates (log-space). Shape ``[B, T, H, M]``.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape ``[N+1]`` used for variable-length training,
            consistent with the FlashAttention API.
        max_seqlen (Optional[int]):
            Maximum sequence length (required for varlen mode).
        scale (Optional[float]):
            Scale factor for attention scores.
            If not provided, it will default to ``1 / sqrt(D)``. Default: ``None``.
        window_size_left (int):
            Left window size for SWA (``-1`` = unlimited). Default: ``-1``.
        window_size_right (int):
            Right window size for SWA (``-1`` = unlimited, ``0`` = causal). Default: ``-1``.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: ``False``.
            This argument has been deprecated.
        gsa_kv_shift (int):
            Shift GSA K/V right by this many positions. Default: ``0``.
        initial_state (Optional[Tuple[torch.Tensor, torch.Tensor]]):
            Initial GSA states ``(hk0, hv0)`` of shape ``[N, H, K, M]`` /
            ``[N, H, M, V]`` (``N`` = batch size, or the number of sequences
            in varlen mode). Default: ``None``.
        output_final_state (bool):
            Whether to return the final GSA states. Default: ``False``.

    Returns:
        o (torch.Tensor):
            Fused attention output. Shape ``[B, T, H, V]``.
        lse_swa (torch.Tensor):
            Log-sum-exp from SWA branch. Shape ``[B, T, H]``.
        lse_gsa (torch.Tensor):
            Log-sum-exp from GSA branch. Shape ``[B, T, H]``.
        final_state (Tuple[torch.Tensor, torch.Tensor]):
            ``(hkt, hvt)`` fp32 final GSA states ``[N, H, K, M]`` /
            ``[N, H, M, V]``, returned (as a 4th element) only when
            ``output_final_state=True``.
    """
    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead.",
        )
    if scale is None:
        scale = q_swa.shape[-1] ** -0.5

    is_varlen = cu_seqlens is not None

    assert q_swa.dim() == 4, "Expected [B, T, H, D] for input"
    B, T, H, D = q_swa.shape
    M = s.shape[-1]

    if not is_varlen:
        # Batched mode: flatten to varlen
        # Build cu_seqlens
        cu_seqlens = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=q_swa.device)
        max_seqlen = T

    hk0, hv0 = (None, None) if initial_state is None else initial_state

    q_swa = rearrange(q_swa, 'b t h d -> (b t) h d').contiguous()
    k_swa = rearrange(k_swa, 'b t h d -> (b t) h d').contiguous()
    q_gsa = rearrange(q_gsa, 'b t h d -> (b t) h d').contiguous()
    k_gsa = rearrange(k_gsa, 'b t h d -> (b t) h d').contiguous()
    v = rearrange(v, 'b t h d -> (b t) h d').contiguous()
    s = rearrange(s, 'b t h m -> (b t) h m').contiguous()
    g = rearrange(g, 'b t h m -> (b t) h m').contiguous()

    output, lse_swa, lse_gsa, hkt, hvt = ChunkNativeHybridAttentionFunction.apply(
        q_swa, k_swa, q_gsa, k_gsa, v, s, g,
        cu_seqlens, max_seqlen, scale,
        window_size_left, window_size_right,
        False, gsa_kv_shift,
        hk0, hv0, output_final_state,
        is_varlen, B,
    )

    # Reshape back to [B, T, H, D]
    output = rearrange(output, '(b t) h d -> b t h d', b=B, t=T)
    lse_swa = rearrange(lse_swa, '(b t) h -> b t h', b=B, t=T)
    lse_gsa = rearrange(lse_gsa, '(b t) h -> b t h', b=B, t=T)
    # the final-state tuple is only returned when requested, keeping the
    # long-standing 3-tuple return for existing callers
    if output_final_state:
        return output, lse_swa, lse_gsa, (hkt, hvt)
    return output, lse_swa, lse_gsa
