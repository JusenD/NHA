# Copyright (c) 2024-2026, Jusen Du, Songlin Yang, Yu Zhang

from __future__ import annotations

import torch
from einops import reduce

from fla.ops.gsa.chunk import (
    chunk_gsa_fwd as _fla_chunk_gsa_fwd,
    chunk_gsa_bwd_v,
    chunk_gsa_bwd_k,
)
from fla.ops.utils.softmax import softmax_bwd


def chunk_gsa_fwd_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    scale: float = 1.,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple:
    r"""
    ``chunk_gsa_fwd`` with additional LSE output.

    Calls fla's ``chunk_gsa_fwd`` internally, then computes LSE from the
    returned slot logits ``ok``.

    Returns:
        ``(Ak, hk, hkt, ok, p, Av, hv, hvt, ov, lse)``
        where ``lse = logsumexp(ok, dim=-1)``
    """
    Ak, hk, hkt, ok, p, Av, hv, hvt, ov = _fla_chunk_gsa_fwd(
        q=q, k=k, v=v, s=s, g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )

    lse = torch.logsumexp(ok.float(), dim=-1)

    return Ak, hk, hkt, ok, p, Av, hv, hvt, ov, lse


def chunk_gsa_bwd_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    ok: torch.Tensor,
    p: torch.Tensor,
    A: tuple[torch.Tensor | None, torch.Tensor],
    h: tuple[torch.Tensor | None, torch.Tensor | None],
    initial_state: tuple[torch.Tensor, torch.Tensor] | None,
    scale: float,
    do: torch.Tensor,
    dht: tuple[torch.Tensor | None, torch.Tensor | None],
    dlse: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    r"""
    ``chunk_gsa_bwd`` with ``dlse`` support.

    When ``dlse`` is provided, it is folded into the softmax backward gradient::

        dok += dlse.unsqueeze(-1) * p

    This is because  :math:`\partial \text{lse} / \partial \text{ok}_j = p_j`  (softmax probabilities).
    """
    hk0, hv0 = None, None
    if initial_state is not None:
        hk0, hv0 = initial_state

    _, Av = A
    hk, hv = h
    dhkt, dhvt = dht

    # V-branch backward
    qv = p.to(q.dtype)
    dqv, dsv, dv, dg, dhv0 = chunk_gsa_bwd_v(
        q=qv, k=s, v=v, g=g,
        h0=hv0, h=hv, A=Av,
        do=do, dht=dhvt, dg=None,
        scale=1.,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )

    # Softmax backward: p, dqv -> dok
    dok = softmax_bwd(p, dqv, dtype=ok.dtype)

    # Fold dlse gradient: ∂lse/∂ok_j = p_j
    if dlse is not None:
        dok = dok + (dlse.unsqueeze(-1) * p).to(dok.dtype)

    # K-branch backward
    dq, dk, dsk, dg, dhk0 = chunk_gsa_bwd_k(
        q=q, k=k, v=s, g=g,
        h0=hk0, h=hk, o=ok,
        do=dok, dht=dhkt, dg=dg,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )

    ds = dsv.add_(dsk)
    if q.shape[2] != k.shape[2]:
        dk, dv, ds, dg = map(
            lambda x: reduce(x, 'b t (h g) d -> b t h d', 'sum', h=k.shape[2]),
            (dk, dv, ds, dg),
        )
    dg = dg.to(s.dtype)
    return dq, dk, dv, ds, dg, dhk0, dhv0
