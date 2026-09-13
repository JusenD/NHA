# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

import os
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .configuration_qwen3_moe_nha import Qwen3MoeNHAConfig
from nha_fla.ops.nha_naive import chunk_nha, fused_recurrent_nha, fused_recurrent_nha_decode

from nha_fla.models.nha_cache import NHACache

from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    repeat_kv,
    apply_rotary_pos_emb,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeRMSNorm,
)

from transformers.utils import (
    logging,
)

try:
    from flash_attn import flash_attn_func
except Exception:  # flash-attn is only needed for the full-attention layers
    flash_attn_func = None

logger = logging.get_logger(__name__)

# ablation switches (default on): NHA_FUSED_PREFILL=0 forces the original
# chunk_nha prefill chain; NHA_DECODE_FUSED_ROTARY=0 keeps the eager rotary
# plus the caller-computed slot residual on the decode path (bit-exact
# against the pre-optimization layout).
_FUSED_PREFILL_ENV = os.environ.get("NHA_FUSED_PREFILL", "1") != "0"
_DECODE_FUSED_ROTARY_ENV = os.environ.get("NHA_DECODE_FUSED_ROTARY", "1") != "0"


class Qwen3MoeNativeHybridAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3MoeNHAConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = config.num_key_value_heads

        self.num_slots = config.num_slots
        self.window_size = config.window_size
        self.gate_logit_normalizer = 8

        if config.block_size is not None and config.transformer_idx is not None:
            if self.layer_idx % config.block_size == config.transformer_idx:
                self.window_size = 2048
            else:
                self.window_size = config.window_size

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = getattr(config, "sliding_window", None)

        self.g_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.num_slots, bias=False)
        self.rotary_emb = Qwen3MoeRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[NHACache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        g = self.g_proj(hidden_states).view(bsz, q_len, -1, self.num_slots).transpose(1, 2)

        last_state = None
        if past_key_value is not None and len(past_key_value) > self.layer_idx:
            last_state = past_key_value[self.layer_idx]

        # k/v/g stay in the kv-head layout; they are expanded to the query-head
        # count only where the consumer requires it (the prefill chain and the
        # full-attention training path).
        q = query_states
        k = key_states
        v = value_states
        g = F.logsigmoid(g) / self.gate_logit_normalizer # (b, h, n, m)

        # single-token decode: the current token's slot residual s is only
        # consumed by the prefill/training chain; the decode op recomputes the
        # residual of the *popped* token from its log-gate in-kernel, so
        # computing s here (exp + rsub + cast + mask-mul + transpose) is dead.
        decode_step = (not self.training) and q_len == 1
        if decode_step:
            s = None
        else:
            s = 1 - torch.exp(g).to(g.dtype)

        # dealing with left-padding
        if attention_mask is not None:
            t_len = q_len if s is None else s.shape[2]
            if len(attention_mask.shape) == 4:
                if s is not None:
                    s = s.mul_(attention_mask[:, :, :, -t_len:])
                v = v.mul_(attention_mask[:, :, :, -t_len:])
            else:
                if s is not None:
                    s = s.mul_(attention_mask[:, None, -t_len:, None])
                v = v.mul_(attention_mask[:, None, -v.shape[2]:, None])

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None

        if s is None:
            q, k, v, g = (x.transpose(1, 2).contiguous() for x in (q, k, v, g))
        else:
            q, k, v, s, g = (x.transpose(1, 2).contiguous() for x in (q, k, v, s, g))

        if past_key_value is not None:
            # prepare gsa input
            last_k, last_v, last_g = past_key_value.get_pop_kvf(self.layer_idx, self.window_size)

            if last_k is None:
                # dummy no-op pop entry; must use kv-head count, not q-head
                # count (the fused decode op infers the group layout from it)
                b, d_k, d_v = q.shape[0], k.shape[-1], v.shape[-1]
                h = self.num_key_value_heads
                last_k, last_v, last_g = torch.zeros((b, 1, h, d_k), dtype=k.dtype, device=k.device), \
                                            torch.zeros((b, 1, h, d_v), dtype=v.dtype, device=v.device), \
                                            torch.zeros((b, 1, h, self.num_slots), dtype=g.dtype, device=g.device)
            else:
                last_k = rearrange(last_k, '... (h d) -> ... h d', h=self.num_key_value_heads)
                last_v = rearrange(last_v, '... (h d) -> ... h d', h=self.num_key_value_heads)
                last_g = rearrange(last_g, '... (h d) -> ... h d', h=self.num_key_value_heads)
            # update swa cache
            k_cached, v_cached, g_cached = past_key_value.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1), g.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=0,
                cache_kwargs=dict(window_size=self.window_size)
            )['attn_state']
            cache_has_content = past_key_value.get_seq_length(self.layer_idx) > 0
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, '... (h d) -> ... h d', h=self.num_key_value_heads)
                v = rearrange(v, '... (h d) -> ... h d', h=self.num_key_value_heads)


        # Single-token decode applies rotary to the un-repeated kv heads:
        # rotary is identical across the heads of a GQA group, so rotating
        # before `repeat_kv` is equivalent and cheaper. The NHA (window <= 64)
        # layers go one step further and pass the raw keys/queries plus cos/sin
        # to the fused decode kernel, which applies the rotate-half embedding
        # in-kernel (fp32); the full-attention (window > 64) layers rotate here
        # and call flash-attn, which handles GQA natively.
        if decode_step:
            window_len = k.shape[1]
            # All layers share the same rotary config and receive the same
            # position_ids object within a forward pass, so the decode-time
            # swa cos/sin are identical across layers. Memoize on the shared
            # cache object keyed by the position_ids identity (exact reuse,
            # no GPU sync, no approximation).
            memo = getattr(past_key_value, '_swa_rope_memo', None) if past_key_value is not None else None
            if memo is not None and memo[0] is position_ids and window_len in memo[1]:
                cos, sin = memo[1][window_len]
            else:
                swa_pos = position_ids - torch.arange(window_len - 1, -1, -1, device=position_ids.device).unsqueeze(0)
                cos, sin = self.rotary_emb(v.transpose(1, 2), swa_pos)
                if past_key_value is not None:
                    # holding a reference to position_ids keeps the identity
                    # check exact (no id() reuse across steps)
                    table = memo[1] if memo is not None and memo[0] is position_ids else {}
                    table[window_len] = (cos, sin)
                    past_key_value._swa_rope_memo = (position_ids, table)
            if self.window_size <= 64 and _DECODE_FUSED_ROTARY_ENV and cos.shape[-1] == k.shape[-1]:
                # rotary folded into the fused decode kernel
                sq, sk = None, k
            else:
                # rotary on the un-repeated kv heads (equivalent, NG x cheaper)
                sq = apply_rotary_pos_emb(q, q, cos[:, -1:, ...], sin[:, -1:, ...], unsqueeze_dim=2)[0]
                sk = apply_rotary_pos_emb(k, k, cos, sin, unsqueeze_dim=2)[0]
                cos = sin = None
        else:
            k_rot = repeat_kv(k.transpose(1, 2), self.num_key_value_groups).transpose(1, 2)

            if position_embeddings is None:
                logger.warning_once(
                    "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                    "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                    "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                    "removed and `position_embeddings` will be mandatory."
                )
                cos, sin = self.rotary_emb(v, position_ids)
                sq, sk = apply_rotary_pos_emb(q, k_rot, cos, sin, unsqueeze_dim=2)
            else:
                cos, sin = position_embeddings
                sq, sk = apply_rotary_pos_emb(q, k_rot, cos, sin, unsqueeze_dim=2)

        input_dtype = q.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            q = q.to(target_dtype)
            k = k.to(target_dtype)
            v = v.to(target_dtype)
            g = g.to(target_dtype)
            if s is not None:
                s = s.to(target_dtype)
            if sq is not None:
                sq = sq.to(target_dtype)
            if sk is not None:
                sk = sk.to(target_dtype)

        if self.training or q.shape[1] > 1:
            if self.window_size <= 64:
                rotary_q, rotary_k = sq, sk

                prefix_k = torch.zeros(k.size(0), self.window_size, k.size(2), k.size(3), dtype=k.dtype, device=k.device)
                prefix_v = torch.zeros(v.size(0), self.window_size, v.size(2), v.size(3), dtype=v.dtype, device=v.device)
                prefix_s = torch.zeros(s.size(0), self.window_size, s.size(2), s.size(3), dtype=s.dtype, device=s.device)
                prefix_g = torch.zeros(g.size(0), self.window_size, g.size(2), g.size(3), dtype=g.dtype, device=g.device)

                shift_k = torch.cat([prefix_k, k], dim=1)
                shift_v = torch.cat([prefix_v, v], dim=1)
                shift_s = torch.cat([prefix_s, s], dim=1)
                shift_g = torch.cat([prefix_g, g], dim=1)

                prefix_k_rot = torch.zeros(k.size(0), self.window_size, rotary_k.size(2), rotary_k.size(3), dtype=rotary_k.dtype, device=rotary_k.device)
                rotary_k = torch.cat([prefix_k_rot, rotary_k], dim=1)
                chunk_k = repeat_kv(shift_k.transpose(1, 2), self.num_key_value_groups).transpose(1, 2)
                chunk_v = repeat_kv(shift_v.transpose(1, 2), self.num_key_value_groups).transpose(1, 2)
                chunk_s = repeat_kv(shift_s.transpose(1, 2), self.num_key_value_groups).transpose(1, 2)
                chunk_g = repeat_kv(shift_g.transpose(1, 2), self.num_key_value_groups).transpose(1, 2)

                if recurrent_state is not None and recurrent_state[0].shape[1] == self.num_key_value_heads \
                        and self.num_key_value_groups > 1:
                    recurrent_state = (
                        recurrent_state[0].repeat_interleave(self.num_key_value_groups, dim=1),
                        recurrent_state[1].repeat_interleave(self.num_key_value_groups, dim=1),
                    )

                o, recurrent_state = chunk_nha(
                    q=q,
                    k=chunk_k,
                    v=chunk_v,
                    rotary_q=rotary_q,
                    rotary_k=rotary_k,
                    window_size=self.window_size,
                    s=chunk_s,
                    g=chunk_g,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    scale=None,
                    head_first=False,
                    rotary=None,
                )
                if use_cache and self.num_key_value_groups > 1:
                    # chunk_nha tracks one state per (expanded) query head;
                    # group members are bit-identical copies, so keep one
                    # state per kv head for the decode op.
                    recurrent_state = (
                        recurrent_state[0][:, ::self.num_key_value_groups].contiguous(),
                        recurrent_state[1][:, ::self.num_key_value_groups].contiguous(),
                    )
            else:
                rotary_q, rotary_k = sq, sk
                v_hq = repeat_kv(v.transpose(1, 2), self.num_key_value_groups).transpose(1, 2)
                prefix_k = torch.zeros(k.size(0), self.num_slots, rotary_k.size(2), rotary_k.size(3), dtype=rotary_k.dtype, device=rotary_k.device)
                prefix_v = torch.zeros(k.size(0), self.num_slots, v_hq.size(2), v_hq.size(3), dtype=v_hq.dtype, device=v_hq.device)
                shift_q = torch.cat([prefix_k, rotary_q], dim=1)
                shift_k = torch.cat([prefix_k, rotary_k], dim=1)
                shift_v = torch.cat([prefix_v, v_hq], dim=1)

                sliding_window = self.naive_swa(shift_q, shift_k, self.window_size)
                sliding_window_prob = sliding_window.softmax(-1)
                o = torch.einsum('bthw,bwhd->bthd',
                                sliding_window_prob,
                                shift_v)
                o = o[:, self.num_slots:, :, :]
        else:
            if self.window_size <= 64:
                if sq is None:
                    # rotary and the slot residual are folded into the kernels
                    last_s = None
                else:
                    last_s = (1 - last_g.exp()).to(last_g.dtype)
                o, recurrent_state = fused_recurrent_nha_decode(
                    q=q,
                    sq=sq,
                    k=last_k,
                    v=last_v,
                    g=last_g,
                    sw_k=sk,
                    sw_v=v,
                    s=last_s,
                    cos=cos,
                    sin=sin,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    scale=None,
                )
            else:
                # decode with GQA-native flash-attn: sk/v keep the kv-head
                # layout, flash-attn expands the groups internally
                o = flash_attn_func(sq, sk, v, causal=False)
                recurrent_state = None

        if past_key_value is not None:
            past_key_value.update(
                recurrent_state=recurrent_state,
                layer_idx=self.layer_idx,
                offset=q.shape[1]
            )

        o = rearrange(o, 'b n h d -> b n (h d)')
        o = self.o_proj(o)

        return o, None

    def naive_swa(self, q: torch.Tensor, k: torch.Tensor, W: int):
        seq_len = q.shape[1]
        i = torch.arange(seq_len, device=q.device).view(-1, 1)  # (T, 1)
        j = torch.arange(seq_len, device=q.device).view(1, -1)  # (1, T)

        left_bound = torch.clamp(i - W + 1, min=0)          # (T, 1)
        valid_mask = (j >= left_bound) & (j <= i)                    # (T, T)

        scale_factor = 1 / (q.shape[-1] ** 0.5)
        qk = torch.einsum('bthd,bnhd->bhtn', q, k) * scale_factor

        qk = qk.masked_fill(~valid_mask.unsqueeze(0).unsqueeze(0), -1e7)

        return qk.transpose(1, 2)
