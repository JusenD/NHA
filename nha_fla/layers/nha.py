# Copyright (c) 2024-2026, Jusen Du, Songlin Yang, Yu Zhang

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import RMSNorm, RotaryEmbedding, ShortConvolution
from fla.modules.feature_map import ReLUFeatureMap, SwishFeatureMap, T2RFeatureMap
from fla.modules.layernorm import rms_norm_linear
from nha_fla.ops.nha import chunk_nha, fused_recurrent_nha

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from nha_fla.models.nha_cache import NHACache


class NativeHybridAttention(nn.Module):

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        expand_k: float = 1.,
        expand_v: float = 1.,
        num_heads: int = 4,
        num_kv_heads: int | None = None,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        num_slots: int | None = None,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-5,
        gate_logit_normalizer: int = 8,
        feature_map: str = 'swish',
        use_output_gate: bool = False,
        use_norm: bool = True,
        fuse_norm: bool = True,
        layer_idx: int | None = None,
        scale: float | None = 1.,
        window_size: int = 2048,
        rope_theta: float = 10000.,
        max_position_embeddings: int | None = None,
        gsa_aux_loss_enable: bool = False,
        gsa_aux_loss_coeff: float | None = None,
        gsa_aux_loss_target_ratio: float = 1.0,
        gsa_kv_shift: int = 0,
        **kwargs,
    ) -> NativeHybridAttention:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups
        self.head_k_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.gate_logit_normalizer = gate_logit_normalizer

        self.use_output_gate = use_output_gate
        self.use_norm = use_norm
        self.fuse_norm = fuse_norm
        self.scale = scale

        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.gsa_aux_loss_enable = gsa_aux_loss_enable
        self.gsa_aux_loss_coeff = gsa_aux_loss_coeff
        self.gsa_aux_loss_target_ratio = gsa_aux_loss_target_ratio
        self.gsa_kv_shift = gsa_kv_shift

        if num_slots is None:
            num_slots = self.head_k_dim
        self.num_slots = num_slots

        self.layer_idx = layer_idx

        if layer_idx is None:
            warnings.warn(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class.",
            )

        self.register_module('feature_map', None)
        if feature_map == 'swish':
            self.feature_map = SwishFeatureMap()
        elif feature_map == 'relu':
            self.feature_map = ReLUFeatureMap()
        elif feature_map == 't2r':
            self.feature_map = T2RFeatureMap(self.head_k_dim, self.head_k_dim)
        else:
            raise NotImplementedError(f"Feature map `{feature_map}` is not supported now.")

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim_per_group, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim_per_group, bias=False)
        self.f_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.num_slots, bias=False)

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim_per_group,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim_per_group,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )

        self.g_norm = RMSNorm(self.hidden_size, elementwise_affine, eps=norm_eps, dtype=torch.float32)
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.rotary = RotaryEmbedding(dim=self.head_k_dim, base=self.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: NHACache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, NHACache | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None

        if q_len == 1 and last_state is not None:
            mode = 'fused_recurrent'
        else:
            mode = self.mode

        cu_seqlens = kwargs.get('cu_seqlens')
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)     

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        f = self.f_proj(hidden_states)

        q = rearrange(q, '... (h d) -> ... h d', d=self.head_k_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_k_dim)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)
        f = rearrange(f, '... (h m) -> ... h m', m=self.num_slots)

        if self.feature_map is not None:
            q_gsa, k_gsa = map(lambda x: self.feature_map(x), (q, k))
        else:
            q_gsa, k_gsa = q, k
        v = F.silu(v)

        f = F.logsigmoid(f) / self.gate_logit_normalizer
        s = (1 - f.exp()).to(f.dtype)
        g = f

        if self.num_kv_groups > 1:
            k, v, k_gsa, s, g = map(lambda x: repeat(x, '... h d -> ... (h g) d', g=self.num_kv_groups), (k, v, k_gsa, s, g))

        seqlen_offset = 0
        if last_state is not None and 'seqlen_offset' in last_state:
            seqlen_offset = last_state['seqlen_offset']
        q_swa, k_swa = self.rotary(q, k, seqlen_offset=seqlen_offset)

        if mode == 'chunk':
            output, lse_swa, lse_gsa = chunk_nha(
                q_swa=q_swa,
                k_swa=k_swa,
                q_gsa=q_gsa,
                k_gsa=k_gsa,
                v=v,
                s=s,
                g=g,
                scale=self.scale,
                window_size_left=self.window_size,
                window_size_right=0,
                head_first=False,
                gsa_kv_shift=self.gsa_kv_shift,
                cu_seqlens=cu_seqlens,
            )

            # aux loss
            if self.training:
                lse_max = torch.maximum(lse_swa, lse_gsa)
                w_swa = torch.exp(lse_swa - lse_max)
                w_gsa = torch.exp(lse_gsa - lse_max)
                self._store_compression_stats(w_swa, w_gsa)

                if self.gsa_aux_loss_enable:
                    output, gsa_aux_loss = self._apply_gsa_aux_loss(output, lse_swa, lse_gsa)
                    self._store_gsa_aux_loss(gsa_aux_loss)

        elif mode == 'fused_recurrent':
            from nha_fla.ops.nha.fused_recurrent import fused_recurrent_nha_step

            if recurrent_state is not None:
                hk0, hv0 = recurrent_state
            else:
                H_kv = self.num_kv_heads * self.num_kv_groups
                hk0 = q.new_zeros(batch_size, H_kv, self.head_k_dim, self.num_slots)
                hv0 = q.new_zeros(batch_size, H_kv, self.num_slots, self.head_v_dim)

            output_gsa, hk_new, hv_new = fused_recurrent_nha_step(
                q=rearrange(q_gsa, 'b t h d -> (b t) h d'),
                k=rearrange(k_gsa, 'b t h d -> (b t) h d'),
                v=rearrange(v, 'b t h d -> (b t) h d'),
                s=rearrange(s, 'b t h m -> (b t) h m'),
                g=rearrange(g, 'b t h m -> (b t) h m'),
                hk=hk0,
                hv=hv0,
                scale=self.scale,
            )
            output_gsa = rearrange(output_gsa, '(b t) h d -> b t h d', b=batch_size)

            try:
                from flash_attn import flash_attn_func
                output_swa = flash_attn_func(
                    q_swa, k_swa, v,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=(self.window_size, 0),
                )
            except ImportError:
                scale = self.scale or self.head_k_dim ** -0.5
                attn = torch.einsum('bthd,bshd->bths', q_swa * scale, k_swa)
                causal_mask = torch.ones(q_len, q_len, dtype=torch.bool, device=attn.device).tril()
                attn = attn.masked_fill(~causal_mask, float('-inf'))
                attn = F.softmax(attn, dim=-1)
                output_swa = torch.einsum('bths,bshd->bthd', attn, v)

            output = 0.5 * output_swa + 0.5 * output_gsa
            recurrent_state = (hk_new, hv_new)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        o = rearrange(output[:, :, :, :self.head_v_dim], '... h d -> ... (h d)')
        o = rms_norm_linear(F.silu(o), self.g_norm.weight, self.g_norm.bias, self.o_proj.weight, self.o_proj.bias)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values

    # ----------------------------------------------------------------
    # Aux loss helpers
    # ----------------------------------------------------------------

    @torch.no_grad()
    def _store_compression_stats(self, w_swa: torch.Tensor, w_gsa: torch.Tensor):
        """Store GSA vs SWA ratio for monitoring."""
        w_sum = w_swa + w_gsa
        gsa_ratio = (w_gsa / w_sum).mean().item()
        if not hasattr(self, '_compression_stats'):
            self._compression_stats = []
        self._compression_stats.append({
            'gsa_ratio': gsa_ratio,
            'swa_ratio': 1.0 - gsa_ratio,
        })

    def _apply_gsa_aux_loss(
        self,
        output: torch.Tensor,
        lse_swa: torch.Tensor,
        lse_gsa: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply single-side GSA aux loss via lse_gsa.

        Uses SWA as stop-grad baseline (SWA LSE may not support backward).
        """
        gsa_logits_delta = lse_gsa.float() - lse_swa.detach().float()
        gsa_target = torch.full_like(gsa_logits_delta, self.gsa_aux_loss_target_ratio)
        gsa_aux_loss = F.binary_cross_entropy_with_logits(
            gsa_logits_delta, gsa_target, reduction='mean'
        )
        output = _EnableCoeffGrad.apply(output, gsa_aux_loss, self.gsa_aux_loss_coeff)
        return output, gsa_aux_loss

    @torch.no_grad()
    def _store_gsa_aux_loss(self, gsa_aux_loss: torch.Tensor):
        """Store aux loss value for monitoring."""
        if not hasattr(self, '_gsa_aux_losses'):
            self._gsa_aux_losses = []
        self._gsa_aux_losses.append(gsa_aux_loss.item())


class _EnableCoeffGrad(torch.autograd.Function):
    """Attach an aux-loss gradient to the output hidden states."""

    @staticmethod
    def forward(ctx, hidden: torch.Tensor, loss: torch.Tensor, coeff: float | None):
        ctx.save_for_backward(loss)
        ctx.coeff = coeff
        return hidden

    @staticmethod
    def backward(ctx, *grads):
        dhidden = grads[0]
        coeff = ctx.coeff
        loss: torch.Tensor = ctx.saved_tensors[0]
        if coeff is None or coeff == 0.0:
            return dhidden, None, None
        dloss = torch.full((1,), coeff, dtype=loss.dtype, device=loss.device)
        return dhidden, dloss, None
