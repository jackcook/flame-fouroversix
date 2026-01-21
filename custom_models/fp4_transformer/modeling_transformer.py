from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Unpack

import torch
from einops import rearrange
from fla.layers.utils import pad_input, unpad_input
from fla.models.transformer.modeling_transformer import (
    TransformerBlock,
    TransformerForCausalLM,
    TransformerModel,
    TransformerPreTrainedModel,
)
from fla.models.utils import FLAGenerationMixin
from fla.modules import RMSNorm, RotaryEmbedding
from fla.modules.activations import ACT2FN
from fla.ops.utils.index import prepare_lens_from_mask
from fouroversix import AdaptiveBlockScalingRule, QuantizeBackend
from fouroversix.model import TrainableFP4Linear
from torch import nn

from .configuration_transformer import FP4TransformerConfig

if TYPE_CHECKING:
    from fla.models.utils import Cache

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None


try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer


class AttentionWithFP4Projections(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: int | None = None,
        rope_theta: float | None = 10000.0,
        max_position_embeddings: int | None = None,
        layer_idx: int = None,
        scale_rule: AdaptiveBlockScalingRule | None = None,
        quantize_backend: QuantizeBackend | None = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm

        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        if flash_attn_func is None:
            raise ImportError(
                "Please install Flash Attention via `pip install flash-attn --no-build-isolation` first",
            )

        linear_kwargs = {
            "a_scale_rule": scale_rule,
            "w_scale_rule": scale_rule,
            "g_scale_rule": scale_rule,
            "quantize_backend": quantize_backend,
        }

        self.q_proj = TrainableFP4Linear(
            self.hidden_size,
            self.hidden_size,
            bias=self.qkv_bias,
            **linear_kwargs,
        )
        self.k_proj = TrainableFP4Linear(
            self.hidden_size,
            self.kv_dim,
            bias=self.qkv_bias,
            **linear_kwargs,
        )
        self.v_proj = TrainableFP4Linear(
            self.hidden_size,
            self.kv_dim,
            bias=self.qkv_bias,
            **linear_kwargs,
        )
        self.o_proj = TrainableFP4Linear(
            self.hidden_size, self.hidden_size, bias=False, **linear_kwargs,
        )

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, dtype=torch.float32)
            self.k_norm = RMSNorm(self.head_dim, dtype=torch.float32)

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()

        q = rearrange(
            self.q_proj(hidden_states), "... (h d) -> ... h d", d=self.head_dim,
        )
        k = rearrange(
            self.k_proj(hidden_states), "... (h d) -> ... h d", d=self.head_dim,
        )
        v = rearrange(
            self.v_proj(hidden_states), "... (h d) -> ... h d", d=self.head_dim,
        )

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # equivalent to cu_seqlens in `flash_attn`
        cu_seqlens = kwargs.get("cu_seqlens")

        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to deliminate the offsets of padding tokens
                seqlen_offset = (
                    seqlen_offset
                    + prepare_lens_from_mask(attention_mask)
                    - attention_mask.shape[-1]
                )
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        q, k = self.rotary(
            q,
            k,
            seqlen_offset=seqlen_offset,
            max_seqlen=max_seqlen,
            cu_seqlens=cu_seqlens,
        )

        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size),
            )["attn_state"]
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, "... (h d) -> ... h d", d=self.head_dim)
                v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            if q.shape[1] == 1 and self.window_size is not None:
                attention_mask = attention_mask[:, -self.window_size :]
            q, (k, v), indices_q, cu_seqlens, max_seq_lens = unpad_input(
                q, (k, v), attention_mask, q_len,
            )
            cu_seqlens_q, cu_seqlens_k = cu_seqlens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
                window_size=(
                    (-1, -1) if self.window_size is None else (self.window_size - 1, 0)
                ),
            )
            o = pad_input(o, indices_q, batch_size, q_len)
        elif cu_seqlens is not None:
            o = flash_attn_varlen_func(
                q.squeeze(0),
                k.squeeze(0),
                v.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(
                    (-1, -1) if self.window_size is None else (self.window_size - 1, 0)
                ),
            ).unsqueeze(0)
        else:
            o = flash_attn_func(
                q,
                k,
                v,
                causal=True,
                window_size=(
                    (-1, -1) if self.window_size is None else (self.window_size - 1, 0)
                ),
            )
        o = o.reshape(batch_size, q_len, -1)
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, past_key_values


class FP4GatedMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: int | None = None,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        fuse_swiglu: bool = True,
        scale_rule: AdaptiveBlockScalingRule | None = None,
        quantize_backend: QuantizeBackend | None = None,
    ) -> FP4GatedMLP:
        super().__init__()

        if fuse_swiglu:
            raise ValueError("fuse_swiglu is not supported for FP4GatedMLP")

        self.hidden_size = hidden_size
        # the final number of params is `hidden_ratio * hidden_size^2`
        # `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.fuse_swiglu = fuse_swiglu

        if hidden_act != "swish":
            raise ValueError(f"Unsupported hidden_act: {hidden_act}")

        linear_kwargs = {
            "a_scale_rule": scale_rule,
            "w_scale_rule": scale_rule,
            "g_scale_rule": scale_rule,
            "quantize_backend": quantize_backend,
            "bias": False,
        }

        self.gate_proj = TrainableFP4Linear(
            self.hidden_size,
            self.intermediate_size,
            **linear_kwargs,
        )
        self.up_proj = TrainableFP4Linear(
            self.hidden_size,
            self.intermediate_size,
            **linear_kwargs,
        )
        self.down_proj = TrainableFP4Linear(
            self.intermediate_size,
            self.hidden_size,
            **linear_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Unpack[Any],
    ) -> torch.Tensor:
        gate, y = self.gate_proj(x), self.up_proj(x)
        return self.down_proj(ACT2FN[self.hidden_act](gate) * y)


class FP4TransformerBlock(GradientCheckpointingLayer):

    def __init__(self, config: FP4TransformerConfig, layer_idx: int):
        super(GradientCheckpointingLayer, self).__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.precision_config = config.layer_precision_configs[layer_idx]

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size, eps=config.norm_eps,
        )
        self.attn = AttentionWithFP4Projections(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            qkv_bias=config.qkv_bias,
            qk_norm=config.qk_norm,
            window_size=config.window_size,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx,
            scale_rule=self.precision_config["scale_rule"],
            quantize_backend=self.precision_config.get("quantize_backend"),
        )

        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size, eps=config.norm_eps,
        )
        self.mlp = FP4GatedMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
            scale_rule=self.precision_config["scale_rule"],
            quantize_backend=self.precision_config.get("quantize_backend"),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        **kwargs: Unpack[Any],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:

        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attentions,)

        if use_cache:
            outputs += (past_key_values,)

        return outputs


class FP4TransformerPreTrainedModel(TransformerPreTrainedModel):
    config_class = FP4TransformerConfig
    _no_split_modules = ["FP4TransformerBlock"]


class FP4TransformerModel(FP4TransformerPreTrainedModel, TransformerModel):

    def __init__(
        self,
        config: FP4TransformerConfig,
    ) -> FP4TransformerModel:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
        )
        self.layers = nn.ModuleList(
            [
                (
                    FP4TransformerBlock
                    if config.layer_precision_configs[layer_idx].get(
                        "precision", "bf16",
                    )
                    == "fp4"
                    else TransformerBlock
                )(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ],
        )
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size,
            eps=config.norm_eps,
        )

        self.gradient_checkpointing = False

        self.post_init()


class FP4TransformerForCausalLM(
    FP4TransformerPreTrainedModel,
    TransformerForCausalLM,
    FLAGenerationMixin,
):

    def __init__(self, config):
        super().__init__(config)
        self.model = FP4TransformerModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        # Initialize weights and apply final processing
        self.post_init()
