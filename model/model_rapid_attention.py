from transformers import PretrainedConfig

from rapid_attention.utils.global_context import (
    rapid_attention_global_context as GCTX,
)

class RapidAttentionLMConfig(PretrainedConfig):
    model_type = "rapid_attention"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dropout = GCTX.model_config.dropout
        self.vocab_size = GCTX.tokenizer_config.vocab_size
        self.common_eps = GCTX.common_config.common_eps
        self.hidden_size = GCTX.model_config.hidden_size
        self.num_hidden_layers = GCTX.model_config.num_hidden_layers
        self.use_moe = GCTX.model_config.use_moe
        self.epochs = GCTX.train_config.epochs
        self.wandb_run_name = f"rapid_attention_pretrain-Epoch-{self.epochs}-BatchSize-{GCTX.train_config.batch_size}-LearningRate-{GCTX.train_config.learning_rate}"
        self.max_seq_len = GCTX.model_config.max_seq_len
        self.num_attention_heads = GCTX.model_config.num_attention_heads
        self.num_key_value_heads = GCTX.model_config.num_key_value_heads
        self.bos_token_id = GCTX.tokenizer_config.bos_token_id
        self.eos_token_id = GCTX.tokenizer_config.eos_token_id
        self.use_flash_attention = GCTX.model_config.use_flash_attention
        self.max_position_embeddings = GCTX.model_config.max_position_embeddings
        self.rope_theta = GCTX.model_config.rope_theta
        self.intermediate_size = GCTX.model_config.intermediate_size
        self.hidden_act = GCTX.model_config.hidden_act

from typing import Optional, Tuple, List, Union
import math
import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_fregs_cis(dim: int, end: int = int(32*1024), theta: float = 1e6):
    fregs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=fregs.device)
    fregs = torch.outer(t, fregs)
    fregs_cos = torch.cat([torch.cos(fregs), torch.cos(fregs)], dim=-1)
    fregs_sin = torch.cat([torch.sin(fregs),torch.sin(fregs)], dim=-1)
    return fregs_cos, fregs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_helf(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_helf(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_helf(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class RapidAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_key_value_heads = config.num_key_value_heads if config.num_key_value_heads is not None else config.num_attention_heads
        assert config.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and config.use_flash_attention

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False,
                attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(batch_size, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        if past_key_values is not None:
            xk = torch.cat([past_key_values[0], xk], dim=1)
            xv = torch.cat([past_key_values[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        if self.flash and seq_len != 1:
            dropout_p = self.dropout if self.training else 0.0
            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask.view(batch_size, 1, 1, -1).expand(batch_size, self.n_local_heads, seq_len, -1)
                attn_mask = attn_mask.bool() if attention_mask is not None else None
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1
            ).unsqueeze(0).unsqueeze(0)
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask
            scores = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 /3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]


    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class RapidBlock(nn.Module):
    def __init__(self, layer_id, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attention = RapidAttention(config)
        self.layer_id = layer_id
        self.input_layer_norm = RMSNorm(config.hidden_size, eps = config.common_eps)
        self.post_attention_layer_norm = RMSNorm(config.hidden_size, eps = config.common_eps)
        self.mlp = FeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attention(
            self.input_layer_norm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layer_norm(hidden_states))
        return hidden_states, present_key_value


class RapidAttentionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            RapidBlock(layer_id, config) for layer_id in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)
        freqs_cos, fregs_sin = precompute_fregs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, theta=config.rope_theta)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", fregs_sin, persistent=False)

    def forward(self, input_ids: Optional[torch.Tensor]=None,
                attention_mask: Optional[torch.Tensor]=None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]=None,
                use_cache: bool=False, **args):
        batch_size, seq_len = input_ids.shape
        if hasattr(past_key_values, "layers"): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)

        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embedding = (
            self.freqs_cos[start_pos : start_pos + seq_len,],
            self.freqs_sin[start_pos : start_pos + seq_len,],
        )
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embedding,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = 0# sum(layer.mlp.aux_loss for layer in self.layers if isinstance(layer.mlp, MOEFeedForward))

        return hidden_states, presents, aux_loss


class RapidAttentionForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = RapidAttentionLMConfig

    def __init__(self, config):
        self.config = config
        super().__init__(config)
        self.model = RapidAttentionModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        hidden_states, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_kvs, hidden_states=hidden_states)
        output.aux_loss =aux_loss
        return output
