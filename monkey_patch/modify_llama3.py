import math
import types
import warnings
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint

from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
)

from pdb import set_trace as st 


def llama3_zero_ablate_attention_forward(
    self,
    hidden_states,
    attention_mask = None,
    position_ids = None,
    past_key_value = None,
    output_attentions = False,
    use_cache = False,
    cache_position = None,
    position_embeddings = None,  # will become mandatory in v4.45
    **kwargs,
):
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        # logger.warning_once(
        #     "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
        #     'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        # )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    bsz, q_len, _ = hidden_states.size()
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # query_states = self.q_proj(hidden_states)
    # key_states = self.k_proj(hidden_states)
    # value_states = self.v_proj(hidden_states)

    # query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    # value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if position_embeddings is None:
        # logger.warning_once(
        #     "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
        #     "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
        #     "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
        #     "removed and `position_embeddings` will be mandatory."
        # )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # ######################################################
    #### set attention head output to zero
    for head_id in self.ablate_head_id_list:
        value_states[:, head_id, :, :] = 0 
    # value_states[:, self.ablate_head_id, :, :] = 0 
    # ######################################################

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    return attn_output, None


def zero_ablate_llama3_attn_head(layer, head_id_list):
    """
    replace the forward function of LlamaAttention with a custom forward function `llama_custom_attention_forward`
    """
    modified_module = layer.self_attn
    modified_module.ablate_head_id_list = head_id_list

    # num_heads = modified_module.num_heads 
    # head_dim = modified_module.head_dim
    # modified_module.mean_head_stats = torch.zeros((num_heads, head_dim)).cuda()
    # modified_module.num_samples = 0

    modified_module.forward = types.MethodType(llama3_zero_ablate_attention_forward, modified_module)

    return modified_module


def llama3_mean_ablate_attention_forward(
    self,
    hidden_states,
    attention_mask = None,
    position_ids = None,
    past_key_value = None,
    output_attentions = False,
    use_cache = False,
    cache_position = None,
    position_embeddings = None,  # will become mandatory in v4.45
    **kwargs,
):
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        # logger.warning_once(
        #     "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
        #     'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        # )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        # logger.warning_once(
        #     "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
        #     "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
        #     "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
        #     "removed and `position_embeddings` will be mandatory."
        # )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    ##################################################################
    if self.collect_stats:
        self.mean_head_stats = self.mean_head_stats * self.num_samples + attn_output.mean(dim=2).sum(dim=0)
        self.num_samples += attn_output.shape[0]
        self.mean_head_stats /= self.num_samples
    else:
        attn_output[:, self.ablate_head_id, :, :] = self.mean_head_stats[self.ablate_head_id]
    ##################################################################

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

def mean_ablate_llama3_attn_head(layer, head_id, collect_stats=False):
    """
    replace the forward function of LlamaAttention with a custom forward function `llama_custom_attention_forward`
    """
    modified_module = layer.self_attn
    modified_module.ablate_head_id = head_id
    modified_module.collect_stats = collect_stats

    if collect_stats:
        num_heads = modified_module.num_heads 
        head_dim = modified_module.head_dim
        modified_module.mean_head_stats = torch.zeros((num_heads, head_dim)).cuda()
        modified_module.num_samples = 0

    modified_module.forward = types.MethodType(llama3_mean_ablate_attention_forward, modified_module)

    return modified_module




def decoderlayer_forward(self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache,
        cache_position,
        position_embeddings,  # necessary, but kept here for BC
        **kwargs):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    self.feat = hidden_states.clone().cpu()

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)

    return outputs


def modify_llama3_decoderlayer(layer):
    """
    replace the forward function of LlamaAttention with a custom forward function `llama_custom_attention_forward`
    """

    layer.forward = types.MethodType(decoderlayer_forward, layer)

    return layer