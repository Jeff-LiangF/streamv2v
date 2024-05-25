from importlib import import_module
from typing import Callable, Optional, Union
from collections import deque

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.models.attention_processor import Attention
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.lora import LoRACompatibleLinear, LoRALinearLayer

from .utils import get_nn_feats, random_bipartite_soft_matching

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None
    
class CachedSTAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, name=None, use_feature_injection=False,
                 feature_injection_strength=0.8, 
                 feature_similarity_threshold=0.98,
                 interval=4, 
                 max_frames=1, 
                 use_tome_cache=False, 
                 tome_metric="keys", 
                 use_grid=False, 
                 tome_ratio=0.5):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.name = name
        self.use_feature_injection = use_feature_injection
        self.fi_strength = feature_injection_strength
        self.threshold = feature_similarity_threshold
        self.zero_tensor = torch.tensor(0)
        self.frame_id = torch.tensor(0)
        self.interval = torch.tensor(interval)
        self.max_frames = max_frames
        self.cached_key = None
        self.cached_value = None
        self.cached_output = None
        self.use_tome_cache = use_tome_cache
        self.tome_metric = tome_metric
        self.use_grid = use_grid
        self.tome_ratio = tome_ratio
    
    def _tome_step_kvout(self, keys, values, outputs):
        keys = torch.cat([self.cached_key, keys], dim=1)
        values = torch.cat([self.cached_value, values], dim=1)
        outputs = torch.cat([self.cached_output, outputs], dim=1)
        m_kv_out, _, _= random_bipartite_soft_matching(metric=keys, use_grid=self.use_grid, ratio=self.tome_ratio)
        compact_keys, compact_values, compact_outputs = m_kv_out(keys, values, outputs)
        self.cached_key = compact_keys
        self.cached_value = compact_values
        self.cached_output = compact_outputs
        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        is_selfattn = False
        if encoder_hidden_states is None:
            is_selfattn = True
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        if is_selfattn:
            cached_key = key.clone()
            cached_value = value.clone()
            
            # Avoid if statement -> replace the dynamic graph to static graph
            if torch.equal(self.frame_id, self.zero_tensor):
            # ONNX
                self.cached_key = cached_key
                self.cached_value = cached_value

            key = torch.cat([key, self.cached_key], dim=1)
            value = torch.cat([value, self.cached_value], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        if is_selfattn:
            cached_output = hidden_states.clone()

            if torch.equal(self.frame_id, self.zero_tensor):
                self.cached_output = cached_output

            if self.use_feature_injection and ("up_blocks.0" in self.name or "up_blocks.1" in self.name or 'mid_block' in self.name):
                nn_hidden_states = get_nn_feats(hidden_states, self.cached_output, threshold=self.threshold)
                hidden_states = hidden_states * (1-self.fi_strength) + self.fi_strength * nn_hidden_states

        mod_result = torch.remainder(self.frame_id, self.interval)
        if torch.equal(mod_result, self.zero_tensor) and is_selfattn:
                self._tome_step_kvout(cached_key, cached_value, cached_output)
        
        self.frame_id = self.frame_id + 1
        
        return hidden_states



class CachedSTXFormersAttnProcessor:
    r"""
    Processor for implementing memory efficient attention using xFormers.

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(self, attention_op: Optional[Callable] = None, name=None, 
                 use_feature_injection=False, feature_injection_strength=0.8, feature_similarity_threshold=0.98,
                 interval=4, max_frames=4, use_tome_cache=False, tome_metric="keys", use_grid=False, tome_ratio=0.5):
        self.attention_op = attention_op
        self.name = name
        self.use_feature_injection = use_feature_injection
        self.fi_strength = feature_injection_strength
        self.threshold = feature_similarity_threshold
        self.frame_id = 0
        self.interval = interval
        self.cached_key = deque(maxlen=max_frames)
        self.cached_value = deque(maxlen=max_frames)
        self.cached_output = deque(maxlen=max_frames)
        self.use_tome_cache = use_tome_cache
        self.tome_metric = tome_metric
        self.use_grid = use_grid
        self.tome_ratio = tome_ratio

    def _tome_step_kvout(self, keys, values, outputs):
        if len(self.cached_value) == 1:
            keys = torch.cat(list(self.cached_key) + [keys], dim=1)
            values = torch.cat(list(self.cached_value) + [values], dim=1)
            outputs = torch.cat(list(self.cached_output) + [outputs], dim=1)
            m_kv_out, _, _= random_bipartite_soft_matching(metric=eval(self.tome_metric), use_grid=self.use_grid, ratio=self.tome_ratio)
            compact_keys, compact_values, compact_outputs = m_kv_out(keys, values, outputs)
            self.cached_key.append(compact_keys)
            self.cached_value.append(compact_values)
            self.cached_output.append(compact_outputs)
        else:
            self.cached_key.append(keys)
            self.cached_value.append(values)
            self.cached_output.append(outputs)

    def _tome_step_kv(self, keys, values):
        if len(self.cached_value) == 1:
            keys = torch.cat(list(self.cached_key) + [keys], dim=1)
            values = torch.cat(list(self.cached_value) + [values], dim=1)
            _, m_kv, _= random_bipartite_soft_matching(metric=eval(self.tome_metric), use_grid=self.use_grid, ratio=self.tome_ratio)
            compact_keys, compact_values = m_kv(keys, values)
            self.cached_key.append(compact_keys)
            self.cached_value.append(compact_values)
        else:
            self.cached_key.append(keys)
            self.cached_value.append(values)
            
    def _tome_step_out(self, outputs):
        if len(self.cached_value) == 1:
            outputs = torch.cat(list(self.cached_output) + [outputs], dim=1)
            _, _, m_out= random_bipartite_soft_matching(metric=outputs, use_grid=self.use_grid, ratio=self.tome_ratio)
            compact_outputs = m_out(outputs)
            self.cached_output.append(compact_outputs)
        else:
            self.cached_output.append(outputs)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        is_selfattn = False
        if encoder_hidden_states is None:
            is_selfattn = True
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        if is_selfattn:
            cached_key = key.clone()
            cached_value = value.clone()

            if len(self.cached_key) > 0:
                key = torch.cat([key] + list(self.cached_key), dim=1)
                value = torch.cat([value] + list(self.cached_value), dim=1)

            ## Code for storing and visualizing features 
            # if self.frame_id % self.interval == 0:
            #     # if "down_blocks.0" in self.name or "up_blocks.3" in self.name:
            #     #     feats = {
            #     #                 "hidden_states": hidden_states.clone().cpu(),
            #     #                 "query": query.clone().cpu(),
            #     #                 "key": cached_key.cpu(),
            #     #                 "value": cached_value.cpu(),
            #     #             }
            #     #     torch.save(feats, f'./outputs/self_attn_feats_SD/{self.name}.frame{self.frame_id}.pt')
            #     if self.use_tome_cache:
            #         cached_key, cached_value = self._tome_step(cached_key, cached_value)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        if is_selfattn:
            cached_output = hidden_states.clone()
            if self.use_feature_injection and ("up_blocks.0" in self.name or "up_blocks.1" in self.name or 'mid_block' in self.name):
                if len(self.cached_output) > 0:
                    nn_hidden_states = get_nn_feats(hidden_states, self.cached_output, threshold=self.threshold)
                    hidden_states = hidden_states * (1-self.fi_strength) + self.fi_strength * nn_hidden_states
            
        if self.frame_id % self.interval == 0:
            if is_selfattn:
                if self.use_tome_cache:
                    self._tome_step_kvout(cached_key, cached_value, cached_output)
                else:
                    self.cached_key.append(cached_key)
                    self.cached_value.append(cached_value)
                    self.cached_output.append(cached_output)
        self.frame_id += 1

        return hidden_states
    