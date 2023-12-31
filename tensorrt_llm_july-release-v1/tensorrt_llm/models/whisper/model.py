import enum
import math
from dataclasses import dataclass
from typing import Optional
from collections import OrderedDict

import torch
import numpy as np

import tensorrt as trt

from ..._common import default_net, precision
from ..._utils import str_dtype_to_trt
from ...functional import (Tensor, RaggedTensor, ACT2FN, 
                           unsqueeze, gelu, shape, gather, 
                           concat, view, permute, constant, 
                           elementwise_binary, matmul, softmax, cast,
                           identity, slice)
from ...layers import Attention, LayerNorm, ColumnLinear, Conv2d, Embedding
from ...module import Module, ModuleList
from ...parameter import Parameter
from ...layers.linear import ColumnLinear, RowLinear

def squeeze(input, axis):
    dims = input.ndim()
    input_shape = shape(input)
    out_shapes = []
    for i in range(dims):
        if i == axis:
            continue
        out_shapes.append(gather(input_shape, 0, i))
    out_shape = concat(out_shapes)
    input = view(input, out_shape)
    return input

class WhisperEncoderLayer(Module):

    def __init__(self, d_model=512, encoder_attention_heads=8, activation_function='gelu', encoder_ffn_dim=2048):
        super().__init__()
        self.embed_dim = d_model
        self.self_attn = Attention(self.embed_dim, encoder_attention_heads, 1)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.activation_fn = ACT2FN[activation_function]
        self.fc1 = ColumnLinear(self.embed_dim, encoder_ffn_dim)
        self.fc2 = ColumnLinear(encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, hidden_states: RaggedTensor):

        input_lengths = hidden_states.row_lengths
        max_input_length = hidden_states.max_row_length
        hidden_states = hidden_states.data

        residual = hidden_states

        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(RaggedTensor.from_row_lengths(hidden_states, input_lengths, max_input_length))
        hidden_states = residual + hidden_states.data

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
    
class WhisperEncoder(Module):
    def __init__(self, 
                 d_model=512, num_mel_bins=80, max_source_positions=1500,
                 encoder_layers=6,
                 encoder_attention_heads=8, activation_function='gelu', encoder_ffn_dim=2048):
        super().__init__()

        embed_dim = d_model

        # 原本应该是Conv1d的，但trtllm还没实现，先用Conv2d替换
        self.conv1 = Conv2d(num_mel_bins, embed_dim, kernel_size=(1,3), padding=(0,1))
        self.conv2 = Conv2d(embed_dim, embed_dim, kernel_size=(1,3), stride=(1,2), padding=(0,1))

        self.embed_positions_weight = torch.zeros(1,max_source_positions,embed_dim).numpy()
        
        self.layers = ModuleList([WhisperEncoderLayer(d_model=d_model, 
                                                      encoder_attention_heads=encoder_attention_heads, 
                                                      activation_function=activation_function, 
                                                      encoder_ffn_dim=encoder_ffn_dim) for _ in range(encoder_layers)])
        
        self.layer_norm = LayerNorm(embed_dim)
        
    def forward(self, input_features: RaggedTensor):

        input_lengths = input_features.row_lengths
        max_input_length = input_features.max_row_length
        input_features = input_features.data
        
        input_features = unsqueeze(input_features,2)
        inputs_embeds = gelu(self.conv1(input_features))
        inputs_embeds = gelu(self.conv2(inputs_embeds))
        inputs_embeds = squeeze(inputs_embeds,2)
        inputs_embeds = permute(inputs_embeds,[0,2,1])

        hidden_states = inputs_embeds + constant(self.embed_positions_weight)

        for layer in self.layers:
            hidden_states = layer(RaggedTensor.from_row_lengths(hidden_states, input_lengths, max_input_length))

        hidden_states = self.layer_norm(hidden_states)

        hidden_states.mark_output('hidden_states', str_dtype_to_trt('float32'))

        return hidden_states

    def prepare_inputs(self):

        input_features_data = Tensor(name='data',
                    dtype=trt.float32,
                    shape=[1, 80, 3000])
        input_features_length = Tensor(name='length',
                    dtype=trt.float32,
                    shape=[1])

        input_features = RaggedTensor.from_row_lengths(input_features_data, input_features_length)

        return (input_features)

class AttentionMaskType(enum.Enum):
    padding = 0
    causal = 1
    bidirectional = 2

class PositionEmbeddingType(enum.Enum):
    learned_absolute = enum.auto()
    rope = enum.auto()
    alibi = enum.auto()

@dataclass
class InflightBatchingParam:
    host_beam_widths: Tensor
    cache_indir_pointers: Tensor
    host_req_cache_max_seq_lengths: Tensor
    host_input_lengths: Tensor
    past_key_value_pointers: Tensor
    max_input_length: int
    max_beam_width: int
    kv_orig_quant_scale: Optional[Tensor] = None
    kv_quant_orig_scale: Optional[Tensor] = None
    use_int8_kv_cache: bool = False

    def __post_init__(self):
        assert self.max_input_length > 0, f"max_input_length must be positive, got {self.max_input_length}"
        assert self.max_beam_width > 0, f"max_beam_width must be positive, got {self.max_beam_width}"

class WhisperDecoderAttention(Module):

    def __init__(self,
                 hidden_size=512,
                 num_attention_heads=8,
                 max_position_embeddings=0,
                 num_layers=1,
                 apply_query_key_layer_scaling=False,
                 bias=True,
                 dtype=None,
                 position_embedding_type=PositionEmbeddingType.learned_absolute,
                 neox_rotary_style=False,
                 use_int8_kv_cache=False,
                 rotary_embedding_percentage=1.0,
                 tp_group=None,
                 tp_size=1,
                 multi_block_mode=False,
                 multi_query_mode=False):
        super().__init__()

        self.attention_head_size = hidden_size // num_attention_heads
        self.num_attention_heads = num_attention_heads // tp_size
        self.num_attention_kv_heads = 1 if multi_query_mode else self.num_attention_heads
        self.hidden_size = hidden_size // tp_size
        self.max_position_embeddings = max_position_embeddings

        self.num_layers = num_layers
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.norm_factor = math.sqrt(self.attention_head_size)
        self.q_scaling = 1
        if self.apply_query_key_layer_scaling:
            self.norm_factor *= self.num_layers
            self.q_scaling *= self.num_layers

        self.position_embedding_type = position_embedding_type
        self.multi_block_mode = multi_block_mode
        self.multi_query_mode = multi_query_mode

        self.rotary_embedding_dim = 0
        self.neox_rotary_style = neox_rotary_style
        if self.position_embedding_type == PositionEmbeddingType.rope:
            self.rotary_embedding_dim = int(self.attention_head_size *
                                            rotary_embedding_percentage)
            # TODO: Once we add RotaryEmbedding outside GPTAttention plugin,
            #       we need to set it up here

        self.dtype = dtype

        self.use_int8_kv_cache = use_int8_kv_cache
        if self.use_int8_kv_cache:
            self.kv_orig_quant_scale = Parameter(shape=(1, ), dtype='float32')
            self.kv_quant_orig_scale = Parameter(shape=(1, ), dtype='float32')
        else:
            self.register_parameter('kv_orig_quant_scale', None)
            self.register_parameter('kv_quant_orig_scale', None)

        # Note: in multi_query_mode, only query heads are split between multiple GPUs,
        # while key/value head are not split as there is only one head per key/value.
        # The output feature size is therefore (h/tp + 2) * d, where h is num_heads,
        # d is head_size, and tp is tensor_parallel_size.
        # In ColumnLinear op, the output dim is calculated by (h + 2*tp) * d / tp,
        # which matches the desired output size (h/tp + 2) * d after splitting
        self.q_proj = ColumnLinear(hidden_size,
                                hidden_size,
                                bias=bias,
                                dtype=dtype,
                                tp_group=tp_group,
                                tp_size=tp_size)
        self.k_proj = ColumnLinear(hidden_size,
                                hidden_size,
                                bias=False,
                                dtype=dtype,
                                tp_group=tp_group,
                                tp_size=tp_size)
        self.v_proj = ColumnLinear(hidden_size,
                                hidden_size,
                                bias=bias,
                                dtype=dtype,
                                tp_group=tp_group,
                                tp_size=tp_size)
        self.dense = RowLinear(hidden_size,
                               hidden_size,
                               bias=bias,
                               dtype=dtype,
                               tp_group=tp_group,
                               tp_size=tp_size)

    def forward(self,
                hidden_states: RaggedTensor,
                key_value_states: Optional[RaggedTensor] = None,
                past_key: Optional[Tensor] = None,
                past_value: Optional[Tensor] = None,
                cache_mask: Tensor = None,
                ):

        input_lengths = hidden_states.row_lengths
        max_input_length = hidden_states.max_row_length
        hidden_states = hidden_states.data

        def transpose_for_scores(x):
            new_x_shape = concat([
                shape(x, 0),
                shape(x, 1), self.num_attention_heads, self.attention_head_size
            ])
            return x.view(new_x_shape).permute([0, 2, 1, 3])

        query_states = transpose_for_scores(self.q_proj(hidden_states))

        if key_value_states is not None: # cross attention
            # 用slice来控制是计算还是用past
            # 计算一下用多少cache和多少current
            cache_length = shape(cache_mask,0) - 1
            curr_size = concat([1,1500-cache_length,self.hidden_size])
            past_size = concat([1,self.num_attention_heads,cache_length,self.attention_head_size])
            # 按照所需计算current
            curr_key_states = transpose_for_scores(self.k_proj(slice(key_value_states,[0,0,0],curr_size)))
            curr_value_states = transpose_for_scores(self.v_proj(slice(key_value_states,[0,0,0],curr_size)))
            # 取所需cache与所需current拼接
            key_states = concat([slice(past_key,[0,0,0,0],past_size), curr_key_states],dim=2)
            value_states = concat([slice(past_value,[0,0,0,0],past_size), curr_value_states],dim=2)
        else: # self attention
            # 一定要有的部分
            curr_key_states = transpose_for_scores(self.k_proj(hidden_states))
            curr_value_states = transpose_for_scores(self.v_proj(hidden_states))
            # 用slice来控制past的多少
            cache_length = elementwise_binary(shape(cache_mask,0) - 1, shape(past_key,2), trt.ElementWiseOperation.MIN)
            past_size = concat([1,self.num_attention_heads,cache_length,self.attention_head_size])
            key_states = concat([slice(past_key,[0,0,0,0],past_size), curr_key_states], dim=2)
            value_states = concat([slice(past_value,[0,0,0,0],past_size), curr_value_states], dim=2)

        query = query_states
        key = key_states
        value = value_states
        
        past_key = identity(key)
        past_value = identity(value)

        key = key.permute([0, 1, 3, 2])
        
        with precision('float32'):
            attention_scores = matmul(cast(query, 'float32'), cast(key, 'float32'))
            attention_scores = attention_scores / self.norm_factor
            attention_probs = softmax(attention_scores, dim=-1)

        context = matmul(attention_probs, value).permute([0, 2, 1, 3])
        context = context.view(concat([shape(context, 0), shape(context, 1), self.hidden_size]))

        context = self.dense(context)

        context = RaggedTensor.from_row_lengths(context, input_lengths, max_input_length)

        return context, past_key, past_value

class WhisperDecoderLayer(Module):
    def __init__(self, d_model=512, decoder_attention_heads=8, activation_function='gelu', decoder_ffn_dim=2048):
        super().__init__()
        self.embed_dim = d_model

        self.self_attn = WhisperDecoderAttention(self.embed_dim,decoder_attention_heads)
        self.activation_fn = ACT2FN[activation_function]

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = WhisperDecoderAttention(self.embed_dim,decoder_attention_heads)
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = ColumnLinear(self.embed_dim, decoder_ffn_dim)
        self.fc2 = ColumnLinear(decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self,
        hidden_states: RaggedTensor,
        encoder_hidden_states: Optional[Tensor] = None,
        self_past_key: Optional[Tensor] = None,
        self_past_value: Optional[Tensor] = None,
        self_cache_mask: Optional[Tensor] = None,
        cross_past_key: Optional[Tensor] = None,
        cross_past_value: Optional[Tensor] = None,
        cross_cache_mask: Optional[Tensor] = None,
    ):

        input_lengths = hidden_states.row_lengths
        max_input_length = hidden_states.max_row_length
        hidden_states = hidden_states.data

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, present_key, present_value = self.self_attn(
            hidden_states=RaggedTensor.from_row_lengths(hidden_states, input_lengths, max_input_length),
            key_value_states=None,
            past_key=self_past_key,
            past_value=self_past_value,
            cache_mask=self_cache_mask,
        )
        hidden_states = residual + hidden_states.data

        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Cross Attention
        hidden_states, cross_attn_present_key, cross_attn_present_value = self.encoder_attn(
            hidden_states=RaggedTensor.from_row_lengths(hidden_states, input_lengths, max_input_length),
            key_value_states=encoder_hidden_states,
            past_key=cross_past_key,
            past_value=cross_past_value,
            cache_mask=cross_cache_mask,
        )
        hidden_states = residual + hidden_states.data

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key, present_value, cross_attn_present_key, cross_attn_present_value

class WhisperDecoder(Module):
    def __init__(self,
                 pad_token_id=50256,
                 max_target_positions=448,
                 max_source_positions=1500,
                 d_model=512,
                 scale_embedding=False,
                 vocab_size=51864,
                 decoder_layers=6,
                 decoder_attention_heads=8,
                 activation_function='gelu',
                 decoder_ffn_dim=2048,):
        super().__init__()
        
        self.padding_idx = pad_token_id
        self.max_target_positions = max_target_positions
        self.max_source_positions = max_source_positions
        self.d_model = d_model
        self.embed_scale = math.sqrt(d_model) if scale_embedding else 1.0
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.d_head = d_model // decoder_attention_heads

        self.embed_tokens = Embedding(vocab_size, d_model)
        self.embed_positions = Embedding(self.max_target_positions, d_model)

        self.layers = ModuleList([WhisperDecoderLayer(d_model=d_model,
                                                        decoder_attention_heads=decoder_attention_heads, 
                                                        activation_function=activation_function, 
                                                        decoder_ffn_dim=decoder_ffn_dim) 
                                    for _ in range(decoder_layers)])
        
        self.layer_norm = LayerNorm(d_model)

        self.proj_out = ColumnLinear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: RaggedTensor, # (1,1)
        encoder_hidden_states: Tensor, # (1,1500,512)
        past_self_keys: Tensor,
        past_self_values: Tensor,
        past_cross_keys: Tensor,
        past_cross_values: Tensor,
        past_self_cache_mask: Tensor,
        past_cross_cache_mask: Tensor,
    ):

        input_lengths = input_ids.row_lengths
        max_input_length = input_ids.max_row_length
        input_ids = input_ids.data

        inputs_embeds = self.embed_tokens(input_ids)        
        position = unsqueeze(slice(self.embed_positions.weight.value,concat([shape(past_self_cache_mask,0)-1,0]),concat([shape(input_ids,1),self.d_model])),0)
        hidden_states = inputs_embeds + position
        
        self_len = shape(past_self_keys,2)
        self_size = concat([1,self.decoder_attention_heads,self_len,self.d_head])
        next_self_keys = []
        next_self_values = []
        next_cross_keys = []
        next_cross_values = []
        for idx in range(self.decoder_layers):
            past_self_key = slice(past_self_keys,[idx,0,0,0],self_size)
            past_self_value = slice(past_self_values,[idx,0,0,0],self_size)
            past_cross_key = slice(past_cross_keys,[idx,0,0,0],[1,self.decoder_attention_heads,self.max_source_positions,self.d_head])
            past_cross_value = slice(past_cross_values,[idx,0,0,0],[1,self.decoder_attention_heads,self.max_source_positions,self.d_head])
        
            hidden_states, next_self_key, next_self_value, next_cross_key, next_cross_value = self.layers[idx](
                hidden_states=RaggedTensor.from_row_lengths(hidden_states, input_lengths, max_input_length),
                encoder_hidden_states=encoder_hidden_states,
                self_past_key=past_self_key,
                self_past_value=past_self_value,
                self_cache_mask=past_self_cache_mask,
                cross_past_key=past_cross_key,
                cross_past_value=past_cross_value,
                cross_cache_mask=past_cross_cache_mask,
            )

            next_self_keys.append(next_self_key)
            next_self_values.append(next_self_value)
            next_cross_keys.append(next_cross_key)
            next_cross_values.append(next_cross_value)
            
        hidden_states = self.layer_norm(hidden_states)

        hidden_states = self.proj_out(hidden_states)

        next_self_keys = concat(next_self_keys,dim=0)
        next_self_values = concat(next_self_values,dim=0)
        next_cross_keys = concat(next_cross_keys,dim=0)
        next_cross_values = concat(next_cross_values,dim=0)

        hidden_states.mark_output('hidden_states', str_dtype_to_trt('float32'))
        next_self_keys.mark_output('next_self_keys', str_dtype_to_trt('float32'))
        next_self_values.mark_output('next_self_values', str_dtype_to_trt('float32'))
        next_cross_keys.mark_output('next_cross_keys', str_dtype_to_trt('float32'))
        next_cross_values.mark_output('next_cross_values', str_dtype_to_trt('float32'))

        return hidden_states, next_self_keys, next_self_values, next_cross_keys, next_cross_values
        
    def prepare_inputs(self):

        input_features_data = Tensor(name='data',
                    dtype=trt.int32,
                    shape=[1, 1],
                    dim_range=OrderedDict([('batch_size',[1]),('id_len',[1])]))
        input_features_length = Tensor(name='length',
                    dtype=trt.int32,
                    shape=[1],
                    dim_range=OrderedDict([('batch_size',[1])]))
        input_features = RaggedTensor.from_row_lengths(input_features_data, input_features_length)

        encoder_hidden_states = Tensor(name='encoder_hidden_states',
                    dtype=trt.float32,
                    shape=[1, self.max_source_positions, self.d_model],
                    dim_range=OrderedDict([('batch_size',[1]),('cross_seq_len',[self.max_source_positions]),('embed_size',[self.d_model])]))
        
        self_past_key = Tensor(name='self_past_key',
                    dtype=trt.float32,
                    shape=[self.decoder_layers, self.decoder_attention_heads, -1, self.d_head],
                    dim_range=OrderedDict([('num_layers',[self.decoder_layers]),('num_head',[self.decoder_attention_heads]),('kv_seq_len',[[1,1,self.max_target_positions+1]]),('embed_per_head',[self.d_head])]))
        self_past_value = Tensor(name='self_past_value',
                    dtype=trt.float32,
                    shape=[self.decoder_layers, self.decoder_attention_heads, -1, self.d_head],
                    dim_range=OrderedDict([('num_layers',[self.decoder_layers]),('num_head',[self.decoder_attention_heads]),('kv_seq_len',[[1,1,self.max_target_positions+1]]),('embed_per_head',[self.d_head])]))
        cross_past_key = Tensor(name='cross_past_key',
                    dtype=trt.float32,
                    shape=[self.decoder_layers, self.decoder_attention_heads, self.max_source_positions, self.d_head],
                    dim_range=OrderedDict([('num_layers',[self.decoder_layers]),('num_head',[self.decoder_attention_heads]),('kv_seq_len',[self.max_source_positions]),('embed_per_head',[self.d_head])]))
        cross_past_value = Tensor(name='cross_past_value',
                    dtype=trt.float32,
                    shape=[self.decoder_layers, self.decoder_attention_heads, self.max_source_positions, self.d_head],
                    dim_range=OrderedDict([('num_layers',[self.decoder_layers]),('num_head',[self.decoder_attention_heads]),('kv_seq_len',[self.max_source_positions]),('embed_per_head',[self.d_head])]))

        past_self_cache_mask = Tensor(name='past_self_cache_mask',
                    dtype=trt.float32,
                    shape=[-1],
                    dim_range=OrderedDict([('past_self_cache_length',[[1,1,self.max_target_positions+1]])]))
        
        past_cross_cache_mask = Tensor(name='past_cross_cache_mask',
                    dtype=trt.float32,
                    shape=[-1],
                    dim_range=OrderedDict([('past_cross_cache_length',[[1,1,self.max_source_positions+1]])]))

        return (input_features, encoder_hidden_states, self_past_key, self_past_value, cross_past_key, cross_past_value, past_self_cache_mask, past_cross_cache_mask)
