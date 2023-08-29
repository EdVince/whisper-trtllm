import tensorrt as trt

import torch

from ..._utils import str_dtype_to_trt
from ...functional import (Tensor, RaggedTensor, ACT2FN, 
                           unsqueeze, gelu, shape, gather, concat, view, permute, constant)
from ...layers import Attention, LayerNorm, ColumnLinear, Conv2d
from ...module import Module, ModuleList

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

        return hidden_states


class SimpleConvTRTLLMNet(Module):

    def __init__(self):
        super().__init__()
        self.encoder = WhisperEncoder()

    def forward(self, input_features: RaggedTensor):

        hidden_states = self.encoder(input_features)

        hidden_states.mark_output('output', str_dtype_to_trt('float32'))

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


if __name__ == '__main__':
    net = SimpleConvTRTLLMNet()
