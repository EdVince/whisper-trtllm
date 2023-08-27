import tensorrt as trt

from ..._utils import str_dtype_to_trt
from ...functional import Tensor, RaggedTensor, ACT2FN
from ...layers import Attention, LayerNorm, ColumnLinear
from ...module import Module

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

class SimpleConvTRTLLMNet(Module):

    def __init__(self):
        super().__init__()
        self.encoder_layer = WhisperEncoderLayer()

    def forward(self, hidden_states: RaggedTensor):

        hidden_states = self.encoder_layer(hidden_states)

        hidden_states.mark_output('output', str_dtype_to_trt('float32'))

        return hidden_states

    def prepare_inputs(self):

        hidden_states_data = Tensor(name='data',
                    dtype=trt.float32,
                    shape=[1, 1500, 512])
        hidden_states_length = Tensor(name='length',
                    dtype=trt.float32,
                    shape=[1])

        hidden_states = RaggedTensor.from_row_lengths(hidden_states_data, hidden_states_length)

        return (hidden_states)


if __name__ == '__main__':
    net = SimpleConvTRTLLMNet()
