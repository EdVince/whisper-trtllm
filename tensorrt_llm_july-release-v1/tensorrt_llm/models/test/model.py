import math
from collections import OrderedDict

import tensorrt as trt

from ..._utils import str_dtype_to_trt
from ...functional import Tensor, RaggedTensor
from ...layers import Conv2d, Attention
from ...module import Module

class SimpleConvTRTLLMNet(Module):

    def __init__(self):
        super().__init__()

        self.attn = Attention(512,8,1)

    def prepare_inputs(self):

        hidden_states_data = Tensor(name='data',
                    dtype=trt.float32,
                    shape=[1, 1500, 512])
        hidden_states_length = Tensor(name='length',
                    dtype=trt.float32,
                    shape=[1])

        hidden_states = RaggedTensor.from_row_lengths(hidden_states_data, hidden_states_length)

        return (hidden_states)

    def forward(self,
                hidden_states: RaggedTensor):

        hidden_states = self.attn(hidden_states)
        hidden_states = hidden_states.data
        hidden_states.mark_output('output', str_dtype_to_trt('float32'))

        return hidden_states

if __name__ == '__main__':
    net = SimpleConvTRTLLMNet()
