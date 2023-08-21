import math
from collections import OrderedDict

import tensorrt as trt

from ..._common import default_net
from ..._utils import pad_vocab_size, str_dtype_to_trt
from ...functional import activation, Tensor
from ...layers import Conv2d
from ...module import Module, ModuleList

class SimpleConvTRTLLMNet(Module):

    def __init__(self):
        super().__init__()

        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1), groups=1, bias=True)
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1), groups=1, bias=True)

    def prepare_inputs(self):

        input_ids = Tensor(name='input',
                    dtype=trt.float32,
                    shape=[1, 3, 224, 224])

        return (input_ids)

    def forward(self,x):

        x = self.conv1(x)
        x = activation(x,trt.ActivationType.RELU)
        x = self.conv2(x)
        x = activation(x,trt.ActivationType.RELU)

        x.mark_output('output', str_dtype_to_trt('float32'))

        return x

if __name__ == '__main__':
    net = SimpleConvTRTLLMNet()
