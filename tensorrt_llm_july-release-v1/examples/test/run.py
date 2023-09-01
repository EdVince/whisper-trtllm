import argparse
import csv
import json
from pathlib import Path
import contextlib
import numpy as np
import torch

import tensorrt as trt
import tensorrt_llm
from tensorrt_llm.runtime import Session, TensorInfo

from create import SimpleConvTorchNet

@contextlib.contextmanager
def _scoped_stream():
    '''Create a scoped cuda stream, and synchronize it when the context is destroyed
    '''
    #TODO: delete torch, use cuda native python bindings
    import torch
    stream = torch.cuda.current_stream()
    try:
        # return a handle, trt and other lib does not recognize torch.cuda.Stream
        yield stream.cuda_stream
    finally:
        stream.synchronize()

if __name__ == '__main__':

    tensorrt_llm.logger.set_level('info')
    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(1, 0)
    torch.cuda.set_device(0)

    # load engine
    with open('simplewhisper.engine', 'rb') as f:
        engine_buffer = f.read()
    session = Session.from_serialized_engine(engine_buffer)

    # inference output shape
    inputs_shape = [
        TensorInfo('data',trt.float32,(1,1,512)),
        TensorInfo('length',trt.float32,(1,)),
        TensorInfo('key_value_states',trt.float32,(1,1500,512)),
        TensorInfo('past_key_value',trt.float32,(2,8,1500,64))
    ]
    outputs_shape = session.infer_shapes(inputs_shape)
    
    # malloc buffer
    inputs = {
        'data': torch.rand(1,1,512).cuda(),
        'length': torch.Tensor([1.0]).cuda(),
        'key_value_states': torch.rand(1,1500,512).cuda(),
        'past_key_value': torch.rand(2,8,1500,64).cuda(),
    }
    outputs = {}
    for output in outputs_shape:
        outputs[output.name] = torch.zeros(*output.shape).cuda()

    # execute
    with _scoped_stream() as stream:
        ok = session.run(inputs, outputs, stream)
    torch.cuda.synchronize()
    trtllm_out = outputs['output0']
    trtllm_cache = outputs['output1']

    # print(outputs['output'].shape)
    # print(outputs['output0'].shape,outputs['output1'].shape)


    torch_net = SimpleConvTorchNet()
    torch_net.load_state_dict(torch.load('weight.pth',map_location='cpu'))
    torch_net.cuda()
    with torch.inference_mode():
        torch_out, (torch_kcache, torch_vcache) = torch_net(inputs['data'],inputs['key_value_states'],[inputs['past_key_value'][0:1],inputs['past_key_value'][1:2]])
    
    a = trtllm_cache[1:2].cpu().numpy()
    b = torch_vcache.cpu().numpy()
    diff = np.abs(a-b)
    print(a.shape,a.min(),a.mean(),a.max(),a.var())
    print(b.shape,b.min(),b.mean(),b.max(),b.var())
    print(diff.shape,diff.min(),diff.mean(),diff.max(),diff.var())