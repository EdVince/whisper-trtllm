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

_trt_to_torch_dtype_dict = {
    trt.float16: torch.float16,
    trt.float32: torch.float32,
    trt.int32: torch.int32,
    trt.int8: torch.int8,
}

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
        TensorInfo('length',trt.int32,(1,)),
        TensorInfo('key_value_states',trt.float32,(1,1500,512)),
        TensorInfo('past_key',trt.float32,(1,8,1500,64)),
        TensorInfo('past_value',trt.float32,(1,8,1500,64)),
        TensorInfo('cache_mask',trt.float32,(1+1500,)),
    ]
    outputs_shape = session.infer_shapes(inputs_shape)
    
    # malloc buffer
    inputs = {
        'data': torch.rand(1,1,512).cuda(),
        'length': torch.Tensor([1.0]).to(dtype=torch.int32).cuda(),
        'key_value_states': torch.rand(1,1500,512).cuda(),
        'past_key': torch.rand(1,8,1500,64).cuda(),
        'past_value': torch.rand(1,8,1500,64).cuda(),
        'cache_mask': torch.zeros(1+1500).cuda(),
    }
    outputs = {}
    for output in outputs_shape:
        outputs[output.name] = torch.zeros(*output.shape,dtype=_trt_to_torch_dtype_dict[output.dtype]).cuda()

    # execute
    with _scoped_stream() as stream:
        ok = session.run(inputs, outputs, stream)
    torch.cuda.synchronize()

    trtllm_o = outputs['output0']
    trtllm_k = outputs['output1']
    trtllm_v = outputs['output2']


    torch_net = SimpleConvTorchNet()
    torch_net.load_state_dict(torch.load('weight.pth',map_location='cpu'))
    torch_net.cuda()
    with torch.inference_mode():
        torch_o, (torch_k, torch_v) = torch_net(inputs['data'],None,
                                                (inputs['past_key'],inputs['past_value']))

    a = trtllm_o.cpu().numpy()
    b = torch_o.cpu().numpy()
    diff = np.abs(a-b)
    print(a.shape,a.min(),a.mean(),a.max(),a.var())
    print(b.shape,b.min(),b.mean(),b.max(),b.var())
    print(diff.shape,diff.min(),diff.mean(),diff.max(),diff.var())
    
    a = trtllm_k.cpu().numpy()
    b = torch_k.cpu().numpy()
    diff = np.abs(a-b)
    print(a.shape,a.min(),a.mean(),a.max(),a.var())
    print(b.shape,b.min(),b.mean(),b.max(),b.var())
    print(diff.shape,diff.min(),diff.mean(),diff.max(),diff.var())
    
    a = trtllm_v.cpu().numpy()
    b = torch_v.cpu().numpy()
    diff = np.abs(a-b)
    print(a.shape,a.min(),a.mean(),a.max(),a.var())
    print(b.shape,b.min(),b.mean(),b.max(),b.var())
    print(diff.shape,diff.min(),diff.mean(),diff.max(),diff.var())
