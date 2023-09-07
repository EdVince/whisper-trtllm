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

_torch_to_trt_dtype_dict = {
    torch.float16: trt.float16,
    torch.float32: trt.float32,
    torch.int32: trt.int32,
    torch.int8: trt.int8,
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

    tensorrt_llm.logger.set_level('verbose')
    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(1, 0)
    torch.cuda.set_device(0)

    # load engine
    with open('simplewhisper.engine', 'rb') as f:
        engine_buffer = f.read()
    session = Session.from_serialized_engine(engine_buffer)

    inputs = {
        'data': torch.rand(1,1,512).cuda(),
        'length': torch.Tensor([1.0]).to(dtype=torch.int32).cuda(),
        'encoder_hidden_states': torch.rand(1,1500,512).cuda(),
        'self_past_key': torch.rand(1,8,23,64).cuda(),
        'self_past_value': torch.rand(1,8,23,64).cuda(),
        'self_cache_mask': torch.zeros(1+23).cuda(),
        'cross_past_key': torch.rand(1,8,1500,64).cuda(),
        'cross_past_value': torch.rand(1,8,1500,64).cuda(),
        'cross_cache_mask': torch.zeros(1+1500).cuda(),
    }

    # inference output shape
    inputs_shape = [TensorInfo(k,_torch_to_trt_dtype_dict[v.dtype],tuple(v.shape)) for k,v in inputs.items()]
    outputs_shape = session.infer_shapes(inputs_shape)
    
    # malloc buffer
    outputs = {}
    for output in outputs_shape:
        outputs[output.name] = torch.zeros(*output.shape,dtype=_trt_to_torch_dtype_dict[output.dtype]).cuda()

    # execute
    with _scoped_stream() as stream:
        ok = session.run(inputs, outputs, stream)
    torch.cuda.synchronize()

    trtllm_o = outputs['output0']
    trtllm_sk = outputs['output1']
    trtllm_sv = outputs['output2']
    trtllm_ck = outputs['output3']
    trtllm_cv = outputs['output4']


    torch_net = SimpleConvTorchNet()
    torch_net.load_state_dict(torch.load('weight.pth',map_location='cpu'))
    torch_net.cuda()
    with torch.inference_mode():
        torch_o, (torch_sk, torch_sv, torch_ck, torch_cv) = torch_net(inputs['data'],inputs['encoder_hidden_states'],
                    (inputs['self_past_key'],inputs['self_past_value'],inputs['cross_past_key'],inputs['cross_past_value']))

    a = trtllm_o.cpu().numpy()
    b = torch_o.cpu().numpy()
    diff = np.abs(a-b)
    print(a.shape,a.min(),a.mean(),a.max(),a.var())
    print(b.shape,b.min(),b.mean(),b.max(),b.var())
    print(diff.shape,diff.min(),diff.mean(),diff.max(),diff.var())
    
    diff = np.abs(trtllm_sk.cpu().numpy()-torch_sk.cpu().numpy())
    print(diff.shape,diff.min(),diff.mean(),diff.max(),diff.var())
    
    diff = np.abs(trtllm_sv.cpu().numpy()-torch_sv.cpu().numpy())
    print(diff.shape,diff.min(),diff.mean(),diff.max(),diff.var())

    diff = np.abs(trtllm_ck.cpu().numpy()-torch_ck.cpu().numpy())
    print(diff.shape,diff.min(),diff.mean(),diff.max(),diff.var())
    
    diff = np.abs(trtllm_cv.cpu().numpy()-torch_cv.cpu().numpy())
    print(diff.shape,diff.min(),diff.mean(),diff.max(),diff.var())