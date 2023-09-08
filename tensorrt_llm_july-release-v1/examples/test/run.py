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

    tensorrt_llm.logger.set_level('info')
    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(1, 0)
    torch.cuda.set_device(0)

    # load engine
    with open('simplewhisper.engine', 'rb') as f:
        engine_buffer = f.read()
    session = Session.from_serialized_engine(engine_buffer)

    inputs = {
        'data': torch.Tensor([[50257]]).to(dtype=torch.int32).cuda(),
        'length': torch.Tensor([1.0]).to(dtype=torch.int32).cuda(),
        'encoder_hidden_states': torch.rand(1,1500,512).cuda(),
        'self_past_key': torch.rand(6,8,23,64).cuda(),
        'self_past_value': torch.rand(6,8,23,64).cuda(),
        'cross_past_key': torch.rand(6,8,1500,64).cuda(),
        'cross_past_value': torch.rand(6,8,1500,64).cuda(),
        'past_self_cache_mask': torch.rand(1,).cuda(),
        'past_cross_cache_mask': torch.rand(1,).cuda(),
    }

    # inference output shape
    inputs_shape = [TensorInfo(k,_torch_to_trt_dtype_dict[v.dtype],tuple(v.shape)) for k,v in inputs.items()]
    print(inputs_shape)
    outputs_shape = session.infer_shapes(inputs_shape)
    
    # malloc buffer
    outputs = {}
    for output in outputs_shape:
        outputs[output.name] = torch.zeros(*output.shape,dtype=_trt_to_torch_dtype_dict[output.dtype]).cuda()

    # execute
    with _scoped_stream() as stream:
        ok = session.run(inputs, outputs, stream)
    torch.cuda.synchronize()

    trtllm_o = outputs['output']



    torch_net = SimpleConvTorchNet()
    torch_net.load_state_dict(torch.load('weight.pth',map_location='cpu'))
    torch_net.cuda()
    with torch.inference_mode():
        torch_o, cache = torch_net(inputs['data'],inputs['encoder_hidden_states'],
                            None)
                    # ((inputs['self_past_key'][0:1],inputs['self_past_value'][0:1],inputs['cross_past_key'][0:1],inputs['cross_past_value'][0:1]),
                    # (inputs['self_past_key'][1:2],inputs['self_past_value'][1:2],inputs['cross_past_key'][1:2],inputs['cross_past_value'][1:2]),
                    # (inputs['self_past_key'][2:3],inputs['self_past_value'][2:3],inputs['cross_past_key'][2:3],inputs['cross_past_value'][2:3]),
                    # (inputs['self_past_key'][3:4],inputs['self_past_value'][3:4],inputs['cross_past_key'][3:4],inputs['cross_past_value'][3:4]),
                    # (inputs['self_past_key'][4:5],inputs['self_past_value'][4:5],inputs['cross_past_key'][4:5],inputs['cross_past_value'][4:5]),
                    # (inputs['self_past_key'][5:6],inputs['self_past_value'][5:6],inputs['cross_past_key'][5:6],inputs['cross_past_value'][5:6])))

    cache = [torch.cat([cache[j][i] for j in range(6)],dim=0) for i in range(4)]

    a = trtllm_o.cpu().numpy()
    b = torch_o.cpu().numpy()
    diff = np.abs(a-b)
    print(a.shape,a.min(),a.mean(),a.max(),a.var())
    print(b.shape,b.min(),b.mean(),b.max(),b.var())
    print(diff.shape,diff.min(),diff.mean(),diff.max(),diff.var())

    diff = np.abs(outputs['output0'].cpu().numpy()-cache[0].cpu().numpy())
    print(diff.shape,diff.min(),diff.mean(),diff.max(),diff.var())
    diff = np.abs(outputs['output1'].cpu().numpy()-cache[1].cpu().numpy())
    print(diff.shape,diff.min(),diff.mean(),diff.max(),diff.var())
    diff = np.abs(outputs['output2'].cpu().numpy()-cache[2].cpu().numpy())
    print(diff.shape,diff.min(),diff.mean(),diff.max(),diff.var())
    diff = np.abs(outputs['output3'].cpu().numpy()-cache[3].cpu().numpy())
    print(diff.shape,diff.min(),diff.mean(),diff.max(),diff.var())