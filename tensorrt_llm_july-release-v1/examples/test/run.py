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
    with open('simpleattn.engine', 'rb') as f:
        engine_buffer = f.read()
    session = Session.from_serialized_engine(engine_buffer)

    # inference output shape
    inputs_shape = [
        TensorInfo('data',trt.float32,(1,1500,512)),
        TensorInfo('length',trt.float32,(1,))
    ]
    outputs_shape = session.infer_shapes(inputs_shape)
    
    # malloc buffer
    inputs = {
        'data': torch.rand(1,1500,512).cuda(),
        'length': torch.Tensor([1.0]).cuda()
    }
    outputs = {}
    for output in outputs_shape:
        outputs[output.name] = torch.zeros(*output.shape).cuda()

    # execute
    with _scoped_stream() as stream:
        ok = session.run(inputs, outputs, stream)
    torch.cuda.synchronize()
    trtllm_out = outputs['output']
    

    torch_net = SimpleConvTorchNet()
    torch_net.load_state_dict(torch.load('weight.pth',map_location='cpu'))
    torch_net.cuda()
    with torch.inference_mode():
        torch_out = torch_net(inputs['data'])
    
    trtllm_out = trtllm_out.cpu().numpy()
    torch_out = torch_out.cpu().numpy()
    diff = np.abs(trtllm_out-torch_out)
    print(trtllm_out.shape,trtllm_out.min(),trtllm_out.mean(),trtllm_out.max(),trtllm_out.var())
    print(torch_out.shape,torch_out.min(),torch_out.mean(),torch_out.max(),torch_out.var())
    print(diff.shape,diff.min(),diff.mean(),diff.max(),diff.var())