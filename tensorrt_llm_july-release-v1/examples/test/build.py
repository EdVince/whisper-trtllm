import time
import torch

import tensorrt_llm
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from tensorrt_llm.network import net_guard

def serialize_engine(engine, path):
    logger.info(f'Serializing engine to {path}...')
    tik = time.time()
    with open(path, 'wb') as f:
        f.write(bytearray(engine))
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')

if __name__ == '__main__':

    logger.set_level('info')
    torch.cuda.set_device(0)
    tensorrt_llm.logger.set_level('info')

    # create builder
    builder = Builder()
    builder_config = builder.create_builder_config(
        name='SimpleAttn',
        precision='float32',
        timing_cache='model.cache',
        tensor_parallel=1,
        parallel_build=False,
        int8=False,
        opt_level=None,
        )

    # create tensort-llm model
    tensorrt_llm_test = tensorrt_llm.models.SimpleConvTRTLLMNet()

    # # load weight from torch
    ckpt = torch.load('weight.pth',map_location='cpu')
    tensorrt_llm_test.encoder_layer.self_attn.qkv.weight.value = torch.cat([ckpt['attn.self_attn.q_proj.weight'],ckpt['attn.self_attn.k_proj.weight'],ckpt['attn.self_attn.v_proj.weight']],dim=0).numpy()
    tensorrt_llm_test.encoder_layer.self_attn.qkv.bias.value = torch.cat([ckpt['attn.self_attn.q_proj.bias'],torch.zeros_like(ckpt['attn.self_attn.q_proj.bias']),ckpt['attn.self_attn.v_proj.bias']],dim=0).numpy()
    tensorrt_llm_test.encoder_layer.self_attn.dense.weight.value = ckpt['attn.self_attn.out_proj.weight'].numpy()
    tensorrt_llm_test.encoder_layer.self_attn.dense.bias.value = ckpt['attn.self_attn.out_proj.bias'].numpy()
    tensorrt_llm_test.encoder_layer.self_attn_layer_norm.weight.value = ckpt['attn.self_attn_layer_norm.weight'].numpy()
    tensorrt_llm_test.encoder_layer.self_attn_layer_norm.bias.value = ckpt['attn.self_attn_layer_norm.bias'].numpy()
    tensorrt_llm_test.encoder_layer.fc1.weight.value = ckpt['attn.fc1.weight'].numpy()
    tensorrt_llm_test.encoder_layer.fc1.bias.value = ckpt['attn.fc1.bias'].numpy()
    tensorrt_llm_test.encoder_layer.fc2.weight.value = ckpt['attn.fc2.weight'].numpy()
    tensorrt_llm_test.encoder_layer.fc2.bias.value = ckpt['attn.fc2.bias'].numpy()
    tensorrt_llm_test.encoder_layer.final_layer_norm.weight.value = ckpt['attn.final_layer_norm.weight'].numpy()
    tensorrt_llm_test.encoder_layer.final_layer_norm.bias.value = ckpt['attn.final_layer_norm.bias'].numpy()

    network = builder.create_network()
    network.trt_network.name = 'SimpleAttn'

    with net_guard(network):

        network.set_named_parameters(tensorrt_llm_test.named_parameters())

        inputs = tensorrt_llm_test.prepare_inputs()
    
        tensorrt_llm_test(inputs)

    engine = builder.build_engine(network, builder_config)

    assert engine is not None, f'Failed to build engine'

    serialize_engine(engine, 'simpleattn.engine')