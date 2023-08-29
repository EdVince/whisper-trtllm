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
        name='SimpleWhisper',
        precision='float32',
        timing_cache='model.cache',
        tensor_parallel=1,
        parallel_build=False,
        int8=False,
        opt_level=None,
        )

    # create tensort-llm model
    tensorrt_llm_test = tensorrt_llm.models.SimpleConvTRTLLMNet()

    # load weight from torch
    ckpt = torch.load('weight.pth',map_location='cpu')
    print(ckpt.keys())
    tensorrt_llm_test.encoder.conv1.weight.value = ckpt['encoder.conv1.weight'].unsqueeze(2).numpy()
    tensorrt_llm_test.encoder.conv1.bias.value = ckpt['encoder.conv1.bias'].numpy()
    tensorrt_llm_test.encoder.conv2.weight.value = ckpt['encoder.conv2.weight'].unsqueeze(2).numpy()
    tensorrt_llm_test.encoder.conv2.bias.value = ckpt['encoder.conv2.bias'].numpy()
    tensorrt_llm_test.encoder.embed_positions_weight = ckpt['encoder.embed_positions.weight'].unsqueeze(0).numpy()
    for k, v in ckpt.items():
        if 'layers' in k:
            idx = int(k.split('.')[2])
            layers_idx = '.'.join(k.split('.')[:3])
            tensorrt_llm_test.encoder.layers[idx].self_attn.qkv.weight.value = torch.cat([ckpt[layers_idx+'.self_attn.q_proj.weight'],ckpt[layers_idx+'.self_attn.k_proj.weight'],ckpt[layers_idx+'.self_attn.v_proj.weight']],dim=0).numpy()
            tensorrt_llm_test.encoder.layers[idx].self_attn.qkv.bias.value = torch.cat([ckpt[layers_idx+'.self_attn.q_proj.bias'],torch.zeros_like(ckpt[layers_idx+'.self_attn.q_proj.bias']),ckpt[layers_idx+'.self_attn.v_proj.bias']],dim=0).numpy()
            tensorrt_llm_test.encoder.layers[idx].self_attn.dense.weight.value = ckpt[layers_idx+'.self_attn.out_proj.weight'].numpy()
            tensorrt_llm_test.encoder.layers[idx].self_attn.dense.bias.value = ckpt[layers_idx+'.self_attn.out_proj.bias'].numpy()
            tensorrt_llm_test.encoder.layers[idx].self_attn_layer_norm.weight.value = ckpt[layers_idx+'.self_attn_layer_norm.weight'].numpy()
            tensorrt_llm_test.encoder.layers[idx].self_attn_layer_norm.bias.value = ckpt[layers_idx+'.self_attn_layer_norm.bias'].numpy()
            tensorrt_llm_test.encoder.layers[idx].fc1.weight.value = ckpt[layers_idx+'.fc1.weight'].numpy()
            tensorrt_llm_test.encoder.layers[idx].fc1.bias.value = ckpt[layers_idx+'.fc1.bias'].numpy()
            tensorrt_llm_test.encoder.layers[idx].fc2.weight.value = ckpt[layers_idx+'.fc2.weight'].numpy()
            tensorrt_llm_test.encoder.layers[idx].fc2.bias.value = ckpt[layers_idx+'.fc2.bias'].numpy()
            tensorrt_llm_test.encoder.layers[idx].final_layer_norm.weight.value = ckpt[layers_idx+'.final_layer_norm.weight'].numpy()
            tensorrt_llm_test.encoder.layers[idx].final_layer_norm.bias.value = ckpt[layers_idx+'.final_layer_norm.bias'].numpy()
    tensorrt_llm_test.encoder.layer_norm.weight.value = ckpt['encoder.layer_norm.weight'].numpy()
    tensorrt_llm_test.encoder.layer_norm.bias.value = ckpt['encoder.layer_norm.bias'].numpy()


    network = builder.create_network()
    network.trt_network.name = 'SimpleWhisper'

    with net_guard(network):

        network.set_named_parameters(tensorrt_llm_test.named_parameters())

        inputs = tensorrt_llm_test.prepare_inputs()
    
        tensorrt_llm_test(inputs)

    engine = builder.build_engine(network, builder_config)

    assert engine is not None, f'Failed to build engine'

    serialize_engine(engine, 'simplewhisper.engine')