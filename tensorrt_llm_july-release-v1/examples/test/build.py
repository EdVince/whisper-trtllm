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
    
    ckpt = torch.load('weight.pth',map_location='cpu')
    tensorrt_llm_test.decoder.embed_tokens.weight.value = ckpt['decoder.embed_tokens.weight'].numpy()
    tensorrt_llm_test.decoder.embed_positions.weight.value = ckpt['decoder.embed_positions.weight'].numpy()
    for idx in range(6):
        tensorrt_llm_test.decoder.layers[idx].self_attn.q_proj.weight.value = ckpt[f'decoder.layers.{idx}.self_attn.q_proj.weight'].numpy()
        tensorrt_llm_test.decoder.layers[idx].self_attn.q_proj.bias.value = ckpt[f'decoder.layers.{idx}.self_attn.q_proj.bias'].numpy()
        tensorrt_llm_test.decoder.layers[idx].self_attn.k_proj.weight.value = ckpt[f'decoder.layers.{idx}.self_attn.k_proj.weight'].numpy()
        tensorrt_llm_test.decoder.layers[idx].self_attn.v_proj.weight.value = ckpt[f'decoder.layers.{idx}.self_attn.v_proj.weight'].numpy()
        tensorrt_llm_test.decoder.layers[idx].self_attn.v_proj.bias.value = ckpt[f'decoder.layers.{idx}.self_attn.v_proj.bias'].numpy()
        tensorrt_llm_test.decoder.layers[idx].self_attn.dense.weight.value = ckpt[f'decoder.layers.{idx}.self_attn.out_proj.weight'].numpy()
        tensorrt_llm_test.decoder.layers[idx].self_attn.dense.bias.value = ckpt[f'decoder.layers.{idx}.self_attn.out_proj.bias'].numpy()
        tensorrt_llm_test.decoder.layers[idx].self_attn_layer_norm.weight.value = ckpt[f'decoder.layers.{idx}.self_attn_layer_norm.weight'].numpy()
        tensorrt_llm_test.decoder.layers[idx].self_attn_layer_norm.bias.value = ckpt[f'decoder.layers.{idx}.self_attn_layer_norm.bias'].numpy()
        tensorrt_llm_test.decoder.layers[idx].encoder_attn.q_proj.weight.value = ckpt[f'decoder.layers.{idx}.encoder_attn.q_proj.weight'].numpy()
        tensorrt_llm_test.decoder.layers[idx].encoder_attn.q_proj.bias.value = ckpt[f'decoder.layers.{idx}.encoder_attn.q_proj.bias'].numpy()
        tensorrt_llm_test.decoder.layers[idx].encoder_attn.k_proj.weight.value = ckpt[f'decoder.layers.{idx}.encoder_attn.k_proj.weight'].numpy()
        tensorrt_llm_test.decoder.layers[idx].encoder_attn.v_proj.weight.value = ckpt[f'decoder.layers.{idx}.encoder_attn.v_proj.weight'].numpy()
        tensorrt_llm_test.decoder.layers[idx].encoder_attn.v_proj.bias.value = ckpt[f'decoder.layers.{idx}.encoder_attn.v_proj.bias'].numpy()
        tensorrt_llm_test.decoder.layers[idx].encoder_attn.dense.weight.value = ckpt[f'decoder.layers.{idx}.encoder_attn.out_proj.weight'].numpy()
        tensorrt_llm_test.decoder.layers[idx].encoder_attn.dense.bias.value = ckpt[f'decoder.layers.{idx}.encoder_attn.out_proj.bias'].numpy()
        tensorrt_llm_test.decoder.layers[idx].encoder_attn_layer_norm.weight.value = ckpt[f'decoder.layers.{idx}.encoder_attn_layer_norm.weight'].numpy()
        tensorrt_llm_test.decoder.layers[idx].encoder_attn_layer_norm.bias.value = ckpt[f'decoder.layers.{idx}.encoder_attn_layer_norm.bias'].numpy()
        tensorrt_llm_test.decoder.layers[idx].fc1.weight.value = ckpt[f'decoder.layers.{idx}.fc1.weight'].numpy()
        tensorrt_llm_test.decoder.layers[idx].fc1.bias.value = ckpt[f'decoder.layers.{idx}.fc1.bias'].numpy()
        tensorrt_llm_test.decoder.layers[idx].fc2.weight.value = ckpt[f'decoder.layers.{idx}.fc2.weight'].numpy()
        tensorrt_llm_test.decoder.layers[idx].fc2.bias.value = ckpt[f'decoder.layers.{idx}.fc2.bias'].numpy()
        tensorrt_llm_test.decoder.layers[idx].final_layer_norm.weight.value = ckpt[f'decoder.layers.{idx}.final_layer_norm.weight'].numpy()
        tensorrt_llm_test.decoder.layers[idx].final_layer_norm.bias.value = ckpt[f'decoder.layers.{idx}.final_layer_norm.bias'].numpy()
    tensorrt_llm_test.decoder.layer_norm.weight.value = ckpt['decoder.layer_norm.weight'].numpy()
    tensorrt_llm_test.decoder.layer_norm.bias.value = ckpt['decoder.layer_norm.bias'].numpy()
    
    '''
    ckpt = torch.load('weight.pth',map_location='cpu')
    print(ckpt.keys())
    tensorrt_llm_test.layer.self_attn.q_proj.weight.value = ckpt['layer.self_attn.q_proj.weight'].numpy()
    tensorrt_llm_test.layer.self_attn.q_proj.bias.value = ckpt['layer.self_attn.q_proj.bias'].numpy()
    tensorrt_llm_test.layer.self_attn.k_proj.weight.value = ckpt['layer.self_attn.k_proj.weight'].numpy()
    tensorrt_llm_test.layer.self_attn.v_proj.weight.value = ckpt['layer.self_attn.v_proj.weight'].numpy()
    tensorrt_llm_test.layer.self_attn.v_proj.bias.value = ckpt['layer.self_attn.v_proj.bias'].numpy()
    tensorrt_llm_test.layer.self_attn.dense.weight.value = ckpt['layer.self_attn.out_proj.weight'].numpy()
    tensorrt_llm_test.layer.self_attn.dense.bias.value = ckpt['layer.self_attn.out_proj.bias'].numpy()
    tensorrt_llm_test.layer.self_attn_layer_norm.weight.value = ckpt['layer.self_attn_layer_norm.weight'].numpy()
    tensorrt_llm_test.layer.self_attn_layer_norm.bias.value = ckpt['layer.self_attn_layer_norm.bias'].numpy()
    tensorrt_llm_test.layer.encoder_attn.q_proj.weight.value = ckpt['layer.encoder_attn.q_proj.weight'].numpy()
    tensorrt_llm_test.layer.encoder_attn.q_proj.bias.value = ckpt['layer.encoder_attn.q_proj.bias'].numpy()
    tensorrt_llm_test.layer.encoder_attn.k_proj.weight.value = ckpt['layer.encoder_attn.k_proj.weight'].numpy()
    tensorrt_llm_test.layer.encoder_attn.v_proj.weight.value = ckpt['layer.encoder_attn.v_proj.weight'].numpy()
    tensorrt_llm_test.layer.encoder_attn.v_proj.bias.value = ckpt['layer.encoder_attn.v_proj.bias'].numpy()
    tensorrt_llm_test.layer.encoder_attn.dense.weight.value = ckpt['layer.encoder_attn.out_proj.weight'].numpy()
    tensorrt_llm_test.layer.encoder_attn.dense.bias.value = ckpt['layer.encoder_attn.out_proj.bias'].numpy()
    tensorrt_llm_test.layer.encoder_attn_layer_norm.weight.value = ckpt['layer.encoder_attn_layer_norm.weight'].numpy()
    tensorrt_llm_test.layer.encoder_attn_layer_norm.bias.value = ckpt['layer.encoder_attn_layer_norm.bias'].numpy()
    tensorrt_llm_test.layer.fc1.weight.value = ckpt['layer.fc1.weight'].numpy()
    tensorrt_llm_test.layer.fc1.bias.value = ckpt['layer.fc1.bias'].numpy()
    tensorrt_llm_test.layer.fc2.weight.value = ckpt['layer.fc2.weight'].numpy()
    tensorrt_llm_test.layer.fc2.bias.value = ckpt['layer.fc2.bias'].numpy()
    tensorrt_llm_test.layer.final_layer_norm.weight.value = ckpt['layer.final_layer_norm.weight'].numpy()
    tensorrt_llm_test.layer.final_layer_norm.bias.value = ckpt['layer.final_layer_norm.bias'].numpy()
    '''
    
    network = builder.create_network()
    network.trt_network.name = 'SimpleWhisper'
    network.plugin_config.set_identity_plugin(dtype='float32')

    with net_guard(network):

        network.set_named_parameters(tensorrt_llm_test.named_parameters())

        inputs = tensorrt_llm_test.prepare_inputs()
    
        tensorrt_llm_test(*inputs)

    engine = builder.build_engine(network, builder_config)

    assert engine is not None, f'Failed to build engine'

    serialize_engine(engine, 'simplewhisper.engine')