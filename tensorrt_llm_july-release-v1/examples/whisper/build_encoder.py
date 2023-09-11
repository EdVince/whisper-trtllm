import time
import torch
import pickle

from transformers import WhisperForConditionalGeneration

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

    # create huggingface model
    hf_model = WhisperForConditionalGeneration.from_pretrained("whisper-medium.en")
    config = hf_model.config.to_dict()
    with open('engine/config.pkl', 'wb') as f:
        pickle.dump(config, f)

    # create tensort-llm model
    trtllm_model = tensorrt_llm.models.WhisperEncoder(
        d_model=config['d_model'],
        num_mel_bins=config['num_mel_bins'],
        max_source_positions=config['max_source_positions'],
        encoder_layers=config['encoder_layers'],
        encoder_attention_heads=config['encoder_attention_heads'],
        activation_function=config['activation_function'],
        encoder_ffn_dim=config['encoder_ffn_dim']
    )

    # create builder
    builder = Builder()
    builder_config = builder.create_builder_config(
        name='WhisperEncoder',
        precision='float32',
        timing_cache='model.cache',
        tensor_parallel=1,
        parallel_build=False,
        int8=False,
        opt_level=None,
    )

    # load hf weight to trtllm weight
    ckpt = hf_model.state_dict()
    trtllm_model.conv1.weight.value = ckpt['model.encoder.conv1.weight'].unsqueeze(2).numpy()
    trtllm_model.conv1.bias.value = ckpt['model.encoder.conv1.bias'].numpy()
    trtllm_model.conv2.weight.value = ckpt['model.encoder.conv2.weight'].unsqueeze(2).numpy()
    trtllm_model.conv2.bias.value = ckpt['model.encoder.conv2.bias'].numpy()
    trtllm_model.embed_positions_weight = ckpt['model.encoder.embed_positions.weight'].unsqueeze(0).numpy()
    for idx in range(config['encoder_layers']):
        trtllm_model.layers[idx].self_attn.qkv.weight.value = torch.cat([ckpt[f'model.encoder.layers.{idx}.self_attn.q_proj.weight'],ckpt[f'model.encoder.layers.{idx}.self_attn.k_proj.weight'],ckpt[f'model.encoder.layers.{idx}.self_attn.v_proj.weight']],dim=0).numpy()
        trtllm_model.layers[idx].self_attn.qkv.bias.value = torch.cat([ckpt[f'model.encoder.layers.{idx}.self_attn.q_proj.bias'],torch.zeros_like(ckpt[f'model.encoder.layers.{idx}.self_attn.q_proj.bias']),ckpt[f'model.encoder.layers.{idx}.self_attn.v_proj.bias']],dim=0).numpy()
        trtllm_model.layers[idx].self_attn.dense.weight.value = ckpt[f'model.encoder.layers.{idx}.self_attn.out_proj.weight'].numpy()
        trtllm_model.layers[idx].self_attn.dense.bias.value = ckpt[f'model.encoder.layers.{idx}.self_attn.out_proj.bias'].numpy()
        trtllm_model.layers[idx].self_attn_layer_norm.weight.value = ckpt[f'model.encoder.layers.{idx}.self_attn_layer_norm.weight'].numpy()
        trtllm_model.layers[idx].self_attn_layer_norm.bias.value = ckpt[f'model.encoder.layers.{idx}.self_attn_layer_norm.bias'].numpy()
        trtllm_model.layers[idx].fc1.weight.value = ckpt[f'model.encoder.layers.{idx}.fc1.weight'].numpy()
        trtllm_model.layers[idx].fc1.bias.value = ckpt[f'model.encoder.layers.{idx}.fc1.bias'].numpy()
        trtllm_model.layers[idx].fc2.weight.value = ckpt[f'model.encoder.layers.{idx}.fc2.weight'].numpy()
        trtllm_model.layers[idx].fc2.bias.value = ckpt[f'model.encoder.layers.{idx}.fc2.bias'].numpy()
        trtllm_model.layers[idx].final_layer_norm.weight.value = ckpt[f'model.encoder.layers.{idx}.final_layer_norm.weight'].numpy()
        trtllm_model.layers[idx].final_layer_norm.bias.value = ckpt[f'model.encoder.layers.{idx}.final_layer_norm.bias'].numpy()
    trtllm_model.layer_norm.weight.value = ckpt['model.encoder.layer_norm.weight'].numpy()
    trtllm_model.layer_norm.bias.value = ckpt['model.encoder.layer_norm.bias'].numpy()

    # set plugin
    network = builder.create_network()
    network.trt_network.name = 'WhisperEncoder'
    network.plugin_config.set_identity_plugin(dtype='float32')

    # get static graph
    with net_guard(network):
        network.set_named_parameters(trtllm_model.named_parameters())
        inputs = trtllm_model.prepare_inputs()
        trtllm_model(inputs)

    # build engine
    engine = builder.build_engine(network, builder_config)

    # save engine
    assert engine is not None, f'Failed to build engine'
    serialize_engine(engine, 'engine/WhisperEncoder.engine')