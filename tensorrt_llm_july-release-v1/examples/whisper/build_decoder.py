import os
import time
import torch
import argparse

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

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--whisper', type=str, default='whisper-tiny.en')
    parser.add_argument('--engine_precision', type=str, default='float32')
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument('--engine_dir', type=str, default='whisper_outputs')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()

    logger.set_level(args.log_level)
    torch.cuda.set_device(0)
    tensorrt_llm.logger.set_level(args.log_level)

    os.makedirs(args.engine_dir, exist_ok=True)

    # create huggingface model
    hf_model = WhisperForConditionalGeneration.from_pretrained(args.whisper)
    config = hf_model.config.to_dict()

    # create tensort-llm model
    trtllm_model = tensorrt_llm.models.WhisperDecoder(
        pad_token_id=config['pad_token_id'],
        max_target_positions=config['max_target_positions'],
        max_source_positions=config['max_source_positions'],
        d_model=config['d_model'],
        scale_embedding=config['scale_embedding'],
        vocab_size=config['vocab_size'],
        decoder_layers=config['decoder_layers'],
        decoder_attention_heads=config['decoder_attention_heads'],
        activation_function=config['activation_function'],
        decoder_ffn_dim=config['decoder_ffn_dim']
    )

    # create builder
    builder = Builder()
    builder_config = builder.create_builder_config(
        name='WhisperDecoder',
        precision=args.engine_precision,
        timing_cache='model.cache',
        tensor_parallel=1,
        parallel_build=False,
        int8=False,
        opt_level=None,
    )

    # load hf weight to trtllm weight
    ckpt = hf_model.state_dict()
    trtllm_model.embed_tokens.weight.value = ckpt['model.decoder.embed_tokens.weight'].numpy()
    trtllm_model.embed_positions.weight.value = ckpt['model.decoder.embed_positions.weight'].numpy()
    for idx in range(config['decoder_layers']):
        trtllm_model.layers[idx].self_attn.q_proj.weight.value = ckpt[f'model.decoder.layers.{idx}.self_attn.q_proj.weight'].numpy()
        trtllm_model.layers[idx].self_attn.q_proj.bias.value = ckpt[f'model.decoder.layers.{idx}.self_attn.q_proj.bias'].numpy()
        trtllm_model.layers[idx].self_attn.k_proj.weight.value = ckpt[f'model.decoder.layers.{idx}.self_attn.k_proj.weight'].numpy()
        trtllm_model.layers[idx].self_attn.v_proj.weight.value = ckpt[f'model.decoder.layers.{idx}.self_attn.v_proj.weight'].numpy()
        trtllm_model.layers[idx].self_attn.v_proj.bias.value = ckpt[f'model.decoder.layers.{idx}.self_attn.v_proj.bias'].numpy()
        trtllm_model.layers[idx].self_attn.dense.weight.value = ckpt[f'model.decoder.layers.{idx}.self_attn.out_proj.weight'].numpy()
        trtllm_model.layers[idx].self_attn.dense.bias.value = ckpt[f'model.decoder.layers.{idx}.self_attn.out_proj.bias'].numpy()
        trtllm_model.layers[idx].self_attn_layer_norm.weight.value = ckpt[f'model.decoder.layers.{idx}.self_attn_layer_norm.weight'].numpy()
        trtllm_model.layers[idx].self_attn_layer_norm.bias.value = ckpt[f'model.decoder.layers.{idx}.self_attn_layer_norm.bias'].numpy()
        trtllm_model.layers[idx].encoder_attn.q_proj.weight.value = ckpt[f'model.decoder.layers.{idx}.encoder_attn.q_proj.weight'].numpy()
        trtllm_model.layers[idx].encoder_attn.q_proj.bias.value = ckpt[f'model.decoder.layers.{idx}.encoder_attn.q_proj.bias'].numpy()
        trtllm_model.layers[idx].encoder_attn.k_proj.weight.value = ckpt[f'model.decoder.layers.{idx}.encoder_attn.k_proj.weight'].numpy()
        trtllm_model.layers[idx].encoder_attn.v_proj.weight.value = ckpt[f'model.decoder.layers.{idx}.encoder_attn.v_proj.weight'].numpy()
        trtllm_model.layers[idx].encoder_attn.v_proj.bias.value = ckpt[f'model.decoder.layers.{idx}.encoder_attn.v_proj.bias'].numpy()
        trtllm_model.layers[idx].encoder_attn.dense.weight.value = ckpt[f'model.decoder.layers.{idx}.encoder_attn.out_proj.weight'].numpy()
        trtllm_model.layers[idx].encoder_attn.dense.bias.value = ckpt[f'model.decoder.layers.{idx}.encoder_attn.out_proj.bias'].numpy()
        trtllm_model.layers[idx].encoder_attn_layer_norm.weight.value = ckpt[f'model.decoder.layers.{idx}.encoder_attn_layer_norm.weight'].numpy()
        trtllm_model.layers[idx].encoder_attn_layer_norm.bias.value = ckpt[f'model.decoder.layers.{idx}.encoder_attn_layer_norm.bias'].numpy()
        trtllm_model.layers[idx].fc1.weight.value = ckpt[f'model.decoder.layers.{idx}.fc1.weight'].numpy()
        trtllm_model.layers[idx].fc1.bias.value = ckpt[f'model.decoder.layers.{idx}.fc1.bias'].numpy()
        trtllm_model.layers[idx].fc2.weight.value = ckpt[f'model.decoder.layers.{idx}.fc2.weight'].numpy()
        trtllm_model.layers[idx].fc2.bias.value = ckpt[f'model.decoder.layers.{idx}.fc2.bias'].numpy()
        trtllm_model.layers[idx].final_layer_norm.weight.value = ckpt[f'model.decoder.layers.{idx}.final_layer_norm.weight'].numpy()
        trtllm_model.layers[idx].final_layer_norm.bias.value = ckpt[f'model.decoder.layers.{idx}.final_layer_norm.bias'].numpy()
    trtllm_model.layer_norm.weight.value = ckpt['model.decoder.layer_norm.weight'].numpy()
    trtllm_model.layer_norm.bias.value = ckpt['model.decoder.layer_norm.bias'].numpy()
    trtllm_model.proj_out.weight.value = ckpt['proj_out.weight'].numpy()

    # set plugin
    network = builder.create_network()
    network.trt_network.name = 'WhisperDecoder'
    network.plugin_config.set_identity_plugin(dtype=args.engine_precision)

    # get static graph
    with net_guard(network):
        network.set_named_parameters(trtllm_model.named_parameters())
        inputs = trtllm_model.prepare_inputs()
        trtllm_model(*inputs)

    # build engine
    engine = builder.build_engine(network, builder_config)

    # save engine
    assert engine is not None, f'Failed to build engine'
    serialize_engine(engine, os.path.join(args.engine_dir,'WhisperDecoder.engine'))