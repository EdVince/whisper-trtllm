import warnings
warnings.filterwarnings("ignore")

import os
import time
import contextlib
import torch
import pickle
import argparse
from tqdm import tqdm

from transformers import WhisperProcessor
from transformers.generation.logits_process import LogitsProcessorList, SuppressTokensLogitsProcessor, SuppressTokensAtBeginLogitsProcessor, ForceTokensLogitsProcessor
from transformers.generation.stopping_criteria import StoppingCriteriaList, MaxLengthCriteria
import datasets

import tensorrt as trt
import tensorrt_llm
from tensorrt_llm.runtime import Session, TensorInfo

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

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--whisper', type=str, default='whisper-tiny.en', required=True)
    parser.add_argument('--engine_precision', type=str, default='float32')
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument('--engine_dir', type=str, default='whisper_outputs')
    parser.add_argument('--compare', action='store_true')
    return parser.parse_args()

class WhisperEncoder:
    def __init__(self,args=None,config=None):
        
        # load engine
        with open(os.path.join(args.engine_dir,'WhisperEncoder.engine'), 'rb') as f:
            engine_buffer = f.read()
        self.session = Session.from_serialized_engine(engine_buffer)

        # inference output shape
        inputs_shape = [
            TensorInfo('data',trt.float32,(1,80,3000)),
            TensorInfo('length',trt.float32,(1,))
        ]
        outputs_shape = self.session.infer_shapes(inputs_shape)

        # malloc buffer
        self.inputs = {
            'data': torch.rand(1,80,3000).cuda(),
            'length': torch.Tensor([1.0]).cuda()
        }
        self.outputs = {}
        for output in outputs_shape:
            self.outputs[output.name] = torch.zeros(*output.shape,dtype=_trt_to_torch_dtype_dict[output.dtype]).cuda()

    def __call__(self, input):
        self.inputs['data'] = input
        
        # execute
        with _scoped_stream() as stream:
            ok = self.session.run(self.inputs, self.outputs, stream)
        torch.cuda.synchronize()
        
        hidden_states = self.outputs['hidden_states'].clone()
        
        return hidden_states

class WhisperDecoder:
    def __init__(self,args=None,config=None):

        self.config = config
        
        # load engine
        with open(os.path.join(args.engine_dir,'WhisperDecoder.engine'), 'rb') as f:
            engine_buffer = f.read()
        self.session = Session.from_serialized_engine(engine_buffer)

    def __call__(self, decoder_input_ids, encoder_outputs, past_key_values):
        inputs = {
            'data': decoder_input_ids.to(dtype=torch.int32,device='cuda:0'),
            'length': torch.Tensor([1.0]).to(dtype=torch.int32).cuda(),
            'encoder_hidden_states': encoder_outputs.to(dtype=torch.float32,device='cuda:0'),
        }
        if past_key_values is None:
            decoder_layers = config['decoder_layers']
            decoder_attention_heads = config['decoder_attention_heads']
            d_head = config['d_model'] // config['decoder_attention_heads']
            
            inputs['self_past_key'] = torch.rand(decoder_layers,decoder_attention_heads,1,d_head).cuda()
            inputs['self_past_value'] = torch.rand(decoder_layers,decoder_attention_heads,1,d_head).cuda()
            inputs['cross_past_key'] = torch.rand(decoder_layers,decoder_attention_heads,1500,d_head).cuda()
            inputs['cross_past_value'] = torch.rand(decoder_layers,decoder_attention_heads,1500,d_head).cuda()
            inputs['past_self_cache_mask'] = torch.rand(1,).cuda()
            inputs['past_cross_cache_mask'] = torch.rand(1,).cuda()
        else:
            inputs['self_past_key'] = past_key_values[0].to(dtype=torch.float32,device='cuda:0')
            inputs['self_past_value'] = past_key_values[1].to(dtype=torch.float32,device='cuda:0')
            inputs['cross_past_key'] = past_key_values[2].to(dtype=torch.float32,device='cuda:0')
            inputs['cross_past_value'] = past_key_values[3].to(dtype=torch.float32,device='cuda:0')
            inputs['past_self_cache_mask'] = torch.rand(int(1+past_key_values[0].shape[2]),dtype=torch.float32).cuda()
            inputs['past_cross_cache_mask'] = torch.rand(int(1+1500),dtype=torch.float32).cuda()

        # inference output shape
        inputs_shape = [TensorInfo(k,_torch_to_trt_dtype_dict[v.dtype],tuple(v.shape)) for k,v in inputs.items()]
        outputs_shape = self.session.infer_shapes(inputs_shape)
        
        # malloc buffer
        outputs = {}
        for output in outputs_shape:
            outputs[output.name] = torch.zeros(*output.shape,dtype=_trt_to_torch_dtype_dict[output.dtype]).cuda()

        # execute
        with _scoped_stream() as stream:
            ok = self.session.run(inputs, outputs, stream)
        torch.cuda.synchronize()

        hidden_states = outputs['hidden_states'].clone()
        next_self_keys = outputs['next_self_keys'].clone()
        next_self_values = outputs['next_self_values'].clone()
        next_cross_keys = outputs['next_cross_keys'].clone()
        next_cross_values = outputs['next_cross_values'].clone()

        return hidden_states, (next_self_keys, next_self_values, next_cross_keys, next_cross_values)

def get_logits_processor(config,input_ids_seq_length):
    processors = LogitsProcessorList()

    processors.append(SuppressTokensLogitsProcessor(config['suppress_tokens']))

    begin_index = input_ids_seq_length
    begin_index = begin_index if config['forced_bos_token_id'] is None else begin_index + 1
    begin_index += config['forced_decoder_ids'][-1][0]
    processors.append(SuppressTokensAtBeginLogitsProcessor(config['begin_suppress_tokens'], begin_index))

    processors.append(ForceTokensLogitsProcessor(config['forced_decoder_ids']))

    return processors

def get_stopping_criteria(config):
    criteria = StoppingCriteriaList()

    criteria.append(MaxLengthCriteria(max_length=config['max_length'],max_position_embeddings=None))

    return criteria

def greedy_search(
    model,
    encoder_outputs,
    input_ids: torch.Tensor,
    logits_processor = None,
    stopping_criteria = None,
    pad_token_id = None,
    eos_token_id = None,
):

    # init values
    eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device)
    
    # init attention / hidden states / scores tuples
    scores = None

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.int32, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only
    
    past_key_values = None
    
    while True:

        output, past_key_values = model(input_ids[:, -1:],encoder_outputs,past_key_values)

        next_token_logits = output[:, -1, :]

        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        # if eos_token was found in one sentence, set sentence to finished
        unfinished_sequences = unfinished_sequences.mul(next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0))

        # stop when each sentence is finished
        if unfinished_sequences.max() == 0:
            this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished:
            break

    return input_ids

if __name__ == '__main__':

    args = parse_arguments()

    tensorrt_llm.logger.set_level(args.log_level)
    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(1, 0)
    torch.cuda.set_device(0)

    # load processor
    hf_processor = WhisperProcessor.from_pretrained(args.whisper)

    # load dataset
    if os.path.exists('./librispeech_asr_dummy'):
        print('loading dataset from disk...')
        ds = datasets.load_from_disk('./librispeech_asr_dummy')
    else:
        print('loading dataset from huggingface')
        ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        ds.save_to_disk('./librispeech_asr_dummy')

    # load whisper config
    with open(os.path.join(args.engine_dir,'config.pkl'), 'rb') as f:
        config = pickle.load(f)

    # init whisper
    whisperencoder = WhisperEncoder(args=args,config=config)
    whisperdecoder = WhisperDecoder(args=args,config=config)
    
    
    # go through dataset by trtllm twice
    for j in range(1):
        
        trtllm_transcriptions = []
        
        start_time = time.time()
        for i in tqdm(range(len(ds))):
            
            # fetch audio
            sample = ds[i]["audio"]
            input_features = hf_processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features
            input_features = input_features.cuda()
            
            # encode
            encoder_outputs = whisperencoder(input_features)

            # prepare
            input_ids = torch.Tensor([[config['decoder_start_token_id']]]).to(dtype=torch.int32).cuda()
            logits_processor = get_logits_processor(config,input_ids.shape[-1])
            stopping_criteria = get_stopping_criteria(config)

            # greedy decode
            predicted_ids = greedy_search(
                model=whisperdecoder,
                encoder_outputs=encoder_outputs,
                input_ids=input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=config['pad_token_id'],
                eos_token_id=config['eos_token_id'])

            # id to text
            transcription = hf_processor.batch_decode(predicted_ids, skip_special_tokens=True)
            
            trtllm_transcriptions.extend(transcription)

        end_time = time.time()
        
    trtllm_time = end_time - start_time


    if args.compare:
        from transformers import WhisperForConditionalGeneration
        hf_model = WhisperForConditionalGeneration.from_pretrained(args.whisper)
        
        for j in range(1):
            
            hf_transcriptions = []
            
            start_time = time.time()
            for i in tqdm(range(len(ds))):
                
                # fetch audio
                sample = ds[i]["audio"]
                input_features = hf_processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features

                # generate token ids
                predicted_ids = hf_model.generate(input_features)

                # decode token ids to text
                transcription = hf_processor.batch_decode(predicted_ids, skip_special_tokens=True)
                
                hf_transcriptions.extend(transcription)
                
            end_time = time.time()
        
        hf_time = end_time - start_time
        
        print('TensorRT-LLM time: ',trtllm_time)
        print('Huggingface  time: ',hf_time)
        
        diff_transcription = []
        for a, b in zip(trtllm_transcriptions,hf_transcriptions):
            if a != b:
                diff_transcription.append((a,b))
                
        print(f'Compare Result: same [{len(trtllm_transcriptions)-len(diff_transcription)}], diff [{len(diff_transcription)}]')
        if len(diff_transcription) > 0:
            for a,b in diff_transcription:
                print('-------------------------')
                print(f'TensorRT-LLM: {a}')
                print(f'Huggingface : {b}')