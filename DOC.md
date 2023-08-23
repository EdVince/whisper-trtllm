# Whisper

## 安装修改过的transformers代码
```
cd transformers
pip install -e .
```

## Whipser的运行过程

1. 只调用一次WhisperEncoder
2. 使用greedy策略多次调用WhisperDecoder

- [WhisperForConditionalGeneration.generate](transformers/src/transformers/models/whisper/modeling_whisper.py)
    - [GenerationMixin.generate](transformers/src/transformers/generation/utils.py)
        - [GenerationMixin._prepare_encoder_decoder_kwargs_for_generation](transformers/src/transformers/generation/utils.py)
            - [WhisperEncoder.forward](transformers/src/transformers/models/whisper/modeling_whisper.py)
                - [WhisperEncoderLayer.forward](transformers/src/transformers/models/whisper/modeling_whisper.py)
        - [GenerationMixin.greedy_search](transformers/src/transformers/generation/utils.py)
            - [WhisperForConditionalGeneration.forward](transformers/src/transformers/models/whisper/modeling_whisper.py)
                - [WhisperModel.forward](transformers/src/transformers/models/whisper/modeling_whisper.py)
                    - [WhisperDecoder.forward](transformers/src/transformers/models/whisper/modeling_whisper.py)
                        - [WhisperDecoderLayer.forward](transformers/src/transformers/models/whisper/modeling_whisper.py)