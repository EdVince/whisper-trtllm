### 总述

本工作是 [NVIDIA TensorRT Hackathon 2023](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/Hackathon2023) 的参赛题目，具体选题是 2.用TensorRT-LLM实现新模型

所实现的模型为OpenAI开发的语音识别模型[whisper](https://github.com/openai/whisper)，whisper是一个通用的语音识别模型，其语音识别功能强大，超越很多闭源的商用模型，官方仓库在github上累计获得超过44K的star。本工作主要是使用TensorRT-LLM实现了whipser的四个英语语音识别模型，分别是：[whisper-tiny.en](https://huggingface.co/openai/whisper-tiny.en), [whisper-base.en](https://huggingface.co/openai/whisper-base.en), [whisper-small.en](https://huggingface.co/openai/whisper-small.en)和[whisper-medium.en](https://huggingface.co/openai/whisper-medium.en)

优化效果：使用float32精度构建模型，wer指标均优于或等于官方模型的精度，相较于huggingface加速比在1.2~1.4X

运行步骤(假设当前仓库/README.md所在路径为：```/root/workspace/trt2023```)：
```
# 更新docker内python的tensor-llm包的模型代码
cp -r tensorrt_llm_july-release-v1/tensorrt_llm/models/whisper \
        /usr/local/lib/python3.8/dist-packages/tensorrt_llm/models
cp -r tensorrt_llm_july-release-v1/tensorrt_llm/models/__init__.py \
        /usr/local/lib/python3.8/dist-packages/tensorrt_llm/models/__init__.py
cp -r tensorrt_llm_july-release-v1/tensorrt_llm/builder.py \
        /usr/local/lib/python3.8/dist-packages/tensorrt_llm/builder.py

# 进入example/whisper目录
cd tensorrt_llm_july-release-v1/examples/whisper

# 下载whisper-tiny.en模型
git lfs install
git clone https://huggingface.co/openai/whisper-tiny.en

# 构建encoder和decoder的engine
python build_encoder.py --whisper whisper-tiny.en
python build_decoder.py --whisper whisper-tiny.en

# 运行engine并于huggingface进行比较
python run.py --whisper whisper-tiny.en --compare

# 下载librispeech数据集，在docker中无法安装torchaudio，可以在宿主机中运行，数据会存储在librispeech.cache
python get_LibriSpeech.py
# 计算engine的wer值
python cal_wer.py --whisper whisper-tiny.en
```


### 主要开发工作

#### 开发工作的难点

请在这一节里总结你的工作难点与亮点。
- 如果使用 TensorRT 进行优化，请介绍一下在模型在导出时、或用polygraphy/trtexec解析时，或在使用TensorRT中，遇到了什么问题并解决了。换句话说，针对这个模型，我们为什么需要额外的工程手段。
- 如果使用 TensorRT-LLM 进行优化，描述以下方面可供选手参考：如果搭建了新模型， 请介绍模型结构有无特别之处，在模型的搭建过程中使用了什么算子，有没有通过plugin支持的新算子。如果支持新feature，请介绍这个feature具体需要修改哪些模块才能实现。如果优化已有模型，请介绍模型性能瓶颈以及解决方法。另外还可以包含工程实现以及debug过程中的难点。

### 开发与优化过程

这一部分是报告的主体。请把自己假定为老师，为 TensorRT 或 TensorRT-LLM 的初学者讲述如何从原始模型出发，经过一系列开发步骤，得到优化后的 TensorRT 或 TensorRT-LLM 模型。或者你是如何一步步通过修改哪些模块添加了新feature的。

建议：

- 分步骤讲清楚开发过程
- 最好能介绍为什么需要某个特别步骤，通过这个特别步骤解决了什么问题
  - 比如，通过Nsight Systems绘制timeline做了性能分析，发现attention时间占比高且有优化空间（贴图展示分析过程），所以决定要写plugin。然后介绍plugin的设计与实现，并在timeline上显示attention这一部分的性能改进。

### 优化效果

环境配置：测试环境为复赛提供的云主机和复赛docker镜像

测试代码：测试代码具体可以跳转到```tensorrt_llm_july-release-v1/examples/whisper```目录下，可以使用```python run.py --whisper whisper-xxxx.en --compare```计算相对于Huggingface的加速比，可以使用```python cal_wer --whisper whisper-xxxx.en```计算wer指标

- 精度：对于Whisper模型，我们使用wer来评价模型的精度，并于OpenAI的官方指标进行比较。对于wer指标，值越小模型精度越高。参考值来源于[leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)

***fp32+fp32表示分别表示encoder和decoder的精度***
| model  | fp32+fp32 | fp16+fp16 | bf16+bf16 | ref wer |
| ------ | --------- | --------- | --------- | ------- |
| tiny   | 5.61%     | 5.60%     | 5.67%     | 5.66%   |
| base   | 4.25%     |           | 4.19%     | 4.27%   |
| small  | 3.05%     |           | 3.02%     | 3.05%   |
| medium | 3.01%     |           | 2.84%     | 3.02%   |

- 性能：在具有73条测试音频的[librispeech_asr_dummy](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy)数据集上对比原始Huggingface模型和TensorRT-LLM模型的加速比

***加速比***
| model  | fp32+fp32 | fp16+fp16 | bf16+bf16 |
| ------ | --------- | --------  | --------- |
| tiny   | 1.6X      | 1.7X      | 1.6X      |
| base   | 1.8X      |           | 1.7X      |
| small  | 1.3X      |           | 1.3X      |
| medium | 1.2X      |           | 1.2X      |

### Bug报告（可选）

对应的代码在bug0的commit上
1. 实现了支持self/cross以及w/wo cache的WhisperDecoderAttention，WhisperDecoderLayer调用WhisperDecoderAttention两次分别做self attn和cross attn
2. WhisperDecoderAttention的四种用法单独测试正常，集成到WhisperDecoderLayer里面后，self attn的value cache异常
3. 代码在"tensorrt_llm_july-release-v1/tensorrt_llm/models/test/model.py"的WhisperDecoderAttention类的forward方法的"elif is_reuse"分支内，正常用法是不用mark output的，但这样出来的是全0，加上mark output是正常的，猜测是fusion有问题
4. 在"workspace/trt2023/tensorrt_llm_july-release-v1/examples/test/"目录下，先运行create.py得到torch的权重，然后运行build.py生成engine，最后运行run.py对比数据精度

### 送分题答案（可选）

1. 请在报告中写出 /root/workspace/tensorrt_llm_july-release-v1/examples/gpt/README 里面 “Single node, single GPU” 部分如下命令的输出（10分）[模型为gpt2-medium](https://huggingface.co/gpt2-medium)
```
python3 run.py --max_output_len=8 

Input: Born in north-east France, Soyer trained as a
Output:  chef before moving to London in the early
```

2. 请在报告中写出 /root/workspace/tensorrt_llm_july-release-v1/examples/gpt/README 里面 “Summarization using the GPT model” 部分如下命令的rouge 分数（10分）[模型为gpt2-medium](https://huggingface.co/gpt2-medium)
```
python3 summarize.py --engine_dirtrt_engine/gpt2/fp16/1-gpu --test_hf  --batch_size1  --test_trt_llm  --hf_model_location=gpt2 --check_accuracy --tensorrt_llm_rouge1_threshold=14

[08/20/2023-05:19:46] [TRT-LLM] [I] TensorRT-LLM (total latency: 2.513667583465576 sec)
[08/20/2023-05:19:46] [TRT-LLM] [I] TensorRT-LLM beam 0 result
[08/20/2023-05:19:47] [TRT-LLM] [I]   rouge1 : 15.361040799540035
[08/20/2023-05:19:47] [TRT-LLM] [I]   rouge2 : 3.854022269668396
[08/20/2023-05:19:47] [TRT-LLM] [I]   rougeL : 12.078455591738333
[08/20/2023-05:19:47] [TRT-LLM] [I]   rougeLsum : 13.547802733617264
[08/20/2023-05:19:47] [TRT-LLM] [I] Hugging Face (total latency: 10.179735660552979 sec)
[08/20/2023-05:19:47] [TRT-LLM] [I] HF beam 0 result
[08/20/2023-05:19:47] [TRT-LLM] [I]   rouge1 : 14.75593024343394
[08/20/2023-05:19:47] [TRT-LLM] [I]   rouge2 : 3.3647470801871733
[08/20/2023-05:19:47] [TRT-LLM] [I]   rougeL : 11.124766996533
[08/20/2023-05:19:47] [TRT-LLM] [I]   rougeLsum : 13.031128048110618
```

### 经验与体会（可选）

1. 目前开发上有一点不太方便，每次修改代码后都需要先编译whl再install，源码开发不太方便
2. 手动用trtllm的layers来拼模型感觉最不方便的其实是怎么获取原始模型的结构，现在的模型设计的又大又复杂，还要用各种config来配置才能得到最终的模型
3. 我的经验是先从原始的模型中抽取出来它的pytorch模型部分，然后按照Module来一个一个实现，实现一个就要跟torch对比一次精度，最后再构成完成的大模型
