### 总述

本工作是 [NVIDIA TensorRT Hackathon 2023](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/Hackathon2023) 的参赛题目，具体选题是 2.用TensorRT-LLM实现新模型 + 4.为TensorRT-LLM添加新feature

所实现的模型为OpenAI开发的语音识别模型[whisper](https://github.com/openai/whisper)，whisper是一个通用的语音识别模型，其语音识别功能强大，超越很多闭源的商用模型，官方仓库在github上累计获得超过44K的star。本工作主要是使用TensorRT-LLM实现了whipser的四个英语语音识别模型，分别是：[whisper-tiny.en](https://huggingface.co/openai/whisper-tiny.en), [whisper-base.en](https://huggingface.co/openai/whisper-base.en), [whisper-small.en](https://huggingface.co/openai/whisper-small.en)和[whisper-medium.en](https://huggingface.co/openai/whisper-medium.en)

所添加的新feature为支持self/cross以及with/without kv cache的Attention实现。

优化效果：使用float32精度构建模型，wer指标均优于或等于官方模型的精度，相较于huggingface加速比在1.2~1.4X

运行步骤(假设当前仓库/README.md所在路径为：```/root/workspace/trt2023```)：
```
# docker启动比赛镜像
bash start_docker.sh

# 进入挂载的代码路径
cd trt2023

# 更新docker内python的tensor-llm包的模型代码
bash update_code.sh
# 或者手动逐个cp
cp -r tensorrt_llm_july-release-v1/tensorrt_llm/models/whisper /usr/local/lib/python3.8/dist-packages/tensorrt_llm/models
cp -r tensorrt_llm_july-release-v1/tensorrt_llm/models/__init__.py /usr/local/lib/python3.8/dist-packages/tensorrt_llm/models/__init__.py
cp -r tensorrt_llm_july-release-v1/tensorrt_llm/builder.py /usr/local/lib/python3.8/dist-packages/tensorrt_llm/builder.py

# 进入example/whisper目录
cd tensorrt_llm_july-release-v1/examples/whisper

# 下载whisper-tiny.en模型
git lfs install
git clone https://huggingface.co/openai/whisper-tiny.en

# 构建encoder和decoder的engine
python build_encoder.py --whisper whisper-tiny.en
python build_decoder.py --whisper whisper-tiny.en

# 安装依赖(无法访问github的话，whipser库会安装失败)
pip install -r requirements.txt

# 运行engine并与huggingface进行比较
python run.py --whisper whisper-tiny.en --compare

# 下载librispeech数据集，在docker中无法安装torchaudio，可以在宿主机中运行，数据会存储在librispeech.cache
python get_LibriSpeech.py
# 计算trtllm engine的wer指标
python cal_wer.py --whisper whisper-tiny.en
```

### 主要开发工作

#### 开发工作的难点

Whisper这个模型是由encoder和decoder两部分组成的。其对输入音频进行一次encoder编码，然后使用各种策略(实现的是greedy search)进行多次decoder解码得到最后的文本。所以需要制作的是三部分：encoder模型、decoder模型、解码Session。

对于decoder模型所需要的功能丰富的Attention，这里实现了这个新[feature](tensorrt_llm_july-release-v1/tensorrt_llm/models/whisper/model.py)，具体实现是WhisperDecoderAttention这个类，基于自带的Attention修改得来，具体修改是修改了对key和value的proj计算，通过引入一个mask，用于支持cache。对于self attention的情况，是最简单的，就是记录下第一次计算的k和v的proj值，后面重复使用就行了；而对于cross attention的情况，每次都要计算当前输入的k和v的proj值，还需要跟历史的kv cache进行拼接cat，得到本次的最终的kv proj量，这里同样使用一个mask来控制所cat的历史cache长度。

##### encoder模型
核心算子就是trtllm已经有了的Attention层就足够了，因为它只需要计算一次，所以不涉及到kv cache的问题，同时它只包含self attention部分，是一个很简单的模型，上手难度低。

##### decoder模型
核心算子还是Attention，但是trtllm自带的不满足我们的需求，在这里，我们既需要self也需要cross的attention，同时还需要支持with/without cache的情况，一共有四种组合。

self和cross是编译时候就能确定下来的，所以不是太大的问题，可以直接用if分支做编译时候的判断决定。但cache是动态的，正常来说，decoder在第一次计算时候，会完整计算kv，并把这个cache记录下来，供给后面的decoder使用，同时后面的decoder在计算的时候也会更新这个cache。

参考[transformers库里的实现](transformers/src/transformers/models/whisper/modeling_whisper.py)，所以这里我们需要引入一个额外的变量，来告知模型：用不用cache，用多少cache这个问题。一开始我的想法是引入一个shape为(1,)的mask输入，通过gather(mask,0)获取mask里面记录的cache长度数值，但后来发现这种实现是数值改变后面的shape，需要运行时决断，导致模型构建失败，因为在构建时候后面的算子无法获取准确的shape。所以后面就改成了用一个(-1,)shape的mask，用mask的shape来记录cache的长度，这样的话，在给定输入的shape后，模型就能立马推断出输出的shape，输出的shape不再依赖于输入的具体数值。

### 开发与优化过程

1. 简化pytorch代码：以whisper为例，其在transformers库里的实现是很繁琐的，由于transformers需要支持大量的模型，代码中存在大量的分支，但对于我们需要的whisper来说，其中很多的代码都是冗余的，甚至会干扰我们的开发。因此最开始要先对pytorch代码进行抽丝剥茧，找到模型最本质的实现。代码中的transformers目录就是被我修改简化过的代码。
2. 明确模型的运行流程：以whisper为例，模型分为encoder和deocder，解码pipeline还需要greedy search。我们需要阅读代码，一步一步的找出模型运行的流程，对于whisper模型，可以看[DOC](./DOC.md)文档。
3. 挑软柿子下手：以whisper为例，通过阅读代码，可以发现，encoder是最简单的，核心需要的Attention也可以从trtllm中直接获取，先制作encoder模型，一层一层的实现，具体可以参照github中的commit：[add:WhisperEncoderAttention torch&trtllm](https://github.com/EdVince/whisper-trtllm/commit/32e6c86348501dbdb439c8781f61d17270171005) --> [add:WhisperEncoderLayer torch&trtllm](https://github.com/EdVince/whisper-trtllm/commit/a032479660de452ff1968b3099aa19b95352604c) --> [add:WhisperEncoder torch](https://github.com/EdVince/whisper-trtllm/commit/db4ddb1caa73397a0ccdefa5cb25f232a99434a9) --> [add:WhisperEncoder trtllm](https://github.com/EdVince/whisper-trtllm/commit/1ce15ae9bfdd8c0d9a51b5aecfa4a17c30702833)
4. 攻坚难处：以whisper为例，decoder模型所用的Attention需要支持self/cross和with/without cache。要认真思考各种实现的可能并进行尝试，找到一个可行的方向，具体不赘述。

### 优化效果

环境配置：测试环境为复赛提供的云主机和复赛docker镜像

测试代码：测试代码具体可以跳转到```tensorrt_llm_july-release-v1/examples/whisper```目录下，可以使用```python run.py --whisper whisper-xxxx.en --compare```计算相对于Huggingface的加速比，可以使用```python cal_wer --whisper whisper-xxxx.en```计算wer指标

- 精度：对于Whisper模型，我们使用wer来评价模型的精度，并与OpenAI的官方指标进行比较。对于wer指标，值越小模型精度越高。参考值来源于[leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)

***fp32+fp32表示分别表示encoder和decoder的精度***
| model  | fp32+fp32 | ref wer |
| ------ | --------- | ------- |
| tiny   | 5.61%     | 5.66%   |
| base   | 4.25%     | 4.27%   |
| small  | 3.05%     | 3.05%   |
| medium | 3.01%     | 3.02%   |

- 性能：在具有73条测试音频的[librispeech_asr_dummy](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy)数据集上对比原始Huggingface模型和TensorRT-LLM模型的加速比

***加速比***
| model  | fp32+fp32 |
| ------ | --------- |
| tiny   | 1.6X      |
| base   | 1.8X      |
| small  | 1.3X      |
| medium | 1.2X      |

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
