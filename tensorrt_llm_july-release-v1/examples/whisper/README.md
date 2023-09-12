
# Download huggingface model

Please download the model you want from Huggingface to the local.

Now we support [whisper-tiny.en](https://huggingface.co/openai/whisper-tiny.en), [whisper-base.en](https://huggingface.co/openai/whisper-base.en), [whisper-small.en](https://huggingface.co/openai/whisper-small.en), and [whisper-medium.en](https://huggingface.co/openai/whisper-medium.en)

e.g. You can download the medium model by following command:
```
git lfs install
git clone https://huggingface.co/openai/whisper-medium.en
```

# Build Whisper model

Before buliding, please make sure you have downloaded model.

You can build the encoder and decoder engine by following command, e.g. building medium model:
```
python build_encoder.py --whisper whisper-medium.en
python build_decoder.py --whisper whisper-medium.en
```

By default, the generated engine will be saved in ```whisper_outputs``` directory with float32 precision.

# Run Whisper Engine and Compare with Huggingface

Here we run engine in [hf-internal-testing/librispeech_asr_dummy](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy) dataset. In the first time, we will download the dataset by network and save it to disk, so we will only download it one time.

You can run the TensorRT-LLM Whisper engine by following command:
```
python run.py --whisper whisper-medium.en
```

If you want to compare results with Huggingface, you can add flag ```--compare```.

# Evaluate Engine

Here, we calculate the [wer](https://huggingface.co/spaces/evaluate-metric/wer) of the engine on [LibriSpeech/clean/test](https://huggingface.co/datasets/librispeech_asr/viewer/clean/test).

You can find the reference wer on [leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard).

First, we need to get the dataset by following command, it will save the dataset in disk:
```
python get_LibriSpeech.py
```

Then, you can calculate the wer by following command:
```
python cal_wer.py --whisper whisper-medium.en
```