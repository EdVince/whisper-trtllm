import warnings
warnings.filterwarnings("ignore")

import numpy as np

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, load_from_disk

# load model and processor
processor = WhisperProcessor.from_pretrained("whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("whisper-tiny.en")

# load dummy dataset and read audio files
# ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# ds.save_to_disk('./librispeech_asr_dummy')
ds = load_from_disk('./librispeech_asr_dummy')
sample = ds[0]["audio"]

input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features
# print(sample['array'].shape,input_features.shape)

# print(input_features)

# generate token ids
predicted_ids = model.generate(input_features)

# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
# print(transcription)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription)