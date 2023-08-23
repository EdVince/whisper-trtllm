from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("whisper-base.en")

for name, param in model.named_parameters():
    _, module_id, layer_id, *weight_name = name.split(".")
    print(module_id,layer_id,weight_name)