from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("whisper-medium.en")

for name, param in model.named_parameters():
    _, module_id, layer_id, *weight_name = name.split(".")
    if layer_id == 'layers':
        print(module_id,weight_name)