rm -rf whisper_outputs

python build_encoder.py --whisper whisper-medium.en --engine_precision bfloat16
python build_decoder.py --whisper whisper-medium.en --engine_precision bfloat16

du -sh whisper_outputs/*

python cal_wer.py --whisper whisper-medium.en

python run.py --whisper whisper-medium.en --compare