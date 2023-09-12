import os
import torch
import whisper
import torchaudio
import pickle

class LibriSpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, split="test-clean", device='cpu'):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=os.path.expanduser("~/.cache"),
            url=split,
            download=True,
        )
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)
        
        return (mel, text)

if __name__ == '__main__':
    dataset = LibriSpeech("test-clean")
    
    inputs = []
    for i in range(dataset.__len__()):
        input = dataset.__getitem__(i)
        inputs.append(input)
        
    with open('librispeech.cache', 'wb') as f:
        pickle.dump(inputs, f)