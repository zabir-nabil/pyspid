from pyspid import get_pyspid_dataloader
import torch
import torchaudio

dummy_psd = get_pyspid_dataloader(filenames = ["test.wav", "test.wav"], speaker_names = ["test", "test2"], features = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400,  hop_length=160, f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80), batch_size = 16)

for d, l in dummy_psd:
    print(d.shape)
    print(l.shape)

