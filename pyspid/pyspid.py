import torch
import torchaudio

from pyspid.effects import PreEmphasis

def load_audio(path):
    wav, sr = torchaudio.load(path)

class PySpidDataset(torch.utils.data.Dataset):
    """PySpid data pipeline."""

    def __init__(self, filenames, speaker_names, sampling_rate = 16000, audio_augmentation = [], features = None, feature_augmentation = [], device = torch.device('cpu'), apply_preemphasis = False):
        """
        augmentation = [(aug_1, p = 0.2), ...]
        """
        # assuming no vad is required
        self.filenames = filenames
        self.speaker_names = speaker_names
        self.audio_augmentation = audio_augmentation
        self.sampling_rate = sampling_rate
        self.feature_augmentation = feature_augmentation
        self.features = features
        self.device = device
        self.apply_preemphasis = apply_preemphasis

        # speaker classes
        self.speaker_classes = sorted(list(set(self.speaker_names)))


    def __len__(self):
        return len(self.speaker_names)

    def __getitem__(self, idx):
        # load audio
        y, sr = torchaudio.load(self.filenames[idx])
        print(y.shape)
        if sr != self.sampling_rate:
            # resample
            resample = torchaudio.transforms.Resample(sr, self.sampling_rate)
            y = resample(y)
        # audio augmentation
        # https://github.com/iver56/audiomentations

        """
        In FM broadcasting, preemphasis improvement is the improvement in the signal-to-noise ratio of the high-frequency portion of the baseband, i.e., modulating signal, which improvement results from passing the modulating signal through a preemphasis network before transmission.
        """ 
        print(y.shape)
        if self.apply_preemphasis:
            y = PreEmphasis(y)
        print(y.shape)

        # feature extraction
        if self.features != None:
            # assuming single feature right now
            y = self.features(y)
        print(y.shape)


        # speaker label
        sp_lab = self.speaker_classes.index(self.speaker_names[idx])
        return y, sp_lab

def get_pyspid_dataloader(filenames, speaker_names, sampling_rate = 16000, audio_augmentation = [], features = None, feature_augmentation = [], device = torch.device('cpu'), apply_preemphasis = False, batch_size = 1, num_workers = 0):
    pyspid_dataset = PySpidDataset(filenames, speaker_names, sampling_rate, audio_augmentation, features, feature_augmentation, device, apply_preemphasis)
    pyspid_dataloader = torch.utils.data.DataLoader(pyspid_dataset, batch_size = batch_size, num_workers = num_workers)
    return pyspid_dataloader