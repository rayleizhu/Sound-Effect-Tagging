import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np

import torchaudio


# class conf:
#     sampling_rate = 44100
#     duration = 1 # sec
#     hop_length = 345*duration # to make time steps 128
#     fmin = 20
#     fmax = sampling_rate // 2
#     n_mels = 128
#     n_fft = n_mels * 20
#     padmode = 'constant'
#     samples = sampling_rate * duration


class Common(object):
    def __init__(self, same_size=True, duration=2.0, resample_rate=44100):
        self.duration = duration
        self.resample_rate = resample_rate
        self.same_size = same_size
        
    def __call__(self, sound_tuple):
        
        # if there are only one channels, expand to two
        waveform = sound_tuple[0].expand(2, sound_tuple[0].size(1))
        new_sample_rate = sound_tuple[1]
        
        if self.same_size:
            waveform = torchaudio.transforms.Resample(sound_tuple[1], 
                                                      self.resample_rate)(waveform)
            new_sample_rate = self.resample_rate
            exp_len = int(self.duration * self.resample_rate)

            act_len = waveform.size(1)
            if act_len < exp_len:
                pad_left = (exp_len - act_len) // 2
                pad_right = exp_len - act_len - pad_left
                waveform = F.pad(waveform, pad=(pad_left, pad_right))
            else: # center cropping
                s = (act_len - exp_len) // 2
                e = s + exp_len 
                waveform = waveform[:, s:e]

        return waveform, new_sample_rate

class myMFCC:
    def __call__(self, sound_tuple):
        mfcc = torchaudio.compliance.kaldi.mfcc(sound_tuple[0],
                                                sample_frequency=sound_tuple[1])
        return mfcc, sound_tuple[1]
        

class myPermute(object):        
    def __call__(self, data_sr_tuple):
        # (c, L) -> (L, c)
        return data_sr_tuple[0].transpose(1, 0), data_sr_tuple[1]
    

class AudioTagging(Dataset):
    def __init__(self, data_root, csv_path=None, transform=None, duration=2.0):
        self.data_root = data_root
        
        if csv_path is None:
            self.audio_names = sorted(os.listdir(self.data_root))
        else:
            df =  pd.read_csv(csv_path)
            self.audio_names = df['file_name']
            self.class_names = list(df.columns)[1:]
            self.label = df.iloc[:, 1:].to_numpy()
            
        self.transform = transform
        self.duration = duration
        
    def __getitem__(self, ind):
        path = os.path.join(self.data_root, self.audio_names[ind])
        audio_data, sample_rate = torchaudio.load(path, normalization=True)
        
        if self.transform is not None:
            audio_data, sample_rate = self.transform((audio_data, sample_rate))
            
        if hasattr(self, 'label'):
            return audio_data, torch.from_numpy(self.label[ind]).float()
             
        return audio_data
    
    
    def __len__(self):
        return len(self.audio_names)
            
    