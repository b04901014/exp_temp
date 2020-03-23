import numpy as np
import random
import os
import librosa
from scipy.io.wavfile import read, write
from torch.utils import data
import torch
from hparams import hp

def writewav(path, arr):
    arr = np.clip(arr * 32768 / 8, -32768, 32767).astype(np.int16)
#    arr = arr + hp.datamean
#    arr = np.clip(arr * 32768, -32768, 32767).astype(np.int16)
    write(path, hp.audio.sample_rate, arr)

def readwav(path):
    _, y = read(path)
    assert _ == hp.audio.sample_rate
    y = y.astype(np.float32) / 32768. * 8
    y = y - np.mean(y)
    return y

def padtomaxlen(wav):
    if wav.shape[0] > hp.audio.time_step:
        rn = random.randint(0, wav.shape[0] - hp.audio.time_step)
#        rn = 0
        wav = wav[rn: rn + hp.audio.time_step]
    else:
        pad = np.zeros([hp.audio.time_step - wav.shape[0]] + list(wav.shape[1:]))
        wav = np.concatenate([wav, pad], axis=0)
    wav = wav.astype(np.float32)
    return wav
 
class LJSpeechDataset(data.Dataset):
    def __init__(self):
        self.dataset = os.listdir(hp.path.input.dataset)
        self.dataset = [os.path.join(hp.path.input.dataset, x) for x in self.dataset if x[-4:] == '.wav']
        self.n_examples = len(self.dataset)
        print ("Total number of wavs: %r" %self.n_examples)

    def __len__(self):
        return self.n_examples

    def __getitem__(self, i):
        ret = readwav(self.dataset[i])
        ret = padtomaxlen(ret)
#        ret = ret / ret.std() * 0.5
#        ret = ret - hp.datamean
        return ret

class buff():
    def __init__(self, loss_names):
        self.loss_names = loss_names
        self.buffers = [0. for x in self.loss_names]
        self.count = [0 for x in self.loss_names]
        self.loss_string = "Step : %r / %r "
        for loss_name in loss_names:
            self.loss_string = self.loss_string + loss_name + " : %.6f "

    def put(self, x, index):
        assert len(x) == len(index)
        for y, idx in zip(x, index):
            self.buffers[idx] += y
            self.count[idx] += 1.

    def getstring(self, prefix):
        losses = tuple(prefix + [x / c if c != 0 else 0 for x, c in zip(self.buffers, self.count)])
        self.buffers = [0. for x in self.buffers]
        self.count = [0 for x in self.buffers]
        return self.loss_string %losses

    def get(self, index):
        return self.buffers[index] / self.count[index] if self.count[index] != 0 else 0

class MovingDistribution:
    def __init__(self, initial_samples):
        self.buffer = initial_samples
        self.past = torch.zeros_like(self.buffer)
        self.counter = torch.zeros(initial_samples.size(0), 1)
    
    def sample(self, n): 
        idx = np.random.choice(len(self.buffer), n)
        ret = self.buffer[idx]
        return ret, idx 

    def replace(self, i, x): 
        self.past[i] += x
        self.counter[i] += 1
        j = i[self.counter[i, 0] >= hp.train.update_count]
        self.buffer[j] = self.past[j] / self.counter[j]
        self.past[j] = 0 
        self.counter[j] = 0 

