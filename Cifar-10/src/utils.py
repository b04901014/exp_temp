import numpy as np
import random
import os
from PIL import Image
from scipy.io.wavfile import read, write
from torch.utils import data
import torchvision.transforms as transforms
import multiprocessing
import torch
import torchvision
from hparams import hp

class CenterCropLongEdge(object):
    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__

def noise_adder(img):
    return torch.empty_like(img, dtype=img.dtype).uniform_(0.0, 1/128.0) + img

dset = torchvision.datasets
if not hp.train.use_myreader:
    if hp.dataset.name == 'ImageNet':
        hp.dataset.data = dset.ImageNet(root=hp.path.input.dataset, download=True,
                                        transform=transforms.Compose([
                                            CenterCropLongEdge(),
                                            transforms.Resize(hp.image.size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            noise_adder,
                                        ]))
        hp.dataset.nchannel=3

    elif hp.dataset.name == 'cifar10':
        hp.dataset.data = dset.CIFAR10(root=hp.path.input.dataset, download=True,
                                       transform=transforms.Compose([
                                           transforms.Resize(hp.image.size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           noise_adder,
                                       ]))
        hp.dataset.data, _ = torch.utils.data.random_split(hp.dataset.data, [5000, 45000])
        hp.dataset.nchannel=3

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
        self.counter = torch.zeros(initial_samples.size(0), 1, 1, 1)

    def sample(self, n):
        idx = np.random.choice(len(self.buffer), n)
        ret = self.buffer[idx]
        return ret, idx

    def replace(self, i, x):
        self.past[i] += x
        self.counter[i] += 1
        j = i[self.counter.data[i, 0, 0, 0] >= hp.train.update_count]
        self.buffer[j] = self.past[j] / self.counter[j]
        self.past[j] = torch.zeros_like(self.past[j])
        self.counter[j] = torch.zeros_like(self.counter[j])

