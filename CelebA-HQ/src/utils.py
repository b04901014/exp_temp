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

class MyReader():
    def __init__(self):
        self.manager = multiprocessing.Manager()
        self.queue = multiprocessing.Queue(hp.train.queue_size)
#        self.sum = multiprocessing.Value('f', 0.0)
#        self.count = multiprocessing.Value('i', 0)
        self.load_datasets()

    def load_datasets(self):
        self.dataset = os.listdir(hp.path.input.dataset)
        self.dataset = [os.path.join(hp.path.input.dataset, x) for x in self.dataset if x[-4:] == '.npy']
        self.n_examples = len(self.dataset)
        print ("Total number of Images: %r" %self.n_examples)
        self.dataset = self.manager.list(self.dataset)
        self.run_num = hp.train.batch_size

    def main_proc(self, seed):
        r = np.random.RandomState(seed * 7)
        while True:
            As = []
            idx = r.choice(self.n_examples, size=self.run_num)
            for i in idx:
                ret = np.load(self.dataset[i])[0].astype(np.float32)
                ret = (ret / 255. - 0.5) * 2.# - hp.datamean
                ret = ret - ret.mean()
#                self.sum.value += np.mean(ret)
#                self.count.value += 1
#                mean = self.sum.value / self.count.value
#                ret -= mean
                As.append(ret)
            self.queue.put(torch.from_numpy(np.array(As)).to(torch.device("cuda")))

    def dequeue(self):
        return self.queue.get()

    def start_enqueue(self):
        procs = []
        for i in range(hp.train.num_proc):
            p = multiprocessing.Process(target=self.main_proc, args=(i + 1,))
            p.start()
            procs.append(p)
        return procs

    def printqsize(self):
        print ("Queue Size : ", self.queue.qsize())

class ImageDataset():
    def __init__(self, transform):
        self.dataset = os.listdir(hp.path.input.dataset)
        self.dataset = [os.path.join(hp.path.input.dataset, x) for x in self.dataset]
        if hp.dataset.name == 'celebA-HQ':
            self.dataset = [x for x in self.dataset if x[-4:] == '.npy']
        self.transform = transform
        self.n_examples = len(self.dataset)
        print ("Total number of Images: %r" %self.n_examples)

    def __len__(self):
        return self.n_examples

    def __getitem__(self, i):
        if hp.dataset.name == 'celebA-HQ':
            ret = np.load(self.dataset[i])[0].astype(np.float32)
            ret = (ret / 255. - 0.5) * 2.
            ret = ret - hp.datamean
        else:
            ret = Image.open(self.dataset[i])
            ret = self.transform(ret)
        return ret

if not hp.train.use_myreader:
    if hp.dataset.name == 'lsun':
        hp.dataset.data = dset.LSUN(root=hp.path.input.dataset, classes=['bedroom_train'],
                                    transform=transforms.Compose([
                                        transforms.Resize(hp.image.size),
                                        transforms.CenterCrop(hp.image.size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ]))
        hp.dataset.nchannel=3

    elif hp.dataset.name == 'cifar10':
        hp.dataset.data = dset.CIFAR10(root=hp.path.input.dataset, download=True,
                                       transform=transforms.Compose([
                                           transforms.Resize(hp.image.size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ]))
        hp.dataset.nchannel=3

    elif hp.dataset.name == 'celebA':
        hp.dataset.data = ImageDataset(transform=transforms.Compose([
                              transforms.Resize(hp.image.size),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]))
        
        hp.dataset.nchannel=3

    elif hp.dataset.name == 'celebA-HQ':
        hp.dataset.data = ImageDataset(transform=transforms.Compose([
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]))
        
        hp.dataset.nchannel=3

    elif hp.dataset.name == 'mnist':
        hp.dataset.data = dset.MNIST(root=hp.path.input.dataset, download=True,
                                     transform=transforms.Compose([
                                         transforms.Resize(hp.image.size),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,)),
                                     ]))
        hp.dataset.nchannel=1


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

