import os
import numpy as np
from hparams import hp
from model import ImageGAN
import torch

def main():
    if hp.path.output.save_dir:
        if not os.path.exists(hp.path.output.save_dir):
            os.makedirs(hp.path.output.save_dir)
    if hp.path.output.summary:
        dirname = os.path.dirname(hp.path.output.summary)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    if not os.path.exists(hp.path.output.infer):
        os.makedirs(hp.path.output.infer)

    model = ImageGAN()
    model.train()

if __name__ == '__main__':
    main()
