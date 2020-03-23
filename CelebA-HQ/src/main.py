import os
import numpy as np
from hparams import hp
from model import ImageGAN
import torch

def main():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12459'
    if hp.path.output.save_dir:
        if not os.path.exists(hp.path.output.save_dir):
            os.makedirs(hp.path.output.save_dir)
    if hp.path.output.sample:
        if not os.path.exists(hp.path.output.sample):
            os.makedirs(hp.path.output.sample)
    if hp.path.output.summary:
        dirname = os.path.dirname(hp.path.output.summary)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    model = ImageGAN()
    ngpu = torch.cuda.device_count()
    torch.multiprocessing.spawn(model.train, nprocs=ngpu)

if __name__ == '__main__':
    main()
