from utils import readwav, padtomaxlen 
from hparams import hp
import numpy as np
import os

dataset = os.listdir(hp.path.input.dataset)
dataset = [os.path.join(hp.path.input.dataset, x) for x in dataset if x[-4:] == '.wav']

a = 0
b = 0
for d in dataset:
    ret = readwav(d)
    ret = padtomaxlen(ret)
    a += ret
    b += 1
    print (b)
np.save('mean', a/b)
