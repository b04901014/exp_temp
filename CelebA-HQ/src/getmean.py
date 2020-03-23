import numpy as np
import os

pdataset = '/home/b04901014/dataset'
dataset = os.listdir(pdataset)
dataset = [os.path.join(pdataset, x) for x in dataset]
c = 0
gg = 0
for i in range(len(dataset)):
    print (i)
    ret = np.load(dataset[i])[0].astype(np.float32)
    ret = (ret / 255. - 0.5) * 2.
    c += ret
    gg += 1
    datamean = c / gg
np.save('mean', datamean)
