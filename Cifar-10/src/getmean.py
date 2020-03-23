from hparams import hp
from utils import *
from tqdm import tqdm

params = {'batch_size': hp.train.batch_size,
          'shuffle': False,
          'num_workers': hp.train.num_proc}
loader = data.DataLoader(hp.dataset.data, **params)

pbar = tqdm(total=60000, ascii=True, desc="Calculating Progress")
nproc = 0
acc = 0
for i, data in enumerate(loader, 0):
    inputs, labels = data
    nproc += hp.train.batch_size
    acc += inputs.sum(0).numpy()
    pbar.update(hp.train.batch_size)
pbar.close()
datamean = acc / nproc
np.save('mean', datamean)
