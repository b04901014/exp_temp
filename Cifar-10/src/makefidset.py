from hparams import hp
from utils import *
from tqdm import tqdm

params = {'batch_size': hp.train.batch_size,
          'shuffle': True,
          'num_workers': hp.train.num_proc}
loader = data.DataLoader(hp.dataset.data, **params)

pbar = tqdm(total=hp.logging.infer_num, ascii=True, desc="Generating Progress")
nproc = 0
for i, data in enumerate(loader, 0):
    inputs, labels = data
    for j in range(inputs.size(0)):
        fp = os.path.join(hp.path.input.fidset, str(nproc)+'.png')
        img = (inputs[j] + 1) / 2.
        img = (img * 255 + 0.5).clamp_(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
        img = Image.fromarray(img)
        img.save(fp)
        nproc += 1
        pbar.update(1)
    if nproc >= hp.logging.infer_num:
        pbar.close()
        break
