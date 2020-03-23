from utils  import writewav
import torch
from tqdm import tqdm
import sys

path = sys.argv[1]
checkpoint = torch.load(path)
base = checkpoint['base'].numpy()

for i, x in tqdm(enumerate(base)):
    writewav('../temp/%r.wav' %i, x)


