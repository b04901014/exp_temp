import numpy as np
import argparse

class HParams():
    pass

hp = HParams()

#Image
hp.image = HParams()
hp.image.size = 1024

hp.datamean = np.load('mean.npy')

#Model
hp.md = HParams()
#hp.md.channels = np.array([64, 128, 128, 256, 256, 512, 512, 1024])
#hp.md.channels = np.array([16] * 5 + [32] * 5 + [64] * 5 + [256] * 5 + [512] * 4)
hp.md.channels = np.array([128, 128, 256, 256, 512, 512, 1024])
#hp.md.filter = 5
#hp.md.filters = np.array([3] * 24)
hp.md.filters = np.array([5, 5, 3, 3, 3, 3, 3])
#hp.md.stride = 2
#hp.md.strides = np.array([2, 2, 1, 1, 1] * 2 + [2, 1, 1, 1, 1] * 1 + [1, 1, 1, 1, 1] * 2)
hp.md.strides = np.array([4, 4, 2, 2, 2, 2, 2])
#hp.md.padding = (hp.md.filter - 1) // 2
hp.md.sample_base = 1024

#Training
hp.train = HParams()
hp.train.batch_size = 2
hp.train.learning_rate = 2e-4
hp.train.steps = 10000000
hp.train.pre_step = 0
hp.train.use_myreader = False
hp.train.num_proc = 8
hp.train.queue_size = 32
hp.train.load_checkpoint = True
hp.train.update_count = 1
hp.train.use_generator = True

#Logging
hp.logging = HParams()
hp.logging.step = 500
hp.logging.savestep = 1000
hp.logging.save_maxtokeep = 1
hp.logging.samplestep = 1000
hp.logging.sample_num = 9

#Path
hp.path = HParams()
hp.path.input = HParams()
hp.path.output = HParams()
hp.path.input.dataset = '/home/b04901014/dataset'
hp.path.input.load_path = '../model/Step_965000.pt'
hp.path.output.save_dir = '../model'
hp.path.output.sample = '../samples'
hp.path.output.summary = '../summary'

#Dataset
hp.dataset = HParams()
hp.dataset.name = 'celebA-HQ'
hp.dataset.nchannel = 3
