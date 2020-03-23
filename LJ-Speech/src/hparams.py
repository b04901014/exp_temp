import numpy as np
import argparse

class HParams():
    pass

hp = HParams()

#Audio
hp.audio = HParams()
hp.audio.sample_rate = 22050
hp.audio.time_step = 2 ** 17
hp.datamean = np.load('mean.npy')

#Model
hp.md = HParams()
hp.md.zdims = 128
hp.md.dis = HParams()
#hp.md.dis.channels = np.array([64] * 2 + [128] * 4 + [256] * 4 + [512] * 4 + [1024])
#hp.md.dis.channels = np.array([16, 32, 64, 128, 256, 512, 512, 512] + [512] * 8 + [512] * 8)
hp.md.dis.channels = np.array([128] * 16)
#hp.md.dis.filter = 64
hp.md.dis.filters = np.array([5] * 16)
hp.md.dis.strides = np.array([2] * 16)
hp.md.dis.padding = (hp.md.dis.filters - 1) // 2
#hp.md.dis.padding = 0
hp.md.generator = True
hp.md.gen = HParams()
hp.md.gen.init_timestep = 2 ** 3
hp.md.gen.init_channel = 512
hp.md.gen.channels = np.array([128] * 14)
hp.md.gen.filters = np.array([4] * 14)
hp.md.gen.strides = np.array([2] * 14)
hp.md.gen.padding = (hp.md.gen.filters - hp.md.gen.strides) // 2
hp.md.sample_base = 256

#Initialization
hp.init = HParams()
hp.init.he = True
hp.init.std = 1.0

#Training
hp.train = HParams()
hp.train.batch_size = 1
hp.train.learning_rate = 2e-4
hp.train.steps = 2000000
hp.train.num_proc = 4
hp.train.queue_size = 32
hp.train.pre_step = 0
hp.train.load_checkpoint = False
hp.train.update_count = 16

#Logging
hp.logging = HParams()
hp.logging.step = 1000
hp.logging.savestep = 2000
hp.logging.save_maxtokeep = 2
hp.logging.samplestep = 2000
hp.logging.sample_num = 3

#Path
hp.path = HParams()
hp.path.input = HParams()
hp.path.output = HParams()
hp.path.input.dataset = '/home/jovyan/LJSpeech-1.1/wavs'
hp.path.input.load_path = '../model/Step_942000.pt'
hp.path.output.save_dir = '../model'
hp.path.output.sample = '../samples'
hp.path.output.summary = '../summary'
