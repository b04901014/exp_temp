import numpy as np
import argparse

class HParams():
    pass

hp = HParams()

#Image
hp.image = HParams()
hp.image.size = 32

#Other Settings
hp.fid_batch_size = 100

#Model
hp.md = HParams()
hp.md.imp = HParams()
hp.md.imp.channels = np.array([64, 64, 64, 64, 64, 64, 64, 64, 64, 128, 1])
hp.md.imp.filters = np.array([3, 3, 4, 3, 3, 4, 3, 3, 4, 3, 4])
hp.md.imp.strides = np.array([1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1])
hp.md.imp.pads = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
hp.md.imp.with_bias = False

#hp.md.imp.channels = np.array([32, 64, 128, 256, 512, 1])
#hp.md.imp.filters = np.array([11, 8, 5, 4, 4, 4])
#hp.md.imp.strides = np.array([1, 2, 1, 2, 2, 1])
#hp.md.imp.pads = np.array([5, 3, 2, 1, 1, 0])

hp.md.gen = HParams()
hp.md.gen.noisedim = 128
hp.md.gen.initc = 512
hp.md.gen.inits = 4
hp.md.gen.channels = np.array([256, 128, 64, 3])
hp.md.gen.filters = np.array([4, 4, 4, 3])
hp.md.gen.strides = np.array([2, 2, 2, 1])
hp.md.gen.pads = np.array([1, 1, 1, 1])

hp.md.resblock = True

#Training
hp.train = HParams()
hp.train.batch_size = 64
hp.train.learning_rate = 2e-4
hp.train.learning_rate_decay = True
hp.train.learning_rate_decay_start = 0
hp.train.beta1 = 0.0
hp.train.beta2 = 0.9
hp.train.steps = 150000
hp.train.use_myreader = False
hp.train.num_proc = 4
hp.train.queue_size = 32
hp.train.ndis = 1
hp.train.ngen = 1
hp.train.load_checkpoint = False
hp.train.update_count = 1
hp.train.GAN_type = 'OGAN'
hp.train.spnorm = False
hp.train.bn_g = True
#hp.train.threading = False

#Logging
hp.logging = HParams()
hp.logging.step = 500
hp.logging.savestep = 20000
hp.logging.save_maxtokeep = 1
hp.logging.inferstep = 5000
hp.logging.infer_num = 10000
hp.logging.enable = True
hp.logging.infer_after_F = False
hp.logging.stds = False

#Path
hp.path = HParams()
hp.path.input = HParams()
hp.path.output = HParams()
hp.path.input.dataset = '/home/jovyan/cifar10'
hp.path.input.fidset = 'calfid/fid_stats_cifar10_train.npz'
hp.path.input.load_path = '../model/Step_100000.pt'
hp.path.output.save_dir = '../model'
hp.path.output.infer = '../outputs'
hp.path.output.infer_post = '../outputs_post'
hp.path.output.summary = '../summary'

#Dataset
hp.dataset = HParams()
hp.dataset.name = 'cifar10'
hp.dataset.nchannel = 3
hp.dataset.nclasses = 10
