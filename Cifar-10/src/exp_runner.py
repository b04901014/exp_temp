import os
import numpy as np
from hparams import hp
from model import *

#Global Config
hp.logging.enable = True
hp.train.batch_size = 64
hp.train.learning_rate = 2e-4
hp.train.steps = 120000
hp.logging.inferstep = 5000
hp.train.load_checkpoint = False
hp.path.output.summary = None
hp.path.output.expres = '../expres'
hp.result_name = 'results.npy'
hp.logging.stds = False
if hp.logging.stds:
    hp.result_name = 'results_std.npy'

#Global Vars
runiters = 1
fn = os.path.join(hp.path.output.expres, hp.result_name)
try:
    results = np.load(fn, allow_pickle=True).item()
    print ('Loading Training Results from Previous Runs and will Override...')
    print ('Already Completed Experiments: ', results.keys())
except:
    results = {}
    print ('Initializing new Results')

if hp.path.output.expres:
    if not os.path.exists(hp.path.output.expres):
        os.makedirs(hp.path.output.expres)

def run1conf(hp):
    print ('Running Experiment: %s ...' %hp.name)
    global results
    if hp.logging.stds:
        Fr, r, Fg, g = [], [], [], []
    else:
        runs = []
    if hp.logging.infer_after_F:
        runs_post = []
    override_hp(hp)
    for i in range(runiters):
        print ('Running Iteration %i / %i ...' %(i+1, runiters))
        model = ImageGAN()
        model.train()
        if hp.logging.stds:
            Fr.append(model.std_Fr)
            r.append(model.std_r)
            Fg.append(model.std_Fg)
            g.append(model.std_g)
        else:
            runs.append(model.fids)
            if hp.logging.infer_after_F:
                runs_post.append(model.post_fids)
        del model
    if hp.logging.stds:
        results[hp.name+'_Fr'] = np.array(Fr)
        results[hp.name+'_r'] = np.array(r)
        results[hp.name+'_Fg'] = np.array(Fg)
        results[hp.name+'_g'] = np.array(g)
    else:
        results[hp.name] = np.array(runs)
        if hp.logging.infer_after_F:
            results[hp.name + '_after_F'] = np.array(runs_post)
    np.save(fn, results)


'''
hp.name = 'Proposed_64'
hp.train.GAN_type = 'OGAN'
hp.train.with_bias = False
run1conf(hp)

hp.name = 'WGAN-GP_64'
hp.train.GAN_type = 'WGAN-GP'
hp.train.with_bias = True
run1conf(hp)

hp.name = 'LSGAN_64'
hp.train.GAN_type = 'LSGAN'
hp.train.with_bias = True
run1conf(hp)

hp.name = 'LSGAN_64_nb_nr'
hp.train.GAN_type = 'LSGAN'
hp.train.with_bias = False
run1conf(hp)

hp.name = 'NSGAN_64'
hp.train.GAN_type = 'NSGAN'
hp.train.with_bias = True
run1conf(hp)

hp.name = 'Proposed_64_2'
hp.train.GAN_type = 'OGAN'
hp.train.with_bias = False
hp.logging.infer_after_F = True
run1conf(hp)
'''

hp.name = 'Proposed_new_64_ndis5'
hp.train.GAN_type = 'OGAN'
hp.md.imp.with_bias = False
hp.train.midinit = False
hp.train.spnorm = False
hp.logging.infer_after_F = False
hp.train.steps = 200000
run1conf(hp)

