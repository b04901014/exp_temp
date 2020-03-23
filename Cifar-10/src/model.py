import os
import traceback
import itertools
import torch
from tensorboard_logger import configure, log_value
from tqdm import tqdm
from hparams import hp
from utils import *
from module import *
from module_res import *
import threading

import torch.distributed as dist
from torch.nn.parallel import DataParallel as DP

from calfid.fid_score import calculate_fid_given_paths

def override_hp(_hp):
    global hp
    hp = _hp

def decay_lr(opt, max_iter, start_iter, initial_lr):
    coeff = -initial_lr / (max_iter - start_iter)
    for pg in opt.param_groups:
        pg['lr'] += coeff

class ImageGAN():
    def thread_data(self):
        try:
            real_data, label = next(self.dataloader)
            self.real_data = real_data.to(self.device)
        except StopIteration:
            self.dataloader = iter(self.dataiterable)
            if hp.train.num_proc > 0:
                self.dataloader.data_queue.maxsize = hp.train.queue_size
            real_data, label = next(self.dataloader)
            self.real_data = real_data.to(self.device)

    def thread_post(self, go, io):
        if not hp.logging.stds:
            self.s1 = np.std(self.x.detach().cpu().numpy())
            self.s2 = np.std(np.std(self.x.detach().cpu().numpy(), axis=(1, 2, 3)))
            self.s3 = np.std(io.detach().cpu().numpy())
            self.s4 = np.std(np.std(io.detach().cpu().numpy(), axis=(1, 2, 3)))
        self.pbar.set_postfix(s=self.s1,
                              std2=self.s2,
                              std3=self.s3,
                              std4=self.s4)
        if hp.path.output.summary:
            try: 
                log_value('l1', self.loss1, self.step)
                log_value('l2', self.loss2, self.step)
                log_value('l3', self.loss3, self.step)
            except:
                tqdm.write("Failed adding summary, maybe no space left...")
        self.buffers.put([self.loss1,
                     self.loss2,
                     self.loss3],
                     [0, 1, 2])

    def train(self):
        self.device = torch.device("cuda")
        self.model = Improver_res() if hp.md.resblock else Improver()
        self.model.loginfo()
        if hp.train.GAN_type == 'OGAN' or hp.train.GAN_type == 'LSGAN':
            self.criterion = lambda x, y: torch.mean((x - y) ** 2)
        elif hp.train.GAN_type == 'SNGAN':
            self.criterion = nn.BCEWithLogitsLoss()
        self.angle = lambda x, y: torch.mean((x * y).sum((1, 2, 3)) / (x ** 2).sum((1, 2, 3)) ** 0.5 / (y ** 2).sum((1, 2, 3)) ** 0.5)
        params = {'batch_size': hp.train.batch_size,
                  'shuffle': True,
                  'pin_memory': True,
                  'drop_last': True,
                  'num_workers': hp.train.num_proc}
        self.dataiterable = data.DataLoader(hp.dataset.data, **params)
        self.dataloader = iter(self.dataiterable)
        if hp.train.num_proc > 0:
            self.dataloader.data_queue.maxsize = hp.train.queue_size

        self.checkpoints = []
        self.fids = []
        if hp.logging.infer_after_F:
            self.post_fids = []
        if hp.logging.stds:
            self.std_r, self.std_g = [], []
            self.std_Fr, self.std_Fg = [], []
        self.model.weight_init()
        print ("Using ", torch.cuda.device_count(), "GPUs")
        self.model.to(self.device)
        self.model = DP(self.model)
        self.step = 1
        self.generator = Generator_res() if hp.md.resblock else Generator()
        self.generator.loginfo()
        self.generator.weight_init()
        self.generator.to(self.device)
        self.generator = DP(self.generator)
        params = self.model.parameters()
        self.optimizer_imp = torch.optim.Adam(params, lr=hp.train.learning_rate, betas=(0.0, 0.9))
        params = self.generator.parameters()
        self.optimizer_gen = torch.optim.Adam(params, lr=hp.train.learning_rate, betas=(0.0, 0.9))
        if hp.train.load_checkpoint:
            self.load()
        else:
            print ("Training Model From Beginning!")

        if hp.path.output.summary:
            configure(hp.path.output.summary)

        loss_names = ["Loss1",
                      "Loss2",
                      "Loss3",
                      "Loss4"]
        self.buffers = buff(loss_names)
        self.pbar = tqdm(total=hp.train.steps, ascii=True, desc="Training Progress")
        if self.step != 1:
            self.pbar.update(self.step)
        try:
            self.thread_data()
#            self.slider = 1.0
            while self.step <= hp.train.steps:
#                if hp.train.learning_rate_decay:
#                    decay_lr(self.optimizer_gen, hp.train.steps, hp.train.learning_rate_decay_start, hp.train.learning_rate)
#                    decay_lr(self.optimizer_imp, hp.train.steps, hp.train.learning_rate_decay_start, hp.train.learning_rate)
                self.thread_data()
                for i in range(hp.train.ndis):
                    self.model.zero_grad()
                    io, l2 = self.model(self.real_data)
                    noise = torch.randn(hp.train.batch_size, hp.md.gen.noisedim, 1, 1, device=self.device)
                    self.x = self.generator(noise)
                    go, l1 = self.model(self.x.detach())
                    if hp.train.GAN_type == 'NSGAN' or hp.train.GAN_type == 'LSGAN':
                        loss1 = self.criterion(l1, torch.zeros_like(l1))
                        loss2 = self.criterion(l2, torch.ones_like(l2))
                        losses = loss1 + loss2
                    elif hp.train.GAN_type == 'WGAN-GP':
                        loss1 = -l1.mean()
                        loss2 = l2.mean()
                        a = torch.rand(hp.train.batch_size, 1, 1, 1, device=self.device)
                        interp = a * self.real_data + (1 - a) * self.x
                        interp = interp.requires_grad_()
                        _, out = self.model(interp)
                        grads = autograd.grad(outputs=out,
                                              inputs=interp,
                                              grad_outputs=torch.ones_like(out),
                                              create_graph=True,
                                              retain_graph=True,
                                              only_inputs=True)[0]
                        gploss = ((grads.view(grads.size(0), -1).norm(2, dim=1) - 1) ** 2).mean() * 10.
                        losses = loss1 + loss2 + gploss
                    elif hp.train.GAN_type == 'OGAN':
                        loss1 = self.criterion(io, self.real_data * 2)# + sum(l2)# + (l2 ** 2).mean()
                        loss_log = self.criterion(io, self.real_data)
                        loss2 = torch.mean(go ** 2)
                        losses = loss1 + loss2
                    elif hp.train.GAN_type == 'SNGAN':
#                        loss1 = torch.mean(F.relu(1 + l1))
#                        loss2 = torch.mean(F.relu(1 - l2))
                        loss1 = self.criterion(l1, torch.zeros_like(l1))
                        loss2 = self.criterion(l2, torch.ones_like(l2))
                        loss_log = loss1
                        losses = loss1 + loss2
#                        tt = io.detach()
#                        tt = tt / tt.std() * self.real_data.std()
#                        print (torch.mean((tt - self.real_data) ** 2))
                    losses.backward()
                    self.optimizer_imp.step()
                        
                for i in range(hp.train.ngen):
                    self.optimizer_gen.zero_grad()
                    noise = torch.randn(hp.train.batch_size, hp.md.gen.noisedim, 1, 1, device=self.device)
                    self.x = self.generator(noise)
                    go, l1 = self.model(self.x)
                    if hp.train.GAN_type == 'NSGAN' or hp.train.GAN_type == 'LSGAN':
                        loss3 = self.criterion(l1, torch.ones_like(l1))
                    elif hp.train.GAN_type == 'WGAN' or hp.train.GAN_type == 'WGAN-GP':
                        loss3 = l1.mean()
                    elif hp.train.GAN_type == 'OGAN':
                        tar = (go).detach()#.clamp(-1, 1)
                        loss3 = self.criterion(self.x, tar)
#                        loss3 = -torch.mean((self.x * go.detach()).sum((1, 2, 3))) / 100000
                    elif hp.train.GAN_type == 'SNGAN':
#                        tar = (go*32*32).detach()#.clamp(-1, 1)
#                        loss3 = -torch.mean(torch.log(torch.sigmoid((go.detach() * self.x).sum((1, 2, 3)))))
                        loss3 = self.criterion(l1, torch.ones_like(l1))
#                        loss3 = -torch.mean(l1)
                    loss3.backward()
                    self.optimizer_gen.step()
                self.loss1 = loss_log.item()
                self.loss2 = loss2.item()
                self.loss3 = loss3.item()
                if hp.logging.stds:
                    self.s1 = np.std(go.detach().cpu().numpy())
                    self.s2 = np.std(self.x.detach().cpu().numpy())
                    self.s3 = np.std(io.detach().cpu().numpy())
                    self.s4 = np.std(self.real_data.detach().cpu().numpy())
                    self.std_Fr.append(self.s3)
                    self.std_r.append(self.s4)
                    self.std_Fg.append(self.s1)
                    self.std_g.append(self.s2)
                self.thread_post(go, io)
                self.pbar.update(1)
                self.step += 1
                if self.step % hp.logging.savestep == 0 and hp.path.output.save_dir and hp.logging.enable:
                    try :
                        self.save()
                    except:
                        tqdm.write("Failed saving model, maybe no space left...")
                        traceback.print_exc()
                if self.step % hp.logging.inferstep == 0:
                    self.infer()
                if self.step % hp.logging.step == 0 and hp.logging.enable:
                    tqdm.write(self.buffers.getstring([self.step, hp.train.steps]))
                if self.step > hp.train.steps:
                    break
        except KeyboardInterrupt:
            print ("KeyboardInterrupt")
        except:
            traceback.print_exc()
        finally:
            if hp.train.use_myreader:
                for x in self.procs:
                    x.terminate()
            self.pbar.close()
            if hp.path.output.summary:
                unconfigure()

    def save(self):
        if len(self.checkpoints) == hp.logging.save_maxtokeep:
            to_remove = os.path.join(hp.path.output.save_dir, self.checkpoints[0])
            tqdm.write("Remove old checkpoint %r from disk" %to_remove)
            os.remove(to_remove)
            self.checkpoints = self.checkpoints[1:]
        name = 'Step_' + str(self.step) + '.pt'
        saving_path = os.path.join(hp.path.output.save_dir, name)
        self.save_config = {
                               'step': self.step,
                               'generator_state_dict': self.generator.module.state_dict(),
                               'improver_state_dict': self.model.module.state_dict(),
                               'optimizer_state_dict_gen': self.optimizer_gen.state_dict(),
                               'optimizer_state_dict_imp': self.optimizer_imp.state_dict()
                           }
        tqdm.write("Saving model to %r" %saving_path)
        torch.save(self.save_config, saving_path)
        self.checkpoints.append(name)

    def load(self):
        assert os.path.isfile(hp.path.input.load_path), "Checkpoint %r does not exists" %hp.path.input.load_path
        print ("Loading Model From %r" %hp.path.input.load_path)
        checkpoint = torch.load(hp.path.input.load_path)
        self.model.module.load_state_dict(checkpoint['improver_state_dict'])
        opt_dict_imp = checkpoint['optimizer_state_dict_imp']
        self.optimizer_imp.load_state_dict(opt_dict_imp)
        opt_dict_gen = checkpoint['optimizer_state_dict_gen']
        self.optimizer_gen.load_state_dict(opt_dict_gen)
        self.step = checkpoint['step']
        self.generator.module.load_state_dict(checkpoint['generator_state_dict'])

    def infer(self):
        try:
            procimages = 0
            dirname = hp.path.output.infer
            tqdm.write('Sampling to %r ...' %dirname)
            pbar = tqdm(total=hp.logging.infer_num, ascii=True, desc="Inference Progress")
            while procimages < hp.logging.infer_num:
                if procimages + hp.train.batch_size > hp.logging.infer_num:
                    nproc = hp.logging.infer_num - procimages
                else:
                    nproc = hp.train.batch_size
                noise = torch.randn(nproc, hp.md.gen.noisedim, 1, 1, device=self.device)
                samples = self.generator(noise).detach()
                if hp.logging.infer_after_F:
                    samp_post = self.model(samples)[0].detach().cpu()
                samples = samples.cpu()
                samples = (samples + 1) / 2.
                samples = (samples * 255 + 0.5).clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
                if hp.logging.infer_after_F:
                    samp_post = (samp_post + 1) / 2.
                    samp_post = (samp_post * 255 + 0.5).clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
                for i in range(samples.shape[0]):
                    fp = os.path.join(dirname, str(procimages)+'.png')
                    img = Image.fromarray(samples[i])
                    img.save(fp)
                    if hp.logging.infer_after_F:
                        fp2 = os.path.join(hp.path.output.infer_post, str(procimages)+'.png')
                        img = Image.fromarray(samp_post[i])
                        img.save(fp2)
                    procimages += 1
                    pbar.update(1)
            pbar.close()
            tqdm.write('Calculating FID...')
            fid_value = calculate_fid_given_paths([hp.path.output.infer, hp.path.input.fidset],
                                                  hp.fid_batch_size,
                                                  True,
                                                  2048)
            tqdm.write('FID: %.4f' %fid_value)
            self.fids.append(fid_value)
            if hp.logging.infer_after_F:
                post_fid_value = calculate_fid_given_paths([hp.path.output.infer_post, hp.path.input.fidset],
                                                            hp.fid_batch_size,
                                                            True,
                                                            2048)
                self.post_fids.append(post_fid_value)
                tqdm.write('FID POST: %.4f' %post_fid_value)
            name = '%s-%i-FID' %(hp.train.GAN_type, hp.logging.inferstep)
            if hp.path.output.summary:
                np.save(os.path.join(hp.path.output.summary, name), self.fids)
        except KeyboardInterrupt:
            print ("KeyboardInterrupt")
        except:
            traceback.print_exc()
