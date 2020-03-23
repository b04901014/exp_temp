import os
import traceback
import itertools
import torch
from tensorboard_logger import configure, log_value
from tqdm import tqdm
from hparams import hp
from utils import *
from module import *

class SpeechGAN():
    def __init__(self):
        self.device = torch.device("cuda")

    def train(self):
        self.model = CombinedModel()
        self.model.loginfo()
        self.criterion = lambda x, y: torch.mean((x - y) ** 2)
        self.angle = lambda x, y: (x * y).sum(1) / (x ** 2).sum(1) ** 0.5 / (y ** 2).sum(1) ** 0.5
        params = {'batch_size': hp.train.batch_size,
                  'shuffle': True,
                  'pin_memory': True,
                  'num_workers': hp.train.num_proc}
        self.dataiterable = data.DataLoader(LJSpeechDataset(), **params)
        self.dataloader = iter(self.dataiterable)
        if hp.train.num_proc > 0:
            self.dataloader.data_queue.maxsize = hp.train.queue_size

        self.checkpoints = []
        self.model.weight_init()
        self.model.to(self.device)
        pimp = list(self.model.improver.parameters())# + list(self.model.improver2.parameters()) + \
#               list(self.model.improver3.parameters())
        self.optimizer_dis = torch.optim.Adam(pimp, lr=hp.train.learning_rate)
        self.optimizer_gen = torch.optim.Adam(self.model.generator.parameters(), lr=hp.train.learning_rate)
        self.step = 1
        if not hp.md.generator:
            init = torch.randn(hp.md.sample_base, hp.audio.time_step)
#            self.base = self.base / (self.base.norm(dim=1, keepdim=True) + 1e-20)
#            self.countbar = np.zeros([hp.md.sample_base])
            self.dist = MovingDistribution(init)
        if hp.train.load_checkpoint:
            self.load_for_train()
        else:
            print ("Training Model From Beginning!")

        if hp.path.output.summary:
            configure(hp.path.output.summary)

        if hp.md.generator:
            loss_names = ["Loss1",
                          "Loss2",
                          "Loss3"]
        else:
            loss_names = ["Loss1",
                          "Loss2"]
        buffers = buff(loss_names)
        pbar = tqdm(total=hp.train.steps, ascii=True, desc="Training Progress")
        if self.step != 1:
            pbar.update(self.step)
        
 #       ep = 1
 #       decay = 0.9999#85
        allstdr, allstdg = [], []
        try:
            while self.step <= hp.train.steps:
                try:
                    real_data = next(self.dataloader)
                except StopIteration:
                    self.dataloader = iter(self.dataiterable)
                    if hp.train.num_proc > 0:
                        self.dataloader.data_queue.maxsize = hp.train.queue_size
                    real_data = next(self.dataloader)
                if not hp.md.generator:
                    x, c = self.dist.sample(hp.train.batch_size)
                    x = x.to(self.device)
                else:
                    z = torch.randn(hp.train.batch_size, hp.md.zdims, device=self.device)
                    x = self.model.generator(z)
                real_data = real_data.to(self.device)
                go = self.model.improver(x.detach())
                io = self.model.improver(real_data)
#                io2 = self.model.improver2(real_data)
#                io3 = self.model.improver3(real_data + torch.randn_like(real_data))
#                v = io - real_data
#                io2 = real_data + random.random() * 10 * v
#                io2 = self.model.improver(io2.detach())
#                alpha = ((io ** 2).mean().detach() / (go ** 2).mean().detach()) ** 0.5
                loss1 = self.criterion(io, real_data * 2)
#                if self.step < 500:
#                    loss2 = self.criterion(go, 0) * 0
#                else:
                loss2 = self.criterion(go, 0)
#                loss2 = self.criterion(go, 0)
#                diff1 = goo - go
#                diff2 = io.detach() - real_data
#                diff = go - x.detach()
#                gooo = self.model.improver((go + diff).detach())
#                diff = diff / diff.std() * real_data.std()
#                diff = diff.detach()
#                est = torch.mean(self.angle(io - real_data, real_data) ** 2)
#                allstdr.append(((io ** 2).mean() ** 0.5).item())
#                allstdg.append(((go ** 2).mean() ** 0.5).item())
#                print (allstdg[-1], allstdr[-1])
                losses = loss1 + loss2# + est 
                self.optimizer_dis.zero_grad()
                losses.backward()
                self.optimizer_dis.step()
                loss1 = self.criterion(io, real_data)
                loss1 = loss1.item()
                loss2 = loss2.item()
                if not hp.md.generator:
                    self.dist.replace(c, go.detach().cpu())
                else:
#                    z = torch.randn(hp.train.batch_size, hp.md.zdims, device=self.device)
#                    x = self.model.generator(z)
                    go = self.model.improver(x)
                    loss3 = torch.mean((x - go.detach()) ** 2)
                    self.optimizer_gen.zero_grad()
                    loss3.backward()
                    self.optimizer_gen.step()
                    loss3 = loss3.item()
#                for p in self.model.generator.parameters():
#                    if p.grad is not None:
#                        p.grad.data.add_(torch.randn_like(p.grad.data) * self.optimizer_gen.state[p]['exp_avg_sq'] ** 0.5 * 2)

                pbar.set_postfix(s=np.std(go.detach().cpu().numpy()),
                                 std2=np.std(io.detach().cpu().numpy()),
                                 std3=np.std(x.detach().cpu().numpy()))
                if hp.path.output.summary:
                    try: 
                        log_value('l1', loss1)
                        log_value('l2', loss2)
                        if hp.md.generator:
                            log_value('l3', loss3)
                    except:
                        tqdm.write("Failed adding summary, maybe no space left...")
                buffers.put([loss1,
                             loss2],
                             [0, 1])
                if hp.md.generator:
                    buffers.put([loss3],
                                 [2])

                if self.step % hp.logging.step == 0:
                    tqdm.write(buffers.getstring([self.step, hp.train.steps]))
                if self.step % hp.logging.savestep == 0 and hp.path.output.save_dir:
                    try :
                        self.save()
                    except:
                        tqdm.write("Failed saving model, maybe no space left...")
                        traceback.print_exc()
                if self.step % hp.logging.samplestep == 0 and hp.path.output.sample:
                    self.sample()

                pbar.update(1)
                self.step += 1
                if self.step > hp.train.steps:
                    break
        except KeyboardInterrupt:
            print ("KeyboardInterrupt")
        except:
            traceback.print_exc()
        pbar.close()
#        np.save('../sstdr_1.npy', np.array(allstdr))
#        np.save('../sstdg_1.npy', np.array(allstdg))

    def infer(self):
        pass

    def save(self):
        if len(self.checkpoints) == hp.logging.save_maxtokeep:
            to_remove = os.path.join(hp.path.output.save_dir, self.checkpoints[0])
            tqdm.write("Remove old checkpoint %r from disk" %to_remove)
            os.remove(to_remove)
            self.checkpoints = self.checkpoints[1:]
        name = 'Step_' + str(self.step) + '.pt'
        saving_path = os.path.join(hp.path.output.save_dir, name)
        if hp.md.generator:
            self.save_config = {
                                   'step': self.step,
                                   'generator_state_dict': self.model.generator.state_dict(),
                                   'improver_state_dict': self.model.improver.state_dict(),
                                   'optimizer_state_dict_dis': self.optimizer_dis.state_dict(),
                                   'optimizer_state_dict_gen': self.optimizer_gen.state_dict()
                               }
        else:
            self.save_config = {
                                   'step': self.step,
                                   'base': self.dist.buffer,
                                   'improver_state_dict': self.model.improver.state_dict(),
                                   'optimizer_state_dict_dis': self.optimizer_dis.state_dict()
                               }
        tqdm.write("Saving model to %r" %saving_path)
        torch.save(self.save_config, saving_path)
        self.checkpoints.append(name)

    def load_for_train(self):
        assert os.path.isfile(hp.path.input.load_path), "Checkpoint %r does not exists" %hp.path.input.load_path
        print ("Loading Model From %r" %hp.path.input.load_path)
        checkpoint = torch.load(hp.path.input.load_path)
        self.model.improver.load_state_dict(checkpoint['improver_state_dict'])
        opt_dict_dis = checkpoint['optimizer_state_dict_dis']
        self.optimizer_dis.load_state_dict(opt_dict_dis)
        self.step = checkpoint['step']
        if hp.md.generator:
            self.model.generator.load_state_dict(checkpoint['generator_state_dict'])
            opt_dict_gen = checkpoint['optimizer_state_dict_gen']
            self.optimizer_gen.load_state_dict(opt_dict_gen)
        else:
            self.dist.buffer = checkpoint['base']

    def load_for_infer(self):
        pass

    def sample(self):
        try:
            if not hp.md.generator:
                samples, c = self.dist.sample(hp.logging.sample_num)
                samples = samples.numpy()
            else:
                z = torch.randn(hp.logging.sample_num, hp.md.zdims, device=self.device)
#                samples = []
#                for i in range(hp.logging.sample_num):
#                    real_data = next(self.dataloader).to(self.device)
#                    io = self.model.improver(real_data)
#                    v = io - real_data
#                    samples += [(real_data + 4 * v).detach().cpu().numpy()[0]]
                samples = self.model.generator(z).detach()
                samples = samples.cpu().numpy()
            for i in range(hp.logging.sample_num):
                tqdm.write(str(np.mean(samples[i])) + ' ' + str(np.std(samples[i])))
                name = 'Step_%r-%r.wav' %(self.step, i+1)
                path = os.path.join(hp.path.output.sample, name)
                tqdm.write('Sampling to %r ...' %path)
                writewav(path, samples[i])
        except KeyboardInterrupt:
            print ("KeyboardInterrupt")
        except:
            traceback.print_exc()
