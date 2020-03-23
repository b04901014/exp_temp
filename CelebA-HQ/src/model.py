import os
import traceback
import itertools
import torch
from tensorboard_logger import configure, log_value
from tqdm import tqdm
from hparams import hp
from utils import *
from module import *
import threading

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class ImageGAN():
    def __init__(self):
        pass

    def thread_data(self):
        if hp.train.use_myreader:
            self.real_data = self.dataloader.dequeue()
        else:
            try:
                real_data = next(self.dataloader)
                self.real_data = real_data.to(self.device)
            except StopIteration:
                self.dataloader = iter(self.dataiterable)
                if hp.train.num_proc > 0:
                    self.dataloader.data_queue.maxsize = hp.train.queue_size
                real_data = next(self.dataloader)
                self.real_data = real_data.to(self.device)
        if not hp.train.use_generator:
            x, self.c = self.dist.sample(hp.train.batch_size)
            self.x = x.to(self.device)

    def thread_post(self, go, io):
        self.pbar.set_postfix(s=np.std(go.detach().cpu().numpy()),
                         std2=np.std(self.x.detach().cpu().numpy()),
                         std3=np.std(io.detach().cpu().numpy()))
        if hp.path.output.summary:
            try: 
                log_value('l1', self.loss1)
                log_value('l2', self.loss2)
                log_value('l3', self.loss3)
            except:
                tqdm.write("Failed adding summary, maybe no space left...")
        self.buffers.put([self.loss1,
                     self.loss2,
                     self.loss3,
                     self.loss4],
                     [0, 1, 2, 3])


    def train(self, rank):
        self.gpu = rank
        self.device = torch.device("cuda:{}".format(rank))
        dist.init_process_group(backend="nccl", rank=rank, world_size=torch.cuda.device_count())
        self.model = Improver()
        self.model.loginfo()
        self.criterion = lambda x, y: torch.mean((x - y) ** 2)
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(hp.dataset.data,
                                                                             num_replicas=torch.cuda.device_count(),
                                                                             rank=rank)
        params = {'batch_size': hp.train.batch_size,
#                  'shuffle': (self.train_sampler is None),
                  'pin_memory': True,
                  'num_workers': hp.train.num_proc,
                  'sampler': self.train_sampler}
        if hp.train.use_myreader:
            self.dataloader = MyReader() 
            self.procs = self.dataloader.start_enqueue()
        else:
            self.dataiterable = data.DataLoader(hp.dataset.data, **params)
            self.dataloader = iter(self.dataiterable)
            if hp.train.num_proc > 0:
                self.dataloader.data_queue.maxsize = hp.train.queue_size

        self.checkpoints = []
        self.model.weight_init()
        print ("Using ", torch.cuda.device_count(), "GPUs")
        self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[rank])
        self.step = 1
        if hp.train.use_generator:
            self.generator = Generator()
            self.generator.loginfo()
            self.generator.weight_init()
            self.generator.to(self.device)
            self.generator = DDP(self.generator, device_ids=[rank])
        else:
            init = torch.randn(hp.md.sample_base, hp.dataset.nchannel, hp.image.size, hp.image.size)
            self.dist = MovingDistribution(init)
        params = self.model.parameters()
        if hp.train.use_generator:
            params = list(params) + list(self.generator.parameters())
        self.optimizer = torch.optim.Adam(params, lr=hp.train.learning_rate)
        if hp.train.load_checkpoint:
            self.load_for_train()
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
        self.sample() 
        try:
#            fakemean = 0
#            alpha = 0.95
#            for i in tqdm(range(100)):
#                t = threading.Thread(target=self.thread_data, args=(), daemon=True)
#                t.start()
#                t.join()
#                noise = torch.randn(hp.train.batch_size, self.generator.module.noisedim, 1, 1, device=self.device)
#                mean += self.real_data.detach().cpu()
#            mean = (mean / 100 + 1) / 2
#            torchvision.utils.save_image(mean, '../samples/gg.png', nrow=1, normalize=False)
            t = threading.Thread(target=self.thread_data, args=(), daemon=True)
            t.start()
            t.join()
#            samples = self.model(self.real_data).detach().cpu()
#            samples += torch.FloatTensor(hp.datamean)
#            samples = (samples + 1) / 2.
#            torchvision.utils.save_image(samples, '../samples/gg.png', nrow=1, normalize=False)
            update_step = 8
            while self.step <= hp.train.steps:
                t = threading.Thread(target=self.thread_data, args=(), daemon=True)
                t.start()
                t.join()
                if hp.train.use_generator:
                    noise = torch.randn(hp.train.batch_size, self.generator.module.noisedim, 1, 1, device=self.device)
                    self.x = self.generator(noise)
                go = self.model(self.x.detach())
                io = self.model(self.real_data)
                loss1 = self.criterion(io, self.real_data * 2)
#                if self.step < hp.train.pre_step:
#                    loss2 = self.criterion(self.x, 0)
#                    loss3 = 0
#                else:
                loss2 = self.criterion(go, 0)
                loss3 = self.criterion(self.x, go.detach())
                (loss1 + loss2 + loss3).backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.loss1 = loss1.item()
                self.loss2 = loss2.item()
                self.loss4 = self.criterion(io, self.real_data).item()
                if self.step >= hp.train.pre_step:
                    if hp.train.use_generator:
                        self.loss3 = loss3.item()
                    else:
                        t = threading.Thread(target=self.dist.replace, args=(self.c, go.detach().cpu()), daemon=True)
                        t.start()
                if self.gpu == 0:
                    t = threading.Thread(target=self.thread_post, args=(go, io), daemon=True)
                    t.start()
                self.pbar.update(1)
                self.step += 1
                if self.step % hp.logging.savestep == 0 and hp.path.output.save_dir and self.gpu == 0:
                    try :
                        self.save()
                    except:
                        tqdm.write("Failed saving model, maybe no space left...")
                        traceback.print_exc()
                if self.step % hp.logging.samplestep == 0 and hp.path.output.sample and self.gpu == 0:
                    self.sample()
                if self.step % hp.logging.step == 0 and self.gpu == 0:
                    tqdm.write(self.buffers.getstring([self.step, hp.train.steps]))
                if self.step > hp.train.steps:
                    break
        except KeyboardInterrupt:
            print ("KeyboardInterrupt")
        except:
            traceback.print_exc()
        finally:
            dist.destroy_process_group()
            if hp.train.use_myreader:
                for x in self.procs:
                    x.terminate()
            self.pbar.close()

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
        if hp.train.use_generator:
            self.save_config = {
                                   'step': self.step,
                                   'generator_state_dict': self.generator.module.state_dict(),
                                   'improver_state_dict': self.model.module.state_dict(),
                                   'optimizer_state_dict': self.optimizer.state_dict()
                               }
        else:
            self.save_config = {
                                   'step': self.step,
                                   'base': self.dist.buffer,
                                   'improver_state_dict': self.model.module.state_dict(),
                                   'optimizer_state_dict': self.optimizer.state_dict()
                               }
        tqdm.write("Saving model to %r" %saving_path)
        torch.save(self.save_config, saving_path)
        self.checkpoints.append(name)

    def load_for_train(self):
        loc = 'cuda:{}'.format(self.gpu)
        assert os.path.isfile(hp.path.input.load_path), "Checkpoint %r does not exists" %hp.path.input.load_path
        print ("Loading Model From %r" %hp.path.input.load_path)
        checkpoint = torch.load(hp.path.input.load_path, map_location=loc)
        self.model.module.load_state_dict(checkpoint['improver_state_dict'])
        opt_dict = checkpoint['optimizer_state_dict']
        self.optimizer.load_state_dict(opt_dict)
        self.step = checkpoint['step']
        if hp.train.use_generator:
            self.generator.module.load_state_dict(checkpoint['generator_state_dict'])
        else:
            self.dist.buffer = checkpoint['base']
        dist.barrier()

    def load_for_infer(self):
        pass

    def sample(self):
        try:
#            c = np.random.choice(hp.md.sample_base, hp.logging.sample_num)
#            samples = self.base[c]
            if hp.train.use_generator:
                noise = torch.randn(hp.logging.sample_num, self.generator.module.noisedim, 1, 1, device=self.device)
                samples = self.generator(noise).detach().cpu()
            else:
                samples, c = self.dist.sample(hp.logging.sample_num)
            samples += torch.FloatTensor(hp.datamean)
            samples = (samples + 1) / 2.
            name = 'Step_%r.png' %self.step
            path = os.path.join(hp.path.output.sample, name)
            tqdm.write('Sampling to %r ...' %path)
            torchvision.utils.save_image(samples, path, nrow=int(hp.logging.sample_num**0.5), normalize=False)
        except KeyboardInterrupt:
            print ("KeyboardInterrupt")
        except:
            traceback.print_exc()
            if hp.train.use_myreader:
                for x in self.procs:
                    x.terminate()
