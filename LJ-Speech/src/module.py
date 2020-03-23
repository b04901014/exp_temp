import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import torch
from hparams import hp
import numpy as np

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        t, c = hp.md.gen.init_timestep, hp.md.gen.init_channel
        self.linear = nn.Linear(hp.md.zdims, t * c, bias=True)
        pv = hp.md.gen.init_channel
        for i, c in enumerate(hp.md.gen.channels):
            self.__setattr__(
                                'conv_layer'+str(i+1),
                                nn.ConvTranspose1d(pv, c, hp.md.gen.filters[i], stride=hp.md.gen.strides[i], padding=hp.md.gen.padding[i], bias=True)
                            )
            pv = c
        self.final = nn.Conv1d(pv, 1, 1)

    def weight_init(self):
       for m in self.modules():
           if isinstance(m, nn.Conv1d):
               nn.init.zeros_(m.bias)
           if isinstance(m, nn.ConvTranspose1d):
               nfan_out = nn.init._calculate_correct_fan(m.weight, mode='fan_in') / 2# / m.stride[0]
               gain = nn.init.calculate_gain('relu', 0)
               std = gain / nfan_out ** 0.5
               with torch.no_grad():
                   m.weight.normal_(0, std)
           if isinstance(m, nn.Linear):
               nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
           nn.init.kaiming_normal_(self.final.weight, nonlinearity='conv1d')

    def forward(self, z):
        t, c = hp.md.gen.init_timestep, hp.md.gen.init_channel
        x = self.linear(z)
        x = F.relu(x)
        x = x.view(x.size(0), c, t)
        for i in range(len(hp.md.gen.channels)):
            x = self.__getattr__('conv_layer'+str(i+1))(x)
            x = F.relu(x)
        x = self.final(x)
        return x.squeeze(1)

class Improver(nn.Module):
    def __init__(self):
        super(Improver, self).__init__()
        pv = 1
        for i, c in enumerate(hp.md.dis.channels):
            self.__setattr__(
                                'conv_layer'+str(i+1),
                                nn.Conv1d(pv, c, hp.md.dis.filters[i], stride=hp.md.dis.strides[i], padding=hp.md.dis.padding[i], bias=True)
                            )
            pv = c
        self.final_middle = nn.Linear(pv*2, 256, bias=False)
        self.final_middle2 = nn.Linear(256, 64, bias=False)
        self.final = nn.Linear(64, 1, bias=False)

    def weight_init(self):
       for m in self.modules():
           if isinstance(m, nn.Conv1d):
               nfan_out = nn.init._calculate_correct_fan(m.weight, mode='fan_out')# * 2# / m.stride[0]
               gain = nn.init.calculate_gain('relu', 0)
               std = gain / nfan_out ** 0.5
               with torch.no_grad():
                   m.weight.normal_(0, std)
               nn.init.normal_(m.bias, std=0.0)
           nn.init.normal_(self.final.weight)

    def forward(self, inputs):
        inputs = inputs.requires_grad_()
        x = inputs.unsqueeze(1)
        for i in range(len(hp.md.dis.channels)):
            x = self.__getattr__('conv_layer'+str(i+1))(x)
            x = F.relu(x)
        x = F.relu(self.final_middle(x.view(x.size(0), -1)))
        x = F.relu(self.final_middle2(x))
        x = self.final(x)
        x = autograd.grad(outputs=x,
                          inputs=inputs,
                          grad_outputs=torch.ones_like(x),
                          create_graph=True,
                          retain_graph=True,
                          allow_unused=True,
                          only_inputs=True)[0]
        return x

class Improver2(nn.Module):
    def __init__(self):
        super(Improver2, self).__init__()
        pv = 1
        self.cs1 = np.array([64, 128, 256, 512, 1024])
        self.cs2 = np.array([512, 256, 128, 64])
        self.cs2 = np.concatenate([self.cs2, [1]], axis=0)
        self.z = nn.Parameter(torch.randn(1024, 128))
        for i, c in enumerate(self.cs1):
            self.__setattr__(
                                'conv_layer'+str(i+1),
                                nn.Conv1d(pv, c, 6, stride=4, padding=1, bias=False)
                            )
            self.__setattr__(
                                'norm_layer'+str(i+1),
                                nn.InstanceNorm1d(c)
                            )
            pv = c
        for i, c in enumerate(self.cs2):
            self.__setattr__(
                                'deconv_layer'+str(i+1),
                                nn.ConvTranspose1d(pv, c, 6, stride=4, padding=1, bias=False)
                            )
            pv = c

    def weight_init(self):
       for m in self.modules():
           if isinstance(m, nn.Conv1d):
               nfan_out = nn.init._calculate_correct_fan(m.weight, mode='fan_out')# * 2# / m.stride[0]
               gain = nn.init.calculate_gain('relu', 0)
               std = gain / nfan_out ** 0.5
               with torch.no_grad():
                   m.weight.normal_(0, std)
#               nn.init.normal_(m.bias, std=0.05)
           if isinstance(m, nn.ConvTranspose1d):
               nfan_out = nn.init._calculate_correct_fan(m.weight, mode='fan_in') / 2# / m.stride[0]
               gain = nn.init.calculate_gain('relu', 0)
               std = gain / nfan_out ** 0.5
               with torch.no_grad():
                   m.weight.normal_(0, std)
#               nn.init.normal_(m.bias, std=0.05)

    def forward(self, inputs):
        inputs = inputs.requires_grad_()
        x = inputs.unsqueeze(1)
        for i in range(len(self.cs1)):
            x = self.__getattr__('conv_layer'+str(i+1))(x)
            if i != len(self.cs1) - 1: 
                x = F.relu(x)# + torch.randn_like(x) * 0.05
#        x = x * torch.randn_like(x) 
#        print (x.size())
#        x = x.view(1, 1024, 64, 2)
#        x = torch.flip(x, [3])
#        x = x.view(1, 1024, 128)
#        x = x - x.mean()
#        self.norm.track_running_stats = gg
#        x = self.norm(x)
#        x = x / (x ** 2).mean()
#        z = x
#        x = x.detach()
        x = (x > 0).float().detach() * self.z.expand(x.size())
#        print (x.size())
#        print (x.mean(), x.std())
#        x = torch.randn_like(x) * x.std() + x.mean()
#        print ((x ** 2).mean().item())
#        print (x.mean(), x.std())
        for i in range(len(self.cs2)):
            x = self.__getattr__('deconv_layer'+str(i+1))(x)
            if i != len(self.cs2) - 1: 
                x = F.relu(x)
#        z = autograd.grad(outputs=x,
#                          inputs=inputs,
#                          grad_outputs=torch.ones_like(x),
#                          create_graph=True,
#                          retain_graph=True,
#                          allow_unused=True,
#                          only_inputs=True)[0]
#        x = torch.flip(x, [1])
#        x = z
        return x.squeeze(1)#, z


class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        if hp.md.generator:
            self.generator = Generator()
        self.improver = Improver2()

    def weight_init(self):
        if hp.md.generator:
            self.generator.weight_init()
        self.improver.weight_init()

    def loginfo(self, detail=False):
        if hp.md.generator:
            ng = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        ni = sum(p.numel() for p in self.improver.parameters() if p.requires_grad)
        if detail:
            print (self.improver)
            if hp.md.generator:
                print (self.generator)
        else:
            print ("Number of Parameters for Improver: %r" %(ni))
            if hp.md.generator:
                print ("Number of Parameters for Generator: %r" %(ng))
                print ("Total Number of Parameters: %r" %(ni + ng))

