import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import torch
from hparams import hp
import numpy as np
from ops import *

class Improver(nn.Module):
    def __init__(self):
        super(Improver, self).__init__()
        pv = hp.dataset.nchannel
        sh = hp.image.size
        for i, c in enumerate(hp.md.imp.channels):
            self.__setattr__(
                                'conv_layer'+str(i+1),
                                MyConv2d(pv, c, hp.md.imp.filters[i], stride=hp.md.imp.strides[i],
                                          padding=hp.md.imp.pads[i], bias=hp.md.imp.with_bias)
                            )
            self.__setattr__(
                                'bn'+str(i+1),
                                MyFN(c)
                            )
            if hp.train.spnorm:
                self.__setattr__(
                                    'conv_layer'+str(i+1),
                                    nn.utils.spectral_norm(self.__getattr__('conv_layer'+str(i+1)))
                                )
            pv = c
            self.func = MyAct.apply

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                std = 0.02
                with torch.no_grad():
                    m.weight.normal_(0, std)
                    if hp.md.imp.with_bias:
                        m.bias.normal_(0, std)

    def forward(self, inputs):
        inputs = inputs.requires_grad_()
        x = inputs
        layers = []
        layers.append(x)
        for i in range(len(hp.md.imp.channels)):
            x = self.__getattr__('conv_layer'+str(i+1))(x)
            if i in [0, 1, 2, 3, 4, 5, 6]:
                x = self.func(x, self.__getattr__('bn'+str(i+1)))
        x = x.view(x.size(0), 1)
        if hp.train.GAN_type == 'NSGAN':
            raw = torch.sigmoid(x)
        else:
            raw = x
        x = autograd.grad(outputs=x,
                          inputs=layers,
                          grad_outputs=torch.ones_like(x),
                          create_graph=True,
                          only_inputs=True)[0]
        return x, torch.sum(x * inputs, (1, 2, 3))
    
    def loginfo(self, detail=False):
        ni = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if detail:
            print (self)
        else:
            print ("Number of Parameters for Improver: %r" %(ni))

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        pv = hp.md.gen.initc
        self.first = nn.ConvTranspose2d(hp.md.gen.noisedim, pv, hp.md.gen.inits, 1, 0)
        for i, c in enumerate(hp.md.gen.channels):
            self.__setattr__(
                                'conv_layer'+str(i+1),
                                nn.ConvTranspose2d(pv, c, hp.md.gen.filters[i],
                                                   stride=hp.md.gen.strides[i], padding=hp.md.gen.pads[i])
                            )
            if hp.train.bn_g:
                self.__setattr__(
                                    'bn'+str(i+1),
                                    nn.BatchNorm2d(c)
                                )
                
            pv = c

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                std = 0.02
                with torch.no_grad():
                    m.weight.normal_(0, std)
                    m.bias.fill_(0)

    def forward(self, noise):
        x = F.relu(self.first(noise))
        for i in range(len(hp.md.gen.channels)):
            x = self.__getattr__('conv_layer'+str(i+1))(x)
            if i != len(hp.md.gen.channels) - 1:
                if hp.train.bn_g:
                    x = self.__getattr__('bn'+str(i+1))(x)
                x = F.relu(x)
        return torch.tanh(x)

    def loginfo(self, detail=False):
        ni = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if detail:
            print (self)
        else:
            print ("Number of Parameters for Generator: %r" %(ni))


