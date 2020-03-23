import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import torch
from hparams import hp
import numpy as np

class Improver(nn.Module):
    def __init__(self):
        super(Improver, self).__init__()
        pv = hp.dataset.nchannel
        for i, c in enumerate(hp.md.channels):
            self.__setattr__(
                                'conv_layer'+str(i+1),
                                nn.Conv2d(pv, c, hp.md.filters[i], stride=hp.md.strides[i], padding=0, bias=False)
                            )
#            self.__setattr__(
#                                'conv_layer2'+str(i+1),
#                                nn.Conv2d(c, c, 3, stride=1, padding=1, bias=False)
#                            )
#            self.__setattr__(
#                                'conv_layer3'+str(i+1),
#                                nn.Conv2d(c, c, 3, stride=1, padding=1, bias=False)
#                            )
            pv = c
        self.final = nn.Linear(1024, 1, bias=False)

    def weight_init(self):
       for m in self.modules():
           if isinstance(m, nn.Conv2d):
               nfan_out = nn.init._calculate_correct_fan(m.weight, mode='fan_out') * 0.5
               gain = nn.init.calculate_gain('relu', 0)
               std = gain / nfan_out ** 0.5
               with torch.no_grad():
                   m.weight.normal_(0, std)
           nn.init.normal_(self.final.weight)

    def forward(self, inputs):
        inputs = inputs.requires_grad_()
        x = inputs
        for i in range(len(hp.md.channels)):
            x = self.__getattr__('conv_layer'+str(i+1))(x)
            x = F.relu(x)
#            x = self.__getattr__('conv_layer2'+str(i+1))(x)
#            x = F.relu(x)
#            x = self.__getattr__('conv_layer3'+str(i+1))(x)
#            x = F.relu(x)
        x = self.final(x.view(x.size(0), -1))
        x = autograd.grad(outputs=x,
                          inputs=inputs,
                          grad_outputs=torch.ones_like(x),
                          create_graph=True,
                          retain_graph=True,
                          only_inputs=True)[0]
        return x

    def loginfo(self, detail=False):
        ni = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if detail:
            print (self)
        else:
            print ("Number of Parameters for Improver: %r" %(ni))

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.channels = [512, 512, 256, 256, 128, 3]
        self.noisedim = 256
        filters = [3, 5, 3, 3, 5, 3]
        strides = [2, 4, 2, 2, 4, 2]
        self.strides = strides
        pv = 1024
        self.first = nn.ConvTranspose2d(self.noisedim, pv, 4)
        for i, c in enumerate(self.channels):
            self.__setattr__(
                                'conv_layer'+str(i+1),
                                nn.ConvTranspose2d(pv, c, filters[i], stride=strides[i], padding=0)
                            )
#            self.__setattr__(
#                                'conv_layer2'+str(i+1),
#                                nn.Conv2d(c, c, 3, stride=1, padding=1)
#                            )
#            self.__setattr__(
#                                'conv_layer3'+str(i+1),
#                                nn.Conv2d(c, c, 3, stride=1, padding=1)
#                            )
            pv = c
#        self.final2 = nn.Conv2d(pv, 3, 3, stride=1, padding=1, bias=False)

    def weight_init(self):
       for m in self.modules():
           if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
               nfan_out = nn.init._calculate_correct_fan(m.weight, mode='fan_in') * 2
               gain = nn.init.calculate_gain('relu', 0)
               std = gain / nfan_out ** 0.5
               with torch.no_grad():
                   m.weight.normal_(0, std)
                   m.bias.normal_(0, std)
#       with torch.no_grad():
#          self.final2.weight.normal_(0, 0.02)

    def forward(self, noise):
        x = F.relu(self.first(noise))
        for i in range(len(self.channels)):
#            x = F.interpolate(x, scale_factor=self.strides[i])
            x = self.__getattr__('conv_layer'+str(i+1))(x)
#            x = F.relu(x)
#            x = self.__getattr__('conv_layer2'+str(i+1))(x)
#            x = F.relu(x)
#            x = self.__getattr__('conv_layer3'+str(i+1))(x)
            if i != len(self.channels) - 1:
                x = F.relu(x)
        x = x[:, :, 100:100+hp.image.size, 100:100+hp.image.size]
        return x

    def loginfo(self, detail=False):
        ni = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if detail:
            print (self)
        else:
            print ("Number of Parameters for Generator: %r" %(ni))

