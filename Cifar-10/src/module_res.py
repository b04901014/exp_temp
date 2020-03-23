import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import torch
from hparams import hp
import numpy as np
from ops import *

class ResBlock_imp(nn.Module):
    def __init__(self, inc, outc,
                 downsamp=False, first=False):
        super(ResBlock_imp, self).__init__()
        self.first = first
        self.downsamp = downsamp
        if downsamp:
            self.downsample = nn.AvgPool2d(2)
        self.conv1 = MyConv2d(inc, outc, kernel_size=3, padding=1, bias=False)
        self.conv2 = MyConv2d(outc, outc, kernel_size=3, padding=1, bias=False)
        if hp.train.spnorm:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
            self.conv2 = nn.utils.spectral_norm(self.conv2)
        self.lsc = (inc != outc) or downsamp
        if self.lsc:
            self.convs = MyConv2d(inc, outc, kernel_size=1, padding=0, bias=False)
            if hp.train.spnorm:
                self.convs = nn.utils.spectral_norm(self.convs)
        self.func = MyAct.apply
    
    def forward(self, inputs):
        x = inputs
        if not self.first:
            x = self.func(x)
        x = self.conv1(x)
        x = self.func(x)
        x = self.conv2(x)
        if self.downsamp:
            x = self.downsample(x)
        if self.lsc:
            s = self.convs(inputs)
            if self.downsamp:
                s = self.downsample(s)
        else:
            s = inputs
        return x + s


class ResBlock_gen(nn.Module):
    def __init__(self, inc, outc, upsample=True):
        super(ResBlock_gen, self).__init__()
        self.bn1 = nn.BatchNorm2d(inc)
        self.bn2 = nn.BatchNorm2d(outc)
        self.conv1 = nn.Conv2d(inc, outc, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outc, outc, kernel_size=3, padding=1)
        self.convs = nn.Conv2d(inc, outc, kernel_size=1, padding=0)
        self.upsample = upsample
    
    def forward(self, inputs):
        x = F.relu(self.bn1(inputs))
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        x = self.conv1(x)
        x = F.relu(self.bn2(x))
        x = self.conv2(x)
        s = inputs
        if self.upsample:
            s = F.interpolate(s, scale_factor=2)
            s = self.convs(s)
        return x + s

class Improver_res(nn.Module):
    def __init__(self):
        super(Improver_res, self).__init__()
        self.dim = dim = 64
        self.r1 = ResBlock_imp(3, dim, downsamp=True, first=True)
        self.r2 = ResBlock_imp(dim, dim, downsamp=True)
        for i in range(2):
            self.__setattr__(
                                'res'+str(i+1),
                                ResBlock_imp(dim, dim)
                            )
        self.fc = nn.Linear(dim, 1, bias=False)
        if hp.train.spnorm:
            self.fc = nn.utils.spectral_norm(self.fc)
        self.func = MyAct.apply

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                std = 0.02
                with torch.no_grad():
                    m.weight.normal_(0, std)

    def forward(self, inputs):
        inputs = inputs.requires_grad_()
        x = inputs
        x = self.r1(x)
        x = self.r2(x)
        for i in range(2):
            x = self.__getattr__('res'+str(i+1))(x)
        x = self.func(x)
        x = x.sum((2, 3))
        x = self.fc(x.view(x.size(0), self.dim))
        raw = x
        x = autograd.grad(outputs=x,
                          inputs=inputs,
                          grad_outputs=torch.ones_like(x),
                          create_graph=True,
                          retain_graph=True,
                          only_inputs=True)[0]
        return x, torch.sum(x * inputs, (1, 2, 3)) 
    
    def loginfo(self, detail=False):
        ni = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if detail:
            print (self)
        else:
            print ("Number of Parameters for Improver: %r" %(ni))

class Generator_res(nn.Module):
    def __init__(self):
        super(Generator_res, self).__init__()
        self.dim = dim = 64
        self.fc = nn.Linear(128, 4 * 4 * dim)
        self.r1 = ResBlock_gen(dim, dim)
        self.r2 = ResBlock_gen(dim, dim)
        self.r3 = ResBlock_gen(dim, dim)
        self.conv = nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(dim)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                std = 0.02
                with torch.no_grad():
                    m.weight.normal_(0, std)
                    m.bias.fill_(0)
            if isinstance(m, nn.BatchNorm2d):
                std = 0.02
                with torch.no_grad():
                    m.weight.normal_(1.0, std)
                    m.bias.fill_(0)

    def forward(self, noise):
        x = self.fc(noise.view(noise.size(0), -1))
        x = x.view(-1, self.dim, 4, 4)
        x = self.r1(x)
        x = self.r2(x)
        x = self.r3(x)
        x = self.conv(F.relu(self.bn(x)))
        return torch.tanh(x)

    def loginfo(self, detail=False):
        ni = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if detail:
            print (self)
        else:
            print ("Number of Parameters for Generator: %r" %(ni))
