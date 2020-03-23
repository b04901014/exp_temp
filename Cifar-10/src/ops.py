import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import autograd
from hparams import hp
import numpy as np

def _grad_input_padding(grad_output, input_size, stride, padding, kernel_size):
    input_size = list(input_size)
    k = grad_output.dim() - 2

    if len(input_size) == k + 2:
        input_size = input_size[-k:]
    if len(input_size) != k:
        raise ValueError("input_size must have {} elements (got {})"
                         .format(k + 2, len(input_size)))

    def dim_size(d):
        return ((grad_output.size(d + 2) - 1) * stride[d] - 2 * padding[d] +
                kernel_size[d])

    min_sizes = [dim_size(d) for d in range(k)]
    max_sizes = [min_sizes[d] + stride[d] - 1 for d in range(k)]
    for size, min_size, max_size in zip(input_size, min_sizes, max_sizes):
        if size < min_size or size > max_size:
            raise ValueError(
                ("requested an input grad size of {}, but valid sizes range "
                 "from {} to {} (for a grad_output of {})").format(
                     input_size, min_sizes, max_sizes,
                     grad_output.size()[2:]))

    return tuple(input_size[d] - min_sizes[d] for d in range(k))

class MyAct(autograd.Function):
    @staticmethod
    def forward(ctx, input, fn=None):
        mask = (input.detach() > 0).float()
        ctx.save_for_backward(input, mask)
        ctx.fn = fn
        return input * mask# + 0.1 * input * (1 - mask)

    @staticmethod
    def backward(ctx, grad_output):
        input, mask, = ctx.saved_tensors
#        print (mask.mean().item(), mask.std(0).mean(0).mean(0).mean(0).item())
        grad_input = grad_output#.clone()
#        grad_input = 0.5 * (grad_input + input)
        grad_input = grad_input * mask# + 0.1 * grad_input * (1 - mask)
#        grad_input = F.dropout(grad_input, p=0.2, training=True)
        return grad_input, None

class conv2d_gi(autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, s, p, d, g, bk):
        ctx.save_for_backward(w)
        ctx.i = x.size()
        ctx.b = b
        ctx.s = s
        ctx.p = p
        ctx.d = d
        ctx.g = g
        ctx.bk = bk
        return F.conv2d(x, w, b, s, p, d, g)

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        input_size = ctx.i
        bias = ctx.bk
        stride = ctx.s
        padding = ctx.p
        dilation = ctx.d
        kernel_size = (weight.shape[2], weight.shape[3])
        groups = ctx.g
        grad_input_padding = _grad_input_padding(grad_output, input_size, stride,
                                                 padding, kernel_size)
        grad_input = F.conv_transpose2d(grad_output, weight, None, stride,
                                        padding, grad_input_padding, groups, dilation)
        return tuple([grad_input] + [None] * 7)


class MyConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(MyConv2d, self).__init__(*args, **kwargs)
        self.op = conv2d_gi.apply
        self.bias_back = nn.Parameter(torch.zeros(self.in_channels))

    def forward(self, input):
        ret = self.op(input, self.weight, self.bias, self.stride,
                      self.padding, self.dilation, self.groups, self.bias_back)
        return ret

class MyFN(nn.Module):
    def __init__(self, c):
        super(MyFN, self).__init__()
        self.rate = 0.015
        self.a = nn.Parameter(torch.ones(1, c, 1, 1) * self.rate)

    def forward(self, g, i):
        self.alpha = self.a / self.rate * 0.5
        self.alpha = torch.clamp(self.alpha, 0, 1)
        return self.alpha * g + (1 - self.alpha) * i
 
