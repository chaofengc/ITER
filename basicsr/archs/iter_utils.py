from turtle import forward
import torch
from torch.nn import functional as F
from torch import nn as nn


def class_balanced_xentropy(logits, labels, beta=0.9999, eps=1e-8):
    num_class = logits.shape[-1]
    samples_per_cls = torch.bincount(labels, minlength=num_class) 

    effective_num = 1.0 - torch.pow(beta, samples_per_cls)
    weights = (1.0 - beta) / (effective_num + eps) 
    weights = weights / (torch.sum(weights) + eps) * num_class 

    loss = F.cross_entropy(logits, labels, weight=weights)  
    return loss 


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = NormLayer(in_channels, 'gn')
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

class SimpleGate(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2 * in_channels, 1, 1, 0)
    
    def forward(self, x):
        x = self.conv(x)
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NormLayer(nn.Module):
    """Normalization Layers.
    ------------
    # Arguments
        - channels: input channels, for batch norm and instance norm.
        - input_size: input shape without batch size, for layer norm.
    """
    def __init__(self, channels, norm_type='bn'):
        super(NormLayer, self).__init__()
        norm_type = norm_type.lower()
        self.norm_type = norm_type
        self.channels = channels
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels, affine=True)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=False)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        elif norm_type == 'none':
            self.norm = lambda x: x*1.0
        else:
            assert 1==0, 'Norm type {} not support.'.format(norm_type)

    def forward(self, x):
        return self.norm(x)


class ActLayer(nn.Module):
    """activation layer.
    ------------
    # Arguments
        - relu type: type of relu layer, candidates are
            - ReLU
            - LeakyReLU: default relu slope 0.2
            - PRelu 
            - SELU
            - none: direct pass
    """
    def __init__(self, channels, relu_type='leakyrelu'):
        super(ActLayer, self).__init__()
        relu_type = relu_type.lower()
        if relu_type == 'relu':
            self.func = nn.ReLU(True)
        elif relu_type == 'leakyrelu':
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == 'prelu':
            self.func = nn.PReLU(channels)
        elif relu_type == 'none':
            self.func = lambda x: x*1.0
        elif relu_type == 'silu':
            self.func = nn.SiLU(True)
        elif relu_type == 'gelu':
            self.func = nn.GELU()
        elif relu_type == 'simplegate':
            self.func = SimpleGate(channels)
        else:
            assert 1==0, 'activation type {} not support.'.format(relu_type)

    def forward(self, x):
        return self.func(x)


class ResBlock(nn.Module):
    """
    Use preactivation version of residual block, the same as taming
    """
    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='leakyrelu'):
        super(ResBlock, self).__init__()

        self.conv = nn.Sequential(
            NormLayer(in_channel, norm_type),
            ActLayer(in_channel, act_type),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            NormLayer(out_channel, norm_type),
            ActLayer(out_channel, act_type),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
        )

    def forward(self, input):
        res = self.conv(input)
        out = res + input
        return out


class CombineQuantBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_ch1 + in_ch2, out_channel, 3, 1, 1)

    def forward(self, input1, input2=None):
        if input2 is not None:
            input2 = F.interpolate(input2, input1.shape[2:])
            input = torch.cat((input1, input2), dim=1)
        else:
            input = input1
        out = self.conv(input)
        return out


