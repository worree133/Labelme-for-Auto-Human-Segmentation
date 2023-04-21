import einops
import torch
import torch as th
import torch.nn as nn


from einops import rearrange, repeat
from torchvision.utils import make_grid
from .u2net import *

def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class ControlledUnetModel(U2NETP):

    def forward(self, x, control=None, only_mid_control=False, **kwargs):

        with torch.no_grad():
            hx = x
            # stage 1
            hx1 = self.stage1(hx)
            hx = self.pool12(hx1)

            # stage 2
            hx2 = self.stage2(hx)
            hx = self.pool23(hx2)

            # stage 3
            hx3 = self.stage3(hx)
            hx = self.pool34(hx3)

            # stage 4
            hx4 = self.stage4(hx)
            hx = self.pool45(hx4)

            # stage 5
            hx5 = self.stage5(hx)
            hx = self.pool56(hx5)

            # stage 6
            hx6 = self.stage6(hx)

        if control is not None:
            hx6 += control.pop()

        hx6up = _upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)





class ControlNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(ControlNet, self).__init__()

        self.zero_conv0 = self.make_zero_conv(in_ch)

        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.zero_conv1 = self.make_zero_conv(64)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.zero_conv2 = self.make_zero_conv(128)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.zero_conv3 = self.make_zero_conv(256)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.zero_conv4 = self.make_zero_conv(512)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.zero_conv5 = self.make_zero_conv(512)

        self.stage6 = RSU4F(512, 256, 512)
        self.zero_conv6 = self.make_zero_conv(512)


    def make_zero_conv(self, channels):
        return zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))



class ControlNetp(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, dims=2):
        super(ControlNetp, self).__init__()

        self.dims = dims
        self.zero_conv0 = self.make_zero_conv(in_ch)


        self.stage1 = RSU7(in_ch,16,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv1 = self.make_zero_conv(64)

        self.stage2 = RSU6(64,16,64)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv2 = self.make_zero_conv(64)

        self.stage3 = RSU5(64,16,64)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv3 = self.make_zero_conv(64)

        self.stage4 = RSU4(64,16,64)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv4 = self.make_zero_conv(64)

        self.stage5 = RSU4F(64,16,64)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv5 = self.make_zero_conv(64)

        self.stage6 = RSU4F(64,16,64)
        self.zero_conv6 = self.make_zero_conv(64)


    def make_zero_conv(self, channels):
        return zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))

    def forward(self, x, hint):


        hint = self.zero_conv0(hint)
        outs = []

        hx = hint + x
        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        outs.append(self.zero_conv1(hx))

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        outs.append(self.zero_conv2(hx))

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        outs.append(self.zero_conv3(hx))

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        outs.append(self.zero_conv4(hx))

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        outs.append(self.zero_conv5(hx))

        # stage 6
        hx6 = self.stage6(hx)
        outs.append(self.zero_conv6(hx6))

        return outs



class ControlU2Netpmid(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, dims=2):
        super(ControlU2Net, self).__init__()

        self.dims = dims
        self.zero_conv0 = self.make_zero_conv(in_ch)


        self.stage1 = RSU7(in_ch,16,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv1 = self.make_zero_conv(64)

        self.stage2 = RSU6(64,16,64)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv2 = self.make_zero_conv(64)

        self.stage3 = RSU5(64,16,64)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv3 = self.make_zero_conv(64)

        self.stage4 = RSU4(64,16,64)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv4 = self.make_zero_conv(64)

        self.stage5 = RSU4F(64,16,64)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv5 = self.make_zero_conv(64)

        self.stage6 = RSU4F(64,16,64)
        self.zero_conv6 = self.make_zero_conv(64)
        # decoder
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def make_zero_conv(self, channels):
        return zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))

    def forward(self, x):


        hint = self.zero_conv0(x)
        outs = []

        # stage 1
        hx1 = self.stage1(hint)
        hx = self.pool12(hx1)
        outs.append(self.zero_conv1(hx))

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        outs.append(self.zero_conv2(hx))

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        outs.append(self.zero_conv3(hx))

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        outs.append(self.zero_conv4(hx))

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        outs.append(self.zero_conv5(hx))

        # stage 6
        hx6 = self.stage6(hx)
        outs.append(self.zero_conv6(hx6))

        with torch.no_grad():
            hx = x
            # stage 1
            hx1 = self.stage1(hx)
            hx = self.pool12(hx1)

            # stage 2
            hx2 = self.stage2(hx)
            hx = self.pool23(hx2)

            # stage 3
            hx3 = self.stage3(hx)
            hx = self.pool34(hx3)

            # stage 4
            hx4 = self.stage4(hx)
            hx = self.pool45(hx4)

            # stage 5
            hx5 = self.stage5(hx)
            hx = self.pool56(hx5)

            # stage 6
            hx6 = self.stage6(hx)


        hx6 += outs.pop()
        hx6up = _upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)



class ControlU2Netp(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(ControlU2Netp, self).__init__()

        self.dims = 2
        self.zero_conv0 = self.make_zero_conv(in_ch)


        self.stage1 = RSU7(in_ch,16,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv1 = self.make_zero_conv(64)

        self.stage2 = RSU6(64,16,64)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv2 = self.make_zero_conv(64)

        self.stage3 = RSU5(64,16,64)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv3 = self.make_zero_conv(64)

        self.stage4 = RSU4(64,16,64)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv4 = self.make_zero_conv(64)

        self.stage5 = RSU4F(64,16,64)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv5 = self.make_zero_conv(64)

        self.stage6 = RSU4F(64,16,64)
        self.zero_conv6 = self.make_zero_conv(64)
        # decoder
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def make_zero_conv(self, channels):
        return zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))

    def forward(self, x):


        hint = self.zero_conv0(x)
        outs = []

        # stage 1
        chx1 = self.stage1(hint)
        outs.append(self.zero_conv1(chx1))
        chx = self.pool12(chx1)


        # stage 2
        chx2 = self.stage2(chx)
        outs.append(self.zero_conv2(chx2))
        chx = self.pool23(chx2)


        # stage 3
        chx3 = self.stage3(chx)
        outs.append(self.zero_conv3(chx3))
        chx = self.pool34(chx3)


        # stage 4
        chx4 = self.stage4(chx)
        outs.append(self.zero_conv4(chx4))
        chx = self.pool45(chx4)


        # stage 5
        chx5 = self.stage5(chx)
        outs.append(self.zero_conv5(chx5))
        chx = self.pool56(chx5)


        # stage 6
        chx6 = self.stage6(chx)
        outs.append(self.zero_conv6(chx6))

        with torch.no_grad():

            # stage 1
            hx1 = self.stage1(x)
            hx = self.pool12(hx1)

            # stage 2
            hx2 = self.stage2(hx)
            hx = self.pool23(hx2)

            # stage 3
            hx3 = self.stage3(hx)
            hx = self.pool34(hx3)

            # stage 4
            hx4 = self.stage4(hx)
            hx = self.pool45(hx4)

            # stage 5
            hx5 = self.stage5(hx)
            hx = self.pool56(hx5)

            # stage 6
            hx6 = self.stage6(hx)


        hx6 += outs.pop()

        hx6up = _upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5+outs.pop()), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4+outs.pop()), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3+outs.pop()), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2+outs.pop()), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1+outs.pop()), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)

class ControlU2Netpseg2(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(ControlU2Netpseg2, self).__init__()

        self.dims = 2
        self.zero_conv0 = self.make_zero_conv(in_ch)


        self.stage1 = RSU7(in_ch,16,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv1 = self.make_zero_conv(64)

        self.stage2 = RSU6(64,16,64)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv2 = self.make_zero_conv(64)

        self.stage3 = RSU5(64,16,64)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv3 = self.make_zero_conv(64)

        self.stage4 = RSU4(64,16,64)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv4 = self.make_zero_conv(64)

        self.stage5 = RSU4F(64,16,64)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv5 = self.make_zero_conv(64)

        self.stage6 = RSU4F(64,16,64)
        self.zero_conv6 = self.make_zero_conv(64)
        # decoder
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def make_zero_conv(self, channels):
        return zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))

    def forward(self, x, seg):


        hint = self.zero_conv0(seg)
        outs = []

        # stage 1
        chx1 = self.stage1(hint)
        outs.append(self.zero_conv1(chx1))
        chx = self.pool12(chx1)


        # stage 2
        chx2 = self.stage2(chx)
        outs.append(self.zero_conv2(chx2))
        chx = self.pool23(chx2)


        # stage 3
        chx3 = self.stage3(chx)
        outs.append(self.zero_conv3(chx3))
        chx = self.pool34(chx3)


        # stage 4
        chx4 = self.stage4(chx)
        outs.append(self.zero_conv4(chx4))
        chx = self.pool45(chx4)


        # stage 5
        chx5 = self.stage5(chx)
        outs.append(self.zero_conv5(chx5))
        chx = self.pool56(chx5)


        # stage 6
        chx6 = self.stage6(chx)
        outs.append(self.zero_conv6(chx6))

        with torch.no_grad():

            # stage 1
            hx1 = self.stage1(x)
            hx = self.pool12(hx1)

            # stage 2
            hx2 = self.stage2(hx)
            hx = self.pool23(hx2)

            # stage 3
            hx3 = self.stage3(hx)
            hx = self.pool34(hx3)

            # stage 4
            hx4 = self.stage4(hx)
            hx = self.pool45(hx4)

            # stage 5
            hx5 = self.stage5(hx)
            hx = self.pool56(hx5)

            # stage 6
            hx6 = self.stage6(hx)


        hx6 += outs.pop()

        hx6up = _upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5+outs.pop()), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4+outs.pop()), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3+outs.pop()), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2+outs.pop()), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1+outs.pop()), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)

class ControlU2Netpseg(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(ControlU2Netpseg, self).__init__()

        self.dims = 2


        # controlnet encoder
        self.zero_conv0 = self.make_zero_conv(in_ch)
        self.ctrl_stage1 = RSU7(in_ch,16,64)
        self.ctrl_pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv1 = self.make_zero_conv(64)

        self.ctrl_stage2 = RSU6(64,16,64)
        self.ctrl_pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv2 = self.make_zero_conv(64)

        self.ctrl_stage3 = RSU5(64,16,64)
        self.ctrl_pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv3 = self.make_zero_conv(64)

        self.ctrl_stage4 = RSU4(64,16,64)
        self.ctrl_pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv4 = self.make_zero_conv(64)

        self.ctrl_stage5 = RSU4F(64,16,64)
        self.ctrl_pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv5 = self.make_zero_conv(64)

        self.ctrl_stage6 = RSU4F(64,16,64)
        self.zero_conv6 = self.make_zero_conv(64)

        # u2netp encoder

        self.stage1 = RSU7(in_ch,16,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv1 = self.make_zero_conv(64)

        self.stage2 = RSU6(64,16,64)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv2 = self.make_zero_conv(64)

        self.stage3 = RSU5(64,16,64)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv3 = self.make_zero_conv(64)

        self.stage4 = RSU4(64,16,64)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv4 = self.make_zero_conv(64)

        self.stage5 = RSU4F(64,16,64)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv5 = self.make_zero_conv(64)

        self.stage6 = RSU4F(64,16,64)
        self.zero_conv6 = self.make_zero_conv(64)


        # decoder
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def make_zero_conv(self, channels):
        return zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))

    def forward(self, x, seg):


        hint = self.zero_conv0(seg)
        outs = []

        # stage 1
        chx1 = self.ctrl_stage1(hint)
        outs.append(self.zero_conv1(chx1))
        chx = self.ctrl_pool12(chx1)


        # stage 2
        chx2 = self.ctrl_stage2(chx)
        outs.append(self.zero_conv2(chx2))
        chx = self.ctrl_pool23(chx2)


        # stage 3
        chx3 = self.ctrl_stage3(chx)
        outs.append(self.zero_conv3(chx3))
        chx = self.ctrl_pool34(chx3)


        # stage 4
        chx4 = self.ctrl_stage4(chx)
        outs.append(self.zero_conv4(chx4))
        chx = self.ctrl_pool45(chx4)


        # stage 5
        chx5 = self.ctrl_stage5(chx)
        outs.append(self.zero_conv5(chx5))
        chx = self.ctrl_pool56(chx5)


        # stage 6
        chx6 = self.ctrl_stage6(chx)
        outs.append(self.zero_conv6(chx6))

        with torch.no_grad():

            # stage 1
            hx1 = self.stage1(x)
            hx = self.pool12(hx1)

            # stage 2
            hx2 = self.stage2(hx)
            hx = self.pool23(hx2)

            # stage 3
            hx3 = self.stage3(hx)
            hx = self.pool34(hx3)

            # stage 4
            hx4 = self.stage4(hx)
            hx = self.pool45(hx4)

            # stage 5
            hx5 = self.stage5(hx)
            hx = self.pool56(hx5)

            # stage 6
            hx6 = self.stage6(hx)


        hx6 += outs.pop()

        hx6up = _upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5+outs.pop()), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4+outs.pop()), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3+outs.pop()), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2+outs.pop()), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1+outs.pop()), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)


class DoubleControlU2Netpseg(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(DoubleControlU2Netpseg, self).__init__()

        self.dims = 2
        # controlnet encoder 1
        self.small_zero_conv0 = self.make_zero_conv(in_ch)
        self.small_ctrl_stage1 = RSU7(in_ch,16,64)
        self.small_ctrl_pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.small_zero_conv1 = self.make_zero_conv(64)

        self.small_ctrl_stage2 = RSU6(64,16,64)
        self.small_ctrl_pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.small_zero_conv2 = self.make_zero_conv(64)

        self.small_ctrl_stage3 = RSU5(64,16,64)
        self.small_ctrl_pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.small_zero_conv3 = self.make_zero_conv(64)

        # controlnet encoder 2
        self.zero_conv0 = self.make_zero_conv(in_ch)
        self.ctrl_stage1 = RSU7(in_ch,16,64)
        self.ctrl_pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv1 = self.make_zero_conv(64)

        self.ctrl_stage2 = RSU6(64,16,64)
        self.ctrl_pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv2 = self.make_zero_conv(64)

        self.ctrl_stage3 = RSU5(64,16,64)
        self.ctrl_pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv3 = self.make_zero_conv(64)

        self.ctrl_stage4 = RSU4(64,16,64)
        self.ctrl_pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv4 = self.make_zero_conv(64)

        self.ctrl_stage5 = RSU4F(64,16,64)
        self.ctrl_pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv5 = self.make_zero_conv(64)

        self.ctrl_stage6 = RSU4F(64,16,64)
        self.zero_conv6 = self.make_zero_conv(64)

        # u2netp encoder

        self.stage1 = RSU7(in_ch,16,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv1 = self.make_zero_conv(64)

        self.stage2 = RSU6(64,16,64)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv2 = self.make_zero_conv(64)

        self.stage3 = RSU5(64,16,64)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv3 = self.make_zero_conv(64)

        self.stage4 = RSU4(64,16,64)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv4 = self.make_zero_conv(64)

        self.stage5 = RSU4F(64,16,64)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv5 = self.make_zero_conv(64)

        self.stage6 = RSU4F(64,16,64)
        self.zero_conv6 = self.make_zero_conv(64)


        # decoder
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def make_zero_conv(self, channels):
        return zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))

    def forward(self, x, seg):

        small_int = self.small_zero_conv0(seg)
        small_outs = []

        # stage 1
        schx1 = self.small_ctrl_stage1(small_int)
        small_outs.append(self.small_zero_conv1(schx1))
        schx = self.small_ctrl_pool12(schx1)

        # stage 2
        schx2 = self.small_ctrl_stage2(schx)
        small_outs.append(self.small_zero_conv2(schx2))
        schx = self.small_ctrl_pool23(schx2)

        # stage 3
        schx3 = self.small_ctrl_stage3(schx)
        small_outs.append(self.small_zero_conv3(schx3))


        with torch.no_grad():

            hint = self.zero_conv0(seg)
            outs = []

            # stage 1
            chx1 = self.ctrl_stage1(hint)
            outs.append(self.zero_conv1(chx1))
            chx = self.ctrl_pool12(chx1)


            # stage 2
            chx2 = self.ctrl_stage2(chx)
            outs.append(self.zero_conv2(chx2))
            chx = self.ctrl_pool23(chx2)


            # stage 3
            chx3 = self.ctrl_stage3(chx)
            outs.append(self.zero_conv3(chx3))
            chx = self.ctrl_pool34(chx3)


            # stage 4
            chx4 = self.ctrl_stage4(chx)
            outs.append(self.zero_conv4(chx4))
            chx = self.ctrl_pool45(chx4)


            # stage 5
            chx5 = self.ctrl_stage5(chx)
            outs.append(self.zero_conv5(chx5))
            chx = self.ctrl_pool56(chx5)


            # stage 6
            chx6 = self.ctrl_stage6(chx)
            outs.append(self.zero_conv6(chx6))

            # stage 1
            hx1 = self.stage1(x)
            hx = self.pool12(hx1)

            # stage 2
            hx2 = self.stage2(hx)
            hx = self.pool23(hx2)

            # stage 3
            hx3 = self.stage3(hx)
            hx = self.pool34(hx3)

            # stage 4
            hx4 = self.stage4(hx)
            hx = self.pool45(hx4)

            # stage 5
            hx5 = self.stage5(hx)
            hx = self.pool56(hx5)

            # stage 6
            hx6 = self.stage6(hx)


        hx6 += outs.pop()

        hx6up = _upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5+outs.pop()), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4+outs.pop()), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3+outs.pop()+small_outs.pop()), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2+outs.pop()+small_outs.pop()), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1+outs.pop()+small_outs.pop()), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)


class DoubleControlU2Netpsegdepth(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, dims=2):
        super(DoubleControlU2Netpsegdepth, self).__init__()

        self.dims = dims

        self.input_hint_block = nn.Sequential(
            conv_nd(dims, 6, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 64, 3, padding=1),
            nn.SiLU(),
            zero_module(conv_nd(dims, 64, in_ch, 3, padding=1))
        )

        # controlnet encoder

        self.ctrl_stage1 = RSU7(in_ch,16,64)
        self.ctrl_pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv1 = self.make_zero_conv(64)

        self.ctrl_stage2 = RSU6(64,16,64)
        self.ctrl_pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv2 = self.make_zero_conv(64)

        self.ctrl_stage3 = RSU5(64,16,64)
        self.ctrl_pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv3 = self.make_zero_conv(64)

        self.ctrl_stage4 = RSU4(64,16,64)
        self.ctrl_pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv4 = self.make_zero_conv(64)

        self.ctrl_stage5 = RSU4F(64,16,64)
        self.ctrl_pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv5 = self.make_zero_conv(64)

        self.ctrl_stage6 = RSU4F(64,16,64)
        self.zero_conv6 = self.make_zero_conv(64)

        # u2netp encoder

        self.stage1 = RSU7(in_ch,16,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv1 = self.make_zero_conv(64)

        self.stage2 = RSU6(64,16,64)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv2 = self.make_zero_conv(64)

        self.stage3 = RSU5(64,16,64)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv3 = self.make_zero_conv(64)

        self.stage4 = RSU4(64,16,64)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv4 = self.make_zero_conv(64)

        self.stage5 = RSU4F(64,16,64)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.zero_conv5 = self.make_zero_conv(64)

        self.stage6 = RSU4F(64,16,64)
        self.zero_conv6 = self.make_zero_conv(64)


        # decoder
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def make_zero_conv(self, channels):
        return zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))

    def forward(self, x, seg, depth):

        hint = self.input_hint_block(torch.cat((seg,depth), 1))
        outs = []
        with torch.no_grad():
            # stage 1
            chx1 = self.ctrl_stage1(hint)
            outs.append(self.zero_conv1(chx1))
            chx = self.ctrl_pool12(chx1)


            # stage 2
            chx2 = self.ctrl_stage2(chx)
            outs.append(self.zero_conv2(chx2))
            chx = self.ctrl_pool23(chx2)


            # stage 3
            chx3 = self.ctrl_stage3(chx)
            outs.append(self.zero_conv3(chx3))
            chx = self.ctrl_pool34(chx3)


            # stage 4
            chx4 = self.ctrl_stage4(chx)
            outs.append(self.zero_conv4(chx4))
            chx = self.ctrl_pool45(chx4)


            # stage 5
            chx5 = self.ctrl_stage5(chx)
            outs.append(self.zero_conv5(chx5))
            chx = self.ctrl_pool56(chx5)


            # stage 6
            chx6 = self.ctrl_stage6(chx)
            outs.append(self.zero_conv6(chx6))


            # stage 1
            hx1 = self.stage1(x)
            hx = self.pool12(hx1)

            # stage 2
            hx2 = self.stage2(hx)
            hx = self.pool23(hx2)

            # stage 3
            hx3 = self.stage3(hx)
            hx = self.pool34(hx3)

            # stage 4
            hx4 = self.stage4(hx)
            hx = self.pool45(hx4)

            # stage 5
            hx5 = self.stage5(hx)
            hx = self.pool56(hx5)

            # stage 6
            hx6 = self.stage6(hx)


        hx6 += outs.pop()

        hx6up = _upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5+outs.pop()), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4+outs.pop()), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3+outs.pop()), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2+outs.pop()), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1+outs.pop()), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)
