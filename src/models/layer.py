import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import *

from ..utils.params import get_params

# import astra

# import op_ct

class CNR2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, padding_mode='reflection', norm='bnorm', relu=0.0, drop=[], bias=[]):
        super().__init__()

        if bias == []:
            if norm == 'bnorm':
                bias = False
            else:
                bias = True

        layers = []
        layers += [Padding(padding=padding, padding_mode=padding_mode)]
        layers += [Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=0, bias=bias)]

        if norm != []:
            layers += [Norm2d(nch_out, norm)]

        if relu != []:
            layers += [ReLU(relu)]

        if drop != []:
            layers += [nn.Dropout2d(drop)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)

class DECNR2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=0, output_padding=0, norm='bnorm', relu=0.0, drop=[], bias=[]):
        super().__init__()

        if bias == []:
            if norm == 'bnorm':
                bias = False
            else:
                bias = True

        layers = []
        # layers += [Padding(padding=padding, padding_mode='reflection')]
        layers += [Deconv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)]

        if norm != []:
            layers += [Norm2d(nch_out, norm)]

        if relu != []:
            layers += [ReLU(relu)]

        if drop != []:
            layers += [nn.Dropout2d(drop)]

        self.decbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.decbr(x)

class ResBlock(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=3, stride=1, padding=1, padding_mode='reflection', norm='inorm', relu=0.0, drop=[], bias=[]):
        super().__init__()

        if bias == []:
            if norm == 'bnorm':
                bias = False
            else:
                bias = True

        layers = []

        # 1st conv
        layers += [CNR2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, norm=norm, relu=relu, bias=bias)]

        if drop != []:
            layers += [nn.Dropout2d(drop)]

        # 2nd conv
        layers += [CNR2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, norm=norm, relu=[], bias=bias)]

        self.resblk = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.resblk(x)

class Conv2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, bias=True, is_snorm=False):
        super(Conv2d, self).__init__()

        if is_snorm:
            self.conv = spectral_norm(nn.Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        else:
            self.conv = nn.Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)

class Deconv2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, output_padding=0, bias=True, is_snorm=False):
        super(Deconv2d, self).__init__()
        if is_snorm:
            self.deconv = spectral_norm(nn.ConvTranspose2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias))
        else:
            self.deconv = nn.ConvTranspose2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)

        # layers = [nn.Upsample(scale_factor=2, mode='bilinear'),
        #           nn.ReflectionPad2d(1),
        #           nn.Conv2d(nch_in , nch_out, kernel_size=3, stride=1, padding=0)]
        #
        # self.deconv = nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv(x)

class Norm2d(nn.Module):
    def __init__(self, nch, norm_mode):
        super(Norm2d, self).__init__()
        if norm_mode == 'bnorm':
            self.norm = nn.BatchNorm2d(nch)
        elif norm_mode == 'inorm':
            self.norm = nn.InstanceNorm2d(nch)

    def forward(self, x):
        return self.norm(x)


class ReLU(nn.Module):
    def __init__(self, relu):
        super(ReLU, self).__init__()
        if relu > 0:
            self.relu = nn.LeakyReLU(relu, True)
        elif relu == 0:
            self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(x)


class Padding(nn.Module):
    def __init__(self, padding, padding_mode='zeros', value=0):
        super(Padding, self).__init__()
        if padding_mode == 'reflection':
            self. padding = nn.ReflectionPad2d(padding)
        elif padding_mode == 'replication':
            self.padding = nn.ReplicationPad2d(padding)
        elif padding_mode == 'constant':
            self.padding = nn.ConstantPad2d(padding, value)
        elif padding_mode == 'zeros':
            self.padding = nn.ZeroPad2d(padding)

    def forward(self, x):
        return self.padding(x)


class Pooling2d(nn.Module):
    def __init__(self, nch=None, pool=2, type='avg'):
        super().__init__()

        if type == 'avg':
            self.pooling = nn.AvgPool2d(pool)
        elif type == 'max':
            self.pooling = nn.MaxPool2d(pool)
        elif type == 'conv':
            self.pooling = nn.Conv2d(nch, nch, kernel_size=pool, stride=pool)

    def forward(self, x):
        return self.pooling(x)


class UnPooling2d(nn.Module):
    def __init__(self, nch=None, pool=2, type='nearest'):
        super().__init__()

        if type == 'nearest':
            self.unpooling = nn.Upsample(scale_factor=pool, mode='nearest')
        elif type == 'bilinear':
            self.unpooling = nn.Upsample(scale_factor=pool, mode='bilinear', align_corners=True)
        elif type == 'conv':
            self.unpooling = nn.ConvTranspose2d(nch, nch, kernel_size=pool, stride=pool)

    def forward(self, x):
        return self.unpooling(x)



# class Projector(torch.autograd.Function):
#     """
#     We can implement our own custom autograd Functions by subclassing
#     torch.autograd.Function and implementing the forward and backward passes
#     which operate on Tensors.
#     """
#     @staticmethod
#     def forward(ctx, input, params):
#         """
#         In the forward pass we receive a Tensor containing the input and return
#         a Tensor containing the output. ctx is a context object that can be used
#         to stash information for backward computation. You can cache arbitrary
#         objects for use in the backward pass using the ctx.save_for_backward method.
#         """
#         ctx.params = params
#         vol = input.cpu().numpy().squeeze()
#         proj = np.zeros((params['nDctY'], params['nView'], params['nDctX']), dtype=np.float32)
#
#         vol_id = astra.data3d.link('-vol', params['vol_geom'], vol)
#         proj_id = astra.data3d.link('-sino', params['proj_geom'], proj)
#
#
#         cfg = astra.astra_dict('FP3D_CUDA')
#         cfg['ProjectionDataId'] = proj_id
#         cfg['VolumeDataId'] = vol_id
#
#         # Create the algorithm object from the configuration structure
#         alg_id = astra.algorithm.create(cfg)
#
#         # Run the algorithm
#         astra.algorithm.run(alg_id)
#
#         # Get the reconstructed data
#         proj = astra.data3d.get(proj_id)
#         proj = proj[:, np.newaxis, :, :]
#
#         astra.algorithm.delete(alg_id)
#         astra.data3d.delete(vol_id)
#         astra.data3d.delete(proj_id)
#
#         output = torch.from_numpy(proj)
#         if torch.cuda.is_available():
#             output = output.cuda()
#
#         return output
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         In the backward pass we receive a Tensor containing the gradient of the loss
#         with respect to the output, and we need to compute the gradient of the loss
#         with respect to the input.
#         """
#         params = ctx.params
#         proj = grad_output.cpu().numpy().squeeze()
#         vol = np.zeros((params['nImgZ'], params['nImgY'], params['nImgX']), dtype=np.float32)
#
#         proj_id = astra.data3d.link('-sino', params['proj_geom'], proj)
#         vol_id = astra.data3d.link('-vol', params['vol_geom'], vol)
#
#         cfg = astra.astra_dict('BP3D_CUDA')
#         cfg['ProjectionDataId'] = proj_id
#         cfg['ReconstructionDataId'] = vol_id
#
#         alg_id = astra.algorithm.create(cfg)
#
#         astra.algorithm.run(alg_id)
#
#         vol = astra.data3d.get(vol_id)
#         vol = vol[:, np.newaxis, :, :]
#
#         astra.algorithm.delete(alg_id)
#         astra.algorithm.delete(vol_id)
#         astra.algorithm.delete(proj_id)
#
#         grad_input = torch.from_numpy(vol)
#         if torch.cuda.is_available():
#             grad_input = grad_input.cuda()
#
#         return grad_input, None
#
#
# class Backprojector(torch.autograd.Function):
#     """
#     We can implement our own custom autograd Functions by subclassing
#     torch.autograd.Function and implementing the forward and backward passes
#     which operate on Tensors.
#     """
#     @staticmethod
#     def forward(ctx, input, params):
#         """
#         In the forward pass we receive a Tensor containing the input and return
#         a Tensor containing the output. ctx is a context object that can be used
#         to stash information for backward computation. You can cache arbitrary
#         objects for use in the backward pass using the ctx.save_for_backward method.
#         """
#         ctx.params = params
#         proj = input.cpu().numpy().squeeze()
#         vol = np.zeros((params['nImgZ'], params['nImgY'], params['nImgX']), dtype=np.float32)
#
#         proj_id = astra.data3d.link('-sino', params['proj_geom'], proj)
#         vol_id = astra.data3d.link('-vol', params['vol_geom'], vol)
#
#         cfg = astra.astra_dict('BP3D_CUDA')
#         cfg['ProjectionDataId'] = proj_id
#         cfg['ReconstructionDataId'] = vol_id
#
#         alg_id = astra.algorithm.create(cfg)
#
#         astra.algorithm.run(alg_id)
#
#         vol = astra.data3d.get(vol_id)
#         vol = vol[:, np.newaxis, :, :]
#
#         astra.algorithm.delete(alg_id)
#         astra.algorithm.delete(vol_id)
#         astra.algorithm.delete(proj_id)
#
#         output = torch.from_numpy(vol)
#         if torch.cuda.is_available():
#             output = output.cuda()
#
#         return output
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         In the backward pass we receive a Tensor containing the gradient of the loss
#         with respect to the output, and we need to compute the gradient of the loss
#         with respect to the input.
#         """
#         # output = ctx.saved_tensors
#
#         params = ctx.params
#         vol = grad_output.cpu().numpy().squeeze()
#         proj = np.zeros((params['nDctY'], params['nView'], params['nDctX']), dtype=np.float32)
#
#         vol_id = astra.data3d.link('-vol', params['vol_geom'], vol)
#         proj_id = astra.data3d.link('-sino', params['proj_geom'], proj)
#
#
#         cfg = astra.astra_dict('FP3D_CUDA')
#         cfg['ProjectionDataId'] = proj_id
#         cfg['VolumeDataId'] = vol_id
#
#         # Create the algorithm object from the configuration structure
#         alg_id = astra.algorithm.create(cfg)
#
#         # Run the algorithm
#         astra.algorithm.run(alg_id)
#
#         # Get the reconstructed data
#         proj = astra.data3d.get(proj_id)
#         proj = proj[:, np.newaxis, :, :]
#
#         astra.algorithm.delete(alg_id)
#         astra.data3d.delete(vol_id)
#         astra.data3d.delete(proj_id)
#
#         grad_input = torch.from_numpy(proj)
#         if torch.cuda.is_available():
#             grad_input = grad_input.cuda()
#
#         return grad_input, None




class Projector(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, input, params):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        ## Setting operation
        def Projection(vol, params):
            proj = np.zeros((params['nDctY'], params['nView'], params['nDctX']), dtype=np.float32)
            params['projection'](proj, vol)
            return proj.copy()

        ctx.params = params
        vol = input.cpu().numpy().squeeze()

        ##
        proj = Projection(vol.copy(), params)
        proj = proj[:, np.newaxis, :, :].copy()

        ##
        output = torch.from_numpy(proj)
        if torch.cuda.is_available():
            output = output.cuda()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        def ProjectionT(flt, params):
            if flt.ndim == 2:
                flt = flt[:, np.newaxis, :]

            flt = np.transpose(flt, (1, 0, 2)).copy()
            vol = np.zeros((params['nImgZ'], params['nImgY'], params['nImgX']), dtype=np.float32)
            params['backprojection'](vol, flt)
            vol = vol * (params['nView'] / np.pi)
            return vol.copy()

        params = ctx.params
        proj = grad_output.cpu().numpy().squeeze()

        ##
        vol = ProjectionT(proj, params)
        vol = vol[:, np.newaxis, :, :].copy()

        ##
        grad_input = torch.from_numpy(vol)
        if torch.cuda.is_available():
            grad_input = grad_input.cuda()

        return grad_input, None


class Backprojector(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, input, params):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        def Backprojection(flt, params):
            if flt.ndim == 2:
                flt = flt[:, np.newaxis, :]
            flt = np.transpose(flt, (1, 0, 2)).copy()
            vol = np.zeros((params['nImgZ'], params['nImgY'], params['nImgX']), dtype=np.float32)
            params['backprojection'](vol, flt)
            return vol.copy()

        ctx.params = params
        proj = input.cpu().numpy().squeeze()

        ##
        vol = Backprojection(proj, params)
        vol = vol[:, np.newaxis, :, :].copy()

        ##
        output = torch.from_numpy(vol)
        if torch.cuda.is_available():
            output = output.cuda()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        def BackprojectionT(vol, params):
            proj = np.zeros((params['nDctY'], params['nView'], params['nDctX']), dtype=np.float32)
            params['projection'](proj, vol)
            proj = proj / (params['nView'] / np.pi)
            return proj.copy()

        # output = ctx.saved_tensors

        params = ctx.params
        vol = grad_output.cpu().numpy().squeeze()

        ##
        proj = BackprojectionT(vol.copy(), params)
        proj = proj[:, np.newaxis, :, :].copy()

        grad_input = torch.from_numpy(proj)
        if torch.cuda.is_available():
            grad_input = grad_input.cuda()

        return grad_input, None

class Projection(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def forward(self, input):
        output = Projector.apply(input, self.params)

        return output


class Backprojection(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def forward(self, input):
        output = Backprojector.apply(input, self.params)

        return output


class Img2Prj(nn.Module):
    def __init__(self, params, nstage, nslice, nview=768):
        super().__init__()

        self.nstage = nstage
        self.nslice = nslice
        self.nview = nview

        self.params = get_params(name_project=params['name_project'], dir_project=params['dir_project'], dir_dll=params['dir_dll'],
                                 nStage=self.nstage, nslice=self.nslice, nView=self.nview)

    def forward(self, input):
        [BI, CI, HI, WI] = list(input.shape)
        input = input.reshape(-1, 1, HI, WI)

        output = Projector.apply(input, self.params)

        [BP, CP, HP, WP] = list(output.shape)
        output = output.reshape(BI, CI, HP, WP)

        return output



class Prj2Img(nn.Module):
    def __init__(self, params, nstage, nslice, nview=768):
        super().__init__()

        self.nstage = nstage
        self.nslice = nslice
        self.nview = nview

        self.params = get_params(name_project=params['name_project'], dir_project=params['dir_project'], dir_dll=params['dir_dll'],
                                 nStage=self.nstage, nslice=self.nslice, nView=self.nview)

    def forward(self, input):
        [BP, CP, HP, WP] = list(input.shape)
        input = input.reshape(-1, 1, HP, WP)

        output = Backprojector.apply(input, self.params)
        
        [BI, CI, HI, WI] = list(output.shape)
        output = output.reshape(BP, CP, HI, WI)

        return output



class PositionalEmbedding(nn.Module):
    def __init__(self, params, domain, weight, device='cuda:0'):
        super().__init__()

        self.params = params
        self.domain = domain
        self.weight = weight
        self.device = device

        if self.domain == 'image':
            self.pe = nn.Parameter(torch.randn(self.params['nImgZ'] // 2**(2*self.params['nStage']), 1, self.params['nImgY'], self.params['nImgX'], device=self.device) / self.weight)
        elif self.domain == 'projection':
            self.pe = nn.Parameter(torch.randn(self.params['nDctY'] // 2**(2*self.params['nStage']), 1, self.params['nView'], self.params['nDctX'], device=self.device))

    def forward(self, input):

        nch = self.pe.shape[0]
        nbatch = input.shape[0] // nch

        output = input

        for i in range(nbatch):
            output[nch*i:nch*(i+1), :, :, :] += self.pe

        return output


# class Filtration(nn.Module):
#     def __init__(self, params):
#         super().__init__()
#         _, filter_ft = designFilter(params)
#
#         filter_ft = np.tile(filter_ft[np.newaxis, :], (params['nView'], 1))
#
#         filter_ft = torch.from_numpy(filter_ft)
#         if torch.cuda.is_available():
#             filter_ft = filter_ft.cuda()
#
#         self.params = params
#         self.order = filter_ft.shape[1]
#         self.filter_ft = filter_ft
#
#     def forward(self, x):
#         filter_ft = self.filter_ft.repeat(x.shape[0], x.shape[1], 1, 1)
#
#         x = fft(x, n=self.order, dim=3)
#         x = x * filter_ft
#         x = ifft(x, n=self.order, dim=3)
#         x = torch.real(x[:, :, :, :self.params['nDct']])
#
#         return x

class PixelUnshuffle(nn.Module):
    def __init__(self, ry=2, rx=2):
        super().__init__()
        self.ry = ry
        self.rx = rx

    def forward(self, x):
        ry = self.ry
        rx = self.rx

        [B, C, H, W] = list(x.shape)

        x = x.reshape(B, C, H // ry, ry, W // rx, rx)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C * (ry * rx), H // ry, W // rx)

        return x


class PixelShuffle(nn.Module):
    def __init__(self, ry=2, rx=2):
        super().__init__()
        self.ry = ry
        self.rx = rx

    def forward(self, x):
        ry = self.ry
        rx = self.rx

        [B, C, H, W] = list(x.shape)

        x = x.reshape(B, C // (ry * rx), ry, rx, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C // (ry * rx), H * ry, W * rx)

        return x


class Composition(nn.Module):
    def __init__(self, nImgX, nImgY, nStage):
        super().__init__()

        nx = nImgX
        ny = nImgY

        ms = 2 ** nStage

        my = ny // ms
        mx = nx // ms
        
        params = {'nx': nx, 'ny': ny, 'ms': ms, 'mx': mx, 'my':my, 'nStage': nStage}
        self.params = params

    def forward(self, inp):
        out = inp.reshape((1, self.params['ms'], self.params['ms'], self.params['my'], self.params['mx']))
        out = out.permute((0, 1, 3, 2, 4))
        out = out.reshape((-1, 1, self.params['ny'], self.params['nx']))
        return out


class Decomposition(nn.Module):
    def __init__(self, nImgX, nImgY, nStage):
        super().__init__()

        nx = nImgX
        ny = nImgY

        ms = 2 ** nStage

        my = ny // ms
        mx = nx // ms

        params = {'nx': nx, 'ny': ny, 'ms': ms, 'mx': mx, 'my': my, 'nStage': nStage}
        self.params = params

    def forward(self, inp):
        out = inp.reshape((-1, self.params['ms'], self.params['my'], self.params['ms'], self.params['mx']))
        out = out.permute((0, 1, 3, 2, 4))
        out = out.reshape((-1, self.params['ms'] * self.params['ms'], self.params['my'], self.params['mx']))
        return out