import os
import numpy as np

import json
import yaml

# import astra

import torch
import torch.nn as nn
from torch.autograd import Variable

import ctypes
from ctypes import *

##
def Generate_Filter(params):
    def nextpow2(x):
        """returns the smallest power of two that is greater than or equal to the
        absolute value of x.

        This function is useful for optimizing FFT operations, which are
        most efficient when sequence length is an exact power of two.

        :Example:

        .. doctest::

            # >>> from spectrum import nextpow2
            # >>> x = [255, 256, 257]
            # >>> nextpow2(x)
            # array([8, 8, 9])

        """
        res = np.ceil(np.log2(x))
        return res.astype('int')  # we want integer values only but ceil gives float


    def DesignFilter(params):
        dker = params['dDctX']
        # nker = max(64, 2**nextpow2(2*params['nDctX']))
        nker = max(64, 2 ** nextpow2(2 * params['nDctX']))
        mode = params['CT_MODE']
        filter_name = params['FILTER_NAME']
        d = params['CUTOFF']

        j = 0
        ker = np.zeros(nker, dtype=np.float32)

        # if mode == CT_MODE['parallel'] or mode == CT_MODE['fan_spaced']:
        if filter_name == 'parallel' or filter_name == 'fan_spaced':
            for i in range(-nker // 2, nker // 2):
                if i == 0:
                    ker[j] = 1.0 / (4.0 * dker * dker)
                elif i % 2 == 0:
                    ker[j] = 0.0
                else:
                    ker[j] = -1.0 / ((i * np.pi * dker) * (i * np.pi * dker))
                j += 1
        # elif mode == CT_MODE['fan_angular']:
        elif filter_name == 'fan_angular':
            for i in range(-nker // 2, nker // 2):
                if i == 0:
                    ker[j] = 1.0 / (8.0 * dker * dker)
                elif i % 2 == 0:
                    ker[j] = 0.0
                else:
                    ker[j] = -1.0 / (2.0 * (np.pi * np.sin(i * dker)) * (np.pi * np.sin(i * dker)))
                j += 1

        ker_ft = np.abs(np.fft.fft(ker, axis=0))
        filt = ker_ft[:nker // 2 + 1]
        # filt = 2 * np.arange(0, nker // 2 + 1) / nker
        w = 2 * np.pi * np.arange(0, nker // 2 + 1) / nker

        if filter_name == 'ram-lak':
            # print(filter_name)
            pass
        elif filter_name == 'shepp-logan':
            # print(filter_name)
            filt[1:] = filt[1:] * (np.sin(w[1:] / (2 * d)) / (w[1:] / (2 * d)))
        elif filter_name == 'cosine':
            # print(filter_name)
            filt[1:] = filt[1:] * np.cos(w[1:] / (2 * d))
        elif filter_name == 'hamming':
            # print(filter_name)
            filt[1:] = filt[1:] * (.54 + .46 * np.cos(w[1:] / d))
        elif filter_name == 'hann':
            # print(filter_name)
            filt[1:] = filt[1:] * (1 + np.cos(w[1:] / d)) / 2

        filt[w > np.pi * d] = 0
        filt = np.concatenate([filt, filt[-2:0:-1]])

        return ker, filt

    order = max(64, 2 ** (int(np.log2(2 * params['nDctX'])) + 1))
    nch_in = 1
    nch_out = 1
    filtering = nn.Conv2d(nch_in, nch_out, kernel_size=(1, order + 1), stride=1, padding=(0, (order + 1) // 2),
                          bias=False)
    _weight, _ = DesignFilter(params)
    weight = np.zeros(order + 1, np.float32)
    weight[:-1] = _weight
    weight = np.reshape(weight.astype(np.float32), filtering.weight.data.shape)
    filtering.weight.data = torch.from_numpy(weight)

    for param in filtering.parameters():
        param.requires_grad = False

    return filtering



## ref.
# https://www.astra-toolbox.com/docs/geom3d.html

##
# def volume_geometry_generation(params):
#     vol_shape = (params['nImgY'], params['nImgX'], params['nImgZ'])
#
#     vol_geom = astra.create_vol_geom(vol_shape)
#     vol_geom['option']['WindowMinZ'] = -0.5 * params['dImgZ']*params['nImgZ']
#     vol_geom['option']['WindowMaxZ'] = +0.5 * params['dImgZ']*params['nImgZ']
#     vol_geom['option']['WindowMinY'] = -0.5 * params['dImgY'] * params['nImgY']
#     vol_geom['option']['WindowMaxY'] = +0.5 * params['dImgY'] * params['nImgY']
#     vol_geom['option']['WindowMinX'] = -0.5 * params['dImgX'] * params['nImgX']
#     vol_geom['option']['WindowMaxX'] = +0.5 * params['dImgX'] * params['nImgX']
#
#     return vol_geom

##
# def projection_geometry_generation(params):
#     vectors = np.zeros((params['nView'], 12))
#
#     for i in range(params['nView']):
#         beta = params['dView'] * i
#
#         # X-ray source position
#         vectors[i, 0] = +np.sin(beta) * params['dDSO']
#         vectors[i, 1] = -np.cos(beta) * params['dDSO']
#         vectors[i, 2] = 0
#
#         # Center of detector
#         vectors[i, 3] = + np.cos(beta) * (-params['dDctX'] * params['dOffsetDctX']) \
#                         - np.sin(beta) * (params['dDSD'] - params['dDSO'])
#         vectors[i, 4] = + np.sin(beta) * (-params['dDctX'] * params['dOffsetDctX']) \
#                         + np.cos(beta) * (params['dDSD'] - params['dDSO'])
#         vectors[i, 5] = -params['dDctY'] * params['dOffsetDctY']
#
#         # vector from detector pixel (0,0) to (0,1)
#         vectors[i, 6] = +np.cos(beta) * params['dDctX']
#         vectors[i, 7] = +np.sin(beta) * params['dDctX']
#         vectors[i, 8] = 0
#
#         # vector from detector pixel (0,0) to (1,0)
#         vectors[i, 9] = 0
#         vectors[i, 10] = 0
#         vectors[i, 11] = params['dDctY']
#
#     # proj_geom = astra.create_proj_geom('cone_vec', params['nDctY'], params['nDctX'], vectors)
#     proj_geom = astra.create_proj_geom('parallel3d_vec', params['nDctY'], params['nDctX'], vectors)
#
#     return proj_geom



## OPERATION SETTING
# def get_params(nImgX=512, nImgY=512, nImgZ=1, nDctY=1, nDctX=768, nView=1024, downsample=1, nStage=0):
#     params = {}
#
#     params['CT_NAME'] = 'parallel'
#     params['CT_MODE'] = CT_MODE[params['CT_NAME']]
#
#     params['FILTER_NAME'] = 'ram_lak'
#     params['FILTER_MODE'] = FILTER_MODE[params['FILTER_NAME']]
#     params['CUTOFF'] = 1.0
#
#     params['dDSO'] = 800
#     params['dDSD'] = 1200
#
#     params['nDctX'] = int(nDctX // 2**nStage)
#     params['nDctY'] = int(nDctY)
#
#     params['dDctX'] = 1.0
#     params['dDctY'] = 1.0
#
#     params['dOffsetDctX'] = 0
#     params['dOffsetDctY'] = 0
#
#     params['nImgX'] = int(nImgX // 2**nStage)
#     params['nImgY'] = int(nImgY // 2**nStage)
#     params['nImgZ'] = int(nImgZ)
#
#     params['dImgX'] = 1.0
#     params['dImgY'] = 1.0
#     params['dImgZ'] = 1.0
#
#     params['nView'] = int(nView)
#     params['dView'] = 2 * np.pi / params['nView']
#
#     params['threshold'] = 1e-3
#     params['nStage'] = int(nStage)
#     params['downsample'] = int(downsample)
#
#     # params['filtering'] = Generate_Filter(params)
#
#     # params['vol_geom'] = volume_geometry_generation(params)
#     # params['proj_geom'] = projection_geometry_generation(params)
#
#     params['dir_dll'] = './lib_fbp'
#
#     params = get_operations(dir_dll=params['dir_dll'], params=params)
#
#     return params.copy()


def get_params(name_project='parallel', dir_project='./projects', dir_dll='./lib_fbp', nView=None, downsample=1, nStage=0, nslice=None):
    with open(os.path.join(dir_project, "{}.json".format(name_project)), 'r') as f:
        data = json.load(f)
        CT_MODE = data['CT_MODE']
        BEAM_MODE = data['BEAM_MODE']
        FILTER_MODE = data['FILTER_MODE']

    with open(os.path.join(dir_project, "{}.yml".format(name_project)), 'r') as f:
        params = yaml.safe_load(f.read())
        if nView is not None:
            params['nView'] = nView
        params['dView'] = 2 * np.pi / params['nView']
        params['CT_MODE'] = CT_MODE[params['CT_NAME']]
        params['FILTER_MODE'] = FILTER_MODE[params['FILTER_NAME']]

    params['nStage'] = int(nStage)
    params['downsample'] = int(downsample)
    params['nDctX'] = int(params['nDctX'] // 2**params['nStage'])
    params['nImgX'] = int(params['nImgX'] // 2**params['nStage'])
    params['nImgY'] = int(params['nImgY'] // 2**params['nStage'])

    if nslice is not None:
        params['nDctY'] = nslice
        params['nImgZ'] = nslice
    else:
        params['nDctY'] = params['nDctY'] * (2 ** (2 * params['nStage']))
        params['nImgZ'] = params['nImgZ'] * (2 ** (2 * params['nStage']))

    params['name_project'] = name_project
    params['dir_project'] = dir_project
    params['dir_dll'] = dir_dll

    params = get_operations(dir_dll=params['dir_dll'], params=params)

    return params.copy()


def get_operations(dir_dll='./lib', device='cuda', params=None):
    ##
    '''
    Filtered Back-Projectin (FBP) algorithm
    '''
    dir_dll_projection = os.path.join(os.path.dirname(__file__), dir_dll, 'libprojection_gpu.so')
    _dll_projection = ctypes.CDLL(dir_dll_projection)

    # dir_dll_backprojection = os.path.join(os.path.dirname(__file__), dir_dll, 'libbackprojection_gpu.so')
    # _dll_backprojection = ctypes.CDLL(dir_dll_backprojection)

    dir_dll_backprojection = os.path.join(os.path.dirname(__file__), dir_dll, 'libbackprojection_decom_gpu.so')
    _dll_backprojection = ctypes.CDLL(dir_dll_backprojection)

    dir_dll_filtering_w_conv = os.path.join(os.path.dirname(__file__), dir_dll, 'libfiltering_w_conv_gpu.so')
    _dll_filtering_w_conv = ctypes.CDLL(dir_dll_filtering_w_conv)

    dir_dll_filtering_w_ft = os.path.join(os.path.dirname(__file__), dir_dll, 'libfiltering_w_ft_gpu.so')
    _dll_filtering_w_ft = ctypes.CDLL(dir_dll_filtering_w_ft)

    # dir_dll_decomposition = os.path.join(os.path.dirname(__file__), dir_dll, 'libdecomposition_gpu.so')
    # _dll_decomposition = ctypes.CDLL(dir_dll_decomposition)

    ##
    RunProjection = _dll_projection.RunProjection
    RunProjection.argtypes = (POINTER(c_float), POINTER(c_float),
                              c_float, c_float, c_float,
                              c_int, c_int, c_int,
                              c_float, c_float,
                              c_int, c_int,
                              c_float,
                              c_float, c_int,
                              c_float, c_float,
                              c_int)
    RunProjection.restypes = c_void_p

    RunBackprojection = _dll_backprojection.RunBackprojection
    RunBackprojection.argtypes = (POINTER(c_float), POINTER(c_float),
                                  c_float, c_float, c_float,
                                  c_int, c_int, c_int,
                                  c_float, c_float,
                                  c_int, c_int,
                                  c_float,
                                  c_float, c_int,
                                  c_float, c_float,
                                  c_int)
    RunBackprojection.restypes = c_void_p

    RunFiltering_w_conv = _dll_filtering_w_conv.RunFiltering
    RunFiltering_w_conv.argtypes = (POINTER(c_float), POINTER(c_float),
                             c_float, c_float,
                             c_int, c_int,
                             c_int,
                             c_float, c_float,
                             c_int)
    RunFiltering_w_conv.restypes = c_void_p

    RunFiltering_w_ft = _dll_filtering_w_ft.RunFiltering
    RunFiltering_w_ft.argtypes = (POINTER(c_double), POINTER(c_double),
                             c_float, c_float,
                             c_int, c_int,
                             c_int,
                             c_float, c_float,
                             c_int,
                             c_int, c_float)
    RunFiltering_w_ft.restypes = c_void_p


    # RunDecomposition = _dll_decomposition.RunDecomposition
    # RunDecomposition.argtypes = (POINTER(c_float), POINTER(c_float),
    #                              c_int, c_int,
    #                              c_int,
    #                              c_float, c_int,
    #                              c_int)
    # RunDecomposition.restypes = c_void_p


    c_float_p = lambda x: x.ctypes.data_as(POINTER(c_float))
    c_double_p = lambda x: x.ctypes.data_as(POINTER(c_double))

    ##
    params['c_float_p'] = c_float_p
    params['c_double_p'] = c_double_p

    params['projection'] = lambda output, input: \
        RunProjection(
                        params['c_float_p'](output), params['c_float_p'](input),
                        params['dImgX'], params['dImgY'], params['dImgZ'],
                        params['nImgX'], params['nImgY'], params['nImgZ'],
                        params['dDctX'], params['dDctY'],
                        params['nDctX'], params['nDctY'],
                        params['dOffsetDctX'],
                        params['dView'], params['nView'],
                        params['dDSO'], params['dDSD'],
                        params['CT_MODE'])

    params['backprojection'] = lambda output, input: \
        RunBackprojection(
                        params['c_float_p'](output), params['c_float_p'](input),
                        params['dImgX'], params['dImgY'], params['dImgZ'],
                        params['nImgX'], params['nImgY'], params['nImgZ'],
                        params['dDctX'], params['dDctY'],
                        params['nDctX'], params['nDctY'],
                        params['dOffsetDctX'],
                        params['dView'], params['nView'],
                        params['dDSO'], params['dDSD'],
                        params['CT_MODE'])

    params['filtering_w_conv'] = lambda output, input: \
        RunFiltering_w_conv(
                        params['c_float_p'](output), params['c_float_p'](input),
                        params['dDctX'], params['dDctY'],
                        params['nDctX'], params['nDctY'],
                        params['nView'],
                        params['dDSO'], params['dDSD'],
                        params['CT_MODE'])

    params['filtering_w_ft'] = lambda output, input: \
        RunFiltering_w_ft(
                        params['c_double_p'](output), params['c_double_p'](input),
                        params['dDctX'], params['dDctY'],
                        params['nDctX'], params['nDctY'],
                        params['nView'],
                        params['dDSO'], params['dDSD'],
                        params['CT_MODE'],
                        params['FILTER_MODE'],
                        params['CUTOFF'])

    # params['decomposition'] = lambda output, input: \
    #     RunDecomposition(
    #         params['c_float_p'](output), params['c_float_p'](input),
    #         params['nImgX'], params['nImgY'],
    #         params['nDctX'],
    #         params['dView'], params['nView'],
    #         params['nStage'])

    params['device'] = device

    return params

##
def get_decom(path_dll, nStage, params):
    path_dll = os.path.join(os.path.dirname(__file__), path_dll, 'libdecomposition_gpu.so')
    _dll = ctypes.CDLL(path_dll)

    RunDecomposition = _dll.RunDecomposition
    RunDecomposition.argtypes = (POINTER(c_float), POINTER(c_float),
                                 c_int, c_int,
                                 c_int,
                                 c_float, c_int,
                                 c_int)
    RunDecomposition.restypes = c_void_p

    c_float_p = lambda x: x.ctypes.data_as(POINTER(c_float))
    c_double_p = lambda x: x.ctypes.data_as(POINTER(c_double))

    params['c_float_p'] = c_float_p
    params['c_double_p'] = c_double_p

    #
    params['decomposition'] = lambda output, input: \
        RunDecomposition(
            params['c_float_p'](output), params['c_float_p'](input),
            params['nImgX'], params['nImgY'],
            params['nDctX'],
            params['dView'], params['nView'],
            nStage)

    # params['device'] = device

    return params


##
def get_tv(path_dll, params=None):
    #
    # path_dll_tv = os.path.join(os.path.dirname(__file__), path_dll, 'libchambolleTV_3D_gpu.so')
    path_dll_tv = os.path.join(path_dll, 'libchambolleTV_3D_gpu.so')
    _dll_tv = ctypes.CDLL(path_dll_tv)

    #
    RunTV = _dll_tv.RunChambolleProxTV
    RunTV.argtypes = (POINTER(c_float), POINTER(c_float),
                      c_float, c_float,
                      c_int,
                      c_int, c_int, c_int)
    RunTV.restypes = c_void_p

    c_float_p = lambda x: x.ctypes.data_as(POINTER(c_float))
    c_double_p = lambda x: x.ctypes.data_as(POINTER(c_double))

    params['c_float_p'] = c_float_p
    params['c_double_p'] = c_double_p

    #
    params['tv'] = lambda output, input: \
        RunTV(
            params['c_float_p'](output), params['c_float_p'](input),
            params['lambda'], params['tau'],
            params['niter'],
            params['nImgX'], params['nImgY'], params['nImgZ'])

    # params['device'] = device

    return params

