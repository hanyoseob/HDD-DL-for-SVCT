import numpy as np
# import astra
import torch

##
def designFilter(params, filter_name='ram-lak'):
    # filter_name = 'ram-lak'
    # # filter_name = 'shepp-logan'
    # d = 1

    order = max(64, 2 ** (int(np.log2(2 * params['nDct'])) + 1))
    d = 1

    if params['CT_NAME'] == 'parallel':
        d = params['dDct']
    elif params['CT_NAME'] == 'fan':
        d = params['dDct'] * params['dDSO'] / params['dDSD']

    if filter_name == 'none':
        filter_ft = 0.5 * np.ones(order)
        filter_conv = np.zeros(order + 1)
        filter_conv[1:] = np.real(np.fft.ifft(filter_ft))

    else:
        n = np.arange(0, order // 2 + 1)

        filter_imp_resp = np.zeros(len(n))
        filter_imp_resp[0] = 1.0 / (4.0 * d * d)
        filter_imp_resp[1::2] = -1.0 / ((np.pi * n[1::2] * d) ** 2)

        filter_conv = np.zeros(order + 1)
        filter_conv[1:] = np.concatenate((filter_imp_resp[-2:0:-1], filter_imp_resp))
        filter_imp_resp = np.concatenate((filter_imp_resp, filter_imp_resp[-2:0:-1]))
        filter_ft = np.real(np.fft.fft(filter_imp_resp))
        filter_ft = filter_ft[0:order // 2 + 1]

        w = 2 * np.pi * np.arange(0, order // 2 + 1) / order

        if filter_name == 'ram-lak':
            pass
        elif filter_name == 'shepp-logan':
            filter_ft[1:] *= np.sin(w[1:] / (2 * d)) / (w[1:] / (2 * d))
        elif filter_name == 'cosine':
            filter_ft[1:] *= np.cos(w[1:] / (2 * d))
        elif filter_name == 'hamming':
            filter_ft[1:] *= (0.54 + 0.46 * np.cos(w[1:] / d))
        elif filter_name == 'hann':
            filter_ft[1:] *= (1 + np.cos(w[1:] / d)) / 2

        # filter_ft[w > np.pi * d] = 0
        filter_ft = np.concatenate((filter_ft, filter_ft[-2:0:-1]))

    return filter_conv, filter_ft

##
# def Projection_astra(vol, params, type_algorithm='FP3D_CUDA'):
#     # vol_geom = volume_geometry_generation(params)
#     # proj_geom = projection_geometry_generation(params)
#
#     proj = np.zeros((params['nDctY'], params['nView'], params['nDctX']), dtype=np.float32)
#
#     vol_id = astra.data3d.link('-vol', params['vol_geom'], vol)
#     proj_id = astra.data3d.link('-sino', params['proj_geom'], proj)
#
#     # cfg = astra.astra_dict('FDK_CUDA')
#     # cfg = astra.astra_dict('BP3D_CUDA')
#
#     cfg = astra.astra_dict(type_algorithm)
#     cfg['ProjectionDataId'] = proj_id
#     cfg['VolumeDataId'] = vol_id
#
#     # Create the algorithm object from the configuration structure
#     alg_id = astra.algorithm.create(cfg)
#
#     # Run the algorithm
#     astra.algorithm.run(alg_id)
#
#     # Get the reconstructed data
#     proj = astra.data3d.get(proj_id)
#
#     astra.algorithm.delete(alg_id)
#     astra.data3d.delete(vol_id)
#     astra.data3d.delete(proj_id)
#
#     return proj.copy()

# def Filtration_astra(proj, params):
#     filtering = params['filtering'].cuda()
#     filtering.eval()
#
#     numpy2tensor = lambda x: (torch.from_numpy(x[np.newaxis, :, :])).cuda()
#     tensor2numpy = lambda y: (y.cpu()).detach().numpy()[0]
#
#     proj = tensor2numpy(filtering(numpy2tensor(proj))) / (params['nView'] / np.pi)
#
#     return proj.copy()

##
# def ProjectionT_astra(proj, params, type_algorithm='BP3D_CUDA'):
#     # vol_geom = volume_geometry_generation(params)
#     # proj_geom = projection_geometry_generation(params)
#
#     vol = np.zeros((params['nImgZ'], params['nImgY'], params['nImgX']), dtype=np.float32)
#
#     vol_id = astra.data3d.link('-vol', params['vol_geom'], vol)
#     proj_id = astra.data3d.link('-sino', params['proj_geom'], proj)
#
#     # cfg = astra.astra_dict('FDK_CUDA')
#     # cfg = astra.astra_dict('BP3D_CUDA')
#
#     cfg = astra.astra_dict(type_algorithm)
#     cfg['ProjectionDataId'] = proj_id
#     cfg['ReconstructionDataId'] = vol_id
#
#     # Create the algorithm object from the configuration structure
#     alg_id = astra.algorithm.create(cfg)
#
#     # Run the algorithm
#     astra.algorithm.run(alg_id)
#
#     # Get the reconstructed data
#     vol = astra.data3d.get(vol_id)
#
#     astra.algorithm.delete(alg_id)
#     astra.data3d.delete(vol_id)
#     astra.data3d.delete(proj_id)
#
#     return vol.copy()

# def Backprojection_astra(proj, params, type_algorithm='BP3D_CUDA'):
#     # vol_geom = volume_geometry_generation(params)
#     # proj_geom = projection_geometry_generation(params)
#
#     vol = np.zeros((params['nImgZ'], params['nImgY'], params['nImgX']), dtype=np.float32)
#
#     vol_id = astra.data3d.link('-vol', params['vol_geom'], vol)
#     proj_id = astra.data3d.link('-sino', params['proj_geom'], proj)
#
#     # cfg = astra.astra_dict('FDK_CUDA')
#     # cfg = astra.astra_dict('BP3D_CUDA')
#
#     cfg = astra.astra_dict(type_algorithm)
#     cfg['ProjectionDataId'] = proj_id
#     cfg['ReconstructionDataId'] = vol_id
#
#     # Create the algorithm object from the configuration structure
#     alg_id = astra.algorithm.create(cfg)
#
#     # Run the algorithm
#     astra.algorithm.run(alg_id)
#
#     # Get the reconstructed data
#     vol = astra.data3d.get(vol_id)
#
#     astra.algorithm.delete(alg_id)
#     astra.data3d.delete(vol_id)
#     astra.data3d.delete(proj_id)
#
#     vol = vol / (params['nView'] / np.pi)
#
#     return vol.copy()

## Setting operation
def Projection(vol, params):
    proj = np.zeros((params['nDctY'], params['nView'], params['nDctX']), dtype=np.float32)
    params['projection'](proj, vol.copy())
    return proj.copy()

def ProjectionT(flt, params):
    flt = np.transpose(flt, (1, 0, 2)).copy()
    vol = np.zeros((params['nImgZ'], params['nImgY'], params['nImgX']), dtype=np.float32)
    params['backprojection'](vol, flt.copy())
    vol = vol * (params['nView'] / np.pi)
    return vol.copy()

def Filtration(proj, params):
    if params['FILTER_TYPE'] == 'conv':
        flt = np.zeros((params['nDctY'], params['nView'], params['nDctX']), dtype=np.float32)
        params['filtering_w_conv'](flt, proj.copy())
    elif params['FILTER_TYPE'] == 'ft':
        proj = proj.astype(dtype=np.float64).copy()
        flt = np.zeros((params['nDctY'], params['nView'], params['nDctX']), dtype=np.float64)
        params['filtering_w_ft'](flt, proj.copy())
        flt = flt.astype(dtype=np.float32)
    return flt.copy()

# def Backprojection(flt, params):
#     vol = np.zeros((params['nImgZ'], params['nImgY'], params['nImgX']), dtype=np.float32)
#     params['backprojection'](vol, flt)
#     return vol.copy()

def Backprojection(flt, params):
    flt = np.transpose(flt, (1, 0, 2)).copy()
    vol = np.zeros((params['nImgZ'], params['nImgY'], params['nImgX']), dtype=np.float32)
    params['backprojection'](vol, flt.copy())
    return vol.copy()

def BackprojectionT(vol, params):
    proj = np.zeros((params['nDctY'], params['nView'], params['nDctX']), dtype=np.float32)
    params['projection'](proj, vol.copy())
    proj = proj / (params['nView'] / np.pi)
    return proj.copy()

def FBP(proj, params):
    flt = Filtration(proj, params)
    vol = Backprojection(flt, params)
    return vol.copy()
