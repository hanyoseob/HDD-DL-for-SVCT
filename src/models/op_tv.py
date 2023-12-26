import numpy as np
# import astra
import torch

def TotalVariation(src, params):
    src = src.astype(np.float32).copy()
    dst = np.zeros((params['nImgZ'], params['nImgY'], params['nImgX']), dtype=np.float32)
    params['tv'](dst, src)
    return dst.copy()
