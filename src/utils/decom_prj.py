import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import radon, iradon, rescale, resize
from scipy import interpolate


def decom_prj(src, nStage, params):
    ndctx = int(params['nDctX']/2**nStage)
    ndctz = int(2 ** (2*nStage))
    nview = params['nView']

    src = src.transpose((0, 2, 1)).astype(np.float32).copy()
    dst = np.zeros((ndctz, nview, ndctx), dtype=np.float32)
    params['decomposition'](dst, src)
    # dst = np.transpose(dst, (0, 2, 1))
    return dst.copy()