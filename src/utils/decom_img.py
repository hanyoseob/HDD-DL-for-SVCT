import numpy as np
import matplotlib.pyplot as plt

def decom_img(img, nStage, params):
    if nStage == 0:
        return img

    nx = params['nImgX']
    ny = params['nImgY']

    ms = 2 ** nStage

    mx = nx // ms
    my = ny // ms

    img_dec = img.reshape((ms, my, ms, mx))
    img_dec = img_dec.transpose((0, 2, 1, 3))
    img_dec = img_dec.reshape((ms * ms, my, mx))

    return img_dec
