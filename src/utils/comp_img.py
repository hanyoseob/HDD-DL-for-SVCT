import numpy as np
import matplotlib.pyplot as plt


# def comp_img(rec_set, nstage):
#     if rec_set.ndim == 2:
#         return rec_set
#
#     [_, ny, nx] = rec_set.shape
#
#     nsz = ny * (2 ** nstage)
#     ndec = int(nsz / (2 ** nstage))
#     nset = 2 ** nstage
#
#     myh = (ny - ndec) // 2
#     mxh = (nx - ndec) // 2
#
#
#     rec = np.zeros((nsz, nsz))
#
#     # for i in range(nset - 1, 0 - 1, -1):
#     for i in range(nset):
#         for j in range(nset):
#             idx = nset * j + i
#             # rec_ = rec_set[:, :, idx]
#             rec_ = rec_set[idx, myh:ny-myh, mxh:nx-mxh]
#             # rec[N_cur * i:N_cur * (i + 1), N_cur * (nset - 1 - j): N_cur * (nset - 1 - j + 1)] = rec_
#             rec[ndec * (nset - 1 - i): ndec * (nset - 1 - i + 1), ndec * (nset - 1 - j): ndec * (nset - 1 - j + 1)] = rec_
#
#     return rec

def comp_img(img_dec, nStage, params):

    if nStage == 0:
        return img_dec

    # nx = params['nImgX']
    # ny = params['nImgY']
    #
    # [_, my, mx] = img_dec.shape
    #
    # mx_cur = int(nx / (2 ** nStage))
    # my_cur = int(ny / (2 ** nStage))
    #
    # mxh = (mx - mx_cur) // 2
    # myh = (my - my_cur) // 2
    #
    # rec = np.zeros((1, ny, nx))
    # 
    # nset = 2 ** nStage
    # for i in range(nset):
    #     for j in range(nset):
    #         idx = nset * j + i
    #         # rec_ = rec_set[:, :, idx]
    #         rec_ = rec_set[idx, myh:my - myh, mxh:mx - mxh]
    #         rec[0, my_cur * (nset - 1 - i):my_cur * (nset - 1 - i + 1), mx_cur * (nset - 1 - j): mx_cur * (nset - 1 - j + 1)] = rec_
    #
    # return rec

    nx = params['nImgX']
    ny = params['nImgY']

    ms = 2 ** nStage

    mx = nx // ms
    my = ny // ms

    # img_com = img_dec.reshape((1, ms, ms, my, mx))
    # img_com = img_com.transpose((0, 1, 3, 2, 4))
    # img_com = img_com.reshape((-1, 1, ny, nx))


    img_com = img_dec.reshape((ms, ms, my, mx))
    img_com = img_com.transpose((0, 2, 1, 3))
    img_com = img_com.reshape((1, ny, nx))

    return img_com
    
    