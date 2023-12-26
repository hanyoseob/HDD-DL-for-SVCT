import os
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

from src.utils import params

from src.utils.params import get_decom

from src.utils.comp_img import comp_img

from src.utils.decom_img import decom_img
from src.utils.decom_prj import decom_prj_gpu as decom_prj

from skimage.metrics import normalized_root_mse as compare_nrmse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from src.models import op_ct

from src.models.model import *


## Load Networks
def load_dual(dir_checkpoint, net1, net2):
    ckpt = glob(os.path.join(dir_checkpoint, 'model*.pth'))
    ckpt.sort()

    dict_net = torch.load(ckpt[-1], map_location='cuda:0')

    net1.load_state_dict(dict_net['net1'])
    net2.load_state_dict(dict_net['net2'])

    return net1, net2

## Hounsfield Units
def mu2hu(mu):
    # HU (CORRECT)
    muw = 0.0192
    hu = (mu - muw) / muw * 1000

    return hu

## Compute Metrics: PSNR, NRMSE, SSIM
def compute_metrics(gt, pred, scope=''):
    #
    gt[gt < 0] = 0
    pred[pred < 0] = 0

    #
    nor_val = np.max(gt)

    gt /= nor_val
    pred /= nor_val

    # PSNR
    psnr_val = compare_psnr(gt, pred)

    # NRMSE
    nrmse_val = compare_nrmse(gt, pred)

    # SSIM
    ssim_val = compare_ssim(gt, pred)

    # Metrics
    print(f"[{scope}] PSNR = {psnr_val:0.4f}")
    print(f"[{scope}] NRMSE = {nrmse_val:0.4e}")
    print(f"[{scope}] SSIM = {ssim_val:0.4f}")
    print(' ')
   
## Hyper parameters
# FIGURE 6
dir_data = './data'

idx = 240  # 150

nviews = 768
ds = 6

# Network parameters
nstage_img2img = 0
nstage_prj2img = 4

# Model scope
scope_img2img = 'ct_img2img_residual_residual_block4_stage%d_loss_img' % nstage_img2img
scope_prj2img = 'ct_prj2img_consistency_residual_block4_stage%d_loss_img' % nstage_prj2img

# CT parameters
params_gt = get_params(name_project='parallel', dir_project='./projects', dir_dll='./lib_fbp', nView=nviews, nStage=0,
                       nslice=2 ** (2 * 0))
params_ds = get_params(name_project='parallel', dir_project='./projects', dir_dll='./lib_fbp', nView=nviews // ds,
                       nStage=0, nslice=2 ** (2 * 0))

params_img2img = get_params(name_project='parallel', dir_project='./projects', dir_dll='./lib_fbp',
                            nStage=nstage_img2img, nslice=2 ** (2 * nstage_img2img))
params_prj2img = get_params(name_project='parallel', dir_project='./projects', dir_dll='./lib_fbp',
                            nStage=nstage_prj2img, nslice=2 ** (2 * nstage_prj2img))

# Decompoisition operations
params_img2img_dec = get_decom(path_dll='./lib_decom', nStage=nstage_img2img, params=params_gt.copy())
params_prj2img_dec = get_decom(path_dll='./lib_decom', nStage=nstage_prj2img, params=params_gt.copy())

## Load W-Net
dir_checkpoint = os.path.join('checkpoints', scope_img2img)
net_img2img1 = UNet_res(in_chans=1, out_chans=1, chans=32, num_pool_layers=4)
net_img2img2 = UNet_res(in_chans=1, out_chans=1, chans=32, num_pool_layers=4)
net_img2img1, net_img2img2 = load_dual(dir_checkpoint, net_img2img1, net_img2img2)
net_img2img1 = net_img2img1.cuda()
net_img2img2 = net_img2img2.cuda()

## Load Ours DL
dir_checkpoint = os.path.join('checkpoints', scope_prj2img)
net_prj2img1 = UNet_res(in_chans=1, out_chans=1, chans=32, num_pool_layers=4)
net_prj2img2 = UNet_res(in_chans=1, out_chans=1, chans=32, num_pool_layers=4)
net_prj2img1, net_prj2img2 = load_dual(dir_checkpoint, net_prj2img1, net_prj2img2)
net_prj2img1 = net_prj2img1.cuda()
net_prj2img2 = net_prj2img2.cuda()

## Reconstruct image from measurement
## Full measurement
prj_gt = np.load(os.path.join(dir_data, 'prj_%08d.npy' % idx))

## Undersampled measurement
prj_ds = prj_gt[:, ::ds, :].copy()

## Ground Truth
flt_gt = op_ct.Filtration(prj_gt.copy(), params_gt).copy()
img_gt = op_ct.Backprojection(flt_gt.copy(), params_gt)

## FBP (w\ DS = 6)
flt_ds = op_ct.Filtration(prj_ds.copy(), params_ds).copy()
img_ds = op_ct.Backprojection(flt_ds.copy(), params_ds)

## Preprocessed input data for DL methods
prj_interp = op_ct.Projection(img_ds.copy(), params_gt)
flt_interp = op_ct.Filtration(prj_interp.copy(), params_gt)
mask_ds = np.zeros_like(flt_interp)
mask_ds[:, ::ds, :] = 1

flt_interp = (mask_ds * flt_gt + (1 - mask_ds) * flt_interp).copy()
flt_prj2img = decom_prj(flt_interp.copy(), nstage_prj2img, params_prj2img_dec)
flt_img2img = decom_prj(flt_interp.copy(), nstage_img2img, params_img2img_dec)

mask_prj2img = np.zeros_like(flt_prj2img)
mask_prj2img[:, ::ds, :] = 1

input_img2img = op_ct.Backprojection(flt_img2img, params_gt)

## W-Net (Image-Image)
net_img2img1.eval()
net_img2img2.eval()

input_img2img = torch.from_numpy(input_img2img[np.newaxis, :, :, :]).cuda()

[BI, CI, HI, WI] = list(input_img2img.shape)
input_img2img = input_img2img.reshape(-1, 1, HI, WI)

with torch.no_grad():
    output_img2img1 = input_img2img - net_img2img1(input_img2img)
    output_img2img2 = output_img2img1 - net_img2img2(output_img2img1)

# Result reconstructed from 1st phase network of W-Net
output_img2img1 = output_img2img1.reshape(BI, CI, HI, WI).cpu().detach().numpy()[0]
# Result reconstructed from 2nd phase network of W-Net
output_img2img2 = output_img2img2.reshape(BI, CI, HI, WI).cpu().detach().numpy()[0]


## D-Net (Projection-Image)
net_prj2img1.eval()
net_prj2img2.eval()

flt_prj2img = torch.from_numpy(flt_prj2img[np.newaxis, :, :, :]).cuda()
mask_prj2img = torch.from_numpy(mask_prj2img[np.newaxis, :, :, :]).cuda()

[BP, CP, HP, WP] = list(flt_prj2img.shape)
flt_prj2img = flt_prj2img.reshape(-1, 1, HP, WP)
mask_prj2img = mask_prj2img.reshape(-1, 1, HP, WP)

output_prj2img1 = torch.zeros_like(flt_prj2img)

nbatch = 32
with torch.no_grad():
    for i in range(0, BP*CP, nbatch):
        output_prj2img1[i:(i+nbatch)] = (mask_prj2img[i:(i+nbatch)] * flt_prj2img[i:(i+nbatch)] + (1 - mask_prj2img[i:(i+nbatch)]) * net_prj2img1(flt_prj2img[i:(i+nbatch)]))

    output_prj2img1 = op_ct.Backprojection(output_prj2img1.cpu().detach().numpy()[:, 0, :, :].copy(), params_prj2img)
    output_prj2img1 = torch.from_numpy(output_prj2img1[:, np.newaxis, :, :]).cuda()
    output_prj2img2 = output_prj2img1 - net_prj2img2(output_prj2img1)

output_prj2img1 = output_prj2img1.reshape(BP, CP, params_prj2img['nImgY'], params_prj2img['nImgX']).cpu().detach().numpy()[0]
output_prj2img2 = output_prj2img2.reshape(BP, CP, params_prj2img['nImgY'], params_prj2img['nImgX']).cpu().detach().numpy()[0]

# Result reconstructed from 1st phase network of our D-Net
output_prj2img1 = comp_img(output_prj2img1, nstage_prj2img, params_prj2img_dec)
# Result reconstructed from 2nd phase network of our D-Net
output_prj2img2 = comp_img(output_prj2img2, nstage_prj2img, params_prj2img_dec)

## Summary
# Metrics
compute_metrics(img_gt[0].copy(), img_ds[0].copy(), 'FBP')
compute_metrics(img_gt[0].copy(), output_img2img2[0].copy(), 'W-Net')
compute_metrics(img_gt[0].copy(), output_prj2img2[0].copy(), 'Ours')

# Figures
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))

axs[0, 0].imshow(mu2hu(img_gt[0].copy()), vmax=240, vmin=-160, cmap='gray')
axs[0, 0].set_title(f'Ground Truth ({nviews})')
axs[0, 0].axis(False)

axs[0, 1].imshow(mu2hu(img_ds[0].copy()), vmax=240, vmin=-160, cmap='gray')
axs[0, 1].set_title(f'FBP ({nviews//ds})')
axs[0, 1].axis(False)

axs[1, 0].imshow(mu2hu(output_img2img2[0].copy()), vmax=240, vmin=-160, cmap='gray')
axs[1, 0].set_title(f'W-Net (I-domains & Lv{nstage_img2img + 1})')
axs[1, 0].axis(False)

axs[1, 1].imshow(mu2hu(output_prj2img2[0].copy()), vmax=240, vmin=-160, cmap='gray')
axs[1, 1].set_title(f'Ours (D-domains & Lv{nstage_prj2img + 1})')
axs[1, 1].axis(False)

plt.suptitle('Figure 6.')
plt.tight_layout()
plt.show()
