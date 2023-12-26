import os
import numpy as np
import torch
from glob import glob

from skimage.util import pad
from skimage import transform

from src.utils.params import get_params, get_decom

from src.utils.decom_img import decom_img
from src.utils.decom_prj import decom_prj
from src.utils.comp_img import comp_img

from src.models import op_ct

from scipy.stats import poisson

import matplotlib.pyplot as plt


class Dataset(torch.utils.data.Dataset):
    """
    datasets of image files of the form
       stuff<number>_trans.pt
       stuff<number>_density.pt
    """

    def __init__(self, data_dir,
                 transform=None, idx=None,
                 params=None, nStage=None, downsample=None,
                 noise_range=None, downsample_range=None,
                 input_type='input', label_type='label',
                 flip=0.5, weight=5e3, mode='train'):
        if noise_range is None:
            noise_range = [4, 8]
        if downsample_range is None:
            downsample_range = [3, 4, 6, 8, 12]
        self.data_dir = data_dir
        self.transform = transform
        self.params = params
        self.nStage = nStage
        self.noise_range = noise_range
        self.downsample_range = downsample_range
        self.downample = downsample
        self.input_type = input_type
        self.label_type = label_type
        self.flip = flip
        self.weight = weight
        self.mode = mode

        ##
        self.params = get_params(name_project=params['name_project'], dir_project=params['dir_project'], dir_dll=params['dir_dll'], nStage=0, downsample=1, nView=params['nView'])
        self.params_ds = get_params(name_project=params['name_project'], dir_project=params['dir_project'], dir_dll=params['dir_dll'], nStage=0, downsample=1, nView=params['nView'])
        self.params_dec = get_params(name_project=params['name_project'], dir_project=params['dir_project'], dir_dll=params['dir_dll'], nStage=nStage, downsample=1, nView=params['nView'])

        self.params = get_decom(path_dll='lib_decom/', nStage=nStage, params=self.params)

        ##
        lst_input = glob(os.path.join(data_dir, f'{self.input_type}_*.npy'))
        lst_input.sort()

        lst_label = glob(os.path.join(data_dir, f'{self.label_type}_*.npy'))
        lst_label.sort()

        if idx is None:
            self.lst_input = lst_input
            self.lst_label = lst_label
        else:
            self.lst_input = lst_input[idx]
            self.lst_label = lst_label[idx]

        ##
    def __getitem__(self, index):
        prj = np.load(self.lst_input[index])
        
        if np.random.rand() < self.flip:
            prj = np.flip(prj, axis=1)

        if np.random.rand() < self.flip:
            prj = np.flip(prj, axis=2)

        if np.random.rand() < self.flip:
            prj = np.roll(prj, shift=int(np.random.randint(self.params['nView'])), axis=1)

        label_prj = prj.copy()
        downsample = self.downsample_range[np.random.randint(0, len(self.downsample_range))]

        self.params_ds = get_params(name_project=self.params['name_project'], dir_project=self.params['dir_project'],
                                    dir_dll=self.params['dir_dll'], nStage=0, downsample=downsample, nView=self.params['nView'] // downsample)


        label_flt = op_ct.Filtration(label_prj.copy(), self.params)
        input_flt = label_flt[:, ::downsample, :].copy()


        label_fbp = op_ct.Backprojection(label_flt.copy(), self.params)
        input_fbp = op_ct.Backprojection(input_flt.copy(), self.params_ds)
        
        input_prj = op_ct.Projection(input_fbp.copy(), self.params)
        input_flt = op_ct.Filtration(input_prj.copy(), self.params)
        input_mask = np.zeros_like(input_flt)
        input_mask[:, ::downsample, :] = 1

        input_flt = (input_mask * label_flt + (1 - input_mask) * input_flt).copy()


        label_flt_dec = decom_prj(label_flt, self.nStage, self.params)
        input_flt_dec = decom_prj(input_flt, self.nStage, self.params)
        input_mask_dec = np.zeros_like(input_flt_dec)
        input_mask_dec[:, ::downsample, :] = 1

        label_fbp_dec = decom_img(label_fbp, self.nStage, self.params)
        input_fbp_dec = op_ct.Backprojection(input_flt_dec, self.params_dec)

        

        ## Apply FOV - Projection
        data = {'label_fbp': label_fbp_dec.copy(), 'label_flt': label_flt_dec.copy(),
                'input_fbp': input_fbp_dec.copy(), 'input_flt': input_flt_dec.copy(),
                'input_mask': input_mask_dec.copy()}

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.lst_label)



class Normalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        for key, value in data.items():
            data[key] = (value - self.mean) / self.std
        return data


class UnNormalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        for key, value in data.items():
            data[key] = (value * self.std) + self.mean
        return data


class Weighting(object):
    def __init__(self, key):
        self.key = key

    def __call__(self, data):
        wgt = data[self.key]

        for key, value in data.items():
            if not key == self.key:
                data[key] = wgt * value
        return data


class Converter(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, dir='numpy2tensor'):
        self.dir = dir

    def __call__(self, sample):
        if self.dir == 'numpy2tensor':
            for key, value in sample.items():
                sample[key] = torch.from_numpy(value.copy())  # .permute((2, 0, 1))
        elif self.dir == 'tensor2numpy':
            # for key, value in sample.items():
            #     sample[key] = value.numpy()  # .transpose(1, 2, 0)
            sample = sample.cpu().detach().numpy()  # .transpose(1, 2, 0)

        return sample


class RandomFlip(object):
    def __call__(self, data):
        # Random Left or Right Flip
        if np.random.rand() > 0.5:
            for key, value in data.items():
                data[key] = np.flip(value, axis=0)

        if np.random.rand() > 0.5:
            for key, value in data.items():
                data[key] = np.flip(value, axis=1)

        return data


class ZeroPad(object):
    """Rescale the image in a sample to a given size

    Args:
      output_size (tuple or int): Desired output size.
                                  If tuple, output is matched to output_size.
                                  If int, smaller of image edges is matched
                                  to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, data):
        h, w = data['label'].shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        l = (new_w - w) // 2
        r = (new_w - w) - l

        u = (new_h - h) // 2
        b = (new_h - h) - u

        for key, value in data.items():
            data[key] = pad(value, pad_width=((u, b), (l, r), (0, 0)))

        return data


class Rescale(object):
    """Rescale the image in a sample to a given size

    Args:
      output_size (tuple or int): Desired output size.
                                  If tuple, output is matched to output_size.
                                  If int, smaller of image edges is matched
                                  to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, data):
        h, w = data['label'].shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        for key, value in data.items():
            data[key] = transform.resize(value, (new_h, new_w), mode=0)

        return data


class RandomCrop(object):
    """Crop randomly the image in a sample

    Args:
      output_size (tuple or int): Desired output size.
                                  If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        h, w = data['label'].shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
        id_x = np.arange(left, left + new_w, 1)

        for key, value in data.items():
            data[key] = value[id_y, id_x]

        return data


class CenterCrop(object):
    """Crop randomly the image in a sample

    Args:
    output_size (tuple or int): Desired output size.
                                If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        h, w = data['label'].shape[:2]
        new_h, new_w = self.output_size

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        for key, value in data.items():
            data[key] = value[top: top + new_h, left: left + new_w]

        return data

class UnifromSample(object):
    """Crop randomly the image in a sample

    Args:
      output_size (tuple or int): Desired output size.
                                  If int, square crop is made.
    """

    def __init__(self, stride):
        assert isinstance(stride, (int, tuple))
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            assert len(stride) == 2
            self.stride = stride

    def __call__(self, data):
        h, w = data['label'].shape[:2]
        stride_h, stride_w = self.stride
        new_h = h // stride_h
        new_w = w // stride_w

        top = np.random.randint(0, stride_h + (h - new_h * stride_h))
        left = np.random.randint(0, stride_w + (w - new_w * stride_w))

        id_h = np.arange(top, h, stride_h)[:, np.newaxis]
        id_w = np.arange(left, w, stride_w)

        for key, value in data.items():
            data[key] = value[id_h, id_w]

        return data
