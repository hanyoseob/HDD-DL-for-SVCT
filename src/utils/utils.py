from __future__ import absolute_import, division, print_function

import os
import logging
import torch
import argparse

# import astra

import numpy as np
import matplotlib.pyplot as plt

''''
class Logger:
class Parser:
'''


def append_index(dir_result, fileset, step=False):
    index_path = os.path.join(dir_result, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        for key, value in fileset.items():
            index.write("<th>%s</th>" % key)
        index.write('</tr>')

    # for fileset in filesets:
    index.write("<tr>")

    if step:
        index.write("<td>%d</td>" % fileset["step"])
    index.write("<td>%s</td>" % fileset["name"])

    del fileset['name']

    for key, value in fileset.items():
        index.write("<td><img src='images/%s'></td>" % value)

    index.write("</tr>")
    return index_path


def add_plot(output, label, writer, epoch=[], ylabel='Density', xlabel='Radius', namescope=[]):
    fig, ax = plt.subplots()

    ax.plot(output.transpose(1, 0).detach().numpy(), '-')
    ax.plot(label.transpose(1, 0).detach().numpy(), '--')

    ax.set_xlim(0, 400)

    ax.grid(True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    writer.add_figure(namescope, fig, epoch)


def designFilter(params, filter_name='ram-lak'):
    # filter_name = 'ram-lak'
    # # filter_name = 'shepp-logan'
    # d = 1

    order = max(64, 2 ** (int(np.log2(2 * params['nDctX'])) + 1))
    d = 1

    if params['CT_NAME'] == 'parallel':
        d = params['dDctX']
    elif params['CT_NAME'] == 'fan':
        d = params['dDctX'] * params['dDSO'] / params['dDSD']

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


def str2bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ('yes', 'true', 'y', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'n', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Parser:
    def __init__(self, parser):
        self.__parser = parser
        self.__args = parser.parse_args()

        # set gpu ids
        # str_ids = self.__args.gpu_ids.split(',')
        # self.__args.gpu_ids = []
        # for str_id in str_ids:
        #     id = int(str_id)
        #     if id >= 0:
        #         self.__args.gpu_ids.append(id)
        # if len(self.__args.gpu_ids) > 0:
        #     torch.cuda.set_device(self.__args.gpu_ids[0])

    def get_parser(self):
        return self.__parser

    def get_arguments(self):
        return self.__args

    def write_args(self):
        params_dict = vars(self.__args)

        log_dir = os.path.join(params_dict['dir_log'], params_dict['scope'])
        args_name = os.path.join(log_dir, 'args.txt')

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        with open(args_name, 'wt') as args_fid:
            args_fid.write('----' * 10 + '\n')
            args_fid.write('{0:^40}'.format('PARAMETER TABLES') + '\n')
            args_fid.write('----' * 10 + '\n')
            for k, v in sorted(params_dict.items()):
                args_fid.write('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)) + '\n')
            args_fid.write('----' * 10 + '\n')

    def print_args(self, name='PARAMETER TABLES'):
        params_dict = vars(self.__args)

        print('----' * 10)
        print('{0:^40}'.format(name))
        print('----' * 10)
        for k, v in sorted(params_dict.items()):
            if '__' not in str(k):
                print('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)))
        print('----' * 10)


class Logger:
    def __init__(self, info=logging.INFO, name=__name__):
        logger = logging.getLogger(name)
        logger.setLevel(info)

        self.__logger = logger

    def get_logger(self, handler_type='stream_handler'):
        if handler_type == 'stream_handler':
            handler = logging.StreamHandler()
            log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(log_format)
        else:
            handler = logging.FileHandler('utils.log')

        self.__logger.addHandler(handler)

        return self.__logger
