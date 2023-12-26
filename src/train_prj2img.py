from src.models.model import *
from src.data.dataset import *
from src.utils.utils import *
from src.utils.params import get_params, get_decom

# from src.utils.decom_img import decom_img
# from src.utils.decom_prj import decom_prj_gpu as dcom_prj
# from src.utils.comp_img import comp_img

import torch

import torch.nn as nn

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torchsummary import summary

import matplotlib.pyplot as plt

# import wandb

WGT = 5e+3


# MEAN = 6e-5 * WGT
# STD = 6e-5 * WGT

def set_gpu(str_ids):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str_ids

    gpu_ids = []
    for gpu_id, str_id in enumerate(str_ids.split(',')):
        gpu_ids.append(gpu_id)

    return gpu_ids


def print_args(dir_log, args):
    if not os.path.exists(dir_log):
        os.makedirs(dir_log)

    args_name = os.path.join(dir_log, 'args.txt')

    params_dict = vars(args)
    with open(args_name, 'wt') as args_fid:
        args_fid.write('----' * 10 + '\n')
        args_fid.write('{0:^40}'.format('PARAMETER TABLES') + '\n')
        args_fid.write('----' * 10 + '\n')
        for k, v in sorted(params_dict.items()):
            args_fid.write('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)) + '\n')
        args_fid.write('----' * 10 + '\n')


class Train:
    def __init__(self, args):
        self.mode = args.mode
        self.train_continue = args.train_continue

        self.name_data = args.name_data
        self.scope = args.scope

        self.dir_checkpoint = args.dir_checkpoint
        self.dir_log = args.dir_log

        self.dir_data = args.dir_data
        self.dir_result = args.dir_result

        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size

        self.name_project = args.name_project
        self.dir_project = args.dir_project
        self.dir_dll = args.dir_dll

        self.is_consistency = args.is_consistency

        self.lr_prj = args.lr_prj
        self.lr_img = args.lr_img

        self.lr_type_prj = args.lr_type_prj
        self.lr_type_img = args.lr_type_img

        self.loss_type_prj = args.loss_type_prj
        self.loss_type_img = args.loss_type_img

        self.wd_prj = args.wd_prj
        self.wd_img = args.wd_img

        self.optim = args.optim

        self.nstage = args.nstage

        self.downsample = args.downsample

        self.nch_in = args.nch_in
        self.nch_out = args.nch_out

        self.num_block = args.num_block
        self.num_channels = args.num_channels

        self.input_type = args.input_type
        self.label_type = args.label_type

        self.num_freq_disp = args.num_freq_disp
        self.num_freq_save = args.num_freq_save

        self.num_workers = args.num_workers

        # Set CUDA device
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_ids)

        self.gpu_ids = int(0)
        self.device = torch.device('cuda:%d' % self.gpu_ids if torch.cuda.is_available() else 'cpu')

        ##
        self.params = get_params(name_project=self.name_project, dir_project=self.dir_project, dir_dll=self.dir_dll,
                                 nStage=0, nslice=self.batch_size)
        self.params_dec = get_params(name_project=self.name_project, dir_project=self.dir_project, dir_dll=self.dir_dll,
                                     nStage=self.nstage, nslice=self.batch_size * 2 ** (2 * self.nstage))

        self.noise_range = self.params['noise_range']
        self.downsample_range = self.params['downsample_range']

        ##
        # self.scope = self.scope + '_%s' % self.lr_type_prj + '_downsample%s' % str(self.downsample_range)[1:-1].replace(', ', '_') + '_stage%d' % self.nstage + '_loss_%s' % self.loss_type_img
        self.scope = self.scope + '_%s_%s' % (self.lr_type_prj, self.lr_type_img) + '_block%d' % self.num_block + '_stage%d' % self.nstage + '_loss_%s' % self.loss_type_img
        print(self.scope)

        print_args(os.path.join(self.dir_log, self.scope), args)

    ## Train function
    def train(self):
        ## wandb
        # wandb.init(
        #     # set the wandb project where this run will be logged
        #     project=self.name_data,
        #     name=self.scope,
        # )
        #
        # wandb.config.update(self)

        ## CT parameter
        params = self.params
        params_dec = self.params_dec

        ## setup directories
        dir_checkpoint = os.path.join(self.dir_checkpoint, self.scope)
        os.makedirs(dir_checkpoint, exist_ok=True)

        dir_result_train = os.path.join(self.dir_result, self.scope, 'train')
        os.makedirs(os.path.join(dir_result_train, 'images'), exist_ok=True)

        dir_result_valid = os.path.join(self.dir_result, self.scope, 'valid')
        os.makedirs(os.path.join(dir_result_valid, 'images'), exist_ok=True)

        dir_log_train = os.path.join(self.dir_log, self.scope, 'train')
        os.makedirs(dir_log_train, exist_ok=True)

        dir_log_valid = os.path.join(self.dir_log, self.scope, 'valid')
        os.makedirs(dir_log_valid, exist_ok=True)

        name_data = '%s_dct%d_view%d' % (params['CT_NAME'], params['nDctX'], params['nView']) + params['scope_data']

        dir_data_train = os.path.join(self.dir_data, name_data, 'dec%d' % 0, 'train')
        dir_data_valid = os.path.join(self.dir_data, name_data, 'dec%d' % 0, 'valid')

        ## setup transform functions
        transform_train = transforms.Compose([Converter(dir='numpy2tensor'), ])
        transform_valid = transforms.Compose([Converter(dir='numpy2tensor'), ])

        tensor2numpy = Converter(dir='tensor2numpy')

        ## setup dataset loaders about train and validation data
        dataset_train = Dataset(data_dir=dir_data_train,
                                transform=transform_train,  # idx=idx_train,
                                params=params.copy(), nStage=self.nstage,
                                noise_range=self.noise_range, downsample_range=self.downsample_range,
                                input_type=self.input_type, label_type=self.label_type,
                                flip=0.5, weight=WGT, mode='train')
        dataset_valid = Dataset(data_dir=dir_data_valid,
                                transform=transform_valid,  # idx=idx_valid,
                                params=params.copy(), nStage=self.nstage,
                                noise_range=self.noise_range, downsample_range=self.downsample_range,
                                input_type=self.input_type, label_type=self.label_type,
                                flip=0.0, weight=WGT, mode='valid')

        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True,
                                                   num_workers=self.num_workers, drop_last=True)
        loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=self.batch_size, shuffle=True,
                                                   num_workers=self.num_workers, drop_last=True)

        num_train = len(dataset_train)
        num_valid = len(dataset_valid)

        num_batch_train = int((num_train / self.batch_size) + ((num_train % self.batch_size) != 0))
        num_batch_valid = int((num_valid / self.batch_size) + ((num_valid % self.batch_size) != 0))

        ## setup network
        # net_prj1 = UNet_res(in_chans=self.nch_in, out_chans=self.nch_out, chans=self.num_channels,
        #                     num_pool_layers=self.num_block, num_upscale=0, factor_upscale=[0, ])
        net_prj1 = UNet_res(in_chans=self.nch_in, out_chans=self.nch_out, chans=self.num_channels,
                            num_pool_layers=self.num_block, num_upscale=0, factor_upscale=[0, ])

        net_img2 = UNet_res(in_chans=self.nch_in, out_chans=self.nch_out, chans=self.num_channels,
                            num_pool_layers=self.num_block, num_upscale=0, factor_upscale=[0, ])

        net_bp_dec = Backprojection(params=params_dec)

        if torch.cuda.is_available():
            net_prj1 = net_prj1.to(self.device)
            net_img2 = net_img2.to(self.device)
            net_bp_dec = net_bp_dec.to(self.device)

        summary(net_prj1, (1, params_dec['nView'], params_dec['nDctX']))
        summary(net_img2, (1, params_dec['nImgY'], params_dec['nImgX']))

        if self.nstage == 0:
            init_weights(net_prj1, init_type='normal', init_gain=0.02)
            init_weights(net_img2, init_type='normal', init_gain=0.02)
        else:
            _scope = self.scope.split('_stage')[0] + '_stage0' + '_loss_%s' % self.loss_type_prj
            _dir_checkpoint = os.path.join(self.dir_checkpoint, _scope)
            net_prj1, net_img2, _, _, _ = self.load_dual(_dir_checkpoint, net_prj1, net_img2)

        ## setup optimizer
        params_prj1 = net_prj1.parameters()
        params_img2 = net_img2.parameters()

        params_prj2img = [
            {'params': params_prj1},
            {'params': params_img2}
        ]

        optim_prj2img = torch.optim.Adam(params_prj2img, lr=self.lr_prj, weight_decay=self.wd_prj)

        sched_prj2img = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_prj2img, patience=5, verbose=True)

        ## setup loss functions
        loss_l1 = nn.L1Loss()
        loss_l2 = nn.MSELoss()
        if torch.cuda.is_available():
            loss_l1 = loss_l1.to(self.device)
            loss_l2 = loss_l2.to(self.device)

        ## load from checkpoints
        st_epoch = 0

        if self.train_continue == 'on':
            net_prj1, net_img2, optim_prj2img, _, st_epoch = self.load_dual(dir_checkpoint, net_prj1, net_img2,
                                                                            optim1=optim_prj2img, optim2=optim_prj2img)

        if torch.cuda.is_available():
            net_prj1 = net_prj1.to(self.device)
            net_img2 = net_img2.to(self.device)
            net_bp_dec = net_bp_dec.to(self.device)

        ## setup tensorboard
        writer_train = SummaryWriter(log_dir=dir_log_train)
        writer_valid = SummaryWriter(log_dir=dir_log_valid)


        ## setup training loop using epoch
        for epoch in range(st_epoch + 1, self.num_epoch + 1):

            ## training phase
            net_prj1.train()
            net_img2.train()
            net_bp_dec.eval()

            loss_prj1_train = []
            loss_img2_train = []
            loss_prj2img_train = []

            for i, data in enumerate(loader_train):
                def should(freq):
                    return freq > 0 and (i % freq == 0 or i == num_batch_train - 1)

                # get input datasets
                label_fbp = data['label_fbp']
                input_fbp = data['input_fbp']
                label_flt = data['label_flt']
                input_flt = data['input_flt']
                input_mask = data['input_mask']

                # reshape (-1, 1, H, W)
                [BI, CI, HI, WI] = list(label_fbp.shape)
                label_fbp = label_fbp.reshape(-1, 1, HI, WI)
                input_fbp = input_fbp.reshape(-1, 1, HI, WI)

                [BP, CP, HP, WP] = list(label_flt.shape)
                label_flt = label_flt.reshape(-1, 1, HP, WP)
                input_flt = input_flt.reshape(-1, 1, HP, WP)
                input_mask = input_mask.reshape(-1, 1, HP, WP)

                if torch.cuda.is_available():
                    label_fbp = label_fbp.to(self.device)
                    input_fbp = input_fbp.to(self.device)
                    input_flt = input_flt.to(self.device)
                    input_mask = input_mask.to(self.device)

                # initialize optimizer
                optim_prj2img.zero_grad()

                # FORWARD net-img1
                if self.loss_type_prj == 'img':
                    output_prj1 = net_prj1(input_flt.detach())

                    if self.lr_type_prj == 'consistency':
                        output_prj1 = input_mask * input_flt + (1 - input_mask) * output_prj1

                    # BACKPROJECTION
                    output_img1 = net_bp_dec(output_prj1)

                    # Residual or Plain
                    if self.lr_type_prj == 'residual':
                        output_img1 = input_fbp.detach() - output_img1

                    # BACKWARD net-img
                    loss_prj1 = loss_l2(output_img1, label_fbp)


                # FORWARD net-img2
                if self.loss_type_img == 'img':
                    output_img2 = net_img2(output_img1)

                    # Residual or Plain
                    if self.lr_type_img == 'residual':
                        output_img2 = output_img1.detach() - output_img2

                    # BACKWARD net-img
                    loss_img2 = loss_l2(output_img2, label_fbp)


                loss_prj2img = loss_prj1 + loss_img2

                # GET losses
                loss_prj2img.backward()
                optim_prj2img.step()
                loss_prj1_train += [loss_prj1.item()]
                loss_img2_train += [loss_img2.item()]
                loss_prj2img_train += [loss_prj2img.item()]

                print(
                    'TRAIN: EPOCH %d/%d: BATCH %04d/%04d | LR (P1I2) %.2e | LOSS (P1) %.4e | LOSS (I2) %.4e | LOSS (P1I2) %.4e'
                    % (epoch, self.num_epoch, i, num_batch_train, optim_prj2img.param_groups[0]['lr'],
                       np.mean(loss_prj1_train), np.mean(loss_img2_train), np.mean(loss_prj2img_train)))

                # display
                if should(self.num_freq_disp):
                    # show output
                    input_fbp = tensor2numpy(data['input_fbp'].cpu())
                    output_img1 = tensor2numpy(output_img1.reshape(BI, CI, HI, WI).cpu())
                    output_img2 = tensor2numpy(output_img2.reshape(BI, CI, HI, WI).cpu())
                    label_fbp = tensor2numpy(data['label_fbp'].cpu())

                    name = num_batch_train * (epoch - 1) + i

                    _label_fbp = []
                    _input_fbp = []
                    _output_img1 = []
                    _output_img2 = []

                    for j in range(label_fbp.shape[0]):
                        _output_img1.append(comp_img(output_img1[j], self.nstage, self.params) / np.max(label_fbp[j]))
                        _output_img2.append(comp_img(output_img2[j], self.nstage, self.params) / np.max(label_fbp[j]))
                        _input_fbp.append(comp_img(input_fbp[j], self.nstage, self.params) / np.max(label_fbp[j]))
                        _label_fbp.append(comp_img(label_fbp[j], self.nstage, self.params) / np.max(label_fbp[j]))

                        _output_img1[j] = np.clip(_output_img1[j], 0, 1)
                        _output_img2[j] = np.clip(_output_img2[j], 0, 1)
                        _input_fbp[j] = np.clip(_input_fbp[j], 0, 1)
                        _label_fbp[j] = np.clip(_label_fbp[j], 0, 1)


                    writer_train.add_images('input_fbp', np.array(_input_fbp), name, dataformats='NCHW')
                    writer_train.add_images('output_img1', np.array(_output_img1), name, dataformats='NCHW')
                    writer_train.add_images('output_img2', np.array(_output_img2), name, dataformats='NCHW')
                    writer_train.add_images('label_fbp', np.array(_label_fbp), name, dataformats='NCHW')


            writer_train.add_scalar('loss_prj1', np.mean(loss_prj1_train), epoch)
            writer_train.add_scalar('loss_img2', np.mean(loss_img2_train), epoch)
            writer_train.add_scalar('loss_prj2img', np.mean(loss_prj2img_train), epoch)

            # wandb.log({'train/loss_img1': np.mean(loss_img1_train)}, step=epoch)

            ## save
            if (epoch % self.num_freq_save) == 0:
                self.save_dual(dir_checkpoint, net_prj1, net_img2, optim_prj2img, optim_prj2img, epoch)

            ## validation phase
            with torch.no_grad():
                net_prj1.eval()
                net_img2.eval()
                net_bp_dec.eval()

                loss_prj1_valid = []
                loss_img2_valid = []
                loss_prj2img_valid = []

                for i, data in enumerate(loader_valid):
                    def should(freq):
                        return freq > 0 and (i % freq == 0 or i == num_batch_valid - 1)

                    # get input datasets
                    label_fbp = data['label_fbp']
                    input_fbp = data['input_fbp']
                    label_flt = data['label_flt']
                    input_flt = data['input_flt']
                    input_mask = data['input_mask']

                    # reshape (-1, 1, H, W)
                    [BI, CI, HI, WI] = list(label_fbp.shape)
                    label_fbp = label_fbp.reshape(-1, 1, HI, WI)
                    input_fbp = input_fbp.reshape(-1, 1, HI, WI)

                    [BP, CP, HP, WP] = list(label_flt.shape)
                    label_flt = label_flt.reshape(-1, 1, HP, WP)
                    input_flt = input_flt.reshape(-1, 1, HP, WP)
                    input_mask = input_mask.reshape(-1, 1, HP, WP)

                    if torch.cuda.is_available():
                        label_fbp = label_fbp.to(self.device)
                        input_fbp = input_fbp.to(self.device)
                        input_flt = input_flt.to(self.device)
                        input_mask = input_mask.to(self.device)

                    # FORWARD net-prj1
                    if self.loss_type_prj == 'img':
                        output_prj1 = net_prj1(input_flt.detach())

                        if self.lr_type_prj == 'consistency':
                            output_prj1 = input_mask * input_flt + (1 - input_mask) * output_prj1

                        # BACKPROJECTION
                        output_img1 = net_bp_dec(output_prj1)

                        # Residual or Plain
                        if self.lr_type_prj == 'residual':
                            output_img1 = input_fbp.detach() - output_img1

                        # BACKWARD net-img
                        loss_prj1 = loss_l2(output_img1, label_fbp)

                    # FORWARD net-img2
                    if self.loss_type_prj == 'img':
                        output_img2 = net_img2(output_img1)

                        # Residual or Plain
                        if self.lr_type_img == 'residual':
                            output_img2 = output_img1.detach() - output_img2

                        # BACKWARD net-img
                        loss_img2 = loss_l2(output_img2, label_fbp)

                    loss_prj2img = loss_prj1 + loss_img2

                    # GET losses
                    loss_prj1_valid += [loss_prj1.item()]
                    loss_img2_valid += [loss_img2.item()]
                    loss_prj2img_valid += [loss_prj2img.item()]

                    print(
                        'VALID: EPOCH %d/%d: BATCH %04d/%04d | LR (P1I2) %.2e | LOSS (P1) %.4e | LOSS (I2) %.4e | LOSS (P1I2) %.4e'
                        % (epoch, self.num_epoch, i, num_batch_valid, optim_prj2img.param_groups[0]['lr'],
                           np.mean(loss_prj1_valid), np.mean(loss_img2_valid), np.mean(loss_prj2img_valid)))

                    # display
                    if should(self.num_freq_disp):
                        # show output
                        input_fbp = tensor2numpy(data['input_fbp'].cpu())
                        output_img1 = tensor2numpy(output_img1.reshape(BI, CI, HI, WI).cpu())
                        output_img2 = tensor2numpy(output_img2.reshape(BI, CI, HI, WI).cpu())
                        label_fbp = tensor2numpy(data['label_fbp'].cpu())

                        name = num_batch_valid * (epoch - 1) + i

                        _label_fbp = []
                        _input_fbp = []
                        _output_img1 = []
                        _output_img2 = []

                        for j in range(label_fbp.shape[0]):
                            _output_img1.append(
                                comp_img(output_img1[j], self.nstage, self.params) / np.max(label_fbp[j]))
                            _output_img2.append(
                                comp_img(output_img2[j], self.nstage, self.params) / np.max(label_fbp[j]))
                            _input_fbp.append(comp_img(input_fbp[j], self.nstage, self.params) / np.max(label_fbp[j]))
                            _label_fbp.append(comp_img(label_fbp[j], self.nstage, self.params) / np.max(label_fbp[j]))

                            _output_img1[j] = np.clip(_output_img1[j], 0, 1)
                            _output_img2[j] = np.clip(_output_img2[j], 0, 1)
                            _input_fbp[j] = np.clip(_input_fbp[j], 0, 1)
                            _label_fbp[j] = np.clip(_label_fbp[j], 0, 1)

                            # example_images.append(wandb.Image(np.concatenate((label_fbp[j][0], input_fbp[j][0], output_img1[j][0]), axis=1)[:, :, np.newaxis], caption='Iter = %d' % name))

                        writer_valid.add_images('input_fbp', np.array(_input_fbp), name, dataformats='NCHW')
                        writer_valid.add_images('output_img1', np.array(_output_img1), name, dataformats='NCHW')
                        writer_valid.add_images('output_img2', np.array(_output_img2), name, dataformats='NCHW')
                        writer_valid.add_images('label_fbp', np.array(_label_fbp), name, dataformats='NCHW')

                        # wandb.log({'valid/results_img1': example_images}, step=epoch)

                writer_valid.add_scalar('loss_prj1', np.mean(loss_prj1_valid), epoch)
                writer_valid.add_scalar('loss_img2', np.mean(loss_img2_valid), epoch)
                writer_valid.add_scalar('loss_prj2img', np.mean(loss_prj2img_valid), epoch)

            # update schduler
            if sched_prj2img is not None:
                sched_prj2img.step(np.mean(loss_prj2img_valid))

        writer_train.close()
        writer_valid.close()

        # wandb.finish()

    ## Test function
    def test(self):
        ## CT parameter
        params = self.params
        params_dec = self.params_dec

        ## setup directories
        dir_checkpoint = os.path.join(self.dir_checkpoint, self.scope)
        os.makedirs(dir_checkpoint, exist_ok=True)

        dir_result_test = os.path.join(self.dir_result, self.scope, 'test', 'stage%d' % self.nstage, 'ds%d' % self.downsample)
        os.makedirs(os.path.join(dir_result_test, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dir_result_test, 'numpy'), exist_ok=True)

        name_data = '%s_dct%d_view%d' % (params['CT_NAME'], params['nDctX'], params['nView']) + params['scope_data']

        dir_data_test = os.path.join(self.dir_data, name_data, 'dec%d' % 0, 'test')

        ## setup transform functions
        transform_test = transforms.Compose([Converter(dir='numpy2tensor'), ])
        
        tensor2numpy = Converter(dir='tensor2numpy')

        ## setup dataset loaders about train and validation data
        dataset_test = Dataset(data_dir=dir_data_test,
                               transform=transform_test,  # idx=idx_train,
                               params=params.copy(), nStage=self.nstage,
                               noise_range=self.noise_range, downsample_range=[self.downsample, ],
                               input_type=self.input_type, label_type=self.label_type,
                               flip=0.0, weight=WGT, mode='test')

        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)

        num_test = len(dataset_test)

        num_batch_test = int((num_test / self.batch_size) + ((num_test % self.batch_size) != 0))

        ## setup network
        net_prj1 = UNet_res(in_chans=self.nch_in, out_chans=self.nch_out, chans=self.num_channels,
                            num_pool_layers=self.num_block, num_upscale=0, factor_upscale=[0, ])

        net_img2 = UNet_res(in_chans=self.nch_in, out_chans=self.nch_out, chans=self.num_channels,
                            num_pool_layers=self.num_block, num_upscale=0, factor_upscale=[0, ])


        net_bp_dec = Backprojection(params=params_dec)
        if torch.cuda.is_available():
            net_prj1 = net_prj1.to(self.device)
            net_img2 = net_img2.to(self.device)
            net_bp_dec = net_bp_dec.to(self.device)

        summary(net_prj1, (1, params_dec['nView'], params_dec['nDctX']))
        summary(net_img2, (1, params_dec['nImgY'], params_dec['nImgX']))

        if self.nstage == 0:
            init_weights(net_prj1, init_type='normal', init_gain=0.02)
            init_weights(net_img2, init_type='normal', init_gain=0.02)
        else:
            _scope = self.scope.split('_stage')[0] + '_stage0' + '_loss_%s' % self.loss_type_prj
            _dir_checkpoint = os.path.join(self.dir_checkpoint, _scope)
            net_prj1, net_img2, _, _, _ = self.load_dual(_dir_checkpoint, net_prj1, net_img2)

        ## setup optimizer
        params_prj1 = net_prj1.parameters()
        params_img2 = net_img2.parameters()


        params_prj2img = [
            {'params': params_prj1},
            {'params': params_img2}
        ]

        optim_prj2img = torch.optim.Adam(params_prj2img, lr=self.lr_prj, weight_decay=self.wd_prj)
        sched_prj2img = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_prj2img, patience=5, verbose=True)

        ## setup loss functions
        loss_l1 = nn.L1Loss()
        loss_l2 = nn.MSELoss()
        if torch.cuda.is_available():
            loss_l1 = loss_l1.to(self.device)
            loss_l2 = loss_l2.to(self.device)

        ## load from checkpoints
        st_epoch = 0

        # if self.train_continue == 'on':
        net_prj1, net_img2, optim_prj2img, _, st_epoch = self.load_dual(dir_checkpoint, net_prj1, net_img2,
                                                                        optim1=optim_prj2img, optim2=optim_prj2img)


        if torch.cuda.is_available():
            net_prj1 = net_prj1.to(self.device)
            net_img2 = net_img2.to(self.device)
            net_bp_dec = net_bp_dec.to(self.device)

        ## test phase
        with torch.no_grad():
            ## training phase
            net_prj1.eval()
            net_img2.eval()
            net_bp_dec.eval()

            loss_prj1_test = []
            loss_img2_test = []
            loss_prj2img_test = []

            cnt = 0
            for i, data in enumerate(loader_test):
                def should(freq):
                    return freq > 0 and (i % freq == 0 or i == num_batch_test - 1)

                # get input datasets
                label_fbp = data['label_fbp']
                input_fbp = data['input_fbp']
                label_flt = data['label_flt']
                input_flt = data['input_flt']
                input_mask = data['input_mask']

                # reshape (-1, 1, H, W)
                [BI, CI, HI, WI] = list(label_fbp.shape)
                label_fbp = label_fbp.reshape(-1, 1, HI, WI)
                input_fbp = input_fbp.reshape(-1, 1, HI, WI)

                [BP, CP, HP, WP] = list(label_flt.shape)
                label_flt = label_flt.reshape(-1, 1, HP, WP)
                input_flt = input_flt.reshape(-1, 1, HP, WP)
                input_mask = input_mask.reshape(-1, 1, HP, WP)

                if torch.cuda.is_available():
                    label_fbp = label_fbp.to(self.device)
                    input_fbp = input_fbp.to(self.device)
                    input_flt = input_flt.to(self.device)
                    input_mask = input_mask.to(self.device)

                # FORWARD net-prj1
                if self.loss_type_prj == 'img':
                    output_prj1 = net_prj1(input_flt.detach())

                    if self.lr_type_prj == 'consistency':
                        output_prj1 = input_mask * input_flt + (1 - input_mask) * output_prj1

                    # BACKPROJECTION
                    output_img1 = net_bp_dec(output_prj1)

                    # Residual or Plain
                    if self.lr_type_prj == 'residual':
                        output_img1 = input_fbp.detach() - output_img1

                    # BACKWARD net-img
                    loss_prj1 = loss_l2(output_img1, label_fbp)


                # FORWARD net-img2
                if self.loss_type_prj == 'img':
                    output_img2 = net_img2(output_img1)

                    # Residual or Plain
                    if self.lr_type_img == 'residual':
                        output_img2 = output_img1.detach() - output_img2

                    # BACKWARD net-img
                    loss_img2 = loss_l2(output_img2, label_fbp)

                loss_prj2img = loss_prj1 + loss_img2
                # GET losses

                loss_prj1_test += [loss_prj1.item()]
                loss_img2_test += [loss_img2.item()]
                loss_prj2img_test += [loss_prj2img.item()]

                print(
                    'TEST: BATCH %04d/%04d | LR (P1I2) %.2e | LOSS (P1) %.4e | LOSS (I2) %.4e | LOSS (P1I2) %.4e'
                    % (i, num_batch_test, optim_prj2img.param_groups[0]['lr'], np.mean(loss_prj1_test), np.mean(loss_img2_test), np.mean(loss_prj2img_test)))

                # display
                # if should(self.num_freq_disp):
                if True:
                    # show output
                    input_fbp = tensor2numpy(data['input_fbp'].cpu())
                    output_img1 = tensor2numpy(output_img1.reshape(BI, CI, HI, WI).cpu())
                    output_img2 = tensor2numpy(output_img2.reshape(BI, CI, HI, WI).cpu())
                    label_fbp = tensor2numpy(data['label_fbp'].cpu())

                    name = i

                    _label_fbp = []
                    _input_fbp = []
                    _output_img1 = []
                    _output_img2 = []

                    for j in range(label_fbp.shape[0]):
                        _output_img1.append(comp_img(output_img1[j], self.nstage, self.params))
                        _output_img2.append(comp_img(output_img2[j], self.nstage, self.params))
                        _input_fbp.append(comp_img(input_fbp[j], self.nstage, self.params))
                        _label_fbp.append(comp_img(label_fbp[j], self.nstage, self.params))


                        if self.nstage == 0:
                            plt.imshow(_label_fbp[j][0], vmin=0, vmax=0.04, cmap='gray')
                            plt.axis(False)
                            plt.tight_layout()
                            plt.savefig(os.path.join(dir_result_test, 'images', 'label_%04d.png' % cnt))
                            plt.close()

                            plt.imshow(_input_fbp[j][0], vmin=0, vmax=0.04, cmap='gray')
                            plt.axis(False)
                            plt.tight_layout()
                            plt.savefig(os.path.join(dir_result_test, 'images', 'input_%04d.png' % cnt))
                            plt.close()

                        plt.imshow(_output_img1[j][0], vmin=0, vmax=0.04, cmap='gray')
                        plt.axis(False)
                        plt.tight_layout()
                        plt.savefig(os.path.join(dir_result_test, 'images', 'output_img1_%04d.png' % cnt))
                        plt.close()

                        plt.imshow(_output_img2[j][0], vmin=0, vmax=0.04, cmap='gray')
                        plt.axis(False)
                        plt.tight_layout()
                        plt.savefig(os.path.join(dir_result_test, 'images', 'output_img2_%04d.png' % cnt))
                        plt.close()

                        if self.nstage == 0:
                            np.save(os.path.join(dir_result_test, 'numpy', 'label_%04d' % cnt), _label_fbp[j][0])
                            np.save(os.path.join(dir_result_test, 'numpy', 'input_%04d' % cnt), _input_fbp[j][0])
                        np.save(os.path.join(dir_result_test, 'numpy', 'output_img1_%04d' % cnt), _output_img1[j][0])
                        np.save(os.path.join(dir_result_test, 'numpy', 'output_img2_%04d' % cnt), _output_img2[j][0])

                        cnt += 1
        return

    def save_single(self, dir_checkpoint, net1, optim1, epoch):
        if not os.path.exists(dir_checkpoint):
            os.makedirs(dir_checkpoint)

        if self.gpu_ids:
            torch.save({'net1': net1.module.state_dict(),
                        'optim1': optim1.state_dict()},
                       '%s/model_epoch%04d.pth' % (dir_checkpoint, epoch))
        else:
            torch.save({'net1': net1.state_dict(),
                        'optim1': optim1.state_dict()},
                       '%s/model_epoch%04d.pth' % (dir_checkpoint, epoch))

    def load_single(self, dir_checkpoint, net1, optim1=None, epoch=None):
        if not os.path.exists(dir_checkpoint):
            epoch = 0
            return net1, optim1, epoch

        if epoch is None:
            ckpt = os.listdir(dir_checkpoint)
            ckpt.sort()

            if len(ckpt) == 0:
                epoch = 0
                return net1, optim1, epoch

            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_checkpoint, epoch), map_location=self.device)

        print('Loaded %dth network' % epoch)

        net1.load_state_dict(dict_net['net1'])
        if epoch:
            if optim1 is not None:
                optim1.load_state_dict(dict_net['optim1'])

        return net1, optim1, epoch

    def save_dual(self, dir_checkpoint, net1, net2, optim1, optim2, epoch):
        if not os.path.exists(dir_checkpoint):
            os.makedirs(dir_checkpoint)

        if self.gpu_ids:
            torch.save({'net1': net1.module.state_dict(),
                        'net2': net2.module.state_dict(),
                        'optim1': optim1.state_dict(),
                        'optim2': optim2.state_dict()},
                       '%s/model_epoch%04d.pth' % (dir_checkpoint, epoch))
        else:
            torch.save({'net1': net1.state_dict(),
                        'net2': net2.state_dict(),
                        'optim1': optim1.state_dict(),
                        'optim2': optim2.state_dict()},
                       '%s/model_epoch%04d.pth' % (dir_checkpoint, epoch))

    def load_dual(self, dir_checkpoint, net1, net2, optim1=None, optim2=None, epoch=None):
        if not os.path.exists(dir_checkpoint):
            epoch = 0
            return net1, net2, optim1, optim2, epoch

        if epoch is None:
            ckpt = os.listdir(dir_checkpoint)
            ckpt.sort()

            if len(ckpt) == 0:
                epoch = 0
                return net1, net2, optim1, optim2, epoch

            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_checkpoint, epoch), map_location=self.device)

        print('Loaded %dth network' % epoch)

        net1.load_state_dict(dict_net['net1'])
        net2.load_state_dict(dict_net['net2'])
        if epoch:
            if optim1 is not None:
                optim1.load_state_dict(dict_net['optim1'])
            if optim2 is not None:
                optim2.load_state_dict(dict_net['optim2'])

        return net1, net2, optim1, optim2, epoch


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler
