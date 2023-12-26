import torch.backends.cudnn as cudnn
from src.train_prj2img import *
from src.utils.utils import *

import random

# set random seed
SEED = 1234

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

##
GPU_ID = '0'
NUM_WORKER = 0

## setup parse
parser = argparse.ArgumentParser(description='Sparse-view CT using FHBP',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# parser.add_argument('--mode', default='train', choices=['train', 'test'], dest='mode')
parser.add_argument('--mode', default='test', choices=['train', 'test'], dest='mode')
parser.add_argument('--train_continue', default='off', choices=['on', 'off'], dest='train_continue')

parser.add_argument('--scope', default='ct_prj2img', dest='scope')

parser.add_argument('--name_data', type=str, default='Sparse-view CT', dest='name_data')

parser.add_argument('--gpu_ids', default=GPU_ID, dest='gpu_ids')
parser.add_argument('--num_workers', default=0, dest='num_workers')


parser.add_argument('--dir_checkpoint', default='./checkpoints', dest='dir_checkpoint')
parser.add_argument('--dir_log', default='./logs', dest='dir_log')
parser.add_argument('--dir_result', default='./results', dest='dir_result')
parser.add_argument('--dir_data', default='[YOUR DATASET]', dest='dir_data')


parser.add_argument('--num_epoch', type=int, default=30, dest='num_epoch')
parser.add_argument('--batch_size', type=int, default=1, dest='batch_size')


parser.add_argument('--dir_dll', type=str, default='lib_fbp', dest='dir_dll')
parser.add_argument('--dir_project', type=str, default='projects', dest='dir_project')

parser.add_argument('--name_project', type=str, default='parallel', dest='name_project')

parser.add_argument('--is_consistency', type=str2bool, default='True', dest='is_consistency')

parser.add_argument('--lr_prj', type=float, default=1e-4, dest='lr_prj')
parser.add_argument('--lr_img', type=float, default=1e-4, dest='lr_img')

parser.add_argument('--lr_type_prj', type=str, default='consistency', choices=['residual', 'plain', 'consistency'], dest='lr_type_prj')
parser.add_argument('--lr_type_img', type=str, default='residual', choices=['residual', 'plain'], dest='lr_type_img')

parser.add_argument('--loss_type_prj', type=str, default='img', choices=['img', 'prj', 'both'], dest='loss_type_prj')
parser.add_argument('--loss_type_img', type=str, default='img', choices=['img', 'prj', 'both'], dest='loss_type_img')

parser.add_argument('--wd_prj', type=float, default=0e-4, dest='wd_prj')
parser.add_argument('--wd_img', type=float, default=0e-4, dest='wd_img')

parser.add_argument('--optim', default='adam', choices=['sgd', 'adam', 'rmsprop'], dest='optim')

parser.add_argument('--nstage', type=int, default=1, dest='nstage')

parser.add_argument('--downsample', type=int, default=3, dest='downsample')

parser.add_argument('--nch_in', type=int, default=1, dest='nch_in')
parser.add_argument('--nch_out', type=int, default=1, dest='nch_out')

parser.add_argument('--num_block', type=int, default=4, dest='num_block')
parser.add_argument('--num_channels', type=int, default=32, dest='num_channels')

parser.add_argument('--input_type', default='prj', dest='input_type')
parser.add_argument('--label_type', default='label', dest='label_type')

parser.add_argument('--num_freq_disp', type=int, default=50, dest='num_freq_disp')
parser.add_argument('--num_freq_save', type=int, default=1, dest='num_freq_save')

PARSER = Parser(parser)

##
seed_everything(SEED)

##
def main():
    ARGS = PARSER.get_arguments()
    # PARSER.write_args()
    PARSER.print_args()

    TRAINER = Train(ARGS)

    if ARGS.mode == 'train':
        TRAINER.train()
    elif ARGS.mode == 'test':
        TRAINER.test()

if __name__ == '__main__':
    main()
