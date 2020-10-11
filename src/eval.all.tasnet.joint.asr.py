import argparse

import torch

from data import EvalAllDataLoader, EvalAllDataset
from solver_all_tasnet_joint_asr import Solver
from conv_tasnet import ConvTasNet
from attention import AttentionModel


parser = argparse.ArgumentParser(
    "Fully-Convolutional Time-domain Audio Separation Network (Conv-TasNet) "
    "with Permutation Invariant Training")
# General config
# Task related
parser.add_argument('--train_dir', type=str, default=None,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--mix_json', type=str, default=None,
                    help='')
parser.add_argument('--sample_rate', default=16000, type=int,
                    help='Sample rate')

# SEP Network architecture
parser.add_argument('--N', default=256, type=int,
                    help='Number of filters in autoencoder')
parser.add_argument('--L', default=20, type=int,
                    help='Length of the filters in samples (40=5ms at 8kHZ)')
parser.add_argument('--B', default=256, type=int,
                    help='Number of channels in bottleneck 1 × 1-conv block')
parser.add_argument('--H', default=512, type=int,
                    help='Number of channels in convolutional blocks')
parser.add_argument('--P', default=3, type=int,
                    help='Kernel size in convolutional blocks')
parser.add_argument('--X', default=8, type=int,
                    help='Number of convolutional blocks in each repeat')
parser.add_argument('--R', default=4, type=int,
                    help='Number of repeats')
parser.add_argument('--C', default=2, type=int,
                    help='Number of speakers')
parser.add_argument('--norm_type', default='gLN', type=str,
                    choices=['gLN', 'cLN', 'BN'], help='Layer norm type')
parser.add_argument('--causal', type=int, default=0,
                    help='Causal (1) or noncausal(0) training')
parser.add_argument('--mask_nonlinear', default='relu', type=str,
                    choices=['relu', 'softmax'], help='non-linear to generate mask')

# ASR Network architecture
parser.add_argument('--LMFB_DIM', default=40, type=int,
                    help='Number of dimensions in log mel filter bank')
parser.add_argument('--NUM_HIDDEN_NODES', default=320, type=int,
                    help='Number of hidden nodes in bi-LSTM')
parser.add_argument('--NUM_ENC_LAYERS', default=5, type=int,
                    help='Number of layers in Encoder')
parser.add_argument('--NUM_CLASSES', default=2316, type=int,
                    help='Number of classes in utterences')
parser.add_argument('--EOS_ID', default=1, type=int,
                    help='Number of classes in utterences')

# Training config
parser.add_argument('--use_cuda', type=int, default=1,
                    help='Whether use GPU')
parser.add_argument('--epochs', default=40, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--lr_decay', dest='lr_decay', default=0, type=int,
                    help='reduce learning rate when over 20 epochs')
parser.add_argument('--early_stop', dest='early_stop', default=0, type=int,
                    help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')

parser.add_argument('--lambda_si_snr', type=float, default=1,
                    help='')

parser.add_argument('--lambda_asr', type=float, default=2,
                    help='')



# minibatch
parser.add_argument('--batch_size', default=40, type=int,
                    help='Batch size')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers to generate minibatch')
# optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    choices=['sgd', 'adam'],
                    help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Init learning rate')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--l2', default=0.0, type=float,
                    help='weight decay (L2 penalty)')

# save and load model
parser.add_argument('--out_dir', type=str, default='exp/result',
                    help='Directory putting separated wav files')
parser.add_argument('--save_folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=1, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--sep_model_from', default='',
                    help='Pretrained seperate model')
parser.add_argument('--asr_model_from', default='',
                    help='Pretrained asr model')
parser.add_argument('--continue_from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model_path', default='final.pth.tar',
                    help='Location to save best validation model')

# logging
parser.add_argument('--print_freq', default=10, type=int,
                    help='Frequency of printing training infomation')

def main(args):
    # Construct Solver
    # data

    if args.continue_from == '':
        return
    ev_dataset = EvalAllDataset(args.train_dir, args.mix_json, args.batch_size, sample_rate=args.sample_rate)

    ev_loader = EvalAllDataLoader(ev_dataset, batch_size=1,
                                num_workers=args.num_workers)

    data = {'tr_loader': None, 'ev_loader': ev_loader}
    # SEP model
    sep_model = ConvTasNet(args.N, args.L, args.B, args.H, args.P, args.X, args.R,
                       args.C, norm_type=args.norm_type, causal=args.causal,
                       mask_nonlinear=args.mask_nonlinear)

    # ASR model
    asr_model = AttentionModel(args.NUM_HIDDEN_NODES, args.NUM_ENC_LAYERS, args.NUM_CLASSES)
    #print(model)
    if args.use_cuda:
        sep_model = torch.nn.DataParallel(sep_model)
        asr_model = torch.nn.DataParallel(asr_model)
        sep_model.cuda()
        asr_model.cuda()
    # optimizer
    if args.optimizer == 'sgd':
        sep_optimizier = torch.optim.SGD(sep_model.parameters(),
                                     lr=args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.l2)
        asr_optimizier = torch.optim.SGD(asr_model.parameters(),
                                     lr=args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.l2)
    elif args.optimizer == 'adam':
        sep_optimizier = torch.optim.Adam(sep_model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.l2)
        asr_optimizier = torch.optim.Adam(asr_model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.l2)
    else:
        print("Not support optimizer")
        return

    # solver
    solver = Solver(data, sep_model, asr_model, sep_optimizier, asr_optimizier, args, DEVICE, ev=True)
    solver.eval(args.EOS_ID)

if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    #print(args)
    main(args)

