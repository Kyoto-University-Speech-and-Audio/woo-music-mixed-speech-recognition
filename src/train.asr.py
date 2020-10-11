import argparse

import torch

from data import AudioASRDataLoader, AudioASRDataset
from solver_asr import Solver
from attention import AttentionModel


parser = argparse.ArgumentParser(
    "Fully-Convolutional Time-domain Audio Separation Network (Conv-TasNet) "
    "with Permutation Invariant Training")
# General config
# Task related
parser.add_argument('--train_dir', type=str, default=None,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--sample_rate', default=16000, type=int,
                    help='Sample rate')
# Network architecture
parser.add_argument('--LMFB_DIM', default=40, type=int,
                    help='Number of dimensions in log mel filter bank')
parser.add_argument('--NUM_HIDDEN_NODES', default=320, type=int,
                    help='Number of hidden nodes in bi-LSTM')
parser.add_argument('--NUM_ENC_LAYERS', default=5, type=int,
                    help='Number of layers in Encoder')
parser.add_argument('--NUM_CLASSES', default=9515, type=int,
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
parser.add_argument('--save_folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=1, type=int,
                    help='Enables checkpoint saving of model')
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
    tr_dataset = AudioASRDataset(args.train_dir, args.batch_size,
                              sample_rate=args.sample_rate)

    tr_loader = AudioASRDataLoader(tr_dataset, batch_size=1,
                                num_workers=args.num_workers)

    data = {'tr_loader': tr_loader, 'ev_loader' : None}
    # model
    model = AttentionModel(args.NUM_HIDDEN_NODES, args.NUM_ENC_LAYERS, args.NUM_CLASSES)
    #print(model)
    if args.use_cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()
    # optimizer
    if args.optimizer == 'sgd':
        optimizier = torch.optim.SGD(model.parameters(),
                                     lr=args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.l2)
    elif args.optimizer == 'adam':
        optimizier = torch.optim.Adam(model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.l2)
    else:
        print("Not support optimizer")
        return

    # solver
    solver = Solver(data, model, optimizier, args, DEVICE)
    solver.train()


if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(args)
    main(args)

