import os
import time
import sys
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from criterion import cal_loss


class Solver(object):
    
    def __init__(self, data, model, optimizer, args, DEVICE, ev=False):
        self.ev = ev
        self.tr_loader = data['tr_loader']
        self.ev_loader = data['ev_loader']
        self.model = model
        self.optimizer = optimizer
        self.DEVICE = DEVICE
        # Training config
        self.use_cuda = args.use_cuda
        self.epochs = args.epochs
        self.lr_decay = args.lr_decay
        self.lr = args.lr
        self.early_stop = args.early_stop
        self.max_norm = args.max_norm
        self.batch_size = args.batch_size
        self.NUM_CLASSES = args.NUM_CLASSES
        # fourier transform
        self.n_fft = 400
        self.hop_length = 160

        # lmfb
        self.lmfb = torch.tensor(self._lmfb(args.sample_rate, args.LMFB_DIM, self.n_fft), dtype=torch.float32, device=self.DEVICE, requires_grad=False)
        # save and load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.model_path = args.model_path
        # logging
        self.print_freq = args.print_freq
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)

        self._reset()

    def _reset(self):
        # Reset
        if self.continue_from:
            package = torch.load(self.continue_from)
            self.model.module.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start_epoch = int(package.get('epoch', 1)) if not self.ev else 0
            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0
        if self.ev:
            return
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.halving = False

    def eval(self, EOS_ID):
        self.model.eval()
        data_loader = self.ev_loader
        for k, (data) in enumerate(data_loader):
            source, file_name = data
            file_name = file_name[0]
            torchsource = torch.from_numpy(source[0]).to(self.DEVICE).float()
            source_stft = torch.stft(torchsource, self.n_fft, self.hop_length,
                window=torch.hann_window(self.n_fft+1, device=self.DEVICE)[:-1],
                center=False)
            source_stft_abs = torch.sqrt(torch.sum(source_stft**2, dim=2) + 1e-8).t()
            source_lmfb = (torch.matmul(source_stft_abs, self.lmfb.t()) + 1e-8).log()
            torchdat = source_lmfb.unsqueeze(0)
            prediction_beam = self.model.module.evaluate(torchdat, self.DEVICE, EOS_ID)
            file_name = file_name.replace(".wav", ".htk").replace("_s1", "")
            print(file_name, end=" ")
            if len(prediction_beam) > 0:
                best_prediction = prediction_beam[0][0]
                for character in best_prediction:
                    print(character.item(), end=" ")
            print()
            sys.stdout.flush()


    def train(self):
        now = datetime.now()

        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):

            # Adjust learning rate
            if self.lr_decay and epoch >= 20 and epoch < 40:
                self.lr = self.lr * 0.8
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
                print('Learning rate adjusted to: {lr:.6f}'.format(lr=self.lr))

            # Train one epoch
            print("Training...")
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | Total Time {2} | '
                  'Train Loss {3:.3f}'.format(epoch + 1, time.time() - start, datetime.now() - now, tr_avg_loss))
            print('-' * 85)


            self.tr_loss[epoch] = tr_avg_loss
            # Save model each epoch
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save(self.model.module.serialize(self.model.module,
                                                       self.optimizer, epoch + 1,
                                                       tr_loss=self.tr_loss),
                                                       file_path)
                print('Saving checkpoint model to %s' % file_path)



            sys.stdout.flush()

    def _run_one_epoch(self, epoch):
        start = time.time()
        total_loss = 0

        data_loader = self.tr_loader

        for k, (data) in enumerate(data_loader):
            sources, targets = data

            sources = list(map(lambda x: torch.tensor(x, device=self.DEVICE, requires_grad=False).float(), sources))
            targets = list(map(lambda x: torch.tensor(x, device=self.DEVICE, requires_grad=False).long(), targets))
            # stft, B x n_fft/2 + 1 x T x 2
            sources_stft = list(map(lambda x: torch.stft(x, self.n_fft, self.hop_length,
                            window=torch.hann_window(self.n_fft+1, device=self.DEVICE)[:-1],
                            center=False), sources))
            # abs B x T x N(=201)
            sources_stft_abs = list(map(lambda x: torch.sqrt(torch.sum(x**2, dim=2) + 1e-8).t(), sources_stft))

            # lmfb B x T x L(=40)
            sources_lmfb = list(map(lambda x: (torch.matmul(x, self.lmfb.t()) + 1e-8).log(), sources_stft_abs))

            # mean = list(map(lambda x: torch.mean(x, dim=0).unsqueeze(0), sources_lmfb))
            # std = list(map(lambda x:torch.sqrt(torch.mean(x[0]**2, dim=0).unsqueeze(0) - x[1]**2 + 1e-6), zip(sources_lmfb, mean)))

            # sources_lmfb_nor = list(map(lambda x: (x[0]-x[1])/x[2], zip(sources_lmfb, mean, std)))
            sources_lmfb_nor = sources_lmfb

            sources_lengths = torch.tensor(np.array([len(x) for x in sources_lmfb_nor], dtype=np.int32), device=self.DEVICE, requires_grad=False)
            targets_lengths = torch.tensor(np.array([len(t) for t in targets], dtype=np.int32), device=self.DEVICE, requires_grad=False)

            padded_sources_lmfb_nor = nn.utils.rnn.pad_sequence(sources_lmfb_nor, batch_first=True)
            padded_targets = nn.utils.rnn.pad_sequence(targets, batch_first=True)

            sorted_sources_lengths, perm_index = sources_lengths.sort(0, descending=True)
            sorted_targets_lengths = targets_lengths[perm_index]

            padded_sorted_sources_lmfb_nor = padded_sources_lmfb_nor[perm_index]
            padded_sorted_targets = padded_targets[perm_index]

            prediction = self.model(padded_sorted_sources_lmfb_nor, sorted_sources_lengths, padded_sorted_targets, self.DEVICE)

            loss = 0.0
            for i in range(len(sources)):
                num_labels = sorted_targets_lengths[i]
                label = padded_sorted_targets[i, :num_labels]
                onehot_target = torch.zeros((len(label), self.NUM_CLASSES), dtype=torch.float32, device=self.DEVICE, requires_grad=False)
                for j in range(len(label)):
                    onehot_target[j][label[j]] = 1.0
                ls_target = 0.9 * onehot_target + ((1.0 - 0.9) / (self.NUM_CLASSES - 1)) * (1.0 - onehot_target)
                loss += -(F.log_softmax(prediction[i][:num_labels], dim=1) * ls_target).sum()


            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.max_norm)
            self.optimizer.step()

            total_loss += loss.item()

            if k % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                          epoch + 1, k + 1, total_loss / (k + 1),
                          loss.item(), 1000 * (time.time() - start) / (k + 1)),
                      flush=True)
                sys.stdout.flush()

            
        return total_loss / (k + 1)

    def _lmfb(self, sample_rate=16000, nch=40, fftsize=400):
        nbin = fftsize//2 + 1
        low_freq_mel = 0
        high_freq_mel = 2595 * np.log10(1+ (sample_rate/2) /700)
        melcenter = np.linspace(low_freq_mel, high_freq_mel, nch + 2)
        fcenter = 700 * (10**(melcenter / 2595) -1)
        fres = sample_rate / fftsize
        fbank = np.zeros((nch, nbin))
        for c in range(nch):
            # 三角窓の左側の直線の式を計算
            v1 = 1.0 / (fcenter[c + 1] - fcenter[c      ]) * fres * np.arange(nbin) - fcenter[c      ] / (fcenter[c + 1] - fcenter[c      ])
            # 三角窓の右側の直線の式を計算
            v2 = 1.0 / (fcenter[c + 1] - fcenter[c + 2]) * fres * np.arange(nbin) - fcenter[c + 2] / (fcenter[c + 1] - fcenter[c + 2])
            # 下側の直線を選択して、0 以上の部分のみを抽出
            fbank[c] = np.maximum(np.minimum(v1, v2), 0)
        return fbank
