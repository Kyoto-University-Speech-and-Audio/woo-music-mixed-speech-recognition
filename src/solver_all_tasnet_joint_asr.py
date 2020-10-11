# Created on 2018/12
# Author: Kaituo XU

import os
import time
import sys
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import remove_pad
from criterion import get_mask, cal_loss
import soundfile as sf

class Solver(object):
    
    def __init__(self, data, sep_model, asr_model, sep_optimizer, asr_optimizer, args, DEVICE, ev=False):
        self.ev = ev
        self.tr_loader = data['tr_loader']
        self.ev_loader = data['ev_loader']
        self.sep_model = sep_model
        self.asr_model = asr_model
        self.sep_optimizer = sep_optimizer
        self.asr_optimizer = asr_optimizer
        self.DEVICE = DEVICE
        self.sample_rate = args.sample_rate
        # Training config
        self.use_cuda = args.use_cuda
        self.epochs = args.epochs
        self.lr_decay = args.lr_decay
        self.lr = args.lr
        self.early_stop = args.early_stop
        self.max_norm = args.max_norm
        self.batch_size = args.batch_size
        self.NUM_CLASSES = args.NUM_CLASSES

        self.lambda_si_snr = args.lambda_si_snr
        self.lambda_asr = args.lambda_asr
        # fourier transform
        self.n_fft = 400
        self.hop_length = 160

        # lmfb
        self.lmfb = torch.tensor(self._lmfb(args.sample_rate, args.LMFB_DIM, self.n_fft), dtype=torch.float32, device=self.DEVICE, requires_grad=False)
        # save and load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.sep_model_from = args.sep_model_from
        self.asr_model_from = args.asr_model_from
        self.continue_from = args.continue_from
        self.model_path = args.model_path
        # logging
        self.print_freq = args.print_freq
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)

        self._reset()

    def _reset(self):
        # Reset
        if self.sep_model_from and self.asr_model_from:
            sep_package = torch.load(self.sep_model_from)
            asr_package = torch.load(self.asr_model_from)

            self.sep_model.module.load_state_dict(sep_package['state_dict'])
            self.asr_model.module.load_state_dict(asr_package['state_dict'])


            self.start_epoch = 0

        elif self.continue_from:
            package = torch.load(self.continue_from)

            self.sep_model.module.load_state_dict(package['sep_state_dict'])
            self.asr_model.module.load_state_dict(package['asr_state_dict'])

            self.sep_optimizer.load_state_dict(package['sep_optim_dict'])
            self.asr_optimizer.load_state_dict(package['asr_optim_dict'])
            
            self.start_epoch = int(package.get('epoch', 1)) if not self.ev else 0
            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]

        else:
            raise Exception("No model error")
        if self.ev:
            return
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.halving = False

    def eval(self, EOS_ID):

        self.sep_model.eval()
        self.asr_model.eval()
        data_loader = self.ev_loader
        for k, (data) in enumerate(data_loader):

            source, source_length, file_name = data
            file_name = file_name[0]

            with torch.no_grad():
                estimate_sources = self.sep_model(source.to(self.DEVICE))
                mask = get_mask(estimate_sources, source_length)
                estimate_sources *= mask

                speech, music = estimate_sources[:, 0], estimate_sources[:, 1]
                speech = speech / (2 * torch.max(speech.abs(), dim=1, keepdim=True)[0])
                music = music / (2 * torch.max(music.abs(), dim=1, keepdim=True)[0])


                speech = speech[0]
                source_stft = torch.stft(speech, self.n_fft, self.hop_length,
                    window=torch.hann_window(self.n_fft+1, device=self.DEVICE)[:-1],
                    center=False)
                source_stft_abs = torch.sqrt(torch.sum(source_stft**2, dim=2)+1e-12).t()
                source_lmfb = (torch.matmul(source_stft_abs, self.lmfb.t())+1e-12).log()
                torchdat = source_lmfb.unsqueeze(0)
                prediction_beam = self.asr_model.module.evaluate(torchdat, self.DEVICE, EOS_ID)
                file_name = file_name.replace(".wav", ".htk")
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
            # if self.lr_decay and epoch >= 20:
            #     self.lr = self.lr * 0.8
            #     for param_group in self.optimizer.param_groups:
            #         param_group['lr'] = self.lr

            # Train one epoch
            print("Training...")
            self.sep_model.train()  # Turn on BatchNorm & Dropout
            self.asr_model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss, tr_avg_loss_si_snr, tr_avg_loss_asr = self._run_one_epoch(epoch)
            self.tr_loss[epoch] = tr_avg_loss
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | Total Time {2} | '
                  'Train Loss {3:.3f} | Train SI-SNR Loss {4:.3f} | Train ASR Loss {5:.3f}'.format(
                    epoch + 1, time.time() - start, datetime.now() - now, tr_avg_loss, tr_avg_loss_si_snr, tr_avg_loss_asr))
            print('-' * 85)

            # Save model each epoch
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                package = {
                    'sep_state_dict': self.sep_model.module.state_dict(),
                    'asr_state_dict': self.asr_model.module.state_dict(),
                    'sep_optim_dict': self.sep_optimizer.state_dict(),
                    'asr_optim_dict': self.asr_optimizer.state_dict(),
                    'epoch' : epoch + 1 
                }
                package['tr_loss'] = self.tr_loss
                torch.save(package, file_path)
                print('Saving checkpoint model to %s' % file_path)

            sys.stdout.flush()

    def _run_one_epoch(self, epoch):
        start = time.time()
        total_loss = 0
        total_loss_si_snr = 0
        total_loss_asr = 0
        data_loader = self.tr_loader
        
        for k, (data) in enumerate(data_loader):
            padded_mixtures, padded_sources, sources_lengths, targets, mix_infos = data
            #print(list(map(lambda x: x[0], mix_infos)))

            
            sources_lengths = sources_lengths.to(self.DEVICE)
            padded_mixtures = padded_mixtures.to(self.DEVICE)
            padded_sources = padded_sources.to(self.DEVICE)
            #print(padded_sources, file=sys.stderr)


            estimate_sources = self.sep_model(padded_mixtures)

            #print(estimate_sources, file=sys.stderr)
            loss_si_snr, *_ = cal_loss(padded_sources, estimate_sources, sources_lengths)
            mask = get_mask(estimate_sources, sources_lengths)
            estimate_sources *= mask

            speech, music = estimate_sources[:, 0], estimate_sources[:, 1]
            speech = speech / (2 * torch.max(speech.abs(), dim=1, keepdim=True)[0])
            #print(speech, file=sys.stderr)
            targets = list(map(lambda x: torch.tensor(x, device=self.DEVICE, requires_grad=False).long(), targets))
            # stft, B x n_fft/2 + 1 x T x 2
            speeches_stft = torch.stft(speech, self.n_fft, self.hop_length,
                            window=torch.hann_window(self.n_fft+1, device=self.DEVICE)[:-1],
                            center=False)

            sources_lengths = (sources_lengths-400)//160+1
            
            # abs B x T x N(=201)
            speeches_stft_abs = torch.sqrt(torch.sum((speeches_stft + 1e-12)**2, dim=3)).transpose(1, 2)
            #print(sources_stft_abs, file=sys.stderr)
            # lmfb B x T x L(=40)
            speeches_lmfb = (torch.matmul(speeches_stft_abs, self.lmfb.t())+1e-12).log()

            mask = get_mask(speeches_lmfb.transpose(1, 2), sources_lengths)
            speeches_lmfb = speeches_lmfb * mask.transpose(1, 2)

            targets_lengths = torch.tensor(np.array([len(t) for t in targets], dtype=np.int32), device=self.DEVICE, requires_grad=False)

            padded_targets = nn.utils.rnn.pad_sequence(targets, batch_first=True)

            sorted_sources_lengths, perm_index = sources_lengths.sort(0, descending=True)
            sorted_targets_lengths = targets_lengths[perm_index]

            padded_sorted_speeches_lmfb = speeches_lmfb[perm_index]
            padded_sorted_targets = padded_targets[perm_index]


            #print(padded_sorted_sources_lmfb, file=sys.stderr)
            #print("----------------------------------------------", file=sys.stderr)
            prediction = self.asr_model(padded_sorted_speeches_lmfb, sorted_sources_lengths, padded_sorted_targets, self.DEVICE)

            loss_asr = 0.0
            for i in range(len(padded_mixtures)):
                num_labels = sorted_targets_lengths[i]
                label = padded_sorted_targets[i, :num_labels]
                onehot_target = torch.zeros((len(label), self.NUM_CLASSES), dtype=torch.float32, device=self.DEVICE, requires_grad=False)
                for j in range(len(label)):
                    onehot_target[j][label[j]] = 1.0
                ls_target = 0.9 * onehot_target + ((1.0 - 0.9) / (self.NUM_CLASSES - 1)) * (1.0 - onehot_target)
                loss_asr += -(F.log_softmax(prediction[i][:num_labels], dim=1) * ls_target).sum()

            #print("loss", loss, file=sys.stderr)
            self.sep_optimizer.zero_grad()
            self.asr_optimizer.zero_grad()
            #print('before backward', sources_stft_abs.grad)
            loss = self.lambda_si_snr * loss_si_snr + self.lambda_asr * loss_asr
            loss.backward()
            #print('after backward', sources_stft_abs.grad)
            nn.utils.clip_grad_norm_(self.sep_model.parameters(), self.max_norm)
            nn.utils.clip_grad_norm_(self.asr_model.parameters(), self.max_norm)
            self.sep_optimizer.step()
            self.asr_optimizer.step()

            total_loss += loss.item()
            total_loss_si_snr += loss_si_snr.item()
            total_loss_asr += loss_asr.item()

            if k % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | Average SI-SNR Loss {3:.3f} | Average ASR Loss {4:.3f} | '
                      'Current Loss {5:.6f} | Current SI-SNR Loss {6:.6f} | Current ASR Loss {7:.6f} | {8:.1f} ms/batch'.format(
                          epoch + 1, k + 1, total_loss / (k + 1), total_loss_si_snr / (k + 1), total_loss_asr / (k + 1),
                          loss.item(), loss_si_snr.item(), loss_asr.item(), 1000 * (time.time() - start) / (k + 1)),
                      flush=True)
                sys.stdout.flush()
            torch.cuda.empty_cache()

        return total_loss / (k + 1), total_loss_si_snr / (k + 1), total_loss_asr / (k + 1)

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
def write(inputs, filename, sr=16000):
        #librosa.output.write_wav(filename, inputs, sr)# norm=True)
        #librosa.output.write_wav(filename, inputs, sr, norm=True)
        #print(inputs)
        #inputs = inputs / max(np.abs(inputs))
        inputs = inputs / (2 * max(np.abs(inputs)))
        #print(inputs)

        sf.write(filename, inputs, sr)
        #sf.write(filename, inputs, sr, 'PCM_16')
        