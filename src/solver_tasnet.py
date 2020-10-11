import os
import time
import sys
from datetime import datetime
import torch

from criterion import cal_loss


class Solver(object):
    
    def __init__(self, data, model, optimizer, args):
        self.tr_loader = data['tr_loader']
        self.cv_10_loader = data['cv_10_loader']
        self.cv_5_loader = data['cv_5_loader']
        self.cv_0_loader = data['cv_0_loader']
        self.cv_m5_loader = data['cv_m5_loader']
        self.model = model
        self.optimizer = optimizer

        # Training config
        self.use_cuda = args.use_cuda
        self.epochs = args.epochs
        self.half_lr = args.half_lr
        self.early_stop = args.early_stop
        self.max_norm = args.max_norm
        # save and load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.model_path = args.model_path
        # logging
        self.print_freq = args.print_freq
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_10_loss = torch.Tensor(self.epochs)
        self.cv_5_loss = torch.Tensor(self.epochs)
        self.cv_0_loss = torch.Tensor(self.epochs)
        self.cv_m5_loss = torch.Tensor(self.epochs)
        
        

        self._reset()

    def _reset(self):
        # Reset
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            package = torch.load(self.continue_from)
            self.model.module.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start_epoch = int(package.get('epoch', 1))
            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_10_loss[:self.start_epoch] = package['cv_10_loss'][:self.start_epoch]
            self.cv_5_loss[:self.start_epoch] = package['cv_5_loss'][:self.start_epoch]
            self.cv_0_loss[:self.start_epoch] = package['cv_0_loss'][:self.start_epoch]
            self.cv_m5_loss[:self.start_epoch] = package['cv_m5_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False
        self.val_no_impv = 0




    def train(self):
        now = datetime.now()
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            print("Training...")
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | Total Time {2} | '
                  'Train Loss {3:.3f}'.format(epoch + 1, time.time() - start, datetime.now() - now, tr_avg_loss))
            print('-' * 85)

            

            # Cross validation
            print('Cross validation with snr10...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            strat = time.time()
            val_10_loss = self._run_one_epoch(epoch, cross_valid=self.cv_10_loader)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | Total Time {2} | '
                  'Valid Loss {3:.3f}'.format(
                      epoch + 1, time.time() - start, datetime.now() - now, val_10_loss))
            print('-' * 85)

            print('Cross validation with snr5...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            strat = time.time()
            val_5_loss = self._run_one_epoch(epoch, cross_valid=self.cv_5_loader)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | Total Time {2} | '
                  'Valid Loss {3:.3f}'.format(
                      epoch + 1, time.time() - start, datetime.now() - now, val_5_loss))
            print('-' * 85)


            print('Cross validation with snr0...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            strat = time.time()
            val_0_loss = self._run_one_epoch(epoch, cross_valid=self.cv_0_loader)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | Total Time {2} | '
                  'Valid Loss {3:.3f}'.format(
                      epoch + 1, time.time() - start, datetime.now() - now, val_0_loss))
            print('-' * 85)

            print('Cross validation with snr-5...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            strat = time.time()
            val_m5_loss = self._run_one_epoch(epoch, cross_valid=self.cv_m5_loader)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | Total Time {2} | '
                  'Valid Loss {3:.3f}'.format(
                      epoch + 1, time.time() - start, datetime.now() - now, val_m5_loss))
            print('-' * 85)


            # Save the best model
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_10_loss[epoch] = val_10_loss
            self.cv_5_loss[epoch] = val_5_loss
            self.cv_0_loss[epoch] = val_0_loss
            self.cv_m5_loss[epoch] = val_m5_loss
            
            # Save model each epoch
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save(self.model.module.serialize(self.model.module,
                                                       self.optimizer, epoch + 1,
                                                       tr_loss=self.tr_loss,
                                                       cv_10_loss=self.cv_10_loss, 
                                                       cv_5_loss=self.cv_5_loss, 
                                                       cv_0_loss=self.cv_0_loss, 
                                                       cv_m5_loss=self.cv_m5_loss),
                           file_path)
                print('Saving checkpoint model to %s' % file_path)

            # Adjust learning rate (halving)
            if self.half_lr:
                if val_0_loss >= self.prev_val_loss:
                    self.val_no_impv += 1
                    if self.val_no_impv >= 3:
                        self.halving = True
                    if self.val_no_impv >= 10 and self.early_stop:
                        print("No imporvement for 10 epochs, early stopping.")
                        break
                else:
                    self.val_no_impv = 0
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = \
                    optim_state['param_groups'][0]['lr'] / 2.0
                self.optimizer.load_state_dict(optim_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
                self.halving = False
            self.prev_val_loss = val_0_loss

            

            if val_0_loss < self.best_val_loss:
                self.best_val_loss = val_0_loss
                file_path = os.path.join(self.save_folder, self.model_path)
                torch.save(self.model.module.serialize(self.model.module,
                                                       self.optimizer, epoch + 1,
                                                       tr_loss=self.tr_loss,
                                                       cv_10_loss=self.cv_10_loss,
                                                       cv_5_loss=self.cv_5_loss,
                                                       cv_0_loss=self.cv_0_loss,
                                                       cv_m5_loss=self.cv_m5_loss),
                           file_path)
                print("Find better validated model, saving to %s" % file_path)

            sys.stdout.flush()

    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0

        data_loader = self.tr_loader if not cross_valid else cross_valid

        for i, (data) in enumerate(data_loader):
            padded_mixture, mixture_lengths, padded_source = data


            if self.use_cuda:
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                padded_source = padded_source.cuda()
            estimate_source = self.model(padded_mixture)
            loss, max_snr, estimate_source = \
                cal_loss(padded_source, estimate_source, mixture_lengths)
            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_norm)
                self.optimizer.step()

            total_loss += loss.item()

            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                          epoch + 1, i + 1, total_loss / (i + 1),
                          loss.item(), 1000 * (time.time() - start) / (i + 1)),
                      flush=True)
                sys.stdout.flush()

        
            
            
        return total_loss / (i + 1)
