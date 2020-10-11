import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
from operator import itemgetter, attrgetter








#################################################################################################################
###   ASR MODEL
#################################################################################################################

class Encoder(nn.Module):
    def __init__(self, NUM_HIDDEN_NODES, NUM_ENC_LAYERS):
        super(Encoder, self).__init__()
        self.NUM_HIDDEN_NODES = NUM_HIDDEN_NODES
        self.NUM_ENC_LAYERS = NUM_ENC_LAYERS
        # encoder cnn
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels=32, kernel_size = (3, 3), stride=(2, 2), padding=(1, 0))
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 0))
        self.conv2_bn = nn.BatchNorm2d(32)
        # encoder
        self.bi_lstm = nn.LSTM(input_size=320, hidden_size=self.NUM_HIDDEN_NODES, num_layers=self.NUM_ENC_LAYERS, batch_first=True, dropout=0.2, bidirectional=True)
        self.ln = nn.LayerNorm(self.NUM_HIDDEN_NODES*2)

    def _calc_newlengths(self, lengths):
        newlengths = []
        for xlen in lengths:
            q1, mod1 = divmod(xlen.item(), 2)
            if mod1 == 0:
                xlen1 = xlen.item() // 2 - 1
                q2, mod2 = divmod(xlen1, 2)
                if mod2 == 0:
                    xlen2 = xlen1 // 2 - 1
                else:
                    xlen2 = (xlen1 -1) // 2
            else:
                xlen1 = (xlen.item() -1) // 2
                q2, mod2 = divmod(xlen1, 2)
                if mod2 == 0:
                    xlen2 = xlen1 // 2 - 1
                else:
                    xlen2 = (xlen1 - 1) // 2
            newlengths.append(xlen2)
        return newlengths

    def forward(self, x, lengths):
        # NxTxD -> NxTx[1ch]xD -> Nx[1ch]xDxT
        conv_out = self.conv1(x.unsqueeze(2).permute(0, 2, 3, 1))
        batched = self.conv1_bn(conv_out)
        activated = F.relu(batched)

        conv_out = self.conv2(activated)
        batched = self.conv2_bn(conv_out)
        activated = F.relu(batched)

        cnnout = activated.permute(0, 3, 1, 2).reshape(activated.size(0), activated.size(3), -1)
        newlengths = self._calc_newlengths(lengths)
        
        cnnout_packed = nn.utils.rnn.pack_padded_sequence(cnnout, newlengths, batch_first=True)
        self.bi_lstm.flatten_parameters()
        h, (hy, cy) = self.bi_lstm(cnnout_packed)
        h, lengths = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        h_ln = self.ln(h)
        for i in range(h.shape[0]):
            h_ln[i, lengths[i]:] = 0.0
        
        return h_ln, lengths

    def evaluate(self, x):
        # NxTxD -> NxTx[1ch]xD -> Nx[1ch]xDxT
        conv_out = self.conv1(x.unsqueeze(2).permute(0, 2, 3, 1))
        batched = self.conv1_bn(conv_out)
        activated = F.relu(batched)

        conv_out = self.conv2(activated)
        batched = self.conv2_bn(conv_out)
        activated = F.relu(batched)

        cnnout = activated.permute(0, 3, 1, 2).reshape(activated.size(0), activated.size(3), -1)

        h, (hy, cy) = self.bi_lstm(cnnout)
        h_ln = self.ln(h)

        return h_ln

class Decoder2l(nn.Module):
    def __init__(self, NUM_HIDDEN_NODES, NUM_CLASSES):
        super(Decoder2l, self).__init__()

        
        self.NUM_HIDDEN_NODES = NUM_HIDDEN_NODES
        self.NUM_CLASSES = NUM_CLASSES
        self.BEAM_WIDTH = 4
        # attention
        self.L_se = nn.Linear(self.NUM_HIDDEN_NODES, self.NUM_HIDDEN_NODES * 2, bias=False)
        self.L_he = nn.Linear(self.NUM_HIDDEN_NODES * 2, self.NUM_HIDDEN_NODES * 2)
        self.L_ee = nn.Linear(self.NUM_HIDDEN_NODES * 2, 1, bias=False)

        #conv attention
        self.F_conv1d = nn.Conv1d(1, 10, 100, stride=1, padding=50, bias=False)
        self.L_fe = nn.Linear(10, self.NUM_HIDDEN_NODES * 2, bias=False)

        # decoder
        self.L_sy = nn.Linear(self.NUM_HIDDEN_NODES, self.NUM_HIDDEN_NODES, bias=False)
        self.L_gy = nn.Linear(self.NUM_HIDDEN_NODES * 2, self.NUM_HIDDEN_NODES)
        self.L_yy = nn.Linear(self.NUM_HIDDEN_NODES, self.NUM_CLASSES)
        self.L_ys = nn.Embedding(self.NUM_CLASSES, self.NUM_HIDDEN_NODES * 4)
        self.L_ss1 = nn.Linear(self.NUM_HIDDEN_NODES, self.NUM_HIDDEN_NODES * 4, bias=False)
        self.L_gs1 = nn.Linear(self.NUM_HIDDEN_NODES * 2, self.NUM_HIDDEN_NODES * 4)
        self.L_ss12 = nn.Linear(self.NUM_HIDDEN_NODES, self.NUM_HIDDEN_NODES * 4, bias=False)
        self.L_ss2 = nn.Linear(self.NUM_HIDDEN_NODES, self.NUM_HIDDEN_NODES * 4, bias=False)
        self.L_gs2 = nn.Linear(self.NUM_HIDDEN_NODES * 2, self.NUM_HIDDEN_NODES * 4)

    def _lstmcell(self, x, c):
        ingate, forgetgate, cellgate, outgate = x.chunk(4, 1)
        c = (torch.sigmoid(forgetgate) * c) + (torch.sigmoid(ingate) * torch.tanh(cellgate))
        s = torch.sigmoid(outgate) * torch.tanh(c)
        return s, c

    def _attention(self, alpha, s, enc_output, mask = None):
        conv = self.F_conv1d(alpha)
        conv = conv.transpose(1, 2)[:, :-1, :]
        conv = self.L_fe(conv)
        e = torch.tanh(self.L_se(s).unsqueeze(1) + self.L_he(enc_output) + conv)
        e = self.L_ee(e)
        e_nonlin = (e - e.max(1)[0].unsqueeze(1)).exp()
        if mask is not None:
            e_nonlin = e_nonlin * mask
        alpha = e_nonlin / e_nonlin.sum(dim=1, keepdim=True)
        g = (alpha * enc_output).sum(dim=1)
        alpha = alpha.transpose(1, 2)
        return g, alpha

    def forward(self, enc_output, lengths, target, DEVICE):
        batch_size = enc_output.size(0)
        num_frames = enc_output.size(1)
        num_labels = target.size(1)

        e_mask = torch.ones((batch_size, num_frames, 1), device=DEVICE, requires_grad=False)
        s1 = torch.zeros((batch_size, self.NUM_HIDDEN_NODES), device=DEVICE, requires_grad=False)
        c1 = torch.zeros((batch_size, self.NUM_HIDDEN_NODES), device=DEVICE, requires_grad=False)
        s2 = torch.zeros((batch_size, self.NUM_HIDDEN_NODES), device=DEVICE, requires_grad=False)
        c2 = torch.zeros((batch_size, self.NUM_HIDDEN_NODES), device=DEVICE, requires_grad=False)

        prediction = torch.zeros((batch_size, num_labels, self.NUM_CLASSES), device=DEVICE, requires_grad=False)
        alpha = torch.zeros((batch_size, 1, num_frames), device=DEVICE, requires_grad=False)

        for i, tmp in enumerate(lengths):
            if tmp < num_frames:
                e_mask.data[i, tmp:] = 0.0

        for step in range(num_labels):
            g, alpha = self._attention(alpha, s1, enc_output, e_mask)
            y = self.L_yy(torch.tanh(self.L_gy(g) + self.L_sy(s2)))

            rec_input = self.L_ys(target[:, step]) + self.L_ss1(s1) + self.L_gs1(g)
            s1, c1 = self._lstmcell(rec_input, c1)

            rec_input = self.L_ss12(s1) + self.L_ss2(s2) + self.L_gs2(g)
            s2, c2 = self._lstmcell(rec_input, c2)
            prediction[:, step] = y

        return prediction

    def evaluate(self, enc_output, DEVICE, EOS_ID):
        num_frames = enc_output.size(1)

        s1 = torch.zeros((1, self.NUM_HIDDEN_NODES), device=DEVICE, requires_grad=False)
        c1 = torch.zeros((1, self.NUM_HIDDEN_NODES), device=DEVICE, requires_grad=False)
        s2 = torch.zeros((1, self.NUM_HIDDEN_NODES), device=DEVICE, requires_grad=False)
        c2 = torch.zeros((1, self.NUM_HIDDEN_NODES), device=DEVICE, requires_grad=False)

        alpha = torch.zeros((1, 1, num_frames), device=DEVICE, requires_grad=False)
        hypes = [([], 0.0, c1, c2, s1, s2, alpha)]
        token_finalist = []

        for _out_step in range(200):
            new_hypes = []
            
            for hype in hypes:
                out_seq, seq_score, c1, c2, s1, s2, alpha = hype
                g, alpha = self._attention(alpha, s1, enc_output)
                y = self.L_yy(torch.tanh(self.L_gy(g) + self.L_sy(s2)))
                scores = F.log_softmax(y, dim=1).data.squeeze(0)
                best_scores, indices = scores.topk(self.BEAM_WIDTH)
                best_scores2, indices2 = scores.topk(self.BEAM_WIDTH)
                for score, index in zip(best_scores, indices):
                    new_seq = out_seq + [index]
            
                    new_seq_score = seq_score + score
            
            
            
                    rec_input = self.L_ys(index) + self.L_ss1(s1) + self.L_gs1(g)
            
                    new_s1, new_c1 = self._lstmcell(rec_input, c1)
            
                    rec_input = self.L_ss12(new_s1) + self.L_ss2(s2) + self.L_gs2(g)
                    new_s2, new_c2 = self._lstmcell(rec_input, c2)

            
                    new_hypes.append((new_seq, new_seq_score, new_c1, new_c2, new_s1, new_s2, alpha))
            

            

            new_hypes_sorted = sorted(new_hypes, key=itemgetter(1), reverse=True)
            hypes = new_hypes_sorted[:self.BEAM_WIDTH]
            if new_hypes_sorted[0][0][-1] == EOS_ID:
                for i in range(self.BEAM_WIDTH):
                    token_finalist.append(new_hypes_sorted[i])
                break

        return token_finalist

class AttentionModel(nn.Module):
    def __init__(self, NUM_HIDDEN_NODES, NUM_ENC_LAYERS, NUM_CLASSES):
        super(AttentionModel, self).__init__()
        self.NUM_HIDDEN_NODES = NUM_HIDDEN_NODES
        self.NUM_ENC_LAYERS = NUM_ENC_LAYERS
        self.NUM_CLASSES = NUM_CLASSES
        self.encoder = Encoder(NUM_HIDDEN_NODES, NUM_ENC_LAYERS)
        self.decoder = Decoder2l(NUM_HIDDEN_NODES, NUM_CLASSES)

    def forward(self, speech, lengths, target, DEVICE):
        h, lengths = self.encoder(speech, lengths)
        prediction = self.decoder(h, lengths, target, DEVICE)
        return prediction

    def evaluate(self, speech, DEVICE, EOS_ID):
        h = self.encoder.evaluate(speech)
        token_beam_sel = self.decoder.evaluate(h, DEVICE, EOS_ID)
        return token_beam_sel

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None):
        package = {
            # hyper-parameter
            'NUM_HIDDEN_NODES': model.NUM_HIDDEN_NODES, 'NUM_ENC_LAYERS': model.NUM_ENC_LAYERS, 'NUM_CLASSES': model.NUM_CLASSES,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
        return package