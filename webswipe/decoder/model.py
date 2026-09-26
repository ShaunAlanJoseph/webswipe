# PyTorch port of rnn_att_model from Shen et al. 2023 (model.py)
import torch
import torch.nn as nn

from .features import BLANK, N_CLASSES, N_FEATURES


class ShenDecoder(nn.Module):
    def __init__(self, in_dim=N_FEATURES, hidden=400, layers=2, dropout=0.2, n_classes=N_CLASSES):
        super().__init__()
        self.hidden = hidden
        # keras LSTM(dropout=...) drops the inputs of each layer
        self.drops = nn.ModuleList([nn.Dropout(dropout) for _ in range(layers)])
        self.lstms = nn.ModuleList([
            nn.LSTM(in_dim if i == 0 else 2 * hidden, hidden, batch_first=True, bidirectional=True)
            for i in range(layers)])
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x):
        h = x
        for drop, lstm in zip(self.drops, self.lstms):
            h, _ = lstm(drop(h))
        # keras Attention on the split BiLSTM output: query = forward half, value = key = backward half
        q, v = h[..., :self.hidden], h[..., self.hidden:]
        att = torch.softmax(q @ v.transpose(1, 2), dim=-1)
        return self.out(att @ v)  # (B, T, C) logits


def ctc_loss(logits, targets, target_lens):
    logp = logits.log_softmax(-1).transpose(0, 1)  # (T, B, C)
    in_lens = torch.full((logits.shape[0],), logits.shape[1], dtype=torch.long)
    return nn.functional.ctc_loss(logp, targets, in_lens, target_lens, blank=BLANK, zero_infinity=True)
