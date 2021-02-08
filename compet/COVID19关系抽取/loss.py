# -*- coding: utf-8 -*-
import torch.nn as nn

class SeqCceLoss(nn.Module):
    def __init__(self):
        super(SeqCceLoss, self).__init__()
        self.cce = nn.CrossEntropyLoss()
    
    def forward(self, pred, truth):
        """
        parms: 
        pred [batch, tag_seq_len, text_seq_len]
        truth [batch, tag_seq_len]
        """
        text_seq_len = pred.size(2)
        pred_flat = pred.view(-1, text_seq_len)
        truth_flat = truth.view(-1)
        return self.cce(pred_flat, truth_flat)


    