# -*- coding: utf-8 -*- 
import torch
import torch.nn as nn 

class PoStfinder(nn.Module):
    def __init__(self, input_size,  max_len, batch_first=True):
        self.s_lstm = nn.LSTM(input_size, max_len, batch_first=True)
        transformer_encoder_layer = nn.TransformerEncoderLayer(max_len,nhead=6)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, 6)
        self.conv1d = nn.Conv1d(,1)
        self.dense = nn.Linear()
        

    def forward(self, s_idx, context):
        s_context = context[:, s_idx, ...]
        s_feautre, (_, _) = self.s_lstm(s_context)
        a = torch.stack(s_feautre[:, -1, ...], context, dim=1)
        rst = self.transformer_encoder(a)
        self.conv1d(rst)
        self.dense()

        return  


if __name__ == "__main__":
    pass 
    test_tensor = torch.randn(100,20)