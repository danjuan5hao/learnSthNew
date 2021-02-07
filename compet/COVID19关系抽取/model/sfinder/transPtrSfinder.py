# -*- coding: utf-8 -*- 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import AutoModel

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, inputs):
        self.transformer_encoder(inputs)


class Attention(nn.Module):

    def __init__(self, input_size, hidden_size, d3=128):
        super(Attention, self).__init__()
        self.input_dense = nn.Linear(input_size, d3) 
        self.hidden_dense = nn.Linear(hidden_size, d3)

        self.V = nn.Parameter(torch.FloatTensor(d3), requires_grad=True)
        nn.init.uniform_(self.V, -1, 1)
       
    def forward(self, inputs, context):
        context_len = context.size(1)
        batch_size = context.size(0)
    
        inputs = self.input_dense(inputs).unsqueeze(1)
        context = self.hidden_dense(context)

        a = torch.tanh(inputs + context)

        v = self.V.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
        
        att_row = torch.bmm(v, a.permute(0,2,1))
        att = att = F.softmax(att_row, dim=1).squeeze(1)
        return att_row, att
     
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.LSTM = nn.LSTM()

    def forward(self, inputs, h_0):
        return inputs
         
class TransPrtSfinder(nn.Module):
    def __init__(self):
        super(TransPrtSfinder, self).__init__()
        pass 

if __name__ == "__main__":
    pass 
