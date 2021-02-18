# -*- coding: utf-8 -*- 
import torch
import torch.nn as nn 

class PoStfinder(nn.Module):
    """Po seq tag finder"""
    def __init__(self, input_size,  max_len, tag_num, batch_first=True):
        """input_size [feature dim]
        """
        self.s_lstm = nn.LSTM(input_size, max_len, batch_first=True)
        transformer_encoder_layer = nn.TransformerEncoderLayer(max_len,nhead=6)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, 6)
        self.conv1d = nn.Conv1d(input_size,tag_num,1)
        # self.dense = nn.Linear()
        
    def forward(self, s_idx, context):
        s_start, s_end  = s_idx
        s_context = context[:, s_start:s_end , ...]  
        s_feautre, (_, _) = self.s_lstm(s_context)  # 得到的维度，需要和seq_len一样才能拼接。
        ## 这里拼接的方法其实挺奇怪的，不知道为什么要这么做，
        a = torch.stack(s_feautre[:, -1, ...], context, dim=1)
        rst = self.transformer_encoder(a)  # [batch_size, seq_len, feature_dim]
        return self.conv1d(rst.permute(0,2,1)).permute(0,1,2)  # 
        
  
if __name__ == "__main__":
    pass