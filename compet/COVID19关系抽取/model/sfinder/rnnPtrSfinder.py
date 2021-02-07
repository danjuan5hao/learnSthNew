# -*- coding: utf-8 -*-

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from transformers import AutoModel

class Encoder(nn.Module):
    """
    """

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
    
    def forward(self, x):
        # if mask:
        #     x = torch.masked_select(x, mask) 
        outputs, (_, _) = self.lstm(x)       
        return outputs, outputs[:, -1, ...]
           
    
class Attention(nn.Module):
    """
    """

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
    def __init__(self, bert_embedding, hidden_size):
        super(Decoder, self).__init__()
        self.bert_embedding = bert_embedding
        self.attn = Attention(hidden_size, hidden_size) # TODO bert_emb_dim should 
        self.first_decode_input_bert_embedding = nn.Parameter(torch.FloatTensor(768), requires_grad=True)
        self.dense = nn.Linear(768+hidden_size, hidden_size)

    def forward(self, input_ids, mask, first_hidden, context):
        """
        input_ids: [batch, seq_len]
        
        """

        prts = []
        atts = []
        input_feautures = self.bert_embedding(input_ids, 
                            attention_mask=mask) 
        # outputs[0] # batch, seq_len, bert_emb_dim,
        input_feautures = input_feautures[0]
        
        batch_size = first_hidden.size(0)
        
        def step(inp, h):
  
            x = torch.cat([inp, h], dim=1) 

            h = torch.tanh(self.dense(x))

            att_row, att = self.attn(h, context)
            return att_row, att, h

        att_row,att,  h = step(self.first_decode_input_bert_embedding.expand(batch_size, -1), first_hidden)
        # atts.append(att)
        for i in range(input_feautures.size(1)):
            t_i = input_feautures[:, i, ...]
            
            att_row, att, h = step(t_i, h)
            atts.append(att_row)
            max_probs, indices = att.max(1)
            prts.append(indices)

        return torch.stack(atts), prts

         
class PrtNet(nn.Module):
    def __init__(self, pretrain_model_weight="bert-base-cased", hidden_size=256):
        super(PrtNet, self).__init__()
        
        self.bert_embedding = AutoModel.from_pretrained(pretrain_model_weight)
        self.encoder = Encoder(768, hidden_size)  # 768 需要去自动获得？？ TODO
        self.decoder = Decoder(self.bert_embedding, hidden_size=hidden_size*2)
        # self.decoder_input0 = nn.Parameter(torch.FloatTensor(embedding_dim), requires_grad=False)
        # nn.init.uniform_(self.decoder_input0, -1, 1) 
        
    def forward(self, encoder_input_ids, decoder_inputs_ids, enocder_mask=None, decoder_mask=None):
        outputs = self.bert_embedding(encoder_input_ids, 
                                    attention_mask=enocder_mask) 
        context, h, c = self.encoder(outputs[0])

        ptr = self.decoder(decoder_inputs_ids, 
                           decoder_mask, h, context)

        return ptr
     

if __name__ == "__main__":
    decoder_feature_dim = 512
    decoder_hidden_dim = 512
    batch_size = 6
    # seq_len = 10

    bert_embedding = AutoModel.from_pretrained("bert-base-cased")
    # test_attention_layer =  Attention(decoder_feature_dim, decoder_hidden_dim)
    # test_decoder = Decoder(bert_embedding, decoder_hidden_dim)

    # fake_att_input = torch.randn(batch_size, decoder_feature_dim)
    # fake_att_context = torch.randn(batch_size, seq_len, decoder_hidden_dim)
    # test_attention_layer(fake_att_input, fake_att_context)
    test_sentence = {'input_ids': [9218, 1105, 23891, 1104, 1107, 4487, 7912], 
                     'token_type_ids': [0, 0, 0, 0, 0, 0, 0], 
                     'attention_mask': [1, 1, 1, 1, 1, 1, 1]}

    test_sentence_batch_input_ids = [test_sentence.get('input_ids')] * batch_size
    test_sentence_batch_input_ids = torch.tensor(test_sentence_batch_input_ids, dtype=torch.long)
    test_sentence_batch_attention_mask = [test_sentence.get('attention_mask')] * batch_size
    test_sentence_batch_attention_mask = torch.tensor(test_sentence_batch_attention_mask, dtype=torch.long)
    
    # fake_context = torch.randn(batch_size, seq_len, decoder_hidden_dim)
    # fake_first_hidden = torch.randn(batch_size, decoder_hidden_dim)
    
    # test_decoder(test_sentence_batch_input_ids, test_sentence_batch_attention_mask,
    #                 fake_first_hidden, fake_context)


    test_ptrnet = PrtNet()
    rst = test_ptrnet(test_sentence_batch_input_ids, test_sentence_batch_input_ids,
                test_sentence_batch_attention_mask, test_sentence_batch_attention_mask)


