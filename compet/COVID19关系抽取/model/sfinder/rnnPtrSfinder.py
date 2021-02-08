# -*- coding: utf-8 -*-

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from transformers import AutoModel

class Encoder(nn.Module):
    """
    """
    def __init__(self, bert_embedding,  rnn_hidden_size):
        super(Encoder, self).__init__()
        self.bert_embedding_layer = bert_embedding
        self.lstm = nn.LSTM(768, rnn_hidden_size, bidirectional=True, batch_first=True)

        # self.en_h_0 = nn.parameter()
        # self.en_c_0 = nn.parameter()
    
    def forward(self, input_ids, attn_mask, token_type_ids):

        bert_embedded = self.bert_embedding_layer(input_ids, attn_mask, token_type_ids)
        bert_feature = bert_embedded[0]

        outputs, (_, _) = self.lstm(bert_feature)       
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
        
        att_before_softmax = torch.bmm(v, a.permute(0,2,1))
        att_after_softmax = F.softmax(att_before_softmax, dim=1).squeeze(1)
        return att_before_softmax, att_after_softmax
     
    
class Decoder(nn.Module):
    def __init__(self, bert_embedding, hidden_size):
        super(Decoder, self).__init__()
        self.bert_embedding = bert_embedding
        self.attn = Attention(hidden_size*2, hidden_size) # TODO bert_emb_dim should 
        self.first_decode_input_bert_embedding = nn.Parameter(torch.FloatTensor(768), requires_grad=True)
        self.dense_input = nn.Linear(768, hidden_size)
        self.dense_s = nn.Linear(hidden_size, hidden_size)

    def forward(self, 
                decoder_input_ids, decoder_attn_mask, decoder_seq_type_id,
                h_first, context):
        """
        input_ids: [batch, seq_len]
        
        """
        prts = []  # 记录产生的每一步atten的最大值的索引
        atts = []

        decoder_input_feautures = self.bert_embedding(decoder_input_ids, decoder_attn_mask, decoder_seq_type_id) 
        # outputs[0] # batch, seq_len, bert_emb_dim,
        decoder_input_feautures = decoder_input_feautures[0]
        
        batch_size = h_first.size(0)
        
        def step(inp, h):
            h = self.dense_s(h)
            s = torch.cat([self.dense_input(inp), h], dim=1)
            att_before_softmax, att_after_softmax = self.attn(s, context)
            return att_before_softmax, att_after_softmax, h

        h = context[:, -1, ...]

        for i in range(decoder_input_feautures.size(1)):
            t_i = decoder_input_feautures[:, i, ...]
            att_before_softmax, att_after_softmax, h = step(t_i, h)
            atts.append(att_before_softmax)
            max_probs, indices = att_after_softmax.max(1)
            prts.append(indices)

        return torch.stack(atts, dim=1).squeeze(), prts

class RnnPtrSfinder(nn.Module):
    def __init__(self, pretrain_model_weight="bert-base-cased", hidden_size=256):
        super(RnnPtrSfinder, self).__init__()
        
        self.bert_embedding = AutoModel.from_pretrained(pretrain_model_weight)
        self.encoder = Encoder(self.bert_embedding, hidden_size)  # 768 需要去自动获得？？ TODO
        self.decoder = Decoder(self.bert_embedding, hidden_size=hidden_size*2)
        # self.decoder_input0 = nn.Parameter(torch.FloatTensor(embedding_dim), requires_grad=False)
        # nn.init.uniform_(self.decoder_input0, -1, 1) 
        
    def forward(self, encoder_input_ids, encoder_attn_mask, encoder_seq_type_id,
                      decoder_input_ids, decoder_attn_mask, decoder_seq_type_id):

        context, h_first = self.encoder(encoder_input_ids, encoder_attn_mask, encoder_seq_type_id)



        atts, ptrs = self.decoder(decoder_input_ids, decoder_attn_mask, decoder_seq_type_id,
                           h_first, context)

        return atts, ptrs, context
     

if __name__ == "__main__":
    batch_size = 3
    bert_embedding = AutoModel.from_pretrained("bert-base-cased")
    test_sentence = {'input_ids': [9218, 1105, 23891, 1104, 1107, 4487, 7912], 
                     'token_type_ids': [0, 0, 0, 0, 0, 0, 0], 
                     'attention_mask': [1, 1, 1, 1, 1, 1, 1]}

    test_sentence_batch_input_ids = [test_sentence.get('input_ids')] * batch_size
    test_sentence_batch_input_ids = torch.tensor(test_sentence_batch_input_ids, dtype=torch.long)
    test_sentence_batch_attention_mask = [test_sentence.get('attention_mask')] * batch_size
    test_sentence_batch_attention_mask = torch.tensor(test_sentence_batch_attention_mask, dtype=torch.long)

    test_sentence_batch_token_type_ids = [test_sentence.get('token_type_ids')] * batch_size
    test_sentence_batch_token_type_ids = torch.tensor(test_sentence_batch_token_type_ids, dtype=torch.long)
    
    test_ptrnet = RnnPtrSfinder()
    atts, ptrs = test_ptrnet(test_sentence_batch_input_ids, test_sentence_batch_input_ids, test_sentence_batch_token_type_ids,
                      test_sentence_batch_token_type_ids, test_sentence_batch_attention_mask, test_sentence_batch_attention_mask)


    print(atts.size())