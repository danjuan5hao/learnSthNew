# -*- coding: utf-8 -*-
import numpy as np 
import torch 
import torch.nn as nn 

class MergeEmbedding(nn.Module):
    """
    word char merge embedding,
    """
    def __init__(self, word_vocab, char_vocab):
        super(MergeEmbedding, self).__init__()
        self.word_embedding = word_vocab.embedding
        self.word_s2i = word_vocab.s2i
        self.word_i2s = word_vocab.i2s 
        self.word_emb_dim = word_vocab.emb_dim
        # self.word_unk_idx = word_unk_idx

        self.char_embedding = char_vocab.embedding
        self.char_s2i = char_vocab.s2i 
        self.char_i2s = char_vocab.i2s 
        self.char_emb_dim = char_vocab.emb_dim 
        # self.char_unk_idx = char_unk_idx

        self.char_lstm_cell = nn.LSTM(self.char_emb_dim, self.char_emb_dim)
        

    def _embedding_word_by_char(self, chars):
        chars_embeddeing = []
        for c in chars:
            if c not in  self.char_s2i:
                c_emb = np.random.randn(self.char_emb_dim)
            else:
                c_emb = self.char_embedding[self.char_s2i[c]]

            chars_embeddeing.append(c_emb)

        chars_embed_tensor = torch.Tensor(chars_embeddeing).unsqueeze(0)
        # output = []
        lstm_emb_chars, (_, _) = self.char_lstm_cell(chars_embed_tensor)
        
        last_output = lstm_emb_chars.squeeze(0)[-1]
        return last_output

    def _embedding_words(self, words):
        words_embedding = []
        for w  in words:
            if w not in self.word_s2i:
                w_emb = self._embedding_word_by_char(w)
            else:
                w_emb = self.word_embedding[self.word_s2i[w]]
            words_embedding.append(w_emb)
        return torch.Tensor(words_embedding)

    def _embedding_batch_words(self, batch_words):
        batch_words_embedding = []
        for words in batch_words:
            words_embedded = self._embedding_words(words) 
            batch_words_embedding.append(words_embedded)
        
        rst = torch.stack(batch_words_embedding)
        print(rst.size())
        

    def forward(self, sentences): 
        return self._embedding_batch_words(sentences)


if __name__ == "__main__":
    test_sentences = [["我们", "老肥", "大兄嘇"],
                      ["我们", "老肥",  '嘇']]


    import vocab
    char_vocab = vocab.Vocab("fake_c_emb.txt") 
    word_vocab = vocab.Vocab("fake_w_emb.txt")

    me = MergeEmbedding(word_vocab, char_vocab)
    me(test_sentences)



        