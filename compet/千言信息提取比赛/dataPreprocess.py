# -*- coding: utf-8 -*-
import torch
import torchtext
from torchtext.datasets import text_classification
# sample
# {'text': '《邪少兵王》是冰火未央写的网络小说连载于旗峰天下', 'spo_list': [{'predicate': '作者', 'object_type': {'@value': '人物'}, 'subject_type': '图书作品', 'object': {'@value': '冰火未央'}, 'subject': '邪少兵王'}]} 


class SLPreProcessor: # Sequence labeling
    """entity only appears in text once.
    """
    def __init__(self, outputFilePath):
        pass


    def process_one_sample(self, text, labels):
        label_seq = ["O"] * len(text) 
        for label in labels:
            self.process_one_label(label_seq, text, label) 
        return text, label_seq
            
    
    def process_one_label(self, label_seq, text, label):
        value, lype = label
        label_start_idx = text.find(value)
        label_seq[label_start_idx] = f"B-{lype}"
        for i in range(1, len(label)):
            label_seq[i] = f"I-{lype}"
        return 

    def process(self, data):
        for sample in data:
            text = sample.get("text")
            labels_raw = sample.get("spo_list") 
            labels = [i.get("subject") for i in labels_raw]
            text, label_seq = self.process_one_sample(text, labels)
            yield text, label_seq
    
    def o(self, data):
        with open(outputFilePath, "w", encoding="utf-8") as f:  
            for text, label_seq in self.process(data):




class PrePorcessor:
    def __init__(self, ):
        pass 

    pass 


# import torch.nn as nn
# import torch.nn.functional as F
# class TextSentiment(nn.Module):
#     def __init__(self, vocab_size, embed_dim, num_class):
#         super().__init__()
#         self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
#         self.fc = nn.Linear(embed_dim, num_class)
#         self.init_weights()

#     def init_weights(self):
#         initrange = 0.5
#         self.embedding.weight.data.uniform_(-initrange, initrange)
#         self.fc.weight.data.uniform_(-initrange, initrange)
#         self.fc.bias.data.zero_()

#     def forward(self, text, offsets):
#         embedded = self.embedding(text, offsets)
#         return self.fc(embedded)





if __name__ == "__main__":
    # import json
    # with open("data/train.json", "r", encoding="utf-8") as f:
    #     train_data = [json.loads(line) for line in f]
    #     train_data[0]

