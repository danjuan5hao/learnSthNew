# -*- coding: utf-8 -*-
import os 

import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, Subset 
import torch.optim as optim

from dataLoader import PtrDataset
# from ptrNet import PointerNet
from ptrNetMe import PrtNet as PrtNetMe

# model_you = PointerNet(embedding_dim=100,
#              hidden_dim=256,lstm_layers=2,dropout=0.5)
model_me = PrtNetMe()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_me.parameters())

train_path = r"data\head"

dataset = PtrDataset(train_path)
datasubset =    Subset(dataset, range(7, 17))
dataloader = DataLoader(datasubset, batch_size = 2) 
## 这里model 用的应该是 atten的output和label 做拟合， 而不是得到的atten的max和label做拟合。 

model_me.train()
for i in range(60):
    for batch in dataloader:
        optimizer.zero_grad()
        inputs = {"input_ids": batch[0],
                "mask": batch[1],
                "token_type_id": batch[2], 
                "decoder_inputs": batch[3],
                "decoder_mask": torch.tensor([[1]*10]*2, dtype=torch.long),
                "labels": batch[4]}

        outputs = model_me(inputs.get("input_ids"),
                        inputs.get("decoder_inputs"),
                        inputs.get("mask"),
                        inputs.get("decoder_mask"))[0]
        outputs = outputs.contiguous().view(-1, outputs.size()[-1])
        labels = inputs.get("labels").view(-1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(loss.data)
   
