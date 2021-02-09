# -*- coding: utf-8 -*-
import torch 
from torch.utils.data import Subset, DataLoader
from dataLoader import PtrSfinderDataset

from model.sfinder.rnnPtrSfinder import RnnPtrSfinder 

from loss import SeqCceLoss

root = r"./data/head"
dataset = PtrSfinderDataset(root)
subset = Subset(dataset, range(9))

loader = DataLoader(subset, batch_size=3)
model = RnnPtrSfinder()

certi = SeqCceLoss()
optim = torch.optim.Adam(model.parameters())

model.train()
for i in range(10):
    for batch in loader:
        optim.zero_grad()
        all_text_bert_idx_seq = batch[0]
        all_text_bert_attn_mask = batch[1]
        all_text_bert_seq_type_id = batch[2]
        all_tag_bert_idx_seq = batch[3]
        all_tag_bert_attn_mask = batch[4]
        all_tag_bert_seq_type_id = batch[5]
        all_tag_points_truth = batch[6]

        atts, ptrs, context = model(all_text_bert_idx_seq, all_text_bert_attn_mask, all_text_bert_seq_type_id,
                            all_tag_bert_idx_seq, all_tag_bert_attn_mask, all_tag_bert_seq_type_id)

        loss = certi(attns, all_tag_points_truth)
        print(loss.data)
        loss.backward()
        optim.step()
    


    
