# -*- coding: utf-8 -*-
import os 
import torch

from torch.utils.data import Dataset

class PtrDataset(Dataset):
    def __init__(self, data_dir, end_idx=2, tag_max_len=10):
        super(PtrDataset, self).__init__()
        self.end_idx = end_idx
        self.root = data_dir
        self.tag_max_len = tag_max_len
        self.get_all_texts_tags()
        

    def get_text_tag_seq(self, file):
        
        with open(file, "r", encoding="utf-8") as f:
            tag_seq = []
            pretrain_idxs = []
            
            text_seq = [self.end_idx]
            mask_seq = [1]
            
            for line in f:
                split_line = line.strip().split("\t")
                # assert len(split_line) == 7
                
                token_idx, token_string, pretrain_idx, attn_mask, token_start, token_end, token_tag = split_line
                
                text_seq.append(int(pretrain_idx))
               
                    # print(file)
  
                mask_seq.append(int(attn_mask))
  
                if token_tag == "B-head" or token_tag == "U-head":
                    tag_seq.append(int(token_idx)+1)
                    pretrain_idxs.append(int(pretrain_idx))
            
            tag_seq.append(self.end_idx)

            def pad_tags(tags):
                assert len(tags) < self.tag_max_len, f"{len(tags)}"
                # print()
                tags = tags + [self.end_idx] * (self.tag_max_len - len(tags))
                return tags
            
            pretrain_idxs = pad_tags(pretrain_idxs)
            tag_seq = pad_tags(tag_seq)
            token_type_ids = [0]*len(text_seq)

   
        return torch.tensor(text_seq, dtype=torch.long), \
                torch.tensor(mask_seq, dtype=torch.long), \
                torch.tensor(token_type_ids, dtype=torch.long), \
                torch.tensor(pretrain_idxs, dtype=torch.long), \
                torch.tensor(tag_seq, dtype=torch.int64) 

    def get_all_texts_tags(self):
        texts = []
        mask_seqs = []
        token_type_idss = []
        decode_inputs_ids_s = []
        labels = []
        fails = []
        for file in os.listdir(self.root):
            file_path = os.path.join(self.root, file)
            try:
                rst = self.get_text_tag_seq(file_path)
                
                if rst:
                    text_seq, mask_seq, token_type_ids, decode_input_ids, tag_seq = rst
                    
                    texts.append(text_seq)
                    mask_seqs.append(mask_seq)
                    token_type_idss.append(token_type_ids)
                    decode_inputs_ids_s.append(decode_input_ids)
                    labels.append(tag_seq)
            except:
                fails.append(file)

        self.texts = torch.stack(texts)
        self.mask_seqs = torch.stack(mask_seqs)
        self.token_type_idss = torch.stack(token_type_idss)
        self.decode_inputs_ids_s = torch.stack(decode_inputs_ids_s)
        self.labels = torch.stack(labels)
        return 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.mask_seqs[idx], self.token_type_idss[idx], self.decode_inputs_ids_s[idx], self.labels[idx]

        

