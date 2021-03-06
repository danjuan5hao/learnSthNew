# -*- coding: utf-8 -*-
import os 
import torch

from torch.utils.data import Dataset

class PtrSfinderDataset(Dataset):
    """
    """
    def __init__(self, data_dir,  text_max_len=150, tag_max_len=10):
        super(PtrSfinderDataset, self).__init__()
        self.root = data_dir

        self.text_max_len=text_max_len
        self.tag_max_len=tag_max_len
        self.pstart_idx = 2

        self.get_all_texts_tags()

    def get_text_tag_seq(self, file):
        """
        """
        with open(file, "r", encoding="utf-8") as f:
            text_bert_idx_seq = [self.pstart_idx]  # 
            text_bert_attn_mask = [1]
            text_bert_seq_type_id = [0]

            tag_bert_idx_seq = [self.pstart_idx]  
            tag_bert_attn_mask = [1]
            tag_bert_seq_type_id = [0]

            tag_points_truth = []

            for line in f:
                split_line = line.strip().split("\t")
                token_idx, token_string, pretrain_idx, attn_mask, token_start, token_end, token_tag = split_line

                text_bert_idx_seq.append(int(pretrain_idx))
                text_bert_attn_mask.append(int(attn_mask))
                text_bert_seq_type_id.append(0)

                if token_tag.startswith("B") or token_tag.startswith("E"):
                    tag_bert_idx_seq.append(int(pretrain_idx))
                    tag_bert_attn_mask.append(int(attn_mask))
                    tag_bert_seq_type_id.append(0)

                    tag_points_truth.append(int(token_idx)+1)
                tag_points_truth.append(0)

        text_bert_idx_seq = self._pad_and_truncate_one(text_bert_idx_seq, self.text_max_len, 0)
        text_bert_attn_mask = self._pad_and_truncate_one(text_bert_attn_mask, self.text_max_len, 0)
        text_bert_seq_type_id = self._pad_and_truncate_one(text_bert_seq_type_id, self.text_max_len, 0)

        tag_bert_idx_seq = self._pad_and_truncate_one(tag_bert_idx_seq, self.tag_max_len, 0)
        tag_bert_attn_mask = self._pad_and_truncate_one(tag_bert_attn_mask, self.tag_max_len, 0)
        tag_bert_seq_type_id = self._pad_and_truncate_one(tag_bert_seq_type_id, self.tag_max_len, 0)

        tag_points_truth = self._pad_and_truncate_one(tag_points_truth, self.tag_max_len, 0)
        
        return torch.tensor(text_bert_idx_seq, dtype=torch.long), \
            torch.tensor(text_bert_attn_mask, dtype=torch.long), \
            torch.tensor(text_bert_seq_type_id, dtype=torch.long), \
            torch.tensor(tag_bert_idx_seq, dtype=torch.long), \
            torch.tensor(tag_bert_attn_mask, dtype=torch.long), \
            torch.tensor(tag_bert_seq_type_id, dtype=torch.long), \
            torch.tensor(tag_points_truth, dtype=torch.long) 

    def get_all_texts_tags(self):
        all_text_bert_idx_seq = []
        all_text_bert_attn_mask = []
        all_text_bert_seq_type_id = []

        all_tag_bert_idx_seq = []
        all_tag_bert_attn_mask = []
        all_tag_bert_seq_type_id = []

        all_tag_points_truth = []

        for file in os.listdir(self.root):
            file_path = os.path.join(self.root, file)
            fails = []
            rst = self.get_text_tag_seq(file_path)
                

            text_bert_idx_seq, text_bert_attn_mask, text_bert_seq_type_id, \
            tag_bert_idx_seq, tag_bert_attn_mask, tag_bert_seq_type_id, tag_points_truth  = rst
            
            all_text_bert_idx_seq.append(text_bert_idx_seq)
            all_text_bert_attn_mask.append(text_bert_attn_mask) 
            all_text_bert_seq_type_id.append(text_bert_seq_type_id)

            all_tag_bert_idx_seq.append(tag_bert_idx_seq)
            all_tag_bert_attn_mask.append(tag_bert_attn_mask)
            all_tag_bert_seq_type_id.append(tag_bert_seq_type_id)
            
            all_tag_points_truth.append(tag_points_truth)
            # except:
            #     print("herer")
            #     fails.append(file)

        self.all_text_bert_idx_seq = all_text_bert_idx_seq
        self.all_text_bert_attn_mask = all_text_bert_attn_mask
        self.all_text_bert_seq_type_id = all_text_bert_seq_type_id

        self.all_tag_bert_idx_seq = all_tag_bert_idx_seq
        self.all_tag_bert_attn_mask = all_tag_bert_attn_mask
        self.all_tag_bert_seq_type_id = all_tag_bert_seq_type_id

        self.all_tag_points_truth = all_tag_points_truth
        return 

    def _pad_and_truncate_one(self, seq, max_len, pad_item):
        seq = seq[:max_len] 
        seq = seq + [pad_item]*(max_len - len(seq))
        return seq

    def __len__(self):
        return len(self.all_tag_points_truth)

    def __getitem__(self, idx):
        return self.all_text_bert_idx_seq[idx], \
               self.all_text_bert_attn_mask[idx], \
               self.all_text_bert_seq_type_id[idx], \
               self.all_tag_bert_idx_seq[idx], \
               self.all_tag_bert_attn_mask [idx], \
               self.all_tag_bert_seq_type_id[idx], \
               self.all_tag_points_truth[idx]

class SPOdataset(Dataset):
    def __init__(self, head_root, po_root, text_max_len=250, tag_max_len=10):
        self.head_root = head_root
        self.po_root = po_root
        self.union = self._find_union(head_root, po_root) 
        self.pstart_idx = 2

    def _find_union(self, head_root, po_root): 
        a = set(i.split(".")[0] for i in os.listdir(head_root))
        b = set(os.listdir(po_root))
        return a & b

    def _pad_and_truncate_one(self, seq, max_len, pad_item):
        seq = seq[:max_len] 
        seq = seq + [pad_item]*(max_len - len(seq))
        return seq

    def get_text_tag_seq(self, file):
        """
        """
        with open(file, "r", encoding="utf-8") as f:
            text_bert_idx_seq = [self.pstart_idx]  # 
            text_bert_attn_mask = [1]
            text_bert_seq_type_id = [0]

            tag_bert_idx_seq = [self.pstart_idx]  
            tag_bert_attn_mask = [1]
            tag_bert_seq_type_id = [0]

            tag_points_truth = []

            for line in f:
                split_line = line.strip().split("\t")
                token_idx, token_string, pretrain_idx, attn_mask, token_start, token_end, token_tag = split_line

                text_bert_idx_seq.append(int(pretrain_idx))
                text_bert_attn_mask.append(int(attn_mask))
                text_bert_seq_type_id.append(0)

                if token_tag.startswith("B") or token_tag.startswith("E"):
                    tag_bert_idx_seq.append(int(pretrain_idx))
                    tag_bert_attn_mask.append(int(attn_mask))
                    tag_bert_seq_type_id.append(0)

                    tag_points_truth.append(int(token_idx)+1)
                tag_points_truth.append(0)

        text_bert_idx_seq = self._pad_and_truncate_one(text_bert_idx_seq, self.text_max_len, 0)
        text_bert_attn_mask = self._pad_and_truncate_one(text_bert_attn_mask, self.text_max_len, 0)
        text_bert_seq_type_id = self._pad_and_truncate_one(text_bert_seq_type_id, self.text_max_len, 0)

        tag_bert_idx_seq = self._pad_and_truncate_one(tag_bert_idx_seq, self.tag_max_len, 0)
        tag_bert_attn_mask = self._pad_and_truncate_one(tag_bert_attn_mask, self.tag_max_len, 0)
        tag_bert_seq_type_id = self._pad_and_truncate_one(tag_bert_seq_type_id, self.tag_max_len, 0)

        tag_points_truth = self._pad_and_truncate_one(tag_points_truth, self.tag_max_len, 0)
        
        return torch.tensor(text_bert_idx_seq, dtype=torch.long), \
            torch.tensor(text_bert_attn_mask, dtype=torch.long), \
            torch.tensor(text_bert_seq_type_id, dtype=torch.long), \
            torch.tensor(tag_bert_idx_seq, dtype=torch.long), \
            torch.tensor(tag_bert_attn_mask, dtype=torch.long), \
            torch.tensor(tag_bert_seq_type_id, dtype=torch.long), \
            torch.tensor(tag_points_truth, dtype=torch.long) 


    def get_spo_idx(self, file):
        with open(file, "r", encoding="utf-8") as f:
            hs, he, rel, ts, te =  f.read().strip().split("\t")
            return  torch.tensor(hs), \
                    torch.tensor(he), \
                    torch.tensor(rel), \
                    torch.tensor(ts), \
                    torch.tensor(te),

    def get_all_text_tag_seq_(self): 

        all_text_bert_idx_seq = []
        all_text_bert_attn_mask = []
        all_text_bert_seq_type_id = []

        all_tag_bert_idx_seq = []
        all_tag_bert_attn_mask = []
        all_tag_bert_seq_type_id = []

        all_tag_points_truth = []

        sstart = []
        send = []
        rels = []
        ostart = [] 
        oend = []

        for item in self.union:
            sfile = os.path.join(self.head_root, f"{item}.txt")
            pofile = os.path.join(self.po_root, item)

            a,b,c,d,e,f,g = self.get_text_tag_seq(sfile)
            h,i,j,k,l = self.get_spo_idx(pofile)

            all_text_bert_idx_seq.append(a)
            all_text_bert_attn_mask.append(b)
            all_text_bert_seq_type_id.append(c)

            all_tag_bert_idx_seq.append(d)
            all_tag_bert_attn_mask.append(e)
            all_tag_bert_seq_type_id .append(f)

            all_tag_points_truth.append(g)

            sstart.append(h)
            send.append(i)
            rels.append(j)
            ostart.append(k)
            oend.append(l)

        return 

    def __getitem__(self, idx):
        pass 

    def __len__(self):
        return len(self.union)

if __name__ == "__main__":
    from torch.utils.data import Subset, DataLoader
    head_root = "./data/head"
    po_root = "./data/po"
    dataset = PtrSfinderDataset(head_root)

    subset  = Subset(dataset, range(9))
    dataloader = DataLoader(subset, batch_size=3)

    for i in range(5):
        for batch in dataloader:
            print(batch[0].size())
            print(batch[1].size())
            print(batch[2].size())
            print(batch[3].size())
            print(batch[4].size())
            print(batch[5].size())
            print(batch[6].size())

    