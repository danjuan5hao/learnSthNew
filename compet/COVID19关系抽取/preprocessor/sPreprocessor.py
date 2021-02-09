# -*- coding: utf-8 -*-
import json
import os, sys 
sys.path.append(os.path.join(os.path.dirname(__file__),".."))

from transformers import AutoTokenizer

from preprocessor.basePreProcessor import BasePreProcessor, DataConfig, TokenStru, openfile

class SPreprocessor(BasePreProcessor):

    def __init__(self):
        super(SPreprocessor, self).__init__()
    
    def tag_head(self, seq_tag, seq_token_stru):
        """
        为seq_token_stru中的token_tag字段填充值
        """
        for i in range(len(seq_tag)):
            tag = seq_tag[i]
            self.find_tag_in_seq_token_stru(tag, seq_token_stru, "head")    
        return seq_token_stru

def save(seq_token_stru, save_path):
    with open(save_path, "w", encoding="utf-8", newline="") as f:
        for i in seq_token_stru:
            tag = i.token_tag 
            tag = "O" if tag is None else tag
            f.write(f"{i.token_idx}\t{i.string}\t{i.pretrain_vocab_idx}\t{i.attn_mask}\t{i.string_start}\t{i.string_end}\t{tag}\n") 


if __name__ == "__main__":
    train_texts = openfile(os.path.join(DataConfig.ROOT, DataConfig.train_text))
    train_labels = openfile(os.path.join(DataConfig.ROOT, DataConfig.train_label))
    assert len(train_texts) == len(train_labels)
    print(f"{len(train_texts)} in total")

    preprocessor = SPreprocessor()
    
    if not os.path.exists(DataConfig.save_root_s):
        os.makedirs(DataConfig.save_root_s)

    cnt = 0
    for idx in range(len(train_texts)): 
        text = train_texts[idx]["train_{}".format(idx+1)]
        seq_tag  = train_labels[idx][f"train_{idx+1}"]
        try:
            seq_token_stru = preprocessor.build_seq_token_stru(text)  # 还没有
        except ValueError as e:  # TODO 不能对齐就整个丢掉了，浪费数据，需要改。 主要是针对于"[UNK]"
            print(f"{idx+1}: {e}")
            cnt += 1
            continue

        try:
            seq_token_stru = preprocessor.tag_head(seq_tag, seq_token_stru)
            save_path = os.path.join(DataConfig.save_root_s, f"train_{idx+1}.txt")
            save(seq_token_stru, save_path)

        except ValueError as e: 
            print(f"{idx+1}: {e}")
            cnt += 1
            continue
    print(cnt)