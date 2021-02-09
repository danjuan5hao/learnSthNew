# -*- coding: utf-8 -*- 
import json 
import os, sys 
sys.path.append(os.path.join(os.path.dirname(__file__),".."))

from collections import Counter
from random import choice 

from transformers import AutoTokenizer 

from preprocessor.basePreProcessor import BasePreProcessor, DataConfig, TokenStru, openfile



class PoPreprocessor(BasePreProcessor):
    """
    每一个train example 随机取出来一条关系
    把 head 和 tail 的实体，分别在 对应到 pretrain-weight-tokeinzier的vocab中
    把关系进行labelEncodeing。

    最终关系表现为【head-start; head-end】【rel-idx】【tail-start; tail-end】
    用向量表示的话，最终会是一个【关系数量】*【text-max-len】类别的分类。

    """

    def __init__(self, train_text, train_labels, text_max_len, num_rel=None):
        super(PoPreprocessor, self).__init__()
        self._get_all_relation(train_labels)
        if not num_rel:
            self.num_rel = len(self.r2i)
        else:
            self.num_rel = num_rel
        self.text_max_len = text_max_len
    
    def _get_all_relation(self, train_labels):
        all_rel = []
        for idx, spos in enumerate(train_labels):
            spos = spos.get(f"train_{idx+1}")
            for spo in spos:
                rel = spo.get("rel")
                all_rel.append(rel)
        rel_set = set(all_rel)
        self.r2i = {key: idx for idx, key in enumerate(rel_set)}
        self.i2r = {idx: key for idx, key in enumerate(rel_set)}
        return 

    def choice_one(self, spos):
        choiced = choice(spos)
        return choiced


    def tag_text(self, text, spos):
        choiced = choice(spos)
        
        seq_token_stru = self.build_seq_token_stru(text)
        self.find_tag_in_seq_token_stru(choiced, seq_token_stru, tag_str="head")
        self.find_tag_in_seq_token_stru(choiced, seq_token_stru, tag_str="tail")

        rel_idx = self.r2i[choiced["rel"]]
        need = {}
        for stru in seq_token_stru:
            if stru.token_tag in ["B-head", "U-head"]:
                need["B-head"] = stru.token_idx
            elif stru.token_tag == "E-head":
                need["E-head"] = stru.token_idx
            elif stru.token_tag in ["B-tail", "U-tail"]:
                need["B-tail"] = stru.token_idx
            elif stru.token_tag == "E-tail":
                need["E-tail"] = stru.token_idx
        need["rel"] = rel_idx
        return need


def save(file, need:dict):
    with open(file, "w", encoding="utf-8", newline="") as f:
        line = f'{need.get("B-head", "EMPTY")}\t{need.get("E-head", "EMPTY")}\t{need.get("rel", "EMPTY")}\t{need.get("B-tail", "EMPTY")}\t{need.get("E-tail", "EMPTY")}\n'
        f.write(line)

if __name__ == "__main__":
    train_texts = openfile(os.path.join(DataConfig.ROOT, DataConfig.train_text))
    train_labels = openfile(os.path.join(DataConfig.ROOT, DataConfig.train_label))

    preprocessor = PoPreprocessor(train_texts, train_labels, PoPreprocessor.MAX_LEN)
    cnt = 0
    for idx in range(len(train_texts)): 
        try:
            text = train_texts[idx]["train_{}".format(idx+1)]
            seq_tag  = train_labels[idx][f"train_{idx+1}"]
            need = preprocessor.tag_text(text, seq_tag)
            if len(need)  == 5:
                save(os.path.join(DataConfig.save_root_po, f"train_{idx+1}"), need)

        except ValueError as e:
            print(e)
            cnt += 1
            continue
    print(cnt)
    