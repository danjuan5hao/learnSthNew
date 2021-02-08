# -*- coding: utf-8 -*- 
import json 
import os, sys 

from collections import Counter
from random import choice 

from transformers import AutoTokenizer 


class DataConfig:
    ROOT = r"D:\data\数据资料\biendata\covid19赛道二关系抽取\task2_public"
    train_text = "task2_user_train.json"
    train_label = "task2_train_label.json"

    save_root = os.path.join(os.path.dirname(__file__), "../data/po")

class TokenStru:
    """
    方便记录一些token的信息，
    """
    def __init__(self, string):
        self.string = string
        self.string_start = None
        self.string_end = None 
        self.token_idx = None 
        self.token_tag = None 
        self.pretrain_vocab_idx = None
        self.attn_mask = None 

class PoPreprocessor:
    """
    每一个train example 随机取出来一条关系
    把 head 和 tail 的实体，分别在 对应到 pretrain-weight-tokeinzier的vocab中
    把关系进行labelEncodeing。

    最终关系表现为【head-start; head-end】【rel-idx】【tail-start; tail-end】
    用向量表示的话，最终会是一个【关系数量】*【text-max-len】类别的分类。

    """
    pretrain_model_weight = "bert-base-cased"
    SEP_TOKEN = "[SEP]"
    CLS_TOKEN = "[CLS]"
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"

    TOKENIZER = AutoTokenizer.from_pretrained('bert-base-cased')
    PRETRAIN_VOCAB = TOKENIZER.vocab
    
    MAX_LEN = 150

    def __init__(self, train_text, train_labels, text_max_len, num_rel=None):
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

    def build_seq_token_stru(self, text): # 没有add speical token
        seq_token = self.tokenize(text)
        token_idx_start_end = self.chars_token_align(text, seq_token)
        seq_token_stru = [TokenStru(i) for i in seq_token]  # 初始化所有token_stru

        for stru, (ix, start, end) in zip(seq_token_stru, token_idx_start_end):
            stru.string_start, stru.string_end, stru.token_idx = start, end, ix
            stru.pretrain_vocab_idx = self.PRETRAIN_VOCAB[stru.string]
            stru.attn_mask = 1
            stru.token_type_ids = 0

        def build_PAD_struc():
            pad_struc = TokenStru(self.PAD_TOKEN)
            pad_struc.attn_mask = 0
            pad_struc.token_type_ids = 0
            pad_struc.pretrain_vocab_idx = self.PRETRAIN_VOCAB.get(self.PAD_TOKEN)
            return pad_struc
        
        # def pad_trunc_tokened_text(strus):
        #     strus = strus[:self.MAX_LEN] 
        #     strus = strus + [build_PAD_struc()]*(self.MAX_LEN-len(strus))
        #     return strus
        
        # tokened_stru = pad_trunc_tokened_text(seq_token_stru)

        return seq_token_stru
   
    def tokenize(self, text): 
        return self.TOKENIZER.tokenize(text)

    def get_t_char_and_len(self, t):
        """
        用来得到sub word的字符表示和字符长度 
        bert分词，subword非开头会用##标出
        """
        if t.startswith("##"):
            return t[2:], len(t)-2
        else:
            return t, len(t)

    def chars_token_align(self, sentence, tokenized):
        """对齐句子和tokens"""
        # TODO 目前遇到不能对齐就报错了，需要更加robust不然会损失数据。

        s_idx = 0  # 记录找到sentence的哪个位置了
        token_idx_start_end = []  # 记录token的索引， 以及在sentence中的开始位置，结束位置，以及
        for idx, t in enumerate(tokenized):
            t_char, t_len = self.get_t_char_and_len(t)
            
            token_s_idx = sentence.find(t_char, s_idx, s_idx+t_len+2)
            if token_s_idx != -1:
                token_e_idx = token_s_idx + t_len
                token_idx_start_end.append((idx, token_s_idx, token_e_idx))
                s_idx = token_e_idx
            else:  # 表明不能对齐这句sentence
                raise ValueError(f"can't align because {idx}: {t}")         
        return  token_idx_start_end

    def find_tag_in_seq_token_stru(self, tag, seq_token_stru, tag_str):
            word = tag.get(tag_str).get("word")
            start = tag.get(tag_str).get("start")
            end = tag.get(tag_str).get("end")
                
            start_if = False
            end_if= False
            for i in range(len(seq_token_stru)):  # 是通过副作用来
                token_stru = seq_token_stru[i]

                if start == token_stru.string_start:
                    token_stru.token_tag = f"B-{tag_str}"
                    start_if = True
                if end == token_stru.string_end:
                    if token_stru.token_tag == f"B-{tag_str}":
                        token_stru.token_tag = f"U-{tag_str}"
                    else:
                        token_stru.token_tag = f"E-{tag_str}"
                    end_if = True
                if start_if and end_if:
                    break
            if start_if == False or end_if == False:
                raise ValueError(f"head:{word} not matched correctly. start: {start}, end: {end}")
            return 

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



def openfile(file):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)  

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
            save(os.path.join(DataConfig.save_root, f"train_{idx+1}"), need)

        except ValueError as e:
            print(e)
            cnt += 1
            continue
    