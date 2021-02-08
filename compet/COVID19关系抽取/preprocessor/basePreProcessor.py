# -*- coding: utf-8 -*-
import json
import os 
from random import choice 

from transformers import AutoTokenizer


def openfile(file):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)

class DataConfig:
    ROOT = r"D:\data\数据资料\biendata\covid19赛道二关系抽取\task2_public"
    train_text = "task2_user_train.json"
    train_label = "task2_train_label.json"

    save_root_s = os.path.join(os.path.dirname(__file__), "../data/head")
    save_root_po = os.path.join(os.path.dirname(__file__), "../data/po")

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

class BasePreProcessor:
    pretrain_model_weight = "bert-base-cased"
    SEP_TOKEN = "[SEP]"
    CLS_TOKEN = "[CLS]"
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"

    TOKENIZER = AutoTokenizer.from_pretrained(pretrain_model_weight)
    PRETRAIN_VOCAB = TOKENIZER.vocab
    
    MAX_LEN = 150

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
                raise ValueError(f"can't find {idx}: {t}")         
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