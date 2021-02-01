# -*- coding: utf-8 -*-
import json
import os 

from transformers import AutoTokenizer
pretrain_model_weight = "bert-base-cased"
TOKENIZER = AutoTokenizer.from_pretrained('bert-base-cased')
PRETRAIN_VOCAB = TOKENIZER.vocab

SEP_TOKEN = "[SEP]"
CLS_TOKEN = "[CLS]"
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"

MAX_LEN = 100

ROOT = r"D:\data\数据资料\biendata\covid19赛道二关系抽取\task2_public"
train_text = "task2_user_train.json"
train_label = "task2_train_label.json"

path = "COVID19关系抽取\data\head"

class Token_stru:
    def __init__(self, string):
        self.string = string
        self.string_start = None
        self.string_end = None 
        self.token_idx = None 
        self.token_tag = None 
        self.pretrain_vocab_idx = None
        self.attn_mask = None 
    
    # def __repr__(self):
    #     return f"{self.token_idx}\t{self.string}\t{self.string_start}\t{self.string_end}\t{self.token_tag}"


# def convert_text_to_ids(self, text):

#     tokens = TOKENIZER.tokenize(text, add_special_tokens=True)
#     tokens = ["[CLS]"]+tokens[:MAX_LEN-2]+["[SEP]"]
#     text_len = len(tokens)
#     input_ids = TOKENIZER.convert_tokens_to_ids(tokens+["[PAD]"]*(MAX_LEN-text_len))
#     attention_mask = [1]*text_len+[0]*(MAX_LEN-text_len)
#     token_type_ids = [0]*MAX_LEN

#     assert len(input_ids) == MAX_LEN
#     assert len(attention_mask) == MAX_LEN
#     assert len(token_type_ids) == MAX_LEN

#     return tokens, input_ids, attention_mask, token_type_ids
    

def get_t_char_and_len(t):
    if t.startswith("##"):
        return t[2:], len(t)-2
    else:
        return t, len(t)

def chars_token_align(sentence, tokenized):
    """逐个对齐，"""
    s_idx = 0
    token_s_e_idx = []
    for idx, t in enumerate(tokenized):
        t_char, t_len = get_t_char_and_len(t)
        
        token_s_idx = sentence.find(t_char, s_idx, s_idx+t_len+3)
        if token_s_idx != -1:
            token_e_idx = token_s_idx + t_len
        else:
            print(f"error at {idx}: {t}")
            raise ValueError(f"cant find") 
        token_s_e_idx.append((token_s_idx, token_e_idx))

        s_idx = token_e_idx
    return  token_s_e_idx


def tag_head(labels, tokened_struc):
    for label in labels:
        head_word = label.get("head").get("word")
        head_start = label.get("head").get("start")
        head_end = label.get("head").get("end")

        start_ed = False
        end_ed = False

        for j in tokened_struc:
            if head_start == j.string_start:
                j.token_tag = "B-head"
                start_ed = True
            if head_end == j.string_end:
                j.token_tag = "E-head"
                end_ed = True
            if head_start == head_end:
                j.token_tag = "U-head"

        if start_ed == False or end_ed == False:
            raise ValueError(f"label:{head_word} not matched. start: {head_start}, end: {head_end}")
            
    return tokened_struc


def openfile(file):
    with open(file) as f:
        return json.load(f)


def build_token_stru(text): # 没有add speical token
    tokened_text = TOKENIZER.tokenize(text)
    
    token_s_e_idx = chars_token_align(text, tokened_text)

    tokened_stru = [Token_stru(i) for i in tokened_text]
    for ix, (stru, s_e_idx) in enumerate(zip(tokened_stru, token_s_e_idx)):
        stru.string_start = s_e_idx[0]
        stru.string_end = s_e_idx[1]
        stru.token_idx = ix 
        stru.pretrain_vocab_idx = PRETRAIN_VOCAB.get(stru.string, PRETRAIN_VOCAB.get(UNK_TOKEN))
        stru.attn_mask = 1
        stru.token_type_ids = 0

    def build_PAD_struc():
        struc = Token_stru(PAD_TOKEN)
        struc.attn_mask = 0
        struc.token_type_ids = 0
        struc.pretrain_vocab_idx = PRETRAIN_VOCAB.get(PAD_TOKEN)
        return struc
    
    def pad_trunc_tokened_text(strus):
        strus = strus[:MAX_LEN] # pad
        len_strus = len(strus)
        strus = strus + [build_PAD_struc()]*(MAX_LEN-len(strus))
        return strus
    
    tokened_stru = pad_trunc_tokened_text(tokened_stru)

    return tokened_stru

# def pad_label(labels):




if __name__ == "__main__":
    train_texts = openfile(os.path.join(ROOT, train_text))
    train_labels = openfile(os.path.join(ROOT, train_label))

    assert len(train_texts) == len(train_labels)
    print(f"{len(train_texts)} in total")

    for idx in range(len(train_texts)): 
        text = train_texts[idx]["train_{}".format(idx+1)]
        labels  = train_labels[idx][f"train_{idx+1}"]
        try:
            tokened_stru = build_token_stru(text)
        except:
            print(f"1 pass file train_{idx+1}")
            continue

       
        save_path = "data/head"
        try:
            tokened_struc = tag_head(labels, tokened_stru)
            with open(os.path.join(save_path, f"train_{idx+1}"), "w", encoding="utf-8", newline="") as f:
                for i in tokened_struc:
                    tag = i.token_tag 
                    tag = "O" if tag is None else tag
                    
                    f.write(f"{i.token_idx}\t{i.string}\t{i.pretrain_vocab_idx}\t{i.attn_mask}\t{i.string_start}\t{i.string_end}\t{tag}\n")

        except ValueError: 
            print(f"2 pass file train_{idx+1}")
            continue

