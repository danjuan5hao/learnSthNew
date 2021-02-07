# -*- coding: utf-8 -*-
import json
import os, sys 

from transformers import AutoTokenizer

class DataConfig:
    ROOT = r"D:\data\数据资料\biendata\covid19赛道二关系抽取\task2_public"
    train_text = "task2_user_train.json"
    train_label = "task2_train_label.json"

    save_root = os.path.join(os.path.dirname(__file__), "data/head")


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


class BertTokenPreprocessor:
    """
    用来对数据集进行预处理，得到需要的形式，
    因为是seq tagging 任务，基础形式参照 subword-级别的NER任务标注。

    原数据标注根据char级别进行标注，需要改为根据tokenizer的vocab进行标注。
    具体做法如下，
    1. 用tokennizer对text分词， 记录每个token的长度
    2. 通过token长度，对齐char级别的标注。为token打上tag
    3. 因为会出现"[UNK]"，含有"[UNK]"数据不能自动打标，需人工打标或弃置。
    """
    pretrain_model_weight = "bert-base-cased"
    SEP_TOKEN = "[SEP]"
    CLS_TOKEN = "[CLS]"
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"

    TOKENIZER = AutoTokenizer.from_pretrained('bert-base-cased')
    PRETRAIN_VOCAB = TOKENIZER.vocab
    
    MAX_LEN = 100

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
        
        def pad_trunc_tokened_text(strus):
            strus = strus[:self.MAX_LEN] 
            strus = strus + [build_PAD_struc()]*(self.MAX_LEN-len(strus))
            return strus
        
        tokened_stru = pad_trunc_tokened_text(seq_token_stru)

        return tokened_stru
   
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

    def tag_head(self, seq_tag, seq_token_stru):
        """
        为seq_token_stru中的token_tag字段填充值
        """
        def find_tag_in_seq_token_stru(tag, seq_token_stru):
            head_word = tag.get("head").get("word")
            head_start = tag.get("head").get("start")
            head_end = tag.get("head").get("end")
                
            start_if = False
            end_if= False
            for i in range(len(seq_token_stru)):  # 是通过副作用来
                token_stru = seq_token_stru[i]

                if head_start == token_stru.string_start:
                    token_stru.token_tag = "B-head"
                    start_if = True
                if head_end == token_stru.string_end:
                    if token_stru.token_tag == "B-head":
                        token_stru.token_tag = "U-head"
                    else:
                        token_stru.token_tag = "E-head"
                    end_if = True
                if start_if and end_if:
                    break
            if start_if == False or end_if == False:
                raise ValueError(f"head:{head_word} not matched correctly. start: {head_start}, end: {head_end}")
            return 
        # for tag in seq_tag:
        
        for i in range(len(seq_tag)):
            tag = seq_tag[i]
            find_tag_in_seq_token_stru(tag, seq_token_stru)

            
               
        return seq_token_stru

def openfile(file):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)

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

    preprocessor = BertTokenPreprocessor()
    
    if not os.path.exists(DataConfig.save_root):
        os.makedirs(DataConfig.save_root)

    cnt = 0
    for idx in range(len(train_texts)): 
        text = train_texts[idx]["train_{}".format(idx+1)]
        seq_tag  = train_labels[idx][f"train_{idx+1}"]
        try:
            seq_token_stru = preprocessor.build_seq_token_stru(text)  # 还没有
        except ValueError as e:  # TODO 不能对齐就整个丢掉了，浪费数据，需要改。 主要是针对于"[UNK]"
            print(f"{idx}: {e}")
            cnt += 1
            continue

        try:
            seq_token_stru = preprocessor.tag_head(seq_tag, seq_token_stru)
            save_path = os.path.join(DataConfig.save_root, f"train_{idx+1}.txt")
            save(seq_token_stru, save_path)

        except ValueError as e: 
            print(f"{idx}: {e}")
            cnt += 1
            continue
    print(cnt)

