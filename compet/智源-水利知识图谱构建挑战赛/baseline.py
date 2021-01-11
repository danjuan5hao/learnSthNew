# -*- coding: utf-8 -*-

import os

from fancy_nlp.utils import load_ner_data_and_labels
from fancy_nlp.applications import NER

sample_file = r"智源-水利知识图谱构建挑战赛\data2\bmes_train_fancynlp_sample.txt"
# train_file = r'智源-水利知识图谱构建挑战赛\data2\bmes_train_fancynlp.txt'
# dev_file = r'智源-水利知识图谱构建挑战赛\data2\bmes_dev_fancynlp.txt'

checkpoint_dir = r'智源-水利知识图谱构建挑战赛\checkpoint_dir'
model_name = 'fancynlp_ner_bilstm_cnn_crf'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

sample_data, sample_labels = load_ner_data_and_labels(sample_file, delimiter=" ")
# train_data, train_labels = load_ner_data_and_labels(train_file)
# dev_data, dev_labels = load_ner_data_and_labels(dev_file)

ner = NER(use_pretrained=False)

ner.fit(sample_data, sample_labels, sample_data, sample_labels,
        ner_model_type='bilstm_cnn',
        char_embed_trainable=True,
        callback_list=['modelcheckpoint', 'earlystopping', 'swa'],
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
        load_swa_model=True)

ner.save(preprocessor_file=os.path.join(checkpoint_dir, f'{model_name}_preprocessor.pkl'),
         json_file=os.path.join(checkpoint_dir, f'{model_name}.json'))

ner.load(preprocessor_file=os.path.join(checkpoint_dir, f'{model_name}_preprocessor.pkl'),
         json_file=os.path.join(checkpoint_dir, f'{model_name}.json'),
         weights_file=os.path.join(checkpoint_dir, f'{model_name}_swa.hdf5'))

# print(ner.score(dev_data, dev_labels))
# print(ner.analyze(train_data[2]))
# print(ner.analyze_batch(train_data[:3]))
# print(ner.restrict_analyze(train_data[2]))
# print(ner.restrict_analyze_batch(train_data[:3]))
# print(ner.analyze('同济大学位于上海市杨浦区，校长为陈杰'))