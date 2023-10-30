#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script
"""
from tqdm import tqdm
import json
import spacy
import pickle
import os
from datetime import datetime
import time
import logging
from utils import *
from config import parameters as conf
from torch import nn
import torch
import torch.optim as optim
from Model import Bert_model
from transformers import AutoTokenizer, AutoConfig

# shape_special_tokens_path = '/home/ailab/Desktop/ASY/FiNER-139/finer/data/shape_special_tokens.txt'
shape_special_tokens_path = ''

tokenizer = AutoTokenizer.from_pretrained(conf.model_tokenizer)
model_config = AutoConfig.from_pretrained(conf.model_size)

if conf.shape_token == '[SHAPE]':
    with open(shape_special_tokens_path) as fin:
        shape_special_tokens = [shape.strip() for shape in fin.readlines()]
    shape_special_tokens_set = set(shape_special_tokens)

    additional_special_tokens = []
    additional_special_tokens.append('[NUM]')
    additional_special_tokens.extend(shape_special_tokens)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=conf.model_size,
        additional_special_tokens=additional_special_tokens
    )
else:
    tokenizer = tokenizer

saved_model_path = os.path.join(conf.output_path, conf.saved_model_path)
# model_dir_name = datetime.now().strftime("%Y%m%d%H%M%S") + \
#     "_" + conf.model_save_name
model_dir_name = conf.model_tokenizer.split("/")[-1]
model_dir = os.path.join(
    conf.output_path, 'inference_only_' + model_dir_name)
results_path = os.path.join(model_dir, "results")
os.makedirs(results_path, exist_ok=False)
log_file = os.path.join(results_path, 'log.txt')


op_list = read_txt(conf.op_list_file, log_file)
op_list = [op + '(' for op in op_list]
op_list = ['EOF', 'UNK', 'GO', ')'] + op_list
const_list = read_txt(conf.const_list_file, log_file)
const_list = [const.lower().replace('.', '_') for const in const_list]
reserved_token_size = len(op_list) + len(const_list)

train_data, train_examples, op_list, const_list = \
read_examples(input_path=conf.train_file, tokenizer=tokenizer,
                op_list=op_list, const_list=const_list, log_file=log_file)
valid_data, valid_examples, op_list, const_list = \
read_examples(input_path=conf.valid_file, tokenizer=tokenizer,
                op_list=op_list, const_list=const_list, log_file=log_file)

test_data, test_examples, op_list, const_list = \
read_examples(input_path=conf.test_file, tokenizer=tokenizer,
                op_list=op_list, const_list=const_list, log_file=log_file)

kwargs = {"examples": train_examples,
          "tokenizer": tokenizer,
          "option": conf.option,
          "is_training": False,
          "max_seq_length": conf.max_seq_length,
          }

exist_flag = True
model_tokenizer_name = conf.model_tokenizer.split('/')[-1]
if not exist_flag:
    train_features = convert_examples_to_features(**kwargs)
    kwargs["examples"] = valid_examples
    kwargs["is_training"] = False
    valid_features = convert_examples_to_features(**kwargs)
    kwargs["examples"] = test_examples
    test_features = convert_examples_to_features(**kwargs)
    with open(os.path.join(f'{model_tokenizer_name}_features',f'{model_tokenizer_name}_{conf.shape_token}_train_features.pkl'), 'wb') as tfp:
        pickle.dump(train_features, tfp)
    with open(os.path.join(f'{model_tokenizer_name}_features',f'{model_tokenizer_name}_{conf.shape_token}_valid_features.pkl'), 'wb') as tfp:
        pickle.dump(valid_features, tfp)
    with open(os.path.join(f'{model_tokenizer_name}_features',f'{model_tokenizer_name}_{conf.shape_token}_test_features.pkl'), 'wb') as tfp:
        pickle.dump(test_features, tfp)
        
else:
    with open(os.path.join(f"{model_tokenizer_name}_features",f'{model_tokenizer_name}_{conf.shape_token}_train_features.pkl'), 'rb') as tfp :
        train_features = pickle.load(tfp)
    with open(os.path.join(f"{model_tokenizer_name}_features",f'{model_tokenizer_name}_{conf.shape_token}_valid_features.pkl'), 'rb') as tfp :
        valid_features = pickle.load(tfp)
    with open(os.path.join(f"{model_tokenizer_name}_features",f'{model_tokenizer_name}_{conf.shape_token}_test_features.pkl'), 'rb') as tfp :
        test_features = pickle.load(tfp)


def generate(data_name, data, model, ksave_dir, mode='valid'):

    pred_list = []
    pred_unk = []

    ksave_dir_mode = os.path.join(ksave_dir, mode)
    os.makedirs(ksave_dir_mode, exist_ok=True)

    data_iterator = DataLoader(
        is_training=False, data=data, batch_size=conf.batch_size_test, shuffle=False)

    k = 0
    all_logits = []
    all_filename_id = []
    all_ind = []
    with torch.no_grad():
        for x in tqdm(data_iterator):

            input_ids = x['input_ids']
            input_mask = x['input_mask']
            segment_ids = x['segment_ids']
            label = x['label']
            filename_id = x["filename_id"]
            ind = x["ind"]

            ori_len = len(input_ids)
            for each_item in [input_ids, input_mask, segment_ids]:
                if ori_len < conf.batch_size_test:
                    each_len = len(each_item[0])
                    pad_x = [0] * each_len
                    each_item += [pad_x] * (conf.batch_size_test - ori_len)

            input_ids = torch.tensor(input_ids).to(conf.device)
            input_mask = torch.tensor(input_mask).to(conf.device)
            segment_ids = torch.tensor(segment_ids).to(conf.device)

            logits = model(True, input_ids, input_mask,
                           segment_ids, device=conf.device)

            all_logits.extend(logits.tolist())
            all_filename_id.extend(filename_id)
            all_ind.extend(ind)

    output_prediction_file = os.path.join(ksave_dir_mode,
                                          f"{data_name}_predictions.json")

    if mode == "valid":
        print_res = retrieve_evaluate(
            all_logits, all_filename_id, all_ind, output_prediction_file, conf.valid_file, topn=conf.topn)
    elif mode == "test":
        print_res = retrieve_evaluate(
            all_logits, all_filename_id, all_ind, output_prediction_file, conf.test_file, topn=conf.topn)
    else:
        # private data mode
        print_res = retrieve_evaluate_private(
            all_logits, all_filename_id, all_ind, output_prediction_file, conf.test_file, topn=conf.topn)

    write_log(log_file, print_res)
    print(print_res)
    return


def generate_test():
    model = Bert_model(hidden_size=model_config.hidden_size,
                       dropout_rate=conf.dropout_rate,tokenizer = tokenizer)

    # model = nn.DataParallel(model)
    model.to(conf.device)
    model.load_state_dict(torch.load(conf.saved_model_path))
    model.eval()
    generate("train", train_features, model, results_path, mode='test')
    generate("valid", valid_features, model, results_path, mode='test')
    generate("test", test_features, model, results_path, mode='test')


if __name__ == '__main__':
    generate_test()
