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
        pretrained_model_name_or_path=conf.model_tokenizer,
        additional_special_tokens=additional_special_tokens
    )
else:
    tokenizer = tokenizer

# create output paths
if conf.mode == "train":
    model_dir_name = conf.model_save_name + "_" + \
        datetime.now().strftime("%Y%m%d%H%M%S")
    model_dir = os.path.join(conf.output_path, model_dir_name)
    results_path = os.path.join(model_dir, "results")
    saved_model_path = os.path.join(model_dir, "saved_model")
    os.makedirs(saved_model_path, exist_ok=False)
    os.makedirs(results_path, exist_ok=False)
    log_file = os.path.join(results_path, 'log.txt')

else:
    saved_model_path = os.path.join(conf.output_path, conf.saved_model_path)
    model_dir_name = datetime.now().strftime("%Y%m%d%H%M%S")
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
          "is_training": True,
          "max_seq_length": conf.max_seq_length,
          }

# exsit_flag is used to check if the features have been generated
exist_flag = True
model_tokenizer_name = conf.model_tokenizer.split('/')[-1]
if not exist_flag:
    train_features = convert_examples_to_features(**kwargs)
    kwargs["examples"] = valid_examples
    kwargs["is_training"] = False
    valid_features = convert_examples_to_features(**kwargs)
    kwargs["examples"] = test_examples
    test_features = convert_examples_to_features(**kwargs)
    with open(os.path.join(f'{model_tokenizer_name}_features',f'{model_tokenizer_name}_{conf.shape_token}_train_features.pkl'), 'wb') as fp:
        pickle.dump(train_features, fp)
    with open(os.path.join(f'{model_tokenizer_name}_features',f'{model_tokenizer_name}_{conf.shape_token}_valid_features.pkl'), 'wb') as vfp:
        pickle.dump(valid_features, vfp)
    with open(os.path.join(f'{model_tokenizer_name}_features',f'{model_tokenizer_name}_{conf.shape_token}_test_features.pkl'), 'wb') as tfp:
        pickle.dump(test_features, tfp)

else:
    with open(os.path.join(f"{model_tokenizer_name}_features",f'{model_tokenizer_name}_{conf.shape_token}_train_features.pkl'), 'rb') as fp :
        train_features = pickle.load(fp)
    with open(os.path.join(f"{model_tokenizer_name}_features",f'{model_tokenizer_name}_{conf.shape_token}_valid_features.pkl'), 'rb') as vfp :
        valid_features = pickle.load(vfp)
    with open(os.path.join(f"{model_tokenizer_name}_features",f'{model_tokenizer_name}_{conf.shape_token}_test_features.pkl'), 'rb') as tfp :
        test_features = pickle.load(tfp)
       
def train():
    # keep track of all input parameters
    write_log(log_file, "####################INPUT PARAMETERS###################")
    for attr in conf.__dict__:
        value = conf.__dict__[attr]
        write_log(log_file, attr + " = " + str(value))
    write_log(log_file, "#######################################################")
    model = Bert_model(hidden_size=model_config.hidden_size,
                       dropout_rate=conf.dropout_rate,tokenizer=tokenizer)

    # model = nn.DataParallel(model)
    print(conf.device)
    model.to(conf.device)
    optimizer = optim.Adam(model.parameters(), conf.learning_rate)
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    model.train()

    train_iterator = DataLoader(
        is_training=True, data=train_features, batch_size=conf.batch_size, shuffle=True)

    k = 0
    record_k = 0
    record_loss_k = 0
    loss, start_time = 0.0, time.time()
    record_loss = 0.0
    best_result = 0.0
    for epoch in range(conf.epoch):
        train_iterator.reset()
        for x in tqdm(train_iterator):

            input_ids = torch.tensor(x['input_ids']).to(conf.device)
            input_mask = torch.tensor(x['input_mask']).to(conf.device)
            segment_ids = torch.tensor(x['segment_ids']).to(conf.device)
            label = torch.tensor(x['label']).to(conf.device)

            model.zero_grad()
            optimizer.zero_grad()

            this_logits = model(True, input_ids, input_mask,
                                segment_ids, device=conf.device)

            this_loss = criterion(
                this_logits.view(-1, this_logits.shape[-1]), label.view(-1))

            this_loss = this_loss.sum()
            record_loss += this_loss.item() * 100
            record_k += 1
            k += 1

            this_loss.backward()
            optimizer.step()

            if k > 1 and k % conf.report_loss == 0:
                write_log(log_file, "%d : loss = %.3f epoch = %d" %
                          (k, record_loss / record_k, epoch))
                record_loss = 0.0
                record_k = 0

            if k > 1 and k % conf.report == 0:
                print("Round: ", k / conf.report)
                model.eval()
                cost_time = time.time() - start_time
                write_log(log_file, "%d : time = %.3f " %
                          (k // conf.report, cost_time))
                start_time = time.time()
                if k // conf.report >= 1:
                    print("Val test")
                    # save model
                    saved_model_path_cnt = os.path.join(
                        saved_model_path, 'loads', str(k // conf.report))
                    os.makedirs(saved_model_path_cnt, exist_ok=True)
                    if k // conf.report > 100 and k % 5 == 0 :
                        torch.save(model.state_dict(),
                                saved_model_path_cnt + "/model.pt")

                    results_path_cnt = os.path.join(
                        results_path, 'loads', str(k // conf.report))
                    os.makedirs(results_path_cnt, exist_ok=True)
                    validation_result = evaluate(
                        valid_examples, valid_features, model, results_path_cnt, 'valid')
                    # write_log(log_file, validation_result)
                    if validation_result > best_result:
                        best_result = validation_result
                        torch.save(model.state_dict(),
                                   saved_model_path + "/model.pt")
                        write_log(log_file, "Best result: " + str(best_result))
                    

                model.train()


def evaluate(data_ori, data, model, ksave_dir, mode='valid'):

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
                                          "predictions.json")

    if mode == "valid":
        print_res, res_3 = retrieve_evaluate(
            all_logits, all_filename_id, all_ind, output_prediction_file, conf.valid_file, topn=conf.topn)
    else:
        print_res, res_3 = retrieve_evaluate(
            all_logits, all_filename_id, all_ind, output_prediction_file, conf.test_file, topn=conf.topn)

    write_log(log_file, print_res)
    print(print_res)
    return res_3


if __name__ == '__main__':
    train()


