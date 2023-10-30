#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script
"""
from tqdm import tqdm
import json
import os
from datetime import datetime
import time
import logging
import pickle
from utils import *
from config import parameters as conf
from torch import nn
import torch
import torch.optim as optim
from Model_new import Bert_model
from transformers import AutoTokenizer, RobertaTokenizer, AutoConfig

shape_special_tokens_path = ''

tokenizer = AutoTokenizer.from_pretrained(conf.model_tokenizer)
model_config = AutoConfig.from_pretrained(conf.model_size)

# create output paths
conf.model_save_name = conf.model_size.split("/")[-1]
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

print(op_list)
print(const_list)

model_tokenizer_name = conf.model_tokenizer.split('/')[-1]
data_dir = conf.root_path + "/" + model_tokenizer_name + "-" + conf.prog_name

# create data dir
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    exist_flag = False
else:
    exist_flag = True

# train dataset
if not exist_flag:
    train_data, train_examples, op_list, const_list = \
    read_examples(input_path=conf.train_file, tokenizer=tokenizer,
                    op_list=op_list, const_list=const_list, log_file=log_file)
    with open(os.path.join(data_dir, 'train_data_read_example.pickle'), 'wb') as file:
        pickle.dump(train_data, file)
    with open(os.path.join(data_dir, 'train_examples.pickle'), 'wb') as fp:
        pickle.dump(train_examples, fp)
else:
    with open(os.path.join(data_dir, 'train_data_read_example.pickle'), 'rb') as file:
        train_data = pickle.load(file)
    with open(os.path.join(data_dir, 'train_examples.pickle'), 'rb') as fp :
        train_examples = pickle.load(fp)

# valid dataset
if not exist_flag:
    valid_data, valid_examples, op_list, const_list = \
    read_examples(input_path=conf.valid_file, tokenizer=tokenizer,
                    op_list=op_list, const_list=const_list, log_file=log_file)
    with open(os.path.join(data_dir,'val_data_read_example.pickle'), 'wb') as file:
        pickle.dump(valid_data, file)
    with open(os.path.join(data_dir,'val_examples.pickle'), 'wb') as fp:
        pickle.dump(valid_examples, fp)
else:
    with open(os.path.join(data_dir,'val_data_read_example.pickle'), 'rb') as file:
        valid_data = pickle.load(file)
    with open(os.path.join(data_dir,'val_examples.pickle'), 'rb') as fp :
        valid_examples = pickle.load(fp)

# test dataset
if not exist_flag:
    test_data, test_examples, op_list, const_list = \
    read_examples(input_path=conf.test_file, tokenizer=tokenizer,
                    op_list=op_list, const_list=const_list, log_file=log_file)
    with open(os.path.join(data_dir,'test_data_read_example.pickle'), 'wb') as file:
        pickle.dump(test_data, file)
    with open(os.path.join(data_dir,'test_examples.pickle'), 'wb') as fp:
        pickle.dump(test_examples, fp)
else:
    with open(os.path.join(data_dir,'test_data_read_example.pickle'), 'rb') as file:
        test_data = pickle.load(file)
    with open(os.path.join(data_dir,'test_examples.pickle'), 'rb') as fp :
        test_examples = pickle.load(fp)

kwargs = {"examples": train_examples,
          "tokenizer": tokenizer,
          "max_seq_length": conf.max_seq_length,
          "max_program_length": conf.max_program_length,
          "is_training": True,
          "op_list": op_list,
          "op_list_size": len(op_list),
          "const_list": const_list,
          "const_list_size": len(const_list),
          "verbose": True}

train_features = convert_examples_to_features(**kwargs)
kwargs["examples"] = valid_examples
kwargs["is_training"] = False
valid_features = convert_examples_to_features(**kwargs)
kwargs["examples"] = test_examples
test_features = convert_examples_to_features(**kwargs)


def train():
    # keep track of all input parameters
    write_log(log_file, "####################INPUT PARAMETERS###################")
    for attr in conf.__dict__:
        value = conf.__dict__[attr]
        write_log(log_file, attr + " = " + str(value))
    write_log(log_file, "#######################################################")

    model = Bert_model(num_decoder_layers=conf.num_decoder_layers,
                       hidden_size=model_config.hidden_size,
                       dropout_rate=conf.dropout_rate,
                       program_length=conf.max_program_length,
                       input_length=conf.max_seq_length,
                       op_list=op_list,
                       const_list=const_list, tokenizer=tokenizer)

    # model = nn.DataParallel(model)
    model.to(conf.device)
    optimizer = optim.Adam(model.parameters(), conf.learning_rate)
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    model.train()
    # torch.autograd.set_detect_anomaly(True)

    train_iterator = DataLoader(
        is_training=True, data=train_features, batch_size=conf.batch_size, reserved_token_size=reserved_token_size, shuffle=True)

    k = 0
    record_k = 0
    record_loss_k = 0
    loss, start_time = 0.0, time.time()
    record_loss = 0.0
    best_result = 0.0
    for epoch in tqdm(range(conf.epoch)):
        write_log(log_file,"epoch = %d" % epoch)
        train_iterator.reset()
        for x in tqdm(train_iterator):

            input_ids = torch.tensor(x['input_ids']).to(conf.device)
            input_mask = torch.tensor(x['input_mask']).to(conf.device)
            segment_ids = torch.tensor(x['segment_ids']).to(conf.device)
            program_ids = torch.tensor(x['program_ids']).to(conf.device)
            program_mask = torch.tensor(x['program_mask']).to(conf.device)
            option_mask = torch.tensor(x['option_mask']).to(conf.device)

            model.zero_grad()
            optimizer.zero_grad()

            this_logits = model(True, input_ids, input_mask, segment_ids,
                                option_mask, program_ids, program_mask, device=conf.device)

            this_loss = criterion(
                this_logits.view(-1, this_logits.shape[-1]), program_ids.view(-1))
            this_loss = this_loss * program_mask.view(-1)
            # per token loss
            this_loss = this_loss.sum() / program_mask.sum()

            record_loss += this_loss.item()
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
                    # if k // conf.report > 100 and k % 5 == 0 :
                    #     torch.save(model.state_dict(),
                    #             saved_model_path_cnt + "/model.pt")
                    
                    results_path_cnt = os.path.join(
                        results_path, 'loads', str(k // conf.report))
                    os.makedirs(results_path_cnt, exist_ok=True)
                    validation_result = evaluate(
                        valid_examples, valid_features, model, results_path_cnt, 'valid')

                    if validation_result > best_result:
                        best_result = validation_result
                        write_log(log_file, f"best result: {best_result}")
                        torch.save(model.state_dict(),
                            saved_model_path_cnt + "/model.pt")

                    # write_log(log_file, validation_result)

                model.train()


def evaluate(data_ori, data, model, ksave_dir, mode='valid'):

    pred_list = []
    pred_unk = []

    ksave_dir_mode = os.path.join(ksave_dir, mode)
    os.makedirs(ksave_dir_mode, exist_ok=True)

    data_iterator = DataLoader(
        is_training=False, data=data, batch_size=conf.batch_size_test, reserved_token_size=reserved_token_size, shuffle=False)

    k = 0
    all_results = []
    with torch.no_grad():
        for x in tqdm(data_iterator):

            input_ids = x['input_ids']
            input_mask = x['input_mask']
            segment_ids = x['segment_ids']
            program_ids = x['program_ids']
            program_mask = x['program_mask']
            option_mask = x['option_mask']

            ori_len = len(input_ids)
            for each_item in [input_ids, input_mask, segment_ids, program_ids, program_mask, option_mask]:
                if ori_len < conf.batch_size_test:
                    each_len = len(each_item[0])
                    pad_x = [0] * each_len
                    each_item += [pad_x] * (conf.batch_size_test - ori_len)

            input_ids = torch.tensor(input_ids).to(conf.device)
            input_mask = torch.tensor(input_mask).to(conf.device)
            segment_ids = torch.tensor(segment_ids).to(conf.device)
            program_ids = torch.tensor(program_ids).to(conf.device)
            program_mask = torch.tensor(program_mask).to(conf.device)
            option_mask = torch.tensor(option_mask).to(conf.device)

            logits = model(False, input_ids, input_mask,
                           segment_ids, option_mask, program_ids, program_mask, device=conf.device)

            for this_logit, this_id in zip(logits.tolist(), x["unique_id"]):
                all_results.append(
                    RawResult(
                        unique_id=int(this_id),
                        logits=this_logit,
                        loss=None
                    ))

    output_prediction_file = os.path.join(ksave_dir_mode,
                                          "predictions.json")
    output_nbest_file = os.path.join(ksave_dir_mode,
                                     "nbest_predictions.json")
    output_eval_file = os.path.join(ksave_dir_mode, "full_results.json")
    output_error_file = os.path.join(ksave_dir_mode, "full_results_error.json")

    all_predictions, all_nbest = compute_predictions(
        data_ori,
        data,
        all_results,
        n_best_size=conf.n_best_size,
        max_program_length=conf.max_program_length,
        tokenizer=tokenizer,
        op_list=op_list,
        op_list_size=len(op_list),
        const_list=const_list,
        const_list_size=len(const_list))
    write_predictions(all_predictions, output_prediction_file)
    write_predictions(all_nbest, output_nbest_file)

    if mode == "valid":
        original_file = conf.valid_file
    else:
        original_file = conf.test_file

    exe_acc, prog_acc = evaluate_result(
        output_nbest_file, original_file, output_eval_file, output_error_file, program_mode=conf.program_mode)

    prog_res = "exe acc: " + str(exe_acc) + " prog acc: " + str(prog_acc)
    write_log(log_file, prog_res)

    return prog_acc


if __name__ == '__main__':

    if conf.mode == "train":
        train()
