from collections import deque
import numpy as np
import pandas as pd

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from transformers import DataCollatorWithPadding

class FinREDDataset(Dataset) :
    def __init__(self, path, tokenizer, special_tokens, args) :
        self.tokenizer = tokenizer

        # special_tokens dictionary
        self.special_tokens = special_tokens

        # class_dict
        self.class_dict = self.make_class_dict()
        # text iterator
        self.txt = open(path, 'r')
        # make dataframe from iterator
        self.dataset = self.make_df(self.txt)
        # remove nan 
        self.datasef = self.dataset.dropna(axis=0)
        # remove duplicates
        self.dataset.drop_duplicates(subset=['sentence'], inplace=True)
    
    def __len__(self) :
        return len(self.dataset)

    def __getitem__(self, idx) :
        sentence = self.dataset['sentence'].iloc[idx]
        head = self.dataset['head'].iloc[idx]
        tail = self.dataset['tail'].iloc[idx]
        relation = self.dataset['relation'].iloc[idx]

        # apply head tokens
        sentence = sentence.replace(head, ' '.join([special_tokens['head_start'], head, special_tokens['head_end']]))
        # apply tail tokens
        sentence = sentence.replace(tail, ' '.join([special_tokens['tail_start'], tail, special_tokens['tail_end']]))

        inputs = self.tokenizer(
            sentence,
            truncation=False,
            add_special_tokens=True
        )

        return {
            'input_ids' : torch.tensor(inputs['input_ids']),
            'attention_mask' : torch.tensor(inputs['attention_mask']),
            'relation' : relation
        }
    
    # make dictionary from FinRED relations.txt
    def make_class_dict(self) :
        # path of relations.txt
        class_txt = open('/home/ailab/Desktop/NY/FinBERT/Financial-Pre-trained-research/downstream/data/FinRED/relations.txt', 'r')
        class_dict = {}
        for i, c_name in enumerate(class_txt) :
            class_dict[c_name.strip()] = i
        return class_dict

    # cleanse relations to make same with class_dict's keys
    def cleanse_relation(self, relation) :
        relation = relation.replace('_', ' ')
        relation = relation.replace(' or ', '/')
        relation = relation.replace(' / ', '/')
        return relation
    
    # make dataframe from text iterator
    def make_df(self, iterator) :
        sentence = []
        head = []
        tail = []
        relation = []
        for line in iterator :
            line = line.strip()
            splitted = deque(line.split(' | '))
            mini_sen = splitted.popleft()
            for triplet in splitted :
                try :
                    mini_head, mini_tail, mini_relation = triplet.split(' ; ')
                except ValueError : break
                sentence.append(mini_sen)
                head.append(mini_head)
                tail.append(mini_tail)
                relation.append(self.class_dict[self.cleanse_relation(mini_relation)])
        dataset = pd.DataFrame({'sentence' : sentence, 'head' : head, 'tail' : tail, 'relation' : relation})
        return dataset

class FinREDDataModule(pl.LightningDataModule) :
    def __init__(self, path, args, tokenizer, special_tokens) :
        super().__init__()
        self.train_path = path['train']
        self.valid_path = path['valid']
        self.test_path = path['test']

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        
        self.args = args

    def setup(self) :
        self.set_train = FinREDDataset(self.train_path, self.tokenizer, self.special_tokens, self.args)
        self.set_valid = FinREDDataset(self.valid_path, self.tokenizer, self.special_tokens, self.args)
        self.set_test = FinREDDataset(self.test_path, self.tokenizer, self.special_tokens, self.args)

    def train_dataloader(self) :
        train = DataLoader(self.set_train, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.data_collator, shuffle=True)
        return train

    def val_dataloader(self) :
        valid = DataLoader(self.set_valid, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.data_collator)
        return valid

    def test_dataloader(self) :
        test = DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.data_collator)
        return test