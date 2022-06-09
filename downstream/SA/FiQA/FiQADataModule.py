import json
import re

import numpy as np
import pandas as pd

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer

class FiQADataset(Dataset) :
    def __init__(self, path, config) :
        self.model_name = config.model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.max_length = config.max_length
        
        # dataset
        df = pd.read_csv(path)
        # nan 제거
        df = df.dropna(axis=0)
        # 중복 제거
        df.drop_duplicates(subset=['sentence'], inplace=True)
        self.dataset = df
        
        self.cased = ['roberta-base', 'facebook/bart-base']
    
    def __len__(self) :
        return len(self.dataset)
    
    def cleanse(self, sentence) :
        # url 삭제
        sentence = re.sub("http://[A-Za-z./0-9]+", "", sentence)
        
        # $[알파벳]에서 $ 삭제
        p = re.compile(r"\$\w+")
        for f in p.findall(sentence) :
            sentence.replace(f, f[1:])
        
        # @[알파벳] 삭제
        sentence = re.sub("@[A-Za-z0-9]+", "", sentence)

        # 두 개 이상 점 하나로 통일
        sentence = re.sub("\.{2,}", ".", sentence)
        
        # 두 개 이상 스페이스 하나로 통일
        sentence = re.sub("\s{2,}", " ", sentence)

        if self.model_name in self.cased :
            return sentence
        else :
            return sentence.lower()
    
    def __getitem__(self, idx) :
        sentence = self.cleanse(self.dataset['sentence'].iloc[idx])
        
        inputs = self.tokenizer(
            sentence,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            add_special_tokens=True
        )
        
        score = self.dataset['score'].iloc[idx]
        
        return {
            'input_ids' : inputs['input_ids'][0],
            'attention_mask' : inputs['attention_mask'][0],
            'score' : score
        }
        
class FiQADataModule(pl.LightningDataModule) :
    def __init__(self, path, config) :
        '''
        path(dict) : train, valid, test 데이터셋 경로 담은 딕셔너리
        config(class) : config.py의 Config class로 생성한 것
        '''
        super().__init__()
        self.train_path = path['train']
        self.valid_path = path['valid']
        self.test_path = path['test']
        
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.config = config

    def setup(self, stage=None) :
        self.set_train = FiQADataset(self.train_path, config=self.config)
        self.set_valid = FiQADataset(self.valid_path, config=self.config)
        self.set_test = FiQADataset(self.test_path, config=self.config)

    def train_dataloader(self) :
        train = DataLoader(self.set_train, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=True)
        return train
    
    def val_dataloader(self) :
        valid = DataLoader(self.set_valid, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=False)
        return valid
    
    def test_dataloader(self) :
        test = DataLoader(self.set_test, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=False)
        return test