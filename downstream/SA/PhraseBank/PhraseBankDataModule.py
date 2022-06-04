import re

import numpy as np
import pandas as pd

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer

class PhraseBankDataset(Dataset) :
    def __init__(self, path, max_length, config) :
        '''
        path(str) : csv path
        max_length : padding할 최대 길이
        config : tokenizer, model 종류 담는 class
        '''
        # model 이름 저장
        self.model_name = config.model
        
        # tokenizer 지정
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # max_length
        self.max_length = max_length
        # labels_dict : string형의 라벨을 정수로 변환
        self.labels_dict = {
            'negative' : 0,
            'neutral' : 1,
            'positive' : 2
        }
        
        # dataset
        df = pd.read_csv(path, encoding='ISO-8859-1')
        # nan 제거
        df = df.dropna(axis=0)
        # 중복 제거
        df.drop_duplicates(subset=['sentences'], inplace=True)
        self.dataset = df
        
        self.cased = ['roberta-base']
        
    def __len__(self) :
        return len(self.dataset)
    
    def cleanse(self, sentences) :
        '''
        preprocessing
        sentences(str) : 문장
        '''
        # 특수문자 제거
#         sentences = re.sub('[^A-Za-z0-9]+', ' ', sentences)

        # cased model인 경우 소문자화 하지 않고 return
        if self.model_name in self.cased :
            return sentences
        # uncased model인 경우 소문자화 하여 return
        else :
            return sentences.lower()
    
    def __getitem__(self, idx) :
        '''
        dataset의 원소를 idx에서 하나서 빼와서 나중에 batch 만들게 됨
        '''
        # 문장 클렌징
        sentences = self.cleanse(self.dataset['sentences'].iloc[idx])
        # 토크나이저로 클렌징된 문장 토크나이징
        inputs = self.tokenizer(
            sentences,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            add_special_tokens=True
        )
        
        # laebl 가공
        label = self.labels_dict[self.dataset['labels'].iloc[idx]]
        
        return {
            'input_ids' : inputs['input_ids'][0],
            'attention_mask' : inputs['attention_mask'][0],
            'label' : label
        }
    

class PhraseBankDataModule(pl.LightningDataModule) :
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
        self.max_length = config.max_length
        self.num_workers = config.num_workers
        self.config = config

    def setup(self, stage=None) :
        self.set_train = PhraseBankDataset(self.train_path, max_length=self.max_length, config=self.config)
        self.set_valid = PhraseBankDataset(self.valid_path, max_length=self.max_length, config=self.config)
        self.set_test = PhraseBankDataset(self.test_path, max_length=self.max_length, config=self.config)

    def train_dataloader(self) :
        train = DataLoader(self.set_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return train
    
    def val_dataloader(self) :
        valid = DataLoader(self.set_valid, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return valid
    
    def test_dataloader(self) :
        test = DataLoader(self.set_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return test