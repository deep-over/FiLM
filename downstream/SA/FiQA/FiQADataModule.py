import json
import re

import numpy as np
import pandas as pd

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import AutoTokenizer

class FiQADataset(Dataset) :
    def __init__(self, path, args) :
        self.tokenizer_name = args.tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.tok_sep_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        self.max_length = args.max_length

        # absa 세팅
        self.absa = args.absa
        self.aspect = args.aspect
        self.sep = args.sep
        
        # dataset
        df = pd.read_csv(path)
        # nan 제거
        df = df.dropna(axis=0)
        # 중복 제거
        df.drop_duplicates(subset=['sentence'], inplace=True)
        self.dataset = df

    def __len__(self) :
        return len(self.dataset)
    
    def get_target_idx(self, sentence, target) :
        '''
        sentence에서 target(aspect) 위치 찾기
        sentence(list) : sentence input_ids with paddings
        target(list) : target input_ids without padding
        '''
        # [SEP] token 위치 찾기
        sep_idx = list(sentence).index(self.tok_sep_id)

        for i in range(sep_idx+1, self.max_length-len(target)) :
            start = i
            end = i + len(target)
            if torch.equal(target, sentence[start:end]) :
                break
        return (start, end)

    def cleanse(self, sentence) :
        # url 삭제
        sentence = re.sub("http://[A-Za-z./0-9]+", "", sentence)
        
        # $[알파벳]에서 $ 삭제
        p = re.compile(r"\$[a-zA-Z]+")
        for f in p.findall(sentence) :
            sentence = sentence.replace(f, f[1:].upper())
        
        # @[알파벳] 삭제
        # sentence = re.sub("@[A-Za-z0-9]+", "", sentence)
        
        # @ 삭제
        sentence = re.sub("@", "", sentence)

        # 두 개 이상 점 하나로 통일
        sentence = re.sub("\.{2,}", ".", sentence)
        
        # 두 개 이상 스페이스 하나로 통일
        sentence = re.sub("\s{2,}", " ", sentence)

        return sentence
    
    def __getitem__(self, idx) :
        sentence = self.cleanse(self.dataset['sentence'].iloc[idx])
        score = self.dataset['score'].iloc[idx]
        if self.absa == 'none' :
            inputs = self.tokenizer(
                sentence,
                return_tensors='pt',
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                add_special_tokens=True
            )
        
            return {
                'input_ids' : inputs['input_ids'][0],
                'attention_mask' : inputs['attention_mask'][0],
                'score' : score
            }
        else :
            target_aspect = ''
            if self.absa == 'NLI-M' :
                # NLI 방식
                target_aspect = self.dataset['target'].iloc[idx]
            if self.aspect :
                target_aspect = ' '.join([target_aspect, self.sep, self.dataset['aspects'].iloc[idx][2:-2]])

            sentence = ' '.join([sentence, self.tokenizer.sep_token, target_aspect])
            inputs = self.tokenizer(
                sentence,
                return_tensors='pt',
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                add_special_tokens=True
            )

            # ' '가 있어야 같은 토큰으로 분류됨
            target_aspect = ' ' + target_aspect
            ta = self.tokenizer(
                target_aspect,
                return_tensors='pt',
                truncation=True,
                add_special_tokens=True
            )
            # target_idx : target_aspect 위치, 튜플 (start, end)
            target_idx = self.get_target_idx(inputs['input_ids'][0], ta['input_ids'][0][1:-1])
        
            return {
                'input_ids' : inputs['input_ids'][0],
                'attention_mask' : inputs['attention_mask'][0],
                'score' : score,
                'target_idx' : target_idx
            }
        
class FiQADataModule(pl.LightningDataModule) :
    def __init__(self, path, args) :
        '''
        path(dict) : train, valid, test 데이터셋 경로 담은 딕셔너리
        args : argument parser
        '''
        super().__init__()
        self.train_path = path['train']
        self.valid_path = path['valid']
        # self.test_path = path['test']
        
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.args = args

    def setup(self, stage=None) :
        self.set_train = FiQADataset(self.train_path, args=self.args)
        self.set_valid = FiQADataset(self.valid_path, args=self.args)
        # self.set_test = FiQADataset(self.test_path, args=self.args)

    def train_dataloader(self) :
        train = DataLoader(self.set_train, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=True)
        return train
    
    def val_dataloader(self) :
        valid = DataLoader(self.set_valid, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=False)
        return valid
    
    def test_dataloader(self) :
        test = DataLoader(self.set_test, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=False)
        return test