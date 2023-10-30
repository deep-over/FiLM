import re

import numpy as np
import pandas as pd
import spacy
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

class PhraseBankDataset(Dataset) :
    def __init__(self, path, max_length, config) :
        self.tokenizer = config.tokenizer
        self.spacy_tokenizer = spacy.load("en_core_web_sm")
        
        # max_length
        self.max_length = max_length
        self.labels_dict = {
            'negative' : 0,
            'neutral' : 1,
            'positive' : 2
        }
        
        # if path is not str
        if not isinstance(path, str) :
            df = path
            df = df.dropna(axis=0)
            df.drop_duplicates(subset=['sentences'], inplace=True)
            self.dataset = df
        else:
            # dataset
            df = pd.read_csv(path, encoding='ISO-8859-1')
            df = df.dropna(axis=0)
            df.drop_duplicates(subset=['sentences'], inplace=True)
            self.dataset = df
        
        self.cased = ['roberta-base', 'facebook/bart-base']
        
    def __len__(self) :
        return len(self.dataset)
    
    def numeric_token_replace(self, sentence):
        tokens = [t.text for t in self.spacy_tokenizer.tokenizer(sentence)]

        return " ".join(tokens)
    
    def __getitem__(self, idx) :
        # sentence cleaning
        sentences = self.dataset['sentences'].iloc[idx]
        sentences = self.numeric_token_replace(sentences)
        inputs = self.tokenizer(
            sentences,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            add_special_tokens=True
        )
        
        label = self.labels_dict[self.dataset['labels'].iloc[idx]]
        
        return {
            'input_ids' : inputs['input_ids'][0],
            'attention_mask' : inputs['attention_mask'][0],
            'label' : label
        }


class PhraseBankDataModule(pl.LightningDataModule) :
    def __init__(self, path, config) :
        super().__init__()
        self.train_path = path['train']
        self.valid_path = path['valid']
        self.test_path = path['test']
        
        self.batch_size = config.batch_size
        self.max_length = config.max_length
        self.num_workers = config.num_workers
        self.config = config
        self.setup()

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
    
class CrossValidationDataModule(pl.LightningDataModule) :
    def __init__(self, train_df, valid_df, config) :
        super().__init__()

        self.train_df = train_df
        self.valid_df = valid_df
        
        self.batch_size = config.batch_size
        self.max_length = config.max_length
        self.num_workers = config.num_workers
        self.config = config
        self.setup()

    def setup(self, stage=None) :
        self.set_train = PhraseBankDataset(self.train_df, max_length=self.max_length, config=self.config)
        self.set_valid = PhraseBankDataset(self.valid_df, max_length=self.max_length, config=self.config)

    def train_dataloader(self) :
        train = DataLoader(self.set_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return train
    
    def val_dataloader(self) :
        valid = DataLoader(self.set_valid, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return valid    