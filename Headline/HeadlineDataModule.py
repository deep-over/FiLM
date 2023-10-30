# Make Headline Data Module

#1. Import Modules
import re
import pandas as pd
import numpy as np
import torch
import spacy
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def split_data(df, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    df.rename(columns = {'News':'news', 'Price Sentiment':'sentiment'}, inplace = True)
    # drop na
    df = df.dropna()

    df = df.drop_duplicates(subset=['news', 'sentiment'])
    df = df[(df.sentiment != 'none')]

    train, val, test = np.split(df.sample(frac=1, random_state=random_state), [int(train_size*len(df)), int((train_size+val_size)*len(df))])
    return train, val, test

#3. Make Headline Dataset
class HeadlineDataset(Dataset):
    def __init__(self, df, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.labels_dict = {
            'negative' : 0,
            'neutral' : 1,
            'positive' : 2
        }
        self.spacy_tokenizer = spacy.load("en_core_web_sm")
        self.df = df

        # max_len is the maximum length in df['news']
        df_list = self.df['news'].tolist()
        max_len = 0
        headlineInputs = []
        for i, headline in enumerate(df_list):
            if isinstance(headline, str):
                tokens = tokenizer(headline)['input_ids']
                headlineInputs.append(tokens)
                max_len = max(max_len, len(tokens))
            else:
                # delete nan
                self.df = self.df.drop(i)
        self.max_len = 64

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        headline = self.df.iloc[index]['news']
        label = self.df.iloc[index]['sentiment']

        label = self.labels_dict[label]

        encoding = self.tokenizer.encode_plus(
            headline,
            add_special_tokens=True,
            truncation=True,
            return_token_type_ids=False,
            padding='max_length',
            max_length=self.max_len,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        label = torch.tensor(label, dtype=torch.long)

        return {
            'headline_text': headline,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label
        }

#3. Make Headline Data Class
class HeadlineDataModule(pl.LightningDataModule):
    def __init__(self, data_path, tokenizer, config=None):
        super().__init__()

        data_df = pd.read_csv(data_path, usecols=["Price Sentiment", "News"])

        self.tokenizer = tokenizer
        self.batch_size = config.batch_size
        self.config = config

        self.train_df, self.val_df, self.test_df = split_data(data_df, train_size=0.8, val_size=0.1, test_size=0.1, random_state=config.seed)

    def setup(self, stage=None):
        self.train_dataset = HeadlineDataset(self.train_df, self.tokenizer, self.config)
        self.val_dataset = HeadlineDataset(self.val_df, self.tokenizer, self.config)
        self.test_dataset = HeadlineDataset(self.test_df, self.tokenizer, self.config)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)
