import os

import torch
import torchmetrics
from torchmetrics import MeanSquaredError
from torchmetrics import R2Score
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import transformers
from transformers import AdamW, AutoModel, AutoConfig

device = torch.device("cuda")

class RegressionHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, dropout, configuration):
        super().__init__()
        self.dense = nn.Linear(configuration.hidden_size, configuration.hidden_size)
        regresser_dropout = dropout
        self.dropout = nn.Dropout(regresser_dropout)
        self.out_proj = nn.Linear(configuration.hidden_size, 1)

    def forward(self, features, **kwargs):
        # features shape : [batch, seq_length, hidden_size]
        
        # x shape : [batch, hidden_size] -> [CLS]토큰만 모음
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x) # x shape : [batch, hidden_size]
        x = torch.tanh(x) # activation
        x = self.dropout(x)
        x = self.out_proj(x) # x shape : [batch, 1]
        return x
    
class FiQARegression(pl.LightningModule) :
    def __init__(self, config) :
        super().__init__()
        
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.max_length = config.max_length
        self.dropout = config.dropout
        
        # pretrained 모델 configuration
        self.configuration = AutoConfig.from_pretrained(config.model)
        
        # 하이퍼파라미터 저장
        self.save_hyperparameters()
        
        # pretrained 모델 불러오기
        self.model = AutoModel.from_pretrained(config.model)
        
        self.regresser = RegressionHead(self.dropout, self.configuration)
        
        self.loss_func = MeanSquaredError()
        self.r2score = R2Score()
            
    def forward(self, input_ids, attention_mask) :
        outputs = self.model(
            input_ids,
            attention_mask
        )
        
        # last hidden states
        sequence_outputs = outputs[0]
        logits = self.regresser(sequence_outputs)
        logits = logits.view([-1])
        
        return logits
    
    def step(self, batch, batch_idx) :
        score = batch['score'].to(device)
        
        output = self(
            input_ids = batch['input_ids'].to(device),
            attention_mask = batch['attention_mask'].to(device)
        )
        
        loss = self.loss_func(output, score)
        y_true = score.cpu()
        y_pred = output.cpu()
        r2 = self.r2score(y_pred, y_true)
        
        return {
            'loss' : loss,
            'r2' : r2
        }
    
    def training_step(self, batch, batch_idx) :
        return self.step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx) :
        return self.step(batch, batch_idx)
    
    def training_epoch_end(self, outputs, state='train') :
        train_loss = torch.tensor(0, dtype=torch.float)
        for i in outputs :
            train_loss += i['loss'].cpu().detach()
        
        train_loss = train_loss / len(outputs)
        train_r2 = torch.stack([i['r2'] for i in outputs]).mean()
        
        print(f'[Epoch {self.trainer.current_epoch} {state.upper()}] Loss:{train_loss}, R2:{train_r2}')
        
    def validation_epoch_end(self, outputs, state='val') :
        val_loss = torch.tensor(0, dtype=torch.float)
        for i in outputs :
            val_loss += i['loss'].cpu().detach()
        val_r2 = torch.stack([i['r2'] for i in outputs]).mean()
        val_loss = val_loss / len(outputs)
        
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        self.log('val_r2', val_r2, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self) :
        # optimizer 설정 가능
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        # learning scheduler 설정가능
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        
        return {
            'optimizer': optimizer,
            'scheduler': lr_scheduler,
        }
