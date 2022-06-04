import os

import torch
import torchmetrics
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import transformers
from transformers import AdamW, AutoModel, AutoConfig

device = torch.device("cuda")

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, configuration):
        super().__init__()
        self.dense = nn.Linear(configuration.hidden_size, configuration.hidden_size)
        classifier_dropout = (
            configuration.classifier_dropout if configuration.classifier_dropout is not None else configuration.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(configuration.hidden_size, 3)

    def forward(self, features, **kwargs):
        # features shape : [batch, seq_length, hidden_size]
        
        # x shape : [batch, hidden_size] -> [CLS]토큰만 모음
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x) # x shape : [batch, hidden_size]
        x = torch.tanh(x) # activation
        x = self.dropout(x)
        x = self.out_proj(x) # x shape : [batch, 3(num_labels)]
        return x
    
class PhraseBankClassification(pl.LightningModule) :
    def __init__(self, config) :
        super().__init__()
        '''
        config(class) : hyperparameter config
        '''
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.max_length = config.max_length
        
        # pretrained 모델 configuration
        self.configuration = AutoConfig.from_pretrained(config.model)
        
        # 하이퍼파라미터 저장
        self.save_hyperparameters()
        
        # pretrained 모델 불러오기
        self.model = AutoModel.from_pretrained(config.model)
        
        self.classifier = ClassificationHead(self.configuration)
        
        self.metric_acc = torchmetrics.Accuracy(num_classs=3)
        self.metric_f1 = torchmetrics.F1Score(num_classes=3)
        
        # loss function
        self.loss_func = nn.CrossEntropyLoss()
        
        
    def forward(self, input_ids, attention_mask, labels=None) :
        outputs = self.model(
            input_ids,
            attention_mask
        )
        
        # last hidden state
        sequence_outputs = outputs[0]
        logits = self.classifier(sequence_outputs)
        
        return logits
    
    def step(self, batch, batch_idx, state=None) :
        label = batch['label'].to(device) # label shape : [batch]
        
        output = self(
            input_ids = batch['input_ids'].to(device),
            attention_mask = batch['attention_mask'].to(device)
        )
        
        loss = self.loss_func(output, label)
        
        self.log(f"[{state.upper()} LOSS]", loss, prog_bar=True)
        
        y_true = label.cpu()
        y_pred = output.argmax(dim=1).cpu()

        accuracy = self.metric_acc(y_pred, y_true)
        f1 = self.metric_f1(y_pred, y_true)
        return {
            'loss' : loss,
            'accuracy' : accuracy,
            'f1' : f1
              
        }
    
    def training_step(self, batch, batch_idx) :
        return self.step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx) :
        return self.step(batch, batch_idx, 'validation')
    
    # epoch 끝날 때마다 loss, accuracy, f1 score 평균내어 프린트함
    def training_epoch_end(self, outputs, state='train') :
        train_loss = torch.tensor(0, dtype=torch.float)
        for i in outputs :
            train_loss += i['loss'].cpu().detach()
        # epoch 평균 loss
        train_loss = train_loss / len(outputs)
        
        train_acc = torch.stack([i['accuracy'] for i in outputs]).mean()
        train_f1 = torch.stack([i['f1'] for i in outputs]).mean()
        
        print(f'[Epoch {self.trainer.current_epoch} {state.upper()}] Loss:{train_loss}, Acc: {train_acc}, F1: {train_f1}')
    
    # epoch 끝날 때마다 평균 loss, accuracy, f1 score 구하고 모델 저장    
    def validation_epoch_end(self, outputs, state='val') :
        val_loss = torch.tensor(0, dtype=torch.float)
        for i in outputs :
            val_loss += i['loss'].cpu().detach()
        # epoch 평균 loss
        val_loss = val_loss / len(outputs)
        
        val_acc = torch.stack([i['accuracy'] for i in outputs]).mean()
        val_f1 = torch.stack([i['f1'] for i in outputs]).mean()
        
        print(f'[Epoch {self.trainer.current_epoch} {state.upper()}]')
        # train.py에서 model_checkpoint 저장할 때 log에 있는 것 중 하나로 저장 가능하다
        # val_accuracy로 저장함
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', val_acc, on_epoch=True, prog_bar=True)
        self.log('val_f1', val_f1, on_epoch=True, prog_bar=True)
        
    def configure_optimizers(self) :
        # optimizer 설정 가능
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        # learning scheduler 설정가능
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        
        return {
            'optimizer': optimizer,
            'scheduler': lr_scheduler,
        }