import argparse

parser = argparse.ArgumentParser(description='FiQA Sentiment Analysis')

parser.add_argument('--model', type=str, help='model name or path')
parser.add_argument('--tokenizer', type=str, help='tokenizer name or path')
parser.add_argument('--data', type=str, help='headline | post')
parser.add_argument('--dirpath', type=str, help='path to save checkpoint')
# hyperparameters
parser.add_argument('--max_length', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=5)

# absa mode
parser.add_argument('--absa', type=str, help='none : not absa | NLI-M : NLI-M ', default='none')
parser.add_argument('--aspect', type=bool, help='False : train with target | True : train with aspect + target', default=False)
parser.add_argument('--sep', type=str, help='seperate target and aspect with input char', default='-')
# CLS, target
parser.add_argument('--CLS_MEAN', type=str, help='CLS : use only CLS | MEAN : use target_aspect\'s mean | CLS_MEAN : use CLS and target_aspect\'s mean', default='CLS')
args = parser.parse_args()

import os

from pytorch_lightning import loggers as pl_loggers

from FiQADataModule import *
from FiQARegression import *

# data path
if args.data == 'headline' :
    path = {
        'train' : "/home/ailab/Desktop/NY/FinBERT/Financial-Pre-trained-research/downstream/data/fiqa/headline_orig_train.csv",
        "valid" : "/home/ailab/Desktop/NY/FinBERT/Financial-Pre-trained-research/downstream/data/fiqa/headline_orig_valid.csv"
    }
elif args.data == 'post' :
    path = {
        "train" : "/home/ailab/Desktop/NY/FinBERT/Financial-Pre-trained-research/downstream/data/fiqa/post_orig_train.csv",
        "valid" : "/home/ailab/Desktop/NY/FinBERT/Financial-Pre-trained-research/downstream/data/fiqa/post_orig_valid.csv"
    }

model = FiQARegression(args)
dm = FiQADataModule(path, args)

checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                dirpath=args.dirpath,
                                                filename='{epoch:02d}-{val_loss:.3f}-{val_r2:.3f}',
                                                verbose=False,
                                                save_last=True,
                                                mode='min',
                                                save_top_k=1
                                                )

# tensorboard logger
tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.dirpath, 'tb_logs'))
# learning rate logger
lr_logger = pl.callbacks.LearningRateMonitor()
# train 설정
trainer = pl.Trainer(
    default_root_dir = os.path.join(args.dirpath,'checkpoints'),
    logger = tb_logger,
    callbacks = [checkpoint_callback, lr_logger],
    max_epochs=args.epochs,
    gpus=[1]
)
# 학습 시작
trainer.fit(model, dm)
