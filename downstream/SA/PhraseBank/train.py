import os

from pytorch_lightning import loggers as pl_loggers

from PhraseBankDataModule import *
from PhraseBankClassification import *
from config import *

PhraseBank_path = {
    'train' : '/home/ailab/Desktop/NY/FinBERT/Financial-Pre-trained-research/downstream/data/FinancialPhraseBank/train.csv',
    'valid' : '/home/ailab/Desktop/NY/FinBERT/Financial-Pre-trained-research/downstream/data/FinancialPhraseBank/valid.csv',
    'test' : '/home/ailab/Desktop/NY/FinBERT/Financial-Pre-trained-research/downstream/data/FinancialPhraseBank/test.csv'
}

# roberta_config = Config('roberta-base', 64, 64, 1, 2e-5)
# model = PhraseBankClassification(roberta_config)
# dm = PhraseBankDataModule(PhraseBank_path, roberta_config)

# checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_accuracy',
#                                                 dirpath='./roberta_phrasebank',
#                                                 filename='{epoch:02d}-{val_accuracy:.3f}',
#                                                 verbose=False,
#                                                 save_last=True,
#                                                 mode='max',
#                                                 save_top_k=-1,
#                                                 )

# tb_logger = pl_loggers.TensorBoardLogger(os.path.join('./roberta_phrasebank', 'tb_logs'))

# lr_logger = pl.callbacks.LearningRateMonitor()

# trainer = pl.Trainer(
#     default_root_dir='./roberta_phrasebank/checkpoints',
#     logger = tb_logger,
#     callbacks = [checkpoint_callback, lr_logger],
#     max_epochs=6,
#     gpus=1
# )

# trainer.fit(model, dm)

bart_config = Config('facebook/bart-base', 64, 64, 1, 2e-5)
# model 생성
model = PhraseBankClassification(bart_config)
# datamodule 생성
dm = PhraseBankDataModule(PhraseBank_path, bart_config)
# checkpoint 어떻게 저장할지 설정
checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_accuracy',
                                                dirpath='./bart_phrasebank',
                                                filename='{epoch:02d}-{val_accuracy:.3f}',
                                                verbose=False,
                                                save_last=True,
                                                mode='max',
                                                save_top_k=-1,
                                                )
# tensorboard logger
tb_logger = pl_loggers.TensorBoardLogger(os.path.join('./bart_phrasebank', 'tb_logs'))
# learning rate logger
lr_logger = pl.callbacks.LearningRateMonitor()
# train 설정
trainer = pl.Trainer(
    default_root_dir='./bart_phrasebank/checkpoints',
    logger = tb_logger,
    callbacks = [checkpoint_callback, lr_logger],
    max_epochs=6,
    gpus=1
)
# 학습 시작
trainer.fit(model, dm)