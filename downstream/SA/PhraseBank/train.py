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

# config = Config('roberta-base', 'roberta-base', 64, 64, 1, 2e-5)
# config = Config('facebook/bart-base', 'facebook/bart-base', 64, 64, 1, 2e-5)
# config = Config('yiyanghkust/finbert-tone', 'yiyanghkust/finbert-tone', 64, 64, 1, 2e-5)
# config = Config('/home/ailab/Desktop/JY/roberta-retrained', 'roberta-base', 64, 64, 1, 2e-5)
# config = Config('/home/ailab/Desktop/JY/roberta-retrained/deduplicate', 'roberta-base', 64, 64, 1, 2e-5)
config = Config('/home/ailab/Desktop/JY/bart-retrained/logs/checkpoints/epoch=0-step=1052212.ckpt', 'facebook/bart-base', 64, 64, 1, 2e-5)
# config = Config('/home/ailab/Desktop/NY/FinBERT/Financial-Pre-trained-research/downstream/SA/PhraseBank/fin-bart', 'facebook/bart-base', 64, 64, 1, 2e-5)

dir_path = './phrasebank'

model = PhraseBankClassification(config)

dm = PhraseBankDataModule(PhraseBank_path, config)

checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_accuracy',
                                                dirpath=dir_path,
                                                filename='{epoch:02d}-{val_accuracy:.3f}-{val_f1:.3f}',
                                                verbose=False,
                                                save_last=True,
                                                mode='max',
                                                save_top_k=1,
                                                )

tb_logger = pl_loggers.TensorBoardLogger(os.path.join(dir_path, 'tb_logs'))

lr_logger = pl.callbacks.LearningRateMonitor()

trainer = pl.Trainer(
    default_root_dir= os.path.join(dir_path, 'checkpoints'),
    logger = tb_logger,
    callbacks = [checkpoint_callback, lr_logger],
    max_epochs=20,
    gpus=[1]
)

trainer.fit(model, dm)