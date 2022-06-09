import os

from pytorch_lightning import loggers as pl_loggers

from FiQADataModule import *
from FiQARegression import *
from config import *

headline_path = {
    'train' : "/home/ailab/Desktop/NY/FinBERT/Financial-Pre-trained-research/downstream/data/FiQA/headline_orig_train.csv",
    "valid" : "/home/ailab/Desktop/NY/FinBERT/Financial-Pre-trained-research/downstream/data/FiQA/headline_orig_valid.csv",
    "test" : "/home/ailab/Desktop/NY/FinBERT/Financial-Pre-trained-research/downstream/data/FiQA/headline_orig_valid.csv" # 임의
}

post_path = {
    "train" : "/home/ailab/Desktop/NY/FinBERT/Financial-Pre-trained-research/downstream/data/FiQA/post_orig_train.csv",
    "valid" : "/home/ailab/Desktop/NY/FinBERT/Financial-Pre-trained-research/downstream/data/FiQA/post_orig_valid.csv",
    "test" : "/home/ailab/Desktop/NY/FinBERT/Financial-Pre-trained-research/downstream/data/FiQA/post_orig_valid.csv" # 임의
}

# headline_path = {
#     'train' : '/home/ailab/Desktop/NY/FinBERT/Financial-Pre-trained-research/downstream/data/FiQA_ABSA_task1/headline_train.csv',
#     'valid' : '/home/ailab/Desktop/NY/FinBERT/Financial-Pre-trained-research/downstream/data/FiQA_ABSA_task1/headline_valid.csv',
#     'test' : '/home/ailab/Desktop/NY/FinBERT/Financial-Pre-trained-research/downstream/data/FiQA_ABSA_task1/headline_test.csv'
# }

# post_path = {
#     'train' : '/home/ailab/Desktop/NY/FinBERT/Financial-Pre-trained-research/downstream/data/FiQA_ABSA_task1/post_train.csv',
#     'valid' : '/home/ailab/Desktop/NY/FinBERT/Financial-Pre-trained-research/downstream/data/FiQA_ABSA_task1/post_valid.csv',
#     'test' : '/home/ailab/Desktop/NY/FinBERT/Financial-Pre-trained-research/downstream/data/FiQA_ABSA_task1/post_test.csv'
# }

# head_post_path = {
#     'train' : '/home/ailab/Desktop/NY/FinBERT/Financial-Pre-trained-research/downstream/data/FiQA_ABSA_task1/head_post_train.csv',
#     'valid' : '/home/ailab/Desktop/NY/FinBERT/Financial-Pre-trained-research/downstream/data/FiQA_ABSA_task1/head_post_valid.csv',
#     'test' : '/home/ailab/Desktop/NY/FinBERT/Financial-Pre-trained-research/downstream/data/FiQA_ABSA_task1/head_post_test.csv'
# }


bart_config = Config('facebook/bart-base', 64, 64, 1, 2e-5, 0.1)
# model 생성
model = FiQARegression(bart_config)
# datamodule 생성
# dm = FiQADataModule(headline_path, bart_config)
dm = FiQADataModule(post_path, bart_config)

# dirpath = './bart_fiqa_headline'
dirpath = './bart_fiqa_post'
# checkpoint 어떻게 저장할지 설정
checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                dirpath=dirpath,
                                                filename='{epoch:02d}-{val_loss:.3f}-{val_r2:.3f}',
                                                verbose=False,
                                                save_last=True,
                                                mode='min',
                                                save_top_k=1
                                                )
# tensorboard logger
tb_logger = pl_loggers.TensorBoardLogger(os.path.join(dirpath, 'tb_logs'))
# learning rate logger
lr_logger = pl.callbacks.LearningRateMonitor()
# train 설정
trainer = pl.Trainer(
    default_root_dir = os.path.join(dirpath,'checkpoints'),
    logger = tb_logger,
    callbacks = [checkpoint_callback, lr_logger],
    max_epochs=20,
    gpus=1
)
# 학습 시작
trainer.fit(model, dm)