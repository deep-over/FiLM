import os
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import pytorch_lightning as pl

from HeadlineDataModule import HeadlineDataModule
from HeadlineClassification import HeadlineClassification

from transformers import AutoTokenizer
from model_dictionary import model_dict
from itertools import product

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

#1. Set Seed
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

#set config
class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.seed = 42
        self.tokenizer = 'roberta-base'
        self.model = 'roberta-base'
        self.learning_rate = 3e-5
        self.batch_size = 32
        self.num_labels = 3
        self.num_epochs = 100
        self.num_gpus = 1
        self.mode = "test"

root_path = "Headline"
dir_parh = './outputs'
data_path = os.path.join(f'{root_path}','data/gold-dataset-sinha-khandait.csv')

SEED = [4068]
LR = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
BATCH_SIZE = [128, 64]

# unzip to model_dict values after len model_dict
model_dict_len = sum([len(model_list) for model_list in model_dict.values()])
progress_bar = tqdm(total=len(SEED)*len(BATCH_SIZE)*len(LR)*model_dict_len)
print("total: ", progress_bar.total)

config = Config()

output_list = []
for seed, batch_size, lr, model_key in product(SEED, BATCH_SIZE, LR, model_dict.keys()):
    config.seed = seed
    set_seed(config.seed)
    config.tokenizer = model_key

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    config.batch_size = batch_size
    for model_path in model_dict[model_key]:
        model_name = model_path.split('/')[-1]
        config.model = model_path
        dm = HeadlineDataModule(data_path=data_path, tokenizer=tokenizer, config=config)
        # don't save model

        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='val_f1',
            min_delta=0.00,
            patience=7,
            verbose=False,
            mode='max'
        )
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_f1',
            dirpath=dir_parh,
            filename=f'[{model_name}]-{batch_size}-{lr}' + '-{epoch:02d}-{val_accuracy:.3f}-{val_f1:.3f}',
            save_top_k=1,
            mode='max',
            )
        config.learning_rate = lr
        model = HeadlineClassification(config)

        trainer = pl.Trainer(
            gpus=[config.num_gpus],
            max_epochs=config.num_epochs,
            callbacks=[checkpoint_callback, early_stop_callback],
            default_root_dir=os.path.join(dir_parh, 'checkpoints')
        )

        trainer.fit(model, dm)

        #evaluate
        eval_model = HeadlineClassification.load_from_checkpoint(checkpoint_callback.best_model_path, config=config)
        output = trainer.test(model=eval_model, dataloaders=dm.val_dataloader())[0]
        test_output = trainer.test(model=eval_model, dataloaders=dm.test_dataloader())[0]
        output['model_name'] = model_name
        output['seed'] = seed
        output['lr'] = lr
        output['batch_size'] = batch_size
        output['real_test_accuracy'] = test_output['test_accuracy']
        output['real_test_f1'] = test_output['test_f1']
        output_list.append(output)
        progress_bar.update(1)

        df = pd.DataFrame(output_list)
        print(df)
        df.to_csv('output.csv', index=False)