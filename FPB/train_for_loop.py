import glob
import os
import random
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import early_stopping

from PhraseBankDataModule import *
from PhraseBankClassification import *
from config import *
import pandas as pd
from itertools import product
from transformers import AutoTokenizer, AutoModel

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

def set_random_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)

    pl.seed_everything(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Set random seed : {random_seed}")

PhraseBank_path = {
    'train' : 'data/train.csv',
    'valid' : 'data/valid.csv',
    'test' : 'data/test.csv'
}


dir_path = 'output'

output_list = []
model_name_list = []

model_dict = {
    # "SALT-NLP/FLANG-BERT":["SALT-NLP/FLANG-BERT"],
    'roberta-base':[
    # 'roberta-base',
    'HYdsl/FiLM'
    ],
    
    # "ProsusAI/finbert":[f"{root_path}/models/finbert"],
    # "bert-base-uncased":[
    #                     "bert-base-uncased"
    # ],
    # "yiyanghkust/finbert-tone":["yiyanghkust/finbert-tone"],
    # "nlpaueb/sec-bert-base":["nlpaueb/sec-bert-base"],
}

BS = [64, 32]
LR = [1e-5, 2e-5, 3e-5, 5e-5]
SEED = [1004]

# grid search
for seed, batch_size, learning_rate, model_key in product(SEED, BS, LR, model_dict.keys()):
    set_random_seed(random_seed=seed)
    for model_path in model_dict[model_key]:
        config = Config(model = model_path, tokenizer = model_key, max_length=64, batch_size=batch_size, num_workers=1, 
                        learning_rate=learning_rate)
        config.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)

        model = PhraseBankClassification(config)

        dm = PhraseBankDataModule(PhraseBank_path, config)

        model_name = model_path.split('/')[-1]

        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_accuracy',
                                                        dirpath=dir_path,
                                                        filename=f'[{model_name}]' + '-{epoch:02d}-{val_accuracy:.3f}-{val_f1:.3f}',
                                                        verbose=False,
                                                        save_last=False,
                                                        mode='max',
                                                        save_top_k=1,
                                                        )
        early_stopping = pl.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, mode='max')
        tb_logger = pl_loggers.TensorBoardLogger(os.path.join(dir_path, 'tb_logs'))

        lr_logger = pl.callbacks.LearningRateMonitor()

        trainer = pl.Trainer(
            default_root_dir= os.path.join(dir_path, 'checkpoints'),
            logger = tb_logger,
            callbacks = [checkpoint_callback, lr_logger, early_stopping],
            max_epochs=20,
            gpus=[0]
        )
        trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())

        eval_model = PhraseBankClassification.load_from_checkpoint(checkpoint_callback.best_model_path, config=config)
        output = trainer.test(model=eval_model, dataloaders=dm.val_dataloader())
        test_output = trainer.test(model=eval_model, dataloaders=dm.test_dataloader())
        model_name_list.append(model_name)
        result = output[0]
        test_result = test_output[0]
        result['seed'] = seed
        result['batch_size'] = batch_size
        result['learning_rate'] = learning_rate
        result['real_test_accuracy'] = test_result['test_accuracy']
        result['real_test_f1'] = test_result['test_f1']
        output_list.append(result)

        pd.DataFrame(output_list, index=model_name_list).to_csv(os.path.join(dir_path, 'all_result.csv'))
