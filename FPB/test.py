from glob import glob
import os
import random
from pytorch_lightning import loggers as pl_loggers

from PhraseBankDataModule import *
from PhraseBankClassification import *
from config import *
import pandas as pd
from transformers import AutoTokenizer, AutoModel

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'

def set_random_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)

    pl.seed_everything(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Set random seed : {random_seed}")

def load_to_model(config):
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    
    return tokenizer


set_random_seed(42)
PhraseBank_path = {
    'train' : 'data/train.csv',
    'valid' : 'data/valid.csv',
    'test' : 'data/test.csv'
}
ckpt_list = glob('output/*/*.ckpt')
output_list = []
model_name_list = []

tokenizer_name = "roberta-base"
for ckpt in ckpt_list:
    model_path = ckpt
    model_name = model_path.split('/')[-1].split('.ckpt')[0]
    model_name_list.append(model_name)

    config = Config(model_path, tokenizer_name, 
                max_length=64, batch_size=64, num_workers=1, learning_rate=2e-5)
        
    config.tokenizer = load_to_model(config)

    dm = PhraseBankDataModule(PhraseBank_path, config)

    eval_model = PhraseBankClassification.load_from_checkpoint(model_path, config=config)

    trainer = pl.Trainer(gpus=1, max_epochs=1, logger=False, checkpoint_callback=False)

    # let's test it
    output = trainer.test(model=eval_model, dataloaders=dm.test_dataloader())
    output_list.append(output[0])

pd.DataFrame(output_list, index=model_name_list).to_csv('eval_result.csv')